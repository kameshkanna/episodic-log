"""Step-synchronized batch agent loop for vLLM evaluation.

Instead of processing sessions one-by-one (sequential), this module runs all
sessions in a single worker simultaneously:

  Step 1: build initial prompts for all N sessions → one vLLM batch of N
  Step 2: execute tool calls (CPU, instant) → one vLLM batch of M ≤ N
  Step 3: ...repeat until all sessions have a final answer

This keeps the GPU at near-100% utilisation.  With HuggingFace generate()
the GPU sits idle while tool calls execute; here that dead time is filled by
the other sessions' tool executions running in parallel on CPU.

Only used when the provider exposes ``generate_with_tools_batch`` (vLLM).
Falls back gracefully to the regular AgentLoop for providers that don't.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from episodic_log.agent.loop import _MAX_SUMMARY_CHARS, _SYSTEM_PROMPT_GREP, _SYSTEM_PROMPT_LOAD_ONLY
from episodic_log.agent.trace import AgentTrace, ToolCallRecord
from episodic_log.tools.session_tools import format_summaries_as_context, make_session_tools

logger = logging.getLogger(__name__)


@dataclass
class _SessionState:
    """Mutable per-session state for the batch loop."""

    session_id: str
    question: str
    summary_method: str
    messages: list[dict[str, Any]]
    tools: dict[str, Any]           # callable tool functions, session-specific
    tool_schemas: list[dict]        # JSON schema dicts, shared across sessions
    tool_call_records: list[ToolCallRecord] = field(default_factory=list)
    turns_loaded: list[str] = field(default_factory=list)
    step: int = 0
    done: bool = False
    final_answer: str = ""


def _build_first_message(
    question: str,
    summaries_dir: Path,
    summary_method: str,
    mode: str,
) -> str:
    """Build the initial user message for one session."""
    if mode == "grep_and_load":
        return question

    summary_context = format_summaries_as_context(summaries_dir, summary_method)
    if not summary_context:
        return f"No memory index available.\n\nQuestion: {question}"

    if len(summary_context) > _MAX_SUMMARY_CHARS:
        cutoff = summary_context.rfind("\n", 0, _MAX_SUMMARY_CHARS)
        summary_context = summary_context[:cutoff if cutoff != -1 else _MAX_SUMMARY_CHARS]
        summary_context += "\n...[index truncated]"

    n_turns = max(0, summary_context.count("\n"))
    return (
        f"Memory index — {n_turns} turns (columns: turn_id TAB summary):\n"
        f"{summary_context}\n\nQuestion: {question}"
    )


def _call_tool(tool_name: str, tool_args: dict[str, Any], tools: dict[str, Any]) -> str:
    if tool_name not in tools:
        return f"Unknown tool '{tool_name}'. Available: {list(tools.keys())}"
    try:
        return str(tools[tool_name](**tool_args))
    except (TypeError, ValueError, FileNotFoundError) as exc:
        return f"Error calling {tool_name}: {exc}"


def run_batch(
    sessions: list[dict[str, Any]],
    summary_method: str,
    mode: str,
    provider: Any,
    max_tool_calls: int = 15,
) -> list[AgentTrace]:
    """Run the agent loop for all sessions simultaneously via batch inference.

    At each step every active (not-yet-answered) session is submitted to the
    provider in a single ``generate_with_tools_batch`` call.  Sessions that
    produce a text answer are retired; sessions that produce a tool call have
    the tool executed (CPU) and remain active for the next step.

    Args:
        sessions: List of session metadata dicts (same format as evaluate.py).
        summary_method: Summarizer method key (``"lexical"``, ``"scout"``, etc.).
        mode: ``"load_only"`` for recall conditions; ``"grep_and_load"`` for
            grep_recall conditions.
        provider: A provider instance that exposes ``generate_with_tools_batch``.
        max_tool_calls: Maximum tool calls per session before forcing an answer.

    Returns:
        Ordered list of :class:`~episodic_log.agent.trace.AgentTrace` objects,
        one per session, in the same order as *sessions*.
    """
    system_prompt = _SYSTEM_PROMPT_GREP if mode == "grep_and_load" else _SYSTEM_PROMPT_LOAD_ONLY

    # ── Initialise per-session state ────────────────────────────────────────
    states: list[_SessionState] = []
    for meta in sessions:
        session_id: str = meta["session_id"]
        question: str = meta["question"]
        log_path = Path(meta["log_path"])
        summaries_dir = Path(meta["summaries_dir"])

        tools, tool_schemas = make_session_tools(
            summaries_dir=summaries_dir,
            log_path=log_path,
            method=summary_method,
            mode=mode,
        )
        first_message = _build_first_message(question, summaries_dir, summary_method, mode)
        states.append(_SessionState(
            session_id=session_id,
            question=question,
            summary_method=summary_method,
            messages=[{"role": "user", "content": first_message}],
            tools=tools,
            tool_schemas=tool_schemas,
        ))

    # All sessions in the same condition share identical tool schemas.
    shared_schemas: list[dict] = states[0].tool_schemas if states else []

    logger.info(
        "batch_loop.run_batch: %d sessions  mode=%s  method=%s",
        len(states), mode, summary_method,
    )

    # ── Step-synchronised batch loop ────────────────────────────────────────
    for step in range(max_tool_calls):
        active_idx = [i for i, s in enumerate(states) if not s.done]
        if not active_idx:
            break

        batch_messages = [states[i].messages for i in active_idx]
        outputs = provider.generate_with_tools_batch(
            batch_messages=batch_messages,
            tools=shared_schemas,
            system=system_prompt,
            max_tokens=1024,
        )

        logger.debug(
            "batch_loop step=%d active=%d", step + 1, len(active_idx),
        )

        for i, output in zip(active_idx, outputs):
            state = states[i]
            if output["type"] == "text":
                state.final_answer = output["content"]
                state.done = True
                continue

            tool_name: str = output["tool_name"]
            tool_args: dict[str, Any] = output["tool_args"]
            state.messages.append(output["raw_message"])

            tool_result = _call_tool(tool_name, tool_args, state.tools)
            state.tool_call_records.append(ToolCallRecord(
                tool_name=tool_name,
                arguments=tool_args,
                result=tool_result,
                timestamp=datetime.now(tz=timezone.utc),
            ))
            if tool_name == "load_turn" and "turn_id" in tool_args:
                state.turns_loaded.append(str(tool_args["turn_id"]))

            state.messages.append({"role": "tool", "content": tool_result, "name": tool_name})
            state.step += 1

    # ── Force answers for sessions that exhausted their tool budget ──────────
    exhausted_idx = [i for i, s in enumerate(states) if not s.done]
    if exhausted_idx:
        logger.warning(
            "batch_loop: %d/%d sessions exhausted max_tool_calls=%d — forcing answers",
            len(exhausted_idx), len(states), max_tool_calls,
        )
        forced_messages = [
            states[i].messages + [{
                "role": "user",
                "content": f"Based on the turns you loaded, answer now: {states[i].question}",
            }]
            for i in exhausted_idx
        ]
        forced_answers = provider.generate_batch(
            batch_messages=forced_messages,
            system=system_prompt,
            max_tokens=512,
        )
        for i, answer in zip(exhausted_idx, forced_answers):
            states[i].final_answer = answer

    return [
        AgentTrace(
            question=s.question,
            answer=s.final_answer.strip(),
            tool_calls=s.tool_call_records,
            turns_loaded=s.turns_loaded,
            total_tool_calls=len(s.tool_call_records),
            session_id=s.session_id,
            summary_method=s.summary_method,
        )
        for s in states
    ]
