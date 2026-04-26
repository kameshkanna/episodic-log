"""Tool-use agent loop for the CHD (Conversational Hallucination Drift) evaluation.

Design: no retrieval layer.  The agent receives the full 1-2 line summary of
every turn in the session in its first message.  It reads those summaries,
decides which turns are relevant, then calls ``load_turn`` to read the full
verbatim content from ``log.jsonl`` before producing a final answer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from episodic_log.agent.trace import AgentTrace, ToolCallRecord
from episodic_log.providers.base import BaseProvider
from episodic_log.tools.session_tools import format_summaries_as_context, make_session_tools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_LOAD_ONLY: str = (
    "You are answering a question about a past conversation.\n"
    "The memory index in the user message lists every turn with a one-line summary.\n\n"
    "Instructions:\n"
    "1. Read the memory index and identify which turns are relevant to the question.\n"
    "2. Call load_turn for each relevant turn to read its full verbatim content.\n"
    "3. Answer ONLY after reading the supporting turns.\n"
    "4. If the answer spans multiple turns, load all of them.\n"
    "5. If nothing in the index is relevant, say so explicitly — do NOT guess."
)

_SYSTEM_PROMPT_GREP: str = (
    "You are answering a question about a past conversation.\n"
    "You have two tools: grep_memory to search turn summaries, and load_turn to read full turns.\n\n"
    "Instructions:\n"
    "1. Think about what keywords would appear in a summary of the relevant turn.\n"
    "2. Call grep_memory with those keywords to find candidate turns.\n"
    "3. If the first search misses, try different keywords or synonyms.\n"
    "4. Call load_turn on turns that look relevant from their summary.\n"
    "5. Answer ONLY after reading the verbatim turn content.\n"
    "6. Do NOT guess — if you cannot find the answer after searching, say so."
)

# Default — overridden per-call based on mode.
SYSTEM_PROMPT = _SYSTEM_PROMPT_LOAD_ONLY


class AgentLoop:
    """Runs the tool-use agent loop: model reads summary index, loads turns, answers.

    The loop operates as follows:

    1. Format all turn summaries as a text block and inject into the first message.
    2. Inject the system prompt.
    3. The model reads the summary index, then calls ``load_turn`` for relevant turns.
    4. Repeat up to *max_tool_calls* times:

       a. Call :meth:`~episodic_log.providers.base.BaseProvider.generate_with_tools`.
       b. If the response is a final text answer, break.
       c. If the response is a ``load_turn`` call, execute it, append result, continue.

    5. If the budget is exhausted without a text answer, force one via
       :meth:`~episodic_log.providers.base.BaseProvider.generate`.
    6. Return a fully populated :class:`~episodic_log.agent.trace.AgentTrace`.

    Args:
        provider: An instantiated :class:`~episodic_log.providers.base.BaseProvider`.
        max_tool_calls: Maximum ``load_turn`` calls before forcing a final answer.
            Defaults to 15.

    Raises:
        TypeError: If *provider* is not a BaseProvider instance.
        ValueError: If *max_tool_calls* is not a positive integer.
    """

    def __init__(
        self,
        provider: BaseProvider,
        max_tool_calls: int = 15,
        mode: str = "load_only",
    ) -> None:
        if not isinstance(provider, BaseProvider):
            raise TypeError(
                f"provider must be a BaseProvider instance, got {type(provider)}"
            )
        if not isinstance(max_tool_calls, int) or max_tool_calls <= 0:
            raise ValueError(
                f"max_tool_calls must be a positive integer, got {max_tool_calls!r}"
            )
        if mode not in ("load_only", "grep_and_load"):
            raise ValueError(f"mode must be 'load_only' or 'grep_and_load', got {mode!r}")
        self._provider = provider
        self._max_tool_calls = max_tool_calls
        self._mode = mode

    def run(
        self,
        question: str,
        session_meta: dict[str, Any],
        summary_method: str,
    ) -> AgentTrace:
        """Execute one agent question-answering run and return the full trace.

        Args:
            question: The question to answer using the session's memory.
            session_meta: Mapping that must contain ``"session_id"``,
                ``"log_path"``, and ``"summaries_dir"``.
            summary_method: Summarizer method key used to load the correct
                ``<method>.jsonl`` summary file.

        Returns:
            A :class:`~episodic_log.agent.trace.AgentTrace`.

        Raises:
            TypeError: If *question* is not a string or *session_meta* is not a dict.
            KeyError: If *session_meta* is missing required keys.
        """
        if not isinstance(question, str):
            raise TypeError(f"question must be a str, got {type(question)}")
        if not isinstance(session_meta, dict):
            raise TypeError(f"session_meta must be a dict, got {type(session_meta)}")
        if not summary_method or not isinstance(summary_method, str):
            raise ValueError(f"summary_method must be a non-empty string, got {summary_method!r}")

        for key in ("session_id", "log_path", "summaries_dir"):
            if key not in session_meta:
                raise KeyError(f"session_meta is missing required key: '{key}'")

        session_id: str = session_meta["session_id"]
        log_path = Path(session_meta["log_path"])
        summaries_dir = Path(session_meta["summaries_dir"])

        tools, tool_schemas = make_session_tools(
            summaries_dir=summaries_dir,
            log_path=log_path,
            method=summary_method,
            mode=self._mode,
        )

        # Build the first user message depending on mode.
        if self._mode == "load_only":
            # Dump all summaries upfront — model reads and calls load_turn.
            summary_context = format_summaries_as_context(summaries_dir, summary_method)
            if summary_context:
                first_message = (
                    f"Memory index ({summary_context.count(chr(10)) + 1} turns):\n"
                    f"{summary_context}\n\n"
                    f"Question: {question}"
                )
            else:
                first_message = f"No memory index available.\n\nQuestion: {question}"
        else:
            # grep_and_load — model receives only the question and must search.
            first_message = question

        logger.info(
            "AgentLoop.run: session_id=%s method=%s mode=%s question=%r",
            session_id, summary_method, self._mode, question,
        )

        system_prompt = (
            _SYSTEM_PROMPT_GREP if self._mode == "grep_and_load"
            else _SYSTEM_PROMPT_LOAD_ONLY
        )

        tool_call_records: list[ToolCallRecord] = []
        turns_loaded: list[str] = []
        messages: list[dict[str, Any]] = [{"role": "user", "content": first_message}]
        final_answer: str = ""

        for step in range(self._max_tool_calls):
            logger.debug("AgentLoop.run: step=%d/%d", step + 1, self._max_tool_calls)
            result = self._provider.generate_with_tools(
                messages=messages,
                tools=tool_schemas,
                system=system_prompt,
                max_tokens=1024,
            )

            if result["type"] == "text":
                final_answer = result["content"]
                logger.info(
                    "AgentLoop.run: final answer at step=%d len=%d",
                    step + 1, len(final_answer),
                )
                break

            tool_name: str = result["tool_name"]
            tool_args: dict[str, Any] = result["tool_args"]
            raw_message: dict[str, Any] = result["raw_message"]

            messages.append(raw_message)

            tool_result: str = self._call_tool(tool_name, tool_args, tools)
            call_ts = datetime.now(tz=timezone.utc)

            tool_call_records.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=tool_result,
                    timestamp=call_ts,
                )
            )

            if tool_name == "load_turn" and "turn_id" in tool_args:
                turns_loaded.append(str(tool_args["turn_id"]))

            messages.append(
                {"role": "tool", "content": tool_result, "name": tool_name}
            )
        else:
            logger.warning(
                "AgentLoop.run: max_tool_calls=%d exhausted for session=%s — forcing answer.",
                self._max_tool_calls, session_id,
            )
            forced_prompt = f"Based on the turns you loaded, answer now: {question}"
            messages.append({"role": "user", "content": forced_prompt})
            final_answer = self._provider.generate(
                messages=messages,
                system=system_prompt,
            )

        return AgentTrace(
            question=question,
            answer=final_answer,
            tool_calls=tool_call_records,
            turns_loaded=turns_loaded,
            total_tool_calls=len(tool_call_records),
            session_id=session_id,
            summary_method=summary_method,
        )

    def _call_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tools: dict[str, Any],
    ) -> str:
        if tool_name not in tools:
            return (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {list(tools.keys())}"
            )
        try:
            return str(tools[tool_name](**tool_args))
        except (TypeError, ValueError, FileNotFoundError) as exc:
            logger.warning(
                "AgentLoop._call_tool: tool=%s raised %s: %s",
                tool_name, type(exc).__name__, exc,
            )
            return f"Error calling {tool_name}: {exc}"
