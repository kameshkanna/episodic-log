"""Tool-use agent loop for the CHD (Conversational Hallucination Drift) evaluation.

:class:`AgentLoop` drives a model through an iterative tool-calling cycle:
the model calls ``grep_memory`` to search a session's summary index, then
``load_turn`` to read verbatim turn content, until it has enough information
to produce a final answer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from episodic_log.agent.trace import AgentTrace, ToolCallRecord
from episodic_log.providers.base import BaseProvider
from episodic_log.tools.session_tools import TOOL_SCHEMAS, make_session_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "You are an AI assistant with access to a memory index of past conversations.\n"
    "Use grep_memory to search for relevant turns, then load_turn to read their full content.\n"
    "You may call tools multiple times. Once you have enough information, give a direct answer.\n"
    "Do not call tools if you already have the answer."
)


class AgentLoop:
    """Runs the tool-use agent loop: model calls tools until it produces a final answer.

    The loop operates as follows:

    1. Build session-bound tool callables via :func:`~episodic_log.tools.session_tools.make_session_tools`.
    2. Inject the system prompt.
    3. Start the conversation with the user's question.
    4. Repeat up to *max_tool_calls* times:

       a. Call :meth:`~episodic_log.providers.base.BaseProvider.generate_with_tools`.
       b. If the response is a final text answer, break.
       c. If the response is a tool call, execute the tool, append both the
          assistant tool-call message and the tool-result message, then continue.

    5. If the budget is exhausted without a text answer, force one via
       :meth:`~episodic_log.providers.base.BaseProvider.generate`.
    6. Return a fully populated :class:`~episodic_log.agent.trace.AgentTrace`.

    Args:
        provider: An instantiated :class:`~episodic_log.providers.base.BaseProvider`.
        max_tool_calls: Maximum number of tool invocations before forcing a final answer.
            Defaults to 8.

    Raises:
        TypeError: If *provider* is not a BaseProvider instance.
        ValueError: If *max_tool_calls* is not a positive integer.
    """

    def __init__(
        self,
        provider: BaseProvider,
        max_tool_calls: int = 8,
    ) -> None:
        if not isinstance(provider, BaseProvider):
            raise TypeError(
                f"provider must be a BaseProvider instance, got {type(provider)}"
            )
        if not isinstance(max_tool_calls, int) or max_tool_calls <= 0:
            raise ValueError(
                f"max_tool_calls must be a positive integer, got {max_tool_calls!r}"
            )
        self._provider = provider
        self._max_tool_calls = max_tool_calls

    def run(
        self,
        question: str,
        session_meta: dict[str, Any],
        summary_method: str,
    ) -> AgentTrace:
        """Execute one agent question-answering run and return the full trace.

        Args:
            question: The question to answer using the session's memory.
            session_meta: Mapping that must contain:

                - ``"session_id"`` (``str``): Session identifier.
                - ``"log_path"`` (``str | Path``): Path to ``log.jsonl``.
                - ``"summaries_dir"`` (``str | Path``): Path to the summaries directory.

            summary_method: Summarizer method key used to load the correct
                ``<method>.jsonl`` file (e.g. ``"lexical"``, ``"scout"``,
                ``"echo"``).

        Returns:
            A :class:`~episodic_log.agent.trace.AgentTrace` with the question,
            final answer, all tool call records, and metadata.

        Raises:
            TypeError: If *question* is not a string or *session_meta* is not a dict.
            ValueError: If required keys are missing from *session_meta* or
                *summary_method* is empty.
            KeyError: If *session_meta* is missing required keys.
        """
        if not isinstance(question, str):
            raise TypeError(f"question must be a str, got {type(question)}")
        if not isinstance(session_meta, dict):
            raise TypeError(f"session_meta must be a dict, got {type(session_meta)}")
        if not summary_method or not isinstance(summary_method, str):
            raise ValueError(f"summary_method must be a non-empty string, got {summary_method!r}")

        _required_keys = ("session_id", "log_path", "summaries_dir")
        for key in _required_keys:
            if key not in session_meta:
                raise KeyError(f"session_meta is missing required key: '{key}'")

        session_id: str = session_meta["session_id"]
        log_path = Path(session_meta["log_path"])
        summaries_dir = Path(session_meta["summaries_dir"])

        tools = make_session_tools(
            summaries_dir=summaries_dir,
            log_path=log_path,
            method=summary_method,
        )

        logger.info(
            "AgentLoop.run: session_id=%s method=%s question=%r",
            session_id,
            summary_method,
            question,
        )

        # Mutable state accumulated over the loop.
        tool_call_records: list[ToolCallRecord] = []
        turns_loaded: list[str] = []
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        final_answer: str = ""

        for step in range(self._max_tool_calls):
            logger.debug("AgentLoop.run: step=%d/%d", step + 1, self._max_tool_calls)

            result = self._provider.generate_with_tools(
                messages=messages,
                tools=TOOL_SCHEMAS,
                system=SYSTEM_PROMPT,
            )

            if result["type"] == "text":
                final_answer = result["content"]
                logger.info(
                    "AgentLoop.run: final answer received at step=%d len=%d",
                    step + 1,
                    len(final_answer),
                )
                break

            # Tool call branch.
            tool_name: str = result["tool_name"]
            tool_args: dict[str, Any] = result["tool_args"]
            raw_message: dict[str, Any] = result["raw_message"]

            # Append the assistant's tool-call message to history.
            messages.append(raw_message)

            # Execute the tool.
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

            # Track which turns were loaded for downstream analysis.
            if tool_name == "load_turn" and "turn_id" in tool_args:
                turns_loaded.append(str(tool_args["turn_id"]))

            # Append the tool result in Qwen's native tool-result format.
            messages.append(
                {"role": "tool", "content": tool_result, "name": tool_name}
            )
            logger.debug(
                "AgentLoop.run: tool=%s result_len=%d", tool_name, len(tool_result)
            )
        else:
            # Budget exhausted — force a final answer from the model.
            logger.warning(
                "AgentLoop.run: max_tool_calls=%d exhausted for session=%s — forcing answer.",
                self._max_tool_calls,
                session_id,
            )
            forced_prompt = f"Based on what you found, answer now: {question}"
            messages.append({"role": "user", "content": forced_prompt})
            final_answer = self._provider.generate(
                messages=messages,
                system=SYSTEM_PROMPT,
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tools: dict[str, Any],
    ) -> str:
        """Dispatch a tool call by name and return the result as a string.

        Args:
            tool_name: Name of the tool to call.
            tool_args: Argument dict to pass to the tool callable.
            tools: Dict mapping tool names to bound callables.

        Returns:
            Tool output coerced to a string.  If the callable raises, the
            exception message is returned instead so the model can observe the
            error and adjust its next call.
        """
        if tool_name not in tools:
            msg = (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {list(tools.keys())}"
            )
            logger.warning("AgentLoop._call_tool: %s", msg)
            return msg

        callable_ = tools[tool_name]
        try:
            raw_result = callable_(**tool_args)
            result_str = str(raw_result)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            logger.warning(
                "AgentLoop._call_tool: tool=%s raised %s: %s",
                tool_name,
                type(exc).__name__,
                exc,
            )
            result_str = f"Error calling {tool_name}: {exc}"

        return result_str
