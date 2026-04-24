"""Agent trajectory tracing data contracts for the CHD evaluation framework.

:class:`ToolCallRecord` captures a single tool invocation and its result.
:class:`AgentTrace` captures the full trajectory of one agent run, including
the question asked, all tool calls made, and the final answer produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ToolCallRecord:
    """Immutable record of a single tool invocation within an agent run.

    Attributes:
        tool_name: Name of the tool that was called.
        arguments: Argument dict passed to the tool.
        result: Stringified return value of the tool call.
        timestamp: UTC datetime when the tool was called.
    """

    tool_name: str
    arguments: dict[str, Any]
    result: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation of this record.

        The ``timestamp`` is emitted as an ISO-8601 string with UTC suffix.

        Returns:
            Dict with keys ``tool_name``, ``arguments``, ``result``, and
            ``timestamp``.
        """
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentTrace:
    """Full trajectory record for a single agent question-answering run.

    Attributes:
        question: The original question posed to the agent.
        answer: The final answer text produced by the agent.
        tool_calls: Ordered list of all tool invocations made during the run.
        turns_loaded: List of turn IDs retrieved via ``load_turn`` calls.
        total_tool_calls: Total number of tool invocations (redundant with
            ``len(tool_calls)`` but convenient for downstream aggregation).
        session_id: Identifier of the session whose log was queried.
        summary_method: Summarizer method used to build the memory index.
    """

    question: str
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    turns_loaded: list[str] = field(default_factory=list)
    total_tool_calls: int = 0
    session_id: str = ""
    summary_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation of this trace.

        Returns:
            Dict with all fields; ``tool_calls`` is a list of dicts produced
            by :meth:`ToolCallRecord.to_dict`.
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "turns_loaded": self.turns_loaded,
            "total_tool_calls": self.total_tool_calls,
            "session_id": self.session_id,
            "summary_method": self.summary_method,
        }
