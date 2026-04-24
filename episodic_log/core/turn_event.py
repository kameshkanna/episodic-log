"""Core TurnEvent data contract for the episodic execution log.

A TurnEvent is an immutable record of a single agent turn persisted to log.jsonl.
The agent retrieves verbatim TurnEvent records — it never sees summaries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventRole(str, Enum):
    """Role of the participant who produced this event."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class EventType(str, Enum):
    """Semantic type of the event payload."""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    REASONING = "reasoning"


@dataclass(frozen=True)
class TurnEvent:
    """Immutable record of one agent turn stored in log.jsonl.

    Attributes:
        turn_id: Zero-padded string index within the session (e.g. "0000", "0042").
        session_id: Identifier for the enclosing session.
        timestamp: UTC datetime when the event was recorded (or synthetic for ingested data).
        role: Who produced this event (user, assistant, tool, system).
        type: Semantic category of the payload.
        content: Full verbatim payload text.
        raw: Original source dict before normalisation (preserved for auditability).
        tool_name: Name of the tool invoked or returning a result; None otherwise.
        file_path: Primary file path referenced in the event; None otherwise.
    """

    turn_id: str
    session_id: str
    timestamp: datetime
    role: EventRole
    type: EventType
    content: str
    raw: dict[str, Any]
    tool_name: str | None = field(default=None)
    file_path: str | None = field(default=None)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation of this event.

        Datetime is emitted as an ISO-8601 string with UTC timezone suffix.
        Enum members are emitted as their string values.
        """
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["role"] = self.role.value
        d["type"] = self.type.value
        return d

    def to_json(self) -> str:
        """Serialise to a single-line JSON string suitable for JSONL files."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnEvent":
        """Deserialise a TurnEvent from a dict (as produced by :meth:`to_dict`).

        Args:
            data: Mapping with at minimum the required TurnEvent fields.

        Returns:
            A fully reconstructed, frozen TurnEvent instance.

        Raises:
            KeyError: If a required field is absent from *data*.
            ValueError: If an enum value or timestamp string is unrecognised.
        """
        ts_raw = data["timestamp"]
        if isinstance(ts_raw, str):
            timestamp = datetime.fromisoformat(ts_raw)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        elif isinstance(ts_raw, datetime):
            timestamp = ts_raw
        else:
            raise ValueError(f"Unrecognised timestamp type: {type(ts_raw)}")

        return cls(
            turn_id=data["turn_id"],
            session_id=data["session_id"],
            timestamp=timestamp,
            role=EventRole(data["role"]),
            type=EventType(data["type"]),
            content=data["content"],
            raw=data.get("raw", {}),
            tool_name=data.get("tool_name"),
            file_path=data.get("file_path"),
        )

    @classmethod
    def from_json(cls, line: str) -> "TurnEvent":
        """Deserialise a TurnEvent from a JSONL line string.

        Args:
            line: A single JSON-encoded line as written by :meth:`to_json`.

        Returns:
            A fully reconstructed, frozen TurnEvent instance.
        """
        return cls.from_dict(json.loads(line))
