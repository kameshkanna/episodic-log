"""Core TurnSummary data contract for the BM25 search index.

TurnSummary records are stored separately from the execution log and are used
exclusively as a search index.  The agent never reads summaries directly — it
retrieves verbatim TurnEvent records from log.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TurnSummary:
    """One-line summary of a TurnEvent used as a BM25 search index entry.

    Attributes:
        turn_id: Zero-padded string matching the corresponding TurnEvent.turn_id.
        session_id: Identifier for the enclosing session.
        summary: Human-readable (or model-generated) description of the turn.
        method: Which summarizer produced this record ("structured", "haiku", "self").
    """

    turn_id: str
    session_id: str
    summary: str
    method: str

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation of this summary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialise to a single-line JSON string suitable for JSONL files."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnSummary":
        """Deserialise a TurnSummary from a dict (as produced by :meth:`to_dict`).

        Args:
            data: Mapping with at minimum the required TurnSummary fields.

        Returns:
            A fully reconstructed, frozen TurnSummary instance.

        Raises:
            KeyError: If a required field is absent from *data*.
        """
        return cls(
            turn_id=data["turn_id"],
            session_id=data["session_id"],
            summary=data["summary"],
            method=data["method"],
        )

    @classmethod
    def from_json(cls, line: str) -> "TurnSummary":
        """Deserialise a TurnSummary from a JSONL line string.

        Args:
            line: A single JSON-encoded line as written by :meth:`to_json`.

        Returns:
            A fully reconstructed, frozen TurnSummary instance.
        """
        return cls.from_dict(json.loads(line))
