"""LogReader and TurnLoader utilities for reading TurnEvent records from log.jsonl."""

from __future__ import annotations

import logging
from pathlib import Path

from episodic_log.core.turn_event import TurnEvent

logger = logging.getLogger(__name__)


class LogReader:
    """Reads :class:`~episodic_log.core.turn_event.TurnEvent` records from a JSONL log file.

    Args:
        log_path: Absolute path to the ``log.jsonl`` file.

    Raises:
        TypeError: If *log_path* is not a :class:`pathlib.Path`.
        FileNotFoundError: If *log_path* does not exist.
    """

    def __init__(self, log_path: Path) -> None:
        if not isinstance(log_path, Path):
            raise TypeError(f"log_path must be a Path, got {type(log_path)}")
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        self._log_path = log_path

    @property
    def log_path(self) -> Path:
        """Absolute path to the underlying JSONL file."""
        return self._log_path

    def load_all(self) -> list[TurnEvent]:
        """Load every TurnEvent from the log file in order.

        Returns:
            Ordered list of all :class:`~episodic_log.core.turn_event.TurnEvent` records.

        Raises:
            ValueError: If any line cannot be parsed.
        """
        events: list[TurnEvent] = []
        with self._log_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(TurnEvent.from_json(stripped))
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Failed to parse TurnEvent at line {lineno} of {self._log_path}: {exc}"
                    ) from exc
        logger.debug("LogReader: loaded %d events from %s", len(events), self._log_path)
        return events

    def load_by_ids(self, turn_ids: list[str]) -> list[TurnEvent]:
        """Load :class:`~episodic_log.core.turn_event.TurnEvent` records matching the given turn_id strings.

        Args:
            turn_ids: Ordered list of zero-padded turn_id strings to retrieve.

        Returns:
            Events in the same order as *turn_ids*, preserving BM25 relevance ranking.
            Turn IDs not found in the log are silently skipped.
        """
        event_map: dict[str, TurnEvent] = {e.turn_id: e for e in self.load_all()}
        return [event_map[tid] for tid in turn_ids if tid in event_map]

    def count(self) -> int:
        """Return the total number of non-empty lines in the log.

        Returns:
            Integer count of TurnEvent records.
        """
        total = 0
        with self._log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    total += 1
        return total


class TurnLoader:
    """Formats :class:`~episodic_log.core.turn_event.TurnEvent` objects into human-readable blocks for prompt injection.

    The formatter converts each event into a compact header + content block so
    the calling condition can splice retrieved history directly into a prompt.

    Args:
        log_reader: Initialised :class:`LogReader` for the target session.

    Raises:
        TypeError: If *log_reader* is not a :class:`LogReader`.
    """

    def __init__(self, log_reader: LogReader) -> None:
        if not isinstance(log_reader, LogReader):
            raise TypeError(f"log_reader must be a LogReader, got {type(log_reader)}")
        self._reader = log_reader

    def load_and_format(self, turn_ids: list[str]) -> str:
        """Load turn_ids and return a formatted multi-block string.

        Args:
            turn_ids: Ordered list of zero-padded turn_id strings.

        Returns:
            Newline-separated formatted event blocks, or empty string if no IDs given.
        """
        if not turn_ids:
            return ""
        events = self._reader.load_by_ids(turn_ids)
        return "\n\n".join(_format_event(e) for e in events)

    def format_all(self, char_limit: int | None = None) -> str:
        """Load all events and return a formatted multi-block string.

        Args:
            char_limit: Optional character limit — truncates the result with a
                ``"[TRUNCATED]"`` suffix if exceeded.

        Returns:
            Formatted string of all events, optionally truncated.
        """
        events = self._reader.load_all()
        text = "\n\n".join(_format_event(e) for e in events)
        if char_limit is not None and len(text) > char_limit:
            text = text[:char_limit] + "\n\n[TRUNCATED]"
        return text


def _format_event(event: TurnEvent) -> str:
    """Render a single TurnEvent as a structured text block.

    Args:
        event: The :class:`~episodic_log.core.turn_event.TurnEvent` to format.

    Returns:
        A header line followed by the event content.
    """
    tool_suffix = f" | tool={event.tool_name}" if event.tool_name else ""
    file_suffix = f" | file={event.file_path}" if event.file_path else ""
    header = (
        f"[turn_{event.turn_id}] {event.role.value} | {event.type.value}"
        f"{tool_suffix}{file_suffix}"
    )
    return f"{header}\n{event.content}"
