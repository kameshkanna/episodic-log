"""Append-only JSONL writer for TurnEvent records.

The log is the immutable ground truth of a session.  Once written, no event
may be modified or deleted — only new events may be appended.  This invariant
is the core property that distinguishes episodic-log from mutable memory systems.
"""

from __future__ import annotations

import logging
from pathlib import Path

from episodic_log.core.turn_event import TurnEvent

logger = logging.getLogger(__name__)


class LogWriter:
    """Append-only writer for a single session's ``log.jsonl`` file.

    The writer ensures:

    * All writes are append-only — the file is opened in ``"a"`` mode.
    * Each :class:`~episodic_log.core.turn_event.TurnEvent` occupies exactly
      one UTF-8 line terminated by ``"\\n"``.
    * The parent directory is created on first use if it does not exist.

    Args:
        log_path: Absolute path to the target ``log.jsonl`` file.

    Raises:
        TypeError: If *log_path* is not a :class:`pathlib.Path`.
    """

    def __init__(self, log_path: Path) -> None:
        if not isinstance(log_path, Path):
            raise TypeError(f"log_path must be a Path, got {type(log_path)}")
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def log_path(self) -> Path:
        """Absolute path to the underlying JSONL file."""
        return self._log_path

    def append(self, event: TurnEvent) -> None:
        """Append one :class:`~episodic_log.core.turn_event.TurnEvent` to the log.

        The parent directory is created automatically on the first call.

        Args:
            event: The :class:`~episodic_log.core.turn_event.TurnEvent` to persist.

        Raises:
            TypeError: If *event* is not a :class:`~episodic_log.core.turn_event.TurnEvent`.
            OSError: If the file cannot be opened or written.
        """
        if not isinstance(event, TurnEvent):
            raise TypeError(f"event must be a TurnEvent, got {type(event)}")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(event.to_json() + "\n")
        logger.debug("LogWriter: appended turn_id=%s to %s", event.turn_id, self._log_path)

    def append_batch(self, events: list[TurnEvent]) -> None:
        """Append multiple events in a single file-open operation.

        Args:
            events: Ordered list of :class:`~episodic_log.core.turn_event.TurnEvent` objects.

        Raises:
            TypeError: If *events* is not a list or any element is not a TurnEvent.
        """
        if not isinstance(events, list):
            raise TypeError(f"events must be a list, got {type(events)}")
        if not events:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            for event in events:
                if not isinstance(event, TurnEvent):
                    raise TypeError(f"All elements must be TurnEvent, got {type(event)}")
                fh.write(event.to_json() + "\n")
        logger.debug(
            "LogWriter: appended %d events to %s", len(events), self._log_path
        )
