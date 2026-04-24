"""Full-turn loader tool for the CHD evaluation agent.

Scans ``log.jsonl`` for a specific turn by ID and returns a formatted
string representation suitable for inclusion in the agent's context window.
"""

from __future__ import annotations

import logging
from pathlib import Path

from episodic_log.core.turn_event import TurnEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI-format tool schema
# ---------------------------------------------------------------------------

LOAD_TURN_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "load_turn",
        "description": (
            "Load the full content of a specific conversation turn by its ID. "
            "Returns the verbatim turn text including role and content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "turn_id": {
                    "type": "string",
                    "description": "Zero-padded turn identifier (e.g. '0042')",
                },
            },
            "required": ["turn_id"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


def load_turn(turn_id: str, log_path: Path) -> str:
    """Scan ``log.jsonl`` and return the formatted content of the requested turn.

    Reads the log file line-by-line (streaming) so that arbitrarily large logs
    are handled without loading the entire file into memory.

    Args:
        turn_id: Zero-padded string identifier for the turn (e.g. ``"0042"``).
        log_path: Absolute path to the ``log.jsonl`` file for this session.

    Returns:
        A human-readable string in the format::

            [ROLE | turn XXXX]
            Content: <verbatim turn content>

        If the turn is not found, returns ``"Turn {turn_id} not found."``.

    Raises:
        TypeError: If *turn_id* is not a string or *log_path* is not a Path.
        FileNotFoundError: If *log_path* does not exist.
    """
    if not isinstance(turn_id, str):
        raise TypeError(f"turn_id must be a str, got {type(turn_id)}")
    if not isinstance(log_path, Path):
        raise TypeError(f"log_path must be a Path, got {type(log_path)}")
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event: TurnEvent = TurnEvent.from_json(stripped)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "load_turn: skipping malformed line %d in %s: %s",
                    lineno,
                    log_path,
                    exc,
                )
                continue

            if event.turn_id == turn_id:
                role_label = event.role.value.upper()
                formatted = f"[{role_label} | turn {event.turn_id}]\nContent: {event.content}"
                if event.tool_name:
                    formatted += f"\nTool: {event.tool_name}"
                if event.file_path:
                    formatted += f"\nFile: {event.file_path}"
                logger.debug("load_turn: found turn_id=%s at line %d.", turn_id, lineno)
                return formatted

    logger.warning("load_turn: turn_id=%s not found in %s.", turn_id, log_path)
    return f"Turn {turn_id} not found."
