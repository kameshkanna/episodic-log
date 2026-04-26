"""Full-turn loader tool for the CHD evaluation agent.

Performs an O(1) lookup in a pre-built ``turn_id → TurnEvent`` dict built by
:func:`~episodic_log.tools.session_tools.make_session_tools`.  No linear
file scan occurs at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
            "Load the full verbatim content of a conversation turn by its ID. "
            "Use grep_memory first to find relevant turn IDs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "turn_id": {
                    "type": "string",
                    "description": "Turn identifier returned by grep_memory (e.g. '0042')",
                },
            },
            "required": ["turn_id"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


def load_turn(
    turn_id: str,
    turn_map: "dict[str, TurnEvent]",
) -> str:
    """Return the formatted content of a turn via O(1) dict lookup.

    Args:
        turn_id: String identifier for the turn (e.g. ``"0042"``).
        turn_map: Pre-built mapping from turn_id strings to
            :class:`~episodic_log.core.turn_event.TurnEvent` objects,
            constructed in :func:`~episodic_log.tools.session_tools.make_session_tools`.

    Returns:
        A human-readable string in the format::

            [ROLE | turn XXXX]
            Content: <verbatim turn content>

        Returns ``"Turn {turn_id} not found."`` if the ID is not in the map.

    Raises:
        TypeError: If *turn_id* is not a string.
    """
    if not isinstance(turn_id, str):
        raise TypeError(f"turn_id must be a str, got {type(turn_id)}")

    # Normalise: strip whitespace and leading zeros are preserved as-is since
    # turn_ids are zero-padded strings (e.g. "0042" not "42").
    turn_id = turn_id.strip()

    event = turn_map.get(turn_id)
    if event is None:
        logger.warning("load_turn: turn_id=%r not found in turn_map.", turn_id)
        return f"Turn {turn_id} not found. Check the turn_id from grep_memory results."

    role_label = event.role.value.upper()
    formatted = f"[{role_label} | turn {event.turn_id}]\nContent: {event.content}"
    if event.tool_name:
        formatted += f"\nTool: {event.tool_name}"
    if event.file_path:
        formatted += f"\nFile: {event.file_path}"

    logger.debug("load_turn: found turn_id=%r", turn_id)
    return formatted
