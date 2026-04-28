"""Condition registry for the CHD evaluation harness.

Supported condition names
-------------------------
- ``"amnesiac"``               — no memory, pure parametric knowledge.
- ``"recall/lexical"``         — summary index + agent load_turn (lexical summaries).
- ``"recall/scout"``           — summary index + agent load_turn (scout summaries).
- ``"recall/echo"``            — summary index + agent load_turn (echo summaries).
- ``"grep_recall/lexical"``    — grep_memory (top-3 auto-loaded) + load_turn (lexical).
- ``"grep_recall/scout"``      — grep_memory (top-3 auto-loaded) + load_turn (scout).
- ``"grep_recall/echo"``       — grep_memory (top-3 auto-loaded) + load_turn (echo).
"""

from __future__ import annotations

from episodic_log.conditions.amnesiac import AmnesiacCondition
from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.conditions.grep_recall import GrepRecallCondition
from episodic_log.conditions.recall import RecallCondition

__all__ = [
    "ALL_CONDITIONS",
    "get_condition",
    "AmnesiacCondition",
    "RecallCondition",
    "GrepRecallCondition",
    "BaseCondition",
    "ConditionResult",
]

_SUMMARY_METHODS = ("lexical", "scout", "echo")

ALL_CONDITIONS: dict[str, BaseCondition] = {
    "amnesiac": AmnesiacCondition(),
    **{f"recall/{m}": RecallCondition(m, max_tool_calls=8) for m in _SUMMARY_METHODS},
    **{f"grep_recall/{m}": GrepRecallCondition(m, max_tool_calls=5) for m in _SUMMARY_METHODS},
}


def get_condition(name: str) -> BaseCondition:
    """Return the condition registered under *name*.

    Args:
        name: One of the keys in :data:`ALL_CONDITIONS`.

    Returns:
        The corresponding :class:`~episodic_log.conditions.base.BaseCondition`
        instance.

    Raises:
        ValueError: If *name* is not a registered condition.
    """
    if name not in ALL_CONDITIONS:
        raise ValueError(
            f"Unknown condition: {name!r}. Available: {sorted(ALL_CONDITIONS)}"
        )
    return ALL_CONDITIONS[name]
