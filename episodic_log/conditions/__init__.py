"""Condition registry for the CHD evaluation harness.

Provides :data:`ALL_CONDITIONS`, a pre-instantiated mapping of every supported
condition name to its :class:`~episodic_log.conditions.base.BaseCondition`
instance, and :func:`get_condition` for validated lookup.

Supported condition names
-------------------------
- ``"amnesiac"``         — no memory, pure parametric knowledge.
- ``"recall/lexical"``   — tool-use recall with a lexical summary index.
- ``"recall/scout"``     — tool-use recall with a scout summary index.
- ``"recall/echo"``      — tool-use recall with an echo summary index.
"""

from __future__ import annotations

from episodic_log.conditions.amnesiac import AmnesiacCondition
from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.conditions.recall import RecallCondition

__all__ = [
    "ALL_CONDITIONS",
    "get_condition",
    "AmnesiacCondition",
    "RecallCondition",
    "BaseCondition",
    "ConditionResult",
]

ALL_CONDITIONS: dict[str, BaseCondition] = {
    "amnesiac": AmnesiacCondition(),
    "recall/lexical": RecallCondition("lexical"),
    "recall/scout": RecallCondition("scout"),
    "recall/echo": RecallCondition("echo"),
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
