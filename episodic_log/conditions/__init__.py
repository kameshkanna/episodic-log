"""Condition registry for the CHD evaluation harness.

Provides :data:`ALL_CONDITIONS`, a pre-instantiated mapping of every supported
condition name to its :class:`~episodic_log.conditions.base.BaseCondition`
instance, and :func:`get_condition` for validated lookup.

Supported condition names
-------------------------
- ``"amnesiac"``               — no memory, pure parametric knowledge.
- ``"recall/lexical"``         — summary dump + agent load_turn with lexical summaries.
- ``"recall/scout"``           — summary dump + agent load_turn with scout summaries.
- ``"recall/echo"``            — summary dump + agent load_turn with echo summaries.
- ``"topk/lexical/k3"``        — pre-inject top-3 turns by keyword overlap, one-shot answer.
- ``"topk/lexical/k5"``        — pre-inject top-5 turns by keyword overlap, one-shot answer.
- ``"topk/lexical/k10"``       — pre-inject top-10 turns by keyword overlap, one-shot answer.
- ``"topk/scout/k3"``          — same with scout summaries.
- ``"topk/scout/k5"``
- ``"topk/scout/k10"``
- ``"topk/echo/k3"``           — same with echo summaries.
- ``"topk/echo/k5"``
- ``"topk/echo/k10"``
"""

from __future__ import annotations

from episodic_log.conditions.amnesiac import AmnesiacCondition
from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.conditions.grep_recall import GrepRecallCondition
from episodic_log.conditions.recall import RecallCondition
from episodic_log.conditions.topk import TopKCondition

__all__ = [
    "ALL_CONDITIONS",
    "get_condition",
    "AmnesiacCondition",
    "RecallCondition",
    "GrepRecallCondition",
    "TopKCondition",
    "BaseCondition",
    "ConditionResult",
]

_SUMMARY_METHODS = ("lexical", "scout", "echo")
_TOPK_VALUES = (3, 5, 10)

ALL_CONDITIONS: dict[str, BaseCondition] = {
    "amnesiac": AmnesiacCondition(),
    **{f"recall/{m}": RecallCondition(m) for m in _SUMMARY_METHODS},
    **{f"grep_recall/{m}": GrepRecallCondition(m) for m in _SUMMARY_METHODS},
    **{
        f"topk/{m}/k{k}": TopKCondition(m, k)
        for m in _SUMMARY_METHODS
        for k in _TOPK_VALUES
    },
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
