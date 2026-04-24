"""Evaluation conditions for CHD measurement on LongMemEval."""

from episodic_log.conditions.adversarial import AdversarialCondition
from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.conditions.baseline import BaselineCondition
from episodic_log.conditions.episodic import EpisodicCondition
from episodic_log.conditions.external import ExternalCondition
from episodic_log.conditions.full_context import FullContextCondition
from episodic_log.conditions.md_memory import MdMemoryCondition
from episodic_log.conditions.proactive import ProactiveCondition

ALL_CONDITIONS: list[str] = [
    "baseline",
    "episodic",
    "adversarial",
    "proactive",
    "external",
    "md_memory",
    "full_context",
]

_REGISTRY: dict[str, type[BaseCondition]] = {
    "baseline": BaselineCondition,
    "episodic": EpisodicCondition,
    "adversarial": AdversarialCondition,
    "proactive": ProactiveCondition,
    "external": ExternalCondition,
    "md_memory": MdMemoryCondition,
    "full_context": FullContextCondition,
}


def get_condition(name: str, **kwargs) -> BaseCondition:
    """Instantiate a condition by name.

    Args:
        name: One of :data:`ALL_CONDITIONS`.
        **kwargs: Forwarded to the condition constructor (e.g. ``provider=``, ``summary_method=``).

    Returns:
        An initialised :class:`~episodic_log.conditions.base.BaseCondition` subclass.

    Raises:
        KeyError: If *name* is not a registered condition.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown condition '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "ALL_CONDITIONS",
    "get_condition",
    "BaseCondition",
    "ConditionResult",
    "BaselineCondition",
    "EpisodicCondition",
    "AdversarialCondition",
    "ProactiveCondition",
    "ExternalCondition",
    "MdMemoryCondition",
    "FullContextCondition",
]
