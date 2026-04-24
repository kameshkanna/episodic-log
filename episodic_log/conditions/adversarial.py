"""Adversarial condition — retrieves real turns but shifts ordinals ±10 to corrupt grounding."""

from __future__ import annotations

import logging

from episodic_log.conditions.base import ConditionResult
from episodic_log.conditions.episodic import EpisodicCondition
from episodic_log.retrieval.summary_store import SummaryStore

logger = logging.getLogger(__name__)

_SHIFT = 10


class AdversarialCondition(EpisodicCondition):
    """Adversarial corruption of retrieved turn IDs.

    After BM25 retrieval, each retrieved turn ordinal is shifted by ±:data:`_SHIFT`
    (alternating sign) so the model receives verbatim content from *different* turns
    than those it queried for.  This tests whether the model commits commission errors
    when grounded by misaligned verbatim text.

    Turn IDs that fall out of the valid range ``[0000, …, 9999]`` are clamped.
    """

    name: str = "adversarial"

    def _retrieve(self, store: SummaryStore, query: str) -> list[str]:
        """Run BM25 retrieval then shift the returned turn ordinals.

        Args:
            store: Initialised :class:`~episodic_log.retrieval.summary_store.SummaryStore`.
            query: Free-text query string.

        Returns:
            Shifted list of zero-padded turn_id strings.
        """
        original_ids = super()._retrieve(store, query)
        return _shift_ids(original_ids)


def _shift_ids(turn_ids: list[str]) -> list[str]:
    """Apply alternating ±:data:`_SHIFT` shift to a list of turn_id strings.

    Args:
        turn_ids: Ordered list of zero-padded turn_id strings.

    Returns:
        New list with shifted ordinals, clamped to ``[0, 9999]``.
    """
    shifted: list[str] = []
    for i, tid in enumerate(turn_ids):
        ordinal = int(tid)
        delta = _SHIFT if i % 2 == 0 else -_SHIFT
        new_ordinal = max(0, min(9999, ordinal + delta))
        shifted.append(str(new_ordinal).zfill(4))
    return shifted
