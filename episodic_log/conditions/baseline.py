"""Baseline condition — no memory access, single generate call."""

from __future__ import annotations

import logging

from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.ingestor.longmemeval import IngestedSession
from episodic_log.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question concisely and directly. "
    "Do not ask clarifying questions."
)


class BaselineCondition(BaseCondition):
    """No memory condition.

    The model answers purely from parametric knowledge with no access to the
    session log.  This establishes the lower-bound for CHD measurement.
    """

    name: str = "baseline"

    def __init__(self, provider: BaseProvider, summary_method: str = "structured") -> None:
        super().__init__(provider, summary_method)

    def run(self, session: IngestedSession, question: str) -> ConditionResult:
        """Answer *question* with no log access.

        Args:
            session: Ingested session (only session_id is used for bookkeeping).
            question: Free-text question.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult` with empty
            retrieved_turn_ids and num_retrieval_calls=0.
        """
        logger.debug("BaselineCondition: session=%s", session.session_id)
        answer = self._provider.generate(
            messages=[question],
            system=_SYSTEM_PROMPT,
            max_tokens=256,
            temperature=0.0,
        )
        return ConditionResult(
            condition_name=self.name,
            session_id=session.session_id,
            question=question,
            predicted_answer=answer.strip(),
        )
