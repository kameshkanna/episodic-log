"""Amnesiac condition — model answers with no memory tools or log access."""

from __future__ import annotations

import logging

from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_REQUIRED_META_KEYS: frozenset[str] = frozenset(
    {"session_id", "question_id", "question", "answer"}
)


class AmnesiacCondition(BaseCondition):
    """No-memory baseline condition.

    The model answers purely from parametric knowledge with no access to the
    session log, summaries, or any retrieval tools.  This establishes the
    lower-bound for Conversational History Degradation (CHD) measurement.

    The system prompt intentionally avoids any hint about memory retrieval so
    that behaviour is strictly parametric.
    """

    SYSTEM_PROMPT: str = (
        "You are a helpful assistant. Answer the question directly and concisely. "
        "If you don't know, say so."
    )

    @property
    def name(self) -> str:
        """Canonical condition name."""
        return "amnesiac"

    def run(self, session_meta: dict, provider: BaseProvider) -> ConditionResult:
        """Answer the question with no log access.

        Performs a single :meth:`~episodic_log.providers.base.BaseProvider.generate`
        call using only the question text.  Retrieved turn IDs and tool calls are
        always empty for this condition.

        Args:
            session_meta: Metadata dict containing at minimum ``session_id``,
                ``question_id``, ``question``, and ``answer``.
            provider: Initialised LLM provider backend.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult` with empty
            ``tool_calls`` and ``turns_loaded``.

        Raises:
            KeyError: If a required key is absent from *session_meta*.
            TypeError: If *provider* is not a
                :class:`~episodic_log.providers.base.BaseProvider`.
        """
        missing = _REQUIRED_META_KEYS - session_meta.keys()
        if missing:
            raise KeyError(f"session_meta is missing required keys: {sorted(missing)}")
        if not isinstance(provider, BaseProvider):
            raise TypeError(f"provider must be a BaseProvider, got {type(provider)}")

        session_id: str = session_meta["session_id"]
        question_id: str = session_meta["question_id"]
        question: str = session_meta["question"]
        ground_truth: str = session_meta["answer"]

        logger.debug(
            "AmnesiacCondition.run: session_id=%s question_id=%s",
            session_id,
            question_id,
        )

        predicted_answer: str = provider.generate(
            messages=[question],
            system=self.SYSTEM_PROMPT,
            max_tokens=256,
            temperature=0.0,
        )

        logger.debug(
            "AmnesiacCondition.run: completed session_id=%s question_id=%s",
            session_id,
            question_id,
        )

        return ConditionResult(
            session_id=session_id,
            question_id=question_id,
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted_answer.strip(),
            condition=self.name,
            summary_method=None,
            tool_calls=[],
            turns_loaded=[],
        )

    def run_batch(
        self, sessions: list[dict], provider: BaseProvider
    ) -> list[ConditionResult]:
        """Answer all sessions in one generate_batch call.

        Args:
            sessions: List of session metadata dicts.
            provider: Initialised LLM provider with ``generate_batch``.

        Returns:
            Ordered list of :class:`ConditionResult`.
        """
        messages = [[{"role": "user", "content": s["question"]}] for s in sessions]
        answers = provider.generate_batch(
            messages, system=self.SYSTEM_PROMPT, max_tokens=256, temperature=0.0
        )
        return [
            ConditionResult(
                session_id=s["session_id"],
                question_id=s["question_id"],
                question=s["question"],
                ground_truth=s["answer"],
                predicted_answer=a.strip(),
                condition=self.name,
                summary_method=None,
                tool_calls=[],
                turns_loaded=[],
            )
            for s, a in zip(sessions, answers)
        ]
