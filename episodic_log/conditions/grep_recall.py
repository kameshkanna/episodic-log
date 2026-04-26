"""GrepRecall condition — model formulates keywords, greps summaries, loads turns.

Unlike RecallCondition (which dumps all summaries upfront), this condition
gives the model only the question and two tools:

  grep_memory(keywords)  — keyword grep over the summary file; returns
                            all matching summary lines with their turn IDs.
  load_turn(turn_id)     — loads the full verbatim content of a turn.

The model must decide WHAT to search for.  It generates keywords, receives
matching summary lines, then calls load_turn on the ones it wants to read.

Ablation value:
  - Compare with recall/<method> (all summaries in context) to measure whether
    model-driven keyword search is better/worse than seeing all summaries.
  - The grep is simple substring matching — the model chooses the query, not
    a retrieval system.  This is NOT BM25 (no TF-IDF weighting or scoring).
"""

from __future__ import annotations

import logging

from episodic_log.agent.loop import AgentLoop
from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_VALID_SUMMARY_METHODS: frozenset[str] = frozenset({"lexical", "scout", "echo"})

_REQUIRED_META_KEYS: frozenset[str] = frozenset(
    {"session_id", "question_id", "question", "answer"}
)


class GrepRecallCondition(BaseCondition):
    """Agent-loop condition where the model greps summaries before loading turns.

    The model sees only the question.  It calls ``grep_memory`` with
    self-chosen keywords to find relevant summary lines, then calls
    ``load_turn`` to read the verbatim content of interesting turns.

    Args:
        summary_method: Summary index to grep; one of ``"lexical"``,
            ``"scout"``, or ``"echo"``.
        max_tool_calls: Combined budget for grep_memory + load_turn calls.
            Defaults to 15.

    Raises:
        ValueError: If *summary_method* is not accepted or *max_tool_calls* < 1.
    """

    def __init__(self, summary_method: str, max_tool_calls: int = 15) -> None:
        if summary_method not in _VALID_SUMMARY_METHODS:
            raise ValueError(
                f"summary_method must be one of {sorted(_VALID_SUMMARY_METHODS)}, "
                f"got {summary_method!r}"
            )
        if max_tool_calls < 1:
            raise ValueError(
                f"max_tool_calls must be a positive integer, got {max_tool_calls}"
            )
        self._summary_method = summary_method
        self._max_tool_calls = max_tool_calls

    @property
    def name(self) -> str:
        """Canonical condition name, e.g. ``"grep_recall/lexical"``."""
        return f"grep_recall/{self._summary_method}"

    def run(self, session_meta: dict, provider: BaseProvider) -> ConditionResult:
        """Run the grep-search agent loop and return the result.

        Args:
            session_meta: Must contain ``session_id``, ``question_id``,
                ``question``, ``answer``, ``log_path``, ``summaries_dir``.
            provider: Initialised LLM provider backend.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult`.

        Raises:
            KeyError: If required keys are absent from *session_meta*.
            TypeError: If *provider* is not a BaseProvider.
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
            "GrepRecallCondition.run: condition=%s session_id=%s question_id=%s",
            self.name, session_id, question_id,
        )

        loop = AgentLoop(
            provider=provider,
            max_tool_calls=self._max_tool_calls,
            mode="grep_and_load",
        )
        trace = loop.run(
            question=question,
            session_meta=session_meta,
            summary_method=self._summary_method,
        )

        return ConditionResult(
            session_id=session_id,
            question_id=question_id,
            question=question,
            ground_truth=ground_truth,
            predicted_answer=trace.answer.strip(),
            condition=self.name,
            summary_method=self._summary_method,
            tool_calls=[tc.to_dict() for tc in trace.tool_calls],
            turns_loaded=list(trace.turns_loaded),
        )
