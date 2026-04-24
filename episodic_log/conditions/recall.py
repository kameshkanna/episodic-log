"""Recall condition — tool-use memory via grep_memory + load_turn agent loop."""

from __future__ import annotations

import logging

from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_VALID_SUMMARY_METHODS: frozenset[str] = frozenset({"lexical", "scout", "echo"})

_REQUIRED_META_KEYS: frozenset[str] = frozenset(
    {"session_id", "question_id", "question", "answer"}
)


class RecallCondition(BaseCondition):
    """Tool-use memory condition using a summarised index.

    The model is given access to two tools — ``grep_memory`` (BM25 search over
    a pre-built summary index) and ``load_turn`` (load a raw conversation turn
    by ID) — and must answer the question by retrieving relevant evidence from
    the session log.

    The summary index is constructed from one of three summariser variants,
    controlled by *summary_method*.

    Args:
        summary_method: Index construction strategy; one of ``"lexical"``,
            ``"scout"``, or ``"echo"``.
        max_tool_calls: Upper bound on the number of tool invocations the agent
            loop is allowed per question.  Defaults to 8.

    Raises:
        ValueError: If *summary_method* is not one of the accepted values.
    """

    def __init__(self, summary_method: str, max_tool_calls: int = 8) -> None:
        if summary_method not in _VALID_SUMMARY_METHODS:
            raise ValueError(
                f"summary_method must be one of {sorted(_VALID_SUMMARY_METHODS)}, "
                f"got {summary_method!r}"
            )
        if max_tool_calls < 1:
            raise ValueError(
                f"max_tool_calls must be a positive integer, got {max_tool_calls}"
            )
        self._summary_method: str = summary_method
        self._max_tool_calls: int = max_tool_calls

    @property
    def name(self) -> str:
        """Canonical condition name, e.g. ``"recall/lexical"``."""
        return f"recall/{self._summary_method}"

    def run(self, session_meta: dict, provider: BaseProvider) -> ConditionResult:
        """Answer the question by running the agent loop with memory tools.

        Delegates to :class:`~episodic_log.agent.loop.AgentLoop` which manages
        the multi-turn tool-calling cycle.  The resulting
        :class:`~episodic_log.agent.trace.AgentTrace` is unpacked into a
        :class:`~episodic_log.conditions.base.ConditionResult`.

        Args:
            session_meta: Metadata dict containing at minimum ``session_id``,
                ``question_id``, ``question``, and ``answer``.  Additional keys
                (``log_path``, ``summaries_dir``, ``evidence_turn_ids``,
                ``question_type``) are forwarded to the agent loop.
            provider: Initialised LLM provider backend.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult` populated
            with the agent's final answer, all serialised tool call records, and
            the list of turn IDs that were loaded during the run.

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

        # Import deferred to avoid a hard dependency when the agent module is not
        # yet present; the import will fail loudly at call time if the module is
        # missing, which is the correct fail-fast behaviour.
        from episodic_log.agent.loop import AgentLoop  # noqa: PLC0415

        session_id: str = session_meta["session_id"]
        question_id: str = session_meta["question_id"]
        question: str = session_meta["question"]
        ground_truth: str = session_meta["answer"]

        logger.debug(
            "RecallCondition.run: condition=%s session_id=%s question_id=%s",
            self.name,
            session_id,
            question_id,
        )

        loop = AgentLoop(provider=provider, max_tool_calls=self._max_tool_calls)
        trace = loop.run(
            question=question,
            session_meta=session_meta,
            summary_method=self._summary_method,
        )

        logger.debug(
            "RecallCondition.run: completed condition=%s session_id=%s "
            "question_id=%s total_tool_calls=%d turns_loaded=%d",
            self.name,
            session_id,
            question_id,
            trace.total_tool_calls,
            len(trace.turns_loaded),
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
