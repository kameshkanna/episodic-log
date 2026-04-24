"""Abstract base class and result type for all evaluation conditions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from episodic_log.ingestor.longmemeval import IngestedSession
from episodic_log.providers.base import BaseProvider


@dataclass
class ConditionResult:
    """Result of running one condition on one session.

    Attributes:
        condition_name: Name of the condition (e.g. ``"baseline"``, ``"episodic"``).
        session_id: Session identifier from :class:`~episodic_log.ingestor.longmemeval.IngestedSession`.
        question: The evaluation question.
        predicted_answer: The model's answer string.
        retrieved_turn_ids: Turn IDs fetched from the log (empty for non-retrieval conditions).
        num_retrieval_calls: Number of BM25 retrieval calls made during answering.
        metadata: Optional extra data (timing, token counts, etc.).
    """

    condition_name: str
    session_id: str
    question: str
    predicted_answer: str
    retrieved_turn_ids: list[str] = field(default_factory=list)
    num_retrieval_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseCondition(ABC):
    """Abstract condition that defines how a model answers a LongMemEval question.

    Each concrete subclass encodes one memory strategy (e.g. no memory, BM25
    retrieval, full context injection) and returns a :class:`ConditionResult`.

    Args:
        provider: An initialised :class:`~episodic_log.providers.base.BaseProvider`.
        summary_method: Summarizer method to use for BM25 index construction.
            Ignored by conditions that do not perform retrieval.

    Raises:
        TypeError: If *provider* is not a :class:`~episodic_log.providers.base.BaseProvider`.
    """

    name: str = "base"

    def __init__(self, provider: BaseProvider, summary_method: str = "structured") -> None:
        if not isinstance(provider, BaseProvider):
            raise TypeError(f"provider must be a BaseProvider, got {type(provider)}")
        self._provider = provider
        self._summary_method = summary_method

    @abstractmethod
    def run(self, session: IngestedSession, question: str) -> ConditionResult:
        """Answer *question* using the strategy encoded by this condition.

        Args:
            session: Ingested session providing the log, summaries, and metadata.
            question: Free-text question to answer.

        Returns:
            A :class:`ConditionResult` with the model's predicted answer and
            any retrieval metadata.
        """
        raise NotImplementedError
