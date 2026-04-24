"""Abstract base class and result type for all evaluation conditions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from episodic_log.providers.base import BaseProvider


@dataclass
class ConditionResult:
    """Structured result from running a single condition on a single session question.

    Attributes:
        session_id: Identifier of the session being evaluated.
        question_id: Identifier of the question within the session.
        question: The evaluation question text.
        ground_truth: The expected correct answer.
        predicted_answer: The model's generated answer.
        condition: Condition label, one of ``"amnesiac"``, ``"recall/lexical"``,
            ``"recall/scout"``, or ``"recall/echo"``.
        summary_method: Summarizer variant used to build the index; ``None`` for
            amnesiac (no index).
        tool_calls: Serialized :class:`~episodic_log.agent.trace.ToolCallRecord`
            dicts, empty for amnesiac.
        turns_loaded: Turn IDs loaded from the log during answering, empty for
            amnesiac.
    """

    session_id: str
    question_id: str
    question: str
    ground_truth: str
    predicted_answer: str
    condition: str
    summary_method: str | None
    tool_calls: list[dict] = field(default_factory=list)
    turns_loaded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise this result to a plain dictionary for JSON export.

        Returns:
            Dictionary representation of the result with all fields serialised
            to JSON-compatible types.
        """
        return {
            "session_id": self.session_id,
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "predicted_answer": self.predicted_answer,
            "condition": self.condition,
            "summary_method": self.summary_method,
            "tool_calls": self.tool_calls,
            "turns_loaded": self.turns_loaded,
        }


class BaseCondition(ABC):
    """Abstract condition encoding one memory strategy for CHD evaluation.

    Each concrete subclass implements a distinct approach to answering questions
    about past conversations.  All conditions receive a ``session_meta`` dict
    rather than a typed object to remain loosely coupled to the ingestor layer.

    ``session_meta`` must contain the following keys:

    - ``session_id`` (str): Unique session identifier.
    - ``question_id`` (str): Unique question identifier.
    - ``question`` (str): Free-text question to answer.
    - ``answer`` (str): Ground-truth answer string.
    - ``log_path`` (str): Path to the raw conversation log file.
    - ``summaries_dir`` (str): Directory containing pre-built summary index files.
    - ``evidence_turn_ids`` (list[str]): Turn IDs that contain the ground-truth evidence.
    - ``question_type`` (str): Category label for the question.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical condition name used for logging and output labelling."""
        raise NotImplementedError

    @abstractmethod
    def run(self, session_meta: dict, provider: BaseProvider) -> ConditionResult:
        """Answer the question encoded in *session_meta* and return a result.

        Args:
            session_meta: Metadata dict for the session/question (see class
                docstring for required keys).
            provider: Initialised LLM provider backend.

        Returns:
            :class:`ConditionResult` populated with the predicted answer and any
            retrieval bookkeeping.

        Raises:
            KeyError: If a required key is missing from *session_meta*.
            TypeError: If *provider* is not a
                :class:`~episodic_log.providers.base.BaseProvider`.
        """
        raise NotImplementedError
