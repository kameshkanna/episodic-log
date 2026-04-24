"""Markdown memory condition — compress entire log to ≤4000-char markdown, then answer."""

from __future__ import annotations

import logging

from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.core.log_reader import LogReader, TurnLoader
from episodic_log.ingestor.longmemeval import IngestedSession
from episodic_log.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_CHAR_LIMIT = 4000

_COMPRESS_SYSTEM = (
    "Compress the following conversation log into a concise markdown summary of at most "
    f"{_CHAR_LIMIT} characters.  Preserve key facts, decisions, and entities. "
    "Use bullet points."
)

_ANSWER_SYSTEM = (
    "You are a helpful assistant.  Use the provided memory summary to answer the question. "
    "Be concise and direct."
)


class MdMemoryCondition(BaseCondition):
    """Markdown-compressed memory condition.

    Two-step process:
    1. Compress the full session log to a markdown summary ≤ :data:`_CHAR_LIMIT` chars.
    2. Inject the summary into the system prompt and answer in one shot.

    This tests whether lossy compression of episodic memory degrades grounding
    compared to verbatim retrieval (Episodic condition).
    """

    name: str = "md_memory"

    def __init__(self, provider: BaseProvider, summary_method: str = "structured") -> None:
        super().__init__(provider, summary_method)

    def run(self, session: IngestedSession, question: str) -> ConditionResult:
        """Compress session log to markdown, then answer *question*.

        Args:
            session: Ingested session with log_path.
            question: Free-text question.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult`.
        """
        reader = LogReader(session.log_path)
        loader = TurnLoader(reader)

        full_log = loader.format_all(char_limit=16_000)

        md_summary = self._provider.generate(
            messages=[full_log],
            system=_COMPRESS_SYSTEM,
            max_tokens=512,
            temperature=0.0,
        )

        if len(md_summary) > _CHAR_LIMIT:
            md_summary = md_summary[:_CHAR_LIMIT] + "\n\n[TRUNCATED]"

        answer_system = (
            f"{_ANSWER_SYSTEM}\n\n<memory_summary>\n{md_summary}\n</memory_summary>"
        )
        answer = self._provider.generate(
            messages=[question],
            system=answer_system,
            max_tokens=256,
            temperature=0.0,
        )

        return ConditionResult(
            condition_name=self.name,
            session_id=session.session_id,
            question=question,
            predicted_answer=answer.strip(),
            metadata={"md_summary_chars": len(md_summary)},
        )
