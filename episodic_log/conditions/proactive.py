"""Proactive condition — top-5 turns pre-injected into system prompt, one-shot answer."""

from __future__ import annotations

import logging

from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.core.log_reader import LogReader, TurnLoader
from episodic_log.ingestor.longmemeval import IngestedSession
from episodic_log.providers.base import BaseProvider
from episodic_log.retrieval.summary_store import SummaryStore

logger = logging.getLogger(__name__)

_TOP_K = 5

_SYSTEM_TEMPLATE = """\
You are a helpful assistant with access to the following conversation history.

<retrieved_context>
{context}
</retrieved_context>

Answer the question using the context above.  If the answer is not in the
context, say so.  Be concise and direct.
"""


class ProactiveCondition(BaseCondition):
    """Proactive memory injection.

    Before generating, the top-:data:`_TOP_K` BM25 hits for the question are
    retrieved and injected into the system prompt.  No iterative tool use — the
    model answers in a single generate call.  This tests whether pre-loading
    context improves recall without requiring the model to issue retrieval calls.
    """

    name: str = "proactive"

    def __init__(self, provider: BaseProvider, summary_method: str = "structured") -> None:
        super().__init__(provider, summary_method)

    def run(self, session: IngestedSession, question: str) -> ConditionResult:
        """Pre-inject top-5 relevant turns and answer in one shot.

        Args:
            session: Ingested session with log_path and summaries_dir.
            question: Free-text question.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult` with the
            model's predicted answer and pre-injected turn IDs.
        """
        store = SummaryStore(session.summaries_dir)
        reader = LogReader(session.log_path)
        loader = TurnLoader(reader)

        try:
            index = store.get_index(self._summary_method)
            turn_ids = index.query(question, k=_TOP_K)
        except (FileNotFoundError, ValueError):
            logger.warning(
                "No summaries for method=%s in session=%s — injecting no context.",
                self._summary_method,
                session.session_id,
            )
            turn_ids = []

        context = loader.load_and_format(turn_ids) if turn_ids else "(no context available)"
        system = _SYSTEM_TEMPLATE.format(context=context)

        answer = self._provider.generate(
            messages=[question],
            system=system,
            max_tokens=256,
            temperature=0.0,
        )

        return ConditionResult(
            condition_name=self.name,
            session_id=session.session_id,
            question=question,
            predicted_answer=answer.strip(),
            retrieved_turn_ids=turn_ids,
            num_retrieval_calls=1 if turn_ids else 0,
        )
