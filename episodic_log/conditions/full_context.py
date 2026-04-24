"""Full-context condition — inject all turns verbatim into system prompt (truncated at 32k chars)."""

from __future__ import annotations

import logging

from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.core.log_reader import LogReader, TurnLoader
from episodic_log.ingestor.longmemeval import IngestedSession
from episodic_log.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_CHAR_LIMIT = 32_000

_SYSTEM_TEMPLATE = """\
You are a helpful assistant with access to your complete conversation history.

<conversation_history>
{history}
</conversation_history>

Answer the question using only the conversation history above.  Be concise.
"""


class FullContextCondition(BaseCondition):
    """Full-context upper bound condition.

    The entire session log (up to :data:`_CHAR_LIMIT` characters) is injected
    verbatim into the system prompt.  No retrieval — the model has access to
    everything at once.  This establishes the performance ceiling for a model
    with perfect recall of all turns.
    """

    name: str = "full_context"

    def __init__(self, provider: BaseProvider, summary_method: str = "structured") -> None:
        super().__init__(provider, summary_method)

    def run(self, session: IngestedSession, question: str) -> ConditionResult:
        """Inject full log and answer *question* in one shot.

        Args:
            session: Ingested session with log_path.
            question: Free-text question.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult`.
        """
        reader = LogReader(session.log_path)
        loader = TurnLoader(reader)

        history = loader.format_all(char_limit=_CHAR_LIMIT)
        truncated = history.endswith("[TRUNCATED]")

        system = _SYSTEM_TEMPLATE.format(history=history)
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
            metadata={"truncated": truncated, "history_chars": len(history)},
        )
