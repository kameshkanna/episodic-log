"""External condition — retrieved turns labelled as coming from a *different* agent's session."""

from __future__ import annotations

import logging

from episodic_log.conditions.base import ConditionResult
from episodic_log.conditions.episodic import EpisodicCondition
from episodic_log.core.log_reader import LogReader, TurnLoader, _format_event
from episodic_log.retrieval.summary_store import SummaryStore

logger = logging.getLogger(__name__)

_EXTERNAL_HEADER_PREFIX = "[EXTERNAL SESSION]"


class ExternalCondition(EpisodicCondition):
    """Attribution confusion condition.

    Retrieved turns are presented with a forged provenance header so the model
    sees them as originating from ``"a different agent's session"`` rather than
    its own log.  This probes whether agents over-trust labelled context and
    confabulate cross-session memories.
    """

    name: str = "external"

    def _retrieve(self, store: SummaryStore, query: str) -> list[str]:
        """Delegate to base BM25 retrieval — decoration happens in the loader override."""
        return super()._retrieve(store, query)

    def run(self, session, question: str) -> ConditionResult:
        """Answer *question* using turns presented as external session context.

        The turn formatter is overridden to prepend :data:`_EXTERNAL_HEADER_PREFIX`
        to every retrieved block.

        Args:
            session: Ingested session.
            question: Free-text question.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult`.
        """
        from episodic_log.core.log_reader import LogReader, TurnLoader
        from episodic_log.retrieval.summary_store import SummaryStore
        import re

        store = SummaryStore(session.summaries_dir)
        reader = LogReader(session.log_path)

        from episodic_log.conditions.episodic import (
            _MAX_RETRIEVAL_CALLS,
            _TOP_K,
            _SYSTEM_PROMPT,
            _TOOL_CALL_RE,
            _ANSWER_RE,
        )

        messages: list[str] = [question]
        all_retrieved: list[str] = []
        retrieval_calls = 0
        answer = ""
        response = ""

        for _ in range(_MAX_RETRIEVAL_CALLS + 1):
            response = self._provider.generate(
                messages=messages,
                system=_SYSTEM_PROMPT,
                max_tokens=512,
                temperature=0.0,
            )
            messages.append(response)

            answer_match = _ANSWER_RE.search(response)
            if answer_match:
                answer = answer_match.group(1).strip()
                break

            tool_match = _TOOL_CALL_RE.search(response)
            if tool_match and retrieval_calls < _MAX_RETRIEVAL_CALLS:
                query = tool_match.group(1)
                turn_ids = self._retrieve(store, query)
                all_retrieved.extend(turn_ids)
                retrieval_calls += 1
                # Load events and decorate with external header.
                events = reader.load_by_ids(turn_ids)
                blocks = []
                for event in events:
                    block = _format_event(event)
                    decorated = f"{_EXTERNAL_HEADER_PREFIX} {block}"
                    blocks.append(decorated)
                context = "\n\n".join(blocks)
                tool_result = (
                    f"[read_summaries result]\n{context}\n[/read_summaries]"
                    if context
                    else "[read_summaries result]\n(no matching turns found)\n[/read_summaries]"
                )
                messages.append(tool_result)
            else:
                answer = response.strip()
                break

        if not answer:
            answer = response.strip()

        return ConditionResult(
            condition_name=self.name,
            session_id=session.session_id,
            question=question,
            predicted_answer=answer,
            retrieved_turn_ids=list(dict.fromkeys(all_retrieved)),
            num_retrieval_calls=retrieval_calls,
        )
