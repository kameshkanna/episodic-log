"""Episodic condition — BM25 retrieval over TurnSummaries → TurnEvent grounding."""

from __future__ import annotations

import json
import logging
import re

from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.core.log_reader import LogReader, TurnLoader, _format_event
from episodic_log.ingestor.longmemeval import IngestedSession
from episodic_log.providers.base import BaseProvider
from episodic_log.retrieval.summary_store import SummaryStore

logger = logging.getLogger(__name__)

_MAX_RETRIEVAL_CALLS = 3
_TOP_K = 5

_SYSTEM_PROMPT = (
    "You are a memory-augmented assistant.  You have access to a tool called\n"
    "``read_summaries`` that retrieves verbatim conversation turns from your\n"
    "episodic log.\n\n"
    'To call the tool emit a JSON object on its own line:\n'
    '  {"name": "read_summaries", "arguments": {"query": "<your query>"}}\n\n'
    f"The tool returns up to 5 verbatim turn blocks.  You may call it up to\n"
    f"{_MAX_RETRIEVAL_CALLS} times.  When you have enough context, answer the question\n"
    "on a line that starts with \"ANSWER:\".\n\n"
    "Rules:\n"
    "- Call the tool before answering when the question requires factual recall.\n"
    "- Do NOT invent information not present in the retrieved turns.\n"
    "- Be concise.\n"
)

_TOOL_CALL_RE = re.compile(
    r'"name"\s*:\s*"read_summaries".*?"query"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
_ANSWER_RE = re.compile(r"ANSWER:\s*([^\n]+)", re.IGNORECASE)


class EpisodicCondition(BaseCondition):
    """BM25-based episodic retrieval condition.

    The model issues up to :data:`_MAX_RETRIEVAL_CALLS` ``read_summaries``
    tool calls.  Each call runs a BM25 query and returns verbatim TurnEvent
    blocks from ``log.jsonl``.  The model must emit ``ANSWER: <text>`` to
    terminate.
    """

    name: str = "episodic"

    def __init__(self, provider: BaseProvider, summary_method: str = "structured") -> None:
        super().__init__(provider, summary_method)

    def run(self, session: IngestedSession, question: str) -> ConditionResult:
        """Answer *question* using agentic BM25 retrieval from the session log.

        Args:
            session: Ingested session with log_path and summaries_dir.
            question: Free-text question.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult` with the
            model's predicted answer, retrieved turn IDs, and retrieval call count.
        """
        store = SummaryStore(session.summaries_dir)
        reader = LogReader(session.log_path)

        messages: list[str] = [question]
        all_retrieved: list[str] = []
        retrieval_calls = 0
        answer = ""

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
                context = self._format_context(turn_ids, reader)
                tool_result = (
                    f"[read_summaries result]\n{context}\n[/read_summaries]"
                    if context
                    else "[read_summaries result]\n(no matching turns found)\n[/read_summaries]"
                )
                messages.append(tool_result)
            else:
                # No tool call and no ANSWER — treat last response as answer.
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

    def _format_context(self, turn_ids: list[str], reader: LogReader) -> str:
        """Format retrieved turn events into a context string for the prompt.

        Subclasses may override this to decorate turn blocks (e.g. adding
        provenance headers) without duplicating the retrieval loop.

        Args:
            turn_ids: Ordered list of turn IDs to load and format.
            reader: :class:`~episodic_log.core.log_reader.LogReader` for the session.

        Returns:
            Formatted context string ready for prompt injection.
        """
        loader = TurnLoader(reader)
        return loader.load_and_format(turn_ids)

    def _retrieve(self, store: SummaryStore, query: str) -> list[str]:
        """Run a BM25 query and return turn_ids.  Override in subclasses.

        Args:
            store: Initialised :class:`~episodic_log.retrieval.summary_store.SummaryStore`.
            query: Free-text query string.

        Returns:
            Ordered list of top-k turn_id strings.
        """
        try:
            index = store.get_index(self._summary_method)
            return index.query(query, k=_TOP_K)
        except (FileNotFoundError, ValueError):
            logger.warning(
                "No summaries found for method=%s — returning empty retrieval.",
                self._summary_method,
            )
            return []
