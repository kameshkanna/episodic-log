"""BM25 retrieval index over TurnSummary records.

Uses ``rank_bm25.BM25Okapi`` for keyword-based retrieval.  A naïve TF fallback
is provided when ``rank_bm25`` is not installed so unit tests can run without
the optional dependency.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from episodic_log.core.turn_summary import TurnSummary

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_TOKENISE_RE = re.compile(r"\w+")


def _tokenise(text: str) -> list[str]:
    return _TOKENISE_RE.findall(text.lower())


class BM25Index:
    """BM25-based retrieval index built from a list of :class:`~episodic_log.core.turn_summary.TurnSummary` records.

    Args:
        summaries: Ordered list of :class:`~episodic_log.core.turn_summary.TurnSummary`
            objects whose ``summary`` field is used as the retrieval corpus.

    Raises:
        TypeError: If *summaries* is not a list or any element is not a TurnSummary.
        ValueError: If *summaries* is empty.
    """

    def __init__(self, summaries: list[TurnSummary]) -> None:
        if not isinstance(summaries, list):
            raise TypeError(f"summaries must be a list, got {type(summaries)}")
        if not summaries:
            raise ValueError("summaries must be non-empty to build a BM25 index.")
        for item in summaries:
            if not isinstance(item, TurnSummary):
                raise TypeError(f"All elements must be TurnSummary, got {type(item)}")

        self._summaries = summaries
        self._turn_ids: list[str] = [s.turn_id for s in summaries]
        corpus: list[list[str]] = [_tokenise(s.summary) for s in summaries]

        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import]
            self._bm25 = BM25Okapi(corpus)
            self._use_bm25 = True
            logger.debug("BM25Index: built rank_bm25.BM25Okapi index over %d summaries.", len(summaries))
        except ImportError:
            logger.warning(
                "rank_bm25 not installed — falling back to naïve TF scoring. "
                "Install with: pip install rank-bm25"
            )
            self._corpus = corpus
            self._use_bm25 = False

    def query(self, text: str, k: int = 5) -> list[str]:
        """Retrieve the top-k most relevant turn_ids for a query string.

        Args:
            text: Free-text query.
            k: Maximum number of turn_ids to return.

        Returns:
            Ordered list (most relevant first) of zero-padded turn_id strings.
            Zero-score results (complete keyword misses) are excluded.

        Raises:
            TypeError: If *text* is not a string.
            ValueError: If *k* is not a positive integer.
        """
        return [turn_id for turn_id, _ in self.query_with_scores(text, k=k)]

    def query_with_scores(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """Retrieve the top-k most relevant (turn_id, score) pairs.

        Zero-score results are filtered out so the caller can distinguish a
        genuine miss from a low-scoring match.

        Args:
            text: Free-text query.
            k: Maximum number of results to return.

        Returns:
            Ordered list of ``(turn_id, score)`` tuples, most relevant first.
            Empty if no document matched any query token.

        Raises:
            TypeError: If *text* is not a string.
            ValueError: If *k* is not a positive integer.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text)}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        tokens = _tokenise(text)
        k_clamped = min(k, len(self._summaries))

        if self._use_bm25:
            scores = self._bm25.get_scores(tokens)
        else:
            scores = self._naive_tf_scores(tokens)

        import heapq
        top_indices = heapq.nlargest(k_clamped, range(len(scores)), key=lambda i: scores[i])
        return [
            (self._turn_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _naive_tf_scores(self, query_tokens: list[str]) -> list[float]:
        """Compute a simple term-frequency score for each document.

        Args:
            query_tokens: Tokenised query terms.

        Returns:
            Float score per document in corpus order.
        """
        query_set = set(query_tokens)
        scores: list[float] = []
        for doc_tokens in self._corpus:
            score = sum(1.0 for t in doc_tokens if t in query_set)
            scores.append(score)
        return scores
