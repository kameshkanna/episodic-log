"""Dense retrieval index over TurnSummary records using sentence embeddings.

Encodes all turn summaries with a SentenceTransformer once at construction time,
then answers queries via cosine similarity (dot product on L2-normalised vectors).

The embedding model is cached at module level so it is loaded **once per worker
process** regardless of how many sessions are evaluated — the same object is
reused for every ``EmbeddingIndex`` instantiated in that process.

Default model: ``BAAI/bge-large-en-v1.5``
  - 335M parameters, 1024-dim embeddings, top-tier MTEB English retrieval score
  - ~670 MB on GPU; negligible VRAM cost alongside a 72B LLM
  - Encodes ~5000 sentences/sec on a single H100
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from episodic_log.core.turn_summary import TurnSummary

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# Module-level cache: model_name → SentenceTransformer instance.
# Populated lazily on first use; persists for the lifetime of the process.
_MODEL_CACHE: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    """Lazy-load and cache a SentenceTransformer (once per process per model).

    Args:
        model_name: HuggingFace model ID.

    Returns:
        Loaded :class:`sentence_transformers.SentenceTransformer` instance.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install with: pip install sentence-transformers"
            ) from exc

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        logger.info("EmbeddingIndex: loading model=%s on device=%s", model_name, device)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=device)
        logger.info("EmbeddingIndex: model loaded.")

    return _MODEL_CACHE[model_name]


class EmbeddingIndex:
    """Dense cosine-similarity index over a session's turn summaries.

    Encodes all summaries in a single batched forward pass during construction.
    Query encoding is a single fast forward pass at search time (~1 ms on GPU).

    Args:
        summaries: Non-empty list of :class:`~episodic_log.core.turn_summary.TurnSummary`
            objects whose ``summary`` field is used as the retrieval corpus.
        model_name: HuggingFace sentence-transformer model ID.
        min_score: Minimum cosine similarity for a result to be returned.
            Scores below this threshold indicate a semantic miss and are
            filtered out so the caller receives an empty list on a true miss.

    Raises:
        ValueError: If *summaries* is empty.
        TypeError: If any element of *summaries* is not a TurnSummary.
        ImportError: If ``sentence-transformers`` is not installed.
    """

    def __init__(
        self,
        summaries: list[TurnSummary],
        model_name: str = DEFAULT_EMBED_MODEL,
        min_score: float = 0.3,
    ) -> None:
        if not summaries:
            raise ValueError("summaries must be non-empty to build an embedding index.")
        for item in summaries:
            if not isinstance(item, TurnSummary):
                raise TypeError(f"All elements must be TurnSummary, got {type(item)}")

        self._summaries = summaries
        self._turn_ids: list[str] = [s.turn_id for s in summaries]
        self._min_score = min_score

        model = _get_model(model_name)
        texts = [s.summary for s in summaries]

        logger.debug("EmbeddingIndex: encoding %d summaries in batch...", len(texts))
        # normalize_embeddings=True → dot product == cosine similarity.
        self._embeddings: np.ndarray = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=False,
        )  # shape: (n_summaries, hidden_dim)
        self._model = model
        logger.debug(
            "EmbeddingIndex: ready — %d turns, embedding shape=%s",
            len(texts),
            self._embeddings.shape,
        )

    def query(self, text: str, k: int = 5) -> list[str]:
        """Return top-k turn_ids most semantically similar to *text*.

        Args:
            text: Query string.
            k: Maximum results to return.

        Returns:
            Ordered list of turn_id strings (most similar first).
            Empty if no summary exceeds *min_score*.
        """
        return [tid for tid, _ in self.query_with_scores(text, k=k)]

    def query_with_scores(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k ``(turn_id, cosine_similarity)`` pairs for *text*.

        Args:
            text: Query string.
            k: Maximum number of results to return.

        Returns:
            Ordered list of ``(turn_id, score)`` tuples, most similar first.
            Results with ``score < min_score`` are excluded.

        Raises:
            TypeError: If *text* is not a string.
            ValueError: If *k* is not a positive integer.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text)}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        query_emb: np.ndarray = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # shape: (1, hidden_dim)

        # Cosine similarity via dot product on normalised vectors.
        scores: np.ndarray = (query_emb @ self._embeddings.T)[0]  # (n_summaries,)

        k_clamped = min(k, len(self._summaries))
        top_indices = np.argsort(-scores)[:k_clamped]

        return [
            (self._turn_ids[int(i)], float(scores[i]))
            for i in top_indices
            if float(scores[i]) >= self._min_score
        ]
