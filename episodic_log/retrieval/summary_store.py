"""SummaryStore — loads TurnSummary JSONL files and provides BM25 retrieval.

Each summarizer method writes its output to a separate JSONL file:
    <summaries_dir>/<method>.jsonl

SummaryStore loads all files in *summaries_dir* on demand and caches a
per-method :class:`~episodic_log.retrieval.bm25_index.BM25Index`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from episodic_log.core.turn_summary import TurnSummary
from episodic_log.retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class SummaryStore:
    """Lazy-loading store of :class:`~episodic_log.core.turn_summary.TurnSummary` records.

    The store reads ``<summaries_dir>/<method>.jsonl`` files on first access and
    caches a :class:`~episodic_log.retrieval.bm25_index.BM25Index` per method.

    Args:
        summaries_dir: Directory containing ``<method>.jsonl`` summary files.

    Raises:
        TypeError: If *summaries_dir* is not a :class:`pathlib.Path`.
    """

    def __init__(self, summaries_dir: Path) -> None:
        if not isinstance(summaries_dir, Path):
            raise TypeError(f"summaries_dir must be a Path, got {type(summaries_dir)}")
        self._summaries_dir = summaries_dir
        self._summaries_cache: dict[str, list[TurnSummary]] = {}
        self._index_cache: dict[str, BM25Index] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, summary: TurnSummary) -> None:
        """Append a single :class:`~episodic_log.core.turn_summary.TurnSummary` to the appropriate JSONL file.

        The file is ``<summaries_dir>/<summary.method>.jsonl``.  Caches are
        invalidated for the affected method so the next read rebuilds them.

        Args:
            summary: The :class:`~episodic_log.core.turn_summary.TurnSummary` to persist.

        Raises:
            TypeError: If *summary* is not a TurnSummary.
        """
        if not isinstance(summary, TurnSummary):
            raise TypeError(f"summary must be a TurnSummary, got {type(summary)}")
        self._summaries_dir.mkdir(parents=True, exist_ok=True)
        path = self._summaries_dir / f"{summary.method}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(summary.to_json() + "\n")
        # Invalidate caches for this method.
        self._summaries_cache.pop(summary.method, None)
        self._index_cache.pop(summary.method, None)

    def load(self, method: str) -> list[TurnSummary]:
        """Load all :class:`~episodic_log.core.turn_summary.TurnSummary` records for *method*.

        Results are cached in memory; call :meth:`invalidate` to force a reload.

        Args:
            method: Summarizer method name (e.g. ``"structured"``, ``"haiku"``, ``"self"``).

        Returns:
            Ordered list of :class:`~episodic_log.core.turn_summary.TurnSummary` records.

        Raises:
            FileNotFoundError: If no JSONL file exists for *method*.
        """
        if method in self._summaries_cache:
            return self._summaries_cache[method]

        path = self._summaries_dir / f"{method}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"No summary file for method '{method}' at {path}. "
                "Run scripts/summarize.py first."
            )

        summaries: list[TurnSummary] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    summaries.append(TurnSummary.from_json(stripped))
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Failed to parse TurnSummary at line {lineno} of {path}: {exc}"
                    ) from exc

        self._summaries_cache[method] = summaries
        logger.debug("SummaryStore: loaded %d summaries for method=%s", len(summaries), method)
        return summaries

    def get_index(self, method: str) -> BM25Index:
        """Return a cached :class:`~episodic_log.retrieval.bm25_index.BM25Index` for *method*.

        Builds the index on first access.

        Args:
            method: Summarizer method name.

        Returns:
            A :class:`~episodic_log.retrieval.bm25_index.BM25Index` built from the loaded summaries.

        Raises:
            FileNotFoundError: If no JSONL file exists for *method*.
            ValueError: If the summary file is empty.
        """
        if method not in self._index_cache:
            summaries = self.load(method)
            self._index_cache[method] = BM25Index(summaries)
        return self._index_cache[method]

    def available_methods(self) -> list[str]:
        """Return the list of summarizer methods for which a JSONL file exists.

        Returns:
            Sorted list of method name strings.
        """
        if not self._summaries_dir.exists():
            return []
        return sorted(p.stem for p in self._summaries_dir.glob("*.jsonl"))

    def invalidate(self, method: str | None = None) -> None:
        """Invalidate in-memory caches so the next access reloads from disk.

        Args:
            method: If given, invalidate only the cache for that method.
                If ``None``, invalidate all caches.
        """
        if method is None:
            self._summaries_cache.clear()
            self._index_cache.clear()
        else:
            self._summaries_cache.pop(method, None)
            self._index_cache.pop(method, None)
