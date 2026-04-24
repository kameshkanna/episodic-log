"""BM25-based memory search tool for the CHD evaluation agent.

Reads a ``<method>.jsonl`` summary file and returns the top-k most relevant
turns for a free-text query using :class:`~episodic_log.retrieval.bm25_index.BM25Index`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from episodic_log.core.turn_summary import TurnSummary
from episodic_log.retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI-format tool schema
# ---------------------------------------------------------------------------

GREP_MEMORY_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "grep_memory",
        "description": (
            "Search your memory index for turns relevant to a query. "
            "Returns up to k matching turns with their IDs and summaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrase to search for",
                },
                "k": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


def grep_memory(
    query: str,
    summaries_dir: Path,
    method: str,
    k: int = 5,
) -> list[dict[str, str]]:
    """Search the BM25 summary index and return the top-k matching turns.

    Reads ``<summaries_dir>/<method>.jsonl``, builds a
    :class:`~episodic_log.retrieval.bm25_index.BM25Index` over all summaries,
    and returns the top-k results ordered by relevance.

    Args:
        query: Free-text search string.
        summaries_dir: Path to the directory containing ``<method>.jsonl`` files.
        method: Summarizer method name (e.g. ``"lexical"``, ``"scout"``, ``"echo"``).
        k: Maximum number of results to return.  Defaults to 5.

    Returns:
        List of dicts, each with keys ``"turn_id"`` and ``"summary"``,
        ordered from most to least relevant.  Returns an empty list if the
        summary file is missing or contains no parseable entries.

    Raises:
        TypeError: If *query* is not a string or *summaries_dir* is not a Path.
        ValueError: If *k* is not a positive integer.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a str, got {type(query)}")
    if not isinstance(summaries_dir, Path):
        raise TypeError(f"summaries_dir must be a Path, got {type(summaries_dir)}")
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    summary_path = summaries_dir / f"{method}.jsonl"
    if not summary_path.exists():
        logger.warning(
            "grep_memory: summary file not found at %s — returning empty results.",
            summary_path,
        )
        return []

    summaries: list[TurnSummary] = []
    with summary_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                summaries.append(TurnSummary.from_json(stripped))
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "grep_memory: skipping malformed line %d in %s: %s",
                    lineno,
                    summary_path,
                    exc,
                )

    if not summaries:
        logger.warning(
            "grep_memory: summary file %s is empty or has no parseable entries.",
            summary_path,
        )
        return []

    index = BM25Index(summaries)
    top_ids: list[str] = index.query(query, k=k)

    # Build a lookup map for O(1) access when assembling results.
    summary_by_id: dict[str, str] = {s.turn_id: s.summary for s in summaries}

    results: list[dict[str, str]] = []
    for turn_id in top_ids:
        if turn_id in summary_by_id:
            results.append({"turn_id": turn_id, "summary": summary_by_id[turn_id]})

    logger.debug(
        "grep_memory: query=%r method=%s returned %d results.", query, method, len(results)
    )
    return results
