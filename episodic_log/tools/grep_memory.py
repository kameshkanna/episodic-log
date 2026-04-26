"""Keyword grep tool for the CHD evaluation agent.

The model generates its own search keywords based on the question, then calls
this tool to find matching summary lines.  Matching is simple case-insensitive
substring search — no scoring, no ranking.  Every summary line that contains
any of the query words is returned.

This is intentionally different from BM25:
  - BM25 scores every document using TF-IDF weighting and the raw question.
  - This tool returns all lines matching model-chosen keywords (binary match).
  - The model decides WHAT to search for; the system just does the grep.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI-format tool schema
# ---------------------------------------------------------------------------

GREP_MEMORY_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "grep_memory",
        "description": (
            "Search the memory index for turns matching your keywords. "
            "Returns all summary lines that contain any of the keywords. "
            "Choose specific keywords that would appear in a summary of the relevant turn. "
            "Call load_turn with a turn_id to read the full content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": (
                        "Space-separated keywords to search for "
                        "(e.g. 'dentist appointment Tuesday'). "
                        "Use specific nouns, names, dates, or action words."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return (default 20).",
                    "default": 20,
                },
            },
            "required": ["keywords"],
        },
    },
}

_MIN_KEYWORD_LEN = 3  # ignore very short words like "a", "is", "to"
_WORD_RE = re.compile(r"\b\w+\b")


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


def grep_memory(
    keywords: str,
    summaries_text: str,
    max_results: int = 20,
) -> str:
    """Return all summary lines containing any of the given keywords.

    Args:
        keywords: Space-separated search terms chosen by the model.
        summaries_text: The full pre-formatted summary block, one line per turn
            in the format ``[turn_id] summary text``.
        max_results: Maximum lines to return.

    Returns:
        A formatted string listing matching summary lines, or a "no match"
        message with a suggestion to try different keywords.

    Raises:
        TypeError: If *keywords* is not a string.
        ValueError: If *max_results* is not a positive integer.
    """
    if not isinstance(keywords, str):
        raise TypeError(f"keywords must be a str, got {type(keywords)}")
    if not isinstance(max_results, int) or max_results <= 0:
        raise ValueError(f"max_results must be a positive integer, got {max_results!r}")

    if not summaries_text:
        return "No memory index available for this session."

    # Extract meaningful keywords (skip very short stop-words).
    kws = [
        w.lower()
        for w in _WORD_RE.findall(keywords)
        if len(w) >= _MIN_KEYWORD_LEN
    ]
    if not kws:
        return (
            f"No usable keywords in {keywords!r} (all words are too short). "
            "Use specific nouns or action words."
        )

    lines = summaries_text.splitlines()
    matches = [
        line for line in lines
        if any(kw in line.lower() for kw in kws)
    ]

    if not matches:
        return (
            f"No turns matched keywords {kws}. "
            "Try synonyms, a shorter keyword, or a different aspect of the question."
        )

    results = matches[:max_results]
    truncation_note = (
        f"\n(showing {max_results} of {len(matches)} matches — "
        "refine keywords to narrow results)"
        if len(matches) > max_results else ""
    )

    logger.debug(
        "grep_memory: keywords=%r matched %d/%d lines.", keywords, len(matches), len(lines)
    )
    return (
        f"grep_memory: {len(results)} matching turn(s) for '{keywords}':\n\n"
        + "\n".join(results)
        + truncation_note
        + "\n\nCall load_turn(turn_id) to read the full content of any turn."
    )
