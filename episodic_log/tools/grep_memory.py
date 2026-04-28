"""Keyword grep tool for the CHD evaluation agent.

The model generates its own search keywords, calls this tool, and immediately
receives the top-3 matching turns WITH their full verbatim content pre-loaded.
No follow-up load_turn call is needed for those turns — the model can answer
directly from the returned content.  Additional matches (beyond top-3) are
listed as summary-only; the model may call load_turn on them if needed.

Ranking: matches are scored by how many unique query keywords appear in their
summary (higher = more relevant), so the most relevant turns bubble up first.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from episodic_log.core.turn_event import TurnEvent

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
            "Returns the top-3 matching turns with their FULL verbatim content "
            "already loaded — you can answer directly from them without calling "
            "load_turn. Additional matches are listed as summaries; call "
            "load_turn(turn_id) on those if you need their full content."
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
                    "description": "Maximum number of matching turns to list (default 10).",
                    "default": 10,
                },
            },
            "required": ["keywords"],
        },
    },
}

_MIN_KEYWORD_LEN = 3
_WORD_RE = re.compile(r"\b\w+\b")
_INLINE_TOP_K = 3  # always inline content for the top-3 matches
_SEP = "\n" + "─" * 60 + "\n"


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


def grep_memory(
    keywords: str,
    summaries_text: str,
    turn_map: "dict[str, TurnEvent]",
    max_results: int = 10,
) -> str:
    """Return top-3 matching turns with full content, plus remaining summary hits.

    Args:
        keywords: Space-separated search terms chosen by the model.
        summaries_text: Full pre-formatted TSV summary block (turn_id<TAB>summary).
        turn_map: Mapping from turn_id to :class:`~episodic_log.core.turn_event.TurnEvent`.
        max_results: Maximum total matches to include (default 10).

    Returns:
        Formatted string — top-3 matches with verbatim content, remaining as
        summary-only lines.  Returns a no-match message with suggestions if
        no turns matched.

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

    all_lines = summaries_text.splitlines()
    data_lines = [ln for ln in all_lines if "\t" in ln and not ln.startswith("turn_id\t")]

    # Score each line by number of unique keywords matched in the summary column.
    scored: list[tuple[int, str]] = []
    for ln in data_lines:
        summary_col = ln.split("\t", 1)[1].lower()
        score = sum(1 for kw in kws if kw in summary_col)
        if score > 0:
            scored.append((score, ln))

    if not scored:
        return (
            f"No turns matched keywords {kws}. "
            "Try synonyms, a shorter keyword, or a different aspect of the question."
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    results = scored[:max_results]
    total_matched = len(scored)

    logger.debug(
        "grep_memory: keywords=%r matched %d/%d turns (showing %d).",
        keywords, total_matched, len(data_lines), len(results),
    )

    header = (
        f"grep_memory: {total_matched} matching turn(s) for '{keywords}'"
        + (f" (showing top {max_results})" if total_matched > max_results else "")
        + "\n"
    )

    inline_parts: list[str] = []
    summary_parts: list[str] = []

    for i, (score, ln) in enumerate(results):
        turn_id = ln.split("\t", 1)[0]
        summary_text = ln.split("\t", 1)[1] if "\t" in ln else ln

        if i < _INLINE_TOP_K:
            event = turn_map.get(turn_id)
            if event is not None:
                role_label = event.role.value.upper()
                inline_parts.append(
                    f"[{role_label} | turn {turn_id}]  score={score}\n"
                    f"Summary: {summary_text}\n"
                    f"Full content:\n{event.content}"
                )
            else:
                inline_parts.append(
                    f"[turn {turn_id}]  score={score}\n"
                    f"Summary: {summary_text}\n"
                    "(full content unavailable)"
                )
        else:
            summary_parts.append(f"  {turn_id}\t{summary_text}  [score={score}]")

    sections: list[str] = []

    if inline_parts:
        sections.append(
            f"── TOP {min(_INLINE_TOP_K, len(inline_parts))} TURNS (content pre-loaded — answer directly from these) ──\n"
            + _SEP.join(inline_parts)
        )

    if summary_parts:
        sections.append(
            "── ADDITIONAL MATCHES (call load_turn to read full content) ──\n"
            + "\n".join(summary_parts)
        )

    return header + "\n\n".join(sections)
