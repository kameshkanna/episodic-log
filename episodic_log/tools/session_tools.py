"""Session-bound tool callables for the CHD evaluation agent.

Two retrieval modes — controlled by *mode* in :func:`make_session_tools`:

``"load_only"`` (default, used by RecallCondition)
    All summaries are injected into the agent's first message.
    The agent has only ``load_turn`` — it reads the summary index and calls
    load_turn directly for turns it deems relevant.

``"grep_and_load"`` (used by GrepRecallCondition)
    Summaries are NOT injected upfront.  The agent has ``grep_memory`` +
    ``load_turn``.  It must formulate keywords to search, receive matching
    summary lines, then call load_turn for the relevant ones.

Both modes build the turn map once (O(1) lookup at call time) and format the
summary block once.  No retrieval library is involved.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Callable

from episodic_log.core.turn_event import TurnEvent
from episodic_log.core.turn_summary import TurnSummary
from episodic_log.tools.grep_memory import GREP_MEMORY_SCHEMA, grep_memory
from episodic_log.tools.load_turn import LOAD_TURN_SCHEMA, load_turn

logger = logging.getLogger(__name__)

# Default schema set — only load_turn (recall condition).
TOOL_SCHEMAS: list[dict] = [LOAD_TURN_SCHEMA]

# Extended schema set — grep_memory + load_turn (grep_recall condition).
GREP_TOOL_SCHEMAS: list[dict] = [GREP_MEMORY_SCHEMA, LOAD_TURN_SCHEMA]


# Per-summary char cap so all 500 turns always fit in the index.
# 500 turns × (5 id + 1 tab + 150 summary + 1 newline) ≈ 78 k chars — well under
# the 100 k global cap.  Long echo summaries are clipped with "…" rather than
# dropping the whole turn, so the model still sees every turn ID.
_MAX_SUMMARY_LINE_CHARS: int = 150


def format_summaries_as_context(summaries_dir: Path, method: str) -> str:
    """Load all turn summaries and return them as a TSV memory index.

    Format: one row per turn, tab-separated::

        turn_id\\tsummary
        0000\\tUser greeted the assistant
        0001\\tUser asked about scheduling a dentist appointment for Tuesday
        ...

    Multi-line summaries are collapsed to a single line.  Each summary is then
    capped at ``_MAX_SUMMARY_LINE_CHARS`` characters (appending ``…`` if clipped)
    so the full index of 500 turns always fits in the model's context without the
    global truncation in ``AgentLoop`` ever needing to drop entire turns.

    Args:
        summaries_dir: Directory containing ``<method>.jsonl`` summary files.
        method: Summarizer method key (e.g. ``"lexical"``, ``"scout"``).

    Returns:
        TSV string with a header row followed by one row per turn.
        Returns an empty string if the summary file is missing.
    """
    summary_path = summaries_dir / f"{method}.jsonl"
    if not summary_path.exists():
        logger.warning("format_summaries_as_context: summary file not found at %s", summary_path)
        return ""

    rows: list[str] = []
    with summary_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                s = TurnSummary.from_json(stripped)
                # Collapse internal newlines → single line.
                one_line = " ".join(s.summary.split())
                # Cap per-line length so no single verbose summary blows the budget.
                if len(one_line) > _MAX_SUMMARY_LINE_CHARS:
                    one_line = one_line[:_MAX_SUMMARY_LINE_CHARS - 1] + "…"
                rows.append(f"{s.turn_id}\t{one_line}")
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "format_summaries_as_context: skipping malformed line %d in %s: %s",
                    lineno, summary_path, exc,
                )

    logger.debug(
        "format_summaries_as_context: %d summaries loaded from %s/%s",
        len(rows), summaries_dir.name, method,
    )
    if not rows:
        return ""

    header = "turn_id\tsummary"
    return header + "\n" + "\n".join(rows)


def make_session_tools(
    summaries_dir: Path,
    log_path: Path,
    method: str,
    mode: str = "load_only",
) -> tuple[dict[str, Callable], list[dict]]:
    """Build session-bound tool callables and the matching tool schema list.

    Args:
        summaries_dir: Directory containing ``<method>.jsonl`` summary files.
        log_path: Absolute path to this session's ``log.jsonl``.
        method: Summarizer method key (e.g. ``"lexical"``).
        mode: ``"load_only"`` — returns only ``load_turn``.
              ``"grep_and_load"`` — returns ``grep_memory`` + ``load_turn``.

    Returns:
        Tuple of ``(tools_dict, tool_schemas)`` where *tools_dict* maps tool
        name strings to bound callables and *tool_schemas* is the
        corresponding list of OpenAI-format schema dicts.

    Raises:
        TypeError: If *summaries_dir* or *log_path* are not :class:`Path`.
        ValueError: If *method* is empty or *mode* is unrecognised.
    """
    if not isinstance(summaries_dir, Path):
        raise TypeError(f"summaries_dir must be a Path, got {type(summaries_dir)}")
    if not isinstance(log_path, Path):
        raise TypeError(f"log_path must be a Path, got {type(log_path)}")
    if not method or not isinstance(method, str):
        raise ValueError(f"method must be a non-empty string, got {method!r}")
    if mode not in ("load_only", "grep_and_load"):
        raise ValueError(f"mode must be 'load_only' or 'grep_and_load', got {mode!r}")

    # ── Build turn map (once per session) ───────────────────────────────────
    turn_map: dict[str, TurnEvent] = {}
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = TurnEvent.from_json(stripped)
                    turn_map[event.turn_id] = event
                except (KeyError, ValueError) as exc:
                    logger.warning(
                        "make_session_tools: skipping malformed log line %d in %s: %s",
                        lineno, log_path, exc,
                    )
        logger.debug(
            "make_session_tools: indexed %d turns from %s (mode=%s)",
            len(turn_map), log_path.name, mode,
        )
    else:
        logger.warning("make_session_tools: log file not found at %s", log_path)

    load_turn_fn = partial(load_turn, turn_map=turn_map)

    if mode == "load_only":
        return {"load_turn": load_turn_fn}, TOOL_SCHEMAS

    # grep_and_load: pre-load the formatted summary block once, bind to grep_memory.
    summaries_text = format_summaries_as_context(summaries_dir, method)
    grep_memory_fn = partial(grep_memory, summaries_text=summaries_text)

    return (
        {"grep_memory": grep_memory_fn, "load_turn": load_turn_fn},
        GREP_TOOL_SCHEMAS,
    )
