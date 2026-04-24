"""Factory for session-bound tool callables used by the CHD evaluation agent.

:func:`make_session_tools` binds :func:`~episodic_log.tools.grep_memory.grep_memory`
and :func:`~episodic_log.tools.load_turn.load_turn` to a specific session's paths,
returning ready-to-call functions keyed by their tool names.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Callable

from episodic_log.tools.grep_memory import GREP_MEMORY_SCHEMA, grep_memory
from episodic_log.tools.load_turn import LOAD_TURN_SCHEMA, load_turn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aggregated tool schemas list (consumed by the agent loop)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [GREP_MEMORY_SCHEMA, LOAD_TURN_SCHEMA]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_session_tools(
    summaries_dir: Path,
    log_path: Path,
    method: str,
) -> dict[str, Callable]:
    """Return tool callables bound to the specified session's data paths.

    Each callable in the returned dict accepts only the arguments that the
    model controls (i.e. the fields defined in the corresponding tool schema).
    Session-specific arguments (``summaries_dir``, ``log_path``, ``method``)
    are captured via :func:`functools.partial`.

    Args:
        summaries_dir: Path to the directory containing ``<method>.jsonl`` files.
        log_path: Absolute path to the session's ``log.jsonl`` file.
        method: Summarizer method name used to select the correct summary file.

    Returns:
        Dict mapping tool name strings to bound callables:

        - ``"grep_memory"``: ``(query: str, k: int = 5) -> list[dict[str, str]]``
        - ``"load_turn"``: ``(turn_id: str) -> str``

    Raises:
        TypeError: If *summaries_dir* or *log_path* are not :class:`pathlib.Path` objects.
        ValueError: If *method* is an empty string.
    """
    if not isinstance(summaries_dir, Path):
        raise TypeError(f"summaries_dir must be a Path, got {type(summaries_dir)}")
    if not isinstance(log_path, Path):
        raise TypeError(f"log_path must be a Path, got {type(log_path)}")
    if not method or not isinstance(method, str):
        raise ValueError(f"method must be a non-empty string, got {method!r}")

    logger.debug(
        "make_session_tools: summaries_dir=%s log_path=%s method=%s",
        summaries_dir,
        log_path,
        method,
    )

    return {
        "grep_memory": partial(grep_memory, summaries_dir=summaries_dir, method=method),
        "load_turn": partial(load_turn, log_path=log_path),
    }
