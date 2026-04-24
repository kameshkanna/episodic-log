"""Summarizer implementations and factory for the episodic log system.

All summarizers produce :class:`~episodic_log.core.turn_summary.TurnSummary` records
used exclusively as a BM25 search index.  The agent never reads summaries directly.

Exported names
--------------
* :class:`AbstractSummarizer` — abstract base class.
* :class:`LexicalSummarizer` — deterministic, no model required.
* :class:`ScoutSummarizer` — uses a small LLM via :class:`~episodic_log.providers.base.BaseProvider`.
* :class:`EchoSummarizer` — uses the agent model (highest contamination risk).
* :func:`build_summarizer` — factory that instantiates the correct summarizer by name.
"""

from __future__ import annotations

from episodic_log.summarizers.base import AbstractSummarizer
from episodic_log.summarizers.lexical import LexicalSummarizer
from episodic_log.summarizers.scout import ScoutSummarizer
from episodic_log.summarizers.echo import EchoSummarizer
from episodic_log.providers.base import BaseProvider

__all__ = [
    "AbstractSummarizer",
    "LexicalSummarizer",
    "ScoutSummarizer",
    "EchoSummarizer",
    "build_summarizer",
]

_VALID_METHODS = ("lexical", "scout", "echo")


def build_summarizer(method: str, provider: BaseProvider | None = None):
    if method == "lexical":
        return LexicalSummarizer()
    if method == "scout":
        if provider is None:
            raise ValueError("provider required for scout method")
        return ScoutSummarizer(provider)
    if method == "echo":
        if provider is None:
            raise ValueError("provider required for echo method")
        return EchoSummarizer(provider)
    raise ValueError(f"Unknown summarizer method: {method!r}. Valid: {_VALID_METHODS}")
