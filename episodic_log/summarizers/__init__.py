"""Summarizer implementations and factory for the episodic log system.

All summarizers produce :class:`~episodic_log.core.turn_summary.TurnSummary` records
used exclusively as a BM25 search index.  The agent never reads summaries directly.

Exported names
--------------
* :class:`AbstractSummarizer` — abstract base class.
* :class:`StructuredSummarizer` — deterministic, no model required.
* :class:`HaikuSummarizer` — uses a small LLM via :class:`~episodic_log.providers.base.BaseProvider`.
* :class:`SelfSummarizer` — uses the agent model (highest contamination risk).
* :func:`build_summarizer` — factory that instantiates the correct summarizer by name.
"""

from __future__ import annotations

from episodic_log.summarizers.base import AbstractSummarizer
from episodic_log.summarizers.haiku import HaikuSummarizer
from episodic_log.summarizers.self_summarizer import SelfSummarizer
from episodic_log.summarizers.structured import StructuredSummarizer

__all__ = [
    "AbstractSummarizer",
    "StructuredSummarizer",
    "HaikuSummarizer",
    "SelfSummarizer",
    "build_summarizer",
]

# Registry mapping method names to (class, requires_provider) tuples.
_SUMMARIZER_REGISTRY: dict[str, tuple[type[AbstractSummarizer], bool]] = {
    "structured": (StructuredSummarizer, False),
    "haiku": (HaikuSummarizer, True),
    "self": (SelfSummarizer, True),
}


def build_summarizer(
    method: str,
    provider: "episodic_log.providers.base.BaseProvider | None" = None,
) -> AbstractSummarizer:
    """Factory that instantiates the correct summarizer by method name.

    Args:
        method: One of ``"structured"``, ``"haiku"``, or ``"self"``.
        provider: A :class:`~episodic_log.providers.base.BaseProvider` instance.
            Required for ``"haiku"`` and ``"self"`` methods; must be ``None``
            (or omitted) for ``"structured"``.

    Returns:
        A fully initialised :class:`AbstractSummarizer` subclass instance.

    Raises:
        ValueError: If *method* is not one of the registered method names.
        ValueError: If *method* requires a provider but none was supplied.
        TypeError: If a provider is supplied for a method that does not need one.

    Examples:
        >>> s = build_summarizer("structured")
        >>> s.method
        'structured'

        >>> from episodic_log.providers.base import BaseProvider
        >>> s = build_summarizer("haiku", provider=my_provider)
        >>> s.method
        'haiku'
    """
    method = method.strip().lower()

    if method not in _SUMMARIZER_REGISTRY:
        valid = sorted(_SUMMARIZER_REGISTRY.keys())
        raise ValueError(
            f"Unknown summarizer method '{method}'. Valid options: {valid}"
        )

    cls, requires_provider = _SUMMARIZER_REGISTRY[method]

    if requires_provider and provider is None:
        raise ValueError(
            f"Summarizer method '{method}' requires a BaseProvider instance, "
            f"but provider=None was given."
        )

    if not requires_provider and provider is not None:
        raise TypeError(
            f"Summarizer method '{method}' does not accept a provider, "
            f"but provider={provider!r} was given."
        )

    if requires_provider:
        return cls(provider=provider)  # type: ignore[call-arg]

    return cls()
