"""LLM provider abstractions for the episodic log system."""

from __future__ import annotations

from episodic_log.providers.base import BaseProvider, normalize_messages
from episodic_log.providers.groq_provider import GroqProvider
from episodic_log.providers.huggingface_provider import HuggingFaceProvider


def get_provider(spec: str, **kwargs) -> BaseProvider:
    """Instantiate a provider from a ``"backend:model"`` spec string.

    Supported prefixes:
        - ``groq:<model>``   — :class:`GroqProvider`
        - ``hf:<model>``     — :class:`HuggingFaceProvider`

    Args:
        spec: Provider spec string, e.g. ``"groq:llama-3.1-8b-instant"``.
        **kwargs: Extra kwargs forwarded to the provider constructor.

    Returns:
        An initialised :class:`BaseProvider` subclass.

    Raises:
        ValueError: If the prefix is not recognised.
    """
    if ":" not in spec:
        raise ValueError(f"Provider spec must be '<backend>:<model>', got {spec!r}")
    backend, model = spec.split(":", 1)
    backend = backend.lower()
    if backend == "groq":
        return GroqProvider(model=model, **kwargs)
    if backend in ("hf", "huggingface"):
        return HuggingFaceProvider(model_name=model, **kwargs)
    raise ValueError(
        f"Unknown provider backend '{backend}'. Supported: groq, hf"
    )


__all__ = [
    "BaseProvider",
    "normalize_messages",
    "GroqProvider",
    "HuggingFaceProvider",
    "get_provider",
]
