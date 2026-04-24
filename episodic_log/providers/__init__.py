"""LLM provider abstractions for the episodic log system."""

from __future__ import annotations

from episodic_log.providers.base import BaseProvider, normalize_messages
from episodic_log.providers.groq_provider import GroqProvider
from episodic_log.providers.huggingface_provider import HuggingFaceProvider


def get_provider(spec: str, **kwargs) -> BaseProvider:
    """Instantiate a provider from a spec string.

    Spec formats:
        ``groq:<model>``             — :class:`GroqProvider`
        ``hf:<org>/<model>``         — :class:`HuggingFaceProvider` (BF16)
        ``hf:<org>/<model>:4bit``    — :class:`HuggingFaceProvider` (4-bit NF4)
        ``hf:<org>/<model>:8bit``    — :class:`HuggingFaceProvider` (8-bit)

    Args:
        spec: Provider spec string.
        **kwargs: Extra kwargs forwarded to the provider constructor
            (e.g. ``device_map="cuda:0"``).

    Returns:
        An initialised :class:`BaseProvider` subclass.

    Raises:
        ValueError: If the prefix or quantization value is not recognised.
    """
    if ":" not in spec:
        raise ValueError(f"Provider spec must be '<backend>:<model>[:<quant>]', got {spec!r}")
    parts = spec.split(":")
    backend = parts[0].lower()

    if backend == "groq":
        model = ":".join(parts[1:])
        return GroqProvider(model=model, **kwargs)

    if backend in ("hf", "huggingface"):
        _QUANT_VALUES = ("4bit", "8bit")
        if len(parts) >= 3 and parts[-1] in _QUANT_VALUES:
            quantization: str | None = parts[-1]
            model_name = ":".join(parts[1:-1])
        else:
            quantization = None
            model_name = ":".join(parts[1:])
        return HuggingFaceProvider(
            model_name=model_name,
            quantization=quantization,
            **kwargs,
        )

    raise ValueError(f"Unknown provider backend '{backend}'. Supported: groq, hf")


__all__ = [
    "BaseProvider",
    "normalize_messages",
    "GroqProvider",
    "HuggingFaceProvider",
    "get_provider",
]
