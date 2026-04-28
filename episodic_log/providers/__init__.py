"""LLM provider abstractions for the episodic log system."""

from __future__ import annotations

from episodic_log.providers.base import BaseProvider, normalize_messages
from episodic_log.providers.groq_provider import GroqProvider
from episodic_log.providers.huggingface_provider import HuggingFaceProvider


def get_provider(spec: str, **kwargs) -> BaseProvider:
    """Instantiate a provider from a spec string.

    Spec formats:
        ``groq:<model>``               — :class:`GroqProvider`
        ``hf:<org>/<model>``           — :class:`HuggingFaceProvider` (BF16)
        ``hf:<org>/<model>:4bit``      — :class:`HuggingFaceProvider` (4-bit NF4)
        ``hf:<org>/<model>:8bit``      — :class:`HuggingFaceProvider` (8-bit)
        ``vllm:<org>/<model>``         — :class:`VLLMProvider` (tp=1)
        ``vllm:<org>/<model>:tp4``     — :class:`VLLMProvider` (tensor_parallel_size=4)
        ``vllm:<org>/<model>:tp8``     — :class:`VLLMProvider` (tensor_parallel_size=8)

    Args:
        spec: Provider spec string.
        **kwargs: Extra kwargs forwarded to HF/Groq provider constructors.
            Ignored for vLLM (it manages GPU placement internally).

    Returns:
        An initialised :class:`BaseProvider` subclass.

    Raises:
        ValueError: If the prefix or quantization value is not recognised.
    """
    if ":" not in spec:
        raise ValueError(f"Provider spec must be '<backend>:<model>[:<opt>]', got {spec!r}")
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

    if backend == "vllm":
        from episodic_log.providers.vllm_provider import VLLMProvider

        _TP_VALUES = {"tp1": 1, "tp2": 2, "tp4": 4, "tp8": 8}
        raw = ":".join(parts[1:])
        tp = 1
        for suffix, n in _TP_VALUES.items():
            if raw.endswith(f":{suffix}"):
                tp = n
                raw = raw[: -(len(suffix) + 1)]
                break
        return VLLMProvider(model_name=raw, tensor_parallel_size=tp, **kwargs)

    raise ValueError(
        f"Unknown provider backend '{backend}'. Supported: groq, hf, vllm"
    )


__all__ = [
    "BaseProvider",
    "normalize_messages",
    "GroqProvider",
    "HuggingFaceProvider",
    "get_provider",
]
