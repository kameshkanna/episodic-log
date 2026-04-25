"""vLLM offline batch provider for maximum summarization throughput.

Uses vLLM's PagedAttention + continuous batching to process hundreds of
thousands of prompts in a single pass, achieving 5-10x the throughput of
naive HuggingFace ``model.generate()``.

Provider spec: ``vllm:<model_id>`` or ``vllm:<model_id>:tp4``
"""

from __future__ import annotations

import logging
from typing import Any

from episodic_log.providers.base import BaseProvider, normalize_messages

logger = logging.getLogger(__name__)


class VLLMProvider(BaseProvider):
    """Offline batch LLM provider backed by vLLM.

    Loads the model once with tensor parallelism across *tensor_parallel_size*
    GPUs.  All calls to :meth:`generate_batch` feed prompts directly to
    ``vllm.LLM.generate()`` — vLLM schedules them with PagedAttention and
    continuous batching, saturating GPU memory bandwidth at all times.

    Args:
        model_name: HuggingFace model ID or local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            Use 4 for 7B/14B on 4 H100s; 8 for maximum throughput.
        gpu_memory_utilization: Fraction of GPU VRAM to reserve for KV cache
            (default 0.92 leaves headroom for activations).
        max_model_len: Maximum sequence length (input + output tokens).

    Raises:
        ImportError: If ``vllm`` is not installed.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 4096,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "vllm is required. Install with: pip install vllm"
            ) from exc

        try:
            from transformers import AutoTokenizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers is required for chat template formatting."
            ) from exc

        self._model_name = model_name
        self._tp = tensor_parallel_size
        logger.info(
            "VLLMProvider: loading model=%s tp=%d gpu_mem_util=%.2f",
            model_name, tensor_parallel_size, gpu_memory_utilization,
        )

        self._llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self._SamplingParams = SamplingParams

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        logger.info("VLLMProvider: model loaded.")

    @property
    def model_id(self) -> str:
        return self._model_name

    def _build_prompt(
        self,
        messages: list[dict[str, str]],
        system: str | None,
    ) -> str:
        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(normalize_messages(messages))
        try:
            return self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in chat
            ) + "\nASSISTANT:"

    def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a single response.

        Args:
            messages: Conversation messages (strings or dicts).
            system: Optional system prompt.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature; 0.0 = greedy.

        Returns:
            Assistant response string.
        """
        return self.generate_batch(
            [messages],  # type: ignore[list-item]
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )[0]

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate responses for an arbitrarily large batch of message lists.

        All prompts are submitted to ``vllm.LLM.generate()`` in a single call.
        vLLM's continuous batching scheduler saturates GPU memory bandwidth
        regardless of the number of prompts.  There is no practical limit on
        ``len(batch_messages)`` — pass the entire dataset at once for maximum
        throughput.

        Args:
            batch_messages: One message list per item.
            system: Optional system prompt applied uniformly.
            max_tokens: Maximum new tokens per item.
            temperature: Sampling temperature.

        Returns:
            List of response strings in the same order as *batch_messages*.
        """
        prompts = [self._build_prompt(msgs, system) for msgs in batch_messages]

        params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 1e-6) if temperature > 0 else 0.0,
            top_p=1.0,
        )

        logger.info("VLLMProvider.generate_batch: %d prompts", len(prompts))
        outputs = self._llm.generate(prompts, params)
        return [o.outputs[0].text.strip() for o in outputs]

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Not implemented for vLLM offline provider.

        Tool-calling requires the HuggingFace provider with
        ``apply_chat_template(tools=...)``.  Use
        :class:`~episodic_log.providers.HuggingFaceProvider` for evaluation
        conditions that require tool-use.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "VLLMProvider does not support tool-calling. "
            "Use HuggingFaceProvider for recall/agent conditions."
        )
