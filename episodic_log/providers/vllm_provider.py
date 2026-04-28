"""vLLM offline provider — batch summarization + single-step tool-calling.

Uses vLLM's PagedAttention + continuous batching for maximum throughput.
``generate_batch`` processes hundreds of prompts in one pass.
``generate_with_tools`` runs one agent step via vLLM's fast single-prompt path,
using ``apply_chat_template(tools=...)`` + Qwen ``<tool_call>`` parsing — the
same contract as HuggingFaceProvider so the agent loop works with either backend.

Provider spec: ``vllm:<model_id>`` or ``vllm:<model_id>:tp2`` etc.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from episodic_log.providers.base import BaseProvider, normalize_messages

logger = logging.getLogger(__name__)


class VLLMProvider(BaseProvider):
    """Offline LLM provider backed by vLLM.

    Supports both batch inference (summarization, judging) and interactive
    tool-calling (evaluation agent loop) on the same loaded model.

    Args:
        model_name: HuggingFace model ID or local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            tp=1 fits Qwen3-32B BF16 on a single 80 GB H100 (64 GB weights +
            ~12 GB KV cache at max_model_len=32768).
        gpu_memory_utilization: Fraction of VRAM reserved for KV cache.
        max_model_len: Maximum sequence length (input + output tokens).
            49152 gives 16 k headroom for dense echo-recall prompts on A100 40 GB.

    Raises:
        ImportError: If ``vllm`` is not installed.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.87,
        max_model_len: int = 49_152,
        max_num_batched_tokens: int = 65_536,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import]
        except ImportError as exc:
            if "vllm" in str(exc).lower() or "No module named" in str(exc):
                raise ImportError(
                    "vllm is required. Install with: pip install vllm"
                ) from exc
            raise

        try:
            from transformers import AutoTokenizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers is required for chat template formatting."
            ) from exc

        self._model_name = model_name
        self._tp = tensor_parallel_size
        logger.info(
            "VLLMProvider: loading model=%s tp=%d gpu_mem_util=%.2f "
            "max_model_len=%d max_num_batched_tokens=%d",
            model_name, tensor_parallel_size, gpu_memory_utilization,
            max_model_len, max_num_batched_tokens,
        )

        hf_token: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        self._llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self._SamplingParams = SamplingParams

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=hf_token or True
        )
        logger.info("VLLMProvider: model loaded.")

    @property
    def model_id(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict] | None = None,
    ) -> str:
        """Apply the tokenizer chat template, with optional tool schema injection."""
        chat: list[dict[str, Any]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(normalize_messages(messages))  # type: ignore[arg-type]

        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools

        def _try(**extra: Any) -> str:
            try:
                return self._tokenizer.apply_chat_template(chat, **{**kwargs, **extra})
            except TypeError:
                extra.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(chat, **{**kwargs, **extra})

        try:
            return _try(enable_thinking=False)
        except Exception:
            pass
        try:
            return _try()
        except Exception:
            return "\n".join(
                f"{m['role'].upper()}: {m.get('content', '')}" for m in chat
            ) + "\nASSISTANT:"

    def _sampling_params(self, max_tokens: int, temperature: float) -> Any:
        return self._SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 1e-6) if temperature > 0 else 0.0,
            top_p=1.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        """Generate responses for a large batch in a single vLLM pass.

        All prompts are submitted to ``vllm.LLM.generate()`` at once.
        vLLM's continuous batching saturates GPU memory bandwidth regardless
        of batch size — pass the entire dataset for maximum throughput.

        Args:
            batch_messages: One message list per item.
            system: Optional system prompt applied uniformly.
            max_tokens: Maximum new tokens per item.
            temperature: Sampling temperature.

        Returns:
            List of response strings in the same order as *batch_messages*.
        """
        prompts = [self._build_prompt(msgs, system) for msgs in batch_messages]  # type: ignore[arg-type]

        params = self._sampling_params(max_tokens, temperature)

        logger.info("VLLMProvider.generate_batch: %d prompts", len(prompts))
        long_prompts = sum(1 for p in prompts if len(p) > 60_000)
        if long_prompts:
            logger.warning(
                "VLLMProvider.generate_batch: %d/%d prompts exceed 60k chars — "
                "vLLM will truncate to max_model_len=%d",
                long_prompts, len(prompts), self._llm.llm_engine.model_config.max_model_len,
            )
        outputs = self._llm.generate(prompts, params)
        return [o.outputs[0].text.strip() for o in outputs]

    def _parse_tool_or_text(self, raw_text: str) -> dict[str, Any]:
        """Parse one model output into a tool_call or text response dict."""
        parse_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", parse_text, re.DOTALL)
        if match:
            try:
                payload: dict[str, Any] = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"VLLMProvider: failed to parse tool_call JSON: {exc}\n"
                    f"Raw: {match.group(1)!r}"
                ) from exc
            tool_args = payload.get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            return {
                "type": "tool_call",
                "content": "",
                "tool_name": payload.get("name", ""),
                "tool_args": tool_args,
                "raw_message": {"role": "assistant", "content": raw_text},
            }
        return {
            "type": "text",
            "content": parse_text,
            "tool_name": "",
            "tool_args": {},
            "raw_message": {"role": "assistant", "content": raw_text},
        }

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run one tool-calling agent step (single session).

        Delegates to :meth:`generate_with_tools_batch` for a batch of one.
        """
        return self.generate_with_tools_batch(
            [messages], tools, system=system,
            max_tokens=max_tokens, temperature=temperature,
        )[0]

    def generate_with_tools_batch(
        self,
        batch_messages: list[list[dict[str, Any]]],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Run one agent step for many sessions simultaneously in a single vLLM call.

        All sessions are at the same step in their agent loop.  Prompts are
        built in parallel (CPU), submitted to vLLM in one batch, and each
        output is independently parsed into a tool_call or text response.

        This is the core primitive for the batch agent loop — submitting
        N sessions at once keeps the GPU busy instead of waiting for one
        session's tool execution before starting the next forward pass.

        Args:
            batch_messages: One message list per active session.
            tools: OpenAI-format tool schema dicts (same for all sessions).
            system: Optional system prompt applied uniformly.
            max_tokens: Maximum new tokens per output.
            temperature: Sampling temperature; 0.0 = greedy.

        Returns:
            Ordered list of response dicts (same schema as
            :meth:`generate_with_tools`).
        """
        prompts = [self._build_prompt(msgs, system, tools) for msgs in batch_messages]
        params = self._sampling_params(max_tokens, temperature)
        logger.debug("VLLMProvider.generate_with_tools_batch: %d prompts", len(prompts))
        outputs = self._llm.generate(prompts, params)
        return [self._parse_tool_or_text(o.outputs[0].text.strip()) for o in outputs]
