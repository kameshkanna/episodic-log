"""HuggingFace local model provider with 4-bit/8-bit quantization support."""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from typing import Any

from episodic_log.providers.base import BaseProvider, normalize_messages

logger = logging.getLogger(__name__)

# Maximum input tokens before we truncate aggressively to avoid OOM.
_MAX_INPUT_TOKENS: int = 28_000
# Number of recent messages kept when the context is trimmed on OOM retry.
_OOM_TRIM_KEEP_MESSAGES: int = 4


def _flush_cuda() -> None:
    """Clear CUDA cache and run GC."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


class HuggingFaceProvider(BaseProvider):
    """LLM provider backed by a local HuggingFace causal language model.

    Loads the model with ``transformers.AutoModelForCausalLM`` and applies
    optional 4-bit or 8-bit quantization via ``BitsAndBytesConfig``.  Uses
    ``apply_chat_template`` for prompt construction.

    OOM resilience: every generation path catches ``torch.cuda.OutOfMemoryError``,
    flushes the CUDA cache, trims the context to the most recent messages, and
    retries once before re-raising.

    Args:
        model_name: HuggingFace model ID or local path.
        quantization: One of ``"4bit"``, ``"8bit"``, or ``None`` (full precision).
        device_map: Device placement string passed to ``from_pretrained``
            (e.g. ``"auto"``, ``"cuda:0"``).
        max_input_tokens: Hard cap on the number of input tokens.  Prompts
            exceeding this are truncated from the *middle* of the chat history
            (keeping the first user message and recent exchanges).

    Raises:
        ImportError: If ``transformers`` is not installed.
        ValueError: If *quantization* is not one of the accepted values.
    """

    _VALID_QUANTIZATIONS = (None, "4bit", "8bit")

    def __init__(
        self,
        model_name: str,
        quantization: str | None = None,
        device_map: str = "auto",
        max_input_tokens: int = _MAX_INPUT_TOKENS,
    ) -> None:
        if quantization not in self._VALID_QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {self._VALID_QUANTIZATIONS}, got {quantization!r}"
            )
        try:
            import transformers  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required. Install with: pip install transformers"
            ) from exc

        self._model_name = model_name
        self._max_input_tokens = max_input_tokens
        logger.info(
            "HuggingFaceProvider: loading model=%s quantization=%s device_map=%s",
            model_name,
            quantization,
            device_map,
        )

        bnb_config = _build_bnb_config(quantization)
        hf_token: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": "auto",
            "trust_remote_code": True,
            "token": hf_token or True,
        }
        if bnb_config is not None:
            kwargs["quantization_config"] = bnb_config

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=hf_token or True
        )
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, **kwargs
        )
        self._model.eval()
        if hasattr(self._model, "generation_config"):
            self._model.generation_config.temperature = 1.0
            self._model.generation_config.top_p = 1.0
            self._model.generation_config.top_k = 0
            self._model.generation_config.do_sample = False
        logger.info("HuggingFaceProvider: model loaded.")

    @property
    def model_id(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_to_prompt(
        self,
        chat: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> str:
        """Apply the tokenizer chat template, falling back to naive concat."""
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

    def _tokenize(self, prompt: str) -> "torch.Tensor":  # type: ignore[name-defined]
        import torch  # type: ignore[import]
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(self._model.parameters()).device
        return inputs["input_ids"].to(device), inputs.get("attention_mask", None)

    def _trim_chat_to_token_budget(
        self,
        chat: list[dict[str, Any]],
        budget: int,
        tools: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Drop (assistant, tool) exchange pairs until the chat fits in *budget* tokens.

        The anchor (first non-system message — the summary context) is always
        kept.  The most recent *_OOM_TRIM_KEEP_MESSAGES* messages are also kept.
        Old exchanges between the anchor and the tail are removed as atomic pairs
        so the chat history is never left with an orphaned tool message.
        """
        if len(chat) <= 2:
            return chat

        # Index of the first non-system message (anchor — never dropped).
        anchor_idx: int = 1 if (chat and chat[0].get("role") == "system") else 0
        drop_start: int = anchor_idx + 1

        while True:
            prompt = self._chat_to_prompt(chat, tools)
            ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) <= budget:
                break
            if len(chat) <= anchor_idx + 1 + _OOM_TRIM_KEEP_MESSAGES:
                logger.warning(
                    "HuggingFaceProvider: cannot trim below %d messages and %d tokens",
                    len(chat), len(ids),
                )
                break
            # Find and atomically remove the oldest (assistant [+ tool]) exchange.
            dropped = False
            for i in range(drop_start, len(chat)):
                if chat[i].get("role") == "assistant":
                    end = (
                        i + 2
                        if i + 1 < len(chat) and chat[i + 1].get("role") == "tool"
                        else i + 1
                    )
                    removed_roles = [m.get("role") for m in chat[i:end]]
                    del chat[i:end]
                    logger.debug(
                        "HuggingFaceProvider._trim_chat: dropped exchange "
                        "at idx=%d roles=%s remaining=%d",
                        i, removed_roles, len(chat),
                    )
                    dropped = True
                    break
            if not dropped:
                logger.warning(
                    "HuggingFaceProvider._trim_chat: no droppable exchanges found "
                    "(len=%d) — giving up", len(chat),
                )
                break
        return chat

    def _run_generate(
        self,
        input_ids: "torch.Tensor",  # type: ignore[name-defined]
        attention_mask: "torch.Tensor | None",  # type: ignore[name-defined]
        max_tokens: int,
        temperature: float,
    ) -> "torch.Tensor":  # type: ignore[name-defined]
        """Forward pass with explicit gen_kwargs. Does NOT handle OOM."""
        import torch  # type: ignore[import]

        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            return self._model.generate(input_ids, **gen_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate responses for a batch of message lists in a single forward pass.

        Falls back to sequential calls on OOM.

        Args:
            batch_messages: List of message lists, one per item in the batch.
            system: Optional system prompt applied uniformly to every item.
            max_tokens: Maximum new tokens to generate per item.
            temperature: Sampling temperature; 0.0 uses greedy decoding.

        Returns:
            List of stripped response strings, in the same order as *batch_messages*.
        """
        import torch  # type: ignore[import]

        if len(batch_messages) == 1:
            return [self.generate(batch_messages[0], system=system,
                                   max_tokens=max_tokens, temperature=temperature)]

        prompts: list[str] = []
        for messages in batch_messages:
            chat: list[dict[str, str]] = []
            if system:
                chat.append({"role": "system", "content": system})
            chat.extend(normalize_messages(messages))
            prompts.append(self._chat_to_prompt(chat))

        orig_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        try:
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        finally:
            self._tokenizer.padding_side = orig_padding_side

        device = next(self._model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        input_len = input_ids.shape[-1]
        try:
            output_ids = self._run_generate(input_ids, attention_mask, max_tokens, temperature)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "generate_batch: OOM on batch_size=%d input_len=%d — falling back to sequential",
                len(prompts), input_len,
            )
            del input_ids
            if attention_mask is not None:
                del attention_mask
            _flush_cuda()
            return [
                self.generate(msgs, system=system, max_tokens=max_tokens,
                               temperature=temperature)
                for msgs in batch_messages
            ]

        results: list[str] = []
        for i in range(len(prompts)):
            new_ids = output_ids[i][input_len:]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text.strip())

        del input_ids, output_ids
        if attention_mask is not None:
            del attention_mask
        _flush_cuda()

        return results

    def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response from the local model.

        On ``OutOfMemoryError`` the context is trimmed to the last
        ``_OOM_TRIM_KEEP_MESSAGES`` messages and retried once.

        Args:
            messages: Alternating user/assistant strings or chat-message dicts.
            system: Optional system prompt prepended as a system-role message.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature; 0.0 uses greedy decoding.

        Returns:
            The assistant's text response (decoded and stripped).

        Raises:
            torch.cuda.OutOfMemoryError: If OOM persists after trimming.
        """
        import torch  # type: ignore[import]

        chat: list[dict[str, Any]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(normalize_messages(messages))

        chat = self._trim_chat_to_token_budget(chat, self._max_input_tokens)
        prompt = self._chat_to_prompt(chat)

        input_ids, attention_mask = self._tokenize(prompt)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)

        try:
            output_ids = self._run_generate(input_ids, attention_mask, max_tokens, temperature)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "generate: OOM on input_len=%d — trimming to last %d messages and retrying",
                input_ids.shape[-1], _OOM_TRIM_KEEP_MESSAGES,
            )
            del input_ids
            if attention_mask is not None:
                del attention_mask
            _flush_cuda()

            # Trim to system + last N messages and retry.
            trimmed = _keep_system_and_tail(chat, _OOM_TRIM_KEEP_MESSAGES)
            prompt = self._chat_to_prompt(trimmed)
            input_ids, attention_mask = self._tokenize(prompt)
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_ids.device)
            output_ids = self._run_generate(input_ids, attention_mask, max_tokens, temperature)

        new_ids = output_ids[0][input_ids.shape[-1]:]
        text: str = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        del input_ids, output_ids, new_ids
        if attention_mask is not None:
            del attention_mask
        _flush_cuda()

        return text.strip()

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run one inference step with tool-calling support for Qwen2.5/3-Instruct.

        Uses ``apply_chat_template`` with the ``tools=`` parameter so that
        Qwen-Instruct models emit structured ``<tool_call>`` blocks when
        they want to invoke a function.

        On ``OutOfMemoryError`` the context is trimmed and retried once.

        The output is parsed as follows:

        - If the decoded text contains a ``<tool_call>`` block, the JSON
          payload is extracted and returned as a ``type="tool_call"`` response.
        - Otherwise the text is returned as a ``type="text"`` response.

        The ``raw_message`` field is always a complete assistant message dict
        ready to be appended to *messages* before the next call.  Tool result
        messages must use the Qwen-native format::

            {"role": "tool", "content": <result_str>, "name": <tool_name>}

        Args:
            messages: Full conversation history as standard chat-message dicts.
            tools: List of OpenAI-format tool schema dicts.
            system: Optional system prompt.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature; 0.0 uses greedy decoding.

        Returns:
            Dict with keys ``"type"``, ``"content"``, ``"tool_name"``,
            ``"tool_args"``, and ``"raw_message"``.

        Raises:
            RuntimeError: If JSON parsing of a tool call fails.
            torch.cuda.OutOfMemoryError: If OOM persists after trimming.
        """
        import torch  # type: ignore[import]

        chat: list[dict[str, Any]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(messages)

        # Pre-emptively trim if already over budget before we even try.
        chat = self._trim_chat_to_token_budget(chat, self._max_input_tokens, tools)
        prompt = self._chat_to_prompt(chat, tools)

        input_ids, attention_mask = self._tokenize(prompt)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)

        try:
            output_ids = self._run_generate(input_ids, attention_mask, max_tokens, temperature)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "generate_with_tools: OOM on input_len=%d — trimming context and retrying",
                input_ids.shape[-1],
            )
            del input_ids
            if attention_mask is not None:
                del attention_mask
            _flush_cuda()

            trimmed = _keep_system_and_tail(chat, _OOM_TRIM_KEEP_MESSAGES)
            prompt = self._chat_to_prompt(trimmed, tools)
            input_ids, attention_mask = self._tokenize(prompt)
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_ids.device)
            logger.info(
                "generate_with_tools: retrying with trimmed context input_len=%d",
                input_ids.shape[-1],
            )
            output_ids = self._run_generate(input_ids, attention_mask, max_tokens, temperature)

        new_ids = output_ids[0][input_ids.shape[-1]:]
        raw_text: str = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        del input_ids, output_ids, new_ids
        if attention_mask is not None:
            del attention_mask
        _flush_cuda()

        # Strip Qwen3 <think>...</think> blocks before parsing tool calls.
        parse_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        tool_call_match = re.search(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            parse_text,
            re.DOTALL,
        )
        if tool_call_match:
            payload_str = tool_call_match.group(1)
            try:
                payload: dict[str, Any] = json.loads(payload_str)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"generate_with_tools: failed to parse tool_call JSON: {exc}\n"
                    f"Raw payload: {payload_str!r}"
                ) from exc

            tool_name: str = payload.get("name", "")
            tool_args: dict[str, Any] = payload.get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            raw_message: dict[str, Any] = {"role": "assistant", "content": raw_text}
            logger.debug(
                "generate_with_tools: tool_call detected name=%s args=%s",
                tool_name, tool_args,
            )
            return {
                "type": "tool_call",
                "content": "",
                "tool_name": tool_name,
                "tool_args": tool_args,
                "raw_message": raw_message,
            }

        raw_message = {"role": "assistant", "content": raw_text}
        logger.debug("generate_with_tools: text response len=%d", len(parse_text))
        return {
            "type": "text",
            "content": parse_text,
            "tool_name": "",
            "tool_args": {},
            "raw_message": raw_message,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _keep_system_and_tail(
    chat: list[dict[str, Any]],
    keep_tail: int,
) -> list[dict[str, Any]]:
    """Return system message (if any) + the last *keep_tail* messages."""
    system_msgs = [m for m in chat if m.get("role") == "system"]
    non_system = [m for m in chat if m.get("role") != "system"]
    tail = non_system[-keep_tail:] if len(non_system) > keep_tail else non_system
    return system_msgs + tail


def _build_bnb_config(quantization: str | None) -> Any | None:
    """Build a ``BitsAndBytesConfig`` for the requested quantization level.

    Args:
        quantization: ``"4bit"``, ``"8bit"``, or ``None``.

    Returns:
        A ``BitsAndBytesConfig`` instance, or ``None`` for full precision.
    """
    if quantization is None:
        return None

    try:
        from transformers import BitsAndBytesConfig  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for quantization. "
            "Install with: pip install bitsandbytes"
        ) from exc

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    raise ValueError(f"Unsupported quantization: {quantization!r}")
