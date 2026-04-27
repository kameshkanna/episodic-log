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


class HuggingFaceProvider(BaseProvider):
    """LLM provider backed by a local HuggingFace causal language model.

    Loads the model with ``transformers.AutoModelForCausalLM`` and applies
    optional 4-bit or 8-bit quantization via ``BitsAndBytesConfig``.  Uses
    ``apply_chat_template`` for prompt construction.

    Args:
        model_name: HuggingFace model ID or local path.
        quantization: One of ``"4bit"``, ``"8bit"``, or ``None`` (full precision).
        device_map: Device placement string passed to ``from_pretrained``
            (e.g. ``"auto"``, ``"cuda:0"``).

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
        # Reset model's default sampling params to their neutral values so that
        # greedy decoding (do_sample=False) does not trigger a transformers
        # UserWarning about ignored temperature/top_p/top_k defaults.
        if hasattr(self._model, "generation_config"):
            self._model.generation_config.temperature = 1.0
            self._model.generation_config.top_p = 1.0
            self._model.generation_config.top_k = 0
            self._model.generation_config.do_sample = False
        logger.info("HuggingFaceProvider: model loaded.")

    @property
    def model_id(self) -> str:
        return self._model_name

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate responses for a batch of message lists in a single forward pass.

        Left-pads all sequences to the same length so the batch fits in a single
        ``model.generate()`` call, amortising CUDA launch and Python overhead across
        the entire batch.  Falls back to sequential calls if batch size is 1.

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
            try:
                prompt = self._tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in chat
                ) + "\nASSISTANT:"
            prompts.append(prompt)

        # Left-pad for decoder-only batch generation.
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

        input_len = input_ids.shape[-1]
        try:
            with torch.inference_mode():
                output_ids = self._model.generate(input_ids, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "generate_batch: OOM on batch_size=%d input_len=%d — falling back to sequential",
                len(prompts), input_len,
            )
            del input_ids
            if attention_mask is not None:
                del attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return results

    def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response from the local model.

        Args:
            messages: Alternating user/assistant strings or chat-message dicts.
            system: Optional system prompt prepended as a system-role message.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature; 0.0 uses greedy decoding.

        Returns:
            The assistant's text response (decoded and stripped).

        Raises:
            RuntimeError: If generation fails.
        """
        import torch  # type: ignore[import]

        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(normalize_messages(messages))

        try:
            prompt: str = self._tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: concatenate messages naively.
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in chat
            ) + "\nASSISTANT:"

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = next(self._model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        # Pass attention_mask explicitly to suppress the pad==eos UserWarning.
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

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
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        new_ids = output_ids[0][input_ids.shape[-1]:]
        text: str = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        # Explicitly free GPU memory for the intermediate tensors.
        del input_ids, output_ids, new_ids
        if attention_mask is not None:
            del attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return text.strip()

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run one inference step with tool-calling support for Qwen2.5-Instruct.

        Uses ``apply_chat_template`` with the ``tools=`` parameter so that
        Qwen2.5-Instruct models emit structured ``<tool_call>`` blocks when
        they want to invoke a function.

        The output is parsed as follows:

        - If the decoded text contains a ``<tool_call>`` block, the JSON
          payload is extracted and returned as a ``type="tool_call"`` response.
        - Otherwise the text is returned as a ``type="text"`` response.

        The ``raw_message`` field in the returned dict is always a complete
        assistant message dict ready to be appended to *messages* before the
        next call.  Tool result messages must use the Qwen-native format::

            {"role": "tool", "content": <result_str>, "name": <tool_name>}

        Args:
            messages: Full conversation history as standard chat-message dicts.
                Tool result turns should follow the Qwen tool-result format.
            tools: List of OpenAI-format tool schema dicts.
            system: Optional system prompt.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature; 0.0 uses greedy decoding.

        Returns:
            Dict with keys ``"type"``, ``"content"``, ``"tool_name"``,
            ``"tool_args"``, and ``"raw_message"``.  See
            :meth:`~episodic_log.providers.base.BaseProvider.generate_with_tools`
            for the full contract.

        Raises:
            RuntimeError: If generation or JSON parsing of a tool call fails.
        """
        import torch  # type: ignore[import]

        chat: list[dict[str, Any]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(messages)

        def _apply_template(**kwargs: Any) -> str:
            """Apply chat template with graceful degradation for unsupported kwargs."""
            try:
                return self._tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True, **kwargs
                )
            except TypeError:
                # Older tokenizers (e.g. Qwen2.5) don't accept enable_thinking.
                kwargs.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True, **kwargs
                )

        try:
            # enable_thinking=False: skip Qwen3's chain-of-thought reasoning so the
            # agent loop doesn't burn its token budget on <think> blocks per call.
            prompt: str = _apply_template(tools=tools, enable_thinking=False)
        except Exception as exc:
            logger.warning(
                "generate_with_tools: apply_chat_template with tools failed (%s); "
                "falling back to tool-schema-free template.",
                exc,
            )
            try:
                prompt = _apply_template(enable_thinking=False)
            except Exception:
                prompt = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in chat
                ) + "\nASSISTANT:"

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = next(self._model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

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
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        new_ids = output_ids[0][input_ids.shape[-1]:]
        raw_text: str = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        del input_ids, output_ids, new_ids
        if attention_mask is not None:
            del attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ------------------------------------------------------------------ #
        # Parse tool call vs. plain text                                       #
        # Qwen3 prepends <think>...</think> blocks; strip them before parsing  #
        # so that tool_call detection is not confused by reasoning content.   #
        # ------------------------------------------------------------------ #
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

            raw_message: dict[str, Any] = {
                "role": "assistant",
                "content": raw_text,
            }
            logger.debug(
                "generate_with_tools: tool_call detected name=%s args=%s",
                tool_name,
                tool_args,
            )
            return {
                "type": "tool_call",
                "content": "",
                "tool_name": tool_name,
                "tool_args": tool_args,
                "raw_message": raw_message,
            }

        # Plain text response — the model has produced a final answer.
        # Use parse_text (thinking stripped) as the actual answer content so
        # downstream components don't receive raw <think> blocks.
        raw_message = {"role": "assistant", "content": raw_text}
        logger.debug("generate_with_tools: text response len=%d", len(parse_text))
        return {
            "type": "text",
            "content": parse_text,
            "tool_name": "",
            "tool_args": {},
            "raw_message": raw_message,
        }


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
