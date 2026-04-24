"""Groq API provider with exponential backoff retry."""

from __future__ import annotations

import logging
import time
from typing import Any

from episodic_log.providers.base import BaseProvider, normalize_messages

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 1.0  # seconds
_RETRY_MAX_DELAY = 60.0  # seconds


class GroqProvider(BaseProvider):
    """LLM provider backed by the Groq Chat Completions API.

    Retries on ``RateLimitError`` with exponential backoff, honouring the
    ``retry-after`` header when present.

    Args:
        model: Groq model identifier (e.g. ``"llama-3.1-8b-instant"``).
        api_key: Groq API key.  If ``None``, reads from the ``GROQ_API_KEY``
            environment variable via the ``groq`` SDK's default behaviour.

    Raises:
        ImportError: If the ``groq`` package is not installed.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            import groq as groq_sdk  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'groq' package is required. Install with: pip install groq"
            ) from exc

        self._model = model
        self._client = groq_sdk.Groq(api_key=api_key)
        logger.info("GroqProvider initialised: model=%s", model)

    @property
    def model_id(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response via the Groq Chat Completions API.

        Args:
            messages: Alternating user/assistant strings or chat-message dicts.
            system: Optional system prompt.
            max_tokens: Upper bound on tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The assistant's text response.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        import groq as groq_sdk  # type: ignore[import]

        chat_messages: list[dict[str, str]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(normalize_messages(messages))

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=chat_messages,  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text: str = response.choices[0].message.content or ""
                logger.debug(
                    "GroqProvider: model=%s tokens=%d",
                    self._model,
                    response.usage.completion_tokens if response.usage else -1,
                )
                return text
            except groq_sdk.RateLimitError as exc:
                retry_after = _parse_retry_after(exc)
                delay = retry_after if retry_after else min(
                    _RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY
                )
                logger.warning(
                    "GroqProvider: rate limit (attempt %d/%d) — sleeping %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"GroqProvider: all {_MAX_RETRIES} retry attempts exhausted."
                    ) from exc
            except groq_sdk.APIError as exc:
                logger.error("GroqProvider: API error: %s", exc)
                raise RuntimeError(f"GroqProvider: API error: {exc}") from exc

        raise RuntimeError(f"GroqProvider: all {_MAX_RETRIES} retry attempts exhausted.")


def _parse_retry_after(exc: Any) -> float | None:
    """Extract the Retry-After header value from a RateLimitError if present.

    Args:
        exc: The caught RateLimitError exception.

    Returns:
        Float seconds to wait, or ``None`` if the header is absent.
    """
    try:
        headers = exc.response.headers  # type: ignore[union-attr]
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value is not None:
            return float(value)
    except (AttributeError, ValueError):
        pass
    return None
