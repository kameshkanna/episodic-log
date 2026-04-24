"""Abstract base interface for LLM provider backends.

All provider implementations (Groq, HuggingFace, etc.) must subclass
:class:`BaseProvider` so that summarizers, conditions, and the judge remain
provider-agnostic.

Message format:
    Callers may pass either:
    - ``list[str]``: Treated as alternating user/assistant turns
      (index 0 = user, index 1 = assistant, …). Most conditions use this form
      because they build the history as a running list.
    - ``list[dict[str, str]]``: Standard ``{"role": ..., "content": ...}``
      chat-message dicts.

    Use :func:`normalize_messages` to convert either form to the standard dict
    format before passing to an underlying SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def normalize_messages(
    messages: list[str] | list[dict[str, str]],
) -> list[dict[str, str]]:
    """Convert a mixed or string message list to standard chat-message dicts.

    String elements alternate user/assistant (index 0 → user, 1 → assistant, …).
    Dict elements are passed through unchanged.

    Args:
        messages: Either a list of strings or a list of role-content dicts.

    Returns:
        List of ``{"role": str, "content": str}`` dicts.
    """
    _ROLES = ("user", "assistant")
    normalised: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, str):
            role = _ROLES[i % 2]
            normalised.append({"role": role, "content": msg})
        elif isinstance(msg, dict):
            normalised.append(msg)
        else:
            raise TypeError(f"Message at index {i} must be str or dict, got {type(msg)}")
    return normalised


class BaseProvider(ABC):
    """Minimal LLM provider interface consumed by summarizers, conditions, and judge.

    Implementors wrap a specific inference backend (Groq API, local HuggingFace
    model, etc.).  Callers never interact with the underlying SDK directly.
    """

    @property
    def model_id(self) -> str:
        """Short model identifier string (used for logging)."""
        return "unknown"

    @abstractmethod
    def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Run a single inference call and return the assistant text.

        Args:
            messages: Either alternating ``list[str]`` (user/assistant/user/…) or
                standard ``list[dict[str, str]]`` chat-message dicts.
            system: Optional system prompt injected before *messages*.
            max_tokens: Upper bound on generated tokens.
            temperature: Sampling temperature; 0.0 for deterministic decoding.

        Returns:
            The raw assistant-turn text as a single string.

        Raises:
            RuntimeError: If the underlying provider call fails after all retries.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run one inference step with tool-calling support.

        The model may either produce a final text answer or request a tool call.
        The caller is responsible for executing any requested tool, appending
        the result to *messages*, and calling this method again until a text
        response is received or a step budget is exhausted.

        Args:
            messages: Full conversation history as standard
                ``{"role": ..., "content": ...}`` chat-message dicts.  Tool
                result turns must follow the provider's native format.
            tools: List of OpenAI-format tool schema dicts describing the
                available functions (``{"type": "function", "function": {...}}``).
            system: Optional system prompt injected before *messages*.
            max_tokens: Upper bound on generated tokens.
            temperature: Sampling temperature; 0.0 for deterministic decoding.

        Returns:
            A dict with the following keys:

            - ``"type"`` (``str``): ``"tool_call"`` or ``"text"``.
            - ``"content"`` (``str``): Final answer text when
              ``type == "text"``; empty string otherwise.
            - ``"tool_name"`` (``str``): Name of the requested tool when
              ``type == "tool_call"``; empty string otherwise.
            - ``"tool_args"`` (``dict``): Parsed argument dict for the tool
              when ``type == "tool_call"``; empty dict otherwise.
            - ``"raw_message"`` (``dict``): Full assistant message dict
              suitable for appending to *messages* in the next turn.

        Raises:
            RuntimeError: If the underlying provider call fails.
        """
        raise NotImplementedError
