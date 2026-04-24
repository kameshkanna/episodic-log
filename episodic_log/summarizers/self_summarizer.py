"""Self summarizer — agent model writes its own memory log entries.

Uses the same agent model (via any :class:`~episodic_log.providers.base.BaseProvider`
implementation) to write 1-2 sentence summaries of
:class:`~episodic_log.core.turn_event.TurnEvent` records.

This is the *highest contamination risk* summarizer in the ablation study:
because the same model that will be evaluated later also produces the summaries,
it may encode information in a way that inflates recall for that model specifically.
"""

from __future__ import annotations

import logging

from episodic_log.core.turn_event import TurnEvent
from episodic_log.core.turn_summary import TurnSummary
from episodic_log.providers.base import BaseProvider
from episodic_log.summarizers.base import AbstractSummarizer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are maintaining your own memory log. "
    "Write with precision: include every file path, error message, "
    "and tool name exactly as it appears. "
    "Your output must be exactly 1-2 sentences."
)

_USER_PROMPT_TEMPLATE = (
    "You are summarizing your own action for your memory log. "
    "Write 1-2 sentences capturing exactly what you did, what you observed, "
    "and what files or tools were involved. Be precise.\n\n"
    "Turn ID: {turn_id}\n"
    "Session ID: {session_id}\n"
    "Role: {role}\n"
    "Type: {type}\n"
    "Tool: {tool_name}\n"
    "File: {file_path}\n"
    "Content:\n{content}"
)

_MAX_TOKENS = 128
_TEMPERATURE = 0.0


class SelfSummarizer(AbstractSummarizer):
    """Agent model writes its own memory log entries.

    Uses the agent's own model (provided via *provider*) to produce memory
    summaries.  This is the **highest contamination risk** condition in the
    Conversational Hallucination Drift ablation study because the model both
    writes and is later evaluated against the same summaries.

    The prompt instructs the model to write in first-person and to preserve all
    technical identifiers verbatim, maximising the fidelity of the self-log.

    Args:
        provider: A :class:`~episodic_log.providers.base.BaseProvider` instance
            wrapping the agent model.
    """

    def __init__(self, provider: BaseProvider) -> None:
        if not isinstance(provider, BaseProvider):
            raise TypeError(
                f"provider must be a BaseProvider instance, got {type(provider)}"
            )
        self._provider = provider

    @property
    def method(self) -> str:
        """Return the method identifier ``"self"``."""
        return "self"

    def summarize(self, event: TurnEvent) -> TurnSummary:
        """Generate a 1-2 sentence self-authored summary of *event*.

        Formats the event fields into a first-person prompt and calls the agent
        provider with deterministic settings (``temperature=0.0``).

        Args:
            event: The :class:`~episodic_log.core.turn_event.TurnEvent` to summarise.

        Returns:
            A :class:`~episodic_log.core.turn_summary.TurnSummary` whose summary
            was produced by the agent model.

        Raises:
            TypeError: If *event* is not a :class:`~episodic_log.core.turn_event.TurnEvent`.
            RuntimeError: If the provider call fails (propagated from
                :meth:`~episodic_log.providers.base.BaseProvider.generate`).
        """
        if not isinstance(event, TurnEvent):
            raise TypeError(f"event must be a TurnEvent, got {type(event)}")

        user_message = _format_prompt(event)

        logger.debug(
            "SelfSummarizer: calling provider for session=%s turn=%s",
            event.session_id,
            event.turn_id,
        )

        raw_summary = self._provider.generate(
            messages=[{"role": "user", "content": user_message}],
            system=_SYSTEM_PROMPT,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )

        summary = raw_summary.strip()
        logger.debug(
            "SelfSummarizer: session=%s turn=%s summary_len=%d",
            event.session_id,
            event.turn_id,
            len(summary),
        )

        return TurnSummary(
            turn_id=event.turn_id,
            session_id=event.session_id,
            summary=summary,
            method=self.method,
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _format_prompt(event: TurnEvent) -> str:
    """Format the user-turn prompt for the self summarizer.

    Args:
        event: The event whose fields are embedded in the prompt.

    Returns:
        A fully-formatted prompt string ready for model inference.
    """
    return _USER_PROMPT_TEMPLATE.format(
        turn_id=event.turn_id,
        session_id=event.session_id,
        role=event.role.value,
        type=event.type.value,
        tool_name=event.tool_name if event.tool_name is not None else "none",
        file_path=event.file_path if event.file_path is not None else "none",
        content=event.content,
    )
