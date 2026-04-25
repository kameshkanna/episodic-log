"""Scout LLM-based summarizer.

Uses a small language model (via any :class:`~episodic_log.providers.base.BaseProvider`
implementation) to write 1-2 sentence summaries of
:class:`~episodic_log.core.turn_event.TurnEvent` records.

Summaries are consumed exclusively by the BM25 search index and are never
shown directly to the agent.
"""

from __future__ import annotations

import logging

from episodic_log.core.turn_event import TurnEvent
from episodic_log.core.turn_summary import TurnSummary
from episodic_log.providers.base import BaseProvider
from episodic_log.summarizers.base import AbstractSummarizer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise technical archivist. "
    "Your output must be exactly 1-2 sentences. "
    "Preserve exact file paths, error messages, and tool names verbatim. "
    "Do not paraphrase technical identifiers."
)

_USER_PROMPT_TEMPLATE = (
    "Summarize this agent action in 1-2 sentences. "
    "Preserve exact file paths, error messages, and tool names verbatim. "
    "Be specific, not vague.\n\n"
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
_MAX_CONTENT_CHARS = 1500


class ScoutSummarizer(AbstractSummarizer):
    """LLM-based summarizer intended for a small, fast inference model.

    Calls the provided :class:`~episodic_log.providers.base.BaseProvider` once
    per :class:`~episodic_log.core.turn_event.TurnEvent` to generate a concise
    1-2 sentence summary that preserves all technical identifiers verbatim.

    This summarizer introduces a *low contamination risk* relative to
    :class:`~episodic_log.summarizers.echo.EchoSummarizer` because
    it uses a separate, smaller model rather than the agent model itself.

    Args:
        provider: Any :class:`~episodic_log.providers.base.BaseProvider` instance
            that can serve the summary request.  Lightweight models (e.g.
            ``claude-haiku``, ``gpt-3.5-turbo``) are recommended.
    """

    def __init__(self, provider: BaseProvider) -> None:
        if not isinstance(provider, BaseProvider):
            raise TypeError(
                f"provider must be a BaseProvider instance, got {type(provider)}"
            )
        self._provider = provider

    @property
    def method(self) -> str:
        """Return the method identifier ``"scout"``."""
        return "scout"

    def summarize(self, event: TurnEvent) -> TurnSummary:
        """Generate a 1-2 sentence LLM summary of *event*.

        Formats the event fields into a structured prompt and calls the provider
        with deterministic settings (``temperature=0.0``).

        Args:
            event: The :class:`~episodic_log.core.turn_event.TurnEvent` to summarise.

        Returns:
            A :class:`~episodic_log.core.turn_summary.TurnSummary` whose summary
            was produced by the small inference model.

        Raises:
            TypeError: If *event* is not a :class:`~episodic_log.core.turn_event.TurnEvent`.
            RuntimeError: If the provider call fails (propagated from
                :meth:`~episodic_log.providers.base.BaseProvider.generate`).
        """
        if not isinstance(event, TurnEvent):
            raise TypeError(f"event must be a TurnEvent, got {type(event)}")

        user_message = _format_prompt(event)

        logger.debug(
            "ScoutSummarizer: calling provider for session=%s turn=%s",
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
            "ScoutSummarizer: session=%s turn=%s summary_len=%d",
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

    def summarize_batch(
        self,
        events: list[TurnEvent],
        batch_size: int = 32,
    ) -> list[TurnSummary]:
        """Summarize *events* in batches using a single model.generate() per batch.

        When the provider exposes ``generate_batch``, all turns in a batch are
        tokenized together and processed in one forward pass, giving ~20× the
        throughput of sequential single-item calls.  Falls back to the base
        sequential loop for providers that lack ``generate_batch``.

        Args:
            events: All events to summarise (typically one full session).
            batch_size: Number of turns per ``generate_batch`` call. Tune to
                GPU VRAM; 32 is safe for a 7B model on an 80 GB H100.

        Returns:
            List of :class:`TurnSummary` objects in the same order as *events*.
        """
        if not hasattr(self._provider, "generate_batch"):
            return super().summarize_batch(events)

        results: list[TurnSummary] = []
        for start in range(0, len(events), batch_size):
            chunk = events[start: start + batch_size]
            batch_messages = [
                [{"role": "user", "content": _format_prompt(e)}] for e in chunk
            ]
            raw_summaries = self._provider.generate_batch(  # type: ignore[attr-defined]
                batch_messages,
                system=_SYSTEM_PROMPT,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
            )
            for event, raw in zip(chunk, raw_summaries):
                results.append(TurnSummary(
                    turn_id=event.turn_id,
                    session_id=event.session_id,
                    summary=raw.strip(),
                    method=self.method,
                ))
        return results


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _format_prompt(event: TurnEvent) -> str:
    """Format the user-turn prompt for the scout summarizer.

    Args:
        event: The event whose fields are embedded in the prompt.

    Returns:
        A fully-formatted prompt string ready for model inference.
    """
    content = event.content
    if len(content) > _MAX_CONTENT_CHARS:
        content = content[:_MAX_CONTENT_CHARS] + " …[truncated]"
    return _USER_PROMPT_TEMPLATE.format(
        turn_id=event.turn_id,
        session_id=event.session_id,
        role=event.role.value,
        type=event.type.value,
        tool_name=event.tool_name if event.tool_name is not None else "none",
        file_path=event.file_path if event.file_path is not None else "none",
        content=content,
    )
