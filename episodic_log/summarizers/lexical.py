"""Deterministic lexical summarizer.

Produces summaries by extracting fields from a :class:`~episodic_log.core.turn_event.TurnEvent`
into a fixed template.  No model inference is required — this is the cheapest
and most reproducible summarization strategy.
"""

from __future__ import annotations

import json
import logging

from episodic_log.core.turn_event import EventType, TurnEvent
from episodic_log.core.turn_summary import TurnSummary
from episodic_log.summarizers.base import AbstractSummarizer

logger = logging.getLogger(__name__)

# Content truncation limits per event type.
_MESSAGE_CONTENT_LIMIT = 200
_TOOL_RESULT_CONTENT_LIMIT = 150
_DEFAULT_CONTENT_LIMIT = 120


class LexicalSummarizer(AbstractSummarizer):
    """Deterministic, template-based summarizer with no model dependency.

    Produces a single-line summary by extracting metadata fields from a
    :class:`~episodic_log.core.turn_event.TurnEvent` and formatting them as:

    ``[turn_{id}] {role} | {type} | tool={tool_name} | file={file_path} | {content_excerpt}``

    Content excerpts are truncated according to event type:

    * :attr:`~episodic_log.core.turn_event.EventType.MESSAGE`: first 200 chars.
    * :attr:`~episodic_log.core.turn_event.EventType.TOOL_CALL`: key args extracted
      from JSON content (falling back to first 120 chars on parse failure).
    * :attr:`~episodic_log.core.turn_event.EventType.TOOL_RESULT`: first 150 chars.
    * All other types: first 120 chars.

    This summarizer is always available without any provider configuration and
    carries zero risk of model-induced contamination.
    """

    @property
    def method(self) -> str:
        """Return the method identifier ``"lexical"``."""
        return "lexical"

    def summarize(self, event: TurnEvent) -> TurnSummary:
        """Produce a deterministic one-line summary of *event*.

        Args:
            event: The :class:`~episodic_log.core.turn_event.TurnEvent` to summarise.

        Returns:
            A :class:`~episodic_log.core.turn_summary.TurnSummary` populated with
            the template-formatted summary and ``method="lexical"``.

        Raises:
            TypeError: If *event* is not a :class:`~episodic_log.core.turn_event.TurnEvent`.
        """
        if not isinstance(event, TurnEvent):
            raise TypeError(f"event must be a TurnEvent, got {type(event)}")

        content_excerpt = self._extract_content(event)
        tool_label = event.tool_name if event.tool_name is not None else "none"
        file_label = event.file_path if event.file_path is not None else "none"

        summary = (
            f"[turn_{event.turn_id}] {event.role.value} | {event.type.value} "
            f"| tool={tool_label} | file={file_label} | {content_excerpt}"
        )

        logger.debug(
            "LexicalSummarizer: session=%s turn=%s summary_len=%d",
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
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_content(self, event: TurnEvent) -> str:
        """Return an appropriately truncated / parsed content excerpt.

        The extraction strategy varies by :attr:`~episodic_log.core.turn_event.EventType`:

        * ``MESSAGE``: first :data:`_MESSAGE_CONTENT_LIMIT` characters.
        * ``TOOL_CALL``: attempt to parse content as JSON and render key args;
          fall back to first :data:`_DEFAULT_CONTENT_LIMIT` characters.
        * ``TOOL_RESULT``: first :data:`_TOOL_RESULT_CONTENT_LIMIT` characters.
        * Everything else: first :data:`_DEFAULT_CONTENT_LIMIT` characters.

        Args:
            event: Source :class:`~episodic_log.core.turn_event.TurnEvent`.

        Returns:
            A single-line string excerpt of the event content.
        """
        content = event.content

        if event.type == EventType.MESSAGE:
            return _truncate(content, _MESSAGE_CONTENT_LIMIT)

        if event.type == EventType.TOOL_CALL:
            return self._extract_tool_call_args(content)

        if event.type == EventType.TOOL_RESULT:
            return _truncate(content, _TOOL_RESULT_CONTENT_LIMIT)

        return _truncate(content, _DEFAULT_CONTENT_LIMIT)

    @staticmethod
    def _extract_tool_call_args(content: str) -> str:
        """Parse JSON tool-call content and render its key arguments.

        If the content is valid JSON the top-level keys and a short preview of
        their values are formatted as ``key=value`` pairs separated by commas.
        On any parse error the first :data:`_DEFAULT_CONTENT_LIMIT` characters
        are returned verbatim.

        Args:
            content: Raw content string from a TOOL_CALL event.

        Returns:
            A human-readable, single-line representation of the call arguments.
        """
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            logger.debug("LexicalSummarizer: non-JSON tool_call content, using raw truncation.")
            return _truncate(content, _DEFAULT_CONTENT_LIMIT)

        if not isinstance(parsed, dict):
            return _truncate(content, _DEFAULT_CONTENT_LIMIT)

        parts: list[str] = []
        for key, value in parsed.items():
            value_str = str(value)
            if len(value_str) > 60:
                value_str = value_str[:57] + "..."
            parts.append(f"{key}={value_str}")

        return ", ".join(parts) if parts else _truncate(content, _DEFAULT_CONTENT_LIMIT)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    """Return *text* truncated to *limit* characters, appending ``…`` if cut.

    Args:
        text: Source string to truncate.
        limit: Maximum number of characters to retain.

    Returns:
        Truncated string with trailing ``…`` when the input exceeds *limit*,
        or the original string when it fits within the limit.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "…"
