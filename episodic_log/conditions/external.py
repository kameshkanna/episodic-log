"""External condition — retrieved turns labelled as coming from a *different* agent's session."""

from __future__ import annotations

import logging

from episodic_log.conditions.episodic import EpisodicCondition
from episodic_log.core.log_reader import LogReader, _format_event

logger = logging.getLogger(__name__)

_EXTERNAL_HEADER_PREFIX = "[EXTERNAL SESSION]"


class ExternalCondition(EpisodicCondition):
    """Attribution confusion condition.

    Retrieved turns are presented with a forged provenance header so the model
    sees them as originating from ``"a different agent's session"`` rather than
    its own log.  This probes whether agents over-trust labelled context and
    confabulate cross-session memories.
    """

    name: str = "external"

    def _format_context(self, turn_ids: list[str], reader: LogReader) -> str:
        """Decorate each retrieved turn block with an external-session header.

        Args:
            turn_ids: Ordered list of turn IDs to load and format.
            reader: :class:`~episodic_log.core.log_reader.LogReader` for the session.

        Returns:
            Context string with every turn block prefixed by
            :data:`_EXTERNAL_HEADER_PREFIX`.
        """
        events = reader.load_by_ids(turn_ids)
        blocks = [f"{_EXTERNAL_HEADER_PREFIX} {_format_event(e)}" for e in events]
        return "\n\n".join(blocks)
