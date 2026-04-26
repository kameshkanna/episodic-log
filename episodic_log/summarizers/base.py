"""Abstract base class for all TurnEvent summarizers.

Summarizers generate one-line :class:`~episodic_log.core.turn_summary.TurnSummary`
records stored in ``<summaries_dir>/<method>.jsonl``.  These are used by memory
conditions — either dumped in full into the agent's context (recall conditions)
or searched via keyword grep (grep_recall) or keyword-overlap scoring (topk).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from episodic_log.core.turn_event import TurnEvent
from episodic_log.core.turn_summary import TurnSummary


class AbstractSummarizer(ABC):
    """Generates a one-line summary for a :class:`~episodic_log.core.turn_event.TurnEvent`.

    Summaries are stored as ``{summary, turn_id}`` JSONL records consumed by
    memory conditions.  In recall conditions, all summaries are injected into
    the agent's first message.  In grep_recall, the agent searches them by
    keyword.  In topk, the top-k by keyword overlap are pre-loaded verbatim.

    Concrete subclasses must implement :attr:`method` and :meth:`summarize`.
    """

    @property
    @abstractmethod
    def method(self) -> str:
        """Identifier for this summarization strategy.

        Expected values: ``"lexical"`` | ``"scout"`` | ``"echo"``.
        Used as the ``method`` field in every :class:`~episodic_log.core.turn_summary.TurnSummary`
        this summarizer produces and as the JSONL output file name stem.

        Returns:
            A short lowercase string uniquely identifying the strategy.
        """
        ...

    @abstractmethod
    def summarize(self, event: TurnEvent) -> TurnSummary:
        """Produce a one-line summary of *event*.

        Args:
            event: The :class:`~episodic_log.core.turn_event.TurnEvent` to summarise.

        Returns:
            A :class:`~episodic_log.core.turn_summary.TurnSummary` whose
            ``turn_id`` and ``session_id`` match those of *event* and whose
            ``method`` matches :attr:`method`.

        Raises:
            TypeError: If *event* is not a :class:`~episodic_log.core.turn_event.TurnEvent`.
        """
        ...

    def summarize_batch(self, events: list[TurnEvent]) -> list[TurnSummary]:
        """Summarize a list of events, returning one :class:`TurnSummary` per event.

        The default implementation calls :meth:`summarize` sequentially.
        Subclasses backed by a batching-capable provider should override this
        to issue a single batched inference call instead.

        Args:
            events: Ordered list of :class:`TurnEvent` objects to summarise.

        Returns:
            List of :class:`TurnSummary` objects in the same order as *events*.
        """
        return [self.summarize(e) for e in events]
