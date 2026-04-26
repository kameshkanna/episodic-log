"""TopK condition — inject all summaries, pre-load top-k full turns, one-shot answer.

The model receives:
  1. All turn summaries (same as the recall condition).
  2. The full verbatim content of the top-k turns whose summary most closely
     matches the question (simple keyword overlap, no external library needed).

This tests whether pre-loading the most likely relevant turns *before* the
model has to call any tools improves accuracy vs waiting for the model to
issue load_turn calls itself (recall condition).

Ablation axis: k = 3 / 5 / 10 controls how many turns are pre-loaded.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from episodic_log.conditions.base import BaseCondition, ConditionResult
from episodic_log.core.turn_event import TurnEvent
from episodic_log.core.turn_summary import TurnSummary
from episodic_log.providers.base import BaseProvider
from episodic_log.tools.session_tools import format_summaries_as_context

logger = logging.getLogger(__name__)

_VALID_SUMMARY_METHODS: frozenset[str] = frozenset({"lexical", "scout", "echo"})
_VALID_K_VALUES: frozenset[int] = frozenset({3, 5, 10})

_SYSTEM_PROMPT = (
    "You are answering a question about a past conversation.\n"
    "The memory index lists all turns with one-line summaries.\n"
    "The most relevant turns have been pre-loaded below the index.\n"
    "Answer directly and concisely based on the pre-loaded content.\n"
    "If the answer is not in the pre-loaded turns, say so explicitly."
)

_TOKENISE_RE = re.compile(r"\w+")


def _keyword_overlap(query: str, text: str) -> int:
    """Count shared word tokens between query and text (case-insensitive)."""
    q_tokens = set(_TOKENISE_RE.findall(query.lower()))
    t_tokens = _TOKENISE_RE.findall(text.lower())
    return sum(1 for t in t_tokens if t in q_tokens)


class TopKCondition(BaseCondition):
    """Inject all summaries + pre-load top-k turns by keyword match, answer in one shot.

    Args:
        summary_method: One of ``"lexical"``, ``"scout"``, ``"echo"``.
        k: Number of turns to pre-load verbatim.  Supported: 3, 5, 10.

    Raises:
        ValueError: If *summary_method* or *k* are not accepted values.
    """

    def __init__(self, summary_method: str, k: int = 5) -> None:
        if summary_method not in _VALID_SUMMARY_METHODS:
            raise ValueError(
                f"summary_method must be one of {sorted(_VALID_SUMMARY_METHODS)}, "
                f"got {summary_method!r}"
            )
        if k not in _VALID_K_VALUES:
            raise ValueError(
                f"k must be one of {sorted(_VALID_K_VALUES)}, got {k!r}"
            )
        self._summary_method = summary_method
        self._k = k

    @property
    def name(self) -> str:
        return f"topk/{self._summary_method}/k{self._k}"

    def run(self, session_meta: dict, provider: BaseProvider) -> ConditionResult:
        """Inject all summaries + pre-load top-k turns and answer in one generate() call.

        Args:
            session_meta: Must contain ``session_id``, ``question_id``,
                ``question``, ``answer``, ``log_path``, ``summaries_dir``.
            provider: Initialised LLM provider backend.

        Returns:
            :class:`~episodic_log.conditions.base.ConditionResult`.
        """
        _required = {"session_id", "question_id", "question", "answer", "log_path", "summaries_dir"}
        missing = _required - session_meta.keys()
        if missing:
            raise KeyError(f"session_meta missing required keys: {sorted(missing)}")
        if not isinstance(provider, BaseProvider):
            raise TypeError(f"provider must be a BaseProvider, got {type(provider)}")

        session_id: str = session_meta["session_id"]
        question_id: str = session_meta["question_id"]
        question: str = session_meta["question"]
        ground_truth: str = session_meta["answer"]
        summaries_dir = Path(session_meta["summaries_dir"])
        log_path = Path(session_meta["log_path"])

        # ── Load summaries ───────────────────────────────────────────────────
        summary_path = summaries_dir / f"{self._summary_method}.jsonl"
        summaries: list[TurnSummary] = []
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped:
                        try:
                            summaries.append(TurnSummary.from_json(stripped))
                        except (KeyError, ValueError):
                            pass

        # ── Build turn map ───────────────────────────────────────────────────
        turn_map: dict[str, TurnEvent] = {}
        if log_path.exists():
            with log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped:
                        try:
                            event = TurnEvent.from_json(stripped)
                            turn_map[event.turn_id] = event
                        except (KeyError, ValueError):
                            pass

        # ── Select top-k by keyword overlap with the question ────────────────
        scored = sorted(
            summaries,
            key=lambda s: _keyword_overlap(question, s.summary),
            reverse=True,
        )
        top_k = scored[: self._k]
        preloaded_ids = [s.turn_id for s in top_k]

        preloaded_blocks: list[str] = []
        for s in top_k:
            event = turn_map.get(s.turn_id)
            if event is None:
                continue
            role_label = event.role.value.upper()
            preloaded_blocks.append(
                f"[{role_label} | turn {event.turn_id}]\n{event.content}"
            )

        # ── Build prompt ─────────────────────────────────────────────────────
        summary_context = format_summaries_as_context(summaries_dir, self._summary_method)
        sep = "\n" + "─" * 40 + "\n"

        user_msg_parts = []
        if summary_context:
            user_msg_parts.append(
                f"Memory index ({len(summaries)} turns):\n{summary_context}"
            )
        if preloaded_blocks:
            user_msg_parts.append(
                f"Pre-loaded turns (top {self._k} by keyword match):\n"
                + sep.join(preloaded_blocks)
            )
        user_msg_parts.append(f"Question: {question}")
        user_msg = "\n\n".join(user_msg_parts)

        predicted = provider.generate(
            messages=[{"role": "user", "content": user_msg}],
            system=_SYSTEM_PROMPT,
            max_tokens=512,
            temperature=0.0,
        ).strip()

        return ConditionResult(
            session_id=session_id,
            question_id=question_id,
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted,
            condition=self.name,
            summary_method=self._summary_method,
            tool_calls=[],
            turns_loaded=preloaded_ids,
        )
