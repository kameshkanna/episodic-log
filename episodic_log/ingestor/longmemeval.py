"""LongMemEval dataset ingestor.

Converts HuggingFace ``xiaowu0162/longmemeval-cleaned`` instances into
immutable :class:`~episodic_log.core.turn_event.TurnEvent` records persisted
to ``<data_dir>/<session_id>/log.jsonl``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from episodic_log.core.turn_event import EventRole, EventType, TurnEvent

logger = logging.getLogger(__name__)

# Synthetic epoch used when no real timestamps are available for ingested sessions.
_SYNTHETIC_EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Zero-padding width for turn_id strings.
_TURN_ID_WIDTH = 4

# HuggingFace dataset coordinates.
# longmemeval_s_cleaned.json: single-session instances with haystack_sessions,
# answer_session_ids, question, answer, question_type — all fields needed.
_HF_DATASET_NAME = "xiaowu0162/longmemeval-cleaned"
_HF_SESSION_FILE = "longmemeval_s_cleaned.json"


@dataclass
class IngestedSession:
    """Result of ingesting one LongMemEval instance.

    Attributes:
        session_id: Derived from the instance's ``question_id`` (lowercased,
            whitespace replaced with underscores).
        log_path: Absolute path to the ``log.jsonl`` file containing all
            :class:`~episodic_log.core.turn_event.TurnEvent` records.
        summaries_dir: Absolute path to the directory where summarizer JSONL
            outputs for this session should be written.
        question: The evaluation question posed to the agent.
        answer: Ground-truth answer string.  Lists are joined with ``"; "`` during ingestion.
        evidence_turn_ids: Ordered list of zero-padded ``turn_id`` strings that
            correspond to ``evidence_session_ids`` from the original instance.
        question_type: One of ``"single-session-user"``,
            ``"single-session-assistant"``, ``"multi-session"``,
            ``"temporal-reasoning"``, ``"knowledge-update"``.
        question_id: Original ``question_id`` from the dataset instance.
    """

    session_id: str
    log_path: Path
    summaries_dir: Path
    question: str
    answer: str | list[str]
    evidence_turn_ids: list[str]
    question_type: str
    question_id: str


class LongMemEvalIngestor:
    """Converts ``xiaowu0162/longmemeval-cleaned`` instances into our TurnEvent log format.

    Each ``session_history`` turn becomes a :class:`~episodic_log.core.turn_event.TurnEvent` with:

    * ``role``: :attr:`~episodic_log.core.turn_event.EventRole.USER` for ``"user"``
      turns, :attr:`~episodic_log.core.turn_event.EventRole.ASSISTANT` for ``"assistant"`` turns.
    * ``type``: :attr:`~episodic_log.core.turn_event.EventType.MESSAGE` for both.
    * ``turn_id``: Zero-padded index ``"0000"``, ``"0001"``, … within the session.
    * ``session_id``: Derived from ``question_id`` by lowercasing and replacing
      whitespace with underscores.
    * ``content``: Verbatim turn content string.
    * ``timestamp``: Synthetic — ``_SYNTHETIC_EPOCH + turn_index * 1 second``.

    ``evidence_session_ids`` (integer list) are converted to zero-padded
    ``turn_id`` strings using the same padding width.

    Args:
        data_dir: Root directory under which per-session subdirectories are
            created.  Each session lives at ``<data_dir>/<session_id>/``.
    """

    def __init__(self, data_dir: Path) -> None:
        if not isinstance(data_dir, Path):
            raise TypeError(f"data_dir must be a Path, got {type(data_dir)}")
        self._data_dir = data_dir.resolve()
        logger.debug("LongMemEvalIngestor initialised with data_dir=%s", self._data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, instance: dict[str, Any], split: str = "test") -> IngestedSession:
        """Convert one LongMemEval instance to ``log.jsonl`` and return metadata.

        The ``s_cleaned`` format stores conversations as a haystack of multiple
        sessions (``haystack_sessions`` / ``haystack_session_ids``).  All sessions
        are flattened into one ordered turn list; ``answer_session_ids`` (string
        session IDs) are resolved to contiguous ``turn_id`` ranges for
        ``evidence_turn_ids``.

        The method is idempotent: the ``log.jsonl`` is overwritten on every call.

        Args:
            instance: A single LongMemEval dict as loaded from HuggingFace.
            split: Dataset split name used for diagnostic logging (not stored).

        Returns:
            An :class:`IngestedSession` describing the persisted session.

        Raises:
            KeyError: If ``instance`` is missing required fields.
            ValueError: If a turn's role string is unrecognised.
        """
        self._validate_instance(instance)

        question_id: str = instance["question_id"]
        session_id: str = _question_id_to_session_id(question_id)
        question: str = instance["question"]
        raw_answer: str | list[str] = instance["answer"]
        answer: str = _normalise_answer(raw_answer)
        question_type: str = instance.get("question_type", "")

        # Flatten all haystack sessions into one ordered turn list, tracking
        # the [start, end) turn-index range for each named session so we can
        # resolve answer_session_ids to specific turn_ids later.
        haystack_sessions: list[list[dict[str, str]]] = instance.get("haystack_sessions") or []
        haystack_session_ids: list[str] = instance.get("haystack_session_ids") or []
        answer_session_ids: list[str] = instance.get("answer_session_ids") or []

        all_turns: list[dict[str, str]] = []
        session_turn_ranges: dict[str, tuple[int, int]] = {}
        for sess_id, sess_turns in zip(haystack_session_ids, haystack_sessions):
            start = len(all_turns)
            if isinstance(sess_turns, list):
                all_turns.extend(sess_turns)
            end = len(all_turns)
            session_turn_ranges[sess_id] = (start, end)

        if not all_turns:
            logger.warning(
                "haystack_sessions is empty for question_id=%s — session will have no turns.",
                question_id,
            )

        # Map answer session string IDs → turn_ids in the flat log.
        evidence_turn_ids: list[str] = []
        for ans_sess_id in answer_session_ids:
            if ans_sess_id in session_turn_ranges:
                start, end = session_turn_ranges[ans_sess_id]
                evidence_turn_ids.extend(_pad_turn_id(i) for i in range(start, end))
            else:
                logger.warning(
                    "answer_session_id %r not found in haystack_session_ids for question_id=%s",
                    ans_sess_id, question_id,
                )

        session_dir = self._data_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        summaries_dir = session_dir / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)

        log_path = session_dir / "log.jsonl"
        events = self._build_events(all_turns, session_id)
        _write_jsonl(log_path, events)

        logger.info(
            "Ingested session %s | split=%s | turns=%d | evidence_turns=%d | log=%s",
            session_id,
            split,
            len(events),
            len(evidence_turn_ids),
            log_path,
        )

        return IngestedSession(
            session_id=session_id,
            log_path=log_path,
            summaries_dir=summaries_dir,
            question=question,
            answer=answer,
            evidence_turn_ids=evidence_turn_ids,
            question_type=question_type,
            question_id=question_id,
        )

    def ingest_batch(
        self,
        instances: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> list[IngestedSession]:
        """Ingest multiple LongMemEval instances with optional tqdm progress.

        Args:
            instances: List of raw LongMemEval dicts.
            show_progress: When ``True`` (default) display a tqdm progress bar.

        Returns:
            Ordered list of :class:`IngestedSession` objects, one per input instance.

        Raises:
            TypeError: If *instances* is not a list.
        """
        if not isinstance(instances, list):
            raise TypeError(f"instances must be a list, got {type(instances)}")

        results: list[IngestedSession] = []
        iterator = tqdm(
            instances,
            desc="Ingesting LongMemEval sessions",
            unit="session",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for instance in iterator:
            session = self.ingest(instance)
            results.append(session)
            if show_progress:
                iterator.set_postfix({"last": session.session_id[:24]})

        logger.info("Batch ingestion complete: %d sessions written.", len(results))
        return results

    @staticmethod
    def load_dataset(n: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
        """Load LongMemEval single-session instances from HuggingFace.

        Downloads ``longmemeval_s_cleaned.json`` (277 MB) from
        ``xiaowu0162/longmemeval-cleaned`` via ``huggingface_hub``.  Each instance
        contains ``haystack_sessions`` (list of conversation sessions),
        ``haystack_session_ids``, ``answer_session_ids``, ``question``, ``answer``,
        and ``question_type`` — all fields needed by :meth:`ingest`.

        The ``datasets`` library is intentionally bypassed because the repo also
        contains a 2.7 GB multi-session file that overflows PyArrow's int32 block-size.

        The returned list is shuffled with *seed* for reproducibility; when *n*
        is provided only the first *n* instances after shuffling are returned.

        Args:
            n: Optional upper bound on the number of instances to return.
            seed: Random seed used to shuffle the full instance list.

        Returns:
            A plain Python list of instance dicts.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
            ValueError: If *n* is provided but is not a positive integer.
        """
        import json
        import random

        if n is not None and (not isinstance(n, int) or n <= 0):
            raise ValueError(f"n must be a positive integer or None, got {n!r}")

        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required. Install it with: pip install huggingface-hub"
            ) from exc

        logger.info(
            "Downloading %s / %s from HuggingFace…",
            _HF_DATASET_NAME,
            _HF_SESSION_FILE,
        )
        local_path = hf_hub_download(
            repo_id=_HF_DATASET_NAME,
            filename=_HF_SESSION_FILE,
            repo_type="dataset",
        )
        with open(local_path, "r", encoding="utf-8") as fh:
            instances: list[dict[str, Any]] = json.load(fh)

        rng = random.Random(seed)
        rng.shuffle(instances)
        if n is not None:
            instances = instances[:n]

        logger.info("Loaded %d LongMemEval instances.", len(instances))
        return instances

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_events(
        self,
        session_history: list[dict[str, str]],
        session_id: str,
    ) -> list[TurnEvent]:
        """Convert a ``session_history`` list into ordered :class:`TurnEvent` objects.

        Args:
            session_history: List of ``{"role": str, "content": str}`` dicts.
            session_id: Session identifier embedded in every event.

        Returns:
            Ordered list of :class:`TurnEvent` objects.

        Raises:
            ValueError: If a turn's role string is not ``"user"`` or ``"assistant"``.
        """
        events: list[TurnEvent] = []
        for idx, turn in enumerate(session_history):
            role_str: str = turn.get("role", "")
            content: str = turn.get("content", "")

            role = _parse_role(role_str, idx)
            turn_id = _pad_turn_id(idx)
            timestamp = _SYNTHETIC_EPOCH + timedelta(seconds=idx)

            event = TurnEvent(
                turn_id=turn_id,
                session_id=session_id,
                timestamp=timestamp,
                role=role,
                type=EventType.MESSAGE,
                content=content,
                raw={"role": role_str, "content": content},
                tool_name=None,
                file_path=None,
            )
            events.append(event)

        return events

    @staticmethod
    def _validate_instance(instance: dict[str, Any]) -> None:
        """Fail fast if required LongMemEval fields are missing.

        Args:
            instance: Raw instance dict to validate.

        Raises:
            TypeError: If *instance* is not a dict.
            KeyError: If a required field is absent.
        """
        if not isinstance(instance, dict):
            raise TypeError(f"instance must be a dict, got {type(instance)}")
        for field_name in ("question_id", "question", "answer", "haystack_sessions"):
            if field_name not in instance:
                raise KeyError(
                    f"LongMemEval instance is missing required field: '{field_name}'"
                )


# ------------------------------------------------------------------
# Module-level helper functions
# ------------------------------------------------------------------


def _pad_turn_id(idx: int) -> str:
    """Return a zero-padded turn_id string for a given integer index.

    Args:
        idx: Non-negative integer turn index.

    Returns:
        Zero-padded string of width :data:`_TURN_ID_WIDTH`, e.g. ``"0042"``.
    """
    return str(idx).zfill(_TURN_ID_WIDTH)


def _question_id_to_session_id(question_id: str) -> str:
    """Derive a filesystem-safe session_id from a question_id.

    Converts to lowercase and replaces whitespace characters with underscores.

    Args:
        question_id: Raw ``question_id`` string from the dataset.

    Returns:
        Normalised session identifier string.
    """
    return "_".join(question_id.lower().split())


def _normalise_answer(raw: str | list[str]) -> str:
    """Return the answer as a plain string.

    When the raw answer is a list its elements are joined with ``"; "`` so that
    downstream components always receive a plain string.

    Args:
        raw: Original answer value from the dataset instance.

    Returns:
        Normalised answer string.
    """
    if isinstance(raw, list):
        return "; ".join(str(item) for item in raw)
    return str(raw)


def _parse_role(role_str: str, turn_idx: int) -> EventRole:
    """Map a LongMemEval role string to an :class:`~episodic_log.core.turn_event.EventRole`.

    Args:
        role_str: Raw role string from the dataset (``"user"`` or ``"assistant"``).
        turn_idx: Turn index used in the error message for debuggability.

    Returns:
        The corresponding :class:`~episodic_log.core.turn_event.EventRole` member.

    Raises:
        ValueError: If *role_str* is not a recognised role.
    """
    try:
        return EventRole(role_str.lower())
    except ValueError:
        raise ValueError(
            f"Unrecognised role '{role_str}' at turn index {turn_idx}. "
            f"Expected one of: {[r.value for r in EventRole]}"
        )


def _write_jsonl(path: Path, events: list[TurnEvent]) -> None:
    """Serialise a list of :class:`TurnEvent` objects to a JSONL file.

    The file is written atomically with UTF-8 encoding.  Any existing file at
    *path* is overwritten.

    Args:
        path: Destination file path.
        events: Ordered sequence of :class:`TurnEvent` objects to serialise.
    """
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(event.to_json() + "\n")
