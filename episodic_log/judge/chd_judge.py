"""CHD judge — evaluates predicted answers for Conversational Hallucination Drift.

CHD taxonomy (4 categories):
    commission:     The model claims to have done something it never did.
    omission:       The model fails to recall a clearly stated past action.
    distortion:     The model recalls an event but gets key details wrong.
    confabulation:  The model fabricates plausible-sounding but invented history.
    correct:        The predicted answer is factually consistent with the session log.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from episodic_log.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_CHD_CATEGORIES = ("commission", "omission", "distortion", "confabulation", "correct")

_SYSTEM_PROMPT = """\
You are an expert judge evaluating whether a language model's answer about its
own conversation history is accurate or contains a Conversational Hallucination
Drift (CHD) error.

CHD taxonomy:
  commission    — The model claims to have done/said something it never did.
  omission      — The model forgets or denies a clearly stated past action.
  distortion    — The model recalls an event but gets key details wrong (wrong date, wrong value, etc.).
  confabulation — The model invents plausible-sounding history with no basis in the log.
  correct       — The predicted answer is consistent with the ground truth.

You will be given:
1. The evaluation question.
2. The ground-truth answer from the dataset.
3. The model's predicted answer.
4. (Optional) Relevant verbatim turns from the session log.

Respond with ONLY a JSON object on a single line:
{"verdict": "<category>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}

where <category> is one of: commission, omission, distortion, confabulation, correct.
"""

_VERDICT_RE = re.compile(
    r'"verdict"\s*:\s*"([^"]+)".*?"confidence"\s*:\s*([0-9.]+).*?"reason"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


@dataclass
class JudgeVerdict:
    """Result of a single CHD judge evaluation.

    Attributes:
        verdict: One of ``commission``, ``omission``, ``distortion``,
            ``confabulation``, ``correct``.
        confidence: Float in ``[0.0, 1.0]`` indicating judge confidence.
        reason: One-sentence explanation of the verdict.
        raw_response: Full raw judge output for debugging.
    """

    verdict: str
    confidence: float
    reason: str
    raw_response: str


class CHDJudge:
    """LLM-based judge for Conversational Hallucination Drift classification.

    Uses a structured system prompt to classify predicted answers into one of
    five CHD categories using a provider-backed LLM.

    Args:
        provider: Initialised :class:`~episodic_log.providers.base.BaseProvider`.

    Raises:
        TypeError: If *provider* is not a BaseProvider.
    """

    def __init__(self, provider: BaseProvider) -> None:
        if not isinstance(provider, BaseProvider):
            raise TypeError(f"provider must be a BaseProvider, got {type(provider)}")
        self._provider = provider

    def judge(
        self,
        question: str,
        ground_truth: str,
        predicted: str,
        context_turns: str = "",
    ) -> JudgeVerdict:
        """Classify whether *predicted* contains a CHD error relative to *ground_truth*.

        Args:
            question: The evaluation question posed to the model.
            ground_truth: The authoritative answer from the dataset.
            predicted: The model's predicted answer to evaluate.
            context_turns: Optional verbatim turn text from the session log for
                grounding the judge's assessment.

        Returns:
            A :class:`JudgeVerdict` with the CHD category, confidence, and reason.
        """
        user_msg = _build_judge_input(question, ground_truth, predicted, context_turns)
        raw = self._provider.generate(
            messages=[user_msg],
            system=_SYSTEM_PROMPT,
            max_tokens=256,
            temperature=0.0,
        )
        return _parse_verdict(raw)

    def judge_batch_fast(
        self,
        inputs: list[dict[str, Any]],
    ) -> list[JudgeVerdict]:
        """Judge all inputs in a single ``generate_batch()`` call.

        Intended for vLLM where ``generate_batch()`` accepts an arbitrarily
        large list and processes them with continuous batching.  Falls back to
        sequential ``judge()`` calls for providers without ``generate_batch``.

        Args:
            inputs: List of dicts with keys ``question``, ``ground_truth``,
                ``predicted``, and optionally ``context_turns``.

        Returns:
            Ordered list of :class:`JudgeVerdict` objects.
        """
        if not hasattr(self._provider, "generate_batch"):
            return [
                self.judge(
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                    predicted=item["predicted"],
                    context_turns=item.get("context_turns", ""),
                )
                for item in inputs
            ]

        batch_messages = [
            [{"role": "user", "content": _build_judge_input(
                item["question"], item["ground_truth"],
                item["predicted"], item.get("context_turns", ""),
            )}]
            for item in inputs
        ]
        raw_outputs = self._provider.generate_batch(  # type: ignore[attr-defined]
            batch_messages,
            system=_SYSTEM_PROMPT,
            max_tokens=256,
            temperature=0.0,
        )
        return [_parse_verdict(raw) for raw in raw_outputs]

    def judge_batch(
        self,
        inputs: list[dict[str, Any]],
        max_workers: int = 8,
    ) -> list[JudgeVerdict]:
        """Judge a batch of predictions with concurrent Groq requests.

        Uses :class:`~concurrent.futures.ThreadPoolExecutor` so multiple Groq
        API calls run in parallel.  Order of results matches *inputs*.

        Each element of *inputs* must have keys: ``question``, ``ground_truth``,
        ``predicted``, and optionally ``context_turns``.

        Args:
            inputs: List of judge input dicts.
            max_workers: Maximum concurrent requests.  For Groq free tier
                (30 req/min on 70B) keep this ≤ 10 to avoid rate-limit cascades.

        Returns:
            Ordered list of :class:`JudgeVerdict` objects.

        Raises:
            KeyError: If a required key is missing from any input dict.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        verdicts: list[JudgeVerdict | None] = [None] * len(inputs)

        def _call(idx: int, item: dict[str, Any]) -> tuple[int, JudgeVerdict]:
            verdict = self.judge(
                question=item["question"],
                ground_truth=item["ground_truth"],
                predicted=item["predicted"],
                context_turns=item.get("context_turns", ""),
            )
            return idx, verdict

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_call, i, item): i for i, item in enumerate(inputs)}
            for future in as_completed(futures):
                idx, verdict = future.result()
                verdicts[idx] = verdict

        return verdicts  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_judge_input(
    question: str,
    ground_truth: str,
    predicted: str,
    context_turns: str,
) -> str:
    lines = [
        f"Question: {question}",
        f"Ground truth: {ground_truth}",
        f"Predicted answer: {predicted}",
    ]
    if context_turns:
        lines += ["", "Relevant verbatim turns:", context_turns]
    return "\n".join(lines)


def _parse_verdict(raw: str) -> JudgeVerdict:
    """Parse a JudgeVerdict from the model's raw output.

    Falls back to ``confabulation`` with confidence 0.0 if parsing fails.

    Args:
        raw: Raw model output string.

    Returns:
        A :class:`JudgeVerdict`.
    """
    match = _VERDICT_RE.search(raw)
    if match:
        verdict_str = match.group(1).lower()
        if verdict_str not in _CHD_CATEGORIES:
            verdict_str = "confabulation"
        confidence = float(match.group(2))
        reason = match.group(3)
        return JudgeVerdict(
            verdict=verdict_str,
            confidence=min(1.0, max(0.0, confidence)),
            reason=reason,
            raw_response=raw,
        )

    # Try JSON parse fallback.
    try:
        obj = json.loads(raw.strip())
        return JudgeVerdict(
            verdict=obj.get("verdict", "confabulation").lower(),
            confidence=float(obj.get("confidence", 0.0)),
            reason=obj.get("reason", "parse error"),
            raw_response=raw,
        )
    except (json.JSONDecodeError, ValueError):
        logger.warning("CHDJudge: failed to parse verdict from: %r", raw[:200])
        return JudgeVerdict(
            verdict="confabulation",
            confidence=0.0,
            reason="failed to parse judge output",
            raw_response=raw,
        )
