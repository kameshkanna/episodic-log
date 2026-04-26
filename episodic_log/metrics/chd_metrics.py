"""CHD metrics computation and reporting.

Computes per-category and aggregate hallucination rates from judge verdicts,
and renders comparison tables with Rich.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHD_ERROR_CATEGORIES = ("commission", "omission", "distortion", "confabulation")
_ALL_CATEGORIES = (*_CHD_ERROR_CATEGORIES, "correct")


@dataclass
class CHDMetrics:
    """Aggregated CHD metrics for one model × condition × summarizer combination.

    Attributes:
        condition_name: Condition label (e.g. ``"episodic"``).
        summary_method: Summarizer method used (e.g. ``"structured"``).
        total: Total number of evaluated instances.
        model_slug: Short model identifier (e.g. ``"llama-3.1-8b"``).
        counts: Per-category counts including ``"correct"``.
        accuracy: Fraction of instances with verdict ``"correct"``.
        chd_rate: Fraction of instances with any CHD error category.
        per_category_rate: Dict of per-category error rates.
    """

    condition_name: str
    summary_method: str
    total: int
    model_slug: str = "unknown"
    counts: dict[str, int] = field(default_factory=dict)
    accuracy: float = 0.0
    chd_rate: float = 0.0
    per_category_rate: dict[str, float] = field(default_factory=dict)


def compute_metrics(
    results: list[dict[str, Any]],
    condition_name: str,
    summary_method: str = "structured",
    model_slug: str = "unknown",
) -> CHDMetrics:
    """Compute CHD metrics from a list of judge result dicts.

    Each dict must have a ``"verdict"`` key whose value is one of the five
    CHD category strings.

    Args:
        results: List of result dicts, each with at minimum a ``"verdict"`` key.
        condition_name: Name of the condition being evaluated.
        summary_method: Summarizer method used (for labelling).
        model_slug: Short model identifier for the comparison table.

    Returns:
        A populated :class:`CHDMetrics` instance.

    Raises:
        ValueError: If *results* is empty.
    """
    if not results:
        raise ValueError("results must be non-empty to compute metrics.")

    counts: Counter[str] = Counter()
    for r in results:
        verdict = r.get("verdict", "confabulation").lower()
        if verdict not in _ALL_CATEGORIES:
            verdict = "confabulation"
        counts[verdict] += 1

    total = len(results)
    correct = counts.get("correct", 0)
    accuracy = correct / total
    chd_count = total - correct
    chd_rate = chd_count / total

    per_category_rate = {
        cat: counts.get(cat, 0) / total for cat in _CHD_ERROR_CATEGORIES
    }

    return CHDMetrics(
        condition_name=condition_name,
        summary_method=summary_method,
        total=total,
        model_slug=model_slug,
        counts=dict(counts),
        accuracy=accuracy,
        chd_rate=chd_rate,
        per_category_rate=per_category_rate,
    )


def compute_retrieval_quality(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute retrieval quality metrics from evaluated results.

    Args:
        results: List of result dicts with optional ``"retrieved_turn_ids"``
            and ``"evidence_turn_ids"`` keys.

    Returns:
        Dict with keys ``"precision"``, ``"recall"``, ``"f1"`` as macro-averages.
    """
    precisions: list[float] = []
    recalls: list[float] = []

    for r in results:
        retrieved = set(r.get("turns_loaded", r.get("retrieved_turn_ids", [])))
        evidence = set(r.get("evidence_turn_ids", []))
        if not evidence:
            continue
        if not retrieved:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        tp = len(retrieved & evidence)
        precisions.append(tp / len(retrieved))
        recalls.append(tp / len(evidence))

    if not precisions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def print_comparison_table(metrics_list: list[CHDMetrics]) -> None:
    """Print a Rich table comparing CHD metrics across conditions.

    Args:
        metrics_list: List of :class:`CHDMetrics` objects to display.
    """
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        _print_plain_table(metrics_list)
        return

    console = Console()
    table = Table(
        show_header=True,
        header_style="bold cyan",
        title="CHD Evaluation Results",
    )
    table.add_column("Model", style="dim")
    table.add_column("Condition", style="bold")
    table.add_column("Method")
    table.add_column("N", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("CHD Rate", justify="right")
    table.add_column("Commission", justify="right")
    table.add_column("Omission", justify="right")
    table.add_column("Distortion", justify="right")
    table.add_column("Confab.", justify="right")

    for m in sorted(metrics_list, key=lambda x: (x.model_slug, -x.accuracy, x.condition_name)):
        chd_color = "red" if m.chd_rate > 0.3 else "yellow" if m.chd_rate > 0.1 else "green"
        table.add_row(
            m.model_slug,
            m.condition_name,
            m.summary_method,
            str(m.total),
            f"{m.accuracy:.1%}",
            f"[{chd_color}]{m.chd_rate:.1%}[/{chd_color}]",
            f"{m.per_category_rate.get('commission', 0):.1%}",
            f"{m.per_category_rate.get('omission', 0):.1%}",
            f"{m.per_category_rate.get('distortion', 0):.1%}",
            f"{m.per_category_rate.get('confabulation', 0):.1%}",
        )

    console.print(table)


def load_results_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load judge results from a JSONL file.

    Args:
        path: Path to the results JSONL file.

    Returns:
        List of result dicts.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    results = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                results.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON at line {lineno} of {path}: {exc}") from exc
    return results


def _print_plain_table(metrics_list: list[CHDMetrics]) -> None:
    header = f"{'Condition':<20} {'Method':<12} {'N':>5} {'Acc':>7} {'CHD':>7}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(
            f"{m.condition_name:<20} {m.summary_method:<12} {m.total:>5} "
            f"{m.accuracy:>7.1%} {m.chd_rate:>7.1%}"
        )
