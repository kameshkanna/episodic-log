"""Compute and display CHD metrics from evaluate.py results.

Usage
-----
# Score all result files in data/results/
python scripts/score.py

# Score a specific file
python scripts/score.py --results data/results/episodic__structured.jsonl

# Compare multiple conditions
python scripts/score.py --results-dir data/results/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episodic_log.metrics import (
    compute_metrics,
    compute_retrieval_quality,
    load_results_jsonl,
    print_comparison_table,
    CHDMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="score",
    help="Compute and print CHD metrics from evaluate.py results.",
    add_completion=False,
)


@app.command()
def score(
    results: Annotated[
        Path | None,
        typer.Option("--results", help="Path to a single results JSONL file."),
    ] = None,
    results_dir: Annotated[
        Path,
        typer.Option("--results-dir", help="Directory to scan for *.jsonl result files."),
    ] = Path("data/results"),
    show_retrieval: Annotated[
        bool,
        typer.Option("--retrieval/--no-retrieval", help="Show retrieval quality metrics."),
    ] = False,
    breakdown: Annotated[
        bool,
        typer.Option("--breakdown/--no-breakdown", help="Show per-question-type breakdown."),
    ] = False,
) -> None:
    """Load results JSONL file(s) and print CHD metrics table."""
    paths: list[Path] = []

    if results is not None:
        if not results.exists():
            typer.echo(f"ERROR: File not found: {results}", err=True)
            raise typer.Exit(1)
        paths = [results]
    else:
        if not results_dir.exists():
            typer.echo(
                f"ERROR: Results directory not found: {results_dir}. Run evaluate.py first.",
                err=True,
            )
            raise typer.Exit(1)
        # Support both flat layout (results/*.jsonl) and per-model layout (results/<slug>/*.jsonl).
        paths = sorted(results_dir.glob("**/*.jsonl"))
        if not paths:
            typer.echo(f"No *.jsonl files found in {results_dir} (searched recursively).", err=True)
            raise typer.Exit(1)

    all_metrics: list[CHDMetrics] = []

    for path in paths:
        try:
            data = load_results_jsonl(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to load %s: %s", path, exc)
            continue

        # Derive condition, method, and model slug from path.
        # Layout: data/results/<model-slug>/<condition>__<method>.jsonl
        stem = path.stem
        if "__" in stem:
            condition_name, summary_method = stem.split("__", 1)
        else:
            condition_name, summary_method = stem, "-"
        # If file is inside a subdirectory, treat that directory name as model slug.
        model_slug = path.parent.name if path.parent != results_dir else "unknown"

        # Only compute verdict-based metrics if judge has been run.
        judged = [r for r in data if r.get("verdict") is not None]
        if judged:
            metrics = compute_metrics(
                results=judged,
                condition_name=condition_name,
                summary_method=summary_method,
                model_slug=model_slug,
            )
        else:
            logger.warning(
                "%s: no judge verdicts found — metrics will be empty. Run evaluate.py --judge.",
                path.name,
            )
            metrics = CHDMetrics(
                condition_name=condition_name,
                summary_method=summary_method,
                total=len(data),
                model_slug=model_slug,
            )

        all_metrics.append(metrics)

        if show_retrieval:
            rq = compute_retrieval_quality(data)
            typer.echo(
                f"{condition_name}/{summary_method} retrieval: "
                f"P={rq['precision']:.3f}  R={rq['recall']:.3f}  F1={rq['f1']:.3f}"
            )

        if breakdown:
            _print_type_breakdown(data, condition_name, summary_method)

    if all_metrics:
        print_comparison_table(all_metrics)
    else:
        typer.echo("No metrics to display.")


def _print_type_breakdown(
    data: list[dict],
    condition_name: str,
    summary_method: str,
) -> None:
    """Print accuracy by question_type for one condition."""
    from collections import defaultdict

    by_type: dict[str, list[str]] = defaultdict(list)
    for r in data:
        qt = r.get("question_type", "unknown")
        v = r.get("verdict")
        if v:
            by_type[qt].append(v)

    if not by_type:
        return

    typer.echo(f"\n{condition_name}/{summary_method} — per question type:")
    for qt, verdicts in sorted(by_type.items()):
        correct = sum(1 for v in verdicts if v == "correct")
        acc = correct / len(verdicts) if verdicts else 0.0
        typer.echo(f"  {qt:<35} acc={acc:.1%}  n={len(verdicts)}")


if __name__ == "__main__":
    app()
