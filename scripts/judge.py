"""Batch-judge existing prediction JSONL files with a Groq judge model.

Reads result files written by evaluate.py / run_sweep.py, sends all
predictions to the Groq judge in parallel, and writes verdicts back
in-place (or to a new file).

This separates inference (GPU, slow) from judging (Groq API, fast) so
you can re-judge with a different model without re-running inference.

Usage
-----
# Judge all result files under data/results/ (in-place)
python scripts/judge.py \
    --judge-provider groq:llama-3.1-70b-versatile

# Judge a single file
python scripts/judge.py \
    --results data/results/llama-3.1-8b/baseline__structured.jsonl \
    --judge-provider groq:llama-3.1-70b-versatile

# Judge with higher concurrency (be careful with rate limits)
python scripts/judge.py --workers 15

# Only judge rows that don't already have a verdict (default)
python scripts/judge.py --skip-judged

# Re-judge everything, overwrite existing verdicts
python scripts/judge.py --overwrite
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episodic_log.judge import CHDJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Groq free-tier limits (requests per minute):
#   llama-3.1-70b-versatile : 30 rpm
#   llama-3.1-8b-instant    : 30 rpm
# We default to 8 workers which stays safely within limits while
# being ~8× faster than sequential calls.
_DEFAULT_WORKERS = 8

app = typer.Typer(
    name="judge",
    help="Batch-judge prediction JSONL files with a Groq judge model.",
    add_completion=False,
)


@app.command()
def judge_cmd(
    results: Annotated[
        Path | None,
        typer.Option("--results", help="Path to a single results JSONL file. Omit to scan results-dir."),
    ] = None,
    results_dir: Annotated[
        Path,
        typer.Option("--results-dir", help="Directory to scan recursively for *.jsonl result files."),
    ] = Path("data/results"),
    judge_provider: Annotated[
        str,
        typer.Option("--judge-provider", help="Groq (or other) provider spec for the judge."),
    ] = "groq:llama-3.1-70b-versatile",
    workers: Annotated[
        int,
        typer.Option("--workers", help="Concurrent judge requests (default 8, Groq 70B limit: 30 rpm)."),
    ] = _DEFAULT_WORKERS,
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Process this many rows per ThreadPoolExecutor batch."),
    ] = 50,
    skip_judged: Annotated[
        bool,
        typer.Option("--skip-judged/--overwrite", help="Skip rows that already have a verdict (default: skip)."),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Count pending rows without calling the judge."),
    ] = False,
) -> None:
    """Fill in verdicts for all prediction rows using a parallel Groq judge."""
    # ── Collect files ────────────────────────────────────────────────────
    paths: list[Path] = []
    if results is not None:
        if not results.exists():
            typer.echo(f"ERROR: {results} not found.", err=True)
            raise typer.Exit(1)
        paths = [results]
    else:
        if not results_dir.exists():
            typer.echo(f"ERROR: {results_dir} not found. Run evaluate.py or run_sweep.py first.", err=True)
            raise typer.Exit(1)
        paths = sorted(results_dir.glob("**/*.jsonl"))
        if not paths:
            typer.echo("No *.jsonl files found.", err=True)
            raise typer.Exit(1)

    # ── Count pending rows ───────────────────────────────────────────────
    total_pending = 0
    file_stats: list[tuple[Path, list[dict], list[int]]] = []
    for path in paths:
        rows = _load_jsonl(path)
        pending_indices = [
            i for i, r in enumerate(rows)
            if not skip_judged or r.get("verdict") is None
        ]
        file_stats.append((path, rows, pending_indices))
        total_pending += len(pending_indices)

    typer.echo(
        f"Files: {len(paths)}  |  Total pending rows: {total_pending}  |  "
        f"Workers: {workers}  |  Judge: {judge_provider}"
    )
    if dry_run or total_pending == 0:
        if total_pending == 0:
            typer.echo("Nothing to judge.")
        return

    # ── Build judge ──────────────────────────────────────────────────────
    from episodic_log.providers import get_provider
    provider = get_provider(judge_provider)
    judge = CHDJudge(provider=provider)

    # ── Judge each file ──────────────────────────────────────────────────
    overall_bar = tqdm(
        total=total_pending,
        desc="Judging",
        unit="row",
        dynamic_ncols=True,
    )

    for path, rows, pending_indices in file_stats:
        if not pending_indices:
            continue

        typer.echo(f"\n  {path.relative_to(results_dir) if results_dir in path.parents else path.name}"
                   f"  ({len(pending_indices)} rows)")

        # Process in chunks to keep tqdm responsive and avoid one huge batch.
        for chunk_start in range(0, len(pending_indices), chunk_size):
            chunk_indices = pending_indices[chunk_start: chunk_start + chunk_size]
            inputs = [
                {
                    "question": rows[i]["question"],
                    "ground_truth": rows[i]["ground_truth"],
                    "predicted": rows[i]["predicted_answer"],
                }
                for i in chunk_indices
            ]

            verdicts = judge.judge_batch(inputs, max_workers=workers)

            for idx, verdict in zip(chunk_indices, verdicts):
                rows[idx]["verdict"] = verdict.verdict
                rows[idx]["confidence"] = verdict.confidence
                rows[idx]["judge_reason"] = verdict.reason

            overall_bar.update(len(chunk_indices))

        _write_jsonl(path, rows)

    overall_bar.close()
    typer.echo(f"\nDone. {total_pending} verdicts written.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{lineno}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app()
