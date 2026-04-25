"""Batch-judge existing prediction JSONL files using a local HF model or Groq.

Reads result files written by evaluate.py / run_sweep.py, runs the CHD judge
on all unjudged rows, and writes verdicts back in-place.

With --num-gpus > 1 and a HuggingFace provider, result files are distributed
across GPU workers so all GPUs judge in parallel — no API rate limits.

Usage
-----
# Local HF judge across all 8 GPUs (recommended on A100 cluster)
python scripts/judge.py \
    --judge-provider hf:Qwen/Qwen2.5-14B-Instruct

# Explicit GPU count
python scripts/judge.py \
    --judge-provider hf:Qwen/Qwen2.5-14B-Instruct \
    --num-gpus 8

# Groq API judge (rate-limited, single process)
python scripts/judge.py \
    --judge-provider groq:llama-3.1-70b-versatile \
    --workers 8

# Judge a single file
python scripts/judge.py \
    --results data/results/llama-3.1-8b/baseline__structured.jsonl \
    --judge-provider hf:Qwen/Qwen2.5-14B-Instruct

# Dry-run: count pending rows without judging
python scripts/judge.py --dry-run

# Re-judge everything, overwrite existing verdicts
python scripts/judge.py --overwrite
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
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

# Groq free-tier: 30 rpm on llama-3.1-70b-versatile — used only when provider is groq:*
_DEFAULT_API_WORKERS = 8

app = typer.Typer(
    name="judge",
    help="Batch-judge prediction JSONL files with a local HF model or Groq.",
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
        typer.Option(
            "--judge-provider",
            help="Provider spec for the judge model (e.g. hf:Qwen/Qwen2.5-14B-Instruct or groq:llama-3.1-70b-versatile).",
        ),
    ] = "hf:Qwen/Qwen2.5-14B-Instruct",
    num_gpus: Annotated[
        int | None,
        typer.Option(
            "--num-gpus",
            help="GPUs to use for HF judge (default: all available). Ignored for groq: providers.",
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", help="Concurrent API workers (groq: providers only, default 8)."),
    ] = _DEFAULT_API_WORKERS,
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Rows per processing batch (groq: providers only)."),
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
    """Fill in CHD verdicts for all prediction rows."""
    # ── Collect files ────────────────────────────────────────────────────
    paths: list[Path] = []
    if results is not None:
        if not results.exists():
            typer.echo(f"ERROR: {results} not found.", err=True)
            raise typer.Exit(1)
        paths = [results]
    else:
        if not results_dir.exists():
            typer.echo(f"ERROR: {results_dir} not found. Run run_sweep.py first.", err=True)
            raise typer.Exit(1)
        paths = sorted(results_dir.glob("**/*.jsonl"))
        if not paths:
            typer.echo("No *.jsonl files found.", err=True)
            raise typer.Exit(1)

    # ── Count pending rows ───────────────────────────────────────────────
    total_pending = 0
    file_stats: list[tuple[Path, int]] = []
    for path in paths:
        rows = _load_jsonl(path)
        pending = sum(1 for r in rows if not skip_judged or r.get("verdict") is None)
        file_stats.append((path, pending))
        total_pending += pending

    is_vllm = judge_provider.lower().startswith("vllm:")
    is_hf = judge_provider.lower().startswith(("hf:", "huggingface:"))
    mode = "vLLM mega-batch" if is_vllm else ("HF multi-GPU" if is_hf else "API")

    typer.echo(
        f"Files: {len(paths)}  |  Pending rows: {total_pending}  |  "
        f"Provider: {judge_provider}  |  Mode: {mode}"
    )

    if dry_run or total_pending == 0:
        if total_pending == 0:
            typer.echo("Nothing to judge.")
        return

    # ── Route to vLLM mega-batch, HF parallel, or API sequential ─────────
    pending_paths = [p for p, n in file_stats if n > 0]

    if is_vllm:
        _run_vllm_mega_batch(
            paths=pending_paths,
            judge_provider_spec=judge_provider,
            skip_judged=skip_judged,
        )
    elif is_hf:
        _run_hf_parallel(
            paths=pending_paths,
            judge_provider_spec=judge_provider,
            num_gpus=num_gpus,
            skip_judged=skip_judged,
        )
    else:
        _run_api_sequential(
            paths=pending_paths,
            judge_provider_spec=judge_provider,
            workers=workers,
            chunk_size=chunk_size,
            skip_judged=skip_judged,
            results_dir=results_dir,
        )

    typer.echo(f"\nDone. {total_pending} verdicts written.")


# ---------------------------------------------------------------------------
# vLLM mega-batch path
# ---------------------------------------------------------------------------

def _run_vllm_mega_batch(
    paths: list[Path],
    judge_provider_spec: str,
    skip_judged: bool,
) -> None:
    """Load vLLM once, collect every pending row, judge in a single generate_batch call.

    Args:
        paths: Result JSONL files with pending rows.
        judge_provider_spec: vLLM provider spec, e.g. ``vllm:Qwen/...:tp8``.
        skip_judged: Whether to skip rows that already have a verdict.
    """
    from episodic_log.providers import get_provider

    provider = get_provider(judge_provider_spec)
    judge = CHDJudge(provider=provider)

    # Phase 1: load all files and collect pending (file_idx, row_idx) pairs.
    all_files: list[tuple[Path, list[dict]]] = []
    pending_jobs: list[tuple[int, int]] = []  # (file_idx, row_idx)

    for file_idx, path in enumerate(paths):
        rows = _load_jsonl(path)
        all_files.append((path, rows))
        for row_idx, row in enumerate(rows):
            if skip_judged and row.get("verdict") is not None:
                continue
            pending_jobs.append((file_idx, row_idx))

    if not pending_jobs:
        logger.info("vLLM judge: nothing pending.")
        return

    logger.info("vLLM mega-batch judge: %d rows across %d files", len(pending_jobs), len(paths))

    # Phase 2: build inputs list and call judge_batch_fast (single generate_batch).
    inputs = [
        {
            "question":      all_files[fi][1][ri]["question"],
            "ground_truth":  all_files[fi][1][ri]["ground_truth"],
            "predicted":     all_files[fi][1][ri]["predicted_answer"],
            "context_turns": "",
        }
        for fi, ri in pending_jobs
    ]

    bar = tqdm(total=len(inputs), desc="vLLM judging", unit="row", dynamic_ncols=True)
    verdicts = judge.judge_batch_fast(inputs)
    bar.update(len(verdicts))
    bar.close()

    # Phase 3: assign verdicts and write files.
    for (fi, ri), verdict in zip(pending_jobs, verdicts):
        row = all_files[fi][1][ri]
        row["verdict"] = verdict.verdict
        row["confidence"] = verdict.confidence
        row["judge_reason"] = verdict.reason

    for path, rows in all_files:
        _write_jsonl(path, rows)
        logger.info("Wrote verdicts → %s", path)


# ---------------------------------------------------------------------------
# HF multi-GPU path
# ---------------------------------------------------------------------------

def _run_hf_parallel(
    paths: list[Path],
    judge_provider_spec: str,
    num_gpus: int | None,
    skip_judged: bool,
) -> None:
    """Distribute result files across GPU workers and judge in parallel.

    Args:
        paths: Result JSONL files with pending rows.
        judge_provider_spec: HuggingFace provider spec string.
        num_gpus: Number of GPUs to use. Auto-detected if ``None``.
        skip_judged: Whether to skip rows that already have a verdict.
    """
    if num_gpus is None:
        try:
            import torch
            num_gpus = max(1, torch.cuda.device_count())
        except ImportError:
            num_gpus = 1

    typer.echo(f"HF judge — distributing {len(paths)} files across {num_gpus} GPU(s)")

    # Assign files to GPUs round-robin.
    gpu_files: dict[int, list[str]] = {i: [] for i in range(num_gpus)}
    for idx, path in enumerate(paths):
        gpu_files[idx % num_gpus].append(str(path))

    if num_gpus == 1:
        _judge_gpu_worker(
            gpu_id=0,
            judge_provider_spec=judge_provider_spec,
            file_paths=gpu_files[0],
            skip_judged=skip_judged,
        )
        return

    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for gpu_id, file_paths in gpu_files.items():
        if not file_paths:
            continue
        p = ctx.Process(
            target=_judge_gpu_worker,
            args=(gpu_id, judge_provider_spec, file_paths, skip_judged),
            name=f"judge-gpu{gpu_id}",
        )
        p.start()
        processes.append(p)
        typer.echo(f"  Launched judge worker GPU {gpu_id} — {len(file_paths)} file(s)")

    for p in processes:
        p.join()
        if p.exitcode != 0:
            logger.error("Judge worker %s exited with code %d", p.name, p.exitcode)


def _judge_gpu_worker(
    gpu_id: int,
    judge_provider_spec: str,
    file_paths: list[str],
    skip_judged: bool,
) -> None:
    """Load a local judge model and process assigned result files sequentially.

    Intended to run as a child process via ``multiprocessing.spawn``.

    Args:
        gpu_id: Physical GPU index to bind to.
        judge_provider_spec: HuggingFace provider spec string.
        file_paths: Absolute path strings of result JSONL files to judge.
        skip_judged: Whether to skip rows that already have a verdict.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [GPU{gpu_id}] %(name)s: %(message)s",
    )
    worker_logger = logging.getLogger(__name__)

    from episodic_log.providers import get_provider

    worker_logger.info("Loading judge model %s on GPU %d", judge_provider_spec, gpu_id)
    provider = get_provider(judge_provider_spec, device_map="cuda:0")
    judge = CHDJudge(provider=provider)

    for path_str in file_paths:
        path = Path(path_str)
        rows = _load_jsonl(path)
        pending = [i for i, r in enumerate(rows) if not skip_judged or r.get("verdict") is None]

        if not pending:
            continue

        worker_logger.info("Judging %s — %d rows", path.name, len(pending))
        bar = tqdm(pending, desc=path.name, unit="row", dynamic_ncols=True, leave=False)
        for idx in bar:
            try:
                verdict = judge.judge(
                    question=rows[idx]["question"],
                    ground_truth=rows[idx]["ground_truth"],
                    predicted=rows[idx]["predicted_answer"],
                )
                rows[idx]["verdict"] = verdict.verdict
                rows[idx]["confidence"] = verdict.confidence
                rows[idx]["judge_reason"] = verdict.reason
            except Exception as exc:
                worker_logger.error("Row %d in %s failed: %s", idx, path.name, exc)

        _write_jsonl(path, rows)
        worker_logger.info("Wrote verdicts → %s", path)


# ---------------------------------------------------------------------------
# Groq / API path (rate-limited, uses ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _run_api_sequential(
    paths: list[Path],
    judge_provider_spec: str,
    workers: int,
    chunk_size: int,
    skip_judged: bool,
    results_dir: Path,
) -> None:
    """Judge result files using a remote API provider with batched parallel requests.

    Args:
        paths: Result JSONL files with pending rows.
        judge_provider_spec: Provider spec (e.g. ``groq:llama-3.1-70b-versatile``).
        workers: Maximum concurrent API requests.
        chunk_size: Rows per ``ThreadPoolExecutor`` batch.
        skip_judged: Whether to skip rows that already have a verdict.
        results_dir: Root results directory (used for display only).
    """
    from episodic_log.providers import get_provider

    provider = get_provider(judge_provider_spec)
    judge = CHDJudge(provider=provider)

    all_pending = sum(
        sum(1 for r in _load_jsonl(p) if not skip_judged or r.get("verdict") is None)
        for p in paths
    )
    overall_bar = tqdm(total=all_pending, desc="Judging", unit="row", dynamic_ncols=True)

    for path in paths:
        rows = _load_jsonl(path)
        pending_indices = [i for i, r in enumerate(rows) if not skip_judged or r.get("verdict") is None]
        if not pending_indices:
            continue

        rel = path.relative_to(results_dir) if results_dir in path.parents else path.name
        typer.echo(f"\n  {rel}  ({len(pending_indices)} rows)")

        for chunk_start in range(0, len(pending_indices), chunk_size):
            chunk_indices = pending_indices[chunk_start: chunk_start + chunk_size]
            inputs = [
                {
                    "question":     rows[i]["question"],
                    "ground_truth": rows[i]["ground_truth"],
                    "predicted":    rows[i]["predicted_answer"],
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
