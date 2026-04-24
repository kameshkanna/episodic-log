"""Summarize ingested session logs using one or more summarizer methods.

Sessions are split evenly across all available GPUs so all 8 A100s work
simultaneously — 8× faster than single-GPU sequential.  Any HF model size
is supported, including large quantized models for higher-quality summaries.

Usage
-----
# Structured (no model, CPU, fast)
python scripts/summarize.py --method structured

# Haiku — small model, all 8 GPUs in parallel
python scripts/summarize.py --method haiku \
    --provider hf:Qwen/Qwen2.5-7B-Instruct

# Haiku — larger model for better summaries, 4-bit so it fits
python scripts/summarize.py --method haiku \
    --provider hf:Qwen/Qwen2.5-32B-Instruct:4bit

# Self-summarizer with 70B
python scripts/summarize.py --method self \
    --provider hf:meta-llama/Llama-3.3-70B-Instruct:4bit

# Explicit GPU count
python scripts/summarize.py --method haiku \
    --provider hf:Qwen/Qwen2.5-7B-Instruct \
    --num-gpus 4
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

from episodic_log.core.log_reader import LogReader
from episodic_log.retrieval.summary_store import SummaryStore
from episodic_log.summarizers import build_summarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="summarize",
    help="Summarize session logs into per-method JSONL summary files.",
    add_completion=False,
)


@app.command()
def summarize(
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="Summarizer method: structured | haiku | self.",
            case_sensitive=False,
        ),
    ] = "structured",
    sessions_index: Annotated[
        Path,
        typer.Option(help="Path to sessions_index.jsonl produced by ingest.py."),
    ] = Path("data/sessions_index.jsonl"),
    provider_spec: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help=(
                "Provider spec for haiku/self (e.g. hf:Qwen/Qwen2.5-7B-Instruct or "
                "hf:Qwen/Qwen2.5-32B-Instruct:4bit). Any size supported."
            ),
        ),
    ] = None,
    num_gpus: Annotated[
        int | None,
        typer.Option(
            "--num-gpus",
            help="GPUs to use in parallel (default: all available). Ignored for structured.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Overwrite existing summary files."),
    ] = False,
) -> None:
    """Generate TurnSummary JSONL files for every session in the index."""
    if not sessions_index.exists():
        typer.echo(f"ERROR: sessions index not found at {sessions_index}. Run ingest.py first.", err=True)
        raise typer.Exit(1)

    all_sessions = _load_index(sessions_index)

    # Filter out already-summarized sessions unless overwriting.
    if not overwrite:
        pending = [
            s for s in all_sessions
            if not (Path(s["summaries_dir"]) / f"{method}.jsonl").exists()
        ]
    else:
        pending = all_sessions

    typer.echo(
        f"Sessions: {len(all_sessions)} total  |  {len(pending)} pending  |  method={method!r}"
    )
    if not pending:
        typer.echo("Nothing to do.")
        return

    # Structured is CPU-only — no GPU, no parallelization needed.
    if method == "structured":
        _run_sessions(
            gpu_id=None,
            sessions=pending,
            method=method,
            provider_spec=None,
            overwrite=overwrite,
        )
        typer.echo(f"Done. {len(pending)} sessions summarized.")
        return

    # Model-backed methods: parallelize across GPUs.
    if provider_spec is None:
        typer.echo("ERROR: --provider is required for haiku/self methods.", err=True)
        raise typer.Exit(1)

    if num_gpus is None:
        try:
            import torch
            num_gpus = max(1, torch.cuda.device_count())
        except ImportError:
            num_gpus = 1

    typer.echo(f"Provider: {provider_spec}  |  GPUs: {num_gpus}")

    if num_gpus == 1:
        _run_sessions(
            gpu_id=0,
            sessions=pending,
            method=method,
            provider_spec=provider_spec,
            overwrite=overwrite,
        )
    else:
        # Split sessions evenly across GPUs.
        chunks = _split_sessions(pending, num_gpus)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []
        for gpu_id, chunk in enumerate(chunks):
            if not chunk:
                continue
            p = ctx.Process(
                target=_run_sessions,
                args=(gpu_id, chunk, method, provider_spec, overwrite),
                name=f"summarize-gpu{gpu_id}",
            )
            p.start()
            processes.append(p)
            typer.echo(f"  GPU {gpu_id}: {len(chunk)} sessions")

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error("Worker %s exited with code %d", p.name, p.exitcode)

    typer.echo(f"Done. {len(pending)} sessions summarized.")


# ---------------------------------------------------------------------------
# Worker (runs in child process for model-backed methods)
# ---------------------------------------------------------------------------

def _run_sessions(
    gpu_id: int | None,
    sessions: list[dict],
    method: str,
    provider_spec: str | None,
    overwrite: bool,
) -> None:
    """Summarize a subset of sessions, optionally bound to one GPU.

    Args:
        gpu_id: GPU index to bind via ``CUDA_VISIBLE_DEVICES``, or ``None``
            for CPU-only methods.
        sessions: Session metadata records to process.
        method: Summarizer method name.
        provider_spec: Provider spec string, or ``None`` for structured.
        overwrite: Whether to overwrite existing summary files.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s %(levelname)s [GPU{gpu_id}] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    worker_logger = logging.getLogger(__name__)

    provider = None
    if provider_spec:
        from episodic_log.providers import get_provider
        provider = get_provider(provider_spec, device_map="cuda:0" if gpu_id is not None else "cpu")

    summarizer = build_summarizer(method=method, provider=provider)

    tag = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
    bar = tqdm(sessions, desc=f"{tag}/{method}", unit="session", dynamic_ncols=True, leave=True)

    for session_meta in bar:
        summaries_dir = Path(session_meta["summaries_dir"])
        summary_file  = summaries_dir / f"{method}.jsonl"

        if summary_file.exists() and not overwrite:
            continue

        log_path = Path(session_meta["log_path"])
        if not log_path.exists():
            worker_logger.warning("log.jsonl missing for %s — skipping.", session_meta["session_id"])
            continue

        events = LogReader(log_path).load_all()

        if summary_file.exists():
            summary_file.unlink()

        store = SummaryStore(summaries_dir)
        for event in events:
            summary = summarizer.summarize(event)
            store.write(summary)

        bar.set_postfix({"session": session_meta["session_id"][:16]})

    worker_logger.info("%s: finished %d sessions", tag, len(sessions))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sessions(sessions: list[dict], n: int) -> list[list[dict]]:
    """Split *sessions* into *n* roughly equal chunks.

    Args:
        sessions: Full list of session metadata records.
        n: Number of chunks (one per GPU).

    Returns:
        List of *n* sublists; later chunks may be one element shorter.
    """
    chunk_size = (len(sessions) + n - 1) // n
    return [sessions[i: i + chunk_size] for i in range(0, len(sessions), chunk_size)]


def _load_index(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


if __name__ == "__main__":
    app()
