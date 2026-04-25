"""Summarize ingested session logs using one or more summarizer methods.

Sessions are split evenly across all available GPUs so all 8 A100s work
simultaneously — 8× faster than single-GPU sequential.  Any HF model size
is supported.  For large BF16 models (32B+), use ``--gpus-per-worker`` so
each worker spans multiple cards via ``device_map="auto"``.

Usage
-----
# Lexical (no model, CPU, fast)
python scripts/summarize.py --method lexical

# Scout — small model, all 8 GPUs in parallel (1 GPU each)
python scripts/summarize.py --method scout \
    --provider hf:Qwen/Qwen2.5-7B-Instruct

# Scout — 32B BF16, 8 workers × 1 GPU each (32 GB fits in 80 GB A100)
python scripts/summarize.py --method scout \
    --provider hf:Qwen/Qwen2.5-32B-Instruct

# Echo summarizer with 70B BF16, 4 workers × 2 GPUs each
python scripts/summarize.py --method echo \
    --provider hf:meta-llama/Llama-3.3-70B-Instruct \
    --gpus-per-worker 2

# Explicit total GPU count
python scripts/summarize.py --method scout \
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
def summarize(  # noqa: PLR0912
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="Summarizer method: lexical | scout | echo.",
            case_sensitive=False,
        ),
    ] = "lexical",
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
            help="Total GPUs to use (default: all available). Ignored for structured.",
        ),
    ] = None,
    gpus_per_worker: Annotated[
        int,
        typer.Option(
            "--gpus-per-worker",
            help=(
                "GPUs allocated to each worker process (default: 1). "
                "Use 2 for 70B/72B BF16 models that need ~140 GB VRAM."
            ),
        ),
    ] = 1,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Overwrite existing summary files."),
    ] = False,
) -> None:
    """Generate TurnSummary JSONL files for every session in the index.

    Args:
        method: Summarizer method: ``lexical`` | ``scout`` | ``echo``.
        sessions_index: Path to the sessions index JSONL produced by ingest.py.
        provider_spec: Provider spec string for model-backed methods.
        num_gpus: Total GPUs to use (auto-detected if not set).
        gpus_per_worker: GPUs per worker process; determines parallelism as
            ``num_workers = num_gpus // gpus_per_worker``.
        overwrite: Whether to overwrite existing summary files.
    """
    if not sessions_index.exists():
        typer.echo(f"ERROR: sessions index not found at {sessions_index}. Run ingest.py first.", err=True)
        raise typer.Exit(1)

    all_sessions = _load_index(sessions_index)

    # Filter out sessions that already have a valid (non-empty) summary file.
    # Zero-byte or zero-line files are treated as absent so crashed runs don't
    # permanently block re-summarization without --overwrite.
    def _summary_is_valid(s: dict) -> bool:
        p = Path(s["summaries_dir"]) / f"{method}.jsonl"
        return p.exists() and p.stat().st_size > 0 and p.read_text(encoding="utf-8").strip() != ""

    if not overwrite:
        pending = [s for s in all_sessions if not _summary_is_valid(s)]
    else:
        pending = all_sessions

    typer.echo(
        f"Sessions: {len(all_sessions)} total  |  {len(pending)} pending  |  method={method!r}"
    )
    if not pending:
        typer.echo("Nothing to do.")
        return

    # Lexical is CPU-only — no GPU, no parallelization needed.
    if method == "lexical":
        _run_sessions(
            cuda_devices=None,
            sessions=pending,
            method=method,
            provider_spec=None,
            overwrite=overwrite,
        )
        typer.echo(f"Done. {len(pending)} sessions summarized.")
        return

    # Model-backed methods: parallelize across GPUs.
    if provider_spec is None:
        typer.echo("ERROR: --provider is required for scout/echo methods.", err=True)
        raise typer.Exit(1)

    # vLLM path: single-process mega-batch — skip multiprocessing entirely.
    # vLLM manages GPU parallelism internally via tensor parallelism.
    if provider_spec.startswith("vllm:"):
        typer.echo(f"Provider: {provider_spec}  |  mode=vllm-mega-batch")
        _run_vllm_sessions(
            sessions=pending,
            method=method,
            provider_spec=provider_spec,
            overwrite=overwrite,
        )
        typer.echo(f"Done. {len(pending)} sessions summarized.")
        return

    if num_gpus is None:
        try:
            import torch
            num_gpus = max(1, torch.cuda.device_count())
        except ImportError:
            num_gpus = 1

    num_workers = max(1, num_gpus // gpus_per_worker)
    typer.echo(
        f"Provider: {provider_spec}  |  GPUs: {num_gpus}  |  "
        f"Workers: {num_workers}  |  GPUs/worker: {gpus_per_worker}"
    )

    # Build comma-separated CUDA_VISIBLE_DEVICES string per worker.
    all_gpu_ids = list(range(num_gpus))
    worker_devices: list[str] = [
        ",".join(str(g) for g in all_gpu_ids[i * gpus_per_worker:(i + 1) * gpus_per_worker])
        for i in range(num_workers)
    ]

    if num_workers == 1:
        _run_sessions(
            cuda_devices=worker_devices[0],
            sessions=pending,
            method=method,
            provider_spec=provider_spec,
            overwrite=overwrite,
        )
    else:
        # Split sessions evenly across workers.
        chunks = _split_sessions(pending, num_workers)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []
        for worker_idx, (cuda_devs, chunk) in enumerate(zip(worker_devices, chunks)):
            if not chunk:
                continue
            p = ctx.Process(
                target=_run_sessions,
                args=(cuda_devs, chunk, method, provider_spec, overwrite),
                name=f"summarize-worker{worker_idx}",
            )
            p.start()
            processes.append(p)
            typer.echo(f"  Worker {worker_idx} (GPUs {cuda_devs}): {len(chunk)} sessions")

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error("Worker %s exited with code %d", p.name, p.exitcode)

    typer.echo(f"Done. {len(pending)} sessions summarized.")


# ---------------------------------------------------------------------------
# Worker (runs in child process for model-backed methods)
# ---------------------------------------------------------------------------

def _run_sessions(
    cuda_devices: str | None,
    sessions: list[dict],
    method: str,
    provider_spec: str | None,
    overwrite: bool,
) -> None:
    """Summarize a subset of sessions, optionally bound to one or more GPUs.

    Args:
        cuda_devices: Comma-separated GPU indices to bind via
            ``CUDA_VISIBLE_DEVICES`` (e.g. ``"0"`` or ``"6,7"``), or
            ``None`` for CPU-only methods.
        sessions: Session metadata records to process.
        method: Summarizer method name.
        provider_spec: Provider spec string, or ``None`` for lexical.
        overwrite: Whether to overwrite existing summary files.
    """
    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s %(levelname)s [GPU{cuda_devices}] %(name)s: %(message)s",
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
        device_map = "auto" if cuda_devices is not None else "cpu"
        provider = get_provider(provider_spec, device_map=device_map)

    summarizer = build_summarizer(method=method, provider=provider)

    tag = f"GPU{cuda_devices}" if cuda_devices is not None else "CPU"
    bar = tqdm(sessions, desc=f"{tag}/{method}", unit="session", dynamic_ncols=True, leave=True)

    n_skipped = 0
    n_written = 0
    n_empty_log = 0

    for session_meta in bar:
        summaries_dir = Path(session_meta["summaries_dir"])
        summary_file  = summaries_dir / f"{method}.jsonl"

        # Treat zero-byte or zero-line files as absent — a crashed previous run
        # can leave empty placeholder files that would otherwise block re-runs.
        file_is_valid = (
            summary_file.exists()
            and summary_file.stat().st_size > 0
            and summary_file.read_text(encoding="utf-8").strip()
        )
        if file_is_valid and not overwrite:
            n_skipped += 1
            continue

        log_path = Path(session_meta["log_path"])
        if not log_path.exists():
            worker_logger.warning("log.jsonl missing for %s — skipping.", session_meta["session_id"])
            continue

        events = LogReader(log_path).load_all()
        if not events:
            worker_logger.warning(
                "Session %s has 0 events in log.jsonl — no summaries written.",
                session_meta["session_id"],
            )
            n_empty_log += 1
            continue

        if summary_file.exists():
            summary_file.unlink()

        store = SummaryStore(summaries_dir)
        failed_events = 0
        try:
            summaries = summarizer.summarize_batch(events)
        except Exception as exc:
            worker_logger.error(
                "summarize_batch failed: session=%s method=%s: %s",
                session_meta["session_id"], method, exc,
                exc_info=True,
            )
            summaries = []
            failed_events = len(events)

        if failed_events == 0:
            for event, summary in zip(events, summaries):
                try:
                    store.write(summary)
                except Exception as exc:
                    worker_logger.error(
                        "Summarize error: session=%s turn=%s method=%s: %s",
                        session_meta["session_id"], event.turn_id, method, exc,
                        exc_info=True,
                    )
                    failed_events += 1

        if failed_events == len(events):
            if summary_file.exists():
                summary_file.unlink()
            worker_logger.error(
                "ALL %d events failed for session %s — deleted empty %s.jsonl",
                len(events), session_meta["session_id"], method,
            )
        else:
            n_written += 1
            if failed_events:
                worker_logger.warning(
                    "Session %s: %d/%d events failed summarization.",
                    session_meta["session_id"], failed_events, len(events),
                )

        bar.set_postfix({"written": n_written, "skipped": n_skipped, "failed_ev": failed_events})

    worker_logger.info(
        "%s: done — written=%d  skipped(valid)=%d  empty_log=%d  total=%d",
        tag, n_written, n_skipped, n_empty_log, len(sessions),
    )


# ---------------------------------------------------------------------------
# vLLM mega-batch path
# ---------------------------------------------------------------------------

def _run_vllm_sessions(
    sessions: list[dict],
    method: str,
    provider_spec: str,
    overwrite: bool,
) -> None:
    """Single-process vLLM path: collect all events → one LLM.generate() call.

    vLLM's continuous batching scheduler maximally utilises GPU bandwidth —
    no Python multiprocessing needed when using tensor parallelism.

    Args:
        sessions: Session metadata records to process.
        method: Summarizer method name (``"scout"`` or ``"echo"``).
        provider_spec: vLLM provider spec, e.g. ``"vllm:Qwen/...:tp8"``.
        overwrite: Whether to overwrite existing summary files.
    """
    from episodic_log.providers import get_provider
    from episodic_log.retrieval.summary_store import SummaryStore
    from episodic_log.summarizers import build_summarizer

    provider = get_provider(provider_spec)
    summarizer = build_summarizer(method=method, provider=provider)

    # Phase 1: load all events from all pending sessions.
    all_session_data: list[tuple[dict, list]] = []
    for session_meta in tqdm(sessions, desc="Loading logs", unit="session"):
        summaries_dir = Path(session_meta["summaries_dir"])
        summary_file = summaries_dir / f"{method}.jsonl"
        if not overwrite and summary_file.exists() and summary_file.stat().st_size > 0:
            continue
        log_path = Path(session_meta["log_path"])
        if not log_path.exists():
            logger.warning("log.jsonl missing for %s — skipping.", session_meta["session_id"])
            continue
        events = LogReader(log_path).load_all()
        if not events:
            logger.warning("Session %s has 0 events — skipping.", session_meta["session_id"])
            continue
        all_session_data.append((session_meta, events))

    if not all_session_data:
        logger.info("vLLM: nothing to summarize.")
        return

    total_events = sum(len(evs) for _, evs in all_session_data)
    logger.info(
        "vLLM mega-batch: %d sessions, %d total events → single LLM.generate() call",
        len(all_session_data), total_events,
    )

    # Phase 2: flatten all events and generate all summaries in one shot.
    all_events_flat = [e for _, evs in all_session_data for e in evs]
    # Pass batch_size=total_events so vLLM receives all prompts at once.
    all_summaries = summarizer.summarize_batch(all_events_flat, batch_size=total_events)

    # Phase 3: write summaries back per session.
    n_written = 0
    idx = 0
    for session_meta, events in tqdm(all_session_data, desc="Writing summaries", unit="session"):
        n = len(events)
        session_summaries = all_summaries[idx: idx + n]
        idx += n

        summaries_dir = Path(session_meta["summaries_dir"])
        summary_file = summaries_dir / f"{method}.jsonl"
        if summary_file.exists():
            summary_file.unlink()

        store = SummaryStore(summaries_dir)
        for summary in session_summaries:
            try:
                store.write(summary)
            except Exception as exc:
                logger.error(
                    "Write error: session=%s method=%s: %s",
                    session_meta["session_id"], method, exc,
                )
        n_written += 1

    logger.info("vLLM: done — written=%d sessions, %d total summaries", n_written, total_events)


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
