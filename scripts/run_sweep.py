"""Multi-model sweep — queue-based GPU scheduler.

Each model runs on exactly one GPU.  All 9 models fit on a single A100 80 GB
(largest is 72B at 4-bit ≈ 36 GB).  With 8 GPUs the first 8 models start
immediately and the 9th slots in as soon as the fastest GPU finishes.

No phases.  No idle GPUs.  No spreading one model across all 8 cards.

Results are written to:
    ``data/results/<model-slug>/<condition>__<method>.jsonl``

Recommended workflow (8× A100 80 GB)
--------------------------------------
# Step 1 — Full sweep (all 9 models × 7 conditions × 3 methods)
python scripts/run_sweep.py --summary-methods structured,haiku,self

# Step 2 — Local HF judge (8 GPUs in parallel, no API rate limits)
python scripts/judge.py --judge-provider hf:Qwen/Qwen2.5-14B-Instruct

# Step 3 — Score
python scripts/score.py --breakdown

Other examples
--------------
python scripts/run_sweep.py --dry-run
python scripts/run_sweep.py --size-filter small --conditions baseline,episodic
python scripts/run_sweep.py --model hf:Qwen/Qwen2.5-7B-Instruct --num-gpus 1
"""

from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episodic_log.conditions import ALL_CONDITIONS, get_condition
from episodic_log.ingestor.longmemeval import IngestedSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model matrix  (8× A100 80 GB SXM4 — all models fit on a single card)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """One entry in the experiment model matrix.

    Attributes:
        spec: Provider spec string (e.g. ``hf:org/model:4bit``).
        slug: Filesystem-safe short identifier.
        size: ``"small"`` | ``"medium"`` | ``"large"``.
        family: Model family.
        vram_gb: Approximate VRAM required (informational, used in dry-run).
    """

    spec: str
    slug: str
    size: str
    family: str
    vram_gb: int = 0


MODEL_MATRIX: list[ModelSpec] = [
    # --- Small (7–9B BF16, ~14–18 GB) ---
    ModelSpec("hf:meta-llama/Llama-3.1-8B-Instruct",        "llama-3.1-8b",       "small",  "llama",   16),
    ModelSpec("hf:Qwen/Qwen2.5-7B-Instruct",                "qwen-2.5-7b",        "small",  "qwen",    14),
    ModelSpec("hf:google/gemma-2-9b-it",                     "gemma-2-9b",         "small",  "gemma",   18),
    ModelSpec("hf:mistralai/Mistral-7B-Instruct-v0.3",       "mistral-7b",         "small",  "mistral", 14),
    # --- Medium (14–27B BF16, ~28–54 GB — all fit in 80 GB) ---
    ModelSpec("hf:Qwen/Qwen2.5-14B-Instruct",               "qwen-2.5-14b",       "medium", "qwen",    28),
    ModelSpec("hf:google/gemma-2-27b-it",                    "gemma-2-27b",        "medium", "gemma",   54),
    # --- Large (32–72B 4-bit, ~16–36 GB — all fit in 80 GB) ---
    ModelSpec("hf:Qwen/Qwen2.5-32B-Instruct:4bit",          "qwen-2.5-32b-4bit",  "large",  "qwen",    16),
    ModelSpec("hf:meta-llama/Llama-3.3-70B-Instruct:4bit",  "llama-3.3-70b-4bit", "large",  "llama",   35),
    ModelSpec("hf:Qwen/Qwen2.5-72B-Instruct:4bit",          "qwen-2.5-72b-4bit",  "large",  "qwen",    36),
]

_SIZE_LABELS = {"small", "medium", "large"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="run-sweep",
    help="Queue-based multi-GPU CHD evaluation sweep.",
    add_completion=False,
)


@app.command()
def sweep(
    model: Annotated[
        str | None,
        typer.Option("--model", help="Run only this provider spec (overrides matrix)."),
    ] = None,
    model_slug: Annotated[
        str | None,
        typer.Option("--model-slug", help="Slug for --model (auto-derived if omitted)."),
    ] = None,
    size_filter: Annotated[
        str | None,
        typer.Option("--size-filter", help="Only run models of this size: small | medium | large."),
    ] = None,
    family_filter: Annotated[
        str | None,
        typer.Option("--family-filter", help="Only run models from this family."),
    ] = None,
    conditions: Annotated[
        str,
        typer.Option(
            "--conditions",
            help=f"Comma-separated conditions (default: all). Options: {','.join(ALL_CONDITIONS)}",
        ),
    ] = ",".join(ALL_CONDITIONS),
    summary_methods: Annotated[
        str,
        typer.Option("--summary-methods", help="Comma-separated summarizer methods."),
    ] = "structured",
    n: Annotated[
        int | None,
        typer.Option("--n", help="Limit sessions per condition (default: all 500)."),
    ] = None,
    sessions_index: Annotated[
        Path,
        typer.Option(help="Path to sessions_index.jsonl."),
    ] = Path("data/sessions_index.jsonl"),
    output_dir: Annotated[
        Path,
        typer.Option(help="Root results directory."),
    ] = Path("data/results"),
    num_gpus: Annotated[
        int | None,
        typer.Option("--num-gpus", help="GPU slots (default: all available via torch.cuda.device_count())."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Re-run already-completed result files."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the planned queue without executing."),
    ] = False,
) -> None:
    """Run all models through the CHD evaluation sweep using a GPU queue."""
    # ── Parse inputs ──────────────────────────────────────────────────────
    condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
    method_list    = [m.strip() for m in summary_methods.split(",") if m.strip()]

    for c in condition_list:
        if c not in ALL_CONDITIONS:
            typer.echo(f"ERROR: Unknown condition '{c}'. Options: {ALL_CONDITIONS}", err=True)
            raise typer.Exit(1)

    if not sessions_index.exists():
        typer.echo(f"ERROR: {sessions_index} not found. Run ingest.py first.", err=True)
        raise typer.Exit(1)

    sessions_meta = _load_index(sessions_index)
    if n is not None:
        sessions_meta = sessions_meta[:n]

    # ── Select models ────────────────────────────────────────────────────
    if model:
        from scripts.evaluate import _derive_slug  # type: ignore[import]
        slug = model_slug or _derive_slug(model)
        matrix = [ModelSpec(spec=model, slug=slug, size="custom", family="custom")]
    else:
        matrix = list(MODEL_MATRIX)
        if size_filter:
            if size_filter not in _SIZE_LABELS:
                typer.echo(f"ERROR: --size-filter must be one of {_SIZE_LABELS}", err=True)
                raise typer.Exit(1)
            matrix = [m for m in matrix if m.size == size_filter]
        if family_filter:
            matrix = [m for m in matrix if m.family == family_filter.lower()]

    if not matrix:
        typer.echo("ERROR: No models selected after filtering.", err=True)
        raise typer.Exit(1)

    # ── Detect GPU count ─────────────────────────────────────────────────
    if num_gpus is None:
        try:
            import torch
            num_gpus = max(1, torch.cuda.device_count())
        except ImportError:
            num_gpus = 1

    total_runs = len(matrix) * len(condition_list) * len(method_list)
    typer.echo(f"\n{'DRY RUN — ' if dry_run else ''}Queue: {len(matrix)} model(s)  "
               f"{len(condition_list)} condition(s)  {len(method_list)} method(s)  "
               f"= {total_runs} result files  |  GPU slots: {num_gpus}\n")

    for i, m in enumerate(matrix):
        slot = i if i < num_gpus else f"queued (slot free after GPU {i % num_gpus})"
        typer.echo(f"  [{i:>2}] {m.slug:<28} ~{m.vram_gb:>2} GB  →  GPU slot {slot}")

    if dry_run:
        return

    # ── Run queue ────────────────────────────────────────────────────────
    _run_gpu_queue(
        matrix=matrix,
        num_gpus=num_gpus,
        condition_list=condition_list,
        method_list=method_list,
        sessions_meta=sessions_meta,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    typer.echo("\nSweep complete. Run judge.py then score.py.")


# ---------------------------------------------------------------------------
# Queue scheduler
# ---------------------------------------------------------------------------

def _run_gpu_queue(
    matrix: list[ModelSpec],
    num_gpus: int,
    condition_list: list[str],
    method_list: list[str],
    sessions_meta: list[dict],
    output_dir: Path,
    overwrite: bool,
    poll_interval: float = 5.0,
) -> None:
    """Queue-based GPU scheduler.

    Maintains up to *num_gpus* concurrent worker processes.  As soon as a GPU
    slot is freed the next model in the queue starts on it immediately —
    no idle GPUs, no waiting for slow models to synchronise.

    Args:
        matrix: Ordered list of models to evaluate.
        num_gpus: Maximum concurrent worker processes.
        condition_list: Conditions to evaluate per model.
        method_list: Summarizer methods to evaluate per model.
        sessions_meta: Pre-loaded session index.
        output_dir: Root results directory.
        overwrite: Whether to overwrite existing result files.
        poll_interval: Seconds between liveness checks on running workers.
    """
    if num_gpus == 1:
        # Single-GPU path — no subprocess overhead.
        _gpu_worker(
            gpu_id=0,
            assigned_models=matrix,
            condition_list=condition_list,
            method_list=method_list,
            sessions_meta=sessions_meta,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        return

    ctx = mp.get_context("spawn")
    model_queue: list[ModelSpec] = list(matrix)
    free_gpus: list[int]         = list(range(num_gpus))
    running: dict[int, tuple[mp.Process, ModelSpec]] = {}  # gpu_id → (process, spec)

    while model_queue or running:
        # Fill every free GPU slot.
        while model_queue and free_gpus:
            gpu_id     = free_gpus.pop(0)
            model_spec = model_queue.pop(0)
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, [model_spec], condition_list, method_list,
                      sessions_meta, output_dir, overwrite),
                name=f"sweep-{model_spec.slug}",
            )
            p.start()
            running[gpu_id] = (p, model_spec)
            logger.info(
                "GPU %d: started %-28s  [%d queued / %d running]",
                gpu_id, model_spec.slug, len(model_queue), len(running),
            )

        # Poll until at least one worker finishes.
        time.sleep(poll_interval)
        for gpu_id in list(running):
            p, spec = running[gpu_id]
            if not p.is_alive():
                p.join()
                if p.exitcode != 0:
                    logger.error("GPU %d (%s) exited with code %d", gpu_id, spec.slug, p.exitcode)
                else:
                    logger.info("GPU %d: finished %s", gpu_id, spec.slug)
                del running[gpu_id]
                free_gpus.append(gpu_id)


# ---------------------------------------------------------------------------
# GPU worker (runs in a child process)
# ---------------------------------------------------------------------------

def _gpu_worker(
    gpu_id: int,
    assigned_models: list[ModelSpec],
    condition_list: list[str],
    method_list: list[str],
    sessions_meta: list[dict],
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Evaluate assigned models sequentially on one GPU.

    Called as a subprocess by :func:`_run_gpu_queue`.  Binds the process to
    a single GPU via ``CUDA_VISIBLE_DEVICES`` so models never spill over to
    neighbouring cards.

    Args:
        gpu_id: Physical GPU index to bind to.
        assigned_models: Models this worker evaluates in order.
        condition_list: Conditions to run for each model.
        method_list: Summarizer methods to run for each model.
        sessions_meta: Pre-loaded session index records.
        output_dir: Root results directory.
        overwrite: Whether to overwrite existing result files.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [GPU{gpu_id}] %(name)s: %(message)s",
    )
    worker_logger = logging.getLogger(__name__)

    for model_spec in assigned_models:
        worker_logger.info("Loading %s", model_spec.spec)
        try:
            provider = _load_provider(model_spec.spec, device_map="cuda:0")
        except Exception as exc:
            worker_logger.error("Failed to load %s: %s", model_spec.spec, exc, exc_info=True)
            continue

        for cond_name in condition_list:
            for method in method_list:
                out_path = output_dir / model_spec.slug / f"{cond_name}__{method}.jsonl"
                if out_path.exists() and not overwrite:
                    worker_logger.info("Skipping %s/%s — exists", cond_name, method)
                    continue
                worker_logger.info("Running %s/%s/%s", model_spec.slug, cond_name, method)
                _run_condition(
                    provider=provider,
                    condition_name=cond_name,
                    summary_method=method,
                    sessions_meta=sessions_meta,
                    output_path=out_path,
                    model_slug=model_spec.slug,
                )

        _unload_provider(provider)
        del provider
        worker_logger.info("Finished all conditions for %s", model_spec.slug)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_provider(spec: str, device_map: str = "cuda:0"):
    from episodic_log.providers import get_provider
    return get_provider(spec, device_map=device_map)


def _unload_provider(provider) -> None:
    """Release GPU memory held by a HuggingFace provider."""
    try:
        import torch
        if getattr(provider, "_model", None) is not None:
            provider._model = None  # type: ignore[attr-defined]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model unloaded and GPU cache cleared.")
    except ImportError:
        gc.collect()


def _run_condition(
    provider,
    condition_name: str,
    summary_method: str,
    sessions_meta: list[dict],
    output_path: Path,
    model_slug: str,
) -> None:
    cond = get_condition(condition_name, provider=provider, summary_method=summary_method)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    failed = 0
    bar = tqdm(
        sessions_meta,
        desc=f"{model_slug[:18]}/{condition_name}/{summary_method}",
        unit="session",
        dynamic_ncols=True,
        leave=True,
    )
    with output_path.open("w", encoding="utf-8") as fh:
        for meta in bar:
            session = IngestedSession(
                session_id=meta["session_id"],
                log_path=Path(meta["log_path"]),
                summaries_dir=Path(meta["summaries_dir"]),
                question=meta["question"],
                answer=meta["answer"],
                evidence_turn_ids=meta["evidence_turn_ids"],
                question_type=meta["question_type"],
                question_id=meta["question_id"],
            )
            try:
                result = cond.run(session, session.question)
            except Exception as exc:
                logger.error("Session %s failed: %s", session.session_id, exc, exc_info=True)
                failed += 1
                continue

            record: dict = {
                "model_slug": model_slug,
                "session_id": result.session_id,
                "condition": result.condition_name,
                "summary_method": summary_method,
                "question": result.question,
                "ground_truth": meta["answer"],
                "predicted_answer": result.predicted_answer,
                "retrieved_turn_ids": result.retrieved_turn_ids,
                "evidence_turn_ids": meta["evidence_turn_ids"],
                "num_retrieval_calls": result.num_retrieval_calls,
                "question_type": meta["question_type"],
                "metadata": result.metadata,
                "verdict": None,
                "confidence": None,
                "judge_reason": None,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            bar.set_postfix({"failed": failed})

    logger.info(
        "Finished %s/%s/%s — written to %s (failed=%d)",
        model_slug, condition_name, summary_method, output_path, failed,
    )


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
