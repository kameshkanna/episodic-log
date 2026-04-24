"""Multi-model sweep — runs all conditions across a configurable model matrix.

Loads each model once per GPU worker, evaluates all assigned conditions, then
explicitly unloads it before loading the next model.  With --num-gpus > 1 the
matrix is distributed across workers so multiple models run simultaneously.

Results (predictions only, no verdicts) are written to:
    ``data/results/<model-slug>/<condition>__<method>.jsonl``

Run judge.py separately after the sweep to fill in verdicts using local HF
models across all GPUs — no API calls, no rate limits.

Recommended workflow (8x A100)
--------------------------------
# Step 1 — Inference sweep (all 8 GPUs in parallel)
python scripts/run_sweep.py --summary-methods structured,haiku,self

# Step 2 — Local HF judge across all 8 GPUs
python scripts/judge.py --judge-provider hf:Qwen/Qwen2.5-14B-Instruct

# Step 3 — Score
python scripts/score.py --breakdown

Other examples
--------------
# Dry-run: see planned runs without executing
python scripts/run_sweep.py --dry-run

# Small models only, two conditions
python scripts/run_sweep.py --size-filter small --conditions baseline,episodic

# Single model, explicit GPU
python scripts/run_sweep.py --model hf:Qwen/Qwen2.5-7B-Instruct --num-gpus 1

# Override GPU count
python scripts/run_sweep.py --num-gpus 4
"""

from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
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
# Model matrix  (tuned for 8× A100 80 GB SXM4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """One entry in the experiment model matrix.

    Attributes:
        spec: Provider spec string (e.g. ``hf:org/model:4bit``).
        slug: Filesystem-safe short identifier.
        size: ``"small"`` | ``"medium"`` | ``"large"``.
        family: Model family (``"llama"`` | ``"qwen"`` | ``"gemma"`` | ``"mistral"``).
        device_map: HuggingFace device_map value.  ``"cuda:0"`` for models that
            fit on one GPU; ``"auto"`` for large models that should span all
            available GPUs for maximum throughput.
    """

    spec: str
    slug: str
    size: str
    family: str
    device_map: str = "cuda:0"


MODEL_MATRIX: list[ModelSpec] = [
    # --- Small (7–9B, BF16, ~14–18 GB) — one GPU each, run 8 in parallel ---
    ModelSpec("hf:meta-llama/Llama-3.1-8B-Instruct",        "llama-3.1-8b",       "small",  "llama", "cuda:0"),
    ModelSpec("hf:Qwen/Qwen2.5-7B-Instruct",                "qwen-2.5-7b",        "small",  "qwen",  "cuda:0"),
    ModelSpec("hf:google/gemma-2-9b-it",                     "gemma-2-9b",         "small",  "gemma", "cuda:0"),
    ModelSpec("hf:mistralai/Mistral-7B-Instruct-v0.3",       "mistral-7b",         "small",  "mistral","cuda:0"),
    # --- Medium (14–27B, BF16, ~28–54 GB) — one GPU each, run 8 in parallel ---
    ModelSpec("hf:Qwen/Qwen2.5-14B-Instruct",               "qwen-2.5-14b",       "medium", "qwen",  "cuda:0"),
    ModelSpec("hf:google/gemma-2-27b-it",                    "gemma-2-27b",        "medium", "gemma", "cuda:0"),
    # --- Large (32–72B, 4-bit) — device_map="auto" spans all GPUs for peak throughput ---
    ModelSpec("hf:Qwen/Qwen2.5-32B-Instruct:4bit",          "qwen-2.5-32b-4bit",  "large",  "qwen",  "auto"),
    ModelSpec("hf:meta-llama/Llama-3.3-70B-Instruct:4bit",  "llama-3.3-70b-4bit", "large",  "llama", "auto"),
    ModelSpec("hf:Qwen/Qwen2.5-72B-Instruct:4bit",          "qwen-2.5-72b-4bit",  "large",  "qwen",  "auto"),
]

_SIZE_LABELS = {"small", "medium", "large"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="run-sweep",
    help="Multi-model sweep across conditions and summarizer methods.",
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
        typer.Option("--family-filter", help="Only run models from this family (llama|qwen|gemma|mistral)."),
    ] = None,
    conditions: Annotated[
        str,
        typer.Option(
            "--conditions",
            help=f"Comma-separated conditions to run (default: all). Options: {','.join(ALL_CONDITIONS)}",
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
        typer.Option(
            "--num-gpus",
            help="Number of GPUs to use in parallel (default: all available).",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Re-run already-completed result files."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the planned runs without executing any."),
    ] = False,
) -> None:
    """Execute the full multi-model CHD evaluation sweep."""
    # ── Parse inputs ──────────────────────────────────────────────────────
    condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
    method_list = [m.strip() for m in summary_methods.split(",") if m.strip()]

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
    typer.echo(f"GPUs to use: {num_gpus}")

    # ── Plan runs ────────────────────────────────────────────────────────
    runs: list[tuple[ModelSpec, str, str]] = []
    for m in matrix:
        for cond in condition_list:
            for method in method_list:
                runs.append((m, cond, method))

    assignments = _distribute_models([m for m in matrix if m.device_map != "auto"], num_gpus)
    typer.echo(f"\n{'DRY RUN — ' if dry_run else ''}Planned {len(runs)} run(s):\n")
    typer.echo("  Phase 1 — parallel (one GPU each):")
    for gpu_id, specs in assignments.items():
        for m in specs:
            typer.echo(f"    GPU {gpu_id}  {m.slug:<28} [{m.size}]")
    multi_gpu = [m for m in matrix if m.device_map == "auto"]
    if multi_gpu:
        typer.echo("  Phase 2 — sequential (all GPUs via device_map=auto):")
        for m in multi_gpu:
            typer.echo(f"    ALL GPUs  {m.slug:<28} [{m.size}]")
    typer.echo("")
    for m, cond, method in runs:
        out = output_dir / m.slug / f"{cond}__{method}.jsonl"
        status = "[SKIP existing]" if (out.exists() and not overwrite) else ""
        typer.echo(f"  {m.slug:<28} {cond:<16} {method:<12} {status}")

    if dry_run:
        return

    # ── Two-phase execution ───────────────────────────────────────────────
    # Phase 1: small/medium models — one GPU each, run in parallel.
    # Phase 2: large models with device_map="auto" — use all GPUs, sequential.
    single_gpu_models = [m for m in matrix if m.device_map != "auto"]
    multi_gpu_models  = [m for m in matrix if m.device_map == "auto"]

    if single_gpu_models:
        typer.echo(f"\n── Phase 1: {len(single_gpu_models)} model(s) across {num_gpus} GPU(s) in parallel")
        phase_assignments = _distribute_models(single_gpu_models, num_gpus)
        if num_gpus == 1:
            _gpu_worker(
                gpu_id=0,
                assigned_models=single_gpu_models,
                condition_list=condition_list,
                method_list=method_list,
                sessions_meta=sessions_meta,
                output_dir=output_dir,
                overwrite=overwrite,
            )
        else:
            ctx = mp.get_context("spawn")
            processes: list[mp.Process] = []
            for gpu_id, assigned_models in phase_assignments.items():
                if not assigned_models:
                    continue
                p = ctx.Process(
                    target=_gpu_worker,
                    args=(gpu_id, assigned_models, condition_list, method_list,
                          sessions_meta, output_dir, overwrite),
                    name=f"sweep-gpu{gpu_id}",
                )
                p.start()
                processes.append(p)
                typer.echo(f"  Launched GPU {gpu_id} — {[m.slug for m in assigned_models]}")
            for p in processes:
                p.join()
                if p.exitcode != 0:
                    logger.error("Worker %s exited with code %d", p.name, p.exitcode)

    if multi_gpu_models:
        typer.echo(f"\n── Phase 2: {len(multi_gpu_models)} large model(s) with device_map=auto (all GPUs)")
        for model_spec in multi_gpu_models:
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Loading {model_spec.spec}  [{model_spec.size}]  device_map=auto")
            typer.echo(f"{'='*60}")
            try:
                provider = _load_provider(model_spec.spec, device_map="auto")
            except Exception as exc:
                logger.error("Failed to load %s: %s", model_spec.spec, exc, exc_info=True)
                continue
            for cond_name in condition_list:
                for method in method_list:
                    out_path = output_dir / model_spec.slug / f"{cond_name}__{method}.jsonl"
                    if out_path.exists() and not overwrite:
                        typer.echo(f"  Skipping {cond_name}/{method} — exists")
                        continue
                    typer.echo(f"  Running {cond_name}/{method}")
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

    typer.echo("\nSweep complete. Run judge.py then score.py.")


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
    """Load and evaluate each assigned model sequentially on a single GPU.

    Intended to run as a child process via ``multiprocessing.spawn``.
    Sets ``CUDA_VISIBLE_DEVICES`` so the process owns exactly one GPU.

    Args:
        gpu_id: Physical GPU index to bind to.
        assigned_models: Models this worker is responsible for.
        condition_list: Condition names to evaluate.
        method_list: Summarizer methods to evaluate.
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
        worker_logger.info("Loading %s  device_map=%s", model_spec.spec, model_spec.device_map)
        try:
            provider = _load_provider(model_spec.spec, device_map=model_spec.device_map)
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

def _distribute_models(matrix: list[ModelSpec], num_gpus: int) -> dict[int, list[ModelSpec]]:
    """Assign models to GPUs round-robin by index.

    Args:
        matrix: Full list of models to distribute.
        num_gpus: Number of available GPUs.

    Returns:
        Dict mapping GPU index → list of assigned ModelSpecs.
    """
    assignments: dict[int, list[ModelSpec]] = {i: [] for i in range(num_gpus)}
    for idx, model_spec in enumerate(matrix):
        assignments[idx % num_gpus].append(model_spec)
    return assignments


def _load_provider(spec: str, device_map: str = "auto"):
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
        position=0,
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
