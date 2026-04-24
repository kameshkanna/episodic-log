"""Multi-model sweep — queue-based GPU scheduler with hardware auto-detection.

At startup the script measures available GPU count and per-GPU VRAM, then
resolves each catalogue entry to the best-fit quantization level and GPU
allocation:

  1. BF16 on the fewest GPUs that fit  (preferred — no quality loss)
  2. NF4 4-bit on the fewest GPUs that fit  (fallback)
  3. Model is skipped with a warning if it still doesn't fit

Results are written to:
    ``data/results/<model-slug>/<condition>__<method>.jsonl``

Usage examples
--------------
# Full auto — detect hardware, run everything that fits
python scripts/run_sweep.py --summary-methods lexical,scout,echo

# Dry-run: see what would run and on how many GPUs
python scripts/run_sweep.py --dry-run

# Override hardware (useful for testing / partial runs)
python scripts/run_sweep.py --num-gpus 4 --vram-per-gpu 40

# Single custom model
python scripts/run_sweep.py --model hf:Qwen/Qwen2.5-7B-Instruct --num-gpus 1

# Filter to small models only, two conditions
python scripts/run_sweep.py --size-filter small --conditions amnesiac,recall
"""

from __future__ import annotations

import gc
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
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
# Model catalogue  (hardware-independent; quantization is resolved at runtime)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCatalogEntry:
    """Hardware-independent model descriptor.

    Attributes:
        base_spec: Provider spec without quantization suffix,
            e.g. ``"hf:Qwen/Qwen2.5-72B-Instruct"``.
        base_slug: Filesystem-safe short name, e.g. ``"qwen-2.5-72b"``.
        size: ``"small"`` | ``"medium"`` | ``"large"``.
        family: Model family string.
        vram_bf16: Estimated VRAM in GB for full BF16 precision.
        vram_4bit: Estimated VRAM in GB for NF4 4-bit quantization.
            Set to 0 for models where 4-bit is not recommended (e.g. sub-10B).
    """

    base_spec: str
    base_slug: str
    size: str
    family: str
    vram_bf16: int
    vram_4bit: int


@dataclass(frozen=True)
class ModelSpec:
    """Hardware-resolved model descriptor passed to the GPU scheduler.

    Attributes:
        spec: Full provider spec including optional ``:4bit`` suffix.
        slug: Filesystem slug, appended with ``-4bit`` when quantized.
        size: Inherited from catalogue.
        family: Inherited from catalogue.
        vram_gb: Estimated VRAM for the resolved precision.
        num_gpus_needed: How many GPUs this model occupies.
    """

    spec: str
    slug: str
    size: str
    family: str
    vram_gb: int
    num_gpus_needed: int


MODEL_CATALOGUE: list[ModelCatalogEntry] = [
    # ── Small (7–9 B, BF16 14–18 GB) ─────────────────────────────────────
    ModelCatalogEntry("hf:meta-llama/Llama-3.1-8B-Instruct",  "llama-3.1-8b",  "small",  "llama",   16,  5),
    ModelCatalogEntry("hf:Qwen/Qwen2.5-7B-Instruct",          "qwen-2.5-7b",   "small",  "qwen",    14,  5),
    ModelCatalogEntry("hf:google/gemma-2-9b-it",               "gemma-2-9b",    "small",  "gemma",   18,  6),
    ModelCatalogEntry("hf:mistralai/Mistral-7B-Instruct-v0.3", "mistral-7b",    "small",  "mistral", 14,  5),
    # ── Medium (14–27 B) ──────────────────────────────────────────────────
    ModelCatalogEntry("hf:Qwen/Qwen2.5-14B-Instruct",         "qwen-2.5-14b",  "medium", "qwen",    28,  8),
    ModelCatalogEntry("hf:google/gemma-2-27b-it",              "gemma-2-27b",   "medium", "gemma",   54, 15),
    # ── Large (32–72 B) ───────────────────────────────────────────────────
    ModelCatalogEntry("hf:Qwen/Qwen2.5-32B-Instruct",         "qwen-2.5-32b",  "large",  "qwen",    64, 20),
    ModelCatalogEntry("hf:meta-llama/Llama-3.3-70B-Instruct", "llama-3.3-70b", "large",  "llama",  140, 38),
    ModelCatalogEntry("hf:Qwen/Qwen2.5-72B-Instruct",         "qwen-2.5-72b",  "large",  "qwen",   144, 42),
]

_SIZE_LABELS = {"small", "medium", "large"}

# Fraction of each GPU's total VRAM assumed available for model weights.
# The remainder is reserved for KV-cache and activation buffers.
_VRAM_USABLE_FRACTION = 0.88

# Conditions that require per-session BM25 summary files.
_CONDITIONS_NEEDING_SUMMARIES: frozenset[str] = frozenset(
    {"episodic", "adversarial", "proactive", "external"}
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="run-sweep",
    help="Hardware-aware multi-GPU CHD evaluation sweep.",
    add_completion=False,
)


@app.command()
def sweep(
    model: Annotated[
        str | None,
        typer.Option("--model", help="Run only this provider spec (overrides catalogue)."),
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
    ] = "lexical",
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
        typer.Option("--num-gpus", help="Override detected GPU count."),
    ] = None,
    vram_per_gpu: Annotated[
        float | None,
        typer.Option(
            "--vram-per-gpu",
            help="Override detected VRAM per GPU in GB (useful for testing on CPU).",
        ),
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

    # ── Detect hardware ───────────────────────────────────────────────────
    detected_gpus, detected_vram = _detect_hardware()
    num_gpus   = num_gpus    if num_gpus    is not None else detected_gpus
    vram_per_gpu = vram_per_gpu if vram_per_gpu is not None else detected_vram

    typer.echo(
        f"\nHardware: {num_gpus} GPU(s)  ×  {vram_per_gpu:.0f} GB each  "
        f"= {num_gpus * vram_per_gpu:.0f} GB total VRAM"
    )

    # ── Build model matrix ────────────────────────────────────────────────
    if model:
        slug = model_slug or _derive_slug(model)
        # For a custom model we don't know its VRAM; put it on 1 GPU.
        matrix: list[ModelSpec] = [
            ModelSpec(spec=model, slug=slug, size="custom", family="custom",
                      vram_gb=0, num_gpus_needed=1)
        ]
    else:
        matrix = _adapt_matrix(
            catalogue=MODEL_CATALOGUE,
            vram_per_gpu=vram_per_gpu,
            num_gpus=num_gpus,
            size_filter=size_filter,
            family_filter=family_filter,
        )

    if not matrix:
        typer.echo(
            "ERROR: No models selected after filtering / hardware check. "
            "Try --vram-per-gpu to override VRAM detection.",
            err=True,
        )
        raise typer.Exit(1)

    # ── Pre-flight: verify summary files exist for retrieval conditions ───
    _preflight_check_summaries(sessions_meta, condition_list, method_list)

    # ── Summary ───────────────────────────────────────────────────────────
    total_runs = len(matrix) * len(condition_list) * len(method_list)
    typer.echo(
        f"{'DRY RUN — ' if dry_run else ''}Queue: {len(matrix)} model(s)  "
        f"{len(condition_list)} condition(s)  {len(method_list)} method(s)  "
        f"= {total_runs} result files  |  GPU slots: {num_gpus}\n"
    )
    for i, m in enumerate(matrix):
        quant_tag = " (4-bit)" if ":4bit" in m.spec else " (BF16) "
        typer.echo(
            f"  [{i:>2}] {m.slug:<32} ~{m.vram_gb:>3} GB{quant_tag} ×{m.num_gpus_needed} GPU(s)"
        )

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
# Hardware detection
# ---------------------------------------------------------------------------

def _detect_hardware() -> tuple[int, float]:
    """Return (num_gpus, vram_per_gpu_gb) by querying torch.cuda.

    Uses the *smallest* GPU's VRAM if the system has mixed cards so that
    the scheduler never over-allocates on a weaker device.

    Returns:
        A (count, vram_gb) tuple.  Falls back to (1, 0.0) when torch is
        not installed or no CUDA devices are found.
    """
    try:
        import torch

        count = torch.cuda.device_count()
        if count == 0:
            logger.warning("torch.cuda.device_count() == 0; running on CPU.")
            return 1, 0.0

        vram_gb = min(
            torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(count)
        )
        gpu_names = [torch.cuda.get_device_properties(i).name for i in range(count)]
        logger.info("Detected %d GPU(s): %s", count, ", ".join(gpu_names))
        return count, round(vram_gb, 1)

    except ImportError:
        logger.warning("torch not installed; assuming CPU-only.")
        return 1, 0.0


# ---------------------------------------------------------------------------
# Matrix adaptation
# ---------------------------------------------------------------------------

def _resolve_model_spec(
    entry: ModelCatalogEntry,
    usable_vram: float,
    num_gpus: int,
) -> ModelSpec | None:
    """Resolve one catalogue entry to a concrete ModelSpec for this hardware.

    Tries BF16 first (no quality loss), then NF4 4-bit.  Within each
    precision level it finds the minimum number of GPUs required, capped
    at *num_gpus*.  Returns ``None`` if the model cannot fit at all.

    Args:
        entry: Hardware-independent catalogue descriptor.
        usable_vram: Usable VRAM per GPU in GB (after headroom deduction).
        num_gpus: Total physical GPUs available.

    Returns:
        A resolved :class:`ModelSpec`, or ``None`` if the model doesn't fit.
    """
    def _try(vram_needed: int, spec_suffix: str, slug_suffix: str) -> ModelSpec | None:
        if vram_needed <= 0:
            return None
        gpus_needed = math.ceil(vram_needed / usable_vram)
        if gpus_needed > num_gpus:
            return None
        return ModelSpec(
            spec=entry.base_spec + spec_suffix,
            slug=entry.base_slug + slug_suffix,
            size=entry.size,
            family=entry.family,
            vram_gb=vram_needed,
            num_gpus_needed=gpus_needed,
        )

    # BF16 preferred
    result = _try(entry.vram_bf16, "", "")
    if result is not None:
        return result

    # 4-bit fallback
    return _try(entry.vram_4bit, ":4bit", "-4bit")


def _adapt_matrix(
    catalogue: list[ModelCatalogEntry],
    vram_per_gpu: float,
    num_gpus: int,
    size_filter: str | None = None,
    family_filter: str | None = None,
) -> list[ModelSpec]:
    """Build a hardware-adapted :class:`ModelSpec` list from *catalogue*.

    Each entry is resolved to the best available precision for the given
    hardware.  Models that cannot fit on any number of GPUs are logged and
    skipped.

    Args:
        catalogue: Full model catalogue.
        vram_per_gpu: Total VRAM per GPU in GB (raw, not adjusted for headroom).
        num_gpus: Number of physical GPUs.
        size_filter: If set, only include models of this size.
        family_filter: If set, only include models of this family.

    Returns:
        Ordered list of resolved :class:`ModelSpec` objects.
    """
    usable = vram_per_gpu * _VRAM_USABLE_FRACTION

    matrix: list[ModelSpec] = []
    for entry in catalogue:
        if size_filter and entry.size != size_filter:
            continue
        if family_filter and entry.family != family_filter.lower():
            continue

        if usable <= 0:
            # CPU-only / unknown hardware: include all models without GPU assignment.
            matrix.append(ModelSpec(
                spec=entry.base_spec, slug=entry.base_slug,
                size=entry.size, family=entry.family,
                vram_gb=entry.vram_bf16, num_gpus_needed=1,
            ))
            continue

        spec = _resolve_model_spec(entry, usable, num_gpus)
        if spec is not None:
            matrix.append(spec)
        else:
            logger.warning(
                "Skipping %-28s — needs %d GB BF16 / %d GB 4-bit but only "
                "%d × %.0f GB = %.0f GB usable VRAM available.",
                entry.base_slug, entry.vram_bf16, entry.vram_4bit,
                num_gpus, usable, num_gpus * usable,
            )

    return matrix


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------

def _preflight_check_summaries(
    sessions_meta: list[dict],
    condition_list: list[str],
    method_list: list[str],
) -> None:
    """Abort with a clear error if any required summary files are missing.

    Only checks methods for conditions that actually use BM25 retrieval.

    Args:
        sessions_meta: Pre-loaded session index records.
        condition_list: Conditions requested for this sweep run.
        method_list: Summarizer methods requested for this sweep run.
    """
    needs_summaries = any(c in _CONDITIONS_NEEDING_SUMMARIES for c in condition_list)
    if not needs_summaries:
        return

    any_missing = False
    for method in method_list:
        missing = [
            s for s in sessions_meta
            if not (Path(s["summaries_dir"]) / f"{method}.jsonl").exists()
        ]
        if missing:
            any_missing = True
            typer.echo(
                f"PRE-FLIGHT ERROR: {len(missing)}/{len(sessions_meta)} sessions "
                f"are missing '{method}.jsonl' summary files.\n"
                f"  First missing: {missing[0]['session_id']}\n"
                f"  Expected:      {Path(missing[0]['summaries_dir']) / f'{method}.jsonl'}\n"
                f"  Fix: python scripts/summarize.py --method {method}"
                + (f" --provider hf:<model>" if method != "lexical" else ""),
                err=True,
            )
        else:
            typer.echo(
                f"Pre-flight OK: all {len(sessions_meta)} sessions have '{method}.jsonl'."
            )

    if any_missing:
        raise typer.Exit(1)


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
    """Queue-based GPU scheduler supporting variable GPU allocation per model.

    Scans the pending queue on every tick and starts any model whose
    ``num_gpus_needed`` can be satisfied by the current free-GPU pool.
    Multi-GPU models receive a contiguous block and see them via
    ``CUDA_VISIBLE_DEVICES``; HuggingFace ``device_map="auto"`` then
    shards the weights across those cards automatically.

    Args:
        matrix: Ordered list of models to evaluate.
        num_gpus: Total physical GPUs available.
        condition_list: Conditions to evaluate per model.
        method_list: Summarizer methods to evaluate per model.
        sessions_meta: Pre-loaded session index.
        output_dir: Root results directory.
        overwrite: Whether to overwrite existing result files.
        poll_interval: Seconds between liveness checks on running workers.
    """
    if num_gpus == 1:
        _gpu_worker(
            cuda_devices="0",
            assigned_models=matrix,
            condition_list=condition_list,
            method_list=method_list,
            sessions_meta=sessions_meta,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        return

    ctx = mp.get_context("spawn")
    model_queue: list[ModelSpec]                                     = list(matrix)
    free_gpus:   list[int]                                           = list(range(num_gpus))
    running: dict[frozenset[int], tuple[mp.Process, ModelSpec]]      = {}

    while model_queue or running:
        # Scan queue for any model we can start right now.
        started = True
        while started:
            started = False
            for i, model_spec in enumerate(model_queue):
                needed = model_spec.num_gpus_needed
                if len(free_gpus) >= needed:
                    assigned  = free_gpus[:needed]
                    free_gpus = free_gpus[needed:]
                    model_queue.pop(i)
                    cuda_devices = ",".join(str(g) for g in assigned)
                    p = ctx.Process(
                        target=_gpu_worker,
                        args=(cuda_devices, [model_spec], condition_list, method_list,
                              sessions_meta, output_dir, overwrite),
                        name=f"sweep-{model_spec.slug}",
                    )
                    p.start()
                    key = frozenset(assigned)
                    running[key] = (p, model_spec)
                    logger.info(
                        "GPUs [%s]: started %-32s  [%d queued / %d running]",
                        cuda_devices, model_spec.slug, len(model_queue), len(running),
                    )
                    started = True
                    break  # restart scan after each allocation

        # Poll for finished workers and reclaim their GPUs.
        time.sleep(poll_interval)
        for key in list(running):
            p, spec = running[key]
            if not p.is_alive():
                p.join()
                if p.exitcode != 0:
                    logger.error(
                        "GPUs %s (%s) exited with code %d", set(key), spec.slug, p.exitcode
                    )
                else:
                    logger.info("GPUs %s: finished %s", set(key), spec.slug)
                del running[key]
                free_gpus.extend(sorted(key))


# ---------------------------------------------------------------------------
# GPU worker (runs in a child process)
# ---------------------------------------------------------------------------

def _gpu_worker(
    cuda_devices: str,
    assigned_models: list[ModelSpec],
    condition_list: list[str],
    method_list: list[str],
    sessions_meta: list[dict],
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Evaluate assigned models sequentially on one or more GPUs.

    Args:
        cuda_devices: Comma-separated physical GPU indices (e.g. ``"0"`` or
            ``"6,7"``).  Passed directly to ``CUDA_VISIBLE_DEVICES``.
        assigned_models: Models this worker evaluates in order.
        condition_list: Conditions to run for each model.
        method_list: Summarizer methods to run for each model.
        sessions_meta: Pre-loaded session index records.
        output_dir: Root results directory.
        overwrite: Whether to overwrite existing result files.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [GPU{cuda_devices}] %(name)s: %(message)s",
    )
    worker_logger = logging.getLogger(__name__)

    for model_spec in assigned_models:
        worker_logger.info("Loading %s", model_spec.spec)
        try:
            provider = _load_provider(model_spec.spec, device_map="auto")
        except Exception as exc:
            worker_logger.error(
                "Failed to load %s: %s", model_spec.spec, exc, exc_info=True
            )
            continue

        for cond_name in condition_list:
            for method in method_list:
                out_path = output_dir / model_spec.slug / f"{cond_name}__{method}.jsonl"
                if out_path.exists() and not overwrite:
                    worker_logger.info("Skipping %s/%s — exists", cond_name, method)
                    continue
                worker_logger.info(
                    "Running %s/%s/%s", model_spec.slug, cond_name, method
                )
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

def _derive_slug(spec: str) -> str:
    """Derive a filesystem-safe slug from a provider spec string.

    Args:
        spec: Provider spec, e.g. ``"hf:org/model-name:4bit"``.

    Returns:
        Lowercased hyphen-separated slug, e.g. ``"model-name-4bit"``.
    """
    parts = spec.split(":")
    name = parts[-1] if parts[-1] in ("4bit", "8bit") else parts[-1]
    if parts[-1] in ("4bit", "8bit"):
        quant = parts[-1]
        name_part = parts[-2].split("/")[-1].lower()
        return f"{name_part}-{quant}"
    return parts[-1].split("/")[-1].lower()


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
                logger.error(
                    "Session %s failed: %s", session.session_id, exc, exc_info=True
                )
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
