"""Multi-model sweep — runs all conditions across a configurable model matrix.

Loads each model once, evaluates all requested conditions, then explicitly
unloads it and clears GPU memory before loading the next model.  Results
(predictions only, no verdicts) are written to:
    ``data/results/<model-slug>/<condition>__<method>.jsonl``

Run judge.py separately after the sweep to fill in verdicts in parallel
using batched Groq requests — much faster than inline per-row judging.

Recommended workflow
--------------------
# Step 1 — Inference sweep (GPU)
python scripts/run_sweep.py --conditions baseline,episodic --size-filter small

# Step 2 — Batch judge all results (Groq, parallel, fast)
python scripts/judge.py --judge-provider groq:llama-3.1-70b-versatile

# Step 3 — Score
python scripts/score.py

Other examples
--------------
# Dry-run: see planned runs without executing
python scripts/run_sweep.py --dry-run

# Full sweep — all 9 models × all 7 conditions
python scripts/run_sweep.py

# Only Qwen family
python scripts/run_sweep.py --family-filter qwen --conditions baseline,episodic

# Single model
python scripts/run_sweep.py \
    --model hf:Qwen/Qwen2.5-7B-Instruct \
    --conditions baseline,episodic,proactive
"""

from __future__ import annotations

import gc
import json
import logging
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
# Model matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """One entry in the experiment model matrix."""

    spec: str
    slug: str
    size: str  # "small" | "medium" | "large"
    family: str


MODEL_MATRIX: list[ModelSpec] = [
    # --- Small (7–9B, BF16) ---
    ModelSpec("hf:meta-llama/Llama-3.1-8B-Instruct",          "llama-3.1-8b",    "small",  "llama"),
    ModelSpec("hf:Qwen/Qwen2.5-7B-Instruct",                  "qwen-2.5-7b",     "small",  "qwen"),
    ModelSpec("hf:google/gemma-2-9b-it",                       "gemma-2-9b",      "small",  "gemma"),
    ModelSpec("hf:mistralai/Mistral-7B-Instruct-v0.3",         "mistral-7b",      "small",  "mistral"),
    # --- Medium (14–27B) ---
    ModelSpec("hf:Qwen/Qwen2.5-14B-Instruct",                  "qwen-2.5-14b",    "medium", "qwen"),
    ModelSpec("hf:google/gemma-2-27b-it:4bit",                 "gemma-2-27b-4bit","medium", "gemma"),
    # --- Large (32–72B, 4-bit) ---
    ModelSpec("hf:Qwen/Qwen2.5-32B-Instruct:4bit",            "qwen-2.5-32b-4bit","large", "qwen"),
    ModelSpec("hf:meta-llama/Llama-3.3-70B-Instruct:4bit",    "llama-3.3-70b-4bit","large","llama"),
    ModelSpec("hf:Qwen/Qwen2.5-72B-Instruct:4bit",            "qwen-2.5-72b-4bit","large", "qwen"),
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
        typer.Option(
            "--size-filter",
            help="Only run models of this size: small | medium | large.",
        ),
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
    judge: Annotated[
        bool,
        typer.Option(
            "--judge/--no-judge",
            help=(
                "Inline CHD judging after each condition. "
                "Prefer running scripts/judge.py separately for batched parallel judging."
            ),
        ),
    ] = False,
    judge_provider_spec: Annotated[
        str,
        typer.Option("--judge-provider", help="Provider spec for inline CHD judge (only used with --judge)."),
    ] = "groq:llama-3.1-70b-versatile",
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Re-run already-completed result files."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the planned runs without executing any."),
    ] = False,
    device_map: Annotated[
        str,
        typer.Option("--device-map", help="HuggingFace device_map for model loading."),
    ] = "auto",
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

    # ── Plan runs ────────────────────────────────────────────────────────
    runs: list[tuple[ModelSpec, str, str]] = []  # (model, condition, method)
    for m in matrix:
        for cond in condition_list:
            for method in method_list:
                runs.append((m, cond, method))

    typer.echo(f"\n{'DRY RUN — ' if dry_run else ''}Planned {len(runs)} run(s):\n")
    for m, cond, method in runs:
        out = output_dir / m.slug / f"{cond}__{method}.jsonl"
        status = "[SKIP existing]" if (out.exists() and not overwrite) else ""
        typer.echo(f"  {m.slug:<28} {cond:<16} {method:<12} {status}")

    if dry_run:
        return

    # ── Judge provider (shared across all models) ─────────────────────────
    judge_obj = None
    if judge:
        from episodic_log.judge import CHDJudge
        from episodic_log.providers import get_provider
        judge_provider = get_provider(judge_provider_spec)
        judge_obj = CHDJudge(provider=judge_provider)
        typer.echo(f"\nJudge provider: {judge_provider_spec}")

    # ── Main sweep ───────────────────────────────────────────────────────
    for model_spec in matrix:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Loading model: {model_spec.spec}  [{model_spec.size}]")
        typer.echo(f"{'='*60}")

        try:
            provider = _load_provider(model_spec.spec, device_map=device_map)
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_spec.spec, exc, exc_info=True)
            continue

        for cond_name in condition_list:
            for method in method_list:
                out_path = output_dir / model_spec.slug / f"{cond_name}__{method}.jsonl"
                if out_path.exists() and not overwrite:
                    typer.echo(f"  Skipping {cond_name}/{method} — already exists.")
                    continue

                typer.echo(f"  Running {cond_name}/{method} on {len(sessions_meta)} sessions …")
                _run_condition(
                    provider=provider,
                    condition_name=cond_name,
                    summary_method=method,
                    sessions_meta=sessions_meta,
                    output_path=out_path,
                    judge_obj=judge_obj,
                    model_slug=model_spec.slug,
                )

        # Unload model and free GPU memory before next model.
        _unload_provider(provider)
        del provider

    typer.echo("\nSweep complete. Run score.py to view results.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    judge_obj,
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

            if judge_obj is not None:
                verdict = judge_obj.judge(
                    question=result.question,
                    ground_truth=meta["answer"],
                    predicted=result.predicted_answer,
                )
                record["verdict"] = verdict.verdict
                record["confidence"] = verdict.confidence
                record["judge_reason"] = verdict.reason

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
