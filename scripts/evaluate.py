"""Run evaluation conditions on ingested LongMemEval sessions.

Sessions are split evenly across all available GPUs so every worker loads the
model once and processes its chunk in parallel — identical pattern to summarize.py.

Usage
-----
# Amnesiac — 5 sessions, Qwen 7B
python scripts/evaluate.py --condition amnesiac --n 5 \
    --provider hf:Qwen/Qwen2.5-7B-Instruct

# Recall/lexical — all sessions, 4 GPUs auto-detected
python scripts/evaluate.py --condition recall --summary-method lexical \
    --provider hf:Qwen/Qwen2.5-7B-Instruct

# Recall/scout — 32B BF16 across 4 H100s (1 GPU each, fits 80GB)
python scripts/evaluate.py --condition recall --summary-method scout \
    --provider hf:Qwen/Qwen2.5-32B-Instruct

# Force GPU count
python scripts/evaluate.py --condition recall --summary-method echo \
    --provider hf:Qwen/Qwen2.5-7B-Instruct --num-gpus 4
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

from episodic_log.conditions import ALL_CONDITIONS, get_condition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="evaluate",
    help="Run a condition on all ingested sessions and write results JSONL.",
    add_completion=False,
)


@app.command()
def evaluate(
    condition: Annotated[
        str,
        typer.Option(
            "--condition",
            help="Condition family: amnesiac | recall | topk.",
            case_sensitive=False,
        ),
    ] = "amnesiac",
    provider_spec: Annotated[
        str,
        typer.Option("--provider", help="Provider spec (e.g. hf:Qwen/Qwen3-72B)."),
    ] = "groq:llama-3.1-8b-instant",
    summary_method: Annotated[
        str,
        typer.Option(
            "--summary-method",
            help="Summary index to use for recall/topk conditions: lexical | scout | echo.",
        ),
    ] = "lexical",
    retrieval_k: Annotated[
        int,
        typer.Option(
            "--retrieval-k",
            help="Top-k turns to inject for topk condition (3 | 5 | 10).",
        ),
    ] = 5,
    n: Annotated[
        int | None,
        typer.Option("--n", help="Limit to first N sessions (default: all)."),
    ] = None,
    sessions_index: Annotated[
        Path,
        typer.Option(help="Path to sessions_index.jsonl produced by ingest.py."),
    ] = Path("data/sessions_index.jsonl"),
    output_dir: Annotated[
        Path,
        typer.Option(help="Root directory for results JSONL files."),
    ] = Path("data/results"),
    num_gpus: Annotated[
        int | None,
        typer.Option(
            "--num-gpus",
            help="Total GPUs to use (default: all available). Ignored for CPU-only providers.",
        ),
    ] = None,
    gpus_per_worker: Annotated[
        int,
        typer.Option(
            "--gpus-per-worker",
            help="GPUs per worker (default 1). Use 2 for 70B BF16 models.",
        ),
    ] = 1,
    model_slug: Annotated[
        str | None,
        typer.Option(
            "--model-slug",
            help="Short name used in output path. Auto-derived from --provider if omitted.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Overwrite existing result files."),
    ] = False,
) -> None:
    """Run *condition* on all sessions, parallelised across GPUs."""
    # Build the full registry key:
    #   "amnesiac" | "recall/lexical" | "topk/lexical/k5" etc.
    if condition == "amnesiac":
        cond_key = "amnesiac"
    elif condition == "topk":
        cond_key = f"topk/{summary_method}/k{retrieval_k}"
    else:
        cond_key = f"{condition}/{summary_method}"
    if cond_key not in ALL_CONDITIONS:
        typer.echo(
            f"ERROR: Unknown condition {cond_key!r}. Available: {sorted(ALL_CONDITIONS)}",
            err=True,
        )
        raise typer.Exit(1)

    if not sessions_index.exists():
        typer.echo(
            f"ERROR: sessions index not found at {sessions_index}. Run ingest.py first.",
            err=True,
        )
        raise typer.Exit(1)

    all_sessions = _load_index(sessions_index)
    if n is not None:
        all_sessions = all_sessions[:n]

    slug = model_slug or _derive_slug(provider_spec)
    model_output_dir = output_dir / slug
    model_output_dir.mkdir(parents=True, exist_ok=True)

    cond_label = cond_key
    # Use the full cond_key (slashes → double-underscore) as the filename so
    # different k values don't overwrite each other.
    output_stem = cond_key.replace("/", "__")
    output_path = model_output_dir / f"{output_stem}.jsonl"

    if not overwrite and output_path.exists() and output_path.stat().st_size > 0:
        typer.echo(f"Already complete: {output_path}. Use --overwrite to re-run.")
        return

    typer.echo(
        f"Condition: {cond_label!r}  |  Model: {slug}  |  Sessions: {len(all_sessions)}"
    )

    # Detect GPU count for HF providers.
    is_hf = provider_spec.startswith("hf:")
    if is_hf and num_gpus is None:
        try:
            import torch
            num_gpus = max(1, torch.cuda.device_count())
        except ImportError:
            num_gpus = 1

    if not is_hf or num_gpus is None:
        num_gpus = 1

    num_workers = max(1, num_gpus // gpus_per_worker)
    typer.echo(
        f"Provider: {provider_spec}  |  GPUs: {num_gpus}  |  "
        f"Workers: {num_workers}  |  GPUs/worker: {gpus_per_worker}"
    )

    all_gpu_ids = list(range(num_gpus))
    worker_devices: list[str] = [
        ",".join(str(g) for g in all_gpu_ids[i * gpus_per_worker:(i + 1) * gpus_per_worker])
        for i in range(num_workers)
    ]

    # Each worker writes to a temp shard; main process merges them.
    shard_stem = cond_key.replace("/", "_")
    shard_paths = [
        model_output_dir / f".{shard_stem}__{summary_method}__shard{i}.jsonl"
        for i in range(num_workers)
    ]
    chunks = _split_sessions(all_sessions, num_workers)

    if num_workers == 1:
        # The shell already set CUDA_VISIBLE_DEVICES to the correct physical GPUs.
        # Do NOT pass cuda_devices — that would recompute local indices [0,1,...] and
        # override the shell value, putting this worker back on physical GPUs 0,1.
        _run_worker(
            cuda_devices=None,
            sessions=all_sessions,
            condition=cond_key,
            summary_method=summary_method,
            provider_spec=provider_spec,
            shard_path=shard_paths[0],
            device_map="auto" if is_hf else "cpu",
        )
    else:
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []
        for i, (cuda_devs, chunk, shard) in enumerate(
            zip(worker_devices, chunks, shard_paths)
        ):
            if not chunk:
                continue
            p = ctx.Process(
                target=_run_worker,
                args=(cuda_devs, chunk, cond_key, summary_method, provider_spec, shard,
                      "auto" if is_hf else "cpu"),
                name=f"eval-worker{i}",
            )
            p.start()
            processes.append(p)
            typer.echo(f"  Worker {i} (GPUs {cuda_devs}): {len(chunk)} sessions")

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error("Worker %s exited with code %d", p.name, p.exitcode)

    # Merge shards into final output file.
    total = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for shard in shard_paths:
            if shard.exists():
                for line in shard.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out_fh.write(line + "\n")
                        total += 1
                shard.unlink()

    if total == 0:
        logger.error(
            "Merge produced 0 results for %s — all workers may have crashed; check logs/",
            output_path,
        )
        typer.echo(f"WARNING: 0 results written to {output_path} — check worker logs!", err=True)
    else:
        typer.echo(f"Done. {total} results → {output_path}")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_worker(
    cuda_devices: str | None,
    sessions: list[dict],
    condition: str,
    summary_method: str,
    provider_spec: str,
    shard_path: Path,
    device_map: str = "cpu",
) -> None:
    """Load model once, run condition on every session in this chunk.

    ``cuda_devices`` is only set when this function runs in a *spawned* child
    process that needs to select its GPU subset explicitly.  When called from
    the main process (num_workers == 1) it must be ``None`` — the shell has
    already set ``CUDA_VISIBLE_DEVICES`` to the correct physical GPU IDs and
    overriding it here would remap all workers back to GPUs 0,1.
    """
    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        tag = f"GPU{cuda_devices}"
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s %(levelname)s [GPU{cuda_devices}] %(name)s: %(message)s",
        )
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
        tag = f"GPU{visible}"
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s %(levelname)s [{tag}] %(name)s: %(message)s",
        )

    worker_logger = logging.getLogger(__name__)

    from episodic_log.providers import get_provider
    provider = get_provider(provider_spec, device_map=device_map)

    cond = get_condition(condition)

    bar = tqdm(
        sessions,
        desc=f"{tag}/{condition}/{summary_method}",
        unit="session",
        dynamic_ncols=True,
        leave=True,
    )

    failed = 0
    written = 0

    # ── Batch path: vLLM provider with generate_with_tools_batch ────────────
    # For recall/grep_recall conditions, run all sessions simultaneously via
    # step-synchronised batch inference instead of one-by-one.  This keeps the
    # GPU saturated — while one session's tool calls execute on CPU, all other
    # sessions' prompts are in the same vLLM batch.
    _use_batch = (
        hasattr(provider, "generate_with_tools_batch")
        and condition.startswith(("recall/", "grep_recall/"))
    )

    if _use_batch:
        from episodic_log.agent.batch_loop import run_batch
        _mode = "grep_and_load" if condition.startswith("grep_recall/") else "load_only"
        worker_logger.info(
            "%s: batch mode — %d sessions, condition=%s mode=%s",
            tag, len(sessions), condition, _mode,
        )
        try:
            traces = run_batch(
                sessions=sessions,
                summary_method=summary_method,
                mode=_mode,
                provider=provider,
                max_tool_calls=getattr(cond, "_max_tool_calls", 8),
            )
        except Exception as exc:
            worker_logger.error(
                "%s: batch_loop failed: %s", tag, exc, exc_info=True
            )
            traces = []

        with shard_path.open("w", encoding="utf-8") as fh:
            for meta, trace in zip(sessions, traces):
                record = {
                    "session_id": trace.session_id,
                    "question_id": meta.get("question_id", ""),
                    "condition": condition,
                    "summary_method": summary_method,
                    "question": trace.question,
                    "ground_truth": meta.get("answer", ""),
                    "predicted_answer": trace.answer,
                    "tool_calls": [tc.to_dict() for tc in trace.tool_calls],
                    "turns_loaded": trace.turns_loaded,
                    "evidence_turn_ids": meta.get("evidence_turn_ids", []),
                    "question_type": meta.get("question_type", ""),
                    "verdict": None,
                    "confidence": None,
                    "judge_reason": None,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        worker_logger.info(
            "%s: batch done — written=%d  total=%d", tag, written, len(sessions),
        )
        return

    # ── Oneshot batch: amnesiac / topk — single generate_batch call ──────────
    # Conditions that implement run_batch() submit all sessions in one LLM call.
    # No tool loop, no iteration — pure batch inference on the whole session set.
    _use_oneshot_batch = (
        hasattr(cond, "run_batch")
        and hasattr(provider, "generate_batch")
        and not condition.startswith(("recall/", "grep_recall/"))
    )

    if _use_oneshot_batch:
        worker_logger.info(
            "%s: oneshot-batch mode — %d sessions, condition=%s",
            tag, len(sessions), condition,
        )
        try:
            results = cond.run_batch(sessions, provider)
        except Exception as exc:
            worker_logger.error(
                "%s: run_batch failed: %s", tag, exc, exc_info=True,
            )
            results = []

        with shard_path.open("w", encoding="utf-8") as fh:
            for meta, result in zip(sessions, results):
                record = {
                    "session_id": result.session_id,
                    "question_id": result.question_id,
                    "condition": result.condition,
                    "summary_method": result.summary_method,
                    "question": result.question,
                    "ground_truth": result.ground_truth,
                    "predicted_answer": result.predicted_answer,
                    "tool_calls": result.tool_calls,
                    "turns_loaded": result.turns_loaded,
                    "evidence_turn_ids": meta.get("evidence_turn_ids", []),
                    "question_type": meta.get("question_type", ""),
                    "verdict": None,
                    "confidence": None,
                    "judge_reason": None,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        worker_logger.info(
            "%s: oneshot-batch done — written=%d  total=%d", tag, written, len(sessions),
        )
        return

    # ── Sequential path: HuggingFace or single-session vLLM ─────────────────
    with shard_path.open("w", encoding="utf-8") as fh:
        for meta in bar:
            try:
                result = cond.run(session_meta=meta, provider=provider)
            except Exception as exc:
                _is_oom = (
                    "OutOfMemoryError" in type(exc).__name__
                    or "out of memory" in str(exc).lower()
                    or "CUDA out of memory" in str(exc)
                )
                if _is_oom:
                    worker_logger.error(
                        "Session %s OOM — flushing cache and continuing. Error: %s",
                        meta.get("session_id"), exc,
                    )
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    import gc as _gc
                    _gc.collect()
                else:
                    worker_logger.error(
                        "Session %s failed: %s", meta.get("session_id"), exc, exc_info=True
                    )
                failed += 1
                bar.set_postfix({"written": written, "failed": failed})
                continue

            record = {
                "session_id": result.session_id,
                "question_id": result.question_id,
                "condition": result.condition,
                "summary_method": result.summary_method,
                "question": result.question,
                "ground_truth": result.ground_truth,
                "predicted_answer": result.predicted_answer,
                "tool_calls": result.tool_calls,
                "turns_loaded": result.turns_loaded,
                "evidence_turn_ids": meta.get("evidence_turn_ids", []),
                "question_type": meta.get("question_type", ""),
                "verdict": None,
                "confidence": None,
                "judge_reason": None,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            bar.set_postfix({"written": written, "failed": failed})

    worker_logger.info(
        "%s: done — written=%d  failed=%d  total=%d",
        tag, written, failed, len(sessions),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sessions(sessions: list[dict], n: int) -> list[list[dict]]:
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


def _derive_slug(provider_spec: str) -> str:
    parts = provider_spec.split(":")
    if parts[0].lower() in ("groq", "hf", "huggingface"):
        rest = ":".join(parts[1:])
    else:
        rest = provider_spec
    quant_suffix = ""
    if rest.endswith(":4bit"):
        quant_suffix = "-4bit"
        rest = rest[:-5]
    elif rest.endswith(":8bit"):
        quant_suffix = "-8bit"
        rest = rest[:-5]
    model_part = rest.split("/")[-1]
    slug = model_part.lower().replace("_", "-")
    for token in ("-instruct", "-it", "-chat", "-hf"):
        if slug.endswith(token):
            slug = slug[: -len(token)]
        slug = slug.replace(token + "-", "-")
    return slug + quant_suffix


if __name__ == "__main__":
    app()
