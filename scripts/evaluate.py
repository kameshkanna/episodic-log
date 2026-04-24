"""Run evaluation conditions on ingested LongMemEval sessions.

Usage
-----
# Baseline (no memory) — 5 sessions, Groq provider
python scripts/evaluate.py --condition baseline --n 5 --provider groq:llama-3.1-8b-instant

# Episodic with structured summaries
python scripts/evaluate.py --condition episodic --summary-method structured \
    --provider groq:llama-3.1-8b-instant

# Full run — all sessions
python scripts/evaluate.py --condition episodic --provider groq:llama-3.1-8b-instant

# With CHD judging (uses a separate judge provider)
python scripts/evaluate.py --condition episodic --judge --judge-provider groq:llama-3.1-70b-versatile
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
            help=f"Condition name: {', '.join(ALL_CONDITIONS)}.",
            case_sensitive=False,
        ),
    ] = "baseline",
    provider_spec: Annotated[
        str,
        typer.Option("--provider", help="Provider spec (e.g. groq:llama-3.1-8b-instant)."),
    ] = "groq:llama-3.1-8b-instant",
    summary_method: Annotated[
        str,
        typer.Option("--summary-method", help="Summary method for retrieval conditions."),
    ] = "structured",
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
        typer.Option(help="Directory for results JSONL output."),
    ] = Path("data/results"),
    judge: Annotated[
        bool,
        typer.Option("--judge/--no-judge", help="Run CHD judge on each prediction."),
    ] = False,
    judge_provider_spec: Annotated[
        str,
        typer.Option("--judge-provider", help="Provider spec for the CHD judge."),
    ] = "groq:llama-3.1-70b-versatile",
) -> None:
    """Run *condition* on all sessions and write per-result JSONL."""
    if condition not in ALL_CONDITIONS:
        typer.echo(f"ERROR: Unknown condition '{condition}'. Available: {ALL_CONDITIONS}", err=True)
        raise typer.Exit(1)

    if not sessions_index.exists():
        typer.echo(f"ERROR: sessions index not found at {sessions_index}. Run ingest.py first.", err=True)
        raise typer.Exit(1)

    from episodic_log.providers import get_provider
    from episodic_log.ingestor.longmemeval import IngestedSession

    provider = get_provider(provider_spec)
    cond = get_condition(condition, provider=provider, summary_method=summary_method)

    judge_obj = None
    if judge:
        from episodic_log.judge import CHDJudge
        judge_provider = get_provider(judge_provider_spec)
        judge_obj = CHDJudge(provider=judge_provider)

    sessions_meta = _load_index(sessions_index)
    if n is not None:
        sessions_meta = sessions_meta[:n]
    typer.echo(f"Running condition={condition!r} on {len(sessions_meta)} sessions …")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{condition}__{summary_method}.jsonl"

    failed = 0
    iterator = tqdm(sessions_meta, desc=f"{condition}/{summary_method}", unit="session", dynamic_ncols=True)

    with output_path.open("w", encoding="utf-8") as out_fh:
        for meta in iterator:
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

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            iterator.set_postfix({"last": session.session_id[:20], "failed": failed})

    typer.echo(
        f"Done. Results written to {output_path}  (failed={failed})"
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
