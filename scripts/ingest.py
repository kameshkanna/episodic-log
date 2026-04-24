"""Ingest LongMemEval dataset instances into per-session log.jsonl files.

Usage
-----
# Ingest all 500 instances (default)
python scripts/ingest.py

# Quick test — 5 instances
python scripts/ingest.py --n 5

# Custom data directory and seed
python scripts/ingest.py --n 50 --seed 0 --data-dir data/sessions
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episodic_log.ingestor.longmemeval import LongMemEvalIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="ingest",
    help="Ingest LongMemEval into per-session log.jsonl files.",
    add_completion=False,
)


@app.command()
def ingest(
    n: Annotated[
        int | None,
        typer.Option("--n", help="Number of instances (default: all 500)."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Shuffle seed.")] = 42,
    data_dir: Annotated[
        Path,
        typer.Option(help="Root directory for per-session subdirectories."),
    ] = Path("data/sessions"),
    output_index: Annotated[
        Path,
        typer.Option(help="Path to write a JSONL index of ingested sessions."),
    ] = Path("data/sessions_index.jsonl"),
) -> None:
    """Load LongMemEval from HuggingFace and write per-session log.jsonl files."""
    instances = LongMemEvalIngestor.load_dataset(n=n, seed=seed)
    typer.echo(f"Loaded {len(instances)} instances. Ingesting into {data_dir} …")

    ingestor = LongMemEvalIngestor(data_dir=data_dir)
    sessions = ingestor.ingest_batch(instances)

    output_index.parent.mkdir(parents=True, exist_ok=True)
    with output_index.open("w", encoding="utf-8") as fh:
        for s in sessions:
            record = {
                "session_id": s.session_id,
                "question_id": s.question_id,
                "question": s.question,
                "answer": s.answer if isinstance(s.answer, str) else "; ".join(s.answer),
                "evidence_turn_ids": s.evidence_turn_ids,
                "question_type": s.question_type,
                "log_path": str(s.log_path),
                "summaries_dir": str(s.summaries_dir),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    typer.echo(f"Done. {len(sessions)} sessions written. Index: {output_index}")


if __name__ == "__main__":
    app()
