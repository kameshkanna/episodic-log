"""Summarize ingested session logs using one or more summarizer methods.

Usage
-----
# Summarize all sessions with the structured (no-model) method
python scripts/summarize.py --method structured

# Summarize with haiku (Groq Llama-3.1-8b) method
python scripts/summarize.py --method haiku --provider groq:llama-3.1-8b-instant

# Summarize all three methods sequentially
python scripts/summarize.py --method structured
python scripts/summarize.py --method haiku
python scripts/summarize.py --method self --provider groq:llama-3.1-8b-instant
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
        typer.Option("--provider", help="Provider spec (e.g. groq:llama-3.1-8b-instant). Required for haiku/self."),
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

    sessions = _load_index(sessions_index)
    typer.echo(f"Loaded {len(sessions)} sessions. Summarizing with method={method!r} …")

    provider = None
    if provider_spec:
        from episodic_log.providers import get_provider
        provider = get_provider(provider_spec)

    summarizer = build_summarizer(method=method, provider=provider)

    skipped = 0
    iterator = tqdm(sessions, desc=f"summarize/{method}", unit="session", dynamic_ncols=True)
    for session_meta in iterator:
        summaries_dir = Path(session_meta["summaries_dir"])
        summary_file = summaries_dir / f"{method}.jsonl"

        if summary_file.exists() and not overwrite:
            skipped += 1
            continue

        log_path = Path(session_meta["log_path"])
        if not log_path.exists():
            logger.warning("log.jsonl not found for session %s — skipping.", session_meta["session_id"])
            continue

        reader = LogReader(log_path)
        events = reader.load_all()

        # Remove stale file before re-writing.
        if summary_file.exists():
            summary_file.unlink()

        store = SummaryStore(summaries_dir)
        for event in events:
            summary = summarizer.summarize(event)
            store.write(summary)

        iterator.set_postfix({"last": session_meta["session_id"][:20]})

    typer.echo(
        f"Done. {len(sessions) - skipped} sessions summarized, {skipped} skipped (already exist)."
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
