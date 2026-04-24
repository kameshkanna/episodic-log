"""Tests for the three bug-fixes:
1. summarize._run_sessions deletes the output file when all events fail.
2. run_sweep._preflight_check_summaries aborts on missing files.
3. HuggingFaceProvider.generate passes attention_mask and doesn't add
   sampling kwargs when do_sample=False.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(tmp_path: Path, session_id: str, *, write_haiku: bool = False) -> dict:
    """Create a minimal session directory and return a sessions_index record."""
    sd = tmp_path / session_id
    sd.mkdir(parents=True)
    summaries = sd / "summaries"
    summaries.mkdir()
    log = sd / "log.jsonl"
    log.write_text("")  # empty log

    if write_haiku:
        (summaries / "haiku.jsonl").write_text("")  # intentionally empty

    return {
        "session_id": session_id,
        "log_path": str(log),
        "summaries_dir": str(summaries),
        "question": "test?",
        "answer": "ans",
        "evidence_turn_ids": [],
        "question_type": "single-session-user",
        "question_id": session_id,
    }


# ---------------------------------------------------------------------------
# 1. summarize._run_sessions — empty-file cleanup on all-event failure
# ---------------------------------------------------------------------------

def test_summarize_deletes_file_when_all_events_fail(tmp_path: Path) -> None:
    """If every event raises during summarize, the output JSONL is deleted."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.summarize import _run_sessions  # type: ignore[import]

    session = _make_session(tmp_path, "sess_fail")
    # Write a real log with one turn.
    from episodic_log.core.turn_event import EventRole, EventType, TurnEvent
    from datetime import datetime, timezone
    event = TurnEvent(
        turn_id="0000",
        session_id="sess_fail",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        role=EventRole.USER,
        type=EventType.MESSAGE,
        content="hello",
        raw={},
        tool_name=None,
        file_path=None,
    )
    log_path = Path(session["log_path"])
    log_path.write_text(event.to_json() + "\n")

    # Summarizer that always raises.
    bad_provider = MagicMock()
    bad_provider.generate.side_effect = RuntimeError("boom")

    with patch("scripts.summarize.build_summarizer") as mock_build:
        from episodic_log.summarizers.haiku import HaikuSummarizer
        mock_summarizer = MagicMock(spec=HaikuSummarizer)
        mock_summarizer.summarize.side_effect = RuntimeError("boom")
        mock_build.return_value = mock_summarizer

        _run_sessions(
            cuda_devices=None,
            sessions=[session],
            method="haiku",
            provider_spec=None,
            overwrite=False,
        )

    # File must NOT exist — it was created then deleted.
    summary_file = Path(session["summaries_dir"]) / "haiku.jsonl"
    assert not summary_file.exists(), "Empty summary file should be deleted after all-event failure"


def test_summarize_keeps_partial_file_on_some_failures(tmp_path: Path) -> None:
    """If only some events fail, the file stays (partial is better than nothing)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.summarize import _run_sessions

    session = _make_session(tmp_path, "sess_partial")
    from episodic_log.core.turn_event import EventRole, EventType, TurnEvent
    from episodic_log.core.turn_summary import TurnSummary
    from datetime import datetime, timezone

    events = [
        TurnEvent("0000", "sess_partial", datetime(2024, 1, 1, tzinfo=timezone.utc),
                  EventRole.USER, EventType.MESSAGE, "hello", {}, None, None),
        TurnEvent("0001", "sess_partial", datetime(2024, 1, 1, tzinfo=timezone.utc),
                  EventRole.ASSISTANT, EventType.MESSAGE, "world", {}, None, None),
    ]
    log_path = Path(session["log_path"])
    log_path.write_text("\n".join(e.to_json() for e in events) + "\n")

    call_count = 0

    def _sometimes_fail(event: TurnEvent) -> TurnSummary:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("second event fails")
        return TurnSummary(
            turn_id=event.turn_id,
            session_id=event.session_id,
            summary="ok",
            method="haiku",
        )

    with patch("scripts.summarize.build_summarizer") as mock_build:
        mock_summarizer = MagicMock()
        mock_summarizer.summarize.side_effect = _sometimes_fail
        mock_build.return_value = mock_summarizer

        _run_sessions(
            cuda_devices=None,
            sessions=[session],
            method="haiku",
            provider_spec=None,
            overwrite=False,
        )

    summary_file = Path(session["summaries_dir"]) / "haiku.jsonl"
    assert summary_file.exists(), "Partial summary file should be kept"
    lines = [l for l in summary_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 1, "One line for the one successful event"


# ---------------------------------------------------------------------------
# 2. run_sweep._preflight_check_summaries
# ---------------------------------------------------------------------------

def test_preflight_passes_when_all_files_exist(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.run_sweep import _preflight_check_summaries  # type: ignore[import]

    sessions = []
    for i in range(3):
        s = _make_session(tmp_path, f"sess_{i}", write_haiku=True)
        (Path(s["summaries_dir"]) / "haiku.jsonl").write_text('{"turn_id":"0000"}\n')
        sessions.append(s)

    # Should not raise.
    _preflight_check_summaries(sessions, ["episodic"], ["haiku"])


def test_preflight_aborts_when_files_missing(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.run_sweep import _preflight_check_summaries
    import typer

    sessions = [_make_session(tmp_path, f"sess_{i}") for i in range(3)]
    # No haiku.jsonl files created.

    # typer.Exit is not a SystemExit subclass — it's caught by the typer
    # runner but propagates as typer.Exit when called directly.
    with pytest.raises(typer.Exit):
        _preflight_check_summaries(sessions, ["episodic"], ["haiku"])


def test_preflight_skips_baseline_and_full_context(tmp_path: Path) -> None:
    """Conditions that don't use BM25 should never trigger the pre-flight check."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.run_sweep import _preflight_check_summaries

    sessions = [_make_session(tmp_path, f"sess_{i}") for i in range(3)]
    # No summary files, but conditions don't need them.

    # Should not raise.
    _preflight_check_summaries(sessions, ["baseline", "full_context", "md_memory"], ["haiku"])


# ---------------------------------------------------------------------------
# 3. HuggingFaceProvider — attention_mask and greedy decoding kwargs
# ---------------------------------------------------------------------------

def test_hf_provider_passes_attention_mask() -> None:
    """generate() must include attention_mask in gen_kwargs when tokenizer returns it."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import torch
    from episodic_log.providers.huggingface_provider import HuggingFaceProvider

    # Build a provider without loading a real model.
    provider = object.__new__(HuggingFaceProvider)

    fake_ids = torch.ones(1, 5, dtype=torch.long)
    fake_mask = torch.ones(1, 5, dtype=torch.long)
    fake_output = torch.ones(1, 8, dtype=torch.long)

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "PROMPT"
    mock_tokenizer.return_value = {"input_ids": fake_ids, "attention_mask": fake_mask}
    mock_tokenizer.eos_token_id = 2

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    mock_model.generate.return_value = fake_output
    mock_model.generation_config = MagicMock()

    mock_tokenizer.decode.return_value = "answer"

    provider._tokenizer = mock_tokenizer  # type: ignore[attr-defined]
    provider._model = mock_model          # type: ignore[attr-defined]
    provider._model_name = "test-model"

    provider.generate(messages=["hello"], system=None, max_tokens=32, temperature=0.0)

    call_kwargs = mock_model.generate.call_args[1]
    assert "attention_mask" in call_kwargs, "attention_mask must be passed to model.generate()"
    assert "temperature" not in call_kwargs, "temperature must NOT be in gen_kwargs for greedy decoding"
    assert call_kwargs["do_sample"] is False


def test_hf_provider_passes_temperature_when_sampling() -> None:
    """temperature must be included in gen_kwargs when do_sample=True."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import torch
    from episodic_log.providers.huggingface_provider import HuggingFaceProvider

    provider = object.__new__(HuggingFaceProvider)

    fake_ids = torch.ones(1, 5, dtype=torch.long)
    fake_output = torch.ones(1, 8, dtype=torch.long)

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "PROMPT"
    mock_tokenizer.return_value = {"input_ids": fake_ids}
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode.return_value = "answer"

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    mock_model.generate.return_value = fake_output
    mock_model.generation_config = MagicMock()

    provider._tokenizer = mock_tokenizer  # type: ignore[attr-defined]
    provider._model = mock_model          # type: ignore[attr-defined]
    provider._model_name = "test-model"

    provider.generate(messages=["hello"], system=None, max_tokens=32, temperature=0.7)

    call_kwargs = mock_model.generate.call_args[1]
    assert call_kwargs["do_sample"] is True
    assert call_kwargs["temperature"] == pytest.approx(0.7)
