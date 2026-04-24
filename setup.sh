#!/usr/bin/env bash
# Setup script for episodic-log.
# Creates a virtual environment, installs dependencies, and runs a smoke test.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON="${PYTHON:-python3}"

echo "==> episodic-log setup"
echo "    Repo:   ${REPO_ROOT}"
echo "    Python: $("$PYTHON" --version 2>&1)"

# --- GPU info (informational) ---
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "==> GPU info"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi

# --- Create venv ---
if [[ ! -d "$VENV_DIR" ]]; then
    echo ""
    echo "==> Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

echo ""
echo "==> Upgrading pip"
"$VENV_PIP" install --quiet --upgrade pip wheel setuptools

# --- Detect CUDA for PyTorch install ---
echo ""
echo "==> Installing base dependencies"
"$VENV_PIP" install --quiet -e "${REPO_ROOT}[groq]"

# Check if CUDA is available to decide whether to install GPU torch.
CUDA_AVAILABLE=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    CUDA_AVAILABLE=true
fi

if [[ "$CUDA_AVAILABLE" == "true" ]]; then
    echo ""
    echo "==> Installing PyTorch with CUDA support"
    "$VENV_PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cu124
    echo "==> Installing HuggingFace dependencies"
    "$VENV_PIP" install --quiet transformers bitsandbytes accelerate
else
    echo ""
    echo "==> No CUDA detected — installing CPU-only PyTorch (HF local models will run on CPU)"
    "$VENV_PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    "$VENV_PIP" install --quiet transformers accelerate
fi

echo ""
echo "==> Installing dev tools"
"$VENV_PIP" install --quiet pytest pytest-cov ruff

# --- Create data directories ---
echo ""
echo "==> Creating data directories"
mkdir -p "${REPO_ROOT}/data/sessions" \
         "${REPO_ROOT}/data/results" \
         "${REPO_ROOT}/data/submissions"

# --- Smoke test ---
echo ""
echo "==> Running smoke test"
"$VENV_PYTHON" - <<'PYEOF'
import sys
sys.path.insert(0, ".")

# Core
from episodic_log.core.turn_event import TurnEvent, EventRole, EventType
from episodic_log.core.turn_summary import TurnSummary
from episodic_log.core.log_writer import LogWriter
from episodic_log.core.log_reader import LogReader, TurnLoader
from episodic_log.retrieval.bm25_index import BM25Index
from episodic_log.retrieval.summary_store import SummaryStore
from episodic_log.conditions import ALL_CONDITIONS, get_condition
from episodic_log.judge import CHDJudge, JudgeVerdict
from episodic_log.metrics import CHDMetrics, compute_metrics
from episodic_log.providers import BaseProvider, GroqProvider, HuggingFaceProvider, get_provider
print("  All imports OK.")
print(f"  Conditions available: {ALL_CONDITIONS}")
PYEOF

echo ""
echo "==> Setup complete."
echo ""
echo "Activate the environment:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "Quick start:"
echo "    python scripts/ingest.py --n 5"
echo "    python scripts/summarize.py --method structured"
echo "    python scripts/evaluate.py --condition baseline --n 5 --provider groq:llama-3.1-8b-instant"
echo "    python scripts/score.py"
