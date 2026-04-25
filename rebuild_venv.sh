#!/usr/bin/env bash
# Rebuild .venv from scratch — clean of system-site-packages.
# Install vllm first so pip resolves a numpy/scipy/transformers stack
# that is mutually compatible. Then install the project.
#
# Usage:
#   bash rebuild_venv.sh
#
# Time: ~5-10 min on a server with fast internet.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

echo "==> Removing old venv (if any)"
deactivate 2>/dev/null || true
rm -rf "$VENV_DIR"

echo "==> Creating fresh venv (no system-site-packages)"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip/wheel"
pip install --quiet --upgrade pip wheel setuptools

# Install vllm FIRST — it locks numpy, scipy, transformers to a mutually
# compatible set. Installing it after other packages risks ABI conflicts.
echo "==> Installing vllm (pins numpy + transformers compatible versions)"
pip install vllm

# Install PyTorch for CUDA 12.4 — vllm already pulled in a torch but
# we re-pin to the official CUDA 12.4 wheel for HF provider tool-calling.
echo "==> Installing PyTorch cu124"
pip install torch --index-url https://download.pytorch.org/whl/cu124 --upgrade

# Install the project and remaining deps.
echo "==> Installing episodic-log[hf,groq]"
pip install -e "${REPO_ROOT}[hf,groq]"

# Extras needed at runtime.
echo "==> Installing runtime extras"
pip install huggingface_hub bitsandbytes accelerate rank-bm25 tqdm typer rich

echo "==> Installing dev tools"
pip install pytest ruff

echo ""
echo "==> Smoke test"
python - <<'PYEOF'
import sys
sys.path.insert(0, ".")
import torch
print(f"  torch={torch.__version__}  CUDA={torch.cuda.is_available()}  GPUs={torch.cuda.device_count()}")
import numpy, scipy, transformers
print(f"  numpy={numpy.__version__}  scipy={scipy.__version__}  transformers={transformers.__version__}")
from vllm import LLM
print("  vllm import OK")
from episodic_log.providers import get_provider
print("  episodic_log import OK")
PYEOF

echo ""
echo "==> Rebuild complete. Activate with:"
echo "    source .venv/bin/activate"
echo ""
echo "Then run:"
echo "    bash run_pipeline.sh"
