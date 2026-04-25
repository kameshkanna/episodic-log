#!/usr/bin/env bash
# Full CHD evaluation pipeline — ingest → summarize → evaluate → judge → score
# Optimised for 8x H100 SXM5 with HF provider. Total wall time: ~60 min.
#
# Usage:
#   bash run_pipeline.sh
#   MODEL=hf:Qwen/Qwen2.5-7B-Instruct bash run_pipeline.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MODEL="${MODEL:-vllm:Qwen/Qwen2.5-7B-Instruct:tp4}"
JUDGE_MODEL="${JUDGE_MODEL:-vllm:Qwen/Qwen2.5-14B-Instruct:tp4}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ---------------------------------------------------------------------------
# STEP 0: Pull latest code
# ---------------------------------------------------------------------------
log "=== STEP 0: git pull ==="
git pull

# ---------------------------------------------------------------------------
# STEP 1: Ingest all 500 LongMemEval sessions (CPU, ~5 min)
# ---------------------------------------------------------------------------
log "=== STEP 1: Ingest ==="
rm -rf data/sessions data/sessions_index.jsonl
python scripts/ingest.py 2>&1 | tee "$LOG_DIR/ingest.log"

# ---------------------------------------------------------------------------
# STEP 2a: Lexical summarizer — CPU, no model needed (~2 min)
# ---------------------------------------------------------------------------
log "=== STEP 2a: Summarize (lexical, CPU) ==="
python scripts/summarize.py --method lexical 2>&1 | tee "$LOG_DIR/summarize_lexical.log"

# ---------------------------------------------------------------------------
# STEP 2b: Scout (GPUs 0-3) + Echo (GPUs 4-7) in parallel — 4 workers each
#          HF provider, batch_size=32 on H100 80GB → ~15 min total
# ---------------------------------------------------------------------------
log "=== STEP 2b: Summarize scout (GPUs 0-3) + echo (GPUs 4-7) in parallel ==="

HF_MODEL="hf:Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/summarize.py \
    --method scout \
    --provider "$HF_MODEL" \
    --num-gpus 4 2>&1 | tee "$LOG_DIR/summarize_scout.log" &
SCOUT_PID=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/summarize.py \
    --method echo \
    --provider "$HF_MODEL" \
    --num-gpus 4 2>&1 | tee "$LOG_DIR/summarize_echo.log" &
ECHO_PID=$!

wait $SCOUT_PID || { log "ERROR: scout summarizer failed"; exit 1; }
wait $ECHO_PID  || { log "ERROR: echo summarizer failed"; exit 1; }
log "Summarization complete."

# ---------------------------------------------------------------------------
# STEP 3: Evaluate — HF provider (tool-calling needs HF, not vLLM)
#         4 conditions × 2 GPUs each, all parallel (~30 min)
# ---------------------------------------------------------------------------
log "=== STEP 3: Evaluate (4 conditions × 2 GPUs, hf provider) ==="

HF_MODEL="hf:Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py \
    --condition amnesiac \
    --provider "$HF_MODEL" \
    --num-gpus 2 2>&1 | tee "$LOG_DIR/eval_amnesiac.log" &
EVAL0_PID=$!

CUDA_VISIBLE_DEVICES=2,3 python scripts/evaluate.py \
    --condition recall --summary-method lexical \
    --provider "$HF_MODEL" \
    --num-gpus 2 2>&1 | tee "$LOG_DIR/eval_recall_lexical.log" &
EVAL1_PID=$!

CUDA_VISIBLE_DEVICES=4,5 python scripts/evaluate.py \
    --condition recall --summary-method scout \
    --provider "$HF_MODEL" \
    --num-gpus 2 2>&1 | tee "$LOG_DIR/eval_recall_scout.log" &
EVAL2_PID=$!

CUDA_VISIBLE_DEVICES=6,7 python scripts/evaluate.py \
    --condition recall --summary-method echo \
    --provider "$HF_MODEL" \
    --num-gpus 2 2>&1 | tee "$LOG_DIR/eval_recall_echo.log" &
EVAL3_PID=$!

wait $EVAL0_PID || { log "ERROR: amnesiac eval failed"; exit 1; }
wait $EVAL1_PID || { log "ERROR: recall/lexical eval failed"; exit 1; }
wait $EVAL2_PID || { log "ERROR: recall/scout eval failed"; exit 1; }
wait $EVAL3_PID || { log "ERROR: recall/echo eval failed"; exit 1; }
log "Evaluation complete."

# ---------------------------------------------------------------------------
# STEP 4: Judge — HF 14B across all 8 GPUs (~20 min)
# ---------------------------------------------------------------------------
log "=== STEP 4: Judge ==="
python scripts/judge.py \
    --judge-provider "hf:Qwen/Qwen2.5-14B-Instruct" \
    --num-gpus 8 2>&1 | tee "$LOG_DIR/judge.log"
log "Judging complete."

# ---------------------------------------------------------------------------
# STEP 5: Score
# ---------------------------------------------------------------------------
log "=== STEP 5: Score ==="
python scripts/score.py 2>&1 | tee "$LOG_DIR/score.log"

log "=== ALL DONE. Results in data/results/ — logs in $LOG_DIR/ ==="
