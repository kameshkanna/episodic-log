#!/usr/bin/env bash
# Full CHD evaluation pipeline — ingest → summarize → evaluate → judge → score
# Run on a 4x H100 SXM5 node. Total wall time: ~2.5 hours.
#
# Usage:
#   bash run_pipeline.sh
#   bash run_pipeline.sh --model hf:Qwen/Qwen2.5-7B-Instruct
#   bash run_pipeline.sh --judge hf:Qwen/Qwen2.5-14B-Instruct

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MODEL="${MODEL:-hf:Qwen/Qwen2.5-7B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-hf:Qwen/Qwen2.5-14B-Instruct}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
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
# STEP 2b: Scout + Echo in parallel — 2 GPUs each (~45 min)
# ---------------------------------------------------------------------------
log "=== STEP 2b: Summarize scout (GPUs 0,1) + echo (GPUs 2,3) in parallel ==="

CUDA_VISIBLE_DEVICES=0,1 python scripts/summarize.py \
    --method scout \
    --provider "$MODEL" \
    --num-gpus 2 2>&1 | tee "$LOG_DIR/summarize_scout.log" &
SCOUT_PID=$!

CUDA_VISIBLE_DEVICES=2,3 python scripts/summarize.py \
    --method echo \
    --provider "$MODEL" \
    --num-gpus 2 2>&1 | tee "$LOG_DIR/summarize_echo.log" &
ECHO_PID=$!

wait $SCOUT_PID || { log "ERROR: scout summarizer failed"; exit 1; }
wait $ECHO_PID  || { log "ERROR: echo summarizer failed"; exit 1; }
log "Summarization complete."

# ---------------------------------------------------------------------------
# STEP 3: Evaluate 4 conditions in parallel — 1 GPU each (~60 min)
# ---------------------------------------------------------------------------
log "=== STEP 3: Evaluate (4 conditions × 1 GPU) ==="

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --condition amnesiac \
    --provider "$MODEL" \
    --num-gpus 1 2>&1 | tee "$LOG_DIR/eval_amnesiac.log" &
EVAL0_PID=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
    --condition recall --summary-method lexical \
    --provider "$MODEL" \
    --num-gpus 1 2>&1 | tee "$LOG_DIR/eval_recall_lexical.log" &
EVAL1_PID=$!

CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --condition recall --summary-method scout \
    --provider "$MODEL" \
    --num-gpus 1 2>&1 | tee "$LOG_DIR/eval_recall_scout.log" &
EVAL2_PID=$!

CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py \
    --condition recall --summary-method echo \
    --provider "$MODEL" \
    --num-gpus 1 2>&1 | tee "$LOG_DIR/eval_recall_echo.log" &
EVAL3_PID=$!

wait $EVAL0_PID || { log "ERROR: amnesiac eval failed"; exit 1; }
wait $EVAL1_PID || { log "ERROR: recall/lexical eval failed"; exit 1; }
wait $EVAL2_PID || { log "ERROR: recall/scout eval failed"; exit 1; }
wait $EVAL3_PID || { log "ERROR: recall/echo eval failed"; exit 1; }
log "Evaluation complete."

# ---------------------------------------------------------------------------
# STEP 4: Judge — CHD verdict for every prediction (~30 min on 4 GPUs)
# ---------------------------------------------------------------------------
log "=== STEP 4: Judge ==="
python scripts/judge.py \
    --judge-provider "$JUDGE_MODEL" \
    --num-gpus 4 2>&1 | tee "$LOG_DIR/judge.log"
log "Judging complete."

# ---------------------------------------------------------------------------
# STEP 5: Score — print CHD metrics table
# ---------------------------------------------------------------------------
log "=== STEP 5: Score ==="
python scripts/score.py 2>&1 | tee "$LOG_DIR/score.log"

log "=== ALL DONE. Results in data/results/ — logs in $LOG_DIR/ ==="
