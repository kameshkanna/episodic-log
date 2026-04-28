#!/usr/bin/env bash
# Full CHD evaluation pipeline — ingest → summarize → evaluate → judge → score
# Optimised for 8x H100 SXM5. Total wall time: ~40 min from scratch.
# 10 conditions in 2 waves: Wave 1 (8 conditions × 1 GPU, all parallel),
# Wave 2 (topk/scout/k5 + topk/echo/k5).
# Each step is skipped automatically if its output already exists.
#
# Usage:
#   bash run_pipeline.sh              # skip any already-completed steps
#   FORCE_INGEST=1 bash run_pipeline.sh   # force re-ingest even if data exists
#   FORCE_SUMMARIZE=1 bash run_pipeline.sh  # force re-summarize
#   FORCE_EVAL=1 bash run_pipeline.sh       # force re-evaluate (overwrites results)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

JUDGE_MODEL="${JUDGE_MODEL:-vllm:Qwen/Qwen2.5-14B-Instruct:tp8}"
# All eval conditions use tp=8 (all 8 GPUs).
# On 8× A100 40GB: tp=8 puts 8 GB of weights per GPU, leaving 32 GB for KV
# cache + NCCL + activations.  tp=4 left only 24 GB — not enough headroom.
EVAL_MODEL="${EVAL_MODEL:-vllm:Qwen/Qwen3-32B:tp8}"
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
# STEP 1: Ingest — skip if sessions_index.jsonl already exists
# ---------------------------------------------------------------------------
if [[ -f data/sessions_index.jsonl && -z "${FORCE_INGEST:-}" ]]; then
    SESSION_COUNT=$(wc -l < data/sessions_index.jsonl | tr -d ' ')
    log "=== STEP 1: Ingest — SKIPPED ($SESSION_COUNT sessions already in data/sessions_index.jsonl) ==="
else
    log "=== STEP 1: Ingest ==="
    python scripts/ingest.py 2>&1 | tee "$LOG_DIR/ingest.log"
fi

# ---------------------------------------------------------------------------
# STEP 2a: Lexical summaries — skip if all sessions already have lexical.jsonl
# ---------------------------------------------------------------------------
LEXICAL_DONE=$(find data/sessions -name "lexical.jsonl" -size +0c 2>/dev/null | wc -l | tr -d ' ')
TOTAL_SESSIONS=$(wc -l < data/sessions_index.jsonl | tr -d ' ')

if [[ "$LEXICAL_DONE" -ge "$TOTAL_SESSIONS" && -z "${FORCE_SUMMARIZE:-}" ]]; then
    log "=== STEP 2a: Summarize (lexical) — SKIPPED ($LEXICAL_DONE/$TOTAL_SESSIONS sessions done) ==="
else
    log "=== STEP 2a: Summarize (lexical, CPU) ==="
    python scripts/summarize.py --method lexical 2>&1 | tee "$LOG_DIR/summarize_lexical.log"
fi

# ---------------------------------------------------------------------------
# STEP 2b: Scout + Echo — skip if all sessions already have both files
# ---------------------------------------------------------------------------
SCOUT_DONE=$(find data/sessions -name "scout.jsonl" -size +0c 2>/dev/null | wc -l | tr -d ' ')
ECHO_DONE=$(find data/sessions  -name "echo.jsonl"  -size +0c 2>/dev/null | wc -l | tr -d ' ')

if [[ "$SCOUT_DONE" -ge "$TOTAL_SESSIONS" && "$ECHO_DONE" -ge "$TOTAL_SESSIONS" && -z "${FORCE_SUMMARIZE:-}" ]]; then
    log "=== STEP 2b: Summarize (scout/echo) — SKIPPED (scout=$SCOUT_DONE echo=$ECHO_DONE / $TOTAL_SESSIONS) ==="
else
    log "=== STEP 2b: Summarize scout (GPUs 0-3 vllm:tp4) + echo (GPUs 4-7 vllm:tp4) ==="

    if [[ "$SCOUT_DONE" -lt "$TOTAL_SESSIONS" || -n "${FORCE_SUMMARIZE:-}" ]]; then
        CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/summarize.py \
            --method scout \
            --provider "vllm:Qwen/Qwen2.5-7B-Instruct:tp4" \
            2>&1 | tee "$LOG_DIR/summarize_scout.log" &
        SCOUT_PID=$!
    else
        log "  scout — already done, skipping"
        SCOUT_PID=""
    fi

    if [[ "$ECHO_DONE" -lt "$TOTAL_SESSIONS" || -n "${FORCE_SUMMARIZE:-}" ]]; then
        CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/summarize.py \
            --method echo \
            --provider "vllm:Qwen/Qwen2.5-7B-Instruct:tp4" \
            2>&1 | tee "$LOG_DIR/summarize_echo.log" &
        ECHO_PID=$!
    else
        log "  echo — already done, skipping"
        ECHO_PID=""
    fi

    [[ -n "$SCOUT_PID" ]] && { wait $SCOUT_PID || { log "ERROR: scout summarizer failed"; exit 1; }; }
    [[ -n "$ECHO_PID"  ]] && { wait $ECHO_PID  || { log "ERROR: echo summarizer failed";  exit 1; }; }
fi
log "Summarization complete."

# ---------------------------------------------------------------------------
# STEP 3: Evaluate — 7 conditions, all tp=8 (all 8 GPUs per condition).
#         On 8× A100 40GB: tp=8 leaves 32 GB/GPU free vs only 24 GB at tp=4.
#         Each condition submits ALL 500 sessions as one huge batch.
# ---------------------------------------------------------------------------
log "=== STEP 3: Evaluate (7 conditions × tp=8, serial) ==="

OVERWRITE_FLAG=""
[[ -n "${FORCE_EVAL:-}" ]] && OVERWRITE_FLAG="--overwrite"

_eval() {
    local label="$1"; shift
    log "--- $label ---"
    python scripts/evaluate.py "$@" \
        --provider "$EVAL_MODEL" --num-gpus 8 --gpus-per-worker 8 $OVERWRITE_FLAG \
        2>&1 | tee "$LOG_DIR/eval_${label}.log"
    local rc=$?
    [[ $rc -ne 0 ]] && { log "ERROR: $label failed (exit $rc)"; exit $rc; }
}

_eval "amnesiac"          --condition amnesiac
_eval "recall_lexical"    --condition recall      --summary-method lexical
_eval "recall_scout"      --condition recall      --summary-method scout
_eval "recall_echo"       --condition recall      --summary-method echo
_eval "grep_recall_lexical" --condition grep_recall --summary-method lexical
_eval "grep_recall_scout"   --condition grep_recall --summary-method scout
_eval "grep_recall_echo"    --condition grep_recall --summary-method echo

log "Evaluation complete."

# ---------------------------------------------------------------------------
# STEP 4: Judge — vLLM mega-batch, all 8 H100s via tp=8 (~5 min)
# ---------------------------------------------------------------------------
log "=== STEP 4: Judge ==="
python scripts/judge.py \
    --judge-provider "$JUDGE_MODEL" \
    2>&1 | tee "$LOG_DIR/judge.log"
log "Judging complete."

# ---------------------------------------------------------------------------
# STEP 5: Score
# ---------------------------------------------------------------------------
log "=== STEP 5: Score ==="
python scripts/score.py 2>&1 | tee "$LOG_DIR/score.log"

log "=== ALL DONE. Results in data/results/ — logs in $LOG_DIR/ ==="
