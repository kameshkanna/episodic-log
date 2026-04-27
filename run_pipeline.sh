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
# vLLM tp1: Qwen3-32B fits in one 80GB H100 (64GB weights + ~12GB KV cache).
# 8 independent single-GPU workers instead of 4 two-GPU workers → ~3-4x faster.
EVAL_MODEL="${EVAL_MODEL:-vllm:Qwen/Qwen3-32B:tp1}"
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
# STEP 3: Evaluate — 10 conditions in 2 waves, 1 GPU/worker (vLLM tp1)
#         Qwen3-32B fits in a single 80GB H100 → 8 independent workers.
#         Wave 1: 8 conditions × 1 GPU, all in parallel.
#         Wave 2: 2 remaining topk conditions × 1 GPU.
# ---------------------------------------------------------------------------
log "=== STEP 3: Evaluate (2 waves, 1 GPU/worker × 8 workers, $EVAL_MODEL) ==="

OVERWRITE_FLAG=""
[[ -n "${FORCE_EVAL:-}" ]] && OVERWRITE_FLAG="--overwrite"

# Wave 1: 8 conditions in parallel — 1 GPU each, all 8 H100s fully utilised.
log "--- Wave 1: amnesiac + recall×3 + grep_recall×3 + topk/lexical/k5 (8 parallel) ---"

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --condition amnesiac \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_amnesiac.log" &
EVAL0_PID=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
    --condition recall --summary-method lexical \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_recall_lexical.log" &
EVAL1_PID=$!

CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --condition recall --summary-method scout \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_recall_scout.log" &
EVAL2_PID=$!

CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py \
    --condition recall --summary-method echo \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_recall_echo.log" &
EVAL3_PID=$!

CUDA_VISIBLE_DEVICES=4 python scripts/evaluate.py \
    --condition grep_recall --summary-method lexical \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_grep_recall_lexical.log" &
EVAL4_PID=$!

CUDA_VISIBLE_DEVICES=5 python scripts/evaluate.py \
    --condition grep_recall --summary-method scout \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_grep_recall_scout.log" &
EVAL5_PID=$!

CUDA_VISIBLE_DEVICES=6 python scripts/evaluate.py \
    --condition grep_recall --summary-method echo \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_grep_recall_echo.log" &
EVAL6_PID=$!

CUDA_VISIBLE_DEVICES=7 python scripts/evaluate.py \
    --condition topk --summary-method lexical --retrieval-k 5 \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_topk_lexical_k5.log" &
EVAL7_PID=$!

wait $EVAL0_PID || { log "ERROR: amnesiac eval failed";         exit 1; }
wait $EVAL1_PID || { log "ERROR: recall/lexical eval failed";   exit 1; }
wait $EVAL2_PID || { log "ERROR: recall/scout eval failed";     exit 1; }
wait $EVAL3_PID || { log "ERROR: recall/echo eval failed";      exit 1; }
wait $EVAL4_PID || { log "ERROR: grep_recall/lexical failed";   exit 1; }
wait $EVAL5_PID || { log "ERROR: grep_recall/scout failed";     exit 1; }
wait $EVAL6_PID || { log "ERROR: grep_recall/echo failed";      exit 1; }
wait $EVAL7_PID || { log "ERROR: topk/lexical/k5 failed";       exit 1; }
log "Wave 1 complete."

# Wave 2: remaining topk variants (2 GPUs free after wave 1)
log "--- Wave 2: topk/scout/k5 + topk/echo/k5 ---"

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --condition topk --summary-method scout --retrieval-k 5 \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_topk_scout_k5.log" &
EVAL8_PID=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
    --condition topk --summary-method echo --retrieval-k 5 \
    --provider "$EVAL_MODEL" \
    --num-gpus 1 --gpus-per-worker 1 $OVERWRITE_FLAG \
    2>&1 | tee "$LOG_DIR/eval_topk_echo_k5.log" &
EVAL9_PID=$!

wait $EVAL8_PID || { log "ERROR: topk/scout/k5 failed"; exit 1; }
wait $EVAL9_PID || { log "ERROR: topk/echo/k5 failed";  exit 1; }
log "Wave 2 complete."
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
