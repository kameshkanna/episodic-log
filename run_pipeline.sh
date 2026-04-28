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
# Two conditions run in parallel, each on 4 GPUs (tp=4).
# GPUs 0-3 → first condition, GPUs 4-7 → second condition.
# Total throughput = same GPU-hours as tp=8 serial, but half the wall time.
EVAL_MODEL_A="${EVAL_MODEL_A:-vllm:Qwen/Qwen3-32B:tp4}"   # GPUs 0-3
EVAL_MODEL_B="${EVAL_MODEL_B:-vllm:Qwen/Qwen3-32B:tp4}"   # GPUs 4-7
EVAL_MODEL_SOLO="${EVAL_MODEL_SOLO:-vllm:Qwen/Qwen3-32B:tp8}"  # all 8 GPUs, solo run
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
# STEP 3: Evaluate — 7 conditions in 4 waves, 2 conditions × tp=4 in parallel.
#         GPUs 0-3 run one condition, GPUs 4-7 run another simultaneously.
#         Final solo condition gets all 8 GPUs via tp=8.
#         Each condition submits ALL 500 sessions as one huge batch:
#           amnesiac → single generate_batch call (oneshot)
#           recall / grep_recall → step-synchronised batch_loop
# ---------------------------------------------------------------------------
log "=== STEP 3: Evaluate (4 waves, tp=4×2 parallel, 7 conditions) ==="

OVERWRITE_FLAG=""
[[ -n "${FORCE_EVAL:-}" ]] && OVERWRITE_FLAG="--overwrite"

# _eval_pair <label_a> <model_a> <args_a...> -- <label_b> <model_b> <args_b...>
# Runs two evaluate.py jobs in parallel and waits for both.
_eval_pair() {
    local label_a="$1" model_a="$2"; shift 2
    local args_a=()
    while [[ "$1" != "--" ]]; do args_a+=("$1"); shift; done
    shift  # consume "--"
    local label_b="$1" model_b="$2"; shift 2
    local args_b=("$@")

    log "--- parallel: $label_a + $label_b ---"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/evaluate.py "${args_a[@]}" \
        --provider "$model_a" --num-gpus 4 --gpus-per-worker 4 $OVERWRITE_FLAG \
        2>&1 | tee "$LOG_DIR/eval_${label_a}.log" &
    local pid_a=$!

    CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/evaluate.py "${args_b[@]}" \
        --provider "$model_b" --num-gpus 4 --gpus-per-worker 4 $OVERWRITE_FLAG \
        2>&1 | tee "$LOG_DIR/eval_${label_b}.log" &
    local pid_b=$!

    wait $pid_a || { log "ERROR: $label_a failed"; exit 1; }
    wait $pid_b || { log "ERROR: $label_b failed"; exit 1; }
}

_eval_solo() {
    local label="$1"; shift
    log "--- solo: $label (tp=8, all 8 GPUs) ---"
    python scripts/evaluate.py "$@" \
        --provider "$EVAL_MODEL_SOLO" --num-gpus 8 --gpus-per-worker 8 $OVERWRITE_FLAG \
        2>&1 | tee "$LOG_DIR/eval_${label}.log"
    local rc=$?
    [[ $rc -ne 0 ]] && { log "ERROR: $label failed (exit $rc)"; exit $rc; }
}

# Wave 1: amnesiac (oneshot) + recall/lexical (batch_loop)
_eval_pair \
    "amnesiac"       "$EVAL_MODEL_A"  --condition amnesiac \
    -- \
    "recall_lexical" "$EVAL_MODEL_B"  --condition recall --summary-method lexical

# Wave 2: recall/scout + recall/echo
_eval_pair \
    "recall_scout" "$EVAL_MODEL_A" --condition recall --summary-method scout \
    -- \
    "recall_echo"  "$EVAL_MODEL_B" --condition recall --summary-method echo

# Wave 3: grep_recall/lexical + grep_recall/scout
_eval_pair \
    "grep_recall_lexical" "$EVAL_MODEL_A" --condition grep_recall --summary-method lexical \
    -- \
    "grep_recall_scout"   "$EVAL_MODEL_B" --condition grep_recall --summary-method scout

# Wave 4: grep_recall/echo solo — gets all 8 GPUs
_eval_solo "grep_recall_echo" --condition grep_recall --summary-method echo

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
