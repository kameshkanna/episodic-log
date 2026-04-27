# episodic-log

**Measuring Conversational Hallucination Drift (CHD) on [LongMemEval](https://huggingface.co/datasets/xiaowu0162/LongMemEval)**

500 sessions, 500 questions. An LLM is given its own conversation history through different memory conditions and asked to recall facts from it. We measure how often it gets things right — and what kind of errors it makes when it fails.

---

## What is CHD?

**Conversational Hallucination Drift** — when a model misremembers its own conversation history.

| Verdict | Description |
|---|---|
| `correct` | Answer is consistent with the ground truth |
| `omission` | Model forgets or denies a clearly stated past action |
| `commission` | Model claims to have done/said something it never did |
| `distortion` | Model recalls an event but gets key details wrong (wrong date, wrong value) |
| `confabulation` | Model invents plausible-sounding history with no basis in the log |

---

## Architecture

```
episodic_log/
├── core/           TurnEvent, TurnSummary — immutable log record types
├── ingestor/       LongMemEval → session JSONL files
├── summarizers/    lexical / scout / echo — 3 summary index methods
├── conditions/     amnesiac, recall, grep_recall, topk — evaluation conditions
├── agent/          AgentLoop — tool-use loop (load_turn only, or grep+load)
├── tools/          grep_memory, load_turn, session_tools
├── judge/          CHDJudge — LLM classifier for CHD verdicts
├── metrics/        CHDMetrics, compute_metrics, print_comparison_table
└── providers/      GroqProvider, HuggingFaceProvider, VLLMProvider

scripts/
├── ingest.py       Step 1 — HuggingFace dataset → session log files
├── summarize.py    Step 2 — build summary indexes per method
├── evaluate.py     Step 3 — run conditions, write result JSONL
├── judge.py        Step 4 — batch-judge all predictions
└── score.py        Step 5 — print CHD metrics table
```

---

## Summarizer methods

Each turn in a session is reduced to a 1–2 sentence summary. Summaries are stored in `summaries/<method>.jsonl` and consumed by memory conditions at evaluation time.

| Method | How it works |
|---|---|
| `lexical` | Rule-based extraction — no model, deterministic, instant |
| `scout` | Small LLM (Qwen2.5-7B via vLLM) reads each turn, writes a one-line summary |
| `echo` | Small LLM reads each turn and echoes the key facts verbatim |

Summarization uses vLLM mega-batch (`LLM.generate()` once for all turns across the entire dataset) for maximum throughput.

---

## Memory index format

All recall-based conditions inject a **TSV memory index** into the model's first message — one row per turn, tab-separated:

```
turn_id	summary
0000	User greeted the assistant
0001	User asked about scheduling a dentist appointment for Tuesday
0002	Assistant confirmed it noted the appointment for Tuesday
...
0499	User said goodbye
```

Every turn is listed. Multi-line summaries are collapsed to a single line so the index is compact. 500 turns at ~150 chars/row ≈ 75k chars typical — well within Qwen3-32B's 128k-token context window. A hard cap of 100k chars is applied only in pathological cases; if hit, the index is truncated at a line boundary with a notice.

---

## Evaluation conditions

### `amnesiac`
No memory. Single generate call with just the question. Measures pure parametric recall — the floor.

---

### `recall/<method>`
**Full TSV index in context → model scans → calls `load_turn`**

The complete memory index (all turns, TSV format) is injected into the model's first message alongside the question. The model scans the index, identifies relevant turn IDs, and calls `load_turn` to read the full verbatim content before answering. Up to 15 `load_turn` calls are allowed; if the budget is exhausted the loop forces a final answer from whatever was loaded.

No retrieval system. The model does the "search" by reading the index.

```
Memory index — 500 turns (columns: turn_id TAB summary):
turn_id	summary
0000	User greeted the assistant
0001	User asked about scheduling a dentist appointment for Tuesday
0002	Assistant confirmed it noted the appointment
...

Question: When did I schedule my dentist appointment?
```

Model → `load_turn("0001")` → reads full verbatim turn → answers.

---

### `grep_recall/<method>`
**Model formulates keywords → keyword grep on summary column → `load_turn`**

The model sees only the question. It calls `grep_memory(keywords="...")` with self-chosen search terms. The system does a case-insensitive substring match **against the summary column only** (never the turn ID) and returns every matching row. The model then calls `load_turn` on the turns it wants to read.

**Key difference from BM25:** no TF-IDF scoring, no document ranking. The model decides *what to search for*; the system does a plain substring grep on the summary text. The model's intelligence drives the query.

```
Model: grep_memory(keywords="dentist appointment Tuesday")
→ 0001	User asked about scheduling a dentist appointment for Tuesday
→ 0002	Assistant confirmed it noted the appointment

Model: load_turn("0001") → full verbatim content → answers
```

---

### `topk/<method>/k{3,5,10}`
**Full TSV index in context + top-k turns pre-loaded verbatim, one-shot answer**

Same index as `recall` (all summaries shown) plus the top-k turns whose summary has the highest **unique keyword overlap** with the question are pre-loaded verbatim before the model responds. Overlap is computed as the size of the set intersection of word tokens — each shared word counts once regardless of how many times it appears. One-shot generate, no tool calls.

Ablation axis: does pre-loading the most likely relevant turns help vs leaving the model to request them via `load_turn`?

---

## Condition comparison

| Condition | Model sees upfront | Tools available | # LLM calls |
|---|---|---|---|
| `amnesiac` | Question only | none | 1 |
| `recall/<m>` | Full TSV index + question | `load_turn` | 1–16 |
| `grep_recall/<m>` | Question only | `grep_memory` + `load_turn` | 1–16 |
| `topk/<m>/k{n}` | Full TSV index + top-k verbatim turns + question | none | 1 |

16 conditions total: 1 amnesiac + 3 recall + 3 grep_recall + 9 topk (3 methods × k3/k5/k10).

---

## Pipeline

### Full run on 8× H100 SXM5 (~50 min)

```bash
bash rebuild_venv.sh   # first time only — clean venv, installs vllm first
bash run_pipeline.sh
```

**What `run_pipeline.sh` does:**

```
Step 1  ingest.py           CPU        ~5 min   500 sessions → log.jsonl files
                                                 Skipped if data/sessions_index.jsonl exists

Step 2a summarize.py        CPU        ~2 min   lexical summaries (no model)
                                                 Skipped if all 500 sessions have lexical.jsonl

Step 2b summarize.py        vLLM tp4   ~5 min   scout (GPUs 0-3) + echo (GPUs 4-7) in parallel
                                                 Each skipped independently if already complete

Step 3  evaluate.py         HF BF16    ~40 min  8 conditions in 2 waves of 4
         Wave 1             Qwen3-32B           amnesiac + recall × 3
                            2× H100/worker      GPUs 0-1 / 2-3 / 4-5 / 6-7 (parallel)
         Wave 2                                 grep_recall × 3 + topk/lexical/k5
                                                GPUs 0-1 / 2-3 / 4-5 / 6-7 (parallel)

Step 4  judge.py            vLLM tp8   ~5 min   all verdicts in one generate() call
Step 5  score.py            CPU        <1 min   print results table
```

Each step skips automatically if its output already exists. Override with:

```bash
FORCE_INGEST=1 bash run_pipeline.sh      # re-ingest even if sessions exist
FORCE_SUMMARIZE=1 bash run_pipeline.sh   # re-summarize even if summaries exist
FORCE_EVAL=1 bash run_pipeline.sh        # re-evaluate (passes --overwrite to evaluate.py)
```

---

### Run only eval + judge + score (summaries already built)

```bash
# Wave 1: 4 conditions in parallel, 2 GPUs each
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py \
    --condition amnesiac \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 2 --gpus-per-worker 2 &

CUDA_VISIBLE_DEVICES=2,3 python scripts/evaluate.py \
    --condition recall --summary-method lexical \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 2 --gpus-per-worker 2 &

CUDA_VISIBLE_DEVICES=4,5 python scripts/evaluate.py \
    --condition recall --summary-method scout \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 2 --gpus-per-worker 2 &

CUDA_VISIBLE_DEVICES=6,7 python scripts/evaluate.py \
    --condition recall --summary-method echo \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 2 --gpus-per-worker 2 &

wait

# Wave 2: grep_recall + topk
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py \
    --condition grep_recall --summary-method lexical \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 2 --gpus-per-worker 2 &

CUDA_VISIBLE_DEVICES=2,3 python scripts/evaluate.py \
    --condition grep_recall --summary-method scout \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 2 --gpus-per-worker 2 &

CUDA_VISIBLE_DEVICES=4,5 python scripts/evaluate.py \
    --condition grep_recall --summary-method echo \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 2 --gpus-per-worker 2 &

CUDA_VISIBLE_DEVICES=6,7 python scripts/evaluate.py \
    --condition topk --summary-method lexical --retrieval-k 5 \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 2 --gpus-per-worker 2 &

wait

python scripts/judge.py --judge-provider "vllm:Qwen/Qwen2.5-14B-Instruct:tp8"
python scripts/score.py
```

---

### Run topk ablations (all k values)

```bash
for k in 3 5 10; do
  CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py \
      --condition topk --summary-method lexical --retrieval-k $k \
      --provider hf:Qwen/Qwen3-32B \
      --num-gpus 2 --gpus-per-worker 2
done
```

---

### Score with retrieval quality metrics

```bash
# Basic metrics table
python scripts/score.py

# Include retrieval precision/recall/F1 (measures whether load_turn hit evidence turns)
python scripts/score.py --retrieval

# Per question-type breakdown
python scripts/score.py --breakdown
```

---

## OOM resilience

The HuggingFace provider has three layers of defence against GPU out-of-memory errors:

1. **Pre-emptive token budget** — prompts are trimmed to 28k input tokens before the forward pass. Middle messages are dropped first; the anchor (first user message with the memory index) and the most recent exchanges are always kept.
2. **OOM retry** — if a forward pass still hits OOM, the cache is flushed, context is trimmed to the last 4 messages, and the call is retried once at a reduced token budget.
3. **Per-session recovery** — if OOM survives both retries, the worker logs the session as failed, flushes the CUDA cache, and continues to the next session without crashing the whole shard.

The agent loop adds a fourth layer: if a step hits OOM after the provider's own retry, the oldest tool-result exchange is dropped from the message history and the step is retried.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/episodic-log.git
cd episodic-log
bash rebuild_venv.sh
source .venv/bin/activate
cp .env.example .env   # add GROQ_API_KEY if using groq: provider
```

`rebuild_venv.sh` creates a clean venv (no system-site-packages), installs vLLM first to lock a compatible numpy/scipy/transformers ABI, then installs the project.

---

## Provider spec format

```
hf:<org>/<model>              HuggingFace, BF16, device_map=auto
hf:<org>/<model>:4bit         HuggingFace, NF4 4-bit quantization
hf:<org>/<model>:8bit         HuggingFace, LLM.int8()
vllm:<org>/<model>            vLLM offline batch, tp=1
vllm:<org>/<model>:tp4        vLLM, tensor_parallel_size=4
vllm:<org>/<model>:tp8        vLLM, tensor_parallel_size=8
groq:<model>                  Groq API (rate-limited, no GPU needed)
```

When `CUDA_VISIBLE_DEVICES` is set in the shell before launching `evaluate.py`, the provider uses exactly those GPUs with `device_map=auto`. Do not pass `--num-gpus` larger than the number of visible devices.

---

## Output layout

```
data/
├── sessions/
│   └── <session_id>/
│       ├── log.jsonl                immutable turn events
│       └── summaries/
│           ├── lexical.jsonl
│           ├── scout.jsonl
│           └── echo.jsonl
├── sessions_index.jsonl
└── results/
    └── <model-slug>/
        └── <condition>__<method>.jsonl   predictions + verdicts
```

Result row schema:

```json
{
  "session_id": "...",
  "question_id": "...",
  "condition": "recall/lexical",
  "summary_method": "lexical",
  "question": "...",
  "ground_truth": "...",
  "predicted_answer": "...",
  "tool_calls": [...],
  "turns_loaded": ["0001", "0042"],
  "evidence_turn_ids": ["0001"],
  "question_type": "single_session_user",
  "verdict": "correct",
  "confidence": 0.94,
  "judge_reason": "..."
}
```

`turns_loaded` — turn IDs the model actually retrieved (via `load_turn` or topk pre-load).
`evidence_turn_ids` — ground-truth relevant turn IDs from LongMemEval.
`score.py --retrieval` reports macro-averaged precision/recall/F1 between these two sets.

---

## Hardware

| Workload | Minimum |
|---|---|
| Groq-only (no GPU) | Any CPU |
| Summarize + judge (vLLM) | 1× H100 80 GB |
| Single condition eval — Qwen3-32B BF16 | 2× H100 80 GB (tp2, device_map=auto) |
| Full parallel pipeline (2 waves × 4 conditions) | 8× H100 80 GB |

---

## License

MIT
