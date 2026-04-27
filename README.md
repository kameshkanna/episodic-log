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

Each turn in a session is reduced to a 1-2 line summary. These summaries are stored in `summaries/<method>.jsonl` and used by the retrieval conditions.

| Method | How it works |
|---|---|
| `lexical` | Rule-based extraction — no model, deterministic, instant |
| `scout` | Small LLM reads each turn and writes a one-line summary |
| `echo` | Small LLM reads each turn and echoes the key facts verbatim |

Summarization uses vLLM mega-batch (`LLM.generate()` once for all turns) for maximum throughput.

---

## Evaluation conditions

### `amnesiac`
No memory. Single generate call with just the question. Measures pure parametric recall — the floor.

---

### `recall/<method>`
**Dump all summaries → model reads → calls `load_turn`**

All 1-2 line summaries for every turn in the session are injected into the model's first message. The model reads the full summary index, identifies relevant turns by their IDs, and calls `load_turn` to read the verbatim content before answering.

No retrieval system. The model does the "search" by reading the summary block.

```
[0000] User greeted the assistant
[0001] User asked about scheduling a dentist appointment for Tuesday
[0002] Assistant confirmed it noted the appointment
...
[0499] User said goodbye

Question: When did I schedule my dentist appointment?
```

Model → `load_turn("0001")` → reads full turn → answers.

---

### `grep_recall/<method>`
**Model formulates keywords → keyword grep → `load_turn`**

The model sees only the question. It must call `grep_memory(keywords="...")` with self-chosen search terms. The system does a simple case-insensitive substring match against all summary lines and returns the matching ones. The model then calls `load_turn` on the ones it wants to read.

**Key difference from BM25:** there is no TF-IDF scoring. The model decides *what to search for*; the system just does a plain text grep. A BM25 system uses the raw question as input and scores every document — here the model's intelligence drives the query.

```
Model: grep_memory(keywords="dentist appointment Tuesday")
→ [0001] User asked about scheduling a dentist appointment for Tuesday
→ [0002] Assistant confirmed it noted the appointment

Model: load_turn("0001") → full verbatim content → answers
```

---

### `topk/<method>/k{3,5,10}`
**All summaries in context + top-k turns pre-loaded verbatim**

Same as `recall` (all summaries shown) plus the top-k turns whose summary has the highest keyword overlap with the question are pre-loaded verbatim before the model responds. One-shot answer, no tool calls.

Ablation axis: does pre-loading likely relevant turns help vs leaving the model to request them?

---

## Condition comparison

| Condition | Model sees upfront | Tools available | # LLM calls |
|---|---|---|---|
| `amnesiac` | Question only | none | 1 |
| `recall/<m>` | All summaries + question | `load_turn` | 1–16 |
| `grep_recall/<m>` | Question only | `grep_memory` + `load_turn` | 1–16 |
| `topk/<m>/k{n}` | All summaries + top-k verbatim turns + question | none | 1 |

16 conditions total: 1 amnesiac + 3 recall + 3 grep_recall + 9 topk (3 methods × k3/k5/k10).

---

## Pipeline

### Full run on 8× H100 SXM5 (~45 min)

```bash
bash rebuild_venv.sh   # first time only — clean venv, installs vllm first
bash run_pipeline.sh
```

**What `run_pipeline.sh` does:**

```
Step 1  ingest.py           CPU       ~5 min   500 sessions → log.jsonl files
Step 2a summarize.py        CPU       ~2 min   lexical summaries (no model)
Step 2b summarize.py        vLLM tp4  ~5 min   scout + echo in parallel
                            GPUs 0-3 / 4-7
Step 3  evaluate.py         HF BF16   ~30 min  8 conditions in 2 waves × 4
                            Qwen3-32B tp2, 2× H100 per worker
Step 4  judge.py            vLLM tp8  ~5 min   all verdicts in one generate() call
Step 5  score.py            CPU       <1 min   print results table
```

Wave 1 runs in parallel: amnesiac + recall/lexical + recall/scout + recall/echo.
Wave 2 runs in parallel: grep_recall × 3 + topk/lexical/k5.

---

### Run only the eval + judge + score (summaries already built)

```bash
# 4 conditions in parallel — each gets 2 H100s for Qwen3-72B BF16
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py \
    --condition amnesiac \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 1 --gpus-per-worker 1 &

CUDA_VISIBLE_DEVICES=2,3 python scripts/evaluate.py \
    --condition recall --summary-method lexical \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 1 --gpus-per-worker 1 &

CUDA_VISIBLE_DEVICES=4,5 python scripts/evaluate.py \
    --condition recall --summary-method scout \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 1 --gpus-per-worker 1 &

CUDA_VISIBLE_DEVICES=6,7 python scripts/evaluate.py \
    --condition recall --summary-method echo \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 1 --gpus-per-worker 1 &

wait

python scripts/judge.py --judge-provider "vllm:Qwen/Qwen2.5-14B-Instruct:tp8"
python scripts/score.py
```

---

### Run grep_recall ablations

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py \
    --condition grep_recall --summary-method lexical \
    --provider hf:Qwen/Qwen3-32B \
    --num-gpus 1 --gpus-per-worker 1
```

---

### Run topk ablations

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

`turns_loaded` is the set of turn IDs the model actually retrieved. `evidence_turn_ids` is the ground-truth set from LongMemEval. `score.py --retrieval` reports precision/recall/F1 between these two sets.

---

## Hardware

| Workload | Minimum |
|---|---|
| Groq-only (no GPU) | Any CPU |
| Summarize + judge (vLLM) | 1× H100 80 GB |
| Evaluate with Qwen3-32B BF16 | 1× H100 80 GB per condition |
| Full parallel pipeline (8 conditions) | 8× H100 80 GB |

---

## License

MIT
