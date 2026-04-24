# episodic-log

**Measuring Conversational Hallucination Drift (CHD) on [LongMemEval](https://huggingface.co/datasets/xiaowu0162/LongMemEval)**

An agent keeps an immutable, append-only episodic log of every turn in a conversation.  At evaluation time, seven retrieval and memory conditions are tested against 500 questions to measure how accurately the agent recalls its own history — and what kinds of errors it makes when it fails.

---

## What is CHD?

Conversational Hallucination Drift is when an agent misremembers its own conversation history.  We measure four failure modes:

| Category | Description |
|---|---|
| `commission` | Claims to have done/said something it never did |
| `omission` | Forgets or denies a clearly stated past action |
| `distortion` | Recalls an event but gets key details wrong (wrong date, wrong value) |
| `confabulation` | Invents plausible-sounding history with no basis in the log |
| `correct` | Answer is consistent with the ground truth |

---

## Architecture

```
episodic_log/
├── core/          # TurnEvent, TurnSummary, LogWriter, LogReader, TurnLoader
├── ingestor/      # LongMemEval → session JSONL + summaries
├── summarizers/   # structured / haiku / self  (3 ablations)
├── retrieval/     # BM25Index, SummaryStore
├── conditions/    # 7 evaluation conditions (see below)
├── judge/         # CHDJudge — LLM-based CHD verdict classifier
├── metrics/       # CHDMetrics, compute_metrics, print_comparison_table
└── providers/     # GroqProvider, HuggingFaceProvider (4-bit / 8-bit)

scripts/
├── ingest.py      # Step 1: HuggingFace → session logs
├── summarize.py   # Step 2: build BM25 summaries per method
├── evaluate.py    # Step 3a: single model evaluation
├── run_sweep.py   # Step 3b: multi-model GPU sweep
├── judge.py       # Step 4: batch-judge results via Groq (parallel)
└── score.py       # Step 5: print CHD metrics table
```

### Key design constraint

The agent **cannot write to its own log** — the episodic log is append-only and managed externally.  This is the core difference from MemGPT / A-MEM style systems.

---

## Evaluation conditions

| Condition | Description |
|---|---|
| `baseline` | No memory — single generate, no context |
| `episodic` | BM25 retrieval loop (up to 3 tool calls) → verbatim TurnEvent grounding |
| `adversarial` | Retrieved turn IDs shifted ±10 — tests robustness to retrieval noise |
| `proactive` | Top-5 BM25 hits pre-injected, one-shot — no agent tool use |
| `external` | Retrieved turns labelled as from a *different* agent's session — attribution confusion |
| `md_memory` | Full log compressed to ≤4000-char markdown summary — one-shot |
| `full_context` | All turns injected verbatim, truncated at 32k chars |

### Summarizer ablations

| Method | Description |
|---|---|
| `structured` | Rule-based extraction — no model required, deterministic |
| `haiku` | LLM summary via any small HF model (e.g. Qwen2.5-7B) |
| `self` | Agent writes its own summaries during ingestion |

---

## Model matrix (GPU sweep)

| Model | Slug | Size | Family |
|---|---|---|---|
| Llama-3.1-8B-Instruct | `llama-3.1-8b` | small | llama |
| Qwen2.5-7B-Instruct | `qwen-2.5-7b` | small | qwen |
| gemma-2-9b-it | `gemma-2-9b` | small | gemma |
| Mistral-7B-Instruct-v0.3 | `mistral-7b` | small | mistral |
| Qwen2.5-14B-Instruct (4-bit) | `qwen-2.5-14b-4bit` | medium | qwen |
| gemma-2-27b-it (4-bit) | `gemma-2-27b-4bit` | medium | gemma |
| Qwen2.5-32B-Instruct (4-bit) | `qwen-2.5-32b-4bit` | large | qwen |
| Llama-3.3-70B-Instruct (4-bit) | `llama-3.3-70b-4bit` | large | llama |
| Qwen2.5-72B-Instruct (4-bit) | `qwen-2.5-72b-4bit` | large | qwen |

Large models use NF4 4-bit quantization via `bitsandbytes`.

---

## Setup

```bash
# Clone and run the setup script (creates .venv, detects CUDA, installs deps)
git clone https://github.com/YOUR_USERNAME/episodic-log.git
cd episodic-log
bash setup.sh

# Activate
source .venv/bin/activate

# Copy and fill in your API keys
cp .env.example .env
```

**`.env` keys:**
```
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...        # required for gated models (Llama, Gemma)
```

---

## Running the full pipeline

### Option A — Quick smoke test with Groq (no GPU needed)

```bash
# Step 1: Ingest 5 sessions from LongMemEval
python scripts/ingest.py --n 5

# Step 2: Build BM25 summaries
python scripts/summarize.py --method structured

# Step 3: Run baseline + episodic conditions
python scripts/evaluate.py --condition baseline --n 5 \
    --provider groq:llama-3.1-8b-instant

python scripts/evaluate.py --condition episodic --n 5 \
    --provider groq:llama-3.1-8b-instant

# Step 4: Judge predictions (parallel, ~8× faster than sequential)
python scripts/judge.py --judge-provider groq:llama-3.1-70b-versatile

# Step 5: Score
python scripts/score.py
```

### Option B — Full 500-session sweep on 8× A100 80 GB

```bash
# Step 1: Ingest all 500 sessions
python scripts/ingest.py

# Step 2: Build all summary methods
python scripts/summarize.py --method structured
python scripts/summarize.py --method haiku --provider hf:Qwen/Qwen2.5-7B-Instruct
python scripts/summarize.py --method self  --provider hf:Qwen/Qwen2.5-7B-Instruct

# Step 3: Multi-model sweep — 8 models run in parallel across 8 GPUs
# All 9 models × all 7 conditions × all 3 summary methods
python scripts/run_sweep.py --summary-methods structured,haiku,self

# Dry-run to see the GPU assignment plan first
python scripts/run_sweep.py --dry-run

# Override GPU count
python scripts/run_sweep.py --num-gpus 4

# Step 4: Local HF judge across all 8 GPUs — no API calls, no rate limits
python scripts/judge.py --judge-provider hf:Qwen/Qwen2.5-14B-Instruct

# Step 5: Score with per-question-type breakdown
python scripts/score.py --breakdown
```

### Option C — Single model, all conditions

```bash
python scripts/evaluate.py \
    --condition episodic \
    --provider hf:meta-llama/Llama-3.1-8B-Instruct \
    --model-slug llama-3.1-8b \
    --summary-method structured

python scripts/judge.py \
    --results data/results/llama-3.1-8b/episodic__structured.jsonl \
    --judge-provider groq:llama-3.1-70b-versatile

python scripts/score.py --results data/results/llama-3.1-8b/episodic__structured.jsonl
```

---

## Output layout

```
data/
├── sessions/
│   └── <session_id>/
│       ├── log.jsonl           # immutable turn events
│       └── summaries/
│           ├── structured.jsonl
│           ├── haiku.jsonl
│           └── self.jsonl
├── sessions_index.jsonl        # index of all ingested sessions
└── results/
    └── <model-slug>/
        └── <condition>__<method>.jsonl   # predictions + verdicts
```

Each result row:

```json
{
  "model_slug": "llama-3.1-8b",
  "session_id": "...",
  "condition": "episodic",
  "summary_method": "structured",
  "question": "...",
  "ground_truth": "...",
  "predicted_answer": "...",
  "retrieved_turn_ids": ["turn_0003", "turn_0011"],
  "evidence_turn_ids": ["turn_0003"],
  "num_retrieval_calls": 2,
  "question_type": "single_session_user",
  "verdict": "correct",
  "confidence": 0.95,
  "judge_reason": "..."
}
```

---

## Provider spec format

```
groq:<model>                          # Groq API
hf:<org>/<model>                      # HuggingFace, BF16
hf:<org>/<model>:4bit                 # HuggingFace, NF4 4-bit
hf:<org>/<model>:8bit                 # HuggingFace, LLM.int8()
```

Examples:
```
groq:llama-3.1-8b-instant
groq:llama-3.3-70b-versatile
hf:meta-llama/Llama-3.1-8B-Instruct
hf:Qwen/Qwen2.5-72B-Instruct:4bit
```

---

## Hardware requirements

| Workload | Minimum |
|---|---|
| Groq-only (ingest + judge + score) | Any CPU |
| Small models (7–9B, BF16) | 24 GB VRAM (A10 / 3090) |
| Medium models (14–27B, BF16) | 80 GB VRAM (A100 / H100) |
| Large models (32–72B, 4-bit) | 80 GB VRAM (A100 / H100) |
| Full 9-model parallel sweep | 8× A100 80 GB (recommended) |

---

## License

MIT
