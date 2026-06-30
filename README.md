# TriAgent: A three-layer LLM-based agentic framework for web-based process automation

TriAgent decomposes agronomic process automation into three specialized agents:
**Knowledge** (FAISS-based retrieval of disease-management rules from a CIMMYT PDF),
**Reasoning** (ReAct + chain-of-thought LLM with confidence scoring), and
**Execution** (live API tool calls: weather, EPPO, SMS).
It is evaluated on 5 crop-disease processes (wheat rust, tomato blight, olive fruit fly,
wheat fusarium, citrus greening) across 100 benchmark instances, with 3 LLM backbones
and 3 ablation modes.

---

## Repository layout

```
TriAgent/
├── backend/              # Python source: agents, orchestrator, CLI, config
│   ├── agents/           # KnowledgeAgent, ReasoningAgent, ExecutionAgent
│   ├── config/           # pydantic-settings + LLM client factory (OpenAI-compatible)
│   ├── cli.py            # Entry point: batch / monolithic / seed commands
│   └── orchestrator.py   # LangGraph state machine (K → R ↔ E)
├── experiments/          # Dataset builder, evaluator, experiment matrix
├── scheduler/            # Async job scheduler for running the full experiment matrix
├── infra/                # Docker Compose files, Dockerfiles, batch runner scripts
├── data/rules/           # Pre-built FAISS index + rules.json (included, ready to use)
├── datasets/             # Benchmark instances and ground-truth JSONL files
├── results/              # Per-run JSONL outputs + aggregated CSV summaries
├── EXPERIMENTS.md        # Full run matrix, commands, and all numerical results
└── CODE_OVERVIEW.md      # Formal model ↔ code mapping
```

---

## Requirements

### Option A — Docker (recommended; required for local models)

- Docker ≥ 24 with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- GPU with ≥ 24 GB VRAM for local models (Gemma4-26B or Qwen3-27B)
- DeepSeek cloud runs need **no GPU**

### Option B — Local Python (cloud backbones only)

- Python 3.11
- `pip install -r backend/requirements.txt`

---

## Configuration

```bash
cp infra/.env.template infra/.env
# Edit infra/.env and set the variables you need:
```

| Variable                 | Required for            | Where to obtain                                                          |
| ------------------------ | ----------------------- | ------------------------------------------------------------------------ |
| `DEEPSEEK_API_KEY`       | DeepSeek runs           | [platform.deepseek.com](https://platform.deepseek.com)                   |
| `HUGGING_FACE_HUB_TOKEN` | Local GGUF download     | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `OPENWEATHER_API_KEY`    | Live weather tool calls | [openweathermap.org/api](https://openweathermap.org/api)                 |

Twilio SMS variables (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`) are **optional**: used only by the `send_sms` execution tool; the tool silently simulates SMS when these are not set.

---

## Knowledge base

The pre-built FAISS index (MiniLM-L6-v2 embeddings, rules extracted from a CIMMYT wheat disease PDF) is already included in `data/rules/` — no rebuild is needed.

To rebuild from the source PDF:

```bash
python experiments/load_fao_docs.py data/fao_documents data/rules
```

---

## Running experiments

### Build the Docker image

```bash
docker compose -f infra/docker-compose.yml build
```

### Start the local model server (skip for cloud-only runs)

```bash
docker compose -f infra/docker-compose.yml up -d triagent-llama-cpp-server
# First run downloads the GGUF model (~16–18 GB).
# Wait until the health check passes:
docker compose -f infra/docker-compose.yml ps
```

### Run a batch (v2 system — CIMMYT PDF FAISS, `--skip-prefetch`)

```bash
docker compose -f infra/docker-compose.yml run --rm triagent-runner \
  batch \
  --input  /app/datasets/instances.jsonl \
  --output /app/results/$(date +%Y-%m-%d)_run/<backbone>__<ablation>__n100/results.jsonl \
  --backbone <backbone> \
  --provider <provider> \
  --ablation <full|no_knowledge|no_execution> \
  --skip-prefetch
```

**Backbone / provider reference:**

| `--backbone`    | `--provider` | Hardware     | Notes                                        |
| --------------- | ------------ | ------------ | -------------------------------------------- |
| `gemma4-local`  | `llamacpp_b` | 24 GB GPU    | Gemma 4 26B Q4_K_XL (~18 GB)                 |
| `qwen-local`    | `llamacpp_b` | 24 GB GPU    | Qwen3-27B Q4_K_XL (~16 GB)                   |
| `deepseek-chat` | `deepseek`   | None (cloud) | Requires `DEEPSEEK_API_KEY`; use `--no-deps` |

**DeepSeek cloud run (no llama.cpp server needed):**

```bash
docker compose -f infra/docker-compose.yml run --rm --no-deps triagent-runner \
  batch \
  --input  /app/datasets/instances.jsonl \
  --output /app/results/$(date +%Y-%m-%d)_deepseek/deepseek-chat__full_skipprefetch__n100/results.jsonl \
  --backbone deepseek-chat --provider deepseek --ablation full --skip-prefetch
```

**Swap between local models** (only one fits in 24 GB at a time):

```bash
# Switch to Gemma4:
docker compose -f infra/docker-compose.yml stop triagent-llama-cpp-server
LCPP_MODEL_REPO="unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL" \
LCPP_ALIAS=gemma4-local LCPP_CTX_SIZE=8192 \
  docker compose -f infra/docker-compose.yml up -d triagent-llama-cpp-server
# Wait for health, then run batches with --backbone gemma4-local --provider llamacpp_b

# Switch back to Qwen3:
docker compose -f infra/docker-compose.yml stop triagent-llama-cpp-server
docker compose -f infra/docker-compose.yml up -d triagent-llama-cpp-server
```

**Pre-built runner scripts** (reproduce the full v2 experiment matrix inside a `screen` session):

```bash
screen -dmS main bash infra/run_main_skipprefetch.sh    # Gemma4 + DeepSeek, all ablations
screen -dmS qwen  bash infra/run_qwen_newsystem.sh       # Qwen (swap model first)
screen -dmS mono  bash infra/run_monolithic.sh            # Monolithic baseline
screen -dmS tf    bash infra/run_toolforcing.sh           # Tool-forcing evaluation
```

### Local Python (cloud backbones only, Option B)

```bash
cd backend
DEEPSEEK_API_KEY=sk-... python -m cli batch \
  --input  ../datasets/instances.jsonl \
  --output ../results/$(date +%Y-%m-%d)_run/deepseek-chat__full_skipprefetch__n100/results.jsonl \
  --backbone deepseek-chat --provider deepseek --ablation full --skip-prefetch
```

---

## Ablation modes

| `--ablation`   | Knowledge Agent                   | Execution Agent        | What it isolates          |
| -------------- | --------------------------------- | ---------------------- | ------------------------- |
| `full`         | FAISS retrieval → rule            | Live tool calls        | Both improvements (A + B) |
| `no_knowledge` | Bypassed (raw description passed) | Live tool calls        | Execution only (B)        |
| `no_execution` | FAISS retrieval → rule            | Disabled (placeholder) | Knowledge only (A)        |

Always add `--skip-prefetch` in v2 runs: suppresses weather pre-fetch so the Execution agent must issue the live `get_weather` call.

---

## Monolithic baseline

Single-shot LLM with no agents, no FAISS, no ReAct loop — for comparison with Theorem 1:

```bash
docker compose -f infra/docker-compose.yml run --rm --no-deps triagent-runner \
  monolithic \
  --input  /app/datasets/instances.jsonl \
  --output /app/results/monolithic_baseline/deepseek-chat__monolithic__n100/results.jsonl \
  --backbone deepseek-chat --provider deepseek
```

---

## Tool-forcing benchmark

Runs on `tool_forcing_instances_n100.jsonl` (weather absent from descriptions, forcing a live `get_weather` call). Use `full` ablation with `--skip-prefetch` only:

```bash
docker compose -f infra/docker-compose.yml run --rm triagent-runner \
  batch \
  --input  /app/datasets/tool_forcing_instances_n100.jsonl \
  --output /app/results/tool_forcing/<backbone>__full_skipprefetch__n100/results.jsonl \
  --backbone <backbone> \
  --provider <provider> \
  --ablation full \
  --skip-prefetch
```

Pre-built runner scripts: `infra/run_toolforcing.sh` (Gemma4 + DeepSeek), `infra/run_toolforcing_qwen.sh` (Qwen).

---

## Evaluation

```bash
# Main experiments (v2 new system)
python experiments/evaluate.py \
  --results results/v2_new_system \
  --gt datasets/ground_truth.jsonl

# Tool-forcing experiments
python experiments/evaluate.py \
  --results results/tool_forcing \
  --gt datasets/ground_truth_toolforcing_n100.jsonl

# Monolithic baseline
python experiments/evaluate.py \
  --results results/monolithic_baseline \
  --gt datasets/ground_truth.jsonl
```

Scoring covers 6 dimensions per instance: **diagnosis**, **substance**, **dose**,
**tools** (tool call fired), **gateway** (correct go/no-go at each process step),
**escalation**. Output CSVs are written to `results/aggregated/`.

Pre-computed aggregated results for all runs reported in the paper are already included
in `results/aggregated/`.

> **Note:** `results/v1_old_system/` contains results from a preliminary version of the system (seed-only FAISS, weather pre-fetched before the LangGraph loop) that are not reported in the paper tables. They are documented in `EXPERIMENTS.md` for reference.
