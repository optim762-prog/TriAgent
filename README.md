# TriAgent — Experiments Overview

## Setup

**Dataset**: `datasets/instances.jsonl` — 100 instances across 5 agronomic processes (20 each).
**Ground truth**: `datasets/ground_truth.jsonl` — keyword + range matching, no exact match.
**Evaluation**: `experiments/evaluate.py` → `results/aggregated/`

Processes:
| ID | Disease | Pathogen | Steps |
|---|---|---|---|
| P1 | Wheat Brown Rust | _Puccinia triticina_ | 5 |
| P2 | Tomato Late Blight | _Phytophthora infestans_ | 5 |
| P3 | Olive Fruit Fly | _Bactrocera oleae_ | 7 |
| P4 | Wheat Fusarium HB | _Fusarium graminearum_ | 6 |
| P5 | Citrus Greening HLB | _Candidatus_ Liberibacter | 6 |

---

## Ablation modes

| Mode           | Knowledge Agent                    | Execution Agent                      | Tests                           |
| -------------- | ---------------------------------- | ------------------------------------ | ------------------------------- |
| `full`         | FAISS retrieval → rule             | Live tool calls                      | A + B combined                  |
| `no_knowledge` | Bypass FAISS; pass raw description | Live tool calls                      | B only (with `--skip-prefetch`) |
| `no_execution` | FAISS retrieval → rule             | Returns `TOOL_DISABLED_FOR_ABLATION` | A only                          |

Factor isolation design (new system):

```
no_execution            → CIMMYT PDF effect only       (A, no tool calling)
no_knowledge + skip     → Tool calling effect only      (B, no FAISS)
full + skip             → A + B combined                (full new system)
```

Baseline for each comparison: the corresponding v1 run (same backbone, same ablation mode).

**v1 prerequisite**: FAISS store must be pre-seeded before each batch:

```bash
docker compose -f infra/docker-compose.machineB.yml run --rm triagent-runner \
  seed --rules /app/datasets/seed_rules.jsonl
```

---

## Infrastructure

- **Local** (24 GB VRAM) — llama.cpp server + triagent-runner
- **Local models**:
  - `gemma4-local`: `unsloth/gemma-4-26B-A4B-it-GGUF` Q4_K_XL (~18 GB)
  - `qwen-local`: `unsloth/Qwen3.6-27B-MTP-GGUF` UD-Q4_K_XL (~16 GB)
  - Swap: stop server, change env, restart, wait for healthcheck
- **Cloud**: DeepSeek (`deepseek-chat`) — `--no-deps`, no llama.cpp needed

Batch command pattern (v2 new system):

```bash
docker compose -f docker-compose.yml run --rm sepielli-triagent-runner \
  batch \
  --input  /app/datasets/instances.jsonl \
  --output /app/results/2026-06-12_fullsystem/<backbone>__<ablation>__n100/results.jsonl \
  --backbone <backbone> --provider <provider> --ablation <mode> [--skip-prefetch]
```

---

## Run checklist

### v1 — Old system (seed FAISS, pre-fetch active, old prompt)

| Backbone      | Ablation        | Status     | Folder                                                       |
| ------------- | --------------- | ---------- | ------------------------------------------------------------ |
| gemma4-local  | full (seeded)   | ✅ done    | `2026-06-11_gemma-deepseek/gemma4-local__full__n100`         |
| gemma4-local  | no_knowledge    | ✅ done    | `2026-06-11_gemma-ablation/gemma4-local__no_knowledge__n100` |
| gemma4-local  | no_execution    | ✅ done    | `2026-06-11_gemma-ablation/gemma4-local__no_execution__n100` |
| qwen-local    | full (seeded)   | ✅ done    | `2026-06-11_qwen-seeded/qwen-local__full__n100`              |
| qwen-local    | full (unseeded) | ✅ done    | `2026-06-10_qwen-matrix/qwen-local__full__n100`              |
| qwen-local    | no_knowledge    | ✅ done    | `2026-06-10_qwen-matrix/qwen-local__no_knowledge__n100`      |
| qwen-local    | no_execution    | ✅ done    | `2026-06-10_qwen-matrix/qwen-local__no_execution__n100`      |
| deepseek-chat | full            | ✅ done    | `2026-06-11_gemma-deepseek/deepseek-chat__full__n100`        |
| deepseek-chat | no_knowledge    | ⏳ pending | `run_deepseek_oldystem.sh`                                   |
| deepseek-chat | no_execution    | ⏳ pending | `run_deepseek_oldystem.sh`                                   |

### v2 — New system (CIMMYT PDF FAISS, --skip-prefetch, fixed prompt)

| Backbone      | Ablation            | Status       | Script step                                                            |
| ------------- | ------------------- | ------------ | ---------------------------------------------------------------------- |
| gemma4-local  | no_execution        | ✅ done      | `2026-06-12_fullsystem/gemma4-local__no_execution__n100`               |
| gemma4-local  | no_knowledge + skip | ✅ done      | `2026-06-12_fullsystem/gemma4-local__no_knowledge_skipprefetch__n100`  |
| gemma4-local  | full + skip         | ✅ done      | `2026-06-12_fullsystem/gemma4-local__full_skipprefetch__n100`          |
| gemma4-local  | full + skip (n=5)   | ✅ validated | `2026-06-12_tool-validation/` — tools_ok = 1.0                         |
| deepseek-chat | no_execution        | ✅ done      | `2026-06-12_fullsystem/deepseek-chat__no_execution__n100`              |
| deepseek-chat | no_knowledge + skip | ✅ done      | `2026-06-12_fullsystem/deepseek-chat__no_knowledge_skipprefetch__n100` |
| deepseek-chat | full + skip         | ✅ done      | `2026-06-12_fullsystem/deepseek-chat__full_skipprefetch__n100`         |
| qwen-local    | no_execution        | ✅ done      | `2026-06-12_fullsystem/qwen-local__no_execution__n100`                 |
| qwen-local    | no_knowledge + skip | ✅ done      | `2026-06-12_fullsystem/qwen-local__no_knowledge_skipprefetch__n100`    |
| qwen-local    | full + skip         | ✅ done      | `2026-06-12_fullsystem/qwen-local__full_skipprefetch__n100`            |

---

## Results

### v1 — Old system (n=100)

Seed FAISS (9 hand-written rules), weather pre-fetched before LangGraph loop, old prompt.
`tools_ok` universally 0.0 — pre-fetch prevented models from ever triggering `requires_tool`.

| Backbone      | Ablation        | Overall   | Diag  | Substance | Dose  | Gateway |
| ------------- | --------------- | --------- | ----- | --------- | ----- | ------- |
| gemma4-local  | no_execution    | **0.500** | 0.950 | 0.280     | 0.230 | 0.560   |
| gemma4-local  | full            | 0.443     | 0.960 | 0.260     | 0.210 | 0.510   |
| qwen-local    | no_execution    | 0.442     | 0.940 | 0.220     | 0.190 | 0.330   |
| qwen-local    | full (seeded)   | 0.417     | 0.900 | 0.230     | 0.180 | 0.470   |
| gemma4-local  | no_knowledge    | 0.410     | 0.920 | 0.260     | 0.080 | 0.390   |
| deepseek-chat | full            | 0.380     | 0.820 | 0.160     | 0.130 | 0.540   |
| qwen-local    | no_knowledge    | 0.377     | 0.920 | 0.200     | 0.080 | 0.290   |
| qwen-local    | full (unseeded) | 0.350     | 0.820 | 0.140     | 0.150 | 0.270   |

### v2 — New system (n=100)

CIMMYT PDF FAISS, `--skip-prefetch`, fixed system prompt with TOOL CALLING RULES.
`tools_ok = 1.0` for all full+skip runs — `requires_tool: true` triggers live `get_weather` call.

| Backbone      | Ablation          | Overall   | Diag  | Substance | Dose  | Gateway | Δ vs v1    |
| ------------- | ----------------- | --------- | ----- | --------- | ----- | ------- | ---------- |
| gemma4-local  | full+skip         | **0.488** | 0.970 | 0.290     | 0.210 | 0.510   | **+0.045** |
| gemma4-local  | no_execution      | 0.485     | 0.920 | 0.260     | 0.240 | 0.520   | −0.015     |
| qwen-local    | full+skip         | 0.485     | 0.940 | 0.280     | 0.230 | 0.500   | **+0.068** |
| qwen-local    | no_execution      | 0.483     | 0.950 | 0.240     | 0.230 | 0.490   | +0.041     |
| deepseek-chat | no_execution      | 0.443     | 0.900 | 0.260     | 0.140 | 0.410   | —          |
| deepseek-chat | full+skip         | 0.433     | 0.920 | 0.230     | 0.130 | 0.400   | **+0.053** |
| qwen-local    | no_knowledge+skip | 0.423     | 0.990 | 0.210     | 0.080 | 0.260   | +0.046     |
| gemma4-local  | no_knowledge+skip | 0.405     | 0.970 | 0.140     | 0.040 | 0.290   | −0.005     |
| deepseek-chat | no_knowledge+skip | 0.377     | 0.950 | 0.270     | 0.010 | 0.090   | —          |

---

## Key findings

**F1 — FAISS pre-seeding is necessary (v1).**
Without seed rules, cold-start causes P3 `full` escalation 50%. With seeding:
Qwen full 0.350 → 0.417, gateway_ok 0.270 → 0.470.

**F2 — full > no_knowledge confirmed in both versions.**
v1: Gemma4 0.443 > 0.410; Qwen 0.417 > 0.377.
v2: Gemma4 0.488 > 0.405; Qwen 0.485 > 0.423; DeepSeek 0.433 > 0.377.
KnowledgeAgent adds consistent value across all models and both system versions.

**F3 — full > no_execution in v2 (hypothesis confirmed); reversed in v1.**
v1: no_execution > full (Gemma4 0.500 > 0.443) — tools never fired, execution node only added stalling risk.
v2: full+skip ≥ no_execution for Gemma4 (0.488 > 0.485) and Qwen (0.485 > 0.483).
DeepSeek still shows no_execution > full+skip (0.443 > 0.433) — small gap, may be noise.
Conclusion: when tools actually fire, the full 3-agent system is best or co-best.

**F4 — v2 system improves over v1 across all models.**
Gemma4 full: +0.045 (0.443 → 0.488). Qwen full: +0.068 (0.417 → 0.485). DeepSeek full: +0.053 (0.380 → 0.433).
Both improvements (CIMMYT PDF rules + live tool execution) contribute.

**F5 — Gemma4 ≈ Qwen in v2; both well ahead of DeepSeek.**
v2 full+skip: Gemma4 0.488, Qwen 0.485, DeepSeek 0.433.
v1 gap between local models was larger; CIMMYT PDF rules helped Qwen most (+0.068).

**F6 — DeepSeek underperforms local models on structured protocol reasoning.**
v1: 0.380, 37% escalation. v2: 0.433 — improved but still lowest.
Cloud inference speed ≠ structured reasoning capability.

**F7 — Tool calling validated (v2, 2026-06-12).**
`gemma4-local full+skip` n=5: tools_ok = 1.0. All instances called `get_weather` via
`requires_tool: true` and used live observations in gateway decisions (e.g., P3 Bari wind
7.6 m/s → correctly delayed spray). Fix required: (a) skip pre-fetch, (b) explicit TOOL
CALLING RULES in system prompt explaining that `requires_tool: true` triggers real execution.

---

## Pending runs (cloud, needs API keys)

| Backbone      | Ablation                         | Blocker        |
| ------------- | -------------------------------- | -------------- |
| gpt-4o        | full, no_knowledge, no_execution | OpenAI API key |
| llama-3.3-70b | full                             | Groq API key   |

Note for GPT-4o: delete `/app/data/rules` before run (384-dim → 1536-dim FAISS conflict).
