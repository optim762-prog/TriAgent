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

| Mode           | Knowledge Agent                    | Execution Agent                      |
| -------------- | ---------------------------------- | ------------------------------------ |
| `full`         | FAISS retrieval → rule             | Live tool calls                      |
| `no_knowledge` | Bypass FAISS; pass raw description | Live tool calls                      |
| `no_execution` | FAISS retrieval → rule             | Returns `TOOL_DISABLED_FOR_ABLATION` |

**Important**: `full` runs require FAISS store to be pre-seeded before the batch:

```bash
docker compose -f infra/docker-compose.machineB.yml run --rm triagent-runner \
  seed --rules /app/datasets/seed_rules.jsonl
```

---

## Infrastructure

- **Local** (24 GB VRAM):
  `infra/docker-compose.machineB.yml` — llama.cpp server + triagent-runner
- **Local Models**:
  - `gemma4-local`: `unsloth/gemma-4-26B-A4B-it-GGUF` Q4_K_XL (~18 GB)
  - `qwen-local`: `unsloth/Qwen3.6-27B-MTP-GGUF` UD-Q4_K_XL (~16 GB)
  - Swap between them: stop server, change env, restart, wait for healthcheck
- **Cloud**: DeepSeek (`deepseek-chat`), OpenAI (`gpt-4o`), Groq (`llama-3.3-70b-versatile`)

Batch command pattern:

```bash
docker compose -f infra/docker-compose.machineB.yml run --rm triagent-runner \
  batch \
  --input  /app/datasets/instances.jsonl \
  --output /app/results/<DATE>_<phase>/<backbone>__<ablation>__n100/results.jsonl \
  --backbone <backbone> --provider <provider> --ablation <mode>
```

Sequential multi-job runner: `bash infra/run_experiments.sh` (screen-safe).

---

## Results (all n=100)

| Backbone      | Ablation        | Overall   | Diag  | Substance | Dose  | Gateway | Esc% | Latency |
| ------------- | --------------- | --------- | ----- | --------- | ----- | ------- | ---- | ------- |
| gemma4-local  | no_execution    | **0.500** | 0.950 | 0.280     | 0.230 | 0.560   | 2    | 65s     |
| gemma4-local  | full (seeded)   | 0.443     | 0.960 | 0.260     | 0.210 | 0.510   | 28   | 66s     |
| qwen-local    | no_execution    | 0.442     | 0.940 | 0.220     | 0.190 | 0.330   | 3    | 76s     |
| qwen-local    | full (seeded)   | 0.417     | 0.900 | 0.230     | 0.180 | 0.470   | 28   | 65s     |
| gemma4-local  | no_knowledge    | 0.410     | 0.920 | 0.260     | 0.080 | 0.390   | 19   | 26s     |
| deepseek-chat | full            | 0.380     | 0.820 | 0.160     | 0.130 | 0.540   | 37   | 23s     |
| qwen-local    | no_knowledge    | 0.377     | 0.920 | 0.200     | 0.080 | 0.290   | 23   | 27s     |
| qwen-local    | full (unseeded) | 0.350     | 0.820 | 0.140     | 0.150 | 0.270   | 28   | 71s     |

Scores: mean of {diagnosis_ok, substance_ok, dose_ok, gateway_ok, escalation_ok}
(tools_ok excluded — 0.0 universally, no model invoked tools).

---

## Result folder structure

```
results/
├── 2026-06-10_qwen-matrix/          # Qwen unseeded (3 ablations)
├── 2026-06-11_gemma-deepseek/       # Gemma4 full + DeepSeek full (seeded)
├── 2026-06-11_gemma-ablation/       # Gemma4 no_knowledge + no_execution
├── 2026-06-11_qwen-seeded/          # Qwen full (seeded)
├── aggregated/                      # CSV summaries from evaluate.py
│   ├── *_summary.csv                # (backbone, ablation) → scores
│   └── *_per_process.csv            # breakdown by P1–P5
└── */*/results.jsonl                # raw per-instance results
```

Each `results.jsonl` row: `{run_id, instance_id, process_id, backbone, provider,
ablation, latency_s, result, error}` where `result` contains `{diagnosis, treatment,
reasoning_trace, rule_used, hallucination_flags, escalated, cost_usd}`.

---

## Key findings

**F1 — FAISS pre-seeding is necessary.**
Without seed rules, cold-start causes P3 `full` escalation 50%. With seeding:
Qwen full 0.350 → 0.417, gateway_ok 0.270 → 0.470.

**F2 — full > no_knowledge (main hypothesis) confirmed with seeded FAISS.**
Gemma4: 0.443 > 0.410. Qwen: 0.417 > 0.377.

**F3 — no_execution > full (unexpected).**
Gemma4: 0.500 > 0.443. Qwen: 0.442 > 0.417.
When tools return TOOL_DISABLED, models pivot to parametric knowledge and complete cleanly.
Live execution adds stalling risk on 7-step protocols near the confidence threshold.

**F4 — Gemma4 > Qwen.**
Gateway accuracy: 0.560 vs 0.330 (no_execution). Better multi-step decision following.

**F5 — DeepSeek underperforms local models.**
0.380 overall, 37% escalation. Cloud ≠ best for structured protocol reasoning.

---

## Pending runs

| Backbone      | Ablation                         | Blocker        |
| ------------- | -------------------------------- | -------------- |
| gpt-4o        | full, no_knowledge, no_execution | OpenAI API key |
| llama-3.3-70b | full                             | Groq API key   |

Note for GPT-4o: delete `/app/data/rules` before run (384-dim → 1536-dim FAISS conflict).
