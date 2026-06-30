#!/bin/bash
# Qwen new-system ablation (v2)
#
# Factor isolation (new system = CIMMYT PDF FAISS + working tool calling):
#   no_execution          → factor A only  (PDF knowledge, no real tool calls)
#   no_knowledge+skip     → factor B only  (tool calling, FAISS bypassed)
#   full+skip             → A + B combined (full new system)
#
# PREREQUISITE: llama.cpp server running with qwen-local loaded.
#   If gemma4-local is loaded, swap first:
#     docker compose -f infra/docker-compose.yml stop triagent-llama-cpp-server
#     LCPP_MODEL_REPO="unsloth/Qwen3-27B-GGUF:UD-Q4_K_XL" \
#     LCPP_ALIAS=qwen-local LCPP_CTX_SIZE=8192 \
#     docker compose -f infra/docker-compose.yml up -d triagent-llama-cpp-server
#   Then wait for healthcheck before running this script.
#
# Usage:
#   screen -dmS qwen bash infra/run_qwen_newsystem.sh
#   screen -r qwen

set -e
COMPOSE="docker compose -f infra/docker-compose.yml"
INPUT="/app/datasets/instances.jsonl"
OUT_BASE="/app/results/v2_new_system"

echo "============================================"
echo " TriAgent — Qwen new-system ablation (v2)"
echo " FAISS: CIMMYT PDF rules"
echo " Tool calling: --skip-prefetch active"
echo " Started: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"

# ── 1. no_execution: factor A only — PDF knowledge, no real tool calls ────────
echo ""
echo "[1/3] qwen | no_execution | CIMMYT FAISS | n100..."
$COMPOSE run --rm triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/qwen-local__no_execution__n100/results.jsonl" \
  --backbone qwen-local --provider llamacpp_b --ablation no_execution
echo "[1/3] DONE"

# ── 2. no_knowledge + skip-prefetch: factor B only ───────────────────────────
echo ""
echo "[2/3] qwen | no_knowledge + skip-prefetch | n100..."
$COMPOSE run --rm triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/qwen-local__no_knowledge_skipprefetch__n100/results.jsonl" \
  --backbone qwen-local --provider llamacpp_b --ablation no_knowledge --skip-prefetch
echo "[2/3] DONE"

# ── 3. full + skip-prefetch: A + B combined ──────────────────────────────────
echo ""
echo "[3/3] qwen | full + skip-prefetch | CIMMYT FAISS | n100..."
$COMPOSE run --rm triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/qwen-local__full_skipprefetch__n100/results.jsonl" \
  --backbone qwen-local --provider llamacpp_b --ablation full --skip-prefetch
echo "[3/3] DONE"

echo ""
echo "============================================"
echo " Done: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"
echo ""
echo "Tool invocation check (full+skip):"
echo "  grep -c '\"observation\":' ${OUT_BASE}/qwen-local__full_skipprefetch__n100/results.jsonl"
