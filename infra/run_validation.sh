#!/bin/bash
# Validation pipeline — FAO document ingestion + tool-invocation test
# Usage (from TriAgent_docs root on the server):
#   screen -dmS val bash infra/run_validation.sh
#   screen -r val   # reattach to watch progress
#
# PREREQUISITE for step 2 (Gemma4): llama.cpp server must already be running
# with gemma4-local loaded. If it isn't:
#   docker compose -f infra/docker-compose.yml stop triagent-llama-cpp-server
#   LCPP_MODEL_REPO="unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL" \
#   LCPP_ALIAS=gemma4-local LCPP_CTX_SIZE=8192 \
#   docker compose -f infra/docker-compose.yml up -d triagent-llama-cpp-server

set -e
COMPOSE="docker compose -f infra/docker-compose.yml"
DATE="2026-06-12"

echo "============================================"
echo " TriAgent validation pipeline"
echo " Started: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"

# ── 1. FAO document ingestion (DeepSeek cloud, no llama.cpp needed) ──────────
echo ""
echo "[1/2] Ingesting FAO/CIMMYT PDF documents into rule store..."
$COMPOSE run --rm --no-deps --entrypoint python triagent-runner \
  experiments/load_fao_docs.py /app/data/fao_documents
echo "[1/2] DONE"

# ── 2. Tool-forcing validation (Gemma4 local, skip-prefetch) ─────────────────
# --skip-prefetch: suppresses the weather pre-fetch so the model must request
# get_weather via requires_tool=true, proving the execution node is reachable.
# No --no-deps: runner must wait for llama.cpp server to be healthy.
echo ""
echo "[2/2] Tool-forcing validation (5 instances, gemma4-local, skip-prefetch)..."
$COMPOSE run --rm triagent-runner \
  batch \
  --input  /app/datasets/tool_forcing_instances.jsonl \
  --output /app/results/${DATE}_tool-validation/gemma4-local__full_skipprefetch__n5/results.jsonl \
  --backbone gemma4-local --provider llamacpp_b --ablation full --skip-prefetch
echo "[2/2] DONE"

echo ""
echo "============================================"
echo " Validation done: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"
echo ""
echo "Check tool invocation:"
echo "  grep '\"requires_tool\": true' results/${DATE}_tool-validation/gemma4-local__full_skipprefetch__n5/results.jsonl"
