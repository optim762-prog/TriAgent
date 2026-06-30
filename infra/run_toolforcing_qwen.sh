#!/bin/bash
# Tool-forcing evaluation — Qwen
#
# Run AFTER swapping llama.cpp server to Qwen model:
#   docker compose -f infra/docker-compose.yml stop triagent-llama-cpp-server
#   # change LCPP_ALIAS / LCPP_MODEL_REPO in .env for qwen-local
#   docker compose -f infra/docker-compose.yml up -d triagent-llama-cpp-server
#   # wait for /health to return 200
#
# Evaluate together with Gemma4+DeepSeek:
#   python experiments/evaluate.py \
#     --results results/tool_forcing \
#     --gt datasets/ground_truth_toolforcing_n100.jsonl
#
# Usage:
#   screen -dmS tfq bash infra/run_toolforcing_qwen.sh
#   screen -r tfq

set -e
COMPOSE="docker compose -f infra/docker-compose.yml"
INPUT="/app/datasets/tool_forcing_instances_n100.jsonl"
OUT_BASE="/app/results/tool_forcing"

echo "============================================"
echo " TriAgent tool-forcing evaluation — Qwen"
echo " Dataset: tool_forcing_instances_n100.jsonl"
echo " Ablation: full + --skip-prefetch"
echo " Started: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"

echo ""
echo "[1/1] qwen | full + skip-prefetch | tool-forcing n100..."
$COMPOSE run --rm triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/qwen-local__full_skipprefetch__n100/results.jsonl" \
  --backbone qwen-local --provider llamacpp_b --ablation full --skip-prefetch
echo "[1/1] DONE"

echo ""
echo "============================================"
echo " Done: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"
