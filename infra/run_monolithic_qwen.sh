#!/bin/bash
# Monolithic baseline — Qwen3-27B
#
# Single-shot LLM call per instance: no FAISS, no ReAct, no agents.
# Used as empirical baseline for Theorems 1 & 2 (monolithic vs decomposed).
#
# PREREQUISITE: llama.cpp server running with qwen-local loaded.
# Evaluate with:
#   python experiments/evaluate.py \
#     --results results/monolithic_baseline \
#     --gt datasets/ground_truth.jsonl
#
# Usage:
#   screen -dmS mono_qwen bash infra/run_monolithic_qwen.sh
#   screen -r mono_qwen

set -e
COMPOSE="docker compose -f infra/docker-compose.yml"
INPUT="/app/datasets/instances.jsonl"
OUT_BASE="/app/results/monolithic_baseline"

echo "============================================"
echo " TriAgent — monolithic baseline (Qwen)"
echo " No FAISS, no ReAct, single prompt per case"
echo " Started: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"

echo ""
echo "[1/1] qwen | monolithic | n100..."
$COMPOSE run --rm triagent-runner \
  monolithic \
  --input  "$INPUT" \
  --output "${OUT_BASE}/qwen-local__monolithic__n100/results.jsonl" \
  --backbone qwen-local --provider llamacpp_b
echo "[1/1] DONE"

echo ""
echo "============================================"
echo " Done: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"
