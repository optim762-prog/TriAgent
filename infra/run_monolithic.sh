#!/bin/bash
# Monolithic baseline — Gemma4 + DeepSeek
#
# Single-shot LLM call per instance: no FAISS, no ReAct, no agents.
# Used as empirical baseline for Theorems 1 & 2 (monolithic vs decomposed).
#
# PREREQUISITE: llama.cpp server running with gemma4-local loaded.
# Evaluate with:
#   python experiments/evaluate.py \
#     --results results/monolithic_baseline \
#     --gt datasets/ground_truth.jsonl
#
# Usage:
#   screen -dmS mono bash infra/run_monolithic.sh
#   screen -r mono

set -e
COMPOSE="docker compose -f infra/docker-compose.yml"
INPUT="/app/datasets/instances.jsonl"
OUT_BASE="/app/results/monolithic_baseline"

echo "============================================"
echo " TriAgent — monolithic baseline"
echo " No FAISS, no ReAct, single prompt per case"
echo " Started: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"

echo ""
echo "[1/2] gemma4 | monolithic | n100..."
$COMPOSE run --rm triagent-runner \
  monolithic \
  --input  "$INPUT" \
  --output "${OUT_BASE}/gemma4-local__monolithic__n100/results.jsonl" \
  --backbone gemma4-local --provider llamacpp_b
echo "[1/2] DONE"

echo ""
echo "[2/2] deepseek | monolithic | n100..."
$COMPOSE run --rm --no-deps triagent-runner \
  monolithic \
  --input  "$INPUT" \
  --output "${OUT_BASE}/deepseek-chat__monolithic__n100/results.jsonl" \
  --backbone deepseek-chat --provider deepseek
echo "[2/2] DONE"

echo ""
echo "============================================"
echo " Done: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"
