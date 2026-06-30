#!/bin/bash
# Sequential experiment runner — v1 old system (Gemma4 ablations + Qwen seeded)
# Survives SSH disconnect when run inside screen/tmux.
#
# Usage:
#   screen -dmS exp bash infra/run_experiments.sh
#   screen -r exp          # reattach to watch progress

set -e
COMPOSE="docker compose -f infra/docker-compose.yml"

echo "============================================"
echo " TriAgent experiment runner (v1 old system)"
echo " Started: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"

# ── 1. Gemma4 no_knowledge ────────────────────────────────────────────────────
echo ""
echo "[1/3] gemma4-local | no_knowledge | 100 instances"
$COMPOSE run --rm triagent-runner \
  batch \
  --input  /app/datasets/instances.jsonl \
  --output /app/results/v1_old_system/gemma4-local__no_knowledge__n100/results.jsonl \
  --backbone gemma4-local --provider llamacpp_b --ablation no_knowledge
echo "[1/3] DONE"

# ── 2. Gemma4 no_execution ────────────────────────────────────────────────────
echo ""
echo "[2/3] gemma4-local | no_execution | 100 instances"
$COMPOSE run --rm triagent-runner \
  batch \
  --input  /app/datasets/instances.jsonl \
  --output /app/results/v1_old_system/gemma4-local__no_execution__n100/results.jsonl \
  --backbone gemma4-local --provider llamacpp_b --ablation no_execution
echo "[2/3] DONE"

# ── 3. Swap Gemma4 → Qwen ─────────────────────────────────────────────────────
echo ""
echo "[swap] Stopping Gemma4 server..."
$COMPOSE stop triagent-llama-cpp-server

echo "[swap] Starting Qwen server (model already cached)..."
$COMPOSE up -d triagent-llama-cpp-server

echo "[swap] Waiting for Qwen server to become healthy..."
CONTAINER=$($COMPOSE ps -q triagent-llama-cpp-server)
until [ "$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER)" = "healthy" ]; do
  echo "  still loading... $(date -u '+%H:%M:%S')"
  sleep 15
done
echo "[swap] Qwen server healthy"

# ── 4. Qwen full (seeded rules) ───────────────────────────────────────────────
echo ""
echo "[3/3] qwen-local | full (seeded) | 100 instances"
$COMPOSE run --rm triagent-runner \
  batch \
  --input  /app/datasets/instances.jsonl \
  --output /app/results/v1_old_system/qwen-local__full_seeded__n100/results.jsonl \
  --backbone qwen-local --provider llamacpp_b --ablation full
echo "[3/3] DONE"

echo ""
echo "============================================"
echo " All experiments done: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"
