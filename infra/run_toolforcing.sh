#!/bin/bash
# Tool-forcing evaluation — Gemma4 + DeepSeek
#
# Runs on the 100-instance tool-forcing dataset.
# All instances have NO embedded weather data; forcing phrase requires get_weather.
# Ablation: full + --skip-prefetch (live tool execution active).
#
# Evaluate with:
#   python experiments/evaluate.py \
#     --results results/tool_forcing \
#     --gt datasets/ground_truth_toolforcing_n100.jsonl
#
# Usage:
#   screen -dmS tf bash infra/run_toolforcing.sh
#   screen -r tf

set -e
COMPOSE="docker compose -f infra/docker-compose.yml"
INPUT="/app/datasets/tool_forcing_instances_n100.jsonl"
OUT_BASE="/app/results/tool_forcing"

echo "============================================"
echo " TriAgent tool-forcing evaluation (n=100)"
echo " Dataset: tool_forcing_instances_n100.jsonl"
echo " Ablation: full + --skip-prefetch"
echo " Started: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"

# ── 1. Gemma4 full + skip-prefetch ───────────────────────────────────────────
echo ""
echo "[1/2] gemma4 | full + skip-prefetch | tool-forcing n100..."
$COMPOSE run --rm triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/gemma4-local__full_skipprefetch__n100/results.jsonl" \
  --backbone gemma4-local --provider llamacpp_b --ablation full --skip-prefetch
echo "[1/2] DONE"

# ── 2. DeepSeek full + skip-prefetch ─────────────────────────────────────────
echo ""
echo "[2/2] deepseek | full + skip-prefetch | tool-forcing n100..."
$COMPOSE run --rm --no-deps triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/deepseek-chat__full_skipprefetch__n100/results.jsonl" \
  --backbone deepseek-chat --provider deepseek --ablation full --skip-prefetch
echo "[2/2] DONE"

echo ""
echo "============================================"
echo " Done: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"
echo ""
echo "Verify tool invocation (expect ~100 lines with observation):"
echo "  python -c \""
echo "  import json; f=open('${OUT_BASE}/gemma4-local__full_skipprefetch__n100/results.jsonl')"
echo "  rows=[json.loads(l) for l in f]"
echo "  obs=[r for r in rows if any(s.get('observation') for s in (r.get('result') or {}).get('reasoning_trace') or [])]"
echo "  print(f'gemma4: {len(obs)}/100 with real observation')\""
