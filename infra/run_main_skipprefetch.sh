#!/bin/bash
# Full-system ablation runs — Gemma4 + DeepSeek (v2 new system)
#
# Two factors changed vs v1:
#   (A) FAISS rule store populated from real CIMMYT PDF (load_fao_docs.py)
#   (B) --skip-prefetch + updated system prompt → tool calling via execution node
#
# Factor isolation design (Gemma4):
#   no_execution          → factor A only  (PDF knowledge, no real tool calls)
#   no_knowledge+skip     → factor B only  (tool calling, FAISS bypassed)
#   full+skip             → A + B combined (full new system)
#
# PREREQUISITE: llama.cpp server running with gemma4-local loaded.
# Usage:
#   screen -dmS main bash infra/run_main_skipprefetch.sh
#   screen -r main

set -e
COMPOSE="docker compose -f infra/docker-compose.yml"
INPUT="/app/datasets/instances.jsonl"
OUT_BASE="/app/results/v2_new_system"

echo "============================================"
echo " TriAgent full-system ablation (v2)"
echo " FAISS: CIMMYT PDF rules"
echo " Tool calling: --skip-prefetch active"
echo " Started: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"

# ── 1. no_execution: factor A only — PDF knowledge, no real tool calls ────────
echo ""
echo "[1/6] gemma4 | no_execution | CIMMYT FAISS | n100..."
$COMPOSE run --rm triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/gemma4-local__no_execution__n100/results.jsonl" \
  --backbone gemma4-local --provider llamacpp_b --ablation no_execution
echo "[1/6] DONE"

# ── 2. no_knowledge + skip-prefetch: factor B only — tool calling, no FAISS ──
echo ""
echo "[2/6] gemma4 | no_knowledge + skip-prefetch | n100..."
$COMPOSE run --rm triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/gemma4-local__no_knowledge_skipprefetch__n100/results.jsonl" \
  --backbone gemma4-local --provider llamacpp_b --ablation no_knowledge --skip-prefetch
echo "[2/6] DONE"

# ── 3. full + skip-prefetch: A + B combined — full new system ────────────────
echo ""
echo "[3/6] gemma4 | full + skip-prefetch | CIMMYT FAISS | n100..."
$COMPOSE run --rm triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/gemma4-local__full_skipprefetch__n100/results.jsonl" \
  --backbone gemma4-local --provider llamacpp_b --ablation full --skip-prefetch
echo "[3/6] DONE"

# ── 4. DeepSeek no_execution: factor A only (CIMMYT PDF, no tool calls) ───────
echo ""
echo "[4/6] deepseek | no_execution | CIMMYT FAISS | n100..."
$COMPOSE run --rm --no-deps triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/deepseek-chat__no_execution__n100/results.jsonl" \
  --backbone deepseek-chat --provider deepseek --ablation no_execution
echo "[4/6] DONE"

# ── 5. DeepSeek no_knowledge + skip-prefetch: factor B only ──────────────────
echo ""
echo "[5/6] deepseek | no_knowledge + skip-prefetch | n100..."
$COMPOSE run --rm --no-deps triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/deepseek-chat__no_knowledge_skipprefetch__n100/results.jsonl" \
  --backbone deepseek-chat --provider deepseek --ablation no_knowledge --skip-prefetch
echo "[5/6] DONE"

# ── 6. DeepSeek full + skip-prefetch: A + B combined ─────────────────────────
echo ""
echo "[6/6] deepseek | full + skip-prefetch | CIMMYT FAISS | n100..."
$COMPOSE run --rm --no-deps triagent-runner \
  batch \
  --input  "$INPUT" \
  --output "${OUT_BASE}/deepseek-chat__full_skipprefetch__n100/results.jsonl" \
  --backbone deepseek-chat --provider deepseek --ablation full --skip-prefetch
echo "[6/6] DONE"

echo ""
echo "============================================"
echo " Done: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "============================================"
echo ""
echo "Tool invocation check (gemma4 full+skip):"
echo "  grep -c '\"observation\":' ${OUT_BASE}/gemma4-local__full_skipprefetch__n100/results.jsonl"
echo "Tool invocation check (deepseek full+skip):"
echo "  grep -c '\"observation\":' ${OUT_BASE}/deepseek-chat__full_skipprefetch__n100/results.jsonl"
