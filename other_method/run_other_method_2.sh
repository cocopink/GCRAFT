#!/bin/bash
# ===========================================================================
# Run all baseline methods on WikiTQ and TabFact
# ===========================================================================

MODEL="${MODEL:-glm-5.1}" # glm-5.1
BASE_URL="${BASE_URL:-https://113.44.247.131:47851/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-$ANTHROPIC_AUTH_TOKEN}"
N_PROC="${N_PROC:-1}"
FIRST_N="${FIRST_N:--1}"
COT_CONSIST_N="${COT_CONSIST_N:-4}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$SCRIPT_DIR/other_method/run_baseline.py"

echo "============================================"
echo " Model:    $MODEL"
echo " Base URL: $BASE_URL"
echo " N_proc:   $N_PROC"
echo " First_n:  $FIRST_N"
echo " CoT-Consist N: $COT_CONSIST_N"
echo "============================================"

# ---- TabFact ----
echo ""
echo "========== TabFact =========="

for METHOD in   cot cot_consist; do # 
    echo ""
    echo "--- TabFact / $METHOD ---"
    python "$SCRIPT" \
        --dataset tabfact \
        --method "$METHOD" \
        --model "$MODEL" \
        --base_url "$BASE_URL" \
        --api_key "$OPENAI_API_KEY" \
        --first_n "$FIRST_N" \
        --n_proc "$N_PROC" \
        --n_sample "$COT_CONSIST_N"
done

# ---- WikiTQ ----
echo ""
echo "========== WikiTQ =========="

for METHOD in e2e few_shot cot cot_consist; do # e2e few_shot cot 
    echo ""
    echo "--- WikiTQ / $METHOD ---"
    python "$SCRIPT" \
        --dataset wikitq \
        --method "$METHOD" \
        --model "$MODEL" \
        --base_url "$BASE_URL" \
        --api_key "$OPENAI_API_KEY" \
        --first_n "$FIRST_N" \
        --n_proc "$N_PROC" \
        --n_sample "$COT_CONSIST_N"
done

echo ""
echo "============================================"
echo " All baselines completed!"
echo " Results saved to results/other_method/"
echo "============================================"
