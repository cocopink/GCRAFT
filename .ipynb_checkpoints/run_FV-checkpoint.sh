# API Configuration
base_url="${HAOMIAO_URL:-https://113.44.247.131:47851/v1}"
openai_api_key="${HAOMIAO_AUTHEN_TOKEN:-${HAOMIAO_AUTH_TOKEN:-$ANTHROPIC_AUTH_TOKEN}}"
model_name="${MODEL_NAME:-gpt-5.4}"

first_n=-1
n_proc=1
chunk_size=4

# Mode switch: set to "orig" for original mode, "new" for new mode
MODE="${MODE:-new}"

# ============== TableAnalyzer Configuration ==============
USE_TABLE_ANALYZER="true"  # Set to "false" to skip analysis stage

# ============== Graph Enhancement (P1+P2) ==============
ENABLE_P1="${ENABLE_P1:-true}"     # P1: ATGO gamma smoothing for row reranking
ENABLE_P2="${ENABLE_P2:-true}"     # P2: inject graph hints for Refine stage
P1_GAMMA="${P1_GAMMA:-0.1}"       # P1 gamma value (0.0 disables smoothing)

# ============== Results Path Configuration ==============
RESULTS_BASE="test/results"
DATA_DIR="thought/TableFV/data/tabfact"
ORIGINAL_DATA="${DATA_DIR}/test.jsonl"
DATASET_TO_USE="$ORIGINAL_DATA"

ANALYSIS_DATA="${DATA_DIR}/test_analyzed.jsonl"
ATG_RERANK_DATA="${DATA_DIR}/test_reranked.jsonl"
ATGO_ROW_DATA="${DATA_DIR}/test_reranked_row.jsonl"
ATGC_COL_DATA="${DATA_DIR}/test_reranked_col.jsonl"
case "$MODE" in
    orig) MODE_SUFFIX="_orig" ;;
    new) MODE_SUFFIX="" ;;
    *) MODE_SUFFIX="_${MODE}" ;;
esac
THOUGHT_RESULTS="${RESULTS_BASE}/thought/tabfact/${model_name}${MODE_SUFFIX}"
REFINE_RESULTS="${RESULTS_BASE}/refine/tabfact/${model_name}${MODE_SUFFIX}"

# ============== Logging Setup ==============
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SCRIPT_NAME=$(basename "$0" .sh)
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}-${model_name}-${TIMESTAMP}.log"

# Write parameter header to log
_masked_key="${openai_api_key:0:8}...${openai_api_key: -4}"
cat > "$LOG_FILE" <<EOF
=== Run Parameters (${TIMESTAMP}) ===
base_url           = ${base_url}
openai_api_key     = ${_masked_key}
model_name         = ${model_name}
first_n            = ${first_n}
n_proc             = ${n_proc}
chunk_size         = ${chunk_size}
MODE               = ${MODE}
USE_TABLE_ANALYZER = ${USE_TABLE_ANALYZER}
ENABLE_P1          = ${ENABLE_P1} (gamma=${P1_GAMMA})
ENABLE_P2          = ${ENABLE_P2}
GRAPH_SUFFIX       = ${GRAPH_SUFFIX}
RESULTS_BASE       = ${RESULTS_BASE}
ORIGINAL_DATA      = ${ORIGINAL_DATA}
ATG_RERANK_DATA    = ${ATG_RERANK_DATA}
ATGO_ROW_DATA      = ${ATGO_ROW_DATA}
ATGC_COL_DATA      = ${ATGC_COL_DATA}
THOUGHT_RESULTS    = ${THOUGHT_RESULTS}
REFINE_RESULTS     = ${REFINE_RESULTS}
========================================
EOF

# Validate P1/P2 combination: only both-on or both-off are allowed
if [ "$ENABLE_P1" != "$ENABLE_P2" ]; then
    echo "❌ Error: ENABLE_P1 and ENABLE_P2 must be both true or both false."
    echo "   Partial combinations (P1-only or P2-only) are not supported."
    echo "   See docs/graph_experiment_analysis_report.md for experiment details."
    exit 1
fi

# Determine result directory suffix based on P1/P2 state
if [ "$ENABLE_P1" = "true" ]; then
    GRAPH_SUFFIX="_p12"
else
    GRAPH_SUFFIX=""
fi

echo "📝 Log: ${LOG_FILE}"

# Redirect all subsequent output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "📊 Running TableFV Pipeline (ATG + QG-PPR)"
echo "=========================================="

# ============== Stage 0.7a: ATGO Row-only Reranking (P1+P2) ==============
if [ ! -f "$ATGO_ROW_DATA" ] || [ "$ATGO_ROW_DATA" -ot "$ORIGINAL_DATA" ]; then
    echo ""
    echo "Stage 0.7a: ATGO Row-only Reranking (P1=${ENABLE_P1}, P2=${ENABLE_P2})"
    echo "------------------------------------------"
    python preprocess_atgo.py \
        --mode row \
        --dataset_path "$ORIGINAL_DATA" \
        --output_path "$ATGO_ROW_DATA" \
        --gamma $([ "$ENABLE_P1" = "true" ] && echo "$P1_GAMMA" || echo "0.0") \
        --inject_graph_hint $([ "$ENABLE_P2" = "true" ] && echo "True" || echo "False")

    if [ $? -ne 0 ]; then
        echo "❌ Error in ATGO row-only reranking stage"
        exit 1
    fi
    echo "✅ ATGO row-only reranking complete!"
else
    echo "✅ Using existing ATGO row-only reranking: $ATGO_ROW_DATA"
fi

# ============== Stage 1-ATGO: Thought (Row-only, P1+P2) ==============
echo ""
echo "Stage 1-ATGO: Thought (Row-only)"
echo "------------------------------------------"
python thought/TableFV/main.py \
--dataset_path "$ATGO_ROW_DATA" \
--thought_results_dir "${THOUGHT_RESULTS}${GRAPH_SUFFIX}" \
--base_url $base_url \
--openai_api_key $openai_api_key \
--model_name $model_name \
--first_n $first_n \
--n_proc $n_proc \
--chunk_size $chunk_size \
--use_clarifier $([ "$MODE" = "new" ] && echo "True" || echo "False")

if [ $? -ne 0 ]; then
    echo "Error in Stage 1-ATGO"
    exit 1
fi

# ============== Stage 2: Refine (P1+P2, Controller-driven) ==============
echo ""
echo "Stage 2: Refine (Controller-driven)"
echo "------------------------------------------"
python refine/TableFV/main_tree_based.py \
--thought_results_dir "${THOUGHT_RESULTS}${GRAPH_SUFFIX}" \
--refine_results_dir "${REFINE_RESULTS}${GRAPH_SUFFIX}" \
--base_url $base_url \
--openai_api_key $openai_api_key \
--model_name $model_name \
--first_n $first_n \
--n_proc $n_proc \
--chunk_size $chunk_size \
--use_clarifier $([ "$MODE" = "new" ] && echo "True" || echo "False")

if [ $? -ne 0 ]; then
    echo "Error in Stage 2 Refine"
    exit 1
fi

# ============== Stage 0.7b: ATGC Col-only Reranking (optional) ==============
if [ "$ENABLE_ATGC" = "true" ]; then
    if [ ! -f "$ATGC_COL_DATA" ] || [ "$ATGC_COL_DATA" -ot "$ORIGINAL_DATA" ]; then
        echo ""
        echo "Stage 0.7b: ATGC Col-only Reranking (optional)"
        echo "------------------------------------------"
        python preprocess_atgo.py \
            --mode col \
            --dataset_path "$ORIGINAL_DATA" \
            --output_path "$ATGC_COL_DATA"

        if [ $? -ne 0 ]; then
            echo "❌ Error in ATGC col-only reranking stage"
            exit 1
        fi
        echo "✅ ATGC col-only reranking complete!"
    else
        echo "✅ Using existing ATGC col-only reranking: $ATGC_COL_DATA"
    fi

    # ============== Stage 1-ATGC: Thought (Col-only, optional) ==============
    echo ""
    echo "Stage 1-ATGC: Thought (Col-only)"
    echo "------------------------------------------"
    python thought/TableFV/main.py \
    --dataset_path "$ATGC_COL_DATA" \
    --thought_results_dir "${THOUGHT_RESULTS}_col" \
    --base_url $base_url \
    --openai_api_key $openai_api_key \
    --model_name $model_name \
    --first_n $first_n \
    --n_proc $n_proc \
    --chunk_size $chunk_size \
    --use_clarifier $([ "$MODE" = "new" ] && echo "True" || echo "False")

    if [ $? -ne 0 ]; then
        echo "Error in Stage 1-ATGC"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results: $RESULTS_BASE"
