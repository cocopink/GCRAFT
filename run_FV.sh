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

# ============== Graph Enhancement (P1-P6, 可开关) ==============
ENABLE_ATGO="${ENABLE_ATGO:-true}"  # ATGO: Stage 0.7 QG-PPR 行重排本体 (false=raw 表对照, 跳过 Stage 0.7)
ENABLE_P1="${ENABLE_P1:-true}"      # P1: ATGO gamma smoothing for row reranking
ENABLE_P2="${ENABLE_P2:-true}"      # P2: inject graph hints for Refine stage
P1_GAMMA="${P1_GAMMA:-0.1}"         # P1 gamma value (0.0 disables smoothing)
ENABLE_P3="${ENABLE_P3:-false}"     # P3: graph candidate bias for Refine select_row/select_column
P3_BIAS_WEIGHT="${P3_BIAS_WEIGHT:-1.0}"  # P3 bias strength (运行参数注入, 见 docs/modules_p1_p6.md §4)
ENABLE_P6="${ENABLE_P6:-false}"     # P6: graph-grounded judge (一期 CONFLICT/UNKNOWN, 见 docs/modules_p1_p6.md §7)
ENABLE_ATGC="${ENABLE_ATGC:-false}" # ATGC: optional col-only rerank control experiment

# Validate switch combinations
if [ "$ENABLE_ATGO" != "true" ]; then
    for _s in "$ENABLE_P1" "$ENABLE_P2" "$ENABLE_P3" "$ENABLE_P6"; do
        if [ "$_s" = "true" ]; then
            echo "❌ Error: ENABLE_ATGO=false (raw 对照) 时 P1/P2/P3/P6 不可开启 (无图数据可用)."
            exit 1
        fi
    done
fi
if [ "$ENABLE_P3" = "true" ] && { [ "$ENABLE_P1" != "true" ] || [ "$ENABLE_P2" != "true" ]; }; then
    echo "❌ Error: ENABLE_P3=true 需要 ENABLE_P1 与 ENABLE_P2 同时开启 (E2 最强配置叠加位, 见 docs/experiment_plan_p1_p6.md §2.2)."
    exit 1
fi

# ============== Results Path Configuration ==============
RESULTS_BASE="test/results"
DATA_DIR="thought/TableFV/data/tabfact"
ORIGINAL_DATA="${DATA_DIR}/test.jsonl"

case "$MODE" in
    orig) MODE_SUFFIX="_orig" ;;
    new) MODE_SUFFIX="" ;;
    *) MODE_SUFFIX="_${MODE}" ;;
esac
THOUGHT_RESULTS="${RESULTS_BASE}/thought/tabfact/${model_name}${MODE_SUFFIX}"
REFINE_RESULTS="${RESULTS_BASE}/refine/tabfact/${model_name}${MODE_SUFFIX}"

# Determine rerank tag from ATGO/P1/P2/P3 combination (Stage 0.7 输出文件按配置分名, 防止串写)
RERANK_TAG=""
if [ "$ENABLE_ATGO" != "true" ]; then
    RERANK_TAG="_raw"
else
    if [ "$ENABLE_P1" = "true" ] && [ "$ENABLE_P2" = "true" ]; then
        RERANK_TAG="_p12"
    elif [ "$ENABLE_P1" = "true" ]; then
        RERANK_TAG="_p1"
    elif [ "$ENABLE_P2" = "true" ]; then
        RERANK_TAG="_p2"
    fi
    if [ "$ENABLE_P3" = "true" ]; then
        if [ "$RERANK_TAG" = "_p12" ]; then RERANK_TAG="_p123"; else RERANK_TAG="${RERANK_TAG}_p3"; fi
    fi
fi
ATGO_ROW_DATA="${DATA_DIR}/test_reranked_row${RERANK_TAG}.jsonl"
ATGC_COL_DATA="${DATA_DIR}/test_reranked_col.jsonl"

# Thought 输入: ATGO 开启走重排数据, 关闭走 raw 原始表 (refine/thought 的 table_text 回落链天然兼容)
if [ "$ENABLE_ATGO" = "true" ]; then
    THOUGHT_INPUT="$ATGO_ROW_DATA"
else
    THOUGHT_INPUT="$ORIGINAL_DATA"
fi

# Result directory suffix = rerank tag + P6 (P6 只影响 Refine, 不影响重排)
if [ "$ENABLE_P6" = "true" ]; then
    GRAPH_SUFFIX="${RERANK_TAG}_p6"
else
    GRAPH_SUFFIX="${RERANK_TAG}"
fi

# ============== Error Tree Evolution (每实验独立进化树) ==============
# False = 进化模式: 从 canonical 播种本实验私有树并在运行内进化
# True  = 静态树消融: 读 canonical, 永不写
FROZEN_MEMORY="${FROZEN_MEMORY:-False}"
EVOLVE_TREE_JSON="${REFINE_RESULTS}${GRAPH_SUFFIX}/memory/few_shot_critic.json"

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
ENABLE_ATGO        = ${ENABLE_ATGO}
ENABLE_P1          = ${ENABLE_P1} (gamma=${P1_GAMMA})
ENABLE_P2          = ${ENABLE_P2}
ENABLE_P3          = ${ENABLE_P3} (bias_weight=${P3_BIAS_WEIGHT})
ENABLE_P6          = ${ENABLE_P6}
ENABLE_ATGC        = ${ENABLE_ATGC}
GRAPH_SUFFIX       = ${GRAPH_SUFFIX}
RESULTS_BASE       = ${RESULTS_BASE}
ORIGINAL_DATA      = ${ORIGINAL_DATA}
THOUGHT_INPUT      = ${THOUGHT_INPUT}
ATGO_ROW_DATA      = ${ATGO_ROW_DATA}
ATGC_COL_DATA      = ${ATGC_COL_DATA}
THOUGHT_RESULTS    = ${THOUGHT_RESULTS}
REFINE_RESULTS     = ${REFINE_RESULTS}
========================================
EOF

echo "📝 Log: ${LOG_FILE}"

# Redirect all subsequent output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "📊 Running TableFV Pipeline (ATG + QG-PPR)"
echo "=========================================="

# ============== Stage 0.7a: ATGO Row-only Reranking (P1+P2+P3) ==============
if [ "$ENABLE_ATGO" = "true" ]; then
    if [ ! -f "$ATGO_ROW_DATA" ] || [ "$ATGO_ROW_DATA" -ot "$ORIGINAL_DATA" ]; then
        echo ""
        echo "Stage 0.7a: ATGO Row-only Reranking (P1=${ENABLE_P1}, P2=${ENABLE_P2}, P3=${ENABLE_P3})"
        echo "------------------------------------------"
        python preprocess_atgo.py \
            --mode row \
            --dataset_path "$ORIGINAL_DATA" \
            --output_path "$ATGO_ROW_DATA" \
            --gamma $([ "$ENABLE_P1" = "true" ] && echo "$P1_GAMMA" || echo "0.0") \
            --inject_graph_hint $([ "$ENABLE_P2" = "true" ] && echo "True" || echo "False") \
            --bias_weight $([ "$ENABLE_P3" = "true" ] && echo "$P3_BIAS_WEIGHT" || echo "0.0")

        if [ $? -ne 0 ]; then
            echo "❌ Error in ATGO row-only reranking stage"
            exit 1
        fi
        echo "✅ ATGO row-only reranking complete!"
    else
        echo "✅ Using existing ATGO row-only reranking: $ATGO_ROW_DATA"
    fi
else
    echo "ℹ️ ENABLE_ATGO=false: raw 对照模式, 跳过 Stage 0.7 (Thought 输入 = $ORIGINAL_DATA)"
fi

# ============== Stage 1-ATGO: Thought (Row-only, P1+P2) ==============
echo ""
echo "Stage 1-ATGO: Thought (Row-only)"
echo "------------------------------------------"
python thought/TableFV/main.py \
--dataset_path "$THOUGHT_INPUT" \
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

# ============== Stage 2: Refine (P1+P2+P3+P6, Controller-driven) ==============
echo ""
echo "Stage 2: Refine (Controller-driven)"
echo "------------------------------------------"
# 独立进化树: frozen 时读 canonical 静态树, 否则播种/续用本实验私有树
EVOLVE_ARGS=""
if [ "$FROZEN_MEMORY" != "True" ]; then
    EVOLVE_ARGS="--critic_tree_json $EVOLVE_TREE_JSON"
fi
python refine/TableFV/main_tree_based.py \
--thought_results_dir "${THOUGHT_RESULTS}${GRAPH_SUFFIX}" \
--refine_results_dir "${REFINE_RESULTS}${GRAPH_SUFFIX}" \
--base_url $base_url \
--openai_api_key $openai_api_key \
--model_name $model_name \
--first_n $first_n \
--n_proc $n_proc \
--chunk_size $chunk_size \
--use_clarifier $([ "$MODE" = "new" ] && echo "True" || echo "False") \
--frozen_memory $FROZEN_MEMORY \
--bias_weight $([ "$ENABLE_P3" = "true" ] && echo "$P3_BIAS_WEIGHT" || echo "0.0") \
--enable_p6 $([ "$ENABLE_P6" = "true" ] && echo "True" || echo "False") \
$EVOLVE_ARGS

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
