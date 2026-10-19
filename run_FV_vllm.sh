#!/usr/bin/env bash
# run_FV_vllm.sh — TableFV 主线（本地 vLLM 后端）
#
# 基于 run_FV.sh，推理后端改为本地 vLLM 服务（OpenAI 兼容），不使用 ollama。
# 管线逻辑与 run_FV.sh 完全一致：Stage 0.7a ATGO 行重排 → Stage 1 Thought → Stage 2 Refine（→ 可选 ATGC）。
#
# 用法:
#   bash download_model.sh                      # 首次先下载 HF 模型
#   bash run_FV_vllm.sh                         # 自动拉起 vllm serve 并执行管线
#   AUTO_VLLM_SERVER=false bash run_FV_vllm.sh  # vLLM 服务已手动启动时，仅连接
#
# 可调环境变量:
#   VLLM_MODEL=Qwen/Qwen3.5-4B          # HF 模型名；vLLM 启动失败时换
#                                       #   Qwen/Qwen3-4B-Instruct-2507（见 download_model.sh 兼容性注）
#   VLLM_HOST=127.0.0.1 VLLM_PORT=8000  # 服务地址
#   VLLM_MAX_MODEL_LEN=8192             # 表格线性化后 prompt 较长，默认 8192
#   AUTO_VLLM_SERVER=true               # 端口无服务时自动 vllm serve
#   MODE=new|orig  ENABLE_P1=true  ENABLE_P2=true  P1_GAMMA=0.1  ENABLE_ATGC=false

set -u

# ============== vLLM Backend Configuration ==============
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-4B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
AUTO_VLLM_SERVER="${AUTO_VLLM_SERVER:-true}"

# served-model-name 用小写短名：既作 API model 参数（llm.py 依赖小写 qwen 前缀
# 注入 chat_template_kwargs 关思考），也作结果目录名（避免 HF 全名中的 / 拆目录）
SERVED_NAME="${SERVED_NAME:-$(echo "${VLLM_MODEL##*/}" | tr 'A-Z' 'a-z')}"

base_url="http://${VLLM_HOST}:${VLLM_PORT}/v1"
openai_api_key="${VLLM_API_KEY:-EMPTY}"   # vLLM 默认不校验 key，客户端要求非空
model_name="${SERVED_NAME}"

first_n=-1
n_proc="${N_PROC:-8}"          # 客户端并发进程数：调大可吃满 vLLM continuous batching
chunk_size="${CHUNK_SIZE:-4}"

# Mode switch: set to "orig" for original mode, "new" for new mode
MODE="${MODE:-new}"

# ============== TableAnalyzer Configuration ==============
USE_TABLE_ANALYZER="true"  # Set to "false" to skip analysis stage

# ============== Graph Enhancement (P1+P2) ==============
ENABLE_P1="${ENABLE_P1:-true}"     # P1: ATGO gamma smoothing for row reranking
ENABLE_P2="${ENABLE_P2:-true}"     # P2: inject graph hints for Refine stage
P1_GAMMA="${P1_GAMMA:-0.1}"       # P1 gamma value (0.0 disables smoothing)
ENABLE_ATGC="${ENABLE_ATGC:-false}" # ATGC: optional col-only rerank control experiment

# ============== Results Path Configuration ==============
RESULTS_BASE="${RESULTS_BASE:-test/results}"
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
backend            = vllm
vllm_model         = ${VLLM_MODEL}
served_name        = ${SERVED_NAME}
base_url           = ${base_url}
openai_api_key     = ${_masked_key}
model_name         = ${model_name}
auto_vllm_server   = ${AUTO_VLLM_SERVER}
max_model_len      = ${VLLM_MAX_MODEL_LEN}
dtype              = ${VLLM_DTYPE}
gpu_mem_util       = ${VLLM_GPU_MEMORY_UTILIZATION}
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

echo "📝 Log: ${LOG_FILE}"

# Redirect all subsequent output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# ============== vLLM Server Management ==============
VLLM_PID_FILE="${LOG_DIR}/vllm_${model_name}.pid"

wait_for_vllm() {
    local url="$1"
    local attempts=0
    while [ $attempts -lt 300 ]; do
        if curl -sf "${url}/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        attempts=$((attempts + 1))
    done
    return 1
}

start_vllm_server() {
    if ! command -v vllm >/dev/null 2>&1; then
        echo "❌ Error: vllm command not found. Install with: pip install vllm"
        exit 1
    fi
    local server_log="${LOG_DIR}/vllm_${model_name}_${TIMESTAMP}.log"
    echo "🚀 Starting vLLM server: ${VLLM_MODEL} (served as ${SERVED_NAME})"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    nohup vllm serve "${VLLM_MODEL}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --served-model-name "${SERVED_NAME}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --dtype "${VLLM_DTYPE}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        > "${server_log}" 2>&1 &
    echo $! > "${VLLM_PID_FILE}"
    echo "   pid=$(cat "${VLLM_PID_FILE}")  log=${server_log}"
    echo "   Waiting for server to be ready (up to 300s)..."
    if ! wait_for_vllm "${base_url}"; then
        echo "❌ Error: vLLM server failed to start for ${VLLM_MODEL}"
        tail -n 80 "${server_log}" || true
        exit 1
    fi
    echo "✅ vLLM server ready: ${base_url}"
}

ensure_vllm_server() {
    if curl -sf "${base_url}/models" >/dev/null 2>&1; then
        echo "✅ Using existing vLLM server: ${base_url}"
        return 0
    fi
    if [ "$AUTO_VLLM_SERVER" != "true" ]; then
        echo "❌ Error: no vLLM server at ${base_url} and AUTO_VLLM_SERVER=${AUTO_VLLM_SERVER}."
        echo "   Start it manually, e.g.:"
        echo "   vllm serve ${VLLM_MODEL} --host ${VLLM_HOST} --port ${VLLM_PORT} \\"
        echo "       --served-model-name ${SERVED_NAME} --max-model-len ${VLLM_MAX_MODEL_LEN} \\"
        echo "       --dtype ${VLLM_DTYPE} --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION}"
        exit 1
    fi
    start_vllm_server
}

ensure_vllm_server

# NOTE: 脚本结束时服务保持运行（便于连续跑 run_QA_vllm.sh 复用）；如需停止：
#   kill $(cat logs/vllm_${model_name}.pid)

echo "=========================================="
echo "📊 Running TableFV Pipeline (vLLM + ATG + QG-PPR)"
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
echo "vLLM server still running (pid file: ${VLLM_PID_FILE}); stop with: kill \$(cat ${VLLM_PID_FILE})"
