#!/usr/bin/env bash
# run_rethink_ablation.sh — RETHINK_RERANK 全量 QA 验证 (judge-incorrect 触发重排重推)
#
# thought 复用 C1_raw (raw 基线, 不重跑); refine 在其上开启重推。
# 对比基准: results_v2/new_qa/4b (现有修正循环, acc 0.7370)。
#
# 用法:
#   bash scripts/run_rethink_ablation.sh                        # 默认 rowcol_g01 (赢家配置)
#   CONFIGS="rowcol_g01 row_g01" bash scripts/run_rethink_ablation.sh
#   AUTO_VLLM_SERVER=false bash scripts/run_rethink_ablation.sh  # 服务已手动启动
#
# 可调环境变量:
#   VLLM_MODEL=/root/autodl-tmp/models/Qwen3.5-4B  SERVED_NAME=qwen3.5-4b
#   VLLM_MAX_MODEL_LEN=32768  N_PROC=24  CHUNK_SIZE=4
#   CONFIGS="rowcol_g01"  # 每项对应 thought/TableQA/data/wikitq/test_reranked_<cfg>.jsonl

set -u

VLLM_MODEL="${VLLM_MODEL:-/root/autodl-tmp/models/Qwen3.5-4B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
AUTO_VLLM_SERVER="${AUTO_VLLM_SERVER:-true}"
SERVED_NAME="${SERVED_NAME:-$(echo "${VLLM_MODEL##*/}" | tr 'A-Z' 'a-z')}"

base_url="http://${VLLM_HOST}:${VLLM_PORT}/v1"
openai_api_key="${VLLM_API_KEY:-EMPTY}"
model_name="${SERVED_NAME}"

first_n=-1
n_proc="${N_PROC:-24}"
chunk_size="${CHUNK_SIZE:-4}"

CONFIGS="${CONFIGS:-rowcol_g01}"
THOUGHT_DIR="results/graph_experiments/wikitq/C1_raw/thought"
DATA_DIR="thought/TableQA/data/wikitq"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${LOG_DIR}/$(basename "$0" .sh)-${model_name}-${TIMESTAMP}.log"
cat > "$LOG_FILE" <<EOF
=== RETHINK_RERANK Ablation (${TIMESTAMP}) ===
served_name        = ${SERVED_NAME}
base_url           = ${base_url}
thought_dir        = ${THOUGHT_DIR} (C1_raw 复用, 不重跑)
CONFIGS            = ${CONFIGS}
n_proc             = ${n_proc}  chunk_size = ${chunk_size}
========================================
EOF
echo "📝 Log: ${LOG_FILE}"
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
        echo "❌ Error: vllm command not found."
        exit 1
    fi
    local server_log="${LOG_DIR}/vllm_${model_name}_${TIMESTAMP}.log"
    echo "🚀 Starting vLLM server: ${VLLM_MODEL} (served as ${SERVED_NAME})"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    nohup vllm serve "${VLLM_MODEL}" \
        --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
        --served-model-name "${SERVED_NAME}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --dtype "${VLLM_DTYPE}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        > "${server_log}" 2>&1 &
    echo $! > "${VLLM_PID_FILE}"
    if ! wait_for_vllm "${base_url}"; then
        echo "❌ Error: vLLM server failed to start"
        tail -n 80 "${server_log}" || true
        exit 1
    fi
    echo "✅ vLLM server ready: ${base_url}"
}

if curl -sf "${base_url}/models" >/dev/null 2>&1; then
    served=$(curl -sf "${base_url}/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
    if [ "${served}" != "${SERVED_NAME}" ]; then
        echo "❌ Error: ${base_url} 正在 serve '${served}', 与 '${SERVED_NAME}' 不符."
        exit 1
    fi
    echo "✅ Using existing vLLM server: ${base_url} (${served})"
else
    if [ "$AUTO_VLLM_SERVER" != "true" ]; then
        echo "❌ Error: no vLLM server at ${base_url}."
        exit 1
    fi
    start_vllm_server
fi

# ============== Refine × RETHINK_RERANK ==============
for cfg in $CONFIGS; do
    jsonl="${DATA_DIR}/test_reranked_${cfg}.jsonl"
    out="test/results/refine/wikitq/${model_name}_rethink_${cfg}"
    if [ ! -f "$jsonl" ]; then
        echo "❌ Error: 重排数据不存在: $jsonl (先跑 preprocess_atgo.py --gamma 0.1)"
        exit 1
    fi
    echo ""
    echo "=========================================="
    echo "RETHINK_RERANK [$cfg]"
    echo "  thought (复用): $THOUGHT_DIR"
    echo "  rerank jsonl  : $jsonl"
    echo "  output        : $out"
    echo "=========================================="
    python refine/TableQA/main_tree_based.py \
        --thought_results_dir "$THOUGHT_DIR" \
        --refine_results_dir "$out" \
        --base_url "$base_url" \
        --openai_api_key "$openai_api_key" \
        --model_name "$model_name" \
        --first_n "$first_n" \
        --n_proc "$n_proc" \
        --chunk_size "$chunk_size" \
        --use_clarifier True \
        --rethink_rerank True \
        --rethink_rerank_jsonl "$jsonl"
    if [ $? -ne 0 ]; then
        echo "❌ Error in rethink [$cfg]"
        exit 1
    fi
done

# ============== Summary ==============
echo ""
echo "=========================================="
echo "📊 RETHINK_RERANK Summary"
echo "  baseline (v2 现有修正循环): 0.7370"
echo "=========================================="
for cfg in $CONFIGS; do
    out="test/results/refine/wikitq/${model_name}_rethink_${cfg}"
    acc=$(grep -oP 'Accuracy: \K[0-9.]+' "${out}/result.txt" 2>/dev/null || echo "N/A")
    route_stats=$(python3 - "$out/final_result.pkl" <<'PYEOF'
import pickle, sys, collections
data = pickle.load(open(sys.argv[1], 'rb'))
c = collections.Counter(s.get('_route', 'NONE') for s in data if isinstance(s, dict))
print(dict(c))
PYEOF
)
    echo "rethink-${cfg}: acc=${acc}  routes=${route_stats}"
done
echo "=========================================="
echo "Done. vLLM server still running (pid file: ${VLLM_PID_FILE}); stop with: kill \$(cat ${VLLM_PID_FILE})"
