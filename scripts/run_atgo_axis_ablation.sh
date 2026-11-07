#!/usr/bin/env bash
# run_atgo_axis_ablation.sh — ATGO 行/列/行+列 轴消融 (thought-only)
#
# 对比三种重排轴对 Stage 1 Thought 的影响, P1/P2/P3 全关 (gamma=0.0, hint=False, bias=0.0):
#   row    : 仅行重排 (ATGO)
#   col    : 仅列重排 (ATGC, 列间互换)
#   rowcol : 行+列联合重排 (ATGRC, 行序来自行 ATG、列序来自列 ATG, 独立打分)
#
# baseline: results/graph_experiments/{wikitq,tabfact}/C1_raw/thought (同模型 raw 原始表, 不重跑)
#
# 用法:
#   bash scripts/run_atgo_axis_ablation.sh                 # QA + FV 全部 6 组
#   DATASETS="qa" bash scripts/run_atgo_axis_ablation.sh   # 仅 QA
#   AXIS_GROUPS="col rowcol" bash scripts/run_atgo_axis_ablation.sh # 指定组(默认 col rowcol row)
#   AUTO_VLLM_SERVER=false bash scripts/run_atgo_axis_ablation.sh  # 服务已手动启动
#
# 可调环境变量:
#   VLLM_MODEL=/root/autodl-tmp/models/Qwen3.5-4B  SERVED_NAME=qwen3.5-4b
#   VLLM_MAX_MODEL_LEN=32768  N_PROC=24  CHUNK_SIZE=4
#   DATASETS="qa fv"  AXIS_GROUPS="row col rowcol"

set -u

# ============== vLLM Backend Configuration ==============
VLLM_MODEL="${VLLM_MODEL:-/root/autodl-tmp/models/Qwen3.5-4B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
AUTO_VLLM_SERVER="${AUTO_VLLM_SERVER:-true}"

# served-model-name 小写短名: llm.py 依赖小写 qwen 前缀注入 chat_template_kwargs 关思考
SERVED_NAME="${SERVED_NAME:-$(echo "${VLLM_MODEL##*/}" | tr 'A-Z' 'a-z')}"

base_url="http://${VLLM_HOST}:${VLLM_PORT}/v1"
openai_api_key="${VLLM_API_KEY:-EMPTY}"
model_name="${SERVED_NAME}"

first_n=-1
n_proc="${N_PROC:-24}"
chunk_size="${CHUNK_SIZE:-4}"
use_clarifier=True   # 与 C1_raw 对照同口径 (raw+Clarifier)

DATASETS="${DATASETS:-qa fv}"
# 运行顺序: 列重排 → 行列重排 → 行重排 (实验优先级由用户指定)
AXIS_GROUPS="${AXIS_GROUPS:-col rowcol row}"

# ============== Ablation Data Configuration (P1/P2/P3 全关) ==============
# 输出文件名匹配 .gitignore 的 test_reranked*; 缓存目录匹配 *_rerank_cache/
# 独立 cache 子目录按组隔离, 避免 legacy {id}.pkl 误命中
DATA_ARGS="--gamma 0.0 --inject_graph_hint False --bias_weight 0.0"

declare -A DATA_FILES QA_DIRS FV_DIRS
for g in $AXIS_GROUPS; do
    DATA_FILES[$g]="test_reranked_${g}_nograph.jsonl"
    QA_DIRS[$g]="test/results/thought/wikitq/${model_name}_axis_${g}"
    FV_DIRS[$g]="test/results/thought/tabfact/${model_name}_axis_${g}"
done

# ============== Logging Setup ==============
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${LOG_DIR}/$(basename "$0" .sh)-${model_name}-${TIMESTAMP}.log"
_masked_key="${openai_api_key:0:8}...${openai_api_key: -4}"
cat > "$LOG_FILE" <<EOF
=== ATGO Axis Ablation (thought-only) (${TIMESTAMP}) ===
backend            = vllm
vllm_model         = ${VLLM_MODEL}
served_name        = ${SERVED_NAME}
base_url           = ${base_url}
openai_api_key     = ${_masked_key}
max_model_len      = ${VLLM_MAX_MODEL_LEN}
first_n            = ${first_n}
n_proc             = ${n_proc}
chunk_size         = ${chunk_size}
use_clarifier      = ${use_clarifier}
DATASETS           = ${DATASETS}
AXIS_GROUPS             = ${AXIS_GROUPS}
rerank_args        = ${DATA_ARGS}
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
    # 已有服务但模型不符时报错退出, 防止结果串写
    if curl -sf "${base_url}/models" >/dev/null 2>&1; then
        local served
        served=$(curl -sf "${base_url}/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
        if [ "${served}" != "${SERVED_NAME}" ]; then
            echo "❌ Error: ${base_url} 正在 serve '${served}', 与本实验模型 '${SERVED_NAME}' 不符."
            echo "   先停止现有服务: kill \$(cat ${LOG_DIR}/vllm_*.pid) 或手动 kill."
            exit 1
        fi
        echo "✅ Using existing vLLM server: ${base_url} (${served})"
        return 0
    fi
    if [ "$AUTO_VLLM_SERVER" != "true" ]; then
        echo "❌ Error: no vLLM server at ${base_url} and AUTO_VLLM_SERVER=${AUTO_VLLM_SERVER}."
        exit 1
    fi
    start_vllm_server
}

ensure_vllm_server

# ============== Stage 0.7: 生成三组消融数据 (纯 CPU, P1/P2/P3 全关) ==============
declare -A DS_TASK=( [qa]="TableQA" [fv]="TableFV" )
declare -A DS_RAW=( [qa]="thought/TableQA/data/wikitq/test_lower.jsonl" [fv]="thought/TableFV/data/tabfact/test.jsonl" )

for ds in $DATASETS; do
    raw="${DS_RAW[$ds]}"
    for g in $AXIS_GROUPS; do
        out="thought/$( [ "$ds" = qa ] && echo TableQA/data/wikitq || echo TableFV/data/tabfact )/${DATA_FILES[$g]}"
        cache="thought/$( [ "$ds" = qa ] && echo TableQA/data/wikitq || echo TableFV/data/tabfact )/atgo_axis_ablation_rerank_cache/${g}"
        if [ -f "$out" ]; then
            echo "✅ [$ds/$g] Using existing ablation data: $out"
            continue
        fi
        echo ""
        echo "Stage 0.7 [$ds/$g]: ${g} rerank (gamma=0.0, hint=off, bias=0.0)"
        echo "------------------------------------------"
        python preprocess_atgo.py \
            --mode "$g" \
            --dataset_path "$raw" \
            --output_path "$out" \
            --cache_dir "$cache" \
            $DATA_ARGS
        if [ $? -ne 0 ]; then
            echo "❌ Error in [$ds/$g] rerank stage"
            exit 1
        fi
    done
done

# ============== Stage 1: Thought × {数据集} × {组} ==============
for ds in $DATASETS; do
    tdir="${DS_TASK[$ds]}"
    for g in $AXIS_GROUPS; do
        input="thought/$( [ "$ds" = qa ] && echo TableQA/data/wikitq || echo TableFV/data/tabfact )/${DATA_FILES[$g]}"
        results_dir=$( [ "$ds" = qa ] && echo "${QA_DIRS[$g]}" || echo "${FV_DIRS[$g]}" )
        echo ""
        echo "=========================================="
        echo "Stage 1 [$ds/$g]: Thought (${g} reranked input)"
        echo "  input : $input"
        echo "  output: $results_dir"
        echo "=========================================="
        extra_args=""
        [ "$ds" = fv ] && extra_args="--raw2clean_path thought/TableFV/data/tabfact/raw2clean.jsonl"
        python "thought/${tdir}/main.py" \
            --dataset_path "$input" \
            --thought_results_dir "$results_dir" \
            --base_url "$base_url" \
            --openai_api_key "$openai_api_key" \
            --model_name "$model_name" \
            --first_n "$first_n" \
            --n_proc "$n_proc" \
            --chunk_size "$chunk_size" \
            --use_clarifier "$use_clarifier" \
            $extra_args
        if [ $? -ne 0 ]; then
            echo "❌ Error in Stage 1 [$ds/$g]"
            exit 1
        fi
    done
done

# ============== Summary ==============
echo ""
echo "=========================================="
echo "📊 ATGO Axis Ablation Summary (thought acc)"
echo "=========================================="
for ds in $DATASETS; do
    if [ "$ds" = qa ]; then
        baseline="results/graph_experiments/wikitq/C1_raw/thought"
        label="wikitq"
    else
        baseline="results/graph_experiments/tabfact/C1_raw/thought"
        label="tabfact"
    fi
    b_acc=$(grep -oP 'Accuracy: \K[0-9.]+' "${baseline}/acc.txt" 2>/dev/null || echo "N/A")
    echo "[$label]"
    echo "  raw (C1_raw baseline): ${b_acc}"
    for g in $AXIS_GROUPS; do
        results_dir=$( [ "$ds" = qa ] && echo "${QA_DIRS[$g]}" || echo "${FV_DIRS[$g]}" )
        acc=$(grep -oP 'Accuracy: \K[0-9.]+' "${results_dir}/acc.txt" 2>/dev/null || echo "N/A")
        echo "  ${g} : ${acc}"
    done
done
echo "=========================================="
echo "Done. vLLM server still running (pid file: ${VLLM_PID_FILE}); stop with: kill \$(cat ${VLLM_PID_FILE})"
