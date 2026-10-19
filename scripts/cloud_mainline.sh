#!/usr/bin/env bash
# cloud_mainline.sh — 云端一键启动 16 组主实验
#
# 默认策略：
# - 只跑主实验，不跑消融
# - 每个模型只做两种方法：Table-Critic / G-CRAFT(base+P1+P2)
# - 运行顺序按模型分组，便于单机/双机部署
#
# 推荐部署：
# - node=1: Qwen3.5-0.8B + Qwen3.5-2B
# - node=2: Qwen3.5-4B + Qwen3.5-9B
# - node=all: 顺序跑完全部 16 组
#
# 运行前提：
# - 目标机器上已启动兼容 OpenAI API 的推理服务
# - vLLM 方案建议 BASE_URL=http://127.0.0.1:8000/v1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${NODE:-all}"                     # 1 | 2 | all
STAGE="${STAGE:-full}"                  # smoke | pilot | full
BACKEND="${BACKEND:-vllm}"              # vllm | ollama
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
DATASET="${DATASET:-both}"              # wikitq | tabfact | both
VARIANTS="${VARIANTS:-baseline,p1p2}"   # 只跑主实验
RESULTS_BASE="${RESULTS_BASE:-results/cloud}"
N_PROC="${N_PROC:-}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
MODELS="${MODELS:-}"                    # 例如: 0.8b 或 0.8b,2b
AUTO_VLLM_SERVER="${AUTO_VLLM_SERVER:-true}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

if [[ -f "${ROOT_DIR}/.cloud_bundle.env" ]]; then
    # shellcheck disable=SC1090
    source "${ROOT_DIR}/.cloud_bundle.env"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node) NODE="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        --results-base) RESULTS_BASE="$2"; shift 2 ;;
        --n-proc) N_PROC="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage:
  bash scripts/cloud_mainline.sh [--node 1|2|all] [--stage smoke|pilot|full]
                                 [--backend vllm|ollama] [--base-url URL]
                                 [--dataset wikitq|tabfact|both]
                                 [--variants baseline,p1p2]
                                 [--results-base PATH] [--n-proc N] [--chunk-size N]
EOF
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/cloud_mainline_${STAMP}.log"
CURRENT_MODEL="none"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Error: required command not found: $1" >&2
        exit 2
    }
}

print_plan() {
    echo "=========================================="
    echo "Cloud mainline launcher"
    echo "NODE        = ${NODE}"
    echo "STAGE       = ${STAGE}"
    echo "BACKEND     = ${BACKEND}"
    echo "BASE_URL    = ${BASE_URL}"
    echo "DATASET     = ${DATASET}"
    echo "VARIANTS    = ${VARIANTS}"
    echo "MODELS      = ${MODELS:-<node map>}"
    echo "RESULTS_BASE= ${RESULTS_BASE}"
    echo "AUTO_VLLM_SERVER = ${AUTO_VLLM_SERVER}"
    echo "VLLM_HOST   = ${VLLM_HOST}"
    echo "VLLM_PORT   = ${VLLM_PORT}"
    echo "VLLM_MAXLEN = ${VLLM_MAX_MODEL_LEN}"
    echo "VLLM_DTYPE  = ${VLLM_DTYPE}"
    echo "VLLM_GPU_MEM= ${VLLM_GPU_MEMORY_UTILIZATION}"
    echo "LOG_FILE    = ${LOG_FILE}"
    echo "=========================================="
}

models_for_node() {
    if [[ -n "$MODELS" ]]; then
        echo "$MODELS" | tr ',' ' '
        return 0
    fi
    case "$NODE" in
        1) echo "0.8b 2b" ;;
        2) echo "4b 9b" ;;
        all) echo "0.8b 2b 4b 9b" ;;
        *) echo "Error: NODE must be 1|2|all" >&2; exit 2 ;;
    esac
}

model_name_for_backend() {
    case "$1" in
        0.8b) echo "Qwen/Qwen3.5-0.8B" ;;
        2b) echo "Qwen/Qwen3.5-2B" ;;
        4b) echo "Qwen/Qwen3.5-4B" ;;
        9b) echo "Qwen/Qwen3.5-9B" ;;
        *) echo "Error: unknown model key $1" >&2; exit 2 ;;
    esac
}

wait_for_vllm() {
    local url="$1"
    local attempts=0
    while (( attempts < 180 )); do
        if curl -sf "${url}/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        attempts=$((attempts + 1))
    done
    return 1
}

start_vllm_server() {
    local model_key="$1"
    local model_name
    model_name="$(model_name_for_backend "$model_key")"
    local server_log="${LOG_DIR}/vllm_${model_key}_${STAMP}.log"
    echo "Starting vLLM server for ${model_name}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    nohup vllm serve "${model_name}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --dtype "${VLLM_DTYPE}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        > "${server_log}" 2>&1 &
    VLLM_SERVER_PID=$!
    echo "${VLLM_SERVER_PID}" > "${LOG_DIR}/vllm_${model_key}.pid"
    if ! wait_for_vllm "${BASE_URL}"; then
        echo "Error: vLLM server failed to start for ${model_name}" >&2
        tail -n 80 "${server_log}" || true
        exit 1
    fi
    echo "vLLM server ready: ${BASE_URL}"
}

stop_vllm_server() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" || true
            sleep 2
            kill -9 "$pid" >/dev/null 2>&1 || true
        fi
        rm -f "$pid_file"
    fi
}

run_model() {
    local model="$1"
    echo ""
    echo "------------------------------------------"
    echo "Running model=${model}"
    echo "------------------------------------------"

    local pid_file="${LOG_DIR}/vllm_${model}.pid"
    if [[ "$BACKEND" == "vllm" && "$AUTO_VLLM_SERVER" == "true" ]]; then
        stop_vllm_server "$pid_file"
        start_vllm_server "$model"
    fi

    RESULTS_BASE="$RESULTS_BASE" \
    OLLAMA_URL="$BASE_URL" \
    N_PROC="${N_PROC:-}" \
    CHUNK_SIZE="$CHUNK_SIZE" \
    bash "${ROOT_DIR}/run_experiments.sh" \
        --stage "$STAGE" \
        --backend "$BACKEND" \
        --model "$model" \
        --dataset "$DATASET" \
        --variant "$VARIANTS"

    if [[ "$BACKEND" == "vllm" && "$AUTO_VLLM_SERVER" == "true" ]]; then
        stop_vllm_server "$pid_file"
    fi
}

main() {
    require_cmd bash
    require_cmd tar
    require_cmd rsync
    require_cmd sed
    require_cmd git
    require_cmd curl
    require_cmd nohup

    exec > >(tee -a "$LOG_FILE") 2>&1

    print_plan

    if [[ "${GIT_SYNC:-auto}" != "off" && -f "${ROOT_DIR}/scripts/cloud_git_sync.sh" ]]; then
        echo "Git sync start..."
        GIT_REMOTE_URL="${GIT_REMOTE_URL:-}" \
        GIT_BRANCH="${GIT_BRANCH:-main}" \
        bash "${ROOT_DIR}/scripts/cloud_git_sync.sh" --mode "${GIT_SYNC:-auto}"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU status:"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
    fi

    echo "Launcher start: $(date -Is)"

    trap 'stop_vllm_server "${LOG_DIR}/vllm_${CURRENT_MODEL}.pid" || true' EXIT

    for model in $(models_for_node); do
        CURRENT_MODEL="$model"
        run_model "$model"
    done

    echo ""
    echo "=========================================="
    echo "All requested runs complete."
    echo "=========================================="
}

main "$@"
