#!/usr/bin/env bash
# run_batch_vllm.sh — 本地 vLLM 批量实验编排（多模型 × QA/FV 数据集矩阵）
#
# 由于一个 vLLM server 实例只加载一个模型，本脚本按模型**串行**切换服务：
#   对每个模型: 停旧服务 → vllm serve → 跑 run_QA_vllm.sh / run_FV_vllm.sh（复用服务）→ 下一个
# 单个模型内部的样本级并发由各管线脚本的 N_PROC 控制（vLLM continuous batching 自动组批）。
#
# 用法:
#   bash run_batch_vllm.sh                                # 默认: 0.8B × qa+fv
#   MODELS="0.8B 2B 4B 9B" bash run_batch_vllm.sh         # 全模型矩阵
#   MODELS="4B" DATASETS=fv bash run_batch_vllm.sh        # 只跑 tabfact
#   N_PROC=16 CHUNK_SIZE=2 bash run_batch_vllm.sh         # 自定义管线并发
#   DRY_RUN=true bash run_batch_vllm.sh                   # 只打印执行计划，不真正运行
#
# 模型名是 ${MODELS_DIR} 下的目录后缀，如 0.8B → /root/autodl-tmp/models/Qwen3.5-0.8B

set -uo pipefail

MODELS_DIR="${MODELS_DIR:-/root/autodl-tmp/models}"
MODELS="${MODELS:-0.8B}"                 # 空格分隔: 0.8B 2B 4B 9B
DATASETS="${DATASETS:-both}"             # qa | fv | both
N_PROC="${N_PROC:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
DRY_RUN="${DRY_RUN:-false}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${LOG_DIR}/run_batch_vllm-${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "🧪 vLLM Batch Experiment Plan (${TIMESTAMP})"
echo "=========================================="
echo "MODELS     = ${MODELS}  (dir: ${MODELS_DIR})"
echo "DATASETS   = ${DATASETS}"
echo "N_PROC     = ${N_PROC}   CHUNK_SIZE = ${CHUNK_SIZE}"
echo "BASE_URL   = ${BASE_URL}"
echo "DRY_RUN    = ${DRY_RUN}"
echo "LOG_FILE   = ${LOG_FILE}"
echo "=========================================="

# ============== vLLM server lifecycle (single model at a time) ==============
VLLM_PID=""

wait_for_vllm() {
    local attempts=0
    while [ $attempts -lt 300 ]; do
        if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        attempts=$((attempts + 1))
    done
    return 1
}

stop_server() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "🛑 Stopping vLLM server (pid ${VLLM_PID})"
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
    # 等端口释放
    local attempts=0
    while curl -sf "${BASE_URL}/models" >/dev/null 2>&1; do
        sleep 1
        attempts=$((attempts + 1))
        [ $attempts -ge 30 ] && break
    done
}

start_server() {
    local model_path="$1" served_name="$2"
    local server_log="${LOG_DIR}/vllm_${served_name}_${TIMESTAMP}.log"
    echo "🚀 Starting vLLM: ${model_path} (served as ${served_name})"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    nohup vllm serve "${model_path}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --served-model-name "${served_name}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --dtype "${VLLM_DTYPE}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        > "${server_log}" 2>&1 &
    VLLM_PID=$!
    echo "   pid=${VLLM_PID}  log=${server_log}"
    if ! wait_for_vllm; then
        echo "❌ vLLM failed to start: ${model_path}"
        tail -n 80 "${server_log}" || true
        return 1
    fi
    echo "✅ vLLM ready: ${BASE_URL} (${served_name})"
}

trap stop_server EXIT

# ============== Batch loop ==============
declare -A STATUS   # "model:dataset" -> PASS/FAIL

for m in $MODELS; do
    model_path="${MODELS_DIR}/Qwen3.5-${m}"
    served_name="qwen3.5-$(echo "${m}" | tr 'A-Z' 'a-z')"

    echo ""
    echo "############################################"
    echo "# Model: ${model_path}"
    echo "############################################"

    if [ ! -d "$model_path" ]; then
        echo "❌ Model dir not found: ${model_path} — skip"
        STATUS["${m}:server"]=FAIL
        continue
    fi

    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY_RUN] would: stop_server; start_server ${model_path} ${served_name}; run pipeline(s)"
        continue
    fi

    stop_server
    if ! start_server "${model_path}" "${served_name}"; then
        STATUS["${m}:server"]=FAIL
        continue
    fi
    STATUS["${m}:server"]=PASS

    run_one() {
        local script="$1" ds_key="$2"
        echo ""
        echo "===== [${served_name}] ${script} ====="
        if AUTO_VLLM_SERVER=false \
           VLLM_MODEL="${model_path}" \
           SERVED_NAME="${served_name}" \
           N_PROC="${N_PROC}" \
           CHUNK_SIZE="${CHUNK_SIZE}" \
           bash "$script"; then
            STATUS["${m}:${ds_key}"]=PASS
            echo "✅ [${served_name}] ${script} finished"
        else
            STATUS["${m}:${ds_key}"]=FAIL
            echo "❌ [${served_name}] ${script} FAILED"
        fi
    }

    case "$DATASETS" in
        qa)   run_one "run_QA_vllm.sh" "qa" ;;
        fv)   run_one "run_FV_vllm.sh" "fv" ;;
        both) run_one "run_QA_vllm.sh" "qa"
              run_one "run_FV_vllm.sh" "fv" ;;
        *)    echo "❌ DATASETS must be qa|fv|both (got: ${DATASETS})"; exit 1 ;;
    esac
done

# ============== Summary ==============
echo ""
echo "=========================================="
echo "📋 Batch Summary"
echo "=========================================="
if [ "$DRY_RUN" = "true" ]; then
    echo "(dry run — nothing executed)"
    exit 0
fi
fail_count=0
for key in "${!STATUS[@]}"; do
    echo "  ${key} = ${STATUS[$key]}"
    [ "${STATUS[$key]}" = "FAIL" ] && fail_count=$((fail_count + 1))
done
echo "------------------------------------------"
if [ $fail_count -eq 0 ]; then
    echo "✅ All ${#STATUS[@]} entries passed. Results in test/results/"
else
    echo "⚠️  ${fail_count}/${#STATUS[@]} entries FAILED (details above)"
fi
echo "=========================================="
exit $(( fail_count > 0 ? 1 : 0 ))
