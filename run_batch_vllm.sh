#!/usr/bin/env bash
# run_batch_vllm.sh — 本地 vLLM 批量实验编排（多模型 × QA/FV 数据集矩阵）
#
# 由于一个 vLLM server 实例只加载一个模型，本脚本按模型**串行**切换服务：
#   对每个模型: 停旧服务 → vllm serve → 跑 run_QA_vllm.sh / run_FV_vllm.sh（复用服务）→ 下一个
#
# 并发说明（32GB 单卡，bf16）:
#   - 显存与 N_PROC 无关：vllm serve 启动时按 --gpu-memory-utilization 0.85 预分配
#     （权重 + KV cache 池 ≈ 27GB），始终吃满。
#   - N_PROC 决定客户端并发请求数 → vLLM continuous batching 组批大小 → GPU 算力利用率。
#     模型越小推理越快、客户端越容易供不上，因此按模型规模给默认并发：
#       0.8B→48   2B→32   4B→24   9B→16（9B 的 KV 池仅剩 ~7GB，服务端并发受限，再多即排队）
#   - 显式 export N_PROC 时全局覆盖上述映射。
#   - CHUNK_SIZE=1：mp.Pool.imap_unordered 逐任务动态分配，避免大 chunk 造成进程闲置。
#
# 用法:
#   bash run_batch_vllm.sh                                # 默认: 0.8B × qa+fv（N_PROC 按模型自动）
#   MODELS="0.8B 2B 4B 9B" bash run_batch_vllm.sh         # 全模型矩阵
#   MODELS="4B" DATASETS=fv bash run_batch_vllm.sh        # 只跑 tabfact
#   N_PROC=8 bash run_batch_vllm.sh                       # 显式并发，覆盖按模型映射
#   DRY_RUN=true bash run_batch_vllm.sh                   # 只打印执行计划，不真正运行
#
# 模型名是 ${MODELS_DIR} 下的目录后缀，如 0.8B → /root/autodl-tmp/models/Qwen3.5-0.8B

set -uo pipefail

MODELS_DIR="${MODELS_DIR:-/root/autodl-tmp/models}"
MODELS="${MODELS:-0.8B}"                 # 空格分隔: 0.8B 2B 4B 9B
DATASETS="${DATASETS:-both}"             # qa | fv | both
CHUNK_SIZE="${CHUNK_SIZE:-1}"
DRY_RUN="${DRY_RUN:-false}"

# 并发映射:未显式 export N_PROC 时按模型规模取默认（32GB 卡 bf16 经验值）
nproc_for_model() {
    if [ -n "${N_PROC+x}" ]; then
        echo "$N_PROC"
        return
    fi
    case "$1" in
        0.8B) echo 48 ;;
        2B)   echo 32 ;;
        4B)   echo 24 ;;
        9B)   echo 16 ;;
        *)    echo 24 ;;
    esac
}

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
if [ -n "${N_PROC+x}" ]; then
    echo "N_PROC     = ${N_PROC} (explicit, applies to all models)"
else
    echo "N_PROC     = per-model: 0.8B→48 2B→32 4B→24 9B→16"
fi
echo "CHUNK_SIZE = ${CHUNK_SIZE}"
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
    # 1) 停本脚本启动的实例
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "🛑 Stopping vLLM server (pid ${VLLM_PID})"
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
    # 2) 清理任何残留 vLLM 进程（含手动测试遗留、非本脚本启动的外部实例，
    #    以及 APIServer 已死但仍占显存的 EngineCore 子进程）
    if pgrep -f "vllm serve" >/dev/null 2>&1 || pgrep -f "EngineCore" >/dev/null 2>&1; then
        echo "🛑 Killing leftover vLLM process(es): $(pgrep -f 'vllm serve|EngineCore' | tr '\n' ' ')"
        pkill -f "vllm serve" 2>/dev/null || true
        pkill -f "EngineCore" 2>/dev/null || true
        sleep 3
        pkill -9 -f "vllm serve" 2>/dev/null || true
        pkill -9 -f "EngineCore" 2>/dev/null || true
    fi
    # 3) 等端口真正释放（curl 探测，不依赖 ss/lsof）
    local attempts=0
    while curl -sf --max-time 2 "${BASE_URL}/models" >/dev/null 2>&1; do
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
        local model_nproc
        model_nproc="$(nproc_for_model "${m}")"
        echo ""
        echo "===== [${served_name}] ${script} (N_PROC=${model_nproc}) ====="
        if AUTO_VLLM_SERVER=false \
           VLLM_MODEL="${model_path}" \
           SERVED_NAME="${served_name}" \
           N_PROC="${model_nproc}" \
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
