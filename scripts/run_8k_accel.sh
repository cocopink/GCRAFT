#!/usr/bin/env bash
# run_8k_accel.sh — 8192 批次加速: 2B/4B 双实例共卡, QA/FV 四管线并行
#
# 前提: 原 run_batch_vllm.sh 串行批次已停止, GPU 已释放。
# 布局: 2B :8001 (util=0.45) | 4B :8002 (util=0.92, 累计)
#       {2B,4B} × {QA,FV} 四组并行, 组间无共享写路径。
# 结果: 仍写 test/results/（与原批次一致, cache 自动续跑）。

set -u
cd "$(dirname "$0")/.."

MODELS_DIR="${MODELS_DIR:-/root/autodl-tmp/models}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MAX_LEN=8192

declare -A PORT=( [2B]=8001 [4B]=8002 )
declare -A UTIL=( [2B]=0.45 [4B]=0.92 )
declare -A NPROC=( [2B]=48 [4B]=32 )
MODELS="2B 4B"

echo "=== 8k accel start: $(date '+%F %T') ==="

# ---------- Stage 1: 双实例顺序启动 ----------
declare -A STARTED
for m in $MODELS; do
    model_path="${MODELS_DIR}/Qwen3.5-${m}"
    served="qwen3.5-$(echo "$m" | tr 'A-Z' 'a-z')"
    port="${PORT[$m]}"; util="${UTIL[$m]}"
    echo "[server] starting ${served} on :${port} (util=${util}) ..."
    CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$model_path" \
        --host 127.0.0.1 --port "$port" \
        --served-model-name "$served" \
        --max-model-len "$MAX_LEN" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$util" \
        > "${LOG_DIR}/vllm_8k_${served}.log" 2>&1 &
    STARTED[$m]=$!
    ok=0
    for i in $(seq 1 120); do
        curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 "${STARTED[$m]}" 2>/dev/null || break
        sleep 5
    done
    if [ "$ok" = "1" ]; then
        echo "[server] ✅ ${served} ready on :${port}"
    else
        echo "[server] ❌ ${served} failed"; kill -9 "${STARTED[$m]}" 2>/dev/null || true; unset "STARTED[$m]"
    fi
    sleep 5
done

# ---------- Stage 2: 四组并行 (模型 × 数据集) ----------
run_pipe() {
    local m="$1" script="$2" tag="$3"
    local served="qwen3.5-$(echo "$m" | tr 'A-Z' 'a-z')"
    AUTO_VLLM_SERVER=false \
    VLLM_MODEL="${MODELS_DIR}/Qwen3.5-${m}" SERVED_NAME="$served" \
    VLLM_PORT="${PORT[$m]}" VLLM_MAX_MODEL_LEN="$MAX_LEN" \
    N_PROC="${NPROC[$m]}" CHUNK_SIZE=1 \
    bash "$script"
}

pids=()
for m in $MODELS; do
    [ -z "${STARTED[$m]:-}" ] && continue
    served="qwen3.5-$(echo "$m" | tr 'A-Z' 'a-z')"
    (
        echo "[${served}:qa] start $(date '+%T')"
        run_pipe "$m" run_QA_vllm.sh qa
        echo "[${served}:qa] done $(date '+%T')"
    ) > "${LOG_DIR}/accel_8k_${served}_qa.out" 2>&1 &
    pids+=($!)
    (
        echo "[${served}:fv] start $(date '+%T')"
        run_pipe "$m" run_FV_vllm.sh fv
        echo "[${served}:fv] done $(date '+%T')"
    ) > "${LOG_DIR}/accel_8k_${served}_fv.out" 2>&1 &
    pids+=($!)
    echo "[pipe] launched ${served} QA+FM (pids ${pids[*]: -2} ${pids[*]: -1})"
done

# ---------- Stage 3: 等待 + 停服务 + 汇总 ----------
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[summary] all pipes exited (fail=$fail) at $(date '+%F %T')"

for m in $MODELS; do
    served="qwen3.5-$(echo "$m" | tr 'A-Z' 'a-z')"
    for f in test/results/{thought,refine}/{wikitq,tabfact}/${served}_p12/acc.txt; do
        [ -f "$f" ] && echo "[acc] ${f}: $(cat "$f")"
    done
done

for m in "${!STARTED[@]}"; do kill "${STARTED[$m]}" 2>/dev/null || true; done
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "EngineCore" 2>/dev/null || true
echo "=== 8k accel done: $(date '+%F %T') ==="
