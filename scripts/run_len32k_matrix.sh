#!/usr/bin/env bash
# run_len32k_matrix.sh — 等待 8192 批次结束后，4 模型 × 32768 上下文共卡并行全量实验
#
# 编排逻辑:
#   1. 等待当前 run_batch_vllm.sh (8192, 2B/4B) 结束且 GPU 显存释放
#   2. 顺序启动 4 个 vLLM 实例（独立端口, gpu-memory-utilization 为累计上限, 依次递增）:
#        0.8B → :8001 util=0.12     2B → :8002 util=0.28
#        4B   → :8003 util=0.52     9B → :8004 util=0.90
#   3. 4 组并行: 每组串行跑 run_QA_vllm.sh → run_FV_vllm.sh（复用本组服务）
#      结果写 test/results_len32k/{thought,refine}/{wikitq,tabfact}/{model}_p12/
#   4. 全部结束后停止服务并汇总 acc
#
# 用法:
#   WAIT_PID=8905 nohup bash scripts/run_len32k_matrix.sh > logs/len32k_matrix.out 2>&1 &
#   （WAIT_PID 为 8192 批次主进程; 不设则跳过等待直接启动）

set -u
cd "$(dirname "$0")/.."

WAIT_PID="${WAIT_PID:-}"
MODELS_DIR="${MODELS_DIR:-/root/autodl-tmp/models}"
RESULTS_BASE_32K="test/results_len32k"
MAX_LEN=32768
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 模型 → 端口 / util 累计上限 / N_PROC
declare -A PORT=( [0.8B]=8001 [2B]=8002 [4B]=8003 [9B]=8004 )
declare -A UTIL=( [0.8B]=0.12 [2B]=0.28 [4B]=0.52 [9B]=0.90 )
declare -A NPROC=( [0.8B]=48 [2B]=32 [4B]=24 [9B]=16 )
MODELS="0.8B 2B 4B 9B"

echo "=== len32k matrix launcher start: $(date '+%F %T') ==="

# ---------- Stage A: 等待 8192 批次结束 ----------
if [ -n "$WAIT_PID" ]; then
    echo "[wait] waiting for pid ${WAIT_PID} (8192 batch) ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[wait] pid ${WAIT_PID} exited at $(date '+%F %T')"
fi
# 等显存释放（<5GB 视为干净）
for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 5000 ] && break
    sleep 10
done
echo "[wait] GPU free (used=${used}MiB), cleaning leftovers ..."
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 5

# ---------- Stage B: 顺序启动 4 个 vLLM 实例 ----------
declare -A STARTED
for m in $MODELS; do
    port="${PORT[$m]}"; util="${UTIL[$m]}"
    model_path="${MODELS_DIR}/Qwen3.5-${m}"
    served="qwen3.5-$(echo "$m" | tr 'A-Z' 'a-z')"
    if [ ! -d "$model_path" ]; then
        echo "[server] ❌ missing ${model_path}, skip ${m}"
        continue
    fi
    echo "[server] starting ${served} on :${port} (util=${util}) ..."
    CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$model_path" \
        --host 127.0.0.1 --port "$port" \
        --served-model-name "$served" \
        --max-model-len "$MAX_LEN" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$util" \
        > "${LOG_DIR}/vllm_len32k_${served}.log" 2>&1 &
    STARTED[$m]=$!
    ok=0
    for i in $(seq 1 120); do
        curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { ok=1; break; }
        # 进程已死则提前放弃
        kill -0 "${STARTED[$m]}" 2>/dev/null || break
        sleep 5
    done
    if [ "$ok" = "1" ]; then
        echo "[server] ✅ ${served} ready on :${port}"
    else
        echo "[server] ❌ ${served} failed to start, will skip its pipeline"
        kill -9 "${STARTED[$m]}" 2>/dev/null || true
        unset "STARTED[$m]"
    fi
    sleep 5   # 让上一个实例的显存预算稳定后再启动下一个
done

# ---------- Stage C: 4 组并行跑 QA → FV ----------
pids=()
for m in $MODELS; do
    [ -z "${STARTED[$m]:-}" ] && continue
    port="${PORT[$m]}"; nproc="${NPROC[$m]}"
    served="qwen3.5-$(echo "$m" | tr 'A-Z' 'a-z')"
    model_path="${MODELS_DIR}/Qwen3.5-${m}"
    (
        echo "[pipe:${served}] QA start $(date '+%T')"
        AUTO_VLLM_SERVER=false \
        VLLM_MODEL="$model_path" SERVED_NAME="$served" \
        VLLM_PORT="$port" VLLM_MAX_MODEL_LEN="$MAX_LEN" \
        N_PROC="$nproc" CHUNK_SIZE=1 \
        RESULTS_BASE="$RESULTS_BASE_32K" \
        bash run_QA_vllm.sh
        echo "[pipe:${served}] FV start $(date '+%T')"
        AUTO_VLLM_SERVER=false \
        VLLM_MODEL="$model_path" SERVED_NAME="$served" \
        VLLM_PORT="$port" VLLM_MAX_MODEL_LEN="$MAX_LEN" \
        N_PROC="$nproc" CHUNK_SIZE=1 \
        RESULTS_BASE="$RESULTS_BASE_32K" \
        bash run_FV_vllm.sh
        echo "[pipe:${served}] done $(date '+%T')"
    ) > "${LOG_DIR}/len32k_${served}.out" 2>&1 &
    pids+=($!)
    echo "[pipe] launched ${served} (pid $!), log: ${LOG_DIR}/len32k_${served}.out"
done

# ---------- Stage D: 等待全部结束 + 停服务 + 汇总 ----------
fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=$((fail+1))
done
echo "[summary] all pipelines exited (fail=$fail) at $(date '+%F %T')"

for m in $MODELS; do
    served="qwen3.5-$(echo "$m" | tr 'A-Z' 'a-z')"
    for f in "${RESULTS_BASE_32K}"/{thought,refine}/{wikitq,tabfact}/${served}_p12/acc.txt; do
        [ -f "$f" ] && echo "[acc] ${f#*: } $(cat "$f")"
    done
done

# 停止全部 vLLM 实例
for m in "${!STARTED[@]}"; do
    served="qwen3.5-$(echo "$m" | tr 'A-Z' 'a-z')"
    kill "${STARTED[$m]}" 2>/dev/null || true
done
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "EngineCore" 2>/dev/null || true
echo "=== len32k matrix launcher done: $(date '+%F %T') ==="
