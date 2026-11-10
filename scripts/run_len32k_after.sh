#!/usr/bin/env bash
# run_len32k_after.sh — 等待第一批 (8192×2B/4B) 结束后，用 run_batch_vllm.sh 串行跑全量 32768 批次
# 结果目录: test/results_len32k/（与第一批 test/results/ 隔离）
# 用法: nohup bash scripts/run_len32k_after.sh > logs/batch_all_len32k.out 2>&1 &

set -u
cd "$(dirname "$0")/.."

WAIT_PID="${WAIT_PID:?需设置 WAIT_PID=第一批主进程PID}"

echo "=== len32k serial batch: waiting for pid ${WAIT_PID} ($(date '+%F %T')) ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "=== first batch exited ($(date '+%F %T')), starting 32k serial batch ==="

# GPU 清场后交给 run_batch_vllm.sh（其内部自带 stop_server 逻辑）
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 5

MODELS="0.8B 2B 4B 9B" \
VLLM_MAX_MODEL_LEN=32768 \
RESULTS_BASE="test/results_len32k" \
bash run_batch_vllm.sh
