#!/usr/bin/env bash
# run_thought_allscale.sh — Phase 1: 补齐 0.8B/2B/9B 的 raw+Clarifier thought (E2 C1 口径)
# 4B 已有 (results/graph_experiments/*/C1_raw/thought)。vLLM 串行切模型。
set -u
cd /root/code/GCRAFT
mkdir -p results/boost_thought logs

GT=/root/code/GCRAFT

start_vllm() {  # $1=size(0.8b|2b|9b)
    case $1 in
        0.8b) local path=/root/autodl-tmp/models/Qwen3.5-0.8B ;;
        2b)   local path=/root/autodl-tmp/models/Qwen3.5-2B ;;
        9b)   local path=/root/autodl-tmp/models/Qwen3.5-9B ;;
    esac
    pkill -f "vllm serve" 2>/dev/null || true; sleep 8
    echo "[$(date -Is)] vLLM $1 启动..."
    CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$path" \
        --host 127.0.0.1 --port 8000 --served-model-name "qwen3.5-$1" \
        --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-num-seqs 128 \
        > logs/vllm_thought_$1.log 2>&1 &
    for i in $(seq 1 150); do
        curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "qwen3.5-$1" && { echo "[$(date -Is)] ✓ vLLM $1 就绪"; return 0; }
        sleep 4
    done
    echo "❌ vLLM $1 启动失败"; exit 1
}

run_thought() {  # $1=ds $2=task_dir
    local ds=$1 tdir=$2
    local input=$([ "$ds" = wikitq ] && echo "thought/$tdir/data/wikitq/test_lower.jsonl" || echo "thought/$tdir/data/tabfact/test.jsonl")
    local extra=""
    [ "$ds" = tabfact ] && extra="--raw2clean_path thought/$tdir/data/tabfact/raw2clean.jsonl"
    echo "[$(date -Is)] ======== thought ${ds} ========"
    python3 thought/${tdir}/main.py \
        --dataset_path "$input" \
        --thought_results_dir results/boost_thought/${ds} \
        --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
        --model_name "qwen3.5-$CURRENT" \
        --first_n -1 --n_proc 16 --chunk_size 4 --use_clarifier True \
        $extra \
        > logs/bt_${ds}_${CURRENT}.log 2>&1
    echo "[$(date -Is)] thought ${ds}/${CURRENT}: $(grep -oP 'Accuracy: \K[0-9.]+' results/boost_thought/${ds}/acc.txt 2>/dev/null || echo '?')"
}

for size in 0.8b 2b 9b; do
    CURRENT=$size
    start_vllm $size
    run_thought tabfact TableFV
    run_thought wikitq TableQA
done
pkill -f "vllm serve" 2>/dev/null || true

echo "[$(date -Is)] ======== Phase 1 完成, thought acc 汇总 ========"
for ds in tabfact wikitq; do for m in 0.8b 2b 9b; do
    f=results/boost_thought/${ds}/acc.txt
    echo "  ${ds}/${m}: (acc.txt 为该目录最后一次运行; 详细见 logs/bt_${ds}_${m}.log)"
done; done
grep -h "Thought Stage Accuracy" logs/bt_*.log | tail -10
