#!/usr/bin/env bash
# run_remaining_8.sh — 剩余 8 组编排 (9B×4 → 4B orig×2 → 2B new5×2)
# 9B 服务已在跑 (qwen3.5-9b@8000), 直接挂 refine; 完成后切 4B/2B
set -u
TC=/root/code/Table-Critic
G=/root/code/GCRAFT
BASE=http://127.0.0.1:8000/v1

run_refine() {  # $1=tag $2=ds $3=task_dir $4=thought_dir $5=out_dir $6=size $7=max_loop
    local tag=$1 ds=$2 tdir=$3 th=$4 out=$5 size=$6 ml=$7
    local extra=""
    [ "$ds" = tabfact ] && extra="--raw2clean_path $TC/thought/$tdir/data/tabfact/raw2clean.jsonl"
    [ -f "$out/result.txt" ] && { echo "[$(date -Is)] $tag 已有结果, 跳过"; return 0; }
    echo "[$(date -Is)] ======== refine $tag ========"
    mkdir -p "$out"
    python3 - "$tag" "$ds" "$size" "$out" "$ml" << PYEOF
import json, hashlib, os, sys, time
tag, ds, size, out, max_loop = sys.argv[1:6]
tree = "$TC/critic/" + ("TableQA" if ds == "wikitq" else "TableFV") + "/tools/few_shot_critic.json"
m = {"tag": tag, "dataset": ds, "model": "qwen3.5-" + size,
     "refine_repo": "Table-Critic 原版", "max_loop": int(max_loop),
     "thought_input": "G raw+Clarifier" if tag.startswith("new") else "原版第一次基线产物",
     "n_proc": 10, "chunk_size": 4, "backend": "vllm 32k bf16 greedy",
     "tree": {"path": tree, "size_bytes": os.path.getsize(tree),
              "sha256": hashlib.sha256(open(tree, "rb").read()).hexdigest(),
              "origin": "canonical 原点 (每模型块恢复)"},
     "token_log": out + "/token_usage.jsonl",
     "started": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(m, open(out + "/manifest.json", "w"), ensure_ascii=False, indent=2)
PYEOF
    (
        cd $TC
        export TC_TOKEN_LOG="$out/token_usage.jsonl"
        python3 refine/${tdir}/main_tree_based.py \
            --thought_results_dir "$th" \
            --refine_results_dir "$out" \
            --base_url $BASE --openai_api_key EMPTY \
            --model_name "qwen3.5-$size" \
            --first_n -1 --n_proc 10 --chunk_size 4 $extra \
            --max_loop $max_loop \
            > $G/logs/remain_${tag}.log 2>&1
    )
    local acc
    acc=$(grep -oP 'Accuracy: \K[0-9.]+' $out/result.txt 2>/dev/null || echo FAILED)
    python3 - "$out" "$acc" << PYEOF2
import json, sys, time
p, acc = sys.argv[1], sys.argv[2]
m = json.load(open(p + "/manifest.json"))
m["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
m["refine_acc"] = acc
json.dump(m, open(p + "/manifest.json", "w"), ensure_ascii=False, indent=2)
PYEOF2
    echo "[$(date -Is)] $tag: ${acc}"
}

switch_vllm() {  # $1=size
    case $1 in
        4b) local path=/root/autodl-tmp/models/Qwen3.5-4B ;;
        2b) local path=/root/autodl-tmp/models/Qwen3.5-2B ;;
    esac
    pkill -9 -f "vllm [s]erve" 2>/dev/null || true; sleep 8
    echo "[$(date -Is)] vLLM $1 启动..."
    CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$path" \
        --host 127.0.0.1 --port 8000 --served-model-name "qwen3.5-$1" \
        --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-num-seqs 128 \
        > $G/logs/vllm_rem_$1.log 2>&1 &
    for i in $(seq 1 150); do
        curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "qwen3.5-$1" && { echo "[$(date -Is)] ✓ vLLM $1 就绪"; return 0; }
        sleep 4
    done
    echo "❌ vLLM $1 失败"; exit 1
}

echo "[$(date -Is)] ======== 9B 块 4 组 (服务已在跑) ========"
run_refine new_qa_9b  wikitq  TableQA $G/results/boost_thought/wikitq/9b  $G/results_v2/new_qa/9b  9b 5 & J1=$!
run_refine new_fv_9b  tabfact TableFV $G/results/boost_thought/tabfact/9b $G/results_v2/new_fv/9b  9b 5 & J2=$!
run_refine orig_qa_9b wikitq  TableQA $TC/results/thought/wikitq/qwen3.5-9b  $TC/results_v2/orig_qa/9b  9b 5 & J3=$!
run_refine orig_fv_9b tabfact TableFV $TC/results/thought/tabfact/qwen3.5-9b $TC/results_v2/orig_fv/9b  9b 5 & J4=$!
wait $J1 $J2 $J3 $J4
echo "[$(date -Is)] ======== 9B 块完成 ========"

switch_vllm 4b
run_refine orig_qa_4b wikitq  TableQA $TC/results/thought/wikitq/qwen3.5-4b  $TC/results_v2/orig_qa/4b  4b 5 & J1=$!
run_refine orig_fv_4b tabfact TableFV $TC/results/thought/tabfact/qwen3.5-4b $TC/results_v2/orig_fv/4b  4b 5 & J2=$!
wait $J1 $J2
echo "[$(date -Is)] ======== 4B orig 块完成 (new×2 缓存续跑随后) ========"

# 4B new×2 缓存续跑 (693/1025 进度保留)
run_refine new_qa_4b wikitq  TableQA $G/results/graph_experiments/wikitq/C1_raw/thought $G/results_v2/new_qa/4b  4b 5 & J1=$!
run_refine new_fv_4b tabfact TableFV $G/results/graph_experiments/tabfact/C1_raw/thought $G/results_v2/new_fv/4b  4b 5 & J2=$!
wait $J1 $J2
echo "[$(date -Is)] ======== 4B 块完成 ========"

switch_vllm 2b
run_refine new5_qa_2b wikitq  TableQA $G/results/boost_thought/wikitq/2b  $G/results_v2/new5_qa/2b  2b 5 & J1=$!
run_refine new5_fv_2b tabfact TableFV $G/results/boost_thought/tabfact/2b $G/results_v2/new5_fv/2b  2b 5 & J2=$!
wait $J1 $J2
pkill -9 -f "vllm [s]erve" 2>/dev/null || true

echo "[$(date -Is)] ======== 全部 8 组完成 ========"
for d in $G/results_v2/new_qa/9b $G/results_v2/new_fv/9b $TC/results_v2/orig_qa/9b $TC/results_v2/orig_fv/9b \
         $TC/results_v2/orig_qa/4b $TC/results_v2/orig_fv/4b $G/results_v2/new_qa/4b $G/results_v2/new_fv/4b \
         $G/results_v2/new5_qa/2b $G/results_v2/new5_fv/2b; do
    f=$d/result.txt; [ -f "$f" ] && echo "  ${d#$G/}: $(grep -oP 'Accuracy: \K[0-9.]+' $f)"
done
