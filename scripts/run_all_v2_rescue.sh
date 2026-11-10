#!/usr/bin/env bash
# run_all_v2_rescue.sh — v2 恢复编排 (heredoc bug 修复版)
# 顺序: 等 2B wikitq thought 孤儿完成 → 2B 4 组 refine → 0.8B 4 组 refine → 4B 块 → 9B 块
set -u
TC=/root/code/Table-Critic
G=/root/code/GCRAFT

start_vllm() {
    case $1 in
        0.8b) local path=/root/autodl-tmp/models/Qwen3.5-0.8B ;;
        2b)   local path=/root/autodl-tmp/models/Qwen3.5-2B ;;
        4b)   local path=/root/autodl-tmp/models/Qwen3.5-4B ;;
        9b)   local path=/root/autodl-tmp/models/Qwen3.5-9B ;;
    esac
    pkill -f "vllm serve" 2>/dev/null || true; sleep 8
    echo "[$(date -Is)] vLLM $1 启动..."
    CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$path" \
        --host 127.0.0.1 --port 8000 --served-model-name "qwen3.5-$1" \
        --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-num-seqs 128 \
        > $G/logs/vllm_v2_$1.log 2>&1 &
    for i in $(seq 1 150); do
        curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "qwen3.5-$1" && { echo "[$(date -Is)] ✓ vLLM $1 就绪"; return 0; }
        sleep 4
    done
    echo "❌ vLLM $1 失败"; exit 1
}

run_refine() {  # $1=tag $2=ds $3=task_dir $4=thought_dir(绝对) $5=out_dir $6=repo_root $7=nproc $8=max_loop
    local tag=$1 ds=$2 tdir=$3 th=$4 out=$5 repo=$6 nproc=$7 max_loop=${8:-5}
    local extra=""
    [ "$ds" = tabfact ] && extra="--raw2clean_path $repo/thought/$tdir/data/tabfact/raw2clean.jsonl"
    echo "[$(date -Is)] ======== refine $tag ========"
    mkdir -p "$out"
    python3 - "$tag" "$ds" "$CURRENT" "$repo" "$out" "$max_loop" << PYEOF
import json, hashlib, os, sys, time
tag, ds, size, repo, out, max_loop = sys.argv[1:7]
tree = repo + "/critic/" + ("TableQA" if ds == "wikitq" else "TableFV") + "/tools/few_shot_critic.json"
m = {"tag": tag, "dataset": ds, "model": "qwen3.5-" + size,
     "refine_repo": "Table-Critic 原版",
     "max_loop": int(max_loop),
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
        cd $repo
        export TC_TOKEN_LOG="$out/token_usage.jsonl"
        python3 refine/${tdir}/main_tree_based.py \
            --thought_results_dir "$th" \
            --refine_results_dir "$out" \
            --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
            --model_name "qwen3.5-$CURRENT" \
            --first_n -1 --n_proc $nproc --chunk_size 4 $extra \
            --max_loop $max_loop \
            > $G/logs/v2_${tag}.log 2>&1
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

# ---- 1. 等 2B wikitq thought 孤儿完成 ----
echo "[$(date -Is)] 等待 2B wikitq thought 孤儿完成..."
while [ ! -f "$G/results/boost_thought/wikitq/2b/final_result.pkl" ]; do sleep 60; done
echo "[$(date -Is)] 2B thought 孤儿完成"

CURRENT=2b
start_vllm 2b
run_refine v2_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-2b  $TC/results_v2/orig_qa/2b  $TC 10 5 & J1=$!
run_refine v2_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-2b $TC/results_v2/orig_fv/2b  $TC 10 5 & J2=$!
run_refine v2_new_qa  wikitq TableQA $G/results/boost_thought/wikitq/2b  $G/results_v2/new_qa/2b  $TC 10 2 & J3=$!
run_refine v2_new_fv  tabfact TableFV $G/results/boost_thought/tabfact/2b $G/results_v2/new_fv/2b  $TC 10 2 & J4=$!
wait $J1 $J2 $J3 $J4
echo "[$(date -Is)] ======== 2B 块完成 ========"

# ---- 2. 0.8B 4 组 refine ----
CURRENT=0.8b
start_vllm 0.8b
run_refine v2_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-0.8b  $TC/results_v2/orig_qa/0.8b  $TC 10 5 & J1=$!
run_refine v2_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-0.8b $TC/results_v2/orig_fv/0.8b  $TC 10 5 & J2=$!
run_refine v2_new_qa  wikitq TableQA $G/results/boost_thought/wikitq/0.8b  $G/results_v2/new_qa/0.8b  $TC 10 2 & J3=$!
run_refine v2_new_fv  tabfact TableFV $G/results/boost_thought/tabfact/0.8b $G/results_v2/new_fv/0.8b  $TC 10 2 & J4=$!
wait $J1 $J2 $J3 $J4
echo "[$(date -Is)] ======== 0.8B 块完成 ========"

# ---- 3. 4B 块 (G thought 复用 C1_raw) ----
CURRENT=4b
start_vllm 4b
run_refine v2_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-4b  $TC/results_v2/orig_qa/4b  $TC 10 5 & J1=$!
run_refine v2_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-4b $TC/results_v2/orig_fv/4b  $TC 10 5 & J2=$!
run_refine v2_new_qa  wikitq TableQA $G/results/graph_experiments/wikitq/C1_raw/thought  $G/results_v2/new_qa/4b  $TC 10 5 & J3=$!
run_refine v2_new_fv  tabfact TableFV $G/results/graph_experiments/tabfact/C1_raw/thought $G/results_v2/new_fv/4b  $TC 10 5 & J4=$!
wait $J1 $J2 $J3 $J4
echo "[$(date -Is)] ======== 4B 块完成 ========"

# ---- 4. 9B 块 (G thought 现场补) ----
CURRENT=9b
start_vllm 9b
for ds in tabfact wikitq; do
    local_tdir=TableFV; [ "$ds" = wikitq ] && local_tdir=TableQA
    out=$G/results/boost_thought/${ds}/9b
    [ -f "$out/final_result.pkl" ] && continue
    input=$G/thought/${local_tdir}/data/${ds}/$([ "$ds" = wikitq ] && echo test_lower.jsonl || echo test.jsonl)
    extra=""; [ "$ds" = tabfact ] && extra="--raw2clean_path $G/thought/${local_tdir}/data/tabfact/raw2clean.jsonl"
    echo "[$(date -Is)] G thought ${ds}/9b..."
    (cd $G && python3 thought/${local_tdir}/main.py \
        --dataset_path "$input" \
        --thought_results_dir "$out" \
        --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
        --model_name qwen3.5-9b \
        --first_n -1 --n_proc 16 --chunk_size 4 --use_clarifier True $extra \
        > $G/logs/v2_bt_${ds}_9b.log 2>&1)
done
run_refine v2_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-9b  $TC/results_v2/orig_qa/9b  $TC 10 5 & J1=$!
run_refine v2_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-9b $TC/results_v2/orig_fv/9b  $TC 10 5 & J2=$!
run_refine v2_new_qa  wikitq TableQA $G/results/boost_thought/wikitq/9b  $G/results_v2/new_qa/9b  $TC 10 5 & J3=$!
run_refine v2_new_fv  tabfact TableFV $G/results/boost_thought/tabfact/9b $G/results_v2/new_fv/9b  $TC 10 5 & J4=$!
wait $J1 $J2 $J3 $J4
pkill -f "vllm serve" 2>/dev/null || true
echo "[$(date -Is)] ======== 16 组全部完成 ========"
