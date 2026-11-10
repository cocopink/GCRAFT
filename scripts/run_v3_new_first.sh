#!/usr/bin/env bash
# run_v3_new_first.sh — v3: new 组优先编排 (用户指令: 先跑全部 size 的新框架数据)
# 依赖: 2B 的 orig 两组由 rescue 继续跑 (脚本不碰); 剩余组按序:
#   1) 0.8B new×2  2) 4B new×2  3) 9B thought补齐 + 9B new×2
#   4) 0.8B orig×2  5) 4B orig×2  6) 9B orig×2
# 每组 manifest + token (cd 子 shell, 正确仓库加载)
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
        > $G/logs/vllm_v3_$1.log 2>&1 &
    for i in $(seq 1 150); do
        curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "qwen3.5-$1" && { echo "[$(date -Is)] ✓ vLLM $1 就绪"; return 0; }
        sleep 4
    done
    echo "❌ vLLM $1 失败"; exit 1
}

run_refine() {  # $1=tag $2=ds $3=task_dir $4=thought_dir $5=out_dir $6=repo_root $7=nproc $8=max_loop
    local tag=$1 ds=$2 tdir=$3 th=$4 out=$5 repo=$6 nproc=$7 max_loop=${8:-5}
    local extra=""
    [ "$ds" = tabfact ] && extra="--raw2clean_path $repo/thought/$tdir/data/tabfact/raw2clean.jsonl"
    [ -f "$out/result.txt" ] && { echo "[$(date -Is)] $tag 已有结果, 跳过"; return 0; }
    echo "[$(date -Is)] ======== refine $tag ========"
    mkdir -p "$out"
    python3 - "$tag" "$ds" "$CURRENT" "$repo" "$out" "$max_loop" << PYEOF
import json, hashlib, os, sys, time
tag, ds, size, repo, out, max_loop = sys.argv[1:7]
tree = repo + "/critic/" + ("TableQA" if ds == "wikitq" else "TableFV") + "/tools/few_shot_critic.json"
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
        cd $repo
        export TC_TOKEN_LOG="$out/token_usage.jsonl"
        python3 refine/${tdir}/main_tree_based.py \
            --thought_results_dir "$th" \
            --refine_results_dir "$out" \
            --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
            --model_name "qwen3.5-$CURRENT" \
            --first_n -1 --n_proc $nproc --chunk_size 4 $extra \
            --max_loop $max_loop \
            > $G/logs/v3_${tag}.log 2>&1
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

g_new_pair() {  # $1=size — 该模型的 new QA/FV 并发
    CURRENT=$1
    start_vllm $1
    local gt_qa=$G/results/graph_experiments/wikitq/C1_raw/thought
    local gt_fv=$G/results/graph_experiments/tabfact/C1_raw/thought
    [ "$1" != 4b ] && { gt_qa=$G/results/boost_thought/wikitq/$1; gt_fv=$G/results/boost_thought/tabfact/$1; }
    run_refine v3_new_qa wikitq TableQA "$gt_qa" $G/results_v2/new_qa/$1  $TC 10 5 & J1=$!
    run_refine v3_new_fv tabfact TableFV "$gt_fv" $G/results_v2/new_fv/$1  $TC 10 5 & J2=$!
    wait $J1 $J2
    echo "[$(date -Is)] ======== ${1} new 组完成 ========"
}

g_orig_pair() {  # $1=size — 该模型的 orig QA/FV 并发
    CURRENT=$1
    start_vllm $1
    run_refine v3_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-$1  $TC/results_v2/orig_qa/$1  $TC 10 5 & J1=$!
    run_refine v3_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-$1 $TC/results_v2/orig_fv/$1  $TC 10 5 & J2=$!
    wait $J1 $J2
    echo "[$(date -Is)] ======== ${1} orig 组完成 ========"
}

g_thought_pair() {  # $1=size — G thought 补齐 (9B 用)
    CURRENT=$1
    start_vllm $1
    for ds in tabfact wikitq; do
        local tdir=TableFV; [ "$ds" = wikitq ] && tdir=TableQA
        local out=$G/results/boost_thought/${ds}/$1
        [ -f "$out/final_result.pkl" ] && continue
        local input=$G/thought/${tdir}/data/${ds}/$([ "$ds" = wikitq ] && echo test_lower.jsonl || echo test.jsonl)
        local extra=""; [ "$ds" = tabfact ] && extra="--raw2clean_path $G/thought/${tdir}/data/tabfact/raw2clean.jsonl"
        echo "[$(date -Is)] G thought ${ds}/$1..."
        (cd $G && python3 thought/${tdir}/main.py \
            --dataset_path "$input" \
            --thought_results_dir "$out" \
            --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
            --model_name "qwen3.5-$1" \
            --first_n -1 --n_proc 16 --chunk_size 4 --use_clarifier True $extra \
            > $G/logs/v3_bt_${ds}_$1.log 2>&1)
    done
}

# ---- Phase A0: 2B 5 轮组 (复用 wrapper 保留的 2b 服务; 与 2 轮版构成轮数消融对) ----
CURRENT=2b
run_refine v3_new5_qa wikitq TableQA $G/results/boost_thought/wikitq/2b  $G/results_v2/new5_qa/2b  $TC 10 5 & J1=$!
run_refine v3_new5_fv tabfact TableFV $G/results/boost_thought/tabfact/2b $G/results_v2/new5_fv/2b $TC 10 5 & J2=$!
wait $J1 $J2
echo "[$(date -Is)] ======== 2B 5 轮组完成 ========"
g_new_pair 0.8b
g_new_pair 4b
g_thought_pair 9b
g_new_pair 9b

# ---- Phase B: 剩余 orig 组 ----
g_orig_pair 0.8b
g_orig_pair 4b
g_orig_pair 9b
pkill -f "vllm serve" 2>/dev/null || true

echo "[$(date -Is)] ======== v3 全部完成 ========"
echo "参照第一次基线: FV 0.8b=0.2762 2b=0.6359 4b=0.9313 9b=0.9377 | QA 0.8b=0.1573 2b=0.0879 4b=0.7884 9b=0.8072"
echo "交叉消融 4B (G thought × 原 refine): FV=0.9277 QA=0.7895"
