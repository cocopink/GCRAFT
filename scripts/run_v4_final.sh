#!/usr/bin/env bash
# run_v4_final.sh — 最终编排 v4 (new 优先 + 全 5 轮定稿 + 0.8B 块恢复)
# 块序: 0.8B(new→orig) → 4B(new→orig) → 9B(thought→new→orig) → 2B(new5 轮数消融)
# 教训固化: 服务 ready 确认后才起 refine; refine 用 cd 子 shell (正确仓库 import);
#           token append 式; manifest 全记录; result.txt 幂等跳过。
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
        > $G/logs/vllm_v4_$1.log 2>&1 &
    for i in $(seq 1 150); do
        curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "qwen3.5-$1" && { echo "[$(date -Is)] ✓ vLLM $1 就绪"; return 0; }
        sleep 4
    done
    echo "❌ vLLM $1 失败"; exit 1
}

run_refine() {  # $1=tag $2=ds $3=task_dir $4=thought_dir $5=out_dir $6=max_loop
    local tag=$1 ds=$2 tdir=$3 th=$4 out=$5 max_loop=${6:-5}
    local extra=""
    [ "$ds" = tabfact ] && extra="--raw2clean_path $TC/thought/$tdir/data/tabfact/raw2clean.jsonl"
    [ -f "$out/result.txt" ] && { echo "[$(date -Is)] $tag 已有结果, 跳过"; return 0; }
    echo "[$(date -Is)] ======== refine $tag (max_loop=$max_loop) ========"
    mkdir -p "$out"
    python3 - "$tag" "$ds" "$CURRENT" "$out" "$max_loop" << PYEOF
import json, hashlib, os, sys, time
tag, ds, size, out, max_loop = sys.argv[1:6]
tree = "/root/code/Table-Critic/critic/" + ("TableQA" if ds == "wikitq" else "TableFV") + "/tools/few_shot_critic.json"
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
        cd /root/code/Table-Critic
        export TC_TOKEN_LOG="$out/token_usage.jsonl"
        python3 refine/${tdir}/main_tree_based.py \
            --thought_results_dir "$th" \
            --refine_results_dir "$out" \
            --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
            --model_name "qwen3.5-$CURRENT" \
            --first_n -1 --n_proc 10 --chunk_size 4 $extra \
            --max_loop $max_loop \
            > $G/logs/v4_${tag}.log 2>&1
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

g_new_pair() {  # $1=size
    CURRENT=$1
    start_vllm $1
    local gt_qa=$G/results/graph_experiments/wikitq/C1_raw/thought
    local gt_fv=$G/results/graph_experiments/tabfact/C1_raw/thought
    [ "$1" != 4b ] && { gt_qa=$G/results/boost_thought/wikitq/$1; gt_fv=$G/results/boost_thought/tabfact/$1; }
    run_refine v4_new_qa wikitq TableQA "$gt_qa" $G/results_v2/new_qa/$1 5 & J1=$!
    run_refine v4_new_fv tabfact TableFV "$gt_fv" $G/results_v2/new_fv/$1 5 & J2=$!
    wait $J1 $J2
    echo "[$(date -Is)] ======== ${1} new 组完成 ========"
}

g_orig_pair() {  # $1=size
    CURRENT=$1
    run_refine v4_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-$1  $TC/results_v2/orig_qa/$1 5 & J1=$!
    run_refine v4_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-$1 $TC/results_v2/orig_fv/$1 5 & J2=$!
    wait $J1 $J2
    echo "[$(date -Is)] ======== ${1} orig 组完成 ========"
}

# ---- 块 1: 0.8B (new 优先, orig 随后同服务) ----
CURRENT=0.8b
start_vllm 0.8b
gt_qa=$G/results/boost_thought/wikitq/0.8b; gt_fv=$G/results/boost_thought/tabfact/0.8b
run_refine v4_new_qa wikitq TableQA "$gt_qa" $G/results_v2/new_qa/0.8b 5 & J1=$!
run_refine v4_new_fv tabfact TableFV "$gt_fv" $G/results_v2/new_fv/0.8b 5 & J2=$!
wait $J1 $J2
run_refine v4_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-0.8b  $TC/results_v2/orig_qa/0.8b 5 & J1=$!
run_refine v4_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-0.8b $TC/results_v2/orig_fv/0.8b 5 & J2=$!
wait $J1 $J2
echo "[$(date -Is)] ======== 0.8B 块完成 ========"

# ---- 块 2: 4B (new 复用交叉消融? 否——交叉组目录不同, v4 重跑保证同批口径; 幂等跳过已有) ----
CURRENT=4b
start_vllm 4b
gt_qa=$G/results/graph_experiments/wikitq/C1_raw/thought; gt_fv=$G/results/graph_experiments/tabfact/C1_raw/thought
run_refine v4_new_qa wikitq TableQA "$gt_qa" $G/results_v2/new_qa/4b 5 & J1=$!
run_refine v4_new_fv tabfact TableFV "$gt_fv" $G/results_v2/new_fv/4b 5 & J2=$!
wait $J1 $J2
run_refine v4_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-4b  $TC/results_v2/orig_qa/4b 5 & J1=$!
run_refine v4_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-4b $TC/results_v2/orig_fv/4b 5 & J2=$!
wait $J1 $J2
echo "[$(date -Is)] ======== 4B 块完成 ========"

# ---- 块 3: 9B (thought 补齐 → new → orig) ----
CURRENT=9b
start_vllm 9b
for ds in tabfact wikitq; do
    tdir=TableFV; [ "$ds" = wikitq ] && tdir=TableQA
    out=$G/results/boost_thought/${ds}/9b
    [ -f "$out/final_result.pkl" ] && continue
    input=$G/thought/${tdir}/data/${ds}/$([ "$ds" = wikitq ] && echo test_lower.jsonl || echo test.jsonl)
    extra=""; [ "$ds" = tabfact ] && extra="--raw2clean_path $G/thought/${tdir}/data/tabfact/raw2clean.jsonl"
    echo "[$(date -Is)] G thought ${ds}/9b..."
    (cd $G && python3 thought/${tdir}/main.py \
        --dataset_path "$input" \
        --thought_results_dir "$out" \
        --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
        --model_name qwen3.5-9b \
        --first_n -1 --n_proc 16 --chunk_size 4 --use_clarifier True $extra \
        > $G/logs/v4_bt_${ds}_9b.log 2>&1)
done
gt_qa=$G/results/boost_thought/wikitq/9b; gt_fv=$G/results/boost_thought/tabfact/9b
run_refine v4_new_qa wikitq TableQA "$gt_qa" $G/results_v2/new_qa/9b 5 & J1=$!
run_refine v4_new_fv tabfact TableFV "$gt_fv" $G/results_v2/new_fv/9b 5 & J2=$!
wait $J1 $J2
run_refine v4_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-9b  $TC/results_v2/orig_qa/9b 5 & J1=$!
run_refine v4_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-9b $TC/results_v2/orig_fv/9b 5 & J2=$!
wait $J1 $J2
echo "[$(date -Is)] ======== 9B 块完成 ========"

# ---- 块 4: 2B new5 (轮数消融补; 2B orig 已由 rescue 完成) ----
CURRENT=2b
start_vllm 2b
gt_qa=$G/results/boost_thought/wikitq/2b; gt_fv=$G/results/boost_thought/tabfact/2b
run_refine v4_new5_qa wikitq TableQA "$gt_qa" $G/results_v2/new5_qa/2b 5 & J1=$!
run_refine v4_new5_fv tabfact TableFV "$gt_fv" $G/results_v2/new5_fv/2b 5 & J2=$!
wait $J1 $J2
pkill -f "vllm serve" 2>/dev/null || true

echo "[$(date -Is)] ======== v4 全部完成 ========"
