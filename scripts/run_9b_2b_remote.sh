#!/usr/bin/env bash
# run_9b_2b_remote.sh — 9B/2B 远程执行编排（新框架 + 原版重跑，全 5 轮）
# 前置：本脚本位于打包根目录（含 Table-Critic/ 与 GCRAFT/ 两个仓库副本）
# 权重：WEIGHTS_9B / WEIGHTS_2B 环境变量或下方默认（按新服务器路径修改）
set -u
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TC="$PKG/Table-Critic"
G="$PKG/GCRAFT"
WEIGHTS_9B="${WEIGHTS_9B:-/root/autodl-tmp/models/Qwen3.5-9B}"
WEIGHTS_2B="${WEIGHTS_2B:-/root/autodl-tmp/models/Qwen3.5-2B}"
mkdir -p "$TC/results_v2" "$G/results_v2" logs

start_vllm() {  # $1=size(9b|2b)
    local path; [ "$1" = 9b ] && path=$WEIGHTS_9B || path=$WEIGHTS_2B
    pkill -f "vllm serve" 2>/dev/null || true; sleep 8
    echo "[$(date -Is)] vLLM $1 启动 ($path)..."
    CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$path" \
        --host 127.0.0.1 --port 8000 --served-model-name "qwen3.5-$1" \
        --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-num-seqs 128 \
        > logs/vllm_$1.log 2>&1 &
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
tree = "$TC/critic/" + ("TableQA" if ds == "wikitq" else "TableFV") + "/tools/few_shot_critic.json"
m = {"tag": tag, "dataset": ds, "model": "qwen3.5-" + size,
     "refine_repo": "Table-Critic 原版", "max_loop": int(max_loop),
     "thought_input": "G raw+Clarifier" if tag.startswith("new") else "原版第一次基线产物",
     "n_proc": 10, "chunk_size": 4, "backend": "vllm 32k bf16 greedy",
     "tree": {"path": tree, "size_bytes": os.path.getsize(tree),
              "sha256": hashlib.sha256(open(tree, "rb").read()).hexdigest(),
              "origin": "canonical (包内预置干净版)"},
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
            --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
            --model_name "qwen3.5-$CURRENT" \
            --first_n -1 --n_proc 10 --chunk_size 4 $extra \
            --max_loop $max_loop \
            > $PKG/logs/remote_${tag}.log 2>&1
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

# ---- Phase 9B ----
CURRENT=9b
start_vllm 9b
# G thought wikitq/9b 补齐 (tabfact/9b 已在包内)
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
        > logs/bt_${ds}_9b.log 2>&1)
done
gt_qa=$G/results/boost_thought/wikitq/9b; gt_fv=$G/results/boost_thought/tabfact/9b
ot_qa=$TC/results/thought/wikitq/qwen3.5-9b; ot_fv=$TC/results/thought/tabfact/qwen3.5-9b
run_refine new_qa wikitq TableQA "$gt_qa" $G/results_v2/new_qa/9b 5 & J1=$!
run_refine new_fv tabfact TableFV "$gt_fv" $G/results_v2/new_fv/9b 5 & J2=$!
run_refine orig_qa wikitq TableQA "$ot_qa" $TC/results_v2/orig_qa/9b 5 & J3=$!
run_refine orig_fv tabfact TableFV "$ot_fv" $TC/results_v2/orig_fv/9b 5 & J4=$!
wait $J1 $J2 $J3 $J4
echo "[$(date -Is)] ======== 9B 块完成 ========"

# ---- Phase 2B ----
CURRENT=2b
start_vllm 2b
gt_qa=$G/results/boost_thought/wikitq/2b; gt_fv=$G/results/boost_thought/tabfact/2b
ot_qa=$TC/results/thought/wikitq/qwen3.5-2b; ot_fv=$TC/results/thought/tabfact/qwen3.5-2b
run_refine new5_qa wikitq TableQA "$gt_qa" $G/results_v2/new5_qa/2b 5 & J1=$!
run_refine new5_fv tabfact TableFV "$gt_fv" $G/results_v2/new5_fv/2b 5 & J2=$!
run_refine orig_qa wikitq TableQA "$ot_qa" $TC/results_v2/orig_qa_rerun/2b 5 & J3=$!
run_refine orig_fv tabfact TableFV "$ot_fv" $TC/results_v2/orig_fv_rerun/2b 5 & J4=$!
wait $J1 $J2 $J3 $J4
pkill -f "vllm serve" 2>/dev/null || true

echo "[$(date -Is)] ======== 9B/2B 全部完成 ========"
echo "汇总:"
for d in $G/results_v2/new_qa/9b $G/results_v2/new_fv/9b $TC/results_v2/orig_qa/9b $TC/results_v2/orig_fv/9b \
         $G/results_v2/new5_qa/2b $G/results_v2/new5_fv/2b $TC/results_v2/orig_qa_rerun/2b $TC/results_v2/orig_fv_rerun/2b; do
    f=$d/result.txt; [ -f "$f" ] && echo "  ${d#$PKG/}: $(grep -oP 'Accuracy: \K[0-9.]+' $f)"
done
echo "回传: results_v2/ 全目录 + 各组 manifest.json + token_usage.jsonl"
