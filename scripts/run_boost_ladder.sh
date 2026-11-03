#!/usr/bin/env bash
# run_boost_ladder.sh — G-CRAFT 刷分阶梯 (4B, 复用 C1_raw thought 产物, refine-only)
#
# R1: raw thought + max_iterations=5 + canonical 私有树播种 (轮数对齐原版)
# R2: R1 + 原版实战树播种 (QA: 原版进化树快照; FV: orig 大树 306KB, 6KB 进化树异常弃用)
# 树策略: 拷贝至 boost 目录作为私有 EVOLVE 树 (frozen=False, 运行内继续进化, 不碰原版原件)
# 评估: refine 自写 result.txt (match_func 口径)
set -u
cd /root/code/GCRAFT
mkdir -p results/boost logs

BASE_URL="http://127.0.0.1:8000/v1"
KEY="EMPTY"

# ---- 等 Table-Critic 基线队列结束 ----
echo "[$(date -Is)] 等待 Table-Critic 基线队列结束..."
while ps aux | grep run_qwen35_baseline_parallel | grep -v grep > /dev/null 2>&1; do sleep 30; done
echo "[$(date -Is)] 基线队列已结束"

# ---- 快照原版 QA 进化树 (9B 完成后 T4 定格) ----
mkdir -p results/boost/trees
cp /root/code/Table-Critic/critic/TableQA/tools/few_shot_critic.json results/boost/trees/tc_evolved_qa.json
cp /root/code/Table-Critic/critic/TableFV/tools/few_shot_critic.json results/boost/trees/tc_evolved_fv.json
echo "[$(date -Is)] 原版树快照: QA=$(du -h results/boost/trees/tc_evolved_qa.json | cut -f1) FV=$(du -h results/boost/trees/tc_evolved_fv.json | cut -f1)"

# ---- 单实例 + 4 客户端并发 (R1/R2 独立配置共享完整 KV 池; 多实例分池方案已证伪) ----
start_vllm 8000 0.85 /root/autodl-tmp/models/Qwen3.5-4B qwen3.5-4b

run_one() {  # $1=rep $2=ds
    local rep=$1 ds=$2
    local task=TableFV; [ "$ds" = wikitq ] && task=TableQA
    local tree
    if [ "$rep" = R1 ]; then
        tree="results/boost/R1/tree_${ds}.json"
    else
        tree="results/boost/R2/tree_${ds}_seeded.json"
        if [ "$ds" = wikitq ]; then
            cp results/boost/trees/merged_qa.json "$tree"
        else
            cp results/boost/trees/merged_fv.json "$tree"
        fi
    fi
    echo "[$(date -Is)] ======== ${rep}/${ds} refine (5 轮) ========"
    mkdir -p results/boost/${rep}/${ds}
    cp "$tree" "results/boost/${rep}/${ds}/tree_seed_snapshot.json"
    sha256sum "$tree" | awk '{print $1}' > "results/boost/${rep}/${ds}/tree_seed_sha256.txt"
    python3 - << PYEOF
import json, hashlib, os, time
tree = "${tree}"
m = {
  "rep": "${rep}", "dataset": "${ds}", "model": "qwen3.5-4b",
  "thought_input": "results/graph_experiments/${ds}/C1_raw/thought/final_result.pkl",
  "max_iterations": 5, "use_clarifier": True, "enable_p2": False,
  "enable_p6": False, "bias_weight": 0.0, "n_proc": 10, "chunk_size": 4,
  "backend": "vllm 32k bf16 greedy, 单实例共享",
  "tree": {"seed_path": tree, "size_bytes": os.path.getsize(tree),
           "sha256": hashlib.sha256(open(tree, "rb").read()).hexdigest(),
           "origin": "canonical" if "${rep}" == "R1" else ("merged_qa" if "${ds}" == "wikitq" else "merged_fv")},
  "started": time.strftime("%Y-%m-%d %H:%M:%S"),
}
json.dump(m, open("results/boost/${rep}/${ds}/manifest.json", "w"), ensure_ascii=False, indent=2)
PYEOF
    python3 refine/${task}/main_tree_based.py \
        --thought_results_dir results/graph_experiments/${ds}/C1_raw/thought \
        --refine_results_dir results/boost/${rep}/${ds} \
        --base_url $BASE_URL --openai_api_key $KEY \
        --model_name qwen3.5-4b --first_n -1 --n_proc 10 --chunk_size 4 \
        --use_clarifier True --enable_p2 False \
        --max_iterations 5 \
        --critic_tree_json "$tree" \
        > logs/boost_${rep}_${ds}.log 2>&1
    local acc
    acc=$(grep -oP 'Accuracy: \K[0-9.]+' results/boost/${rep}/${ds}/result.txt 2>/dev/null || echo FAILED)
    python3 - << PYEOF2
import json
p = "results/boost/${rep}/${ds}/manifest.json"
m = json.load(open(p))
m["finished"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
m["refine_acc"] = "${acc}"
json.dump(m, open(p, "w"), ensure_ascii=False, indent=2)
PYEOF2
    echo "[$(date -Is)] ${rep}/${ds}: ${acc}"
}

run_one R1 tabfact & J1=$!
run_one R1 wikitq & J2=$!
run_one R2 tabfact & J3=$!
run_one R2 wikitq & J4=$!
wait $J1 $J2 $J3 $J4
pkill -f "vllm serve" 2>/dev/null || true

echo "[$(date -Is)] 刷分阶梯全部完成, 汇总:"
for rep in R1 R2; do for ds in tabfact wikitq; do
    f=results/boost/${rep}/${ds}/result.txt
    [ -f "$f" ] && echo "  ${rep}/${ds}: $(grep -oP 'Accuracy: \K[0-9.]+' $f)"
done; done
echo "  基线 C1_raw: tabfact=0.8987 wikitq=0.7369 | 原版: tabfact=0.9313 wikitq=0.7884"
