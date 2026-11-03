#!/usr/bin/env bash
# run_boost_final.sh — R1/R2 最终编排 (4B 单实例 + 4 refine 并发, 顺序: 服务先行)
set -u
cd /root/code/GCRAFT
mkdir -p results/boost/R1/tabfact results/boost/R1/wikitq results/boost/R2/tabfact results/boost/R2/wikitq logs

# 树就位 (幂等)
[ -f results/boost/R1/tree_tabfact.json ] || cp critic/TableQA/tools/few_shot_critic.json results/boost/R1/tree_tabfact.json
[ -f results/boost/R1/tree_wikitq.json ] || cp critic/TableFV/tools/few_shot_critic.json results/boost/R1/tree_wikitq.json
[ -f results/boost/R2/tree_tabfact_seeded.json ] || cp results/boost/trees/merged_fv.json results/boost/R2/tree_tabfact_seeded.json
[ -f results/boost/R2/tree_wikitq_seeded.json ] || cp results/boost/trees/merged_qa.json results/boost/R2/tree_wikitq_seeded.json
# R1/tabfact 曾被 404 版触碰过 → 强制重置为 canonical (保证 R1 语义)
cp critic/TableFV/tools/few_shot_critic.json results/boost/R1/tree_tabfact.json

# ---- 1. 服务先行: 起 vLLM 4B 并确认 ready ----
pkill -f "vllm serve" 2>/dev/null || true; sleep 8
echo "[$(date -Is)] 启动 vLLM 4B..."
CUDA_VISIBLE_DEVICES=0 nohup vllm serve /root/autodl-tmp/models/Qwen3.5-4B \
    --host 127.0.0.1 --port 8000 --served-model-name qwen3.5-4b \
    --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-num-seqs 128 \
    > logs/vllm_boost.log 2>&1 &
for i in $(seq 1 150); do
    curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen3.5-4b && break
    sleep 4
done
curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen3.5-4b || { echo "❌ vLLM 未就绪"; exit 1; }
echo "[$(date -Is)] ✓ vLLM 4B 就绪"

# ---- 2. 四组 refine 并发 ----
run_one() {  # $1=rep $2=ds $3=task_dir $4=tree
    local rep=$1 ds=$2 tdir=$3 tree=$4
    python3 refine/${tdir}/main_tree_based.py \
        --thought_results_dir results/graph_experiments/${ds}/C1_raw/thought \
        --refine_results_dir results/boost/${rep}/${ds} \
        --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
        --model_name qwen3.5-4b --first_n -1 --n_proc 10 --chunk_size 4 \
        --use_clarifier True --enable_p2 False --max_iterations 5 \
        --critic_tree_json "$tree" \
        > logs/boost_${rep}_${ds}.log 2>&1
    echo "[$(date -Is)] ${rep}/${ds}: $(grep -oP 'Accuracy: \K[0-9.]+' results/boost/${rep}/${ds}/result.txt 2>/dev/null || echo FAILED)"
}

manifest() {  # $1=rep $2=ds $3=tree
    python3 - "$1" "$2" "$3" << 'PYEOF'
import json, hashlib, os, sys, time
rep, ds, tree = sys.argv[1], sys.argv[2], sys.argv[3]
origin = {"R1": "canonical", "R2": {"wikitq": "merged_qa", "tabfact": "merged_fv"}[ds]}[rep]
m = {"rep": rep, "dataset": ds, "model": "qwen3.5-4b",
     "thought_input": f"results/graph_experiments/{ds}/C1_raw/thought/final_result.pkl",
     "max_iterations": 5, "use_clarifier": True, "enable_p2": False, "enable_p6": False,
     "n_proc": 10, "chunk_size": 4, "backend": "vllm 32k bf16 greedy",
     "tree": {"seed_path": tree, "size_bytes": os.path.getsize(tree),
              "sha256": hashlib.sha256(open(tree, "rb").read()).hexdigest(), "origin": origin},
     "started": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(m, open(f"results/boost/{rep}/{ds}/manifest.json", "w"), ensure_ascii=False, indent=2)
PYEOF
}

manifest R1 tabfact results/boost/R1/tree_tabfact.json
manifest R1 wikitq results/boost/R1/tree_wikitq.json
manifest R2 tabfact results/boost/R2/tree_tabfact_seeded.json
manifest R2 wikitq results/boost/R2/tree_wikitq_seeded.json

run_one R1 tabfact TableFV results/boost/R1/tree_tabfact.json & J1=$!
run_one R1 wikitq TableQA results/boost/R1/tree_wikitq.json & J2=$!
run_one R2 tabfact TableFV results/boost/R2/tree_tabfact_seeded.json & J3=$!
run_one R2 wikitq TableQA results/boost/R2/tree_wikitq_seeded.json & J4=$!
wait $J1 $J2 $J3 $J4
pkill -f "vllm serve" 2>/dev/null || true

# ---- 3. 汇总 ----
echo "[$(date -Is)] ======== 刷分阶梯完成 ========"
for rep in R1 R2; do for ds in tabfact wikitq; do
    f=results/boost/${rep}/${ds}/result.txt
    echo "  ${rep}/${ds}: $(grep -oP 'Accuracy: \K[0-9.]+' $f 2>/dev/null || echo FAILED)"
done; done
echo "  基线 R0(C1_raw): tabfact=0.8987 wikitq=0.7369"
echo "  原版: tabfact=0.9313 wikitq=0.7884"
