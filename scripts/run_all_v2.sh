#!/usr/bin/env bash
# run_all_v2.sh — 全规模最终编排 v2 (修正版)
# 每模型块: vLLM → G thought 补齐 → 4 组 refine 并发:
#   orig_qa/orig_fv: 原版 refine × 原 thought (重跑, TC_TOKEN_LOG → 两次 acc 方差 + token)
#   new_qa/new_fv:   原版 refine × G thought (新框架, token)
# 树起点: 每模型块开始 git checkout 恢复 canonical 原点 (对齐第一次基线的起点);
#         模型间不重置 (对齐第一次的顺序共享树语义)。
set -u
TC=/root/code/Table-Critic
G=/root/code/GCRAFT
mkdir -p $TC/results_v2 $G/results_v2 $G/results/boost_thought $G/logs

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
    python3 - "$tag" "$ds" "$CURRENT" "$repo" "$out" << 'PYEOF'
import json, hashlib, os, sys, time
tag, ds, size, repo, out = sys.argv[1:6]
tree = f"{repo}/critic/{'TableQA' if ds=='wikitq' else 'TableFV'}/tools/few_shot_critic.json"
m = {"tag": tag, "dataset": ds, "model": f"qwen3.5-{size}",
     "refine_repo": "Table-Critic 原版 (5 轮上限)",
     "thought_input": {"orig": f"Table-Critic results/thought (第一次基线产物)",
                       "new": "G-CRAFT raw+Clarifier thought (C1_raw 4B / boost_thought 其余)"}["new" if tag.startswith("new") else "orig"],
     "max_loop": ${max_loop}, "use_clarifier": tag.startswith("new"),
     "n_proc": 10, "chunk_size": 4, "backend": "vllm 32k bf16 greedy",
     "tree": {"path": tree, "size_bytes": os.path.getsize(tree),
              "sha256": hashlib.sha256(open(tree, "rb").read()).hexdigest(),
              "origin": "canonical 原点 (每模型块 git checkout 恢复)"},
     "token_log": f"{out}/token_usage.jsonl",
     "started": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(m, open(f"{out}/manifest.json", "w"), ensure_ascii=False, indent=2)
PYEOF
    TC_TOKEN_LOG="$out/token_usage.jsonl" \
    python3 $repo/refine/${tdir}/main_tree_based.py \
        --thought_results_dir "$th" \
        --refine_results_dir "$out" \
        --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
        --model_name "qwen3.5-$CURRENT" \
        --first_n -1 --n_proc $nproc --chunk_size 4 $extra \ \
        --max_loop $max_loop > $G/logs/v2_${tag}.log 2>&1
    local acc
    acc=$(grep -oP 'Accuracy: \K[0-9.]+' $out/result.txt 2>/dev/null || echo FAILED)
    python3 - "$out" "$acc" << 'PYEOF2'
import json, sys, time
p, acc = sys.argv[1], sys.argv[2]
m = json.load(open(f"{p}/manifest.json"))
m["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
m["refine_acc"] = acc
json.dump(m, open(f"{p}/manifest.json", "w"), ensure_ascii=False, indent=2)
PYEOF2
    echo "[$(date -Is)] $tag: ${acc}"
}

model_block() {  # $1=size
    CURRENT=$1
    start_vllm $CURRENT
    (cd $TC && git checkout -- critic/TableQA/tools/few_shot_critic.json critic/TableFV/tools/few_shot_critic.json)
    echo "[$(date -Is)] canonical 树已恢复原点 (对齐第一次基线起点)"

    # G thought 补齐 (0.8B/2B/9B; 4B 复用 C1_raw 产物)
    if [ "$CURRENT" != 4b ]; then
        for ds in tabfact wikitq; do
            local tdir=TableFV; [ "$ds" = wikitq ] && tdir=TableQA
            local out=$G/results/boost_thought/${ds}/${CURRENT}
            [ -f "$out/final_result.pkl" ] && continue
            local input=$G/thought/${tdir}/data/${ds}/$([ "$ds" = wikitq ] && echo test_lower.jsonl || echo test.jsonl)
            local extra=""; [ "$ds" = tabfact ] && extra="--raw2clean_path $G/thought/${tdir}/data/tabfact/raw2clean.jsonl"
            echo "[$(date -Is)] G thought ${ds}/${CURRENT}..."
            (cd $G && python3 thought/${tdir}/main.py \
                --dataset_path "$input" \
                --thought_results_dir "$out" \
                --base_url http://127.0.0.1:8000/v1 --openai_api_key EMPTY \
                --model_name "qwen3.5-$CURRENT" \
                --first_n -1 --n_proc 16 --chunk_size 4 --use_clarifier True $extra \
                > $G/logs/v2_bt_${ds}_${CURRENT}.log 2>&1)
        done
    fi

    local gt_qa=$G/results/graph_experiments/wikitq/C1_raw/thought
    local gt_fv=$G/results/graph_experiments/tabfact/C1_raw/thought
    [ "$CURRENT" != 4b ] && { gt_qa=$G/results/boost_thought/wikitq/${CURRENT}; gt_fv=$G/results/boost_thought/tabfact/${CURRENT}; }

    run_refine v2_orig_qa wikitq TableQA $TC/results/thought/wikitq/qwen3.5-${CURRENT}   $TC/results_v2/orig_qa/${CURRENT}    $TC 10 5 & J1=$!
    run_refine v2_orig_fv tabfact TableFV $TC/results/thought/tabfact/qwen3.5-${CURRENT}  $TC/results_v2/orig_fv/${CURRENT}    $TC 10 5 & J2=$!
    run_refine v2_new_qa  wikitq TableQA "$gt_qa" $G/results_v2/new_qa/${CURRENT}         $TC 10 5 & J3=$!
    run_refine v2_new_fv  tabfact TableFV "$gt_fv" $G/results_v2/new_fv/${CURRENT}        $TC 10 5 & J4=$!
    wait $J1 $J2 $J3 $J4
    echo "[$(date -Is)] ======== 模型 $CURRENT 全部完成 ========"
}

for size in 0.8b 2b 4b 9b; do model_block $size; done
pkill -f "vllm serve" 2>/dev/null || true

echo "[$(date -Is)] ======== 全部完成, 汇总 ========"
echo "组            | acc"
for size in 0.8b 2b 4b 9b; do
    for pair in "orig_qa:orig_qa" "orig_fv:orig_fv" "new_qa:new_qa" "new_fv:new_fv"; do
        tag=${pair%%:*}; sub=${pair##*:}
        repo=$TC; [ "${tag#new}" != "$tag" ] && repo=$G
        f=$repo/results_v2/${tag}/${size}/result.txt
        [ -f "$f" ] && echo "  ${size}/${tag}: $(grep -oP 'Accuracy: \K[0-9.]+' $f)"
    done
done
echo "参照 (第一次基线): $(for m in 0.8b 2b 4b 9b; do for ds in wikitq tabfact; do echo -n "$ds/$m=$(grep -oP 'Accuracy: \K[0-9.]+' /root/code/Table-Critic/results/refine/$ds/qwen3.5-${m}/result.txt 2>/dev/null) "; done; done)"
