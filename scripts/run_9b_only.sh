#!/usr/bin/env bash
# run_9b_only.sh v2 — 用户指令 (11-07): 只跑完 9B 实验, 其余全部停跑
# 顺序: new_qa_9b (缓存续) → orig_qa_9b → orig_fv_9b → 停机
# v2 自愈守护: 12:23 EngineCore 被外部 SIGTERM 杀死导致三组全 FAILED (来源未明),
#   每组开始前 ensure_vllm 检查存活 + 死亡自动重启, 每组失败最多重试 3 次 (缓存续跑幂等)
set -u
echo $$ > /tmp/gcraft_9b.pid   # 供守护进程存活检测
TC=/root/code/Table-Critic
G=/root/code/GCRAFT
BASE=http://127.0.0.1:8000/v1

vllm_alive() { curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "qwen3.5-9b"; }

kill_all_vllm() {
    pkill -9 -f "VLLM::[E]ngineCore" 2>/dev/null || true
    pkill -9 -f "vllm [s]erve" 2>/dev/null || true
    sleep 8
}

ensure_vllm() {
    if vllm_alive; then echo "[$(date -Is)] vLLM 9B 存活, 复用"; return 0; fi
    kill_all_vllm
    echo "[$(date -Is)] vLLM 9B 启动..."
    CUDA_VISIBLE_DEVICES=0 nohup vllm serve /root/autodl-tmp/models/Qwen3.5-9B \
        --host 127.0.0.1 --port 8000 --served-model-name "qwen3.5-9b" \
        --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-num-seqs 128 \
        > $G/logs/vllm_9bonly.log 2>&1 &
    for i in $(seq 1 150); do
        vllm_alive && { echo "[$(date -Is)] ✓ vLLM 9B 就绪"; return 0; }
        sleep 4
    done
    echo "❌ vLLM 9B 启动失败"; return 1
}

run_one() {  # $1=tag $2=ds $3=tdir $4=thought_dir $5=out_dir $6=size $7=max_loop
    local tag=$1 ds=$2 tdir=$3 th=$4 out=$5 size=$6 ml=$7
    [ -f "$out/result.txt" ] && { echo "[$(date -Is)] $tag 已有, 跳过"; return 0; }
    local extra=""
    [ "$ds" = tabfact ] && extra="--raw2clean_path $TC/thought/$tdir/data/tabfact/raw2clean.jsonl"
    echo "[$(date -Is)] ======== $tag ========"
    mkdir -p "$out"
    python3 - "$tag" "$ds" "$size" "$out" "$ml" << PYEOF
import json, hashlib, os, sys, time
tag, ds, size, out, max_loop = sys.argv[1:6]
tree = "$TC/critic/" + ("TableQA" if ds == "wikitq" else "TableFV") + "/tools/few_shot_critic.json"
m = {"tag": tag, "dataset": ds, "model": "qwen3.5-" + size,
     "refine_repo": "Table-Critic 原版", "max_loop": int(max_loop),
     "thought_input": "G raw+Clarifier" if tag.startswith("new") else "原版第一次基线产物",
     "n_proc": 16, "chunk_size": 4, "backend": "vllm 32k bf16 greedy",
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
            --first_n -1 --n_proc 16 --chunk_size 4 $extra \
            --max_loop $ml \
            > $G/logs/only9b_${tag}.log 2>&1
    )
    grep -oP 'Accuracy: \K[0-9.]+' $out/result.txt 2>/dev/null || echo "[$(date -Is)] $tag 本轮无 result.txt"
}

run_with_retry() {  # 参数同 run_one, 失败自动 ensure_vllm + 重试, 最多 3 轮
    local tag=$1; shift
    local attempt
    for attempt in 1 2 3; do
        ensure_vllm || { sleep 30; continue; }
        run_one "$tag" "$@"
        # out_dir 是第 4 参数 (shift 后 $1=ds $2=tdir $3=th $4=out)
        [ -f "$4/result.txt" ] && return 0
        echo "[$(date -Is)] ⚠ $tag 第 $attempt 轮失败 (缓存进度保留), 90s 后重试"
        sleep 90
    done
    echo "❌ $tag 三轮重试均失败"; return 1
}

run_with_retry new_qa_9b  wikitq TableQA $G/results/boost_thought/wikitq/9b $G/results_v2/new_qa/9b 9b 5
run_with_retry orig_qa_9b wikitq TableQA $TC/results/thought/wikitq/qwen3.5-9b $TC/results_v2/orig_qa/9b 9b 5
run_with_retry orig_fv_9b tabfact TableFV $TC/results/thought/tabfact/qwen3.5-9b $TC/results_v2/orig_fv/9b 9b 5
kill_all_vllm

echo "[$(date -Is)] ======== 9B 全部完成, 按指令停机 ========"
for d in $G/results_v2/new_qa/9b $TC/results_v2/orig_qa/9b $TC/results_v2/orig_fv/9b; do
    f=$d/result.txt; [ -f "$f" ] && echo "  ${d#$G/}: $(grep -oP 'Accuracy: \K[0-9.]+' $f)"
done
