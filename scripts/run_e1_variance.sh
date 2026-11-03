#!/usr/bin/env bash
# run_e1_variance.sh — E1 方差与耗时基准 (docs/experiment_plan_p1_p6.md §9 Phase 1)
#
# C2_atgo (E2 对照锚点) × pilot(first_n=100) × 3 次独立重跑 × 2 数据集。
# 每次重复前清 thought/refine 产物目录 (缓存命中 = 假重复, §4.3);
# 预处理产物 data_atgo.jsonl / atgo_cache 保留 (确定性 CPU 计算, 重复无污染)。
# 进化树为每配置私有树 (矩阵脚本注入, 随 refine 目录一并清除 → 每轮从 canonical 重播种)。
#
# 产出: e1_variance.tsv (每轮墙钟时间 + thought/refine acc) → 运行间 SD 与实测吞吐。
#
# 用法: bash scripts/run_e1_variance.sh   (后台: nohup bash scripts/run_e1_variance.sh &)
set -u

REPS="${REPS:-3}"
FIRST_N="${FIRST_N:-100}"
N_PROC="${N_PROC:-24}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-4b}"
OUT_TSV="${OUT_TSV:-results/graph_experiments/e1_variance.tsv}"

echo -e "dataset\trep\tstart\tend\twall_sec\tthought_acc\trefine_acc" > "$OUT_TSV"

for ds in tabfact wikitq; do
    for rep in $(seq 1 "$REPS"); do
        exp="results/graph_experiments/${ds}/C2_atgo"
        rm -rf "$exp/thought" "$exp/refine"
        start=$(date +%s)
        echo "=== [${ds}] rep=${rep} start=$(date -Is) ==="
        python scripts/run_graph_experiment_matrix.py \
            --configs C2_atgo --datasets "$ds" --first_n "$FIRST_N" \
            --n_proc "$N_PROC" --use_clarifier --run_llm \
            --base_url "$BASE_URL" --model_name "$MODEL_NAME"
        rc=$?
        end=$(date +%s)
        t_acc=$(grep -oP 'Accuracy: \K[0-9.]+' "$exp/thought/acc.txt" 2>/dev/null || echo NA)
        r_acc=$(grep -oP 'Accuracy: \K[0-9.]+' "$exp/refine/acc.txt" 2>/dev/null || echo NA)
        echo -e "${ds}\t${rep}\t${start}\t${end}\t$((end - start))\t${t_acc}\t${r_acc}" | tee -a "$OUT_TSV"
        if [ "$rc" -ne 0 ]; then
            echo "!!! rep=${rep} ${ds} FAILED rc=${rc}, 停止该数据集剩余重复 (先排查再重跑)" >&2
            break
        fi
    done
done

echo ""
echo "=== E1 汇总 ==="
column -t -s $'\t' "$OUT_TSV"
echo "E1 done: $(date -Is)"
