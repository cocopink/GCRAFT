#!/usr/bin/env bash
# run_e2_matrix.sh — E2 消融主矩阵启动器 (docs/experiment_plan_p1_p6.md §9 Phase 2)
#
# 一遍 = 18 跑次 (每数据集 thought 3 + refine 6, C4/C5/C6 复用 thought 产物)。
# 重复系数由 E1 判定 (e1_variance.tsv); 多遍重复时重复调用本脚本即可,
# 每遍独立 → 由 run_e2_rep_wrapper 的清产物逻辑保证 (见下)。
#
# 口径: 4B / 32k / 全量 first_n=-1 / clarifier=True / n_proc 24 / 每配置私有进化树。
#
# 用法: REPEAT=2 nohup bash scripts/run_e2_matrix.sh > logs/e2_matrix_rep2.log 2>&1 &
set -u

REPEAT="${REPEAT:-1}"   # 本脚本自身循环的遍数 (E1 判定 ×3 → 建议 REPEAT=3)
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-4b}"

for rep in $(seq 1 "$REPEAT"); do
    echo "=========================================="
    echo "=== E2 全矩阵 rep=${rep}/${REPEAT} start=$(date -Is) ==="
    echo "=========================================="
    # E1 的 pilot thought/refine 产物 (first_n=100) 留有缓存 → 清除防污染全量口径;
    # data_atgo.jsonl / atgo_cache 为全量确定性预处理产物, 保留复用。
    # (C2_atgo 之外的配置无历史产物, 清除为幂等空操作)
    for ds in tabfact wikitq; do
        for cfg in C1_raw C2_atgo C3_p1 C4_p2 C5_p1p2 C6_p1p2p3; do
            rm -rf "results/graph_experiments/${ds}/${cfg}/thought" \
                   "results/graph_experiments/${ds}/${cfg}/refine"
        done
    done
    python3 scripts/run_graph_experiment_matrix.py \
        --configs C1_raw C2_atgo C3_p1 C4_p2 C5_p1p2 C6_p1p2p3 \
        --datasets tabfact wikitq \
        --first_n -1 --n_proc 24 --use_clarifier --run_llm \
        --base_url "$BASE_URL" --model_name "$MODEL_NAME"
    rc=$?
    echo "=== E2 rep=${rep} end=$(date -Is) rc=${rc} ==="
    if [ "$rc" -ne 0 ]; then
        echo "!!! E2 rep=${rep} FAILED rc=${rc}, 终止后续遍 (先排查再重跑)" >&2
        exit "$rc"
    fi
done
echo "E2 all done: $(date -Is)"
