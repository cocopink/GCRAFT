"""P4 信号探测统计报告 (E5, 纯 CPU 零 LLM)

读 p4_extract_features.py 的特征表, 按 docs/experiment_plan_p1_p6.md §6 输出:
- AUROC / AUPRC (附正类率参照) + bootstrap 95% CI
- 固定 FPR @5% / @10% 的 TPR (P6 CONFLICT 阈值依据)
- 链长分层 AUROC (链长混淆控制, 层内增量 vs B2)
- 逐特征 vs 秩融合组合消融
- 特征 Pearson 相关矩阵
- F6 三态可算率 / F1 对齐失败率

判定标准 (§6.5, 预注册): 主特征 F1/F2 在链长分层内且金标准口径下,
相对最强非图基线增量 AUROC >= 0.03 且 CI 不含 0 → 有效信号。

用法:
    python scripts/p4_evaluate.py --features results/p4/features_wikitq_4b.pkl \
        [--features results/p4/features_wikitq_9b.pkl ...] \
        --output results/p4/report_wikitq.md
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p4_probing_lib import (  # noqa: E402
    auprc,
    auroc,
    bootstrap_ci,
    pearson_corr,
    tpr_at_fpr,
)

FEATURE_COLS = [
    "f1_col_gap", "f2_row_loss", "f4_extreme_removal",
    "f5_row_evidence", "f5_col_evidence", "f6_col_rank", "b1_conf", "chain_len",
]
# 方向约定: 值越大越可能为错 (f5/f6 高=证据保留好 → 取负向)
FEATURE_DIRS = {
    "f1_col_gap": 1, "f2_row_loss": 1, "f4_extreme_removal": 1,
    "f5_row_evidence": -1, "f5_col_evidence": -1, "f6_col_rank": -1,
    "b1_conf": -1, "chain_len": 1,
}
MAIN_FEATURES = ["f1_col_gap", "f2_row_loss"]  # 预注册主特征 (闸门判定用)
BASELINES = ["b1_conf", "chain_len"]  # 非图基线 B1/B2


def load_features(paths):
    feats = []
    for path in paths:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        meta = payload["meta"]
        print(f"Loaded {meta['n_features']} features from {path} (task={meta['task']})")
        feats.extend(payload["features"])
    return feats


def rank_fuse(matrix, dirs):
    """多特征秩融合: 各特征转为 [0,1] 归一化秩 (按方向), 取平均作为组合分。"""
    ranks = []
    for j, d in enumerate(dirs):
        col = matrix[:, j]
        order = np.argsort(np.argsort(col, kind="mergesort"), kind="mergesort")
        norm = order / max(len(col) - 1, 1)
        ranks.append(norm if d > 0 else 1 - norm)
    return np.mean(ranks, axis=0)


def main():
    parser = argparse.ArgumentParser(description="P4 信号探测统计报告")
    parser.add_argument("--features", nargs="+", required=True, help="特征表 pkl (可多模型合并)")
    parser.add_argument("--output", default="", help="markdown 报告输出路径")
    parser.add_argument("--layer_min", type=int, default=3, help="链长分层下界")
    parser.add_argument("--layer_max", type=int, default=6, help="链长分层上界 (含), 以上合并为 6+")
    args = parser.parse_args()

    feats = load_features(args.features)
    usable = [x for x in feats if x["label"] in (0, 1)]
    labels = np.array([x["label"] for x in usable])
    n_pos = int(labels.sum())
    err_rate = n_pos / len(usable) * 100
    print(f"\n可用样本: {len(usable)} (正类率/错误率 = {n_pos}/{len(usable)} = {err_rate:.2f}%)")

    lines = []
    lines.append("# P4 图错误信号探测报告 (E5)")
    lines.append(f"- 特征表: {', '.join(args.features)}")
    lines.append(f"- 可用样本: {len(usable)}, 正类率(错误率) = {err_rate:.2f}% ({n_pos}/{len(usable)})")
    lines.append(f"- 口径: 全量 EM 标签 (金标准子集标注完成后须单独重跑报告)")
    lines.append("")

    # ===== 主表: 各特征 AUROC/AUPRC/CI =====
    lines.append("## 1. 逐特征区分力 (主特征 = F1/F2, 预注册闸门)")
    lines.append("")
    lines.append("| 特征 | 方向 | AUROC | 95% CI | AUPRC | TPR@FPR5% | TPR@FPR10% |")
    lines.append("|------|------|-------|--------|-------|-----------|------------|")
    matrix = np.array([[x[c] if x[c] is not None else np.nan for c in FEATURE_COLS] for x in usable])
    scores_map = {}
    for j, col in enumerate(FEATURE_COLS):
        raw = matrix[:, j]
        valid = ~np.isnan(raw)
        if valid.sum() < 10 or n_pos == 0:
            lines.append(f"| {col} | - | nan | - | nan | - | - |")
            continue
        d = FEATURE_DIRS[col]
        s = np.where(valid, raw * d, np.nan)  # 方向统一: 越大越可疑; nan 保留
        # nan 填充为特征最小值 (缺失=最不可疑)
        fill = np.nanmin(s) - 1e-9
        s_full = np.where(valid, s, fill)
        scores_map[col] = s_full
        ci = bootstrap_ci(s_full, labels, auroc)
        lines.append(
            f"| {col} | {'+' if d > 0 else '-'} | {auroc(s_full, labels):.4f} "
            f"| [{ci[0]:.4f}, {ci[1]:.4f}] | {auprc(s_full, labels):.4f} "
            f"| {tpr_at_fpr(s_full, labels, 0.05):.4f} | {tpr_at_fpr(s_full, labels, 0.10):.4f} |"
        )
    lines.append("")

    # ===== 组合消融: 逐特征 vs 主特征组合 vs 秩融合 =====
    lines.append("## 2. 组合消融 (秩融合: 各特征归一化秩平均)")
    combos = {
        "F1+F2 (主特征组合)": ["f1_col_gap", "f2_row_loss"],
        "全部图特征": ["f1_col_gap", "f2_row_loss", "f4_extreme_removal",
                       "f5_row_evidence", "f5_col_evidence", "f6_col_rank"],
        "图特征+B1": ["f1_col_gap", "f2_row_loss", "f4_extreme_removal", "b1_conf"],
    }
    lines.append("")
    lines.append("| 组合 | AUROC | 95% CI | 相对最强基线增量 |")
    lines.append("|------|-------|--------|------------------|")
    base_aurocs = {b: auroc(scores_map[b], labels) for b in BASELINES if b in scores_map}
    strongest_base = max(base_aurocs.values()) if base_aurocs else float("nan")
    for name, cols in combos.items():
        cols_ok = [c for c in cols if c in scores_map]
        if len(cols_ok) < len(cols):
            continue
        fused = rank_fuse(np.stack([scores_map[c] for c in cols_ok], axis=1),
                          [FEATURE_DIRS[c] for c in cols_ok])
        ci = bootstrap_ci(fused, labels, auroc)
        delta = auroc(fused, labels) - strongest_base
        lines.append(f"| {name} | {auroc(fused, labels):.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {delta:+.4f} |")
    lines.append("")

    # ===== 链长分层 (混淆控制) =====
    lines.append("## 3. 链长分层 AUROC (链长混淆控制)")
    chain_lens = np.array([x["chain_len"] for x in usable])
    layers = list(range(args.layer_min, args.layer_max)) + [args.layer_max]  # max 档 = >=layer_max
    lines.append("")
    lines.append("| 链长 | n | 错误率 | F1 AUROC | F2 AUROC | B1 AUROC | 链长 AUROC | F1 相对最强基线增量 |")
    lines.append("|------|---|--------|----------|----------|----------|------------|---------------------|")
    for lo in layers:
        if lo == args.layer_max:
            mask = chain_lens >= lo
            tag = f">={lo}"
        else:
            mask = chain_lens == lo
            tag = str(lo)
        sub = mask & ~np.isnan(matrix[:, FEATURE_COLS.index("f1_col_gap")])
        if sub.sum() < 20 or labels[sub].sum() in (0, sub.sum()):
            lines.append(f"| {tag} | {int(sub.sum())} | - | - | - | - | - | - |")
            continue
        a_f1 = auroc(scores_map["f1_col_gap"][sub], labels[sub])
        a_f2 = auroc(scores_map["f2_row_loss"][sub], labels[sub])
        a_b1 = auroc(scores_map["b1_conf"][sub], labels[sub])
        a_len = auroc(scores_map["chain_len"][sub], labels[sub])
        strongest = max(a_b1, a_len)
        lines.append(
            f"| {tag} | {int(sub.sum())} | {labels[sub].mean()*100:.1f}% "
            f"| {a_f1:.4f} | {a_f2:.4f} | {a_b1:.4f} | {a_len:.4f} | {a_f1 - strongest:+.4f} |"
        )
    lines.append("")

    # ===== 特征相关矩阵 =====
    lines.append("## 4. 特征 Pearson 相关矩阵")
    valid_rows = ~np.isnan(matrix).any(axis=1)
    if valid_rows.sum() > 10:
        corr = pearson_corr(matrix[valid_rows])
        lines.append("")
        lines.append("| | " + " | ".join(FEATURE_COLS) + " |")
        lines.append("|---" * (len(FEATURE_COLS) + 1) + "|")
        for i, c in enumerate(FEATURE_COLS):
            lines.append(f"| {c} | " + " | ".join(f"{corr[i, j]:.2f}" for j in range(len(FEATURE_COLS))) + " |")
    lines.append("")

    # ===== F6 / 对齐失败 =====
    f6_statuses = {}
    for x in usable:
        f6_statuses[x["f6_status"]] = f6_statuses.get(x["f6_status"], 0) + 1
    total_align_calls = sum(
        x["op_types"].get("select_column", 0) + x["op_types"].get("sort_column", 0)
        + x["op_types"].get("group_column", 0) for x in usable)
    unmatched = sum(x["unmatched_cols"] for x in usable)
    lines.append("## 5. 口径说明")
    lines.append(f"- F6 答案可追溯分布: {f6_statuses}")
    lines.append(f"- F1 列名对齐失败: {unmatched}/{total_align_calls} "
                 f"({unmatched/max(total_align_calls,1)*100:.2f}%)")
    lines.append(f"- 特征覆盖边界: F1(推广)+F2 覆盖 select_row/select_column/sort_column/"
                 f"group_column 四类操作; add_column/simple_query 无图特征覆盖 "
                 f"(UNKNOWN 理论下限)")
    lines.append("")
    lines.append("## 判定提醒 (预注册 §6.5)")
    lines.append("- 闸门判定只用主特征 F1/F2; F3-F6 为探索性报告, 不参与 go/no-go")
    lines.append("- 有效信号标准: 链长分层内 + 金标准口径下, 相对最强非图基线增量 AUROC >= 0.03 且 CI 不含 0")
    lines.append("- 本报告为全量 EM 口径; EM 噪声主导时结论降权 (与金标准口径分别报告)")

    report = "\n".join(lines)
    print(report)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n📄 报告已写入: {args.output}")


if __name__ == "__main__":
    main()
