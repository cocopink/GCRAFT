"""P5 定位评估 (E6): Hit@1 / Hit@3, 双基线对照 (docs/experiment_plan_p1_p6.md §7.2)

对照列:
(a) P5 异常分 (execution_graph.step_anomaly_score, 含执行关系)
(b) P4 静态图特征基线 (per-step F1/F2 max, 无执行关系)
(c) 现有 critic 定位基线 (可选 --critic_pkl, GPU 前置, 缺席时标注待补)

判定映射: 双基线胜出 → P5 异常分进 P6; 输给任一基线 → P6 一期退化为仅 P4 特征触发。

用法:
    python scripts/p5_evaluate_localization.py \
        --injected results/p5/injected_wikitq_4b.pkl \
        --output results/p5/localization_wikitq_4b.md
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p4_probing_lib import parse_col_names, trace_row_sets  # noqa: E402
from table_structure.execution_graph import (  # noqa: E402
    build_execution_relations,
    step_anomaly_score,
)
from table_structure.value_similarity import normalize_value  # noqa: E402


def localize_by_anomaly(sample, trace):
    """方法 (a): P5 异常分 top-k 步骤号。"""
    if trace is None:
        return []
    exec_graph = build_execution_relations(trace)
    ranked = step_anomaly_score(exec_graph)
    return [step for step, _ in ranked]


def localize_by_static(sample, trace):
    """方法 (b): P4 静态特征 per-step max (F1 gap 与 F2 loss 的最大步)。"""
    if trace is None:
        return []
    col_scores = trace["col_scores_norm"]
    row_scores = trace["row_scores"]
    max_col = max(col_scores.values()) if col_scores else 0.0
    step_scores = {}
    for st in trace["steps"]:
        score = 0.0
        if st["op"] in ("select_column", "sort_column", "group_column"):
            gaps = [max_col - col_scores[c] for c in st.get("cols_selected", [])
                    if c in col_scores]
            if gaps:
                score = max(gaps)
        elif st["op"] == "select_row" and st["rows_removed"]:
            before = sum(row_scores.get(r, 0.0) for r in st["rows_before"])
            removed = sum(row_scores.get(r, 0.0) for r in st["rows_removed"])
            if before > 0:
                score = removed / before
        step_scores[st["step"]] = score
    if not step_scores or max(step_scores.values()) <= 0:
        return []
    return sorted(step_scores, key=step_scores.get, reverse=True)


def hit_at_k(ranked_steps, label_step, k):
    """label_step 为 None (wrong_aggregate, 无步骤定位) 时计入 n_a (不可判)。"""
    if label_step is None:
        return None
    return 1 if label_step in ranked_steps[:k] else 0


def main():
    parser = argparse.ArgumentParser(description="P5 定位评估 (Hit@1/Hit@3)")
    parser.add_argument("--injected", required=True, help="p5_inject_controlled_errors 输出")
    parser.add_argument("--critic_pkl", default="", help="critic 定位基线产物 (可选, GPU 前置)")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    with open(args.injected, "rb") as f:
        payload = pickle.load(f)
    injected = payload["injected"]
    print(f"Loaded {len(injected)} injected samples from {args.injected}")

    methods = {
        "P5 异常分 (EXCLUDES×DEPENDS_ON)": localize_by_anomaly,
        "P4 静态特征 (F1/F2 max)": localize_by_static,
    }
    hits = {m: {1: [], 3: []} for m in methods}
    by_type = defaultdict(lambda: {m: {1: [0, 0], 3: [0, 0]} for m in methods})

    for item in injected:
        sample = item["sample"]
        label_step = item["injection"]["label_step"]
        itype = item["injection"]["type"]
        trace = trace_row_sets(sample)
        for m, fn in methods.items():
            ranked = fn(sample, trace)
            for k in (1, 3):
                h = hit_at_k(ranked, label_step, k)
                if h is not None:
                    hits[m][k].append(h)
                    by_type[itype][m][k][0] += h
                    by_type[itype][m][k][1] += 1

    lines = []
    lines.append("# P5 定位评估 (E6)")
    lines.append(f"- 注入集: {args.injected} ({len(injected)} 条)")
    lines.append("")
    lines.append("## Hit@1 / Hit@3 对照")
    lines.append("")
    lines.append("| 方法 | Hit@1 | Hit@3 |")
    lines.append("|------|-------|-------|")
    for m in methods:
        h1 = hits[m][1]
        h3 = hits[m][3]
        lines.append(f"| {m} | {sum(h1)}/{len(h1)} ({sum(h1)/max(len(h1),1)*100:.1f}%) "
                     f"| {sum(h3)}/{len(h3)} ({sum(h3)/max(len(h3),1)*100:.1f}%) |")

    lines.append("")
    lines.append("## 按注入类型细分")
    lines.append("")
    lines.append("| 类型 | 方法 | Hit@1 | Hit@3 |")
    lines.append("|------|------|-------|-------|")
    for itype, per_method in sorted(by_type.items()):
        for m in methods:
            h1 = per_method[m][1]
            h3 = per_method[m][3]
            lines.append(f"| {itype} | {m} | {h1[0]}/{h1[1]} | {h3[0]}/{h3[1]} |")

    if args.critic_pkl:
        with open(args.critic_pkl, "rb") as f:
            critic_loc = pickle.load(f)
        # 口径对齐: critic 基线输出形态与排序法不同构——
        #   [Incorrect] Step N → 定位预测 (单步); FIXED → 修正成功但无步号输出;
        #   None → 判错且无步号。双口径: 保守 FIXED=miss, 宽松 FIXED=hit。
        crit_strict = {1: [], 3: []}
        crit_lenient = {1: [], 3: []}
        n_fixed = 0
        for item in injected:
            label_step = item["injection"]["label_step"]
            if label_step is None:
                continue
            sid = item["sample"].get("id") or item["sample"].get("table_id")
            loc = critic_loc.get(sid)
            if loc == "FIXED":
                n_fixed += 1
                crit_strict[1].append(0); crit_strict[3].append(0)
                crit_lenient[1].append(1); crit_lenient[3].append(1)
            else:
                ranked = [loc] if isinstance(loc, int) else []
                crit_strict[1].append(1 if label_step in ranked[:1] else 0)
                crit_strict[3].append(1 if label_step in ranked[:3] else 0)
                crit_lenient[1].append(crit_strict[1][-1])
                crit_lenient[3].append(crit_strict[3][-1])
        lines.append("")
        lines.append("## 方法 (c): critic 定位基线 (双口径)")
        lines.append(f"- 来源: {args.critic_pkl} ({len(critic_loc)} 条)")
        lines.append(f"- 注入集上修正成功 (FIXED, 无步号输出): {n_fixed} 条")
        lines.append(f"- 口径 A 保守 (FIXED=miss): Hit@1 {sum(crit_strict[1])}/{len(crit_strict[1])} "
                     f"({sum(crit_strict[1])/max(len(crit_strict[1]),1)*100:.1f}%), "
                     f"Hit@3 {sum(crit_strict[3])}/{len(crit_strict[3])} "
                     f"({sum(crit_strict[3])/max(len(crit_strict[3]),1)*100:.1f}%)")
        lines.append(f"- 口径 B 宽松 (FIXED=hit, 修正成功隐含定位成功): Hit@1 {sum(crit_lenient[1])}/{len(crit_lenient[1])} "
                     f"({sum(crit_lenient[1])/max(len(crit_lenient[1]),1)*100:.1f}%), "
                     f"Hit@3 {sum(crit_lenient[3])}/{len(crit_lenient[3])} "
                     f"({sum(crit_lenient[3])/max(len(crit_lenient[3]),1)*100:.1f}%)")
    else:
        lines.append("")
        lines.append("## 方法 (c): critic 定位基线")
        lines.append("- 待补: 需对注入集跑一次现有 critic/tree 管线 (E6 GPU 前置, `--critic_pkl` 接入)")
        lines.append("- 判定提醒: 双基线 (a) vs (b) vs (c) 缺一不可, (c) 缺席时不做最终判定")

    report = "\n".join(lines)
    print(report)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n📄 报告已写入: {args.output}")


if __name__ == "__main__":
    main()
