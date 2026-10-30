"""P6 决策实验评估 (E7, docs/experiment_plan_p1_p6.md §8)

对照 P6 开启前后的 refine 产物 pkl, 报告:
1. P6 核验统计: verdict 分布 (CONFLICT/UNKNOWN)、CONFLICT 覆盖次数
2. 覆盖质量: 被覆盖样本 (Judge 判 Correct 但图核验 CONFLICT) 的金标准对错分布
3. Judge 层对照: baseline vs P6 的 Judge FPR/TPR (金标准口径)
4. 端到端净增益: 配对 2x2 表 + McNemar 精确检验 (纯 python, 二项精确)
5. 开销代理: _route 分布对照 (SKIP 率变化 = 纠错触发变化)

标签口径: 复用 refine/*/utils/evaluate 的 match_func (禁止自造匹配, 见实验计划 §6.2)。

用法:
    python scripts/p6_evaluate.py \
        --pkl_baseline test/results/refine/wikitq/qwen3.5-4b_p12/final_result.pkl \
        --pkl_p6 test/results/refine/wikitq/qwen3.5-4b_p12_p6/final_result.pkl \
        --task wikitq
"""

import argparse
import math
import os
import pickle
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)  # utils 包链式 import (validator → thought.*) 需要仓库根


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_matcher(task):
    """按任务注入对应 refine 路径, 返回逐样本对错判定函数。"""
    if task == "wikitq":
        sys.path.insert(0, os.path.join(_REPO_ROOT, "refine/TableQA"))
        from utils.evaluate import (
            tsv_unescape_list,
            to_value_list,
            wikitq_match_func,
        )

        target_values_map = {}
        tagged_dir = "thought/TableQA/data/wikitq/tagged_data"
        # evaluate.py 用 codecs.open 三参形式, 这里显式 encoding 保证独立可用
        for filename in os.listdir(tagged_dir):
            if filename.startswith("."):
                continue
            with open(os.path.join(tagged_dir, filename), "r", encoding="utf-8") as fin:
                header = fin.readline().rstrip("\n").split("\t")
                for line in fin:
                    stuff = dict(zip(header, line.rstrip("\n").split("\t")))
                    original_strings = tsv_unescape_list(stuff["targetValue"])
                    canon_strings = tsv_unescape_list(stuff["targetCanon"])
                    target_values_map[stuff["id"]] = to_value_list(original_strings, canon_strings)

        def judge_correct(sample):
            try:
                return 1 if wikitq_match_func(sample, target_values_map[sample["ids"]]) else 0
            except Exception:
                return -1  # 不可评估 (缺 ids/chain), 记 -1 排除

        return judge_correct
    else:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "refine/TableFV"))
        from utils.evaluate import tabfact_match_func

        def judge_correct(sample):
            try:
                return 1 if tabfact_match_func(sample) else 0
            except Exception:
                return -1

        return judge_correct


def mcnemar_exact(b, c):
    """McNemar 精确检验 (双侧): b/c 为两配置不一致格数。纯 python 实现。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 二项(n, 0.5) 双侧 p = 2 * P(X <= k), 上限截断为 1
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(p, 1.0)


def judge_confusion(samples, labels, key_contains="[Correct]"):
    """从 judge 字段与金标准标签构建混淆: pred_correct = key_contains in judge。"""
    tp = fp = fn = tn = 0  # 正类 = 金标准错误 (Judge 应判 Incorrect)
    for sample, gold in zip(samples, labels):
        if gold < 0 or sample is None:
            continue
        judge_text = str(sample.get("judge", ""))
        pred_correct = key_contains in judge_text
        if pred_correct and gold == 1:
            fp += 1  # 判对但实际错 (漏检)
        elif not pred_correct and gold == 0:
            fn += 1  # 判错但实际对 (误检)
        elif pred_correct and gold == 0:
            tp += 1
        else:
            tn += 1
    return {"tp_correct": tp, "fp_correct_missed_error": fp,
            "fn_incorrect_false_alarm": fn, "tn_incorrect": tn}


def route_stats(samples):
    stats = {}
    for sample in samples:
        if sample is None:
            continue
        route = sample.get("_route", "UNKNOWN")
        stats[route] = stats.get(route, 0) + 1
    return stats


def main():
    parser = argparse.ArgumentParser(description="P6 决策实验评估 (E7)")
    parser.add_argument("--pkl_baseline", required=True, help="P6 关闭的 refine final_result.pkl")
    parser.add_argument("--pkl_p6", required=True, help="P6 开启的 refine final_result.pkl")
    parser.add_argument("--task", choices=["wikitq", "tabfact"], default="wikitq")
    parser.add_argument("--output", default="", help="markdown 报告输出路径 (默认打印)")
    args = parser.parse_args()

    base_samples = load_pkl(args.pkl_baseline)
    p6_samples = load_pkl(args.pkl_p6)
    if len(base_samples) != len(p6_samples):
        print(f"⚠️ 样本数不一致: baseline={len(base_samples)}, p6={len(p6_samples)}; 按最小长度配对")
    n = min(len(base_samples), len(p6_samples))

    judge_correct = build_matcher(args.task)
    base_labels = [judge_correct(s) for s in base_samples[:n]]
    p6_labels = [judge_correct(s) for s in p6_samples[:n]]

    lines = []
    lines.append(f"# P6 决策实验评估 ({args.task})")
    lines.append(f"- baseline: `{args.pkl_baseline}`")
    lines.append(f"- p6:       `{args.pkl_p6}`")
    lines.append(f"- 配对样本数: {n}")
    lines.append("")

    # 1. P6 核验统计 (仅 p6 产物有 _p6_* 字段)
    verdict_stats = {}
    overridden = 0
    for s in p6_samples[:n]:
        if s is None:
            continue
        verdict_stats[s.get("_p6_verdict", "N/A")] = verdict_stats.get(s.get("_p6_verdict", "N/A"), 0) + 1
        if "_p6_original_judge" in s:
            overridden += 1
    lines.append("## 1. P6 核验统计")
    lines.append(f"- verdict 分布: {verdict_stats}")
    lines.append(f"- CONFLICT 覆盖次数 (Judge Correct 被推翻): {overridden}")
    lines.append("")

    # 2. 覆盖质量: 被覆盖样本金标准对错
    saved = 0
    wasted = 0
    for i, s in enumerate(p6_samples[:n]):
        if s is not None and "_p6_original_judge" in s and p6_labels[i] >= 0:
            if p6_labels[i] == 1:
                saved += 1  # 覆盖后最终答案正确 (覆盖有收益)
            else:
                wasted += 1  # 覆盖后最终仍错 (多跑了一次 Critic)
    lines.append("## 2. 覆盖质量")
    lines.append(f"- 覆盖后最终正确 (真被救回): {saved}")
    lines.append(f"- 覆盖后最终仍错 (额外开销): {wasted}")
    lines.append("")

    # 3. Judge 层混淆对照 (正类 = 金标准错误)
    lines.append("## 3. Judge 混淆对照 (正类 = 金标准错误)")
    for name, samples, labels in (("baseline", base_samples[:n], base_labels),
                                  ("p6", p6_samples[:n], p6_labels)):
        cm = judge_confusion(samples, labels)
        lines.append(f"- **{name}**: 判Correct且真对={cm['tp_correct']}, "
                     f"判Correct但实际错(FP)={cm['fp_correct_missed_error']}, "
                     f"判Incorrect但实际对(FN)={cm['fn_incorrect_false_alarm']}, "
                     f"判Incorrect且真错={cm['tn_incorrect']}")
    lines.append("")

    # 4. 端到端配对 2x2 + McNemar
    both = [(b, p) for b, p in zip(base_labels, p6_labels) if b >= 0 and p >= 0]
    n01 = sum(1 for b, p in both if b == 0 and p == 1)
    n10 = sum(1 for b, p in both if b == 1 and p == 0)
    n00 = sum(1 for b, p in both if b == 0 and p == 0)
    n11 = sum(1 for b, p in both if b == 1 and p == 1)
    total = len(both)
    acc_b = (n10 + n11) / total * 100 if total else 0.0
    acc_p6 = (n01 + n11) / total * 100 if total else 0.0
    p_val = mcnemar_exact(n10, n01)
    lines.append("## 4. 端到端净增益 (金标准口径)")
    lines.append(f"- baseline acc: {acc_b:.2f}%  |  p6 acc: {acc_p6:.2f}%  |  Δ = {acc_p6 - acc_b:+.2f}pp")
    lines.append(f"- 配对 2x2: 双对={n11}, 双错={n00}, 仅baseline对={n10}, 仅p6对={n01} (n={total})")
    lines.append(f"- McNemar 精确检验 p = {p_val:.4f}")
    lines.append("")

    # 5. 开销代理: route 分布
    lines.append("## 5. 开销代理 (_route 分布)")
    lines.append(f"- baseline: {route_stats(base_samples[:n])}")
    lines.append(f"- p6:       {route_stats(p6_samples[:n])}")

    report = "\n".join(lines)
    print(report)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n📄 报告已写入: {args.output}")


if __name__ == "__main__":
    main()
