"""E2 六个预注册边际对比的样本级配对分析 (experiment_plan_p1_p6.md §5.3)

每对配置: 2×2 配对表 → McNemar 精确检验 + 配对差值 bootstrap 95% CI。
统计口径 = 修复后的 match_func (与 cal_acc 一致), 按样本索引对齐。

用法:
    python scripts/analyze_e2_pairwise.py                      # rep1 (当前产物)
    python scripts/analyze_e2_pairwise.py --reps 2 3           # 指定遍 (预留多遍合并)
    python scripts/analyze_e2_pairwise.py --out results/graph_experiments/e2_pairwise_rep1.md
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")

from critic.TableFV.tools.read_pkl import read_pkl
from refine.TableFV.utils.evaluate import tabfact_match_func
from refine.TableQA.utils.evaluate import (
    to_value_list,
    tsv_unescape_list,
    wikitq_match_func,
)

ROOT = "results/graph_experiments"
CONFIGS = ["C1_raw", "C2_atgo", "C3_p1", "C4_p2", "C5_p1p2", "C6_p1p2p3"]
# 预注册主对比族 (§3): 8 个 = E2 六边际 + E7 两对照; 本脚本覆盖 E2 六个
PAIRS = [
    ("C2_atgo", "C1_raw", "ATGO 本体"),
    ("C3_p1", "C2_atgo", "P1 边际"),
    ("C4_p2", "C2_atgo", "P2 边际"),
    ("C5_p1p2", "C4_p2", "P1 边际 (P2 在场)"),
    ("C5_p1p2", "C3_p1", "P2 边际 (P1 在场)"),
    ("C6_p1p2p3", "C5_p1p2", "P3 边际"),
]


def wikitq_gold_map(tagged_dir="thought/TableQA/data/wikitq/tagged_data"):
    gold = {}
    for fn in os.listdir(tagged_dir):
        if fn[0] == ".":
            continue
        with open(os.path.join(tagged_dir, fn), "r", encoding="utf8") as fin:
            header = fin.readline().rstrip("\n").split("\t")
            for line in fin:
                row = dict(zip(header, line.rstrip("\n").split("\t")))
                gold[row["id"]] = to_value_list(
                    tsv_unescape_list(row["targetValue"]),
                    tsv_unescape_list(row["targetCanon"]),
                )
    return gold


def correct_flags(dataset, samples, gold_map=None):
    flags = []
    for s in samples:
        try:
            if dataset == "wikitq":
                ok = wikitq_match_func(s, gold_map[s["ids"]])
            else:
                ok = tabfact_match_func(s)
            flags.append(1 if ok else 0)
        except Exception:
            flags.append(0)
    return np.array(flags)


def mcnemar_exact_p(b, c):
    """b: a 对 b 错; c: a 错 b 对。双侧二项精确检验 (无 scipy 依赖)。"""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, tail * 2)  # 双侧


def bootstrap_ci(d0, d1, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(d0)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = d0[idx].mean() - d1[idx].mean()
    return np.percentile(diffs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=[1])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gold = wikitq_gold_map()
    lines = ["# E2 六个预注册边际对比 (配对分析)", ""]

    for ds in ["tabfact", "wikitq"]:
        flags = {}
        for cfg in CONFIGS:
            f = os.path.join(ROOT, ds, cfg, "refine", "final_result.pkl")
            with open(f, "rb") as fh:
                samples = pickle.load(fh)
            flags[cfg] = correct_flags(ds, samples, gold)
        lines.append(f"## {ds} (n={len(flags['C1_raw'])})")
        lines.append("")
        lines.append("| 对比 | Δacc (pp) | 95% CI | a对b错 | a错b对 | McNemar p |")
        lines.append("|------|----------|--------|-------|-------|-----------|")
        for a, b, label in PAIRS:
            fa, fb = flags[a], flags[b]
            n = min(len(fa), len(fb))
            fa, fb = fa[:n], fb[:n]
            delta = (fa.mean() - fb.mean()) * 100
            lo, hi = bootstrap_ci(fa, fb)
            lo, hi = lo * 100, hi * 100
            both_a = int(((fa == 1) & (fb == 0)).sum())
            both_b = int(((fa == 0) & (fb == 1)).sum())
            p = mcnemar_exact_p(both_a, both_b)
            lines.append(
                f"| {a} vs {b} ({label}) | {delta:+.2f} | [{lo:+.2f}, {hi:+.2f}] "
                f"| {both_a} | {both_b} | {p:.4f} |"
            )
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
