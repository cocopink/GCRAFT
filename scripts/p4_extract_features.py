"""P4 特征表构建 (E5 前置, 纯 CPU 零 LLM)

读 thought 阶段 final_result.pkl, 逐样本提取图特征 (F1/F2 主特征 + F4/F5/F6 探索)
与非图基线 (B1 置信 / B2 链长), 金标准标签复用 refine/*/utils/evaluate 的 match 口径
(禁止自造匹配, 见 docs/experiment_plan_p1_p6.md §6.2)。

输出:
1. 特征表 pkl (list[dict] + meta)
2. 金标准标注空表 CSV (--gold_size, 按对/错分层随机抽样, 供人工错误类型标注)

用法:
    python scripts/p4_extract_features.py \
        --pkl test/results/thought/wikitq/qwen3.5-4b_p12/final_result.pkl \
        --task wikitq --output results/p4/features_wikitq_4b.pkl \
        --gold_csv results/p4/gold_wikitq_4b.csv --gold_size 200
"""

import argparse
import csv
import os
import pickle
import random
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)  # table_structure / refine / thought 跨模块 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p4_probing_lib import (  # noqa: E402
    b1_confidence,
    feature_f1,
    feature_f2,
    feature_f4,
    feature_f5,
    feature_f6,
    trace_row_sets,
)


def build_labels(task):
    """按任务返回逐样本金标准判定 (1=错=正类, 0=对, -1=不可评估)。"""
    if task == "wikitq":
        sys.path.insert(0, "refine/TableQA")
        from utils.evaluate import to_value_list, tsv_unescape_list, wikitq_match_func

        target_values_map = {}
        tagged_dir = "thought/TableQA/data/wikitq/tagged_data"
        for filename in os.listdir(tagged_dir):
            if filename.startswith("."):
                continue
            with open(os.path.join(tagged_dir, filename), "r", encoding="utf-8") as fin:
                header = fin.readline().rstrip("\n").split("\t")
                for line in fin:
                    stuff = dict(zip(header, line.rstrip("\n").split("\t")))
                    target_values_map[stuff["id"]] = to_value_list(
                        tsv_unescape_list(stuff["targetValue"]),
                        tsv_unescape_list(stuff["targetCanon"]),
                    )

        # wikitq_match_func 返回 True=对; 统一口径 1=错(正类), 0=对, -1=不可评估
        def label_fn(sample):
            try:
                ok = wikitq_match_func(sample, target_values_map[sample["ids"]])
                return 0 if ok else 1
            except Exception:
                return -1

        return label_fn
    else:
        sys.path.insert(0, "refine/TableFV")
        from utils.evaluate import tabfact_match_func

        def label_fn(sample):
            try:
                ok = tabfact_match_func(sample)
                return 0 if ok else 1
            except Exception:
                return -1

        return label_fn


def extract_sample_features(sample, label_fn):
    """单样本特征 dict; 样本为 None 或缺图数据时返回 None (计数排除)。"""
    if sample is None:
        return None
    trace = trace_row_sets(sample)
    if trace is None:
        return None
    label = label_fn(sample)
    chain = sample.get("chain", [])
    op_types = {}
    for op in chain:
        name = op.get("operation_name", "?")
        op_types[name] = op_types.get(name, 0) + 1

    f5_row, f5_col = feature_f5(trace)
    f6 = feature_f6(sample, trace)
    return {
        "id": sample.get("id"),
        "ids": sample.get("ids"),
        "label": label,
        "chain_len": len(chain),
        "op_types": op_types,
        "f1_col_gap": feature_f1(trace),
        "f2_row_loss": feature_f2(trace),
        "f4_extreme_removal": feature_f4(trace),
        "f5_row_evidence": f5_row,
        "f5_col_evidence": f5_col,
        "f6_status": f6["status"],
        "f6_row_score": f6["row_score"],
        "f6_col_score": f6["col_score"],
        "f6_col_rank": f6["col_rank"],
        "b1_conf": b1_confidence(trace),
        "unmatched_cols": len(trace["unmatched_cols"]),
    }


def main():
    parser = argparse.ArgumentParser(description="P4 特征表构建")
    parser.add_argument("--pkl", required=True, help="thought final_result.pkl")
    parser.add_argument("--task", choices=["wikitq", "tabfact"], required=True)
    parser.add_argument("--output", required=True, help="特征表 pkl 输出路径")
    parser.add_argument("--gold_csv", default="", help="金标准标注空表 CSV 输出路径")
    parser.add_argument("--gold_size", type=int, default=200, help="金标准抽样规模")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples from {args.pkl}")

    label_fn = build_labels(args.task)
    features = []
    excluded = 0
    for sample in samples:
        feat = extract_sample_features(sample, label_fn)
        if feat is None:
            excluded += 1
            continue
        features.append(feat)

    n_err = sum(1 for x in features if x["label"] == 1)
    n_ok = sum(1 for x in features if x["label"] == 0)
    n_skip = sum(1 for x in features if x["label"] == -1)
    f6_calc = sum(1 for x in features if x["f6_status"] != "not_in_table")
    print(f"特征样本: {len(features)} (排除 {excluded}); 错误率={n_err}/{n_err+n_ok}"
          f" ({n_err/max(n_err+n_ok,1)*100:.2f}%, 不可评估 {n_skip})")
    print(f"F6 答案可追溯率: {f6_calc}/{len(features)}")
    if features:
        um = sum(x["unmatched_cols"] for x in features)
        total_cols = sum(sum(v for k, v in x["op_types"].items()
                             if k in ("select_column", "sort_column", "group_column"))
                         for x in features)
        print(f"F1 列对齐失败率: {um}/{total_cols}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"meta": {"pkl": args.pkl, "task": args.task,
                              "n_total": len(samples), "n_features": len(features),
                              "n_error": n_err}, "features": features}, f)
    print(f"特征表已写入: {args.output}")

    # 金标准标注空表: 按对/错分层随机抽样
    if args.gold_csv:
        rng = random.Random(args.seed)
        err_pool = [x for x in features if x["label"] == 1]
        ok_pool = [x for x in features if x["label"] == 0]
        n_half = args.gold_size // 2
        picked = rng.sample(err_pool, min(n_half, len(err_pool))) + \
                 rng.sample(ok_pool, min(args.gold_size - n_half, len(ok_pool)))
        rng.shuffle(picked)
        os.makedirs(os.path.dirname(args.gold_csv) or ".", exist_ok=True)
        with open(args.gold_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "ids", "label", "chain_len",
                             "f1_col_gap", "f2_row_loss",
                             "error_type", "wrong_step", "annotator", "notes"])
            for x in picked:
                writer.writerow([x["id"], x["ids"], x["label"], x["chain_len"],
                                 f"{x['f1_col_gap']:.4f}", f"{x['f2_row_loss']:.4f}",
                                 "", "", "", ""])
        print(f"金标准标注空表 ({len(picked)} 条, 对/错分层): {args.gold_csv}")


if __name__ == "__main__":
    main()
