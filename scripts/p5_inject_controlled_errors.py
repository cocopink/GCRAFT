"""P5 受控错误注入 (E6 GT 自建, 纯 CPU 零 LLM)

在真实**正确**轨迹上注入三类错误 (docs/experiment_plan_p1_p6.md §7.1):
- wrong_column: select_column 主候选替换为另一合法列
- wrong_row:    select_row 主候选行集合替换为随机不同子集
- wrong_aggregate: simple_query 主候选答案替换为最终表中另一单元格值

注入有效性筛选 (确定性): 重放对比注入前后行/列终态集合 (或答案字段) 是否变化,
答案不变的注入剔除并报告剔除率——标签精确性是 P5 定位评估的地基。

用法:
    python scripts/p5_inject_controlled_errors.py \
        --pkl test/results/thought/wikitq/qwen3.5-4b_p12/final_result.pkl \
        --task wikitq --per_type 100 --output results/p5/injected_wikitq_4b.pkl
"""

import argparse
import copy
import os
import pickle
import random
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p4_probing_lib import parse_op_params, parse_row_ids, trace_row_sets  # noqa: E402
from table_structure.value_similarity import normalize_value  # noqa: E402


def _table_of(sample):
    return (
        sample.get("atg_reranked_table")
        or sample.get("atgo_reranked_table")
        or sample.get("table_text")
        or sample.get("table")
    )


def _find_steps(sample, op_name):
    """返回 [(chain 下标, trace 1-based 步号)] 的 select 类操作列表。"""
    trace = trace_row_sets(sample)
    if trace is None:
        return [], None
    hits = []
    for idx, op in enumerate(sample.get("chain", [])):
        if op.get("operation_name") == op_name:
            for st in trace["steps"]:
                if st["op"] == op_name and st["step"] not in [h[1] for h in hits]:
                    hits.append((idx, st["step"]))
                    break
    return hits, trace


def _terminal_state(sample):
    trace = trace_row_sets(sample)
    if trace is None:
        return None
    pred = None
    try:
        pred = str(sample["chain"][-1]["parameter_and_conf"][0][0]).strip().lower()
    except (IndexError, KeyError, TypeError):
        pass
    return (frozenset(trace["rows_final"]), frozenset(trace["cols_final"]), pred)


def inject_wrong_column(sample, rng):
    """select_column 主候选 → 另一合法列 (当前表头中不在原候选内的列)。"""
    hits, trace = _find_steps(sample, "select_column")
    rng.shuffle(hits)
    table = _table_of(sample)
    headers = [str(h) for h in table[0]]
    for idx, step in hits:
        op = sample["chain"][idx]
        params = parse_op_params(op)
        if not isinstance(params, list):
            continue
        old_norm = {normalize_value(str(c)) for c in params}
        candidates = [h for h in headers if normalize_value(h) not in old_norm]
        if not candidates:
            continue
        new_col = rng.choice(candidates)
        new_candidate = str(sorted([str(c) for c in params[:1]] + [new_col]))
        original = copy.deepcopy(op["parameter_and_conf"])
        op["parameter_and_conf"][0] = (new_candidate, op["parameter_and_conf"][0][1])
        return {"type": "wrong_column", "step": step, "chain_idx": idx,
                "original": original, "injected": copy.deepcopy(op["parameter_and_conf"])}
    return None


def inject_wrong_row(sample, rng):
    """select_row 主候选 → 随机不同子集 (至少与原候选差一行)。"""
    hits, trace = _find_steps(sample, "select_row")
    rng.shuffle(hits)
    n_rows = len(_table_of(sample)) - 1
    for idx, step in hits:
        op = sample["chain"][idx]
        old_ids = parse_row_ids(op)
        if not old_ids:
            continue
        pool = [r for r in range(n_rows) if r + 1 not in old_ids]
        if not pool:
            continue
        k = max(1, min(len(old_ids), len(pool)))
        new_ids = rng.sample(pool, k)
        new_candidate = str(sorted(str(r + 1) for r in new_ids))
        original = copy.deepcopy(op["parameter_and_conf"])
        op["parameter_and_conf"][0] = (new_candidate, op["parameter_and_conf"][0][1])
        return {"type": "wrong_row", "step": step, "chain_idx": idx,
                "original": original, "injected": copy.deepcopy(op["parameter_and_conf"])}
    return None


def inject_wrong_aggregate(sample, rng):
    """simple_query 主候选答案 → 最终表中另一单元格值。"""
    table = _table_of(sample)
    chain = sample.get("chain", [])
    if not chain or chain[-1].get("operation_name") != "simple_query":
        return None
    try:
        old_answer = str(chain[-1]["parameter_and_conf"][0][0]).strip().lower()
    except (IndexError, KeyError, TypeError):
        return None
    cells = {str(c).strip().lower() for row in table[1:] for c in row}
    cells.discard(old_answer)
    cells.discard("")
    if not cells:
        return None
    new_answer = rng.choice(sorted(cells))
    original = copy.deepcopy(chain[-1]["parameter_and_conf"])
    chain[-1]["parameter_and_conf"][0] = (new_answer, chain[-1]["parameter_and_conf"][0][1])
    return {"type": "wrong_aggregate", "step": None, "chain_idx": len(chain) - 1,
            "original": original, "injected": copy.deepcopy(chain[-1]["parameter_and_conf"])}


INJECTORS = {
    "wrong_column": inject_wrong_column,
    "wrong_row": inject_wrong_row,
    "wrong_aggregate": inject_wrong_aggregate,
}


def main():
    parser = argparse.ArgumentParser(description="P5 受控错误注入")
    parser.add_argument("--pkl", required=True, help="thought final_result.pkl (含正确轨迹)")
    parser.add_argument("--task", choices=["wikitq", "tabfact"], required=True)
    parser.add_argument("--per_type", type=int, default=100, help="每类型目标注入数 (总计 >= 300)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, "refine/TableQA" if args.task == "wikitq" else "refine/TableFV")
    if args.task == "wikitq":
        from utils.evaluate import to_value_list, tsv_unescape_list, wikitq_match_func
        target_map = {}
        for filename in os.listdir("thought/TableQA/data/wikitq/tagged_data"):
            if filename.startswith("."):
                continue
            with open(os.path.join("thought/TableQA/data/wikitq/tagged_data", filename),
                      "r", encoding="utf-8") as fin:
                header = fin.readline().rstrip("\n").split("\t")
                for line in fin:
                    stuff = dict(zip(header, line.rstrip("\n").split("\t")))
                    target_map[stuff["id"]] = to_value_list(
                        tsv_unescape_list(stuff["targetValue"]),
                        tsv_unescape_list(stuff["targetCanon"]))
        def is_correct(sample):
            try:
                return bool(wikitq_match_func(sample, target_map[sample["ids"]]))
            except Exception:
                return False
    else:
        from utils.evaluate import tabfact_match_func

        def is_correct(sample):
            try:
                return bool(tabfact_match_func(sample))
            except Exception:
                return False

    with open(args.pkl, "rb") as f:
        samples = pickle.load(f)
    correct = [s for s in samples if s is not None and is_correct(s)]
    print(f"Loaded {len(samples)} samples, 正确轨迹 {len(correct)} 条")

    rng = random.Random(args.seed)
    rng.shuffle(correct)

    injected_set = []
    stats = {t: {"attempted": 0, "rejected": 0, "accepted": 0} for t in INJECTORS}
    # 逐样本尝试注入: 每样本只注入一种类型 (错误归因唯一)
    for sample in correct:
        if all(stats[t]["accepted"] >= args.per_type for t in INJECTORS):
            break
        types = [t for t in INJECTORS if stats[t]["accepted"] < args.per_type]
        rng.shuffle(types)
        for t in types:
            stats[t]["attempted"] += 1
            before = _terminal_state(sample)
            record = INJECTORS[t](sample, rng)
            if record is None:
                stats[t]["rejected"] += 1
                continue
            after = _terminal_state(sample)
            changed = before is not None and after is not None and before != after
            if not changed:
                # 回滚无效注入
                idx = record["chain_idx"]
                sample["chain"][idx]["parameter_and_conf"] = record["original"]
                stats[t]["rejected"] += 1
                continue
            stats[t]["accepted"] += 1
            injected_set.append({
                "sample": sample,
                "injection": {**record, "label_step": record["step"]},
            })
            break

    print("\n注入统计:")
    total_acc = 0
    for t, s in stats.items():
        rate = s["rejected"] / max(s["attempted"], 1) * 100
        print(f"  {t}: 尝试 {s['attempted']}, 剔除(答案不变) {s['rejected']} "
              f"({rate:.1f}%), 接受 {s['accepted']}")
        total_acc += s["accepted"]
    print(f"  总计: {total_acc} 条注入 (目标 {args.per_type * len(INJECTORS)})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"meta": {"pkl": args.pkl, "task": args.task, "seed": args.seed,
                              "stats": stats}, "injected": injected_set}, f)
    print(f"注入集已写入: {args.output}")


if __name__ == "__main__":
    main()
