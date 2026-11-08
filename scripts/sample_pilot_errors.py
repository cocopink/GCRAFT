"""Gamma×Axis pilot 错误样本抽样 (管线口径判定)

错误样本判定完全复用管线评估代码:
- QA: thought/TableQA/utils/evaluate.py 的 target_values_map 构建 + wikitq_match_func
- FV: thought/TableFV/utils/evaluate.py 的 tabfact_match_func

从 C1_raw thought 结果判定 gold 错误样本, seed=42 抽 N 条, 回取原始字段写出
子集 jsonl (供 preprocess_atgo.py 生成各 gamma 配置重排数据)。

用法:
    python scripts/sample_pilot_errors.py --dataset qa --size 200
    python scripts/sample_pilot_errors.py --dataset fv --size 200
"""

import json
import os
import pickle
import random
import sys

import fire

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(dataset: str = "qa", size: int = 200, seed: int = 42):
    random.seed(seed)
    if dataset == "qa":
        sys.path.insert(0, os.path.join(ROOT, "thought", "TableQA"))
        sys.path.insert(0, ROOT)
        from utils.evaluate import to_value_list, wikitq_match_func, tsv_unescape_list

        # 照抄 wikitq_match_func_for_samples 的 target map 构建 (evaluate.py:376-395)
        target_values_map = {}
        tagged_dir = os.path.join(ROOT, "thought/TableQA/data/wikitq/tagged_data")
        for filename in os.listdir(tagged_dir):
            if filename[0] == ".":
                continue
            with open(os.path.join(tagged_dir, filename), "r", encoding="utf8") as fin:
                header = fin.readline().rstrip("\n").split("\t")
                for line in fin:
                    stuff = dict(zip(header, line.rstrip("\n").split("\t")))
                    target_values_map[stuff["id"]] = to_value_list(
                        tsv_unescape_list(stuff["targetValue"]),
                        tsv_unescape_list(stuff["targetCanon"]))

        with open(os.path.join(ROOT, "results/graph_experiments/wikitq/C1_raw/thought/final_result.pkl"), "rb") as f:
            thought = pickle.load(f)

        def is_wrong(s):
            try:
                return not wikitq_match_func(s, target_values_map[s["ids"]], "top")
            except Exception:
                return False

        key_of = lambda s: s.get("ids")
        raw_jsonl = os.path.join(ROOT, "thought/TableQA/data/wikitq/test_lower.jsonl")
        out_dir = os.path.join(ROOT, "thought/TableQA/data/wikitq")
    elif dataset == "fv":
        sys.path.insert(0, os.path.join(ROOT, "thought", "TableFV"))
        sys.path.insert(0, ROOT)
        from utils.evaluate import tabfact_match_func

        with open(os.path.join(ROOT, "results/graph_experiments/tabfact/C1_raw/thought/final_result.pkl"), "rb") as f:
            thought = pickle.load(f)

        def is_wrong(s):
            try:
                return not tabfact_match_func(s)
            except Exception:
                return False

        key_of = lambda s: (s.get("statement"), s.get("table_id"))
        raw_jsonl = os.path.join(ROOT, "thought/TableFV/data/tabfact/test.jsonl")
        out_dir = os.path.join(ROOT, "thought/TableFV/data/tabfact")
    else:
        raise ValueError(dataset)

    wrong_thought = [s for s in thought if is_wrong(s)]
    picked_keys = random.sample([key_of(s) for s in wrong_thought], min(size, len(wrong_thought)))
    picked_set = set(picked_keys)

    by_key = {}
    with open(raw_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            by_key[key_of(s)] = s

    out, missing = [], 0
    for k in picked_keys:
        r = by_key.get(k)
        if r is None:
            missing += 1
            continue
        out.append(r)
    print(f"[{dataset}] gold 错误 {len(wrong_thought)}, 抽样 {len(picked_keys)}, 原始对齐缺失 {missing}, 写出 {len(out)}")

    out_path = os.path.join(out_dir, "gamma_pilot_raw.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    keys = [(s.get("ids") or "", s.get("statement") or "", s.get("table_id") or "") for s in out]
    with open(os.path.join(out_dir, "gamma_pilot_ids.json"), "w") as f:
        json.dump(keys, f)
    print(f"写出 {out_path} 与 gamma_pilot_ids.json")


if __name__ == "__main__":
    fire.Fire(main)
