"""构建 judge-incorrect 重推 pilot 子集数据集

从 C1_raw thought 结果中抽取错误样本（judge-incorrect 集合的近似）,
按 ids 从原始数据 jsonl 取回完整样本, 供 preprocess_atgo.py 在子集上做
gamma=0.1 重排, 或直接作为 thought 管线重推的输入。

用法:
    python scripts/build_pilot_rerank_incorrect.py \
        --thought_pkl results/graph_experiments/wikitq/C1_raw/thought/final_result.pkl \
        --raw_jsonl thought/TableQA/data/wikitq/test_lower.jsonl \
        --output thought/TableQA/data/wikitq/pilot_incorrect_raw.jsonl \
        --size 300 --seed 42
"""

import json
import pickle
import random
import sys
import os

import fire

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "thought", "TableQA"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.evaluate import to_value_list, check_denotation  # noqa: E402


def build_target_values_map(tagged_dir):
    """构建 ids -> targetCanon 金标映射 (与 acc.txt 评估口径一致)。"""
    import glob

    def _unescape(s):
        return s.replace("\\n", "\n").replace("\\t", "\t")

    tvm = {}
    for fn in glob.glob(os.path.join(tagged_dir, "*.tagged")):
        with open(fn, "r", encoding="utf8") as fin:
            header = fin.readline().rstrip("\n").split("\t")
            for line in fin:
                stuff = dict(zip(header, line.rstrip("\n").split("\t")))
                tvm[stuff["id"]] = [_unescape(x) for x in stuff["targetCanon"].split(",")]
    return tvm


def is_correct(sample, tvm):
    iid = sample.get("ids") if isinstance(sample.get("ids"), str) else None
    tv = tvm.get(iid) or [a.lower() for a in (sample.get("answer") or [])]
    res = sample["chain"][-1]["parameter_and_conf"][0][0]
    pred = res.lower().split("|")
    return check_denotation(to_value_list(tv), to_value_list(pred))


def main(
    thought_pkl: str = "results/graph_experiments/wikitq/C1_raw/thought/final_result.pkl",
    raw_jsonl: str = "thought/TableQA/data/wikitq/test_lower.jsonl",
    output: str = "thought/TableQA/data/wikitq/pilot_incorrect_raw.jsonl",
    size: int = 300,
    seed: int = 42,
):
    with open(thought_pkl, "rb") as f:
        thought = pickle.load(f)

    tvm = build_target_values_map("thought/TableQA/data/wikitq/tagged_data")
    wrong_ids = [s["ids"] for s in thought if not is_correct(s, tvm)]
    print(f"错误样本: {len(wrong_ids)}")

    random.seed(seed)
    pilot_ids = set(random.sample(wrong_ids, min(size, len(wrong_ids))))
    print(f"抽样 {len(pilot_ids)} 个 (seed={seed})")

    raw_by_ids = {}
    with open(raw_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            iid = s.get("ids")
            if iid in pilot_ids:
                raw_by_ids[iid] = s
    missing = pilot_ids - set(raw_by_ids)
    if missing:
        print(f"⚠️ {len(missing)} 个 ids 未在原始 jsonl 找到, 跳过")

    ordered = [raw_by_ids[i] for i in sorted(raw_by_ids)]
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for s in ordered:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"写出 {len(ordered)} 个样本 -> {output}")

    sidecar = os.path.splitext(output)[0] + "_ids.json"
    with open(sidecar, "w") as f:
        json.dump(sorted(raw_by_ids), f)
    print(f"ids 清单 -> {sidecar}")


if __name__ == "__main__":
    fire.Fire(main)
