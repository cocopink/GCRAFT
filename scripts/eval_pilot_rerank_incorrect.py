"""评估 judge-incorrect 重推 pilot: 修复率 + gamma 增益 + 位置词分组

对比维度:
- 新跑的 gamma=0.1 pilot (qwen3.5-4b_pilot_{axis}_g01) vs 已有 gamma=0 消融数据
  (qwen3.5-4b_axis_{axis}) 在同一 pilot 子集上的修复率
- 位置词 / 非位置词子集分组修复率
- 基准: v2 现有 refine 修正循环修复率

用法:
    python scripts/eval_pilot_rerank_incorrect.py --axes row rowcol
"""

import json
import os
import pickle
import re
import sys

import fire

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "thought", "TableQA"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.evaluate import to_value_list, check_denotation  # noqa: E402

POS_WORDS = re.compile(
    r"\b(first|last|top|bottom|next|previous|earliest|latest|most recent|second|"
    r"third|fourth|fifth|final|initial|beginning|above|below|order|before|after)\b",
    re.I,
)

C1_RAW_PKL = "results/graph_experiments/wikitq/C1_raw/thought/final_result.pkl"
AXIS_TPL = "test/results/thought/wikitq/qwen3.5-4b_axis_{axis}/final_result.pkl"
PILOT_G01_TPL = "test/results/thought/wikitq/qwen3.5-4b_pilot_{axis}_g01/final_result.pkl"
PILOT_IDS = "thought/TableQA/data/wikitq/pilot_incorrect_raw_ids.json"
TAGGED_DIR = "thought/TableQA/data/wikitq/tagged_data"


def _unescape(s):
    return s.replace("\\n", "\n").replace("\\t", "\t")


def build_target_values_map(tagged_dir=TAGGED_DIR):
    import glob

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


def load_by_ids(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {s["ids"]: s for s in data if isinstance(s, dict)}


def main(axes="row rowcol", c1_raw_pkl=C1_RAW_PKL):
    axes = axes.split()
    tvm = build_target_values_map()
    with open(c1_raw_pkl, "rb") as f:
        raw_by_ids = {s["ids"]: s for s in pickle.load(f) if isinstance(s, dict)}
    with open(PILOT_IDS) as f:
        pilot_ids = json.load(f)

    # pilot 子集全部是 C1_raw 错误样本
    wrong = [i for i in pilot_ids if not is_correct(raw_by_ids[i], tvm)]
    pos_wrong = [i for i in wrong if POS_WORDS.search(raw_by_ids[i].get("statement", ""))]
    print(f"pilot 子集 {len(pilot_ids)} | 其中 raw 错误 {len(wrong)} "
          f"(位置词 {len(pos_wrong)} / 非位置词 {len(wrong) - len(pos_wrong)})")
    print()

    header = f"{'配置':<16}{'修复':>5}{'修复率':>8}  {'位置词':>12}  {'非位置词':>12}"
    print(header)
    print("-" * len(header))
    for axis in axes:
        for tag, tpl in [("g0(axis消融)", AXIS_TPL), ("g0.1(pilot)", PILOT_G01_TPL)]:
            by = load_by_ids(tpl.format(axis=axis))
            if by is None:
                print(f"{axis} {tag:<12}    (结果不存在, 跳过)")
                continue
            sub = [i for i in wrong if i in by]
            fixed = [i for i in sub if is_correct(by[i], tvm)]
            pos_fix = [i for i in fixed if POS_WORDS.search(raw_by_ids[i].get("statement", ""))]
            n_pos = len([i for i in sub if POS_WORDS.search(raw_by_ids[i].get("statement", ""))])
            print(f"{axis} {tag:<10} {len(fixed):>4}/{len(sub):<3} {100*len(fixed)/max(len(sub),1):>6.1f}%  "
                  f"{len(pos_fix):>3}/{n_pos:<3} ({100*len(pos_fix)/max(n_pos,1):.1f}%)  "
                  f"{len(fixed)-len(pos_fix):>3}/{len(sub)-n_pos:<3} ({100*(len(fixed)-len(pos_fix))/max(len(sub)-n_pos,1):.1f}%)")
    print()
    print("基准: v2 现有 refine 修正循环全量修复率 33.4% (543/1624)")


if __name__ == "__main__":
    fire.Fire(main)
