"""T-B（修正回退）事后模拟统计——零重跑, 纯 CPU

规则（与运行时 T-B 逐样本等价）：
  refine 最后一轮 judge == [Correct] → 用 refine 答案（现状）
  否则（全轮判错/判错退出）           → 用 thought 原答案（回退）

对照三口径：
  模拟 acc (T-B) / 现状 acc (refine as-is) / 原版第一次基线（手工传入）

用法：
  python3 scripts/simulate_tb_fallback.py --refine <refine.pkl> --thought <thought.pkl> \
      --task wikitq|tabfact --baseline 0.7884 --label "4B new QA"
"""

import argparse
import pickle
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "refine/TableQA")
sys.path.insert(0, "critic/TableQA")
sys.path.insert(0, "refine/TableFV")
sys.path.insert(0, "critic/TableFV")


def load_answer_thought(sample):
    """thought pkl 样本的最终答案（chain[-1] 主候选首项）。"""
    try:
        return str(sample["chain"][-1]["parameter_and_conf"][0][0]).strip().lower()
    except Exception:
        return None


def build_answered_clone(refine_sample, thought_answer):
    """refine 样本拷贝, 将 chain[-1] 主候选首项替换为 thought 答案（模拟回退后的最终样本）。"""
    import copy

    s = copy.deepcopy(refine_sample)
    try:
        old = s["chain"][-1]["parameter_and_conf"][0][0]
        s["chain"][-1]["parameter_and_conf"][0] = (thought_answer, s["chain"][-1]["parameter_and_conf"][0][1])
        return s, str(old)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", required=True, help="refine final_result.pkl")
    ap.add_argument("--thought", required=True, help="对应 thought final_result.pkl（原答案来源）")
    ap.add_argument("--task", choices=["wikitq", "tabfact"], required=True)
    ap.add_argument("--baseline", type=float, required=True, help="原版第一次基线 acc")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    from critic.TableFV.tools.read_pkl import read_pkl
    from refine.TableFV.utils.evaluate import tabfact_match_func
    from refine.TableQA.utils.evaluate import to_value_list, tsv_unescape_list, wikitq_match_func

    gold = {}
    if args.task == "wikitq":
        tdir = "thought/TableQA/data/wikitq/tagged_data"
        for fn in os.listdir(tdir):
            if fn[0] == ".":
                continue
            with open(os.path.join(tdir, fn)) as fin:
                header = fin.readline().rstrip("\n").split("\t")
                for line in fin:
                    row = dict(zip(header, line.rstrip("\n").split("\t")))
                    gold[row["id"]] = to_value_list(
                        tsv_unescape_list(row["targetValue"]),
                        tsv_unescape_list(row["targetCanon"]),
                    )

    def judge_ok(sample):
        return "[Correct]" in str(sample.get("judge", ""))

    def match(sample):
        try:
            if args.task == "wikitq":
                return bool(wikitq_match_func(sample, gold[sample["ids"]]))
            return bool(tabfact_match_func(sample))
        except Exception:
            return False

    with open(args.refine, "rb") as f:
        refine = pickle.load(f)
    with open(args.thought, "rb") as f:
        thought = pickle.load(f)

    n = min(len(refine), len(thought))
    as_is = tb = 0
    n_fallback = 0
    for i in range(n):
        r, t = refine[i], thought[i]
        if r is None:
            continue
        judge_correct = judge_ok(r)
        # 现状: refine 答案原样判定
        as_is += 1 if match(r) else 0
        # T-B: judge 判错 → 回退 thought 答案; 判对 → 用 refine 答案
        if judge_correct:
            s_use = r
        else:
            ta = load_answer_thought(t)
            s_use, _ = build_answered_clone(r, ta)
            n_fallback += 1 if s_use is not None else 0
            if s_use is None:
                s_use = r
        tb += 1 if match(s_use) else 0

    total = n
    print(f"[{args.label}] n={total}")
    print(f"  现状 acc (refine as-is): {as_is/total:.4f} ({as_is}/{total})")
    print(f"  T-B 模拟 acc (判错回退 thought 答案): {tb/total:.4f} ({tb}/{total}), 回退样本数={n_fallback}")
    print(f"  原版第一次基线: {args.baseline:.4f}")
    print(f"  T-B vs 原版: {100*(tb/total - args.baseline):+.2f}pp | T-B vs 现状: {100*(tb/total - as_is/total):+.2f}pp")


if __name__ == "__main__":
    import os

    main()
