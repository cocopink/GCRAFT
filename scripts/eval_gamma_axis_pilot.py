"""Gamma×Axis pilot 汇总 (管线口径)

每组统计: 修复数/修复率 (对照 gold 错误集, 即子集准确率) + ∪修正循环(v2) 并集修复率。
判定完全复用管线评估代码 (QA: wikitq_match_func; FV: tabfact_match_func)。
样本对齐: (ids, statement, table_id) 三元组键 (QA 的 table_id 恒为空串)。

用法:
    python scripts/eval_gamma_axis_pilot.py --out docs/gamma_axis_pilot_results.md
"""

import json
import os
import pickle
import sys

import fire

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMMAS = ["0", "0.1", "0.5", "0.7"]
AXES = ["row", "col", "rowcol"]


def tkey(s):
    return (s.get("ids") or "", s.get("statement") or "", s.get("table_id") or "")


def load_by_tk(p):
    with open(p, "rb") as f:
        data = pickle.load(f)
    return {tkey(s): s for s in data if isinstance(s, dict)}


def main(out: str = "docs/gamma_axis_pilot_results.md"):
    lines = ["# Gamma×Axis Pilot 重推矩阵结果", "",
             "- 判定: 管线评估函数 (QA `wikitq_match_func` / FV `tabfact_match_func`), 无自定义规则",
             "- 子集: 两侧各抽 200 gold 错误样本 (seed=42, 源 C1_raw qwen3.5-4b raw+Clarifier)",
             "- 重推: 重排表上 thought 式动态链从零推理 (raw+Clarifier 口径)",
             "- 修正循环对照: v2 `results_v2/new_{qa,fv}/4b` (原表)",
             "- 并集 = 重推修复集 ∪ 修正循环修复集 (同一错误子集上)", "",
             "| 数据集 | 轴 | gamma | 重推修复 | 重推修复率 | ∪修正循环 | 并集修复率 | 互补增量 |",
             "|---|---|---|---|---|---|---|---|"]

    for ds, res, data_dir, v2_rel, c1_rel in [
        ("QA (WikiTQ)", "wikitq", "wikitq", "results_v2/new_qa/4b/final_result.pkl",
         "results/graph_experiments/wikitq/C1_raw/thought/final_result.pkl"),
        ("FV (TabFact)", "tabfact", "tabfact", "results_v2/new_fv/4b/final_result.pkl",
         "results/graph_experiments/tabfact/C1_raw/thought/final_result.pkl"),
    ]:
        if ds.startswith("QA"):
            sys.path.insert(0, os.path.join(ROOT, "thought", "TableQA"))
            from utils.evaluate import to_value_list, wikitq_match_func, tsv_unescape_list

            tvm = {}
            tagged_dir = os.path.join(ROOT, "thought/TableQA/data/wikitq/tagged_data")
            for filename in os.listdir(tagged_dir):
                if filename[0] == ".":
                    continue
                with open(os.path.join(tagged_dir, filename), "r", encoding="utf8") as fin:
                    header = fin.readline().rstrip("\n").split("\t")
                    for line in fin:
                        stuff = dict(zip(header, line.rstrip("\n").split("\t")))
                        tvm[stuff["id"]] = to_value_list(
                            tsv_unescape_list(stuff["targetValue"]),
                            tsv_unescape_list(stuff["targetCanon"]))
            ok = lambda s: wikitq_match_func(s, tvm[s["ids"]], "top")
        else:
            # 清除 QA 侧已缓存的 utils 模块, 避免 FV 的 utils.evaluate 撞 QA 缓存
            for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
                del sys.modules[k]
            sys.path.insert(0, os.path.join(ROOT, "thought", "TableFV"))
            from utils.evaluate import tabfact_match_func

            ok = lambda s: tabfact_match_func(s)

        raw = load_by_tk(os.path.join(ROOT, c1_rel))
        v2 = load_by_tk(os.path.join(ROOT, v2_rel))
        with open(os.path.join(ROOT, f"thought/Table{'QA' if ds.startswith('QA') else 'FV'}/data/{data_dir}/gamma_pilot_ids.json")) as f:
            pilot_keys = set(tuple(k) for k in json.load(f))

        # 子集: 抽样清单内、管线口径下的 gold 错误
        wrong_tk = [tk for tk in raw
                    if (tk[0], tk[1], tk[2]) in pilot_keys and ok(raw[tk]) is False]
        tree_fixed = {tk for tk in wrong_tk if tk in v2 and ok(v2[tk]) is True}
        n_tree = len([tk for tk in wrong_tk if tk in v2])
        print(f"[{ds}] gold 错误 {len(wrong_tk)} (v2 覆盖 {n_tree}), "
              f"修正循环修复 {len(tree_fixed)} ({100*len(tree_fixed)/max(n_tree,1):.1f}%)")

        for g in GAMMAS:
            for m in AXES:
                p = os.path.join(ROOT, f"test/results/thought/{res}/qwen3.5-4b_gammapilot_{m}_g{g}/final_result.pkl")
                if not os.path.exists(p):
                    lines.append(f"| {ds} | {m} | {g} | (未完成) | — | — | — | — |")
                    continue
                by = load_by_tk(p)
                jud = [tk for tk in wrong_tk if tk in by and ok(by[tk]) is not None]
                fixed = {tk for tk in jud if ok(by[tk]) is True}
                union = fixed | tree_fixed
                lines.append(
                    f"| {ds} | {m} | {g} | {len(fixed)}/{len(jud)} | "
                    f"{100*len(fixed)/max(len(jud),1):.1f}% | {len(union)}/{n_tree} | "
                    f"{100*len(union)/max(n_tree,1):.1f}% | "
                    f"{100*(len(union)-len(tree_fixed))/max(n_tree,1):+.1f}pp |")
                print(lines[-1])
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    fire.Fire(main)
