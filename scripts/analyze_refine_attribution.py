# -*- coding: utf-8 -*-
"""Refine 负收益归因分析（Anchor 训练前置调研）。

从现有 Thought / Refine final_result.pkl 统计：
  1. Thought→Refine 四格表：Correct→Correct / Correct→Wrong / Wrong→Correct / Wrong→Wrong
  2. 每样本 Refine 内部轨迹归因：错误检测（Judge）、错误定位（Critic conclusion Step 解析）、
     回滚位置（incorrect_step vs max_step）、重新推理结果（PLACEHOLDER / 答案未变 / 仍错）

注意：本机 cgroup 限单进程 2GB 内存，因此逐组合加载、处理完立即释放。

用法：
  python scripts/analyze_refine_attribution.py                # 全部 4 模型 x 2 数据集
  python scripts/analyze_refine_attribution.py --datasets wikitq --models 2b
"""
import argparse
import collections
import gc
import json
import pickle
import re
import sys

sys.path.insert(0, ".")

from refine.TableQA.utils.evaluate import wikitq_match_func  # noqa: E402
from refine.TableFV.utils.evaluate import tabfact_match_func  # noqa: E402

MODELS = ["0.8b", "2b", "4b", "9b"]
DATASETS = ["wikitq", "tabfact"]
_TAGGED_QA = "thought/TableQA/data/wikitq/tagged_data"
BASE = "test/results"  # 结果根目录（32k 批次用 test/results_len32k）


def _load_wikitq_gold_map():
    """读 tagged_data 构建 ids → target_values（原评测器用 codecs.open，此处等价）。"""
    import os
    from refine.TableQA.utils.evaluate import to_value_list, tsv_unescape_list
    target_values_map = {}
    for filename in os.listdir(_TAGGED_QA):
        if filename[0] == ".":
            continue
        with open(os.path.join(_TAGGED_QA, filename), "r", encoding="utf8") as fin:
            header = fin.readline().rstrip("\n").split("\t")
            for line in fin:
                stuff = dict(zip(header, line.rstrip("\n").split("\t")))
                target_values_map[stuff["id"]] = to_value_list(
                    tsv_unescape_list(stuff["targetValue"]),
                    tsv_unescape_list(stuff["targetCanon"]),
                )
    return target_values_map


_WIKITQ_GOLD = None  # 惰性单例


def sample_correct(ds, sample):
    """单样本正确性（复用各数据集官方 match 口径，top 策略，与 cal_acc.py 一致）。失败返回 None。"""
    global _WIKITQ_GOLD
    try:
        if ds == "wikitq":
            if _WIKITQ_GOLD is None:
                _WIKITQ_GOLD = _load_wikitq_gold_map()
            return bool(wikitq_match_func(sample, _WIKITQ_GOLD[sample["ids"]]))
        return bool(tabfact_match_func(sample))
    except Exception:
        return None


def pred_answer(sample):
    try:
        return str(sample["chain"][-1]["parameter_and_conf"][0][0]).lower()
    except Exception:
        return "<EXTRACT-FAIL>"


def parse_conclusion(conc):
    """解析 Critic conclusion → (located_step or None, is_placeholder, says_correct)。"""
    conc = (conc or "").strip()
    if "PLACEHOLDER" in conc:
        return None, True, False
    m = re.search(r"\[Incorrect\] Step (\d+)", conc)
    if m:
        return int(m.group(1)), False, False
    if conc.startswith("[Correct]"):
        return None, False, True
    return None, False, False  # 格式坏：既非 PLACEHOLDER 也解析不出 Step


def load_pair(ds, model):
    t = pickle.load(open(f"{BASE}/thought/{ds}/qwen3.5-{model}_p12/final_result.pkl", "rb"))
    r = pickle.load(open(f"{BASE}/refine/{ds}/qwen3.5-{model}_p12/final_result.pkl", "rb"))
    assert len(t) == len(r), f"{ds}/{model}: thought={len(t)} refine={len(r)}"
    for i, (a, b) in enumerate(zip(t, r)):
        assert a["id"] == b["id"], f"{ds}/{model} order mismatch at {i}"
    return t, r


def analyze_pair(ds, model):
    """单组合一次加载、一次遍历，产出全部统计 + 明细 jsonl。"""
    t, r = load_pair(ds, model)
    st = {
        "n": len(r),
        "quad": collections.Counter(),
        "detect_cm": collections.Counter(),
        "judge_first_dist": collections.Counter(),
        "path_c2w": collections.Counter(),
        "path_w2c": collections.Counter(),
        "path_w2w": collections.Counter(),
        "rollback": collections.Counter(),
        "located_step_hist": collections.Counter(),
        "ph_cnt": collections.Counter(),
        "eval_fail": 0,
        "critic_exec": 0,
    }
    detail_path = f"{BASE}/refine/{ds}/qwen3.5-{model}_p12/attribution_detail.jsonl"
    detail_f = open(detail_path, "w")

    for ts, rs in zip(t, r):
        tc, rc = sample_correct(ds, ts), sample_correct(ds, rs)
        if tc is None or rc is None:
            st["eval_fail"] += 1
            st["quad"][("eval-fail", "eval-fail")] += 1
        else:
            st["quad"][("Correct" if tc else "Wrong", "Correct" if rc else "Wrong")] += 1

        # ---- 检测层：judge_list[0] 是对 thought 原链的判定 ----
        jl = rs.get("judge_list") or []
        j0 = (jl[0] if jl else rs.get("judge") or "").strip()
        j0_ok = j0.startswith("[Correct]")
        j0_ph = "PLACEHOLDER" in j0
        if j0_ph:
            st["judge_first_dist"]["PLACEHOLDER"] += 1
        else:
            st["judge_first_dist"][j0.split("]")[0] + "]" if "]" in j0 else (j0[:20] or "<empty>")] += 1
        if not j0_ph and tc is not None:
            # 约定：gold=Correct → judge Correct=TN / Incorrect=FP（误检）
            #       gold=Wrong   → judge Correct=FN（漏检）/ Incorrect=TP（真检出）
            det = ("TN" if j0_ok else "FP") if tc else ("FN" if j0_ok else "TP")
            st["detect_cm"][("Correct" if tc else "Wrong", det)] += 1

        # ---- 定位层：critic 是否执行及其结论解析 ----
        crit_ran = len(rs.get("conclusion_list") or []) > 0
        step, is_ph, _says_correct = parse_conclusion(rs.get("conclusion")) if crit_ran else (None, False, False)
        if crit_ran:
            st["critic_exec"] += 1
            if is_ph:
                st["ph_cnt"]["critic_placeholder"] += 1
            if "PLACEHOLDER" in (rs.get("critique") or ""):
                st["ph_cnt"]["critique_placeholder"] += 1
            if step is not None:
                ms = rs.get("max_step")
                st["rollback"]["REFINE_QUERY(step==max)" if step == ms else "REFINE_CHAIN(step<max)"] += 1
                st["located_step_hist"][step] += 1
            else:
                st["rollback"]["locate-fail(placeholder/format)"] += 1

        ans_changed = pred_answer(ts) != pred_answer(rs)
        loc_key = "located" if step is not None else ("placeholder" if is_ph else ("not-run" if not crit_ran else "locate-fail"))

        if tc is None or rc is None:
            rec_path = "eval-fail"
        else:
            key = ("Correct" if tc else "Wrong", "Correct" if rc else "Wrong")
            rec_path = {
                ("Correct", "Correct"): "path_cc",
                ("Correct", "Wrong"): "path_c2w",
                ("Wrong", "Correct"): "path_w2c",
                ("Wrong", "Wrong"): "path_w2w",
            }[key]

        if rec_path == "path_c2w":
            if j0_ph:
                st["path_c2w"]["judge-placeholder→refined-wrong"] += 1
            elif j0_ok:
                st["path_c2w"]["judge-TN-but-refined-wrong(?)"] += 1
            else:
                st["path_c2w"][f"judge-FP(误检)→critic:{loc_key}→" + ("ans-changed" if ans_changed else "ans-unchanged")] += 1
        elif rec_path == "path_w2c":
            if j0_ph:
                st["path_w2c"]["judge-placeholder→refined-right"] += 1
            elif j0_ok:
                st["path_w2c"]["judge-FN-but-refined-right(?)"] += 1
            else:
                st["path_w2c"][f"judge-TP→critic:{loc_key}→" + ("ans-changed" if ans_changed else "ans-unchanged")] += 1
        elif rec_path == "path_w2w":
            if j0_ph:
                st["path_w2w"]["judge-placeholder"] += 1
            elif j0_ok:
                st["path_w2w"]["judge-FN(漏检,未修正)"] += 1
            else:
                st["path_w2w"][f"judge-TP→critic:{loc_key}→" + ("ans-changed" if ans_changed else "ans-unchanged")] += 1

        # ---- 明细导出（供 Anchor 训练取样）----
        detail_f.write(json.dumps({
            "id": rs["id"],
            "thought_correct": tc,
            "refine_correct": rc,
            "judge_list": jl,
            "conclusion_list": rs.get("conclusion_list", []),
            "max_step": rs.get("max_step"),
            "critique_placeholder": "PLACEHOLDER" in (rs.get("critique") or ""),
            "conclusion_raw": (rs.get("conclusion") or "")[:200],
            "pred_thought": pred_answer(ts),
            "pred_refine": pred_answer(rs),
            "n_ops_thought": len(ts.get("chain", [])),
            "n_ops_refine": len(rs.get("chain", [])),
        }, ensure_ascii=False) + "\n")
    detail_f.close()

    del t, r
    gc.collect()
    return st


def fmt_detect(st):
    tp = st["detect_cm"][("Wrong", "TP")]; fp = st["detect_cm"][("Correct", "FP")]
    fn = st["detect_cm"][("Wrong", "FN")]; tn = st["detect_cm"][("Correct", "TN")]
    lines = [
        "### 检测层：Judge（judge_list[0]，对 Thought 原链）", "",
        "| gold \\ judge | 检出(Incorrect) | 放过(Correct) |",
        "|---|---|---|",
        f"| gold=Wrong | TP={tp} | FN(漏检)={fn} |",
        f"| gold=Correct | FP(误检)={fp} | TN={tn} |",
    ]
    det_tot = tp + fp + fn + tn
    if det_tot:
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        lines.append(f"\n- 检出精确率 {prec:.3f}（{tp}/{tp+fp}），召回 {rec:.3f}（{tp}/{tp+fn}）；"
                     f"judge PLACEHOLDER: {st['judge_first_dist'].get('PLACEHOLDER', 0)}")
    return lines


def fmt_locate(st):
    loc_ok = st["rollback"]["REFINE_CHAIN(step<max)"] + st["rollback"]["REFINE_QUERY(step==max)"]
    lines = ["### 定位层：Critic conclusion 解析（进入 critic 的样本）", ""]
    lines.append(f"- critic 执行样本: {st['critic_exec']}")
    if st["critic_exec"]:
        lines.append(f"- 定位成功（解析出 Step N）: {loc_ok}（{loc_ok/st['critic_exec']:.1%}）；"
                     f"定位失败: {st['rollback']['locate-fail(placeholder/format)']}")
    lines.append(f"- 回滚类型: REFINE_CHAIN(链内回滚)={st['rollback']['REFINE_CHAIN(step<max)']}, "
                 f"REFINE_QUERY(仅末步)={st['rollback']['REFINE_QUERY(step==max)']}")
    if st["located_step_hist"]:
        hist = ", ".join(f"step{k}:{v}" for k, v in sorted(st["located_step_hist"].items()))
        lines.append(f"- 定位步号分布: {hist}")
    if st["ph_cnt"]:
        lines.append(f"- PLACEHOLDER 事件: {dict(st['ph_cnt'])}")
    return lines


def fmt_paths(st):
    lines = ["### 路径分解", ""]
    for name, cnt in (("Correct→Wrong（损失来源）", st["path_c2w"]),
                      ("Wrong→Correct（收益来源）", st["path_w2c"]),
                      ("Wrong→Wrong（纠错失败）", st["path_w2w"])):
        lines.append(f"**{name}**")
        if cnt:
            lines.extend(f"- {k}: {v}" for k, v in cnt.most_common())
        else:
            lines.append("- (空)")
        lines.append("")
    return lines


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--base", default=BASE, help="结果根目录（如 test/results_len32k）")
    args = ap.parse_args()
    BASE = args.base

    # 单次加载逐组合分析，收集结果后统一组装报告（避免双遍加载超出 cgroup 内存）
    results = {}
    for ds in args.datasets:
        for m in args.models:
            print(f"[analyzing] {ds}/{m} ...", flush=True)
            results[(ds, m)] = analyze_pair(ds, m)

    out = ["# Refine 负收益归因分析", "", "## 总览", ""]
    out.append("| dataset | model | C→C | C→W | W→C | W→W | 净收益 |")
    out.append("|---|---|---|---|---|---|---|")
    for ds in args.datasets:
        for m in args.models:
            q = results[(ds, m)]["quad"]
            tot = sum(v for k, v in q.items() if "eval-fail" not in k)
            cc, cw = q[("Correct", "Correct")], q[("Correct", "Wrong")]
            wc, ww = q[("Wrong", "Correct")], q[("Wrong", "Wrong")]
            ta = cc + cw
            ra = cc + wc
            out.append(f"| {ds} | {m} | {cc} | {cw} | {wc} | {ww} | {wc - cw:+d} |")
            out[-1] += f"  (Thought acc {ta}/{tot}={ta/tot:.4f}, Refine acc {ra}/{tot}={ra/tot:.4f})" if tot else "  (无有效样本)"

    for ds in args.datasets:
        for m in args.models:
            st = results[(ds, m)]
            q = st["quad"]
            cc, cw = q[("Correct", "Correct")], q[("Correct", "Wrong")]
            wc, ww = q[("Wrong", "Correct")], q[("Wrong", "Wrong")]
            out += ["", f"## {ds} / qwen3.5-{m}  (n={st['n']})", "", "### 四格表（Thought → Refine）", "",
                    "| Thought \\ Refine | Correct | Wrong |", "|---|---|---|",
                    f"| Correct | {cc} | {cw} |", f"| Wrong | {wc} | {ww} |", "",
                    f"- 保持正确 {cc}；错误修正损失 {cw}；有效修正收益 {wc}；未成功纠错 {ww}",
                    f"- 净收益 = {wc} − {cw} = {wc - cw:+d} 样本"]
            if st["eval_fail"]:
                out.append(f"- 评测抽取失败样本数（chain[-1] 解析异常）: {st['eval_fail']}")
            out += fmt_detect(st)
            out += fmt_locate(st)
            out += fmt_paths(st)
            out.append(f"- 明细已导出: {BASE}/refine/{ds}/qwen3.5-{m}_p12/attribution_detail.jsonl")

    report = "\n".join(out)
    print(report)
    report_path = f"{BASE}/refine_attribution_report.md"
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\n[报告已写入 {report_path}]")


if __name__ == "__main__":
    main()
