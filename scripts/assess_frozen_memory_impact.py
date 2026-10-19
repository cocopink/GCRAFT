"""评估 frozen_memory NameError 事故对历史实验的影响（只读分析，不修改任何数据）。

背景：2026-09-22 起，Refine 修正成功（Judge 判对）后的 UPDATE_TREE 动作在
`if frozen_memory:` 处 NameError 崩溃，worker except 将样本回退为 thought 原始
答案——修正成果被系统性丢弃。本脚本从日志提取崩溃样本，结合 final_result.pkl
量化各 run 的实际损失。

口径注意：
- 仅 "name 'frozen_memory' is not defined" 计入本事故；其他 "Error refining
  sample" 错误（如其它 ImportError）单独列示，不参与损失评估。
- pkl 按模型匹配；若 pkl 修改时间晚于该次日志的启动时间，说明 pkl 已被更晚的
  运行覆盖，该行的「崩溃样本当前正确」配对失效（仅计数仍有意义）。

用法（仓库根目录）：
    python3 scripts/assess_frozen_memory_impact.py \
        --output docs/incidents/2026-10-19_frozen_memory_impact_report.md
"""
import argparse
import glob
import os
import pickle
import re
import sys
from datetime import datetime
from importlib.util import spec_from_file_location, module_from_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LINE_RE = re.compile(r"Error refining sample (\d+): (.*)")
RUN_RE = re.compile(r"^run_(QA|FV)_vllm-(?P<model>.+?)-(?P<ts>\d{8}-\d{6})\.log$")
TASK_DATASET = {"QA": "wikitq", "FV": "tabfact"}
EVAL_SRC = {
    "QA": os.path.join(ROOT, "refine/TableQA/utils/evaluate.py"),
    "FV": os.path.join(ROOT, "refine/TableFV/utils/evaluate.py"),
}
TAGGED_DIR = os.path.join(ROOT, "thought/TableQA/data/wikitq/tagged_data")


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_log(path):
    """返回 (name_error_ids, other_errors)，均为去重后按序排列。"""
    name_err, other_err = {}, {}
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            idx, msg = int(m.group(1)), m.group(2)
            if "frozen_memory" in msg and "not defined" in msg:
                name_err.setdefault(idx, msg)
            else:
                other_err.setdefault(idx, msg.strip()[:80])
    return sorted(name_err), other_err


def load_qa_targets(qa_eval):
    """复刻 wikitq_match_func_for_samples 的 gold 加载逻辑，返回 {id: target_values}。"""
    targets = {}
    for filename in os.listdir(TAGGED_DIR):
        if filename[0] == ".":
            continue
        with open(os.path.join(TAGGED_DIR, filename), "r", encoding="utf8") as fin:
            header = fin.readline().rstrip("\n").split("\t")
            for line in fin:
                stuff = dict(zip(header, line.rstrip("\n").split("\t")))
                original = qa_eval.tsv_unescape_list(stuff["targetValue"])
                canon = qa_eval.tsv_unescape_list(stuff["targetCanon"])
                targets[stuff["id"]] = qa_eval.to_value_list(original, canon)
    return targets


def score_sample(task, sample, qa_eval, qa_targets):
    if task == "QA":
        return bool(qa_eval.wikitq_match_func(sample, qa_targets[sample["ids"]]))
    return bool(fv_evals[task].tabfact_match_func(sample))


def find_pkls(dataset, model, roots, include_broken=False):
    hits = []
    for root in roots:
        hits.extend(glob.glob(os.path.join(root, "refine", dataset, f"{model}*", "final_result.pkl")))
    if not include_broken:
        hits = [p for p in hits if "broken" not in os.path.basename(os.path.dirname(p))]
    return sorted(set(hits))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs_dir", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--results_roots", nargs="+",
                    default=[os.path.join(ROOT, "test/results"), os.path.join(ROOT, "results")])
    ap.add_argument("--output", default=os.path.join(ROOT, "docs/incidents/2026-10-19_frozen_memory_impact_report.md"))
    ap.add_argument("--include_broken", action="store_true")
    args = ap.parse_args()

    global fv_evals
    qa_eval = load_module("assess_qa_evaluate", EVAL_SRC["QA"])
    fv_evals = {"FV": load_module("assess_fv_evaluate", EVAL_SRC["FV"])}
    qa_targets = load_qa_targets(qa_eval) if os.path.isdir(TAGGED_DIR) else None

    rows = []
    for log_path in sorted(glob.glob(os.path.join(args.logs_dir, "run_*_vllm-*.log"))):
        name = os.path.basename(log_path)
        m = RUN_RE.match(name)
        if not m:
            continue
        task, model = m.group(1), m.group("model")
        log_dt = datetime.strptime(m.group("ts"), "%Y%m%d-%H%M%S")
        dataset = TASK_DATASET[task]
        crashed, other_errs = parse_log(log_path)
        if not crashed and not other_errs:
            continue

        for pkl in find_pkls(dataset, model, args.results_roots, args.include_broken) or [None]:
            row = {"log": name, "log_dt": log_dt, "task": task, "dataset": dataset,
                   "model": model, "pkl": pkl or "(未找到 final_result.pkl)",
                   "pkl_stale": False, "total": None,
                   "crashed": len(crashed), "other": len(other_errs), "crashed_oob": 0,
                   "crashed_correct": None, "overall_correct": None, "eval_err": 0}
            if pkl:
                samples = pickle.load(open(pkl, "rb"))
                row["total"] = len(samples)
                row["pkl_stale"] = os.path.getmtime(pkl) > log_dt.timestamp()
                in_range = [i for i in crashed if i < len(samples)]
                row["crashed_oob"] = len(crashed) - len(in_range)
                eval_err = correct_crashed = overall = 0
                for i in in_range:
                    try:
                        correct_crashed += bool(score_sample(task, samples[i], qa_eval, qa_targets))
                    except Exception:
                        eval_err += 1
                for s in samples:
                    try:
                        overall += bool(score_sample(task, s, qa_eval, qa_targets))
                    except Exception:
                        eval_err += 1
                row["crashed_correct"] = correct_crashed
                row["overall_correct"] = overall
                row["eval_err"] = eval_err
            rows.append(row)

    # ---- 渲染 Markdown ----
    lines = [
        "# frozen_memory NameError 事故历史影响评估报告",
        "",
        "生成方式：`python3 scripts/assess_frozen_memory_impact.py`（只读分析）。",
        "",
        "- **NameError 崩溃**：日志中 `Error refining sample N: name 'frozen_memory' is not defined`",
        "  的去重样本数——这些样本的 Refine 修正成果被丢弃、回退 thought 原始答案。",
        "- **其他错误**：同入口的其余异常（如中途重构期的 ImportError），不计入本事故损失。",
        "- **pkl 配对**：pkl 按模型匹配；若其修改时间晚于日志启动时间，说明已被更晚的运行覆盖，",
        "  该行「崩溃样本当前正确」的配对失效（标 ⚠）。",
        "",
        "| 日志 | 数据集/模型 | NameError崩溃 | 其他错误 | 总样本 | 崩溃率 | 崩溃样本当前正确 | 整体正确 | 增益上限 | pkl配对 |",
        "|------|------------|---------------|----------|--------|--------|------------------|----------|----------|---------|",
    ]
    for r in rows:
        pkl_short = os.path.relpath(r["pkl"], ROOT) if r["pkl"].startswith("/") else r["pkl"]
        if r["total"]:
            rate = f"{r['crashed'] / r['total']:.1%}"
            gain = f"+{r['crashed'] - r['crashed_correct']}/{r['total']} ({(r['crashed'] - r['crashed_correct']) / r['total']:.1%})"
            cc = f"{r['crashed_correct']}/{r['crashed']}"
            if r["eval_err"]:
                cc += f" (判分异常 {r['eval_err']})"
            pair = "⚠ 已被更晚运行覆盖" if r["pkl_stale"] else "✓"
            overall = f"{r['overall_correct']}/{r['total']}"
            total = str(r["total"])
        else:
            rate = cc = overall = gain = total = "—"
            pair = "—"
        lines.append(f"| {r['log']} | {r['dataset']}/{r['model']} | {r['crashed']} | {r['other']} "
                     f"| {total} | {rate} | {cc} | {overall} | {gain} | {pair} |")
        if r["crashed_oob"]:
            lines.append(f"  - 注: {r['crashed_oob']} 个崩溃索引超出 pkl 范围（样本数不一致）。")
    lines += [
        "",
        "## 口径说明",
        "",
        "- **崩溃样本当前正确**：这些样本在 final_result.pkl 中的现判分（事故样本预期≈0；",
        "  若明显 >0，说明初始 Judge 误判了本来正确的 thought 答案）。",
        "- **增益上限**：若接线修复后重跑，这些样本恢复 Judge 判对的成果；注意 **Judge 判对 ≠ cal_acc 判对**，",
        "  实际增益 ≤ 上限（以重跑后的 cal_acc 为准）。",
        "- 判分复用 `refine/*/utils/evaluate.py` 的单样本逻辑（WikiTQ 需 tagged_data gold；TabFact 用样本自带 label）。",
        "",
        "## 结论建议",
        "",
        "1. 受影响 run 的 Refine 增益被系统性低估，**用于论文对比前建议在修复后的代码上重跑**；",
        "2. 重跑将默认启用「每实验独立进化树」（`FROZEN_MEMORY=False`），与旧冻结语义不同，需在实验记录中标注；",
        "   若要复现旧静态树语义，用 `FROZEN_MEMORY=True`。",
    ]

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)
    print(f"报告已写入: {args.output}")


if __name__ == "__main__":
    main()
