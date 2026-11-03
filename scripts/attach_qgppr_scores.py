"""为注入集样本补算 QG-PPR 分数 (raw 行序保留, 仅贴分数不重排)。

背景: ATGO 重排本体经 E2 判定为负, 图模块 (P4/P5/P6) 重评宿主切到 raw。
但 raw thought 产物无 row/col_relevance_scores, 图方法无从计算。
本脚本按 experiment_plan §8 预案对原始表 CPU 现算 QG-PPR 分数并贴附,
行序保持 raw 原序 (分数行键 = 原始行号, 与 raw 执行链行号天然对齐)。

γ 取 0.0 (P1 已判定关闭口径, 与 E2 C2_atgo 预处理一致)。

用法:
    python scripts/attach_qgppr_scores.py \
        --injected results/p5_injection/wikitq_injected.pkl \
        --output results/p5_injection/wikitq_injected_scored.pkl
"""

import argparse
import os
import pickle
import sys

from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from table_structure import build_row_atg, rank_row_atg_columns_with_scores, rank_rows_with_scores  # noqa: E402

GAMMA = 0.0  # P1 关闭口径


def score_sample(sample):
    """就地写入 row/col_relevance_scores; 返回 True=成功, False=跳过。"""
    table_text = sample.get("table", sample.get("table_text", []))
    question = sample.get("question", sample.get("statement", ""))
    if not question or not table_text or len(table_text) < 2:
        return False
    row_atg = build_row_atg(table_text)
    _, row_scores = rank_rows_with_scores(row_atg, question, gamma=GAMMA)
    _, col_scores = rank_row_atg_columns_with_scores(row_atg, question, gamma=GAMMA)
    # 分数键与重排口径一致: 行键为字符串化 0-based 数据行号, 列键为列名
    sample["row_relevance_scores"] = row_scores
    sample["col_relevance_scores"] = col_scores
    return True


def main():
    ap = argparse.ArgumentParser(description="注入集补算 QG-PPR 分数 (raw 行序保留)")
    ap.add_argument("--injected", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.injected, "rb") as f:
        payload = pickle.load(f)

    ok, skip = 0, 0
    for item in tqdm(payload["injected"], desc="Scoring"):
        if score_sample(item["sample"]):
            ok += 1
        else:
            skip += 1

    payload["meta"]["scores_attached"] = {"gamma": GAMMA, "ok": ok, "skipped": skip}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)
    print(f"分数贴附完成: 成功 {ok}, 跳过 {skip} → {args.output}")


if __name__ == "__main__":
    main()
