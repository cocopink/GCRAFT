"""从 error_results.pkl 反推 error_data.jsonl

将 Thought 阶段输出 pkl 中的错误样本还原为原始 JSONL 格式，
剥离所有 Thought 阶段字段（chain, clarifier, reranked_table 等），
仅保留原始数据字段。

用法:
    python scripts/build_error_data.py --tabfact
    python scripts/build_error_data.py --wikitq
    python scripts/build_error_data.py --all
"""

import json
import os
import pickle

import fire


# 需要剥离的 Thought 阶段字段（不写入 error_data.jsonl）
_THOUGHT_STAGE_FIELDS = {
    "chain",
    "clarifier",
    "cleaned_statement",
    "atg_reranked_table",
    "atgo_reranked_table",
    "atgc_reranked_table",
    "atg_context",
    "graph_relevance",
    "_atg_order_changed",
    "_atgo_order_changed",
    "_atgc_order_changed",
    "table_analysis",
}

# 各数据集保留的原始字段
_TABFACT_FIELDS = {"statement", "label", "table_caption", "table_text", "table_id"}
_WIKITQ_FIELDS = {"statement", "table_text", "answer", "ids"}


def _strip_thought_fields(sample: dict) -> dict:
    """剥离 Thought 阶段字段，返回仅含原始字段的 dict。"""
    return {k: v for k, v in sample.items() if k not in _THOUGHT_STAGE_FIELDS}


def build_tabfact_error_data() -> str:
    """从 TabFV error_results.pkl 生成 error_data.jsonl。"""
    pkl_path = "test/results/thought/tabfact/gpt-5.4/error_results.pkl"
    output_path = "thought/TableFV/data/tabfact/error_data.jsonl"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 仅保留原始字段
    samples = []
    for s in data:
        cleaned = _strip_thought_fields(s)
        # 确保 id 不混入（thought 阶段生成，load_tabfact_dataset 会重新设置）
        cleaned.pop("id", None)
        samples.append(cleaned)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"TabFV: {len(samples)} samples -> {output_path}")
    return output_path


def build_wikitq_error_data() -> str:
    """从 WikiTQ error_results.pkl 生成 error_data.jsonl。"""
    pkl_path = "test/results/thought/wikitq/gpt-5.4_row/error_results.pkl"
    output_path = "thought/TableQA/data/wikitq/error_data.jsonl"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    samples = []
    for s in data:
        cleaned = _strip_thought_fields(s)
        cleaned.pop("id", None)
        cleaned.pop("table_caption", None)
        samples.append(cleaned)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"WikiTQ: {len(samples)} samples -> {output_path}")
    return output_path


def main(tabfact: bool = False, wikitq: bool = False, all: bool = False):
    """生成 error_data.jsonl。

    Args:
        tabfact: 生成 TabFV error_data.jsonl
        wikitq: 生成 WikiTQ error_data.jsonl
        all: 生成全部
    """
    if all or tabfact:
        build_tabfact_error_data()
    if all or wikitq:
        build_wikitq_error_data()

    if not (all or tabfact or wikitq):
        print("Please specify --tabfact, --wikitq, or --all")


if __name__ == "__main__":
    fire.Fire(main)
