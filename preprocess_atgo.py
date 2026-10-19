"""预处理入口: ATG 行/列单向重排

用法:
    # 仅行排
    python preprocess_atgo.py --mode row --dataset_path input.jsonl --output_path output_row.jsonl

    # 仅列排
    python preprocess_atgo.py --mode col --dataset_path input.jsonl --output_path output_col.jsonl

输出:
    - output.jsonl: 完整数据集，每个样本新增 atgo/atgc_reranked_table 字段
    - output_changed.jsonl: 仅含顺序改变的样本
"""

import json
import os

import fire
from preprocess_utils.atgo_rerank import ATGOReranker, ATGCReranker


def main(
    dataset_path: str,
    output_path: str,
    mode: str = "row",
    cache_dir: str = None,
    gamma: float = 0.0,
    inject_graph_hint: bool = False,
    bias_weight: float = 0.0,
):
    """对数据集执行 ATG 单向重排预处理。

    Args:
        dataset_path: 输入 JSONL 文件路径。
        output_path: 输出 JSONL 文件路径。
        mode: 重排模式, "row" (仅行排) 或 "col" (仅列排)。
        cache_dir: 可选的逐样本 pkl 缓存目录。
        gamma: P1 列间平滑强度, 0.0 表示关闭。
        inject_graph_hint: 是否写入 Refine 图提示。
        bias_weight: P3 Thought 候选偏置强度。
    """
    if mode not in ("row", "col"):
        raise ValueError(f"mode must be 'row' or 'col', got '{mode}'")

    # 加载数据集
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    print(f"Loaded {len(dataset)} samples from {dataset_path}")

    # 默认缓存目录
    if cache_dir is None:
        cache_name = f"atg{mode}_rerank_cache"
        cache_dir = os.path.join(os.path.dirname(output_path), cache_name)

    # 根据模式选择 reranker
    if mode == "row":
        reranker = ATGOReranker(
            cache_dir=cache_dir,
            gamma=gamma,
            inject_graph_hint=inject_graph_hint,
            bias_weight=bias_weight,
        )
        table_key = "atgo_reranked_table"
        changed_key = "_atgo_order_changed"
        desc = "ATGO (row-only)"
    else:
        reranker = ATGCReranker(
            cache_dir=cache_dir,
            gamma=gamma,
            inject_graph_hint=inject_graph_hint,
            bias_weight=bias_weight,
        )
        table_key = "atgc_reranked_table"
        changed_key = "_atgc_order_changed"
        desc = "ATGC (col-only)"

    reranker.rerank_batch(dataset)

    # 写入完整输出
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset)} samples to {output_path}")

    # 统计变化数
    changed = sum(1 for s in dataset if s.get(changed_key))
    print(f"Changed samples: {changed}/{len(dataset)} ({100*changed/len(dataset):.1f}%)")

    # 写入仅含修改样本的精简 JSONL
    changed_samples = []
    for sample in dataset:
        if sample.get(changed_key):
            changed_samples.append({
                "id": sample.get("id", sample.get("table_id")),
                "question": sample.get("question", ""),
                "statement": sample.get("statement", ""),
                "table_text": sample.get("table", sample.get("table_text")),
                table_key: sample[table_key],
                "graph_metadata": sample.get("graph_metadata", {}),
                "row_relevance_scores": sample.get("row_relevance_scores", {}),
                "col_relevance_scores": sample.get("col_relevance_scores", {}),
                "graph_hint": sample.get("graph_hint", ""),
            })

    base, ext = os.path.splitext(output_path)
    changed_path = f"{base}_changed{ext}"
    with open(changed_path, "w", encoding="utf-8") as f:
        for item in changed_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(changed_samples)} changed samples to {changed_path}")


if __name__ == "__main__":
    fire.Fire(main)
