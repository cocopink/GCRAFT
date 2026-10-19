"""预处理入口: ATG 表格重排

用法:
    python preprocess.py --dataset_path input.jsonl --output_path output.jsonl
    python preprocess.py --dataset_path input.jsonl --output_path output.jsonl --cache_dir ./cache

输出:
    - output.jsonl: 完整数据集，每个样本新增 atg_reranked_table 字段
    - output_changed.jsonl: 仅含顺序改变的样本，只保留 id / question / statement / table_text / atg_reranked_table
"""

import json
import os

import fire
from preprocess_utils.atg_rerank import ATGReranker


def main(
    dataset_path: str,
    output_path: str,
    cache_dir: str = None,
):
    """对数据集执行 ATG + QG-PPR 行列重排预处理。

    Args:
        dataset_path: 输入 JSONL 文件路径。
        output_path: 输出 JSONL 文件路径（含 atg_reranked_table 字段）。
        cache_dir: 可选的逐样本 pkl 缓存目录。默认在输出文件同级
                   atg_rerank_cache/ 子目录。
    """
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
        cache_dir = os.path.join(os.path.dirname(output_path), "atg_rerank_cache")

    # 执行重排
    reranker = ATGReranker(cache_dir=cache_dir)
    reranker.rerank_batch(dataset)

    # 写入完整输出
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset)} samples to {output_path}")

    # 写入仅含修改样本的精简 JSONL
    changed = []
    for sample in dataset:
        if sample.get("_atg_order_changed"):
            changed.append({
                "id": sample.get("id"),
                "question": sample.get("question", ""),
                "statement": sample.get("statement", ""),
                "table_text": sample.get("table", sample.get("table_text")),
                "atg_reranked_table": sample["atg_reranked_table"],
            })

    # 从 output_path 推导 changed 路径: output.jsonl → output_changed.jsonl
    base, ext = os.path.splitext(output_path)
    changed_path = f"{base}_changed{ext}"
    with open(changed_path, "w", encoding="utf-8") as f:
        for item in changed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(changed)} changed samples to {changed_path}")


if __name__ == "__main__":
    fire.Fire(main)
