"""ATG 重排预处理模块

将双 ATG + QG-PPR 行列重排作为独立预处理步骤,
结果写入 sample["atg_reranked_table"], 供下游 chain.py 直接消费。

用法:
    from preprocess_utils.atg_rerank import ATGReranker

    reranker = ATGReranker(cache_dir="./cache/atg_rerank")
    dataset = reranker.rerank_batch(dataset)
"""

import os
import pickle
from typing import Dict, List, Optional

from tqdm import tqdm

from table_structure import build_dual_atg, dual_rank, rerank_table
from table_structure.context import build_atg_context


def _tables_equal(t1: list, t2: list) -> bool:
    """判断两个 table_text 列表是否顺序完全相同。"""
    if len(t1) != len(t2):
        return False
    for r1, r2 in zip(t1, t2):
        if len(r1) != len(r2):
            return False
        for v1, v2 in zip(r1, r2):
            if str(v1) != str(v2):
                return False
    return True


def _attach_atg_context(sample: Dict) -> None:
    """生成 thought 前可消费的 ATG 全局上下文。"""
    context = build_atg_context(sample, merge_similar_values=True)
    if context:
        sample["atg_context"] = context


class ATGReranker:
    """基于双 ATG + QG-PPR 的表格行列重排器"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: 可选的逐样本 pkl 缓存目录。
                        提供时启用增量处理（已缓存样本跳过计算）。
        """
        self.cache_dir = cache_dir

    def rerank_sample(self, sample: Dict, idx: int = 0) -> Dict:
        """对单个样本执行 ATG 重排, 结果写入 sample["atg_reranked_table"]。

        同时设置 sample["_atg_order_changed"] 标记顺序是否实际改变。

        Args:
            sample: 数据集样本, 需包含 table_text / table 和 question / statement。
            idx: 样本在数据集中的索引, 用于 sample_id 缺失时的缓存回退。

        Returns:
            同一 sample 对象（原地修改）。
        """
        table_text = sample.get("table", sample.get("table_text", []))
        question = sample.get("question", sample.get("statement", ""))
        sample_id = sample.get("id") or f"idx_{idx}"

        # 无 question 时跳过（QG-PPR 需要问题引导）
        if not question or not table_text:
            sample["atg_reranked_table"] = table_text
            sample["_atg_order_changed"] = False
            _attach_atg_context(sample)
            return sample

        # 尝试从缓存加载
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{sample_id}.pkl")
            if os.path.exists(cache_path):
                reranked = pickle.load(open(cache_path, "rb"))
                sample["atg_reranked_table"] = reranked
                sample["_atg_order_changed"] = not _tables_equal(table_text, reranked)
                _attach_atg_context(sample)
                return sample

        # 执行 ATG + QG-PPR 重排
        try:
            row_atg, col_atg = build_dual_atg(table_text)
            row_order, col_order = dual_rank(row_atg, col_atg, question)
            reranked = rerank_table(table_text, row_order, col_order)
            sample["atg_reranked_table"] = reranked
            sample["_atg_order_changed"] = not _tables_equal(table_text, reranked)
            _attach_atg_context(sample)

            # 写入缓存
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                pickle.dump(reranked, open(cache_path, "wb"))
        except Exception as e:
            print(f"[ATG] Rerank failed for sample {sample_id}: {e}")
            sample["atg_reranked_table"] = table_text
            sample["_atg_order_changed"] = False
            _attach_atg_context(sample)

        return sample

    def rerank_batch(self, dataset: List[Dict]) -> List[Dict]:
        """批量处理数据集。

        Args:
            dataset: 样本列表。

        Returns:
            处理后的样本列表（原地修改）。
        """
        for idx, sample in tqdm(enumerate(dataset), desc="ATG Reranking"):
            self.rerank_sample(sample, idx=idx)
        return dataset
