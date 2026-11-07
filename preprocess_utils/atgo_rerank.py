"""ATG 行/列单向重排预处理模块

提供 ATGO (仅行排)、ATGC (仅列排) 和 ATGRC (行+列联合) 三种模式:
- ATGO: 仅基于行 ATG + QG-PPR 重排行顺序, 列顺序保持原样
- ATGC: 仅基于列 ATG + QG-PPR 重排列顺序, 行顺序保持原样
- ATGRC: 行序来自行 ATG、列序来自列 ATG, 独立打分后一次重构表格

三个重排器均将重排结果双写至 atg_reranked_table (thought 阶段回落链首选字段),
保证任意模式的数据文件可直接作为 thought 输入。

用法:
    from preprocess_utils.atgo_rerank import ATGOReranker, ATGCReranker, ATGRowColReranker

    # 仅行排
    row_reranker = ATGOReranker(cache_dir="./cache/atgo_rerank")
    dataset = row_reranker.rerank_batch(dataset)

    # 仅列排
    col_reranker = ATGCReranker(cache_dir="./cache/atgc_rerank")
    dataset = col_reranker.rerank_batch(dataset)

    # 行+列联合
    rc_reranker = ATGRowColReranker(cache_dir="./cache/atgrc_rerank")
    dataset = rc_reranker.rerank_batch(dataset)
"""

import os
import pickle
from typing import Dict, List, Optional

from tqdm import tqdm

from table_structure import (
    build_col_atg,
    build_relevance_context,
    build_row_atg,
    rank_columns_with_scores,
    rank_row_atg_columns_with_scores,
    rank_rows_with_scores,
    rerank_table,
)


GRAPH_SCHEMA_VERSION = 1


def _order_changed(order: List[int], n: int) -> bool:
    """判断排序是否与恒等序列不同。

    Args:
        order: 排序结果列表
        n: 原始元素数量

    Returns:
        True 表示顺序发生了改变
    """
    return order != list(range(n))


def _cache_path(cache_dir: str, sample_id: str, metadata: Dict) -> str:
    """生成包含实验开关的缓存路径。"""
    name = (
        f"{sample_id}.schema-{metadata['schema_version']}"
        f".mode-{metadata['rerank_mode']}"
        f".gamma-{metadata['gamma']}"
        f".hint-{int(metadata['inject_graph_hint'])}"
    )
    # bias 段仅在 P3 开启 (bias_weight > 0) 时附加, bias=0 保持与既有缓存名兼容
    if metadata.get("bias_weight", 0.0) > 0:
        name += f".bias-{metadata['bias_weight']}"
    name += ".pkl"
    return os.path.join(cache_dir, name)


def _legacy_cache_path(cache_dir: str, sample_id: str) -> str:
    return os.path.join(cache_dir, f"{sample_id}.pkl")


class ATGOReranker:
    """基于行 ATG + QG-PPR 的仅行重排器

    与 ATGReranker 的区别:
    - 仅构建行中心 ATG, 不构建列中心 ATG
    - 仅计算行排序, 列顺序保持原始顺序
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        gamma: float = 0.0,
        inject_graph_hint: bool = False,
        bias_weight: float = 0.0,
    ):
        self.cache_dir = cache_dir
        self.gamma = gamma
        self.inject_graph_hint = inject_graph_hint
        self.bias_weight = bias_weight

    def _metadata(self) -> Dict:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "rerank_mode": "row",
            "gamma": self.gamma,
            "inject_graph_hint": self.inject_graph_hint,
            "bias_weight": self.bias_weight,
        }

    def _load_cache(self, sample: Dict, cache_path: str) -> bool:
        cached = pickle.load(open(cache_path, "rb"))
        if isinstance(cached, dict) and "atgo_reranked_table" in cached:
            sample.update(cached)
            return True
        sample["atgo_reranked_table"] = cached
        return True

    def rerank_sample(self, sample: Dict, idx: int = 0) -> Dict:
        """对单个样本执行 ATGO 重排, 结果写入 sample["atgo_reranked_table"]。

        同时设置 sample["_atgo_order_changed"] 标记行顺序是否实际改变。

        Args:
            sample: 数据集样本, 需包含 table_text / table 和 question / statement。
            idx: 样本在数据集中的索引, 用于 sample_id 缺失时的缓存回退。

        Returns:
            同一 sample 对象（原地修改）。
        """
        table_text = sample.get("table", sample.get("table_text", []))
        question = sample.get("question", sample.get("statement", ""))
        sample_id = sample.get("id") or sample.get("table_id") or f"idx_{idx}"

        if not question or not table_text:
            sample["atgo_reranked_table"] = table_text
            sample["_atgo_order_changed"] = False
            sample["graph_metadata"] = self._metadata()
            return sample

        # 尝试从缓存加载
        if self.cache_dir:
            cache_path = _cache_path(self.cache_dir, sample_id, self._metadata())
            legacy_cache_path = _legacy_cache_path(self.cache_dir, sample_id)
            if os.path.exists(cache_path):
                self._load_cache(sample, cache_path)
                return sample
            if os.path.exists(legacy_cache_path):
                self._load_cache(sample, legacy_cache_path)
                return sample

        try:
            row_atg = build_row_atg(table_text)
            row_order, row_scores = rank_rows_with_scores(
                row_atg,
                question,
                gamma=self.gamma,
            )
            _, col_scores = rank_row_atg_columns_with_scores(
                row_atg,
                question,
                gamma=self.gamma,
            )
            num_rows = len(table_text) - 1
            col_order = list(range(len(table_text[0])))
            reranked = rerank_table(table_text, row_order, col_order)
            sample["atgo_reranked_table"] = reranked
            sample["atg_reranked_table"] = reranked
            sample["_atgo_order_changed"] = _order_changed(row_order, num_rows)
            sample["graph_metadata"] = self._metadata()
            sample["row_relevance_scores"] = row_scores
            sample["col_relevance_scores"] = col_scores
            sample["graph_hint"] = (
                build_relevance_context(sample) if self.inject_graph_hint else ""
            )

            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                pickle.dump(
                    {
                        "atgo_reranked_table": sample["atgo_reranked_table"],
                        "atg_reranked_table": sample["atg_reranked_table"],
                        "_atgo_order_changed": sample["_atgo_order_changed"],
                        "graph_metadata": sample["graph_metadata"],
                        "row_relevance_scores": sample["row_relevance_scores"],
                        "col_relevance_scores": sample["col_relevance_scores"],
                        "graph_hint": sample["graph_hint"],
                    },
                    open(cache_path, "wb"),
                )
        except Exception as e:
            print(f"[ATGO] Rerank failed for sample {sample_id}: {e}")
            sample["atgo_reranked_table"] = table_text
            sample["atg_reranked_table"] = table_text
            sample["_atgo_order_changed"] = False
            sample["graph_metadata"] = self._metadata()

        return sample

    def rerank_batch(self, dataset: List[Dict]) -> List[Dict]:
        for idx, sample in tqdm(enumerate(dataset), desc="ATGO Reranking"):
            self.rerank_sample(sample, idx=idx)
        return dataset


class ATGCReranker:
    """基于列 ATG + QG-PPR 的仅列重排器

    与 ATGReranker 的区别:
    - 仅构建列中心 ATG, 不构建行中心 ATG
    - 仅计算列排序, 行顺序保持原始顺序
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        gamma: float = 0.0,
        inject_graph_hint: bool = False,
        bias_weight: float = 0.0,
    ):
        self.cache_dir = cache_dir
        self.gamma = gamma
        self.inject_graph_hint = inject_graph_hint
        self.bias_weight = bias_weight

    def _metadata(self) -> Dict:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "rerank_mode": "col",
            "gamma": self.gamma,
            "inject_graph_hint": self.inject_graph_hint,
            "bias_weight": self.bias_weight,
        }

    def _load_cache(self, sample: Dict, cache_path: str) -> bool:
        cached = pickle.load(open(cache_path, "rb"))
        if isinstance(cached, dict) and "atgc_reranked_table" in cached:
            sample.update(cached)
            # 旧缓存 (双写改造前) 不含 atg_reranked_table, 补齐保证 thought 回落链可读
            sample.setdefault("atg_reranked_table", sample["atgc_reranked_table"])
            return True
        sample["atgc_reranked_table"] = cached
        sample["atg_reranked_table"] = cached
        return True

    def rerank_sample(self, sample: Dict, idx: int = 0) -> Dict:
        """对单个样本执行 ATGC 重排, 结果写入 sample["atgc_reranked_table"]。

        同时设置 sample["_atgc_order_changed"] 标记列顺序是否实际改变。

        Args:
            sample: 数据集样本, 需包含 table_text / table 和 question / statement。
            idx: 样本在数据集中的索引, 用于 sample_id 缺失时的缓存回退。

        Returns:
            同一 sample 对象（原地修改）。
        """
        table_text = sample.get("table", sample.get("table_text", []))
        question = sample.get("question", sample.get("statement", ""))
        sample_id = sample.get("id") or sample.get("table_id") or f"idx_{idx}"

        if not question or not table_text:
            sample["atgc_reranked_table"] = table_text
            sample["atg_reranked_table"] = table_text
            sample["_atgc_order_changed"] = False
            sample["graph_metadata"] = self._metadata()
            return sample

        # 尝试从缓存加载
        if self.cache_dir:
            cache_path = _cache_path(self.cache_dir, sample_id, self._metadata())
            legacy_cache_path = _legacy_cache_path(self.cache_dir, sample_id)
            if os.path.exists(cache_path):
                self._load_cache(sample, cache_path)
                return sample
            if os.path.exists(legacy_cache_path):
                self._load_cache(sample, legacy_cache_path)
                return sample

        try:
            row_atg = build_row_atg(table_text)
            col_atg = build_col_atg(table_text)
            _, row_scores = rank_rows_with_scores(
                row_atg,
                question,
                gamma=self.gamma,
            )
            col_order, col_scores = rank_columns_with_scores(
                col_atg,
                question,
                gamma=self.gamma,
            )
            num_cols = len(table_text[0])
            row_order = list(range(len(table_text) - 1))
            reranked = rerank_table(table_text, row_order, col_order)
            sample["atgc_reranked_table"] = reranked
            sample["atg_reranked_table"] = reranked
            sample["_atgc_order_changed"] = _order_changed(col_order, num_cols)
            sample["graph_metadata"] = self._metadata()
            sample["row_relevance_scores"] = row_scores
            sample["col_relevance_scores"] = col_scores
            sample["graph_hint"] = (
                build_relevance_context(sample) if self.inject_graph_hint else ""
            )

            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                pickle.dump(
                    {
                        "atgc_reranked_table": sample["atgc_reranked_table"],
                        "atg_reranked_table": sample["atg_reranked_table"],
                        "_atgc_order_changed": sample["_atgc_order_changed"],
                        "graph_metadata": sample["graph_metadata"],
                        "row_relevance_scores": sample["row_relevance_scores"],
                        "col_relevance_scores": sample["col_relevance_scores"],
                        "graph_hint": sample["graph_hint"],
                    },
                    open(cache_path, "wb"),
                )
        except Exception as e:
            print(f"[ATGC] Rerank failed for sample {sample_id}: {e}")
            sample["atgc_reranked_table"] = table_text
            sample["atg_reranked_table"] = table_text
            sample["_atgc_order_changed"] = False
            sample["graph_metadata"] = self._metadata()

        return sample

    def rerank_batch(self, dataset: List[Dict]) -> List[Dict]:
        for idx, sample in tqdm(enumerate(dataset), desc="ATGC Reranking"):
            self.rerank_sample(sample, idx=idx)
        return dataset


class ATGRowColReranker:
    """行+列联合重排器 (ATGRC)

    行序来自行 ATG + QG-PPR, 列序来自列 ATG + QG-PPR, 两者独立打分,
    之后一次 rerank_table 同时完成行间互换与列间互换。
    与单向重排器 (ATGO/ATGC) 的区别: 行、列顺序同时被重排。
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        gamma: float = 0.0,
        inject_graph_hint: bool = False,
        bias_weight: float = 0.0,
    ):
        self.cache_dir = cache_dir
        self.gamma = gamma
        self.inject_graph_hint = inject_graph_hint
        self.bias_weight = bias_weight

    def _metadata(self) -> Dict:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "rerank_mode": "rowcol",
            "gamma": self.gamma,
            "inject_graph_hint": self.inject_graph_hint,
            "bias_weight": self.bias_weight,
        }

    def _load_cache(self, sample: Dict, cache_path: str) -> bool:
        cached = pickle.load(open(cache_path, "rb"))
        if isinstance(cached, dict) and "atgrc_reranked_table" in cached:
            sample.update(cached)
            # 旧缓存 (双写改造前) 不含 atg_reranked_table, 补齐保证 thought 回落链可读
            sample.setdefault("atg_reranked_table", sample["atgrc_reranked_table"])
            return True
        sample["atgrc_reranked_table"] = cached
        sample["atg_reranked_table"] = cached
        return True

    def rerank_sample(self, sample: Dict, idx: int = 0) -> Dict:
        """对单个样本执行行+列联合重排, 结果写入 sample["atgrc_reranked_table"]。

        同时双写 sample["atg_reranked_table"] (thought 回落链首选字段),
        并设置 sample["_atgrc_order_changed"] 标记行/列顺序是否任一发生改变。

        Args:
            sample: 数据集样本, 需包含 table_text / table 和 question / statement。
            idx: 样本在数据集中的索引, 用于 sample_id 缺失时的缓存回退。

        Returns:
            同一 sample 对象（原地修改）。
        """
        table_text = sample.get("table", sample.get("table_text", []))
        question = sample.get("question", sample.get("statement", ""))
        sample_id = sample.get("id") or sample.get("table_id") or f"idx_{idx}"

        if not question or not table_text:
            sample["atgrc_reranked_table"] = table_text
            sample["atg_reranked_table"] = table_text
            sample["_atgrc_order_changed"] = False
            sample["graph_metadata"] = self._metadata()
            return sample

        # 尝试从缓存加载
        if self.cache_dir:
            cache_path = _cache_path(self.cache_dir, sample_id, self._metadata())
            legacy_cache_path = _legacy_cache_path(self.cache_dir, sample_id)
            if os.path.exists(cache_path):
                self._load_cache(sample, cache_path)
                return sample
            if os.path.exists(legacy_cache_path):
                self._load_cache(sample, legacy_cache_path)
                return sample

        try:
            row_atg = build_row_atg(table_text)
            col_atg = build_col_atg(table_text)
            row_order, row_scores = rank_rows_with_scores(
                row_atg,
                question,
                gamma=self.gamma,
            )
            col_order, col_scores = rank_columns_with_scores(
                col_atg,
                question,
                gamma=self.gamma,
            )
            num_rows = len(table_text) - 1
            num_cols = len(table_text[0])
            reranked = rerank_table(table_text, row_order, col_order)
            sample["atgrc_reranked_table"] = reranked
            sample["atg_reranked_table"] = reranked
            sample["_atgrc_order_changed"] = _order_changed(
                row_order, num_rows
            ) or _order_changed(col_order, num_cols)
            sample["graph_metadata"] = self._metadata()
            sample["row_relevance_scores"] = row_scores
            sample["col_relevance_scores"] = col_scores
            sample["graph_hint"] = (
                build_relevance_context(sample) if self.inject_graph_hint else ""
            )

            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                pickle.dump(
                    {
                        "atgrc_reranked_table": sample["atgrc_reranked_table"],
                        "atg_reranked_table": sample["atg_reranked_table"],
                        "_atgrc_order_changed": sample["_atgrc_order_changed"],
                        "graph_metadata": sample["graph_metadata"],
                        "row_relevance_scores": sample["row_relevance_scores"],
                        "col_relevance_scores": sample["col_relevance_scores"],
                        "graph_hint": sample["graph_hint"],
                    },
                    open(cache_path, "wb"),
                )
        except Exception as e:
            print(f"[ATGRC] Rerank failed for sample {sample_id}: {e}")
            sample["atgrc_reranked_table"] = table_text
            sample["atg_reranked_table"] = table_text
            sample["_atgrc_order_changed"] = False
            sample["graph_metadata"] = self._metadata()

        return sample

    def rerank_batch(self, dataset: List[Dict]) -> List[Dict]:
        for idx, sample in tqdm(enumerate(dataset), desc="ATGRC Reranking"):
            self.rerank_sample(sample, idx=idx)
        return dataset
