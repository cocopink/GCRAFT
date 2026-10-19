"""
QG-PPR (Question-Guided Personalized PageRank) 重要性排序

参考论文: Beyond Linearization: Attributed Table Graphs for Table Reasoning
https://arxiv.org/abs/2601.08444

对行中心 ATG 计算行重要性, 对列中心 ATG 计算列重要性
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from table_structure.graph import AttributedTableGraph
from table_structure.value_similarity import should_merge_values


# 默认超参数
DEFAULT_ALPHA = 0.35
DEFAULT_ITERATIONS = 20
DEFAULT_V_COL = 1.0
DEFAULT_V_VAL = 2.0
DEFAULT_W_ROW_ROW_ATG = 0.6  # 行中心 ATG 的行传播权重
DEFAULT_W_COL_ROW_ATG = 0.4
DEFAULT_W_ROW_COL_ATG = 0.4  # 列中心 ATG 的行传播权重
DEFAULT_W_COL_COL_ATG = 0.6
DEFAULT_W_VAL = 0.3


def _value_matches_question(value: str, question: str) -> bool:
    if not value:
        return False
    value_lower = value.lower()
    if value_lower in question:
        return True
    question_tokens = re.findall(r"[a-zA-Z0-9]+", question)
    return any(should_merge_values(token, value_lower, threshold=0.85) for token in question_tokens)


def _triple_value_matches_question(idx: int, atg: AttributedTableGraph, question: str) -> bool:
    triple = atg.triples[idx]
    if _value_matches_question(triple.cell_value, question):
        return True
    if triple.cell_node_idx < 0 or triple.cell_node_idx >= len(atg.cell_nodes):
        return False
    return any(
        _value_matches_question(alias, question)
        for alias in atg.cell_nodes[triple.cell_node_idx].aliases
    )


def _build_personalization_vector(
    question: str,
    atg: AttributedTableGraph,
    v_col: float = DEFAULT_V_COL,
    v_val: float = DEFAULT_V_VAL,
) -> np.ndarray:
    """
    构建个性化向量 p0

    p0(i,j) = v_col * I(h_j in H_q) + v_val * IDF(c_{i,j}) * I(c_{i,j} in C_q)

    Args:
        question: 输入问题
        atg: AttributedTableGraph (行中心或列中心)
        v_col: 列名匹配权重
        v_val: 单元格值匹配权重

    Returns:
        个性化向量 (shape: [num_triples])
    """
    n = len(atg.triples)
    p0 = np.zeros(n)
    question_lower = question.lower()

    # 构建 H_q: 与 question 精确匹配的边属性集合
    # 行中心: edge_attr 是列头
    # 列中心: edge_attr 是行索引, 不匹配列名 → 跳过 v_col
    H_q: set = set()
    if atg.anchor_type == "row":
        for triple in atg.triples:
            if triple.edge_attr.lower() in question_lower:
                H_q.add(triple.edge_attr)

    # 构建 C_q: 与 question 精确匹配的单元格值集合 + 统计 df
    C_q: set = set()
    value_df: Dict[str, int] = {}
    for triple in atg.triples:
        val = triple.cell_value
        value_df[val] = value_df.get(val, 0) + 1
        if val.lower() in question_lower:
            C_q.add(val)

    # 计算 IDF
    N = atg.num_rows if atg.anchor_type == "row" else atg.num_cols
    def idf(val: str) -> float:
        df = value_df.get(val, 0)
        return math.log(1 + N / (1 + df))

    # 填充 p0
    for idx, triple in enumerate(atg.triples):
        score = 0.0
        # 列名匹配
        if atg.anchor_type == "row" and triple.edge_attr in H_q:
            score += v_col
        # 单元格值匹配 (带 IDF)
        if triple.cell_value in C_q or _triple_value_matches_question(idx, atg, question_lower):
            score += v_val * idf(triple.cell_value)
        p0[idx] = score

    # 归一化为概率分布
    total = p0.sum()
    if total > 0:
        p0 /= total
    else:
        # 无匹配时使用均匀分布
        p0 = np.ones(n) / n

    return p0


def _build_propagation_matrix(
    atg: AttributedTableGraph,
    w_row: float,
    w_col: float,
    w_val: float = DEFAULT_W_VAL,
) -> np.ndarray:
    """
    构建三元组级别的传播矩阵 Â

    两个三元组 u, v 相连条件:
    - 同锚点 (行中心: 同行 / 列中心: 同列)
    - 同边属性 (行中心: 同列头 / 列中心: 同行索引)

    传播权重:
    - 同锚点: w_row / |C(anchor)|
    - 同边属性: w_col / |C(edge_attr)|

    Args:
        atg: AttributedTableGraph
        w_row: 同锚点传播权重
        w_col: 同边属性传播权重

    Returns:
        传播矩阵 (shape: [n, n])
    """
    n = len(atg.triples)
    A = np.zeros((n, n))
    cell_node_triple_map: Dict[int, List[int]] = {}
    for triple_idx, triple in enumerate(atg.triples):
        if triple.cell_node_idx < 0:
            continue
        cell_node_triple_map.setdefault(triple.cell_node_idx, []).append(triple_idx)

    for i in range(n):
        ti = atg.triples[i]
        # 同锚点的三元组
        for j in atg.anchor_triple_map.get(ti.anchor_idx, []):
            if i != j:
                A[i, j] = w_row
        # 同边属性的三元组
        for j in atg.edge_attr_triple_map.get(ti.edge_attr, []):
            if i != j:
                A[i, j] = w_col
        # 同合并值节点的三元组
        for j in cell_node_triple_map.get(ti.cell_node_idx, []):
            if i != j:
                A[i, j] = max(A[i, j], w_val)

    # 列归一化: 每行之和为 1
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A /= row_sums

    return A


def power_iteration(
    p0: np.ndarray,
    A: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    K: int = DEFAULT_ITERATIONS,
) -> np.ndarray:
    """
    PageRank 幂迭代

    s^(t+1) = alpha * p0 + (1 - alpha) * A^T * s^(t)

    Args:
        p0: 个性化向量
        A: 传播矩阵
        alpha: 传送概率
        K: 迭代次数

    Returns:
        salience 分数向量
    """
    s = np.copy(p0)
    A_T = A.T
    for _ in range(K):
        s = alpha * p0 + (1 - alpha) * (A_T @ s)
    return s


def qg_ppr(
    atg: AttributedTableGraph,
    question: str,
    alpha: float = DEFAULT_ALPHA,
    K: int = DEFAULT_ITERATIONS,
    v_col: float = DEFAULT_V_COL,
    v_val: float = DEFAULT_V_VAL,
    w_row: Optional[float] = None,
    w_col: Optional[float] = None,
    gamma: float = 0.0,
    column_similarity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    QG-PPR 完整流程: p0 + Â + 幂迭代 → salience scores

    Args:
        atg: AttributedTableGraph
        question: 输入问题
        alpha: 传送概率
        K: 迭代次数
        v_col, v_val: 列名/值匹配权重
        w_row, w_col: 传播权重 (None 时根据 anchor_type 自动选择)

    Returns:
        salience 分数向量 (shape: [num_triples])
    """
    if w_row is None or w_col is None:
        if atg.anchor_type == "row":
            w_row = DEFAULT_W_ROW_ROW_ATG
            w_col = DEFAULT_W_COL_ROW_ATG
        else:
            w_row = DEFAULT_W_ROW_COL_ATG
            w_col = DEFAULT_W_COL_COL_ATG

    p0 = _build_personalization_vector(question, atg, v_col, v_val)
    if gamma > 0 and atg.anchor_type == "row":
        from table_structure.column_similarity import ColumnSimilarityComputer
        from table_structure.column_smoothing import build_smoothed_propagation_matrix

        if column_similarity is None:
            column_similarity = ColumnSimilarityComputer().compute(atg)
        A = build_smoothed_propagation_matrix(
            atg,
            column_similarity,
            w_row,
            w_col,
            gamma=gamma,
        )
    else:
        A = _build_propagation_matrix(atg, w_row, w_col)
    scores = power_iteration(p0, A, alpha, K)
    return scores


def rank_rows_with_scores(
    row_atg: AttributedTableGraph,
    question: str,
    **kwargs,
) -> Tuple[List[int], Dict[str, float]]:
    """
    计算行重要性并返回行排序索引

    S(r_i) = sum_j s_{i,j}

    Args:
        row_atg: 行中心 ATG
        question: 输入问题
        **kwargs: 传递给 qg_ppr 的超参数

    Returns:
        按重要性降序排列的行索引列表
    """
    scores = qg_ppr(row_atg, question, **kwargs)

    # 按锚点 (行) 聚合分数
    row_scores = {}
    for idx, triple in enumerate(row_atg.triples):
        row_scores[triple.anchor_idx] = row_scores.get(triple.anchor_idx, 0) + scores[idx]

    # 按分数降序排列
    row_indices = list(range(row_atg.num_rows))
    row_indices.sort(key=lambda r: row_scores.get(r, 0), reverse=True)
    serializable_scores = {
        str(row_idx): float(row_scores.get(row_idx, 0.0))
        for row_idx in range(row_atg.num_rows)
    }
    return row_indices, serializable_scores


def rank_rows(row_atg: AttributedTableGraph, question: str, **kwargs) -> List[int]:
    """
    计算行重要性并返回行排序索引

    S(r_i) = sum_j s_{i,j}

    Args:
        row_atg: 行中心 ATG
        question: 输入问题
        **kwargs: 传递给 qg_ppr 的超参数

    Returns:
        按重要性降序排列的行索引列表
    """
    row_indices, _ = rank_rows_with_scores(row_atg, question, **kwargs)
    return row_indices


def rank_columns_with_scores(
    col_atg: AttributedTableGraph,
    question: str,
    **kwargs,
) -> Tuple[List[int], Dict[str, float]]:
    """
    计算列重要性并返回列排序索引

    S(c_j) = sum_i s_{i,j}

    Args:
        col_atg: 列中心 ATG
        question: 输入问题
        **kwargs: 传递给 qg_ppr 的超参数

    Returns:
        按重要性降序排列的列索引列表
    """
    scores = qg_ppr(col_atg, question, **kwargs)

    # 按锚点 (列) 聚合分数
    col_scores = {}
    for idx, triple in enumerate(col_atg.triples):
        col_scores[triple.anchor_idx] = col_scores.get(triple.anchor_idx, 0) + scores[idx]

    # 按分数降序排列
    col_indices = list(range(col_atg.num_cols))
    col_indices.sort(key=lambda c: col_scores.get(c, 0), reverse=True)
    serializable_scores = {
        str(col_atg.column_headers[col_idx]).lower(): float(col_scores.get(col_idx, 0.0))
        for col_idx in range(col_atg.num_cols)
    }
    return col_indices, serializable_scores


def rank_row_atg_columns_with_scores(
    row_atg: AttributedTableGraph,
    question: str,
    **kwargs,
) -> Tuple[List[int], Dict[str, float]]:
    """从行中心 ATG 的列边属性聚合列相关性分数。"""
    scores = qg_ppr(row_atg, question, **kwargs)
    col_scores = {}
    for idx, triple in enumerate(row_atg.triples):
        col_scores[triple.col_idx] = col_scores.get(triple.col_idx, 0.0) + scores[idx]

    col_indices = list(range(row_atg.num_cols))
    col_indices.sort(key=lambda c: col_scores.get(c, 0.0), reverse=True)
    serializable_scores = {
        str(row_atg.column_headers[col_idx]).lower(): float(col_scores.get(col_idx, 0.0))
        for col_idx in range(row_atg.num_cols)
    }
    return col_indices, serializable_scores


def rank_columns(col_atg: AttributedTableGraph, question: str, **kwargs) -> List[int]:
    """
    计算列重要性并返回列排序索引

    S(c_j) = sum_i s_{i,j}

    Args:
        col_atg: 列中心 ATG
        question: 输入问题
        **kwargs: 传递给 qg_ppr 的超参数

    Returns:
        按重要性降序排列的列索引列表
    """
    col_indices, _ = rank_columns_with_scores(col_atg, question, **kwargs)
    return col_indices


def dual_rank(
    row_atg: AttributedTableGraph,
    col_atg: AttributedTableGraph,
    question: str,
    **kwargs,
) -> Tuple[List[int], List[int]]:
    """
    双图 QG-PPR: 同时计算行重要性和列重要性

    Args:
        row_atg: 行中心 ATG
        col_atg: 列中心 ATG
        question: 输入问题
        **kwargs: 传递给 qg_ppr 的超参数

    Returns:
        (row_order, col_order) 按重要性降序排列
    """
    row_order = rank_rows(row_atg, question, **kwargs)
    col_order = rank_columns(col_atg, question, **kwargs)
    return row_order, col_order


def dual_rank_with_scores(
    row_atg: AttributedTableGraph,
    col_atg: AttributedTableGraph,
    question: str,
    **kwargs,
) -> Tuple[List[int], List[int], Dict[str, float], Dict[str, float]]:
    """同时返回行列排序及可序列化 relevance scores。"""
    row_order, row_scores = rank_rows_with_scores(row_atg, question, **kwargs)
    col_order, col_scores = rank_columns_with_scores(col_atg, question, **kwargs)
    return row_order, col_order, row_scores, col_scores
