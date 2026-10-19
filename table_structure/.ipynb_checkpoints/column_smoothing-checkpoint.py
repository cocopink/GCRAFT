"""QG-PPR 列间平滑传播矩阵。"""

from typing import Optional

import numpy as np

from table_structure.graph import AttributedTableGraph
from table_structure.qg_ppr import DEFAULT_W_VAL, _build_propagation_matrix


def build_smoothed_propagation_matrix(
    atg: AttributedTableGraph,
    column_similarity: Optional[np.ndarray],
    w_row: float,
    w_col: float,
    w_val: float = DEFAULT_W_VAL,
    gamma: float = 0.1,
) -> np.ndarray:
    """构建加入列间平滑的三元组传播矩阵。

    `gamma=0` 时保持原矩阵，便于 baseline 消融。
    """
    base = _build_propagation_matrix(atg, w_row, w_col, w_val=w_val)
    if gamma <= 0 or column_similarity is None or len(atg.triples) == 0:
        return base

    gamma = min(max(float(gamma), 0.0), 1.0)
    smooth = np.zeros_like(base)
    triples_by_col = {}
    for triple_idx, triple in enumerate(atg.triples):
        triples_by_col.setdefault(triple.col_idx, []).append(triple_idx)

    for src_idx, src in enumerate(atg.triples):
        if src.col_idx < 0 or src.col_idx >= column_similarity.shape[0]:
            continue
        for dst_col, col_prob in enumerate(column_similarity[src.col_idx]):
            targets = triples_by_col.get(dst_col, [])
            if not targets or col_prob <= 0:
                continue
            mass = float(col_prob) / len(targets)
            for dst_idx in targets:
                if dst_idx != src_idx:
                    smooth[src_idx, dst_idx] += mass

    row_sums = smooth.sum(axis=1, keepdims=True)
    non_empty_rows = row_sums[:, 0] > 0
    smooth[non_empty_rows] /= row_sums[non_empty_rows]
    return (1 - gamma) * base + gamma * smooth
