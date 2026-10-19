"""列相似度计算。

用于图增强实验的 P1，不引入额外模型依赖。
"""

import re
from typing import List, Set

import numpy as np

from table_structure.graph import AttributedTableGraph


class ColumnSimilarityComputer:
    """基于列名、列值和位置计算行归一化相似度矩阵。"""

    _SEMANTIC_GROUPS = {
        "person": {"name", "person", "player", "actor", "author", "winner"},
        "location": {"country", "nation", "city", "state", "region", "province"},
        "time": {"year", "date", "month", "season", "time"},
        "identifier": {"id", "code", "rank", "number", "no"},
        "metric": {"score", "points", "population", "total", "average", "count"},
    }

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma_pos: float = 0.2,
        use_rules: bool = True,
        use_embedding: bool = False,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma_pos = gamma_pos
        self.use_rules = use_rules
        self.use_embedding = use_embedding

    def compute(self, atg: AttributedTableGraph) -> np.ndarray:
        """返回 M x M 的行归一化列相似度矩阵。"""
        num_cols = atg.num_cols
        if num_cols == 0:
            return np.zeros((0, 0))

        name_sim = self._name_similarity_by_rules(atg)
        value_sim = self._value_overlap_similarity(atg)
        pos_sim = self._position_similarity(atg)
        scores = self.alpha * name_sim + self.beta * value_sim + self.gamma_pos * pos_sim
        np.fill_diagonal(scores, 0.0)
        return self._normalize_rows(scores)

    def _name_similarity_by_rules(self, atg: AttributedTableGraph) -> np.ndarray:
        headers = [str(h).lower() for h in atg.column_headers]
        groups = [self._header_groups(header) for header in headers]
        n = len(headers)
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if groups[i] and groups[j] and groups[i].intersection(groups[j]):
                    sim[i, j] = 0.8
                elif self._token_jaccard(headers[i], headers[j]) > 0:
                    sim[i, j] = self._token_jaccard(headers[i], headers[j])
                else:
                    sim[i, j] = 0.1 if self.use_rules else 0.0
        return sim

    def _value_overlap_similarity(self, atg: AttributedTableGraph) -> np.ndarray:
        values_by_col = self._values_by_col(atg)
        n = atg.num_cols
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                left = values_by_col[i]
                right = values_by_col[j]
                union = left | right
                if union:
                    sim[i, j] = len(left & right) / len(union)
        return sim

    def _position_similarity(self, atg: AttributedTableGraph) -> np.ndarray:
        n = atg.num_cols
        if n <= 1:
            return np.zeros((n, n))
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sim[i, j] = max(0.0, 1.0 - abs(i - j) / n)
        return sim

    def _normalize_rows(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        row_sums = scores.sum(axis=1, keepdims=True)
        result = scores.copy()
        for idx, total in enumerate(row_sums[:, 0]):
            if total > 0:
                result[idx] /= total
            elif result.shape[1] > 1:
                result[idx] = 1.0 / (result.shape[1] - 1)
                result[idx, idx] = 0.0
        return result

    def _header_groups(self, header: str) -> Set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", header.lower()))
        groups = set()
        for group_name, group_tokens in self._SEMANTIC_GROUPS.items():
            if tokens.intersection(group_tokens):
                groups.add(group_name)
        return groups

    def _token_jaccard(self, left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
        right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return len(left_tokens & right_tokens) / len(union)

    def _values_by_col(self, atg: AttributedTableGraph) -> List[Set[str]]:
        values = [set() for _ in range(atg.num_cols)]
        for triple in atg.triples:
            if 0 <= triple.col_idx < atg.num_cols:
                values[triple.col_idx].add(str(triple.cell_value).strip().lower())
        return values
