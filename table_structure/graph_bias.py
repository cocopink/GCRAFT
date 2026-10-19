"""Thought 候选图分数偏置工具。"""

import ast
import math
import re
from typing import Dict, List, Sequence, Tuple


def _literal_list(candidate: str):
    text = str(candidate).strip()
    if not text:
        return []
    if not text.startswith("["):
        text = "[" + text + "]"
    return ast.literal_eval(text)


def parse_column_candidate(candidate: str) -> List[str]:
    """解析列候选，保留列名内部逗号。"""
    parsed = _literal_list(candidate)
    if isinstance(parsed, str):
        return [parsed.strip()]
    return [str(item).strip() for item in parsed]


def parse_row_candidate(candidate: str) -> List[int]:
    """解析展示行号候选并转换为内部 0-based 行号。"""
    parsed = _literal_list(candidate)
    if isinstance(parsed, (str, int)):
        parsed = [parsed]
    rows = []
    for item in parsed:
        match = re.search(r"\d+", str(item))
        if match:
            rows.append(int(match.group(0)) - 1)
    return rows


def canonicalize_candidate(candidate: str, candidate_type: str) -> str:
    """将 regex 捕获的候选归一化为现有 operation 兼容的字符串列表。"""
    try:
        if candidate_type == "row":
            parsed = _literal_list(candidate)
            if isinstance(parsed, (str, int)):
                parsed = [parsed]
            rows = []
            for item in parsed:
                if str(item).strip() == "*":
                    rows.append("*")
                    continue
                match = re.search(r"\d+", str(item))
                if match:
                    rows.append(match.group(0))
            return str(sorted(rows))
        cols = [col.lower() for col in parse_column_candidate(candidate)]
        return str(sorted(cols))
    except (SyntaxError, ValueError, TypeError):
        if candidate_type == "column":
            cols = [part.strip().strip("'\"").lower() for part in candidate.split(",")]
            cols = [col for col in cols if col]
            return str(sorted(cols))
        rows = []
        for part in candidate.split(","):
            match = re.search(r"\d+", part)
            if match:
                rows.append(match.group(0))
        return str(sorted(rows))


def apply_graph_bias(
    candidates: Sequence[Tuple[str, float]],
    relevance_scores: Dict,
    bias_weight: float,
    candidate_type: str,
) -> List[Tuple[str, float]]:
    """按图分数重排已有候选，不新增候选。"""
    if bias_weight <= 0 or not relevance_scores:
        return sorted(candidates, key=lambda item: item[1], reverse=True)

    normalized_scores = {str(key).lower(): float(value) for key, value in relevance_scores.items()}

    def graph_score(candidate: str) -> float:
        try:
            if candidate_type == "row":
                keys = [str(idx) for idx in parse_row_candidate(candidate)]
            else:
                keys = [col.lower() for col in parse_column_candidate(candidate)]
        except (SyntaxError, ValueError, TypeError):
            return 0.0
        if not keys:
            return 0.0
        return max(normalized_scores.get(key, 0.0) for key in keys)

    return sorted(
        candidates,
        key=lambda item: item[1] * math.exp(bias_weight * graph_score(item[0])),
        reverse=True,
    )
