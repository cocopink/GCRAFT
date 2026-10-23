"""Thought 候选解析与归一化工具。"""

import ast
import re
from typing import List


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
