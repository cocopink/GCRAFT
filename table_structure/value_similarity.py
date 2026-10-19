"""列内单元格值相似度与等价簇工具。"""

from difflib import SequenceMatcher
import re
from typing import Iterable, List


_ARTICLES = {"the", "a", "an"}
_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u s": "united states",
    "u s a": "united states",
    "united states of america": "united states",
}


def normalize_value(value: str) -> str:
    """将值规整到适合相似度比较的轻量形式。"""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = _ALIASES.get(text, text)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    normalized = " ".join(tokens)
    if len(tokens) == 1 and normalized.endswith("s") and len(normalized) > 3:
        normalized = normalized[:-1]
    return _ALIASES.get(normalized, normalized)


def value_similarity(left: str, right: str) -> float:
    """计算两个单元格值的轻量相似度。"""
    a = normalize_value(left)
    b = normalize_value(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    a_tokens = set(a.split())
    b_tokens = set(b.split())
    token_score = 0.0
    if a_tokens and b_tokens:
        token_score = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    seq_score = SequenceMatcher(None, a, b).ratio()
    containment_score = 1.0 if a in b or b in a else 0.0
    return max(token_score, seq_score, containment_score)


def should_merge_values(left: str, right: str, threshold: float = 0.85) -> bool:
    """判断两个值是否应在同一列内合并为等价节点。"""
    return value_similarity(left, right) >= threshold


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    """按首次出现顺序去重。"""
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
