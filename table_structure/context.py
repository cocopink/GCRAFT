"""图相关上下文构造。"""

from typing import Dict


def build_relevance_context(sample: Dict, top_k: int = 3, **_) -> str:
    """从统一图字段生成 Refine 可消费的简短提示。"""
    row_scores = sample.get("row_relevance_scores") or sample.get("_row_relevance_scores") or {}
    col_scores = sample.get("col_relevance_scores") or sample.get("_col_relevance_scores") or {}

    lines = ["[Graph Hint]"]
    top_cols = [
        name
        for name, _ in sorted(col_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    ]
    if top_cols:
        lines.append("Relevant columns: " + ", ".join(str(col) for col in top_cols))

    top_rows = []
    for row_idx, _ in sorted(row_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        try:
            top_rows.append(f"row {int(row_idx) + 1}")
        except (TypeError, ValueError):
            continue
    if top_rows:
        lines.append("Relevant rows: " + ", ".join(top_rows))

    if len(lines) == 1:
        return ""
    lines.append("[End Graph Hint]")
    return "\n".join(lines)


def get_graph_hint(sample: Dict, top_k: int = 3) -> str:
    """按实验开关读取或构造 Graph Hint。"""
    metadata = sample.get("graph_metadata") or {}
    if not metadata.get("inject_graph_hint", False):
        return ""
    hint = sample.get("graph_hint", "")
    if hint:
        return hint
    return build_relevance_context(sample, top_k=top_k)
