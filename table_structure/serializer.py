"""
序列化器: 将行列重排后的表格转为列表格式或 Markdown 文本

- rerank_table(): 返回重排后的列表（与原始 table_text 格式一致, 供 chain 操作使用）
- rerank_table_to_text(): 返回 Markdown 字符串（仅供独立使用）
"""

from typing import List, Optional


def rerank_table(
    table_text: List,
    row_order: List[int],
    col_order: List[int],
) -> List:
    """
    按行列排序重构表格, 返回与原始 table_text 相同的列表格式

    Args:
        table_text: 原始表格, table_text[0] 为表头, table_text[1:] 为数据行
        row_order: 按重要性降序排列的行索引列表
        col_order: 按重要性降序排列的列索引列表

    Returns:
        重排后的列表 [new_headers, new_row1, new_row2, ...]
    """
    headers = table_text[0]
    rows = table_text[1:]

    # 重排列
    reordered_headers = [headers[c] for c in col_order if c < len(headers)]

    # 重排行
    reordered_rows = []
    for orig_row_idx in row_order:
        if orig_row_idx >= len(rows):
            continue
        row = rows[orig_row_idx]
        reordered_row = [str(row[c]) if c < len(row) else "" for c in col_order]
        reordered_rows.append(reordered_row)

    return [reordered_headers] + reordered_rows


def rerank_table_to_text(
    table_text: List,
    row_order: List[int],
    col_order: List[int],
    caption: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> str:
    """
    按行列排序重构 Markdown 表格文本（仅用于独立输出, 不参与管线）

    输出格式与 table2string() 一致:
        col : col1 | col2 | col3
        row 1 : val1 | val2 | val3

    Args:
        table_text: 原始表格, table_text[0] 为表头, table_text[1:] 为数据行
        row_order: 按重要性降序排列的行索引列表
        col_order: 按重要性降序排列的列索引列表
        caption: 可选的表格标题
        extra_context: 可选的额外上下文

    Returns:
        Markdown 格式的表格字符串
    """
    reranked = rerank_table(table_text, row_order, col_order)
    headers = reranked[0]
    rows = reranked[1:]

    linear_table = ""
    if caption is not None:
        linear_table += "table caption : " + caption + "\n"

    linear_table += "col : " + " | ".join(str(h) for h in headers) + "\n"

    if extra_context:
        linear_table += "\n" + extra_context + "\n"

    for row_idx, row in enumerate(rows):
        line = "row {} : ".format(row_idx + 1) + " | ".join(str(v) for v in row)
        if row_idx != len(rows) - 1:
            line += "\n"
        linear_table += line

    return linear_table
