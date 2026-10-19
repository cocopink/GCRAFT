"""
table_structure: 双 ATG + QG-PPR 表格结构增强模块

将表格构建为行中心/列中心两种属性化图,
通过 QG-PPR 计算行列重要性, 重排行列顺序后输出为 Markdown

用法:
    from table_structure import build_dual_atg, dual_rank, rerank_table_to_text

    # 构建
    row_atg, col_atg = build_dual_atg(table_text)

    # 排序
    row_order, col_order = dual_rank(row_atg, col_atg, question)

    # 序列化 (输出格式与 table2string() 一致)
    reranked_table = rerank_table_to_text(table_text, row_order, col_order)
"""

from table_structure.graph import AttributedTableGraph
from table_structure.builder import (
    build_dual_atg,
    build_row_atg,
    build_col_atg,
    flatten_hierarchical_headers,
)
from table_structure.qg_ppr import (
    qg_ppr,
    rank_rows,
    rank_rows_with_scores,
    rank_columns,
    rank_columns_with_scores,
    rank_row_atg_columns_with_scores,
    dual_rank,
    dual_rank_with_scores,
)
from table_structure.column_similarity import ColumnSimilarityComputer
from table_structure.column_smoothing import build_smoothed_propagation_matrix
from table_structure.context import build_relevance_context
from table_structure.graph_bias import apply_graph_bias, parse_column_candidate, parse_row_candidate
from table_structure.serializer import rerank_table, rerank_table_to_text

__all__ = [
    # 数据结构
    "AttributedTableGraph",
    # 构建
    "build_dual_atg",
    "build_row_atg",
    "build_col_atg",
    "flatten_hierarchical_headers",
    # QG-PPR
    "qg_ppr",
    "rank_rows",
    "rank_rows_with_scores",
    "rank_columns",
    "rank_columns_with_scores",
    "rank_row_atg_columns_with_scores",
    "dual_rank",
    "dual_rank_with_scores",
    "ColumnSimilarityComputer",
    "build_smoothed_propagation_matrix",
    "build_relevance_context",
    "apply_graph_bias",
    "parse_column_candidate",
    "parse_row_candidate",
    # 序列化
    "rerank_table",
    "rerank_table_to_text",
]
