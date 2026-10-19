"""
双 ATG 构建器

从表格数据构建行中心 ATG 和列中心 ATG
"""

from typing import List, Optional, Tuple
import pandas as pd

from table_structure.graph import (
    AttributedTableGraph,
    AnchorNode,
    CellValueNode,
    Triple,
)
from table_structure.value_similarity import should_merge_values, unique_preserve_order


def _find_similar_node(value_map: dict, value: str, threshold: float) -> Optional[int]:
    """在同一列/行的已建值节点里查找可合并节点。"""
    for existing_value, node_idx in value_map.items():
        if should_merge_values(existing_value, value, threshold=threshold):
            return node_idx
    return None


def build_row_atg(
    table_text: List,
    headers: Optional[List[str]] = None,
    merge_similar_values: bool = False,
    similarity_threshold: float = 0.85,
) -> AttributedTableGraph:
    """
    构建行中心 ATG

    Root → Row anchors → Cell value nodes (边标注列头)
    三元组: ⟨r_i, h_j, c_{i,j}⟩

    Args:
        table_text: 表格数据, table_text[0] 为表头, table_text[1:] 为数据行
        headers: 可选的层级表头路径列表 (如 ["Region-Worker Type-Metric", ...])
        merge_similar_values: 是否在同一列内合并相似值节点。
        similarity_threshold: 相似度合并阈值。

    Returns:
        行中心 AttributedTableGraph
    """
    if headers is None:
        headers = [str(h) for h in table_text[0]]

    rows = table_text[1:]
    num_rows = len(rows)
    num_cols = len(headers)

    atg = AttributedTableGraph(anchor_type="row")
    atg.num_rows = num_rows
    atg.num_cols = num_cols
    atg.column_headers = headers

    # 为每列维护 value -> unique_index 的映射
    col_value_map: List[dict] = [{} for _ in range(num_cols)]

    # 构建锚点和三元组
    for row_idx in range(num_rows):
        row_data = rows[row_idx]
        anchor = AnchorNode(index=row_idx, label=f"row_{row_idx}")
        anchor_idx = len(atg.anchor_nodes)
        atg.anchor_nodes.append(anchor)
        atg.anchor_triple_map[anchor_idx] = []

        for col_idx in range(num_cols):
            cell_value = str(row_data[col_idx]) if col_idx < len(row_data) else ""

            # 创建或复用 cell value node
            key = cell_value
            if merge_similar_values and key not in col_value_map[col_idx]:
                similar_node_idx = _find_similar_node(
                    col_value_map[col_idx], key, similarity_threshold
                )
                if similar_node_idx is not None:
                    col_value_map[col_idx][key] = similar_node_idx

            if key not in col_value_map[col_idx]:
                col_value_map[col_idx][key] = len(atg.cell_nodes)
                atg.cell_nodes.append(
                    CellValueNode(
                        value=cell_value,
                        unique_index=len(atg.cell_nodes) - 1,
                        aliases=[cell_value],
                    )
                )

            cell_node_idx = col_value_map[col_idx][key]
            node = atg.cell_nodes[cell_node_idx]
            node.aliases = unique_preserve_order(node.aliases + [cell_value])
            atg.cell_nodes[cell_node_idx].anchor_indices.append(anchor_idx)

            # 创建三元组
            triple_idx = len(atg.triples)
            triple = Triple(
                anchor_idx=anchor_idx,
                edge_attr=headers[col_idx],
                cell_value=cell_value,
                row_idx=row_idx,
                col_idx=col_idx,
                cell_node_idx=cell_node_idx,
            )
            atg.triples.append(triple)

            # 更新索引
            atg.anchor_triple_map[anchor_idx].append(triple_idx)
            header_key = headers[col_idx]
            if header_key not in atg.edge_attr_triple_map:
                atg.edge_attr_triple_map[header_key] = []
            atg.edge_attr_triple_map[header_key].append(triple_idx)

            # 记录 cell node 到 anchor 的连接
            anchor.cell_nodes.append(cell_node_idx)

    return atg


def build_col_atg(
    table_text: List,
    headers: Optional[List[str]] = None,
    merge_similar_values: bool = False,
    similarity_threshold: float = 0.85,
) -> AttributedTableGraph:
    """
    构建列中心 ATG

    Root → Column anchors → Cell value nodes (边标注行索引)
    三元组: ⟨c_j, r_i, c_{i,j}⟩

    Args:
        table_text: 表格数据, table_text[0] 为表头, table_text[1:] 为数据行
        headers: 可选的层级表头路径列表
        merge_similar_values: 是否在同一行内合并相似值节点。
        similarity_threshold: 相似度合并阈值。

    Returns:
        列中心 AttributedTableGraph
    """
    if headers is None:
        headers = [str(h) for h in table_text[0]]

    rows = table_text[1:]
    num_rows = len(rows)
    num_cols = len(headers)

    atg = AttributedTableGraph(anchor_type="col")
    atg.num_rows = num_rows
    atg.num_cols = num_cols
    atg.column_headers = headers

    # 为每行维护 value -> unique_index 的映射
    row_value_map: List[dict] = [{} for _ in range(num_rows)]

    # 构建锚点和三元组
    for col_idx in range(num_cols):
        anchor = AnchorNode(index=col_idx, label=f"col_{col_idx}")
        anchor_idx = len(atg.anchor_nodes)
        atg.anchor_nodes.append(anchor)
        atg.anchor_triple_map[anchor_idx] = []

        for row_idx in range(num_rows):
            row_data = rows[row_idx]
            cell_value = str(row_data[col_idx]) if col_idx < len(row_data) else ""

            # 创建或复用 cell value node
            key = cell_value
            if merge_similar_values and key not in row_value_map[row_idx]:
                similar_node_idx = _find_similar_node(
                    row_value_map[row_idx], key, similarity_threshold
                )
                if similar_node_idx is not None:
                    row_value_map[row_idx][key] = similar_node_idx

            if key not in row_value_map[row_idx]:
                row_value_map[row_idx][key] = len(atg.cell_nodes)
                atg.cell_nodes.append(
                    CellValueNode(
                        value=cell_value,
                        unique_index=len(atg.cell_nodes) - 1,
                        aliases=[cell_value],
                    )
                )

            cell_node_idx = row_value_map[row_idx][key]
            node = atg.cell_nodes[cell_node_idx]
            node.aliases = unique_preserve_order(node.aliases + [cell_value])
            atg.cell_nodes[cell_node_idx].anchor_indices.append(anchor_idx)

            # 创建三元组 (edge_attr 用行索引标注)
            triple_idx = len(atg.triples)
            triple = Triple(
                anchor_idx=anchor_idx,
                edge_attr=f"row_{row_idx}",
                cell_value=cell_value,
                row_idx=row_idx,
                col_idx=col_idx,
                cell_node_idx=cell_node_idx,
            )
            atg.triples.append(triple)

            # 更新索引
            atg.anchor_triple_map[anchor_idx].append(triple_idx)
            edge_key = f"row_{row_idx}"
            if edge_key not in atg.edge_attr_triple_map:
                atg.edge_attr_triple_map[edge_key] = []
            atg.edge_attr_triple_map[edge_key].append(triple_idx)

            # 记录 cell node 到 anchor 的连接
            anchor.cell_nodes.append(cell_node_idx)

    return atg


def build_dual_atg(
    table_text: List,
    headers: Optional[List[str]] = None,
    merge_similar_values: bool = False,
    similarity_threshold: float = 0.85,
) -> Tuple[AttributedTableGraph, AttributedTableGraph]:
    """
    同时构建行中心 ATG 和列中心 ATG

    Args:
        table_text: 表格数据
        headers: 可选的层级表头路径列表
        merge_similar_values: 是否合并相似值节点。
        similarity_threshold: 相似度合并阈值。

    Returns:
        (row_atg, col_atg)
    """
    row_atg = build_row_atg(
        table_text,
        headers,
        merge_similar_values=merge_similar_values,
        similarity_threshold=similarity_threshold,
    )
    col_atg = build_col_atg(
        table_text,
        headers,
        merge_similar_values=merge_similar_values,
        similarity_threshold=similarity_threshold,
    )
    return row_atg, col_atg


def flatten_hierarchical_headers(headers: List) -> List[str]:
    """
    将多级表头压平为路径字符串

    如: ["Region", "", "Worker Type", "Metric", "", ""]
    ->  ["Region", "Region-Worker Type", "Region-Metric"]

    Args:
        headers: 原始表头列表 (可能含空字符串表示合并单元格)

    Returns:
        压平后的表头路径列表
    """
    if not headers:
        return []

    # 找出非空的顶级表头
    result = []
    current_parent = ""

    for h in headers:
        h_str = str(h).strip()
        if h_str:
            if current_parent and current_parent != h_str:
                result.append(f"{current_parent}-{h_str}")
            else:
                result.append(h_str)
            current_parent = h_str
        # 空字符串表示延续上一级表头, 跳过

    return result
