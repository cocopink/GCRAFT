"""
Attributed Table Graph (ATG) 数据结构

参考论文: Beyond Linearization: Attributed Table Graphs for Table Reasoning
https://arxiv.org/abs/2601.08444

支持两种视角:
- 行中心 ATG (Row-centric): Root → Row anchors → Cell nodes (边标注列头)
- 列中心 ATG (Column-centric): Root → Column anchors → Cell nodes (边标注行索引)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CellValueNode:
    """
    单元格值节点

    行中心 ATG 中: 列 j 的第 k 个唯一值
    列中心 ATG 中: 行 i 的第 k 个唯一值
    """
    value: str
    unique_index: int  # 在该列/行中的唯一编号
    anchor_indices: List[int] = field(default_factory=list)  # 连接的锚点索引
    aliases: List[str] = field(default_factory=list)  # 合并后的原始值变体


@dataclass
class AnchorNode:
    """
    锚点节点

    行中心 ATG 中: 代表一行 (row anchor)
    列中心 ATG 中: 代表一列 (column anchor)
    """
    index: int  # 行索引 or 列索引
    label: str  # "row_i" or "col_j"
    cell_nodes: List[int] = field(default_factory=list)  # 连接的 CellValueNode 索引


@dataclass
class Triple:
    """
    三元组

    行中心: ⟨anchor(行), edge_attr(列头), cell_value⟩
    列中心: ⟨anchor(列), edge_attr(行索引), cell_value⟩
    """
    anchor_idx: int       # 锚点索引
    edge_attr: str        # 列头 (行中心) or 行索引 (列中心)
    cell_value: str       # 单元格值
    row_idx: int          # 原始行索引 (用于定位)
    col_idx: int          # 原始列索引 (用于定位)
    cell_node_idx: int = -1  # 指向合并后的 CellValueNode


class AttributedTableGraph:
    """
    属性化表格图

    统一数据结构, 通过 anchor_type 区分行中心/列中心视角
    """

    def __init__(self, anchor_type: str = "row"):
        """
        Args:
            anchor_type: "row" (行中心) or "col" (列中心)
        """
        self.anchor_type = anchor_type

        # 节点存储
        self.cell_nodes: List[CellValueNode] = []
        self.anchor_nodes: List[AnchorNode] = []
        self.cell_value_map: Dict[str, int] = {}  # (anchor_idx, value) -> cell_node_idx

        # 三元组列表
        self.triples: List[Triple] = []

        # 三元组索引: anchor_idx -> [triple indices]
        self.anchor_triple_map: Dict[int, List[int]] = {}
        # edge_attr -> [triple indices]
        self.edge_attr_triple_map: Dict[str, List[int]] = {}

        # 元信息
        self.num_rows: int = 0
        self.num_cols: int = 0
        self.column_headers: List[str] = field(default_factory=list) if hasattr(self, 'column_headers') else []
        self._column_headers: List[str] = []

    @property
    def column_headers(self) -> List[str]:
        return self._column_headers

    @column_headers.setter
    def column_headers(self, value: List[str]):
        self._column_headers = value

    def get_triples_by_anchor(self, anchor_idx: int) -> List[Triple]:
        """获取某锚点的所有三元组"""
        return [self.triples[i] for i in self.anchor_triple_map.get(anchor_idx, [])]

    def get_triples_by_edge_attr(self, edge_attr: str) -> List[Triple]:
        """获取某边属性的所有三元组"""
        return [self.triples[i] for i in self.edge_attr_triple_map.get(edge_attr, [])]

    def num_triples(self) -> int:
        return len(self.triples)

    def summary(self) -> str:
        return (
            f"ATG({self.anchor_type}): "
            f"{len(self.anchor_nodes)} anchors, "
            f"{len(self.cell_nodes)} cell nodes, "
            f"{len(self.triples)} triples, "
            f"{self.num_rows}x{self.num_cols}"
        )
