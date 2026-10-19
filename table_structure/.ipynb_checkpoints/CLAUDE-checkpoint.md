# Table Structure 模块文档

[根目录](../CLAUDE.md) > **table_structure**

> 最后更新：2026-06-09 19:39:59

---

## 变更记录 (Changelog)

### 2026-06-09 (新建)
- 初始化 table_structure 模块文档
- 深度扫描 graph.py、builder.py、qg_ppr.py、serializer.py

---

## 模块职责

Table Structure 模块实现**双 ATG (Attributed Table Graph) + QG-PPR (Question-Guided Personalized PageRank)** 表格结构增强，用于表格行列重排。

参考论文: Beyond Linearization: Attributed Table Graphs for Table Reasoning (arXiv:2601.08444)

### 核心功能

1. **双 ATG 构建**: 行中心 ATG (Row-centric) + 列中心 ATG (Column-centric)
2. **QG-PPR 重要性排序**: 基于问题的个性化 PageRank 计算行列重要性
3. **序列化**: 重排后的表格转为列表格式或 Markdown 文本

---

## 入口与启动

```python
from table_structure import build_dual_atg, dual_rank, rerank_table_to_text

# 1. 构建双 ATG
row_atg, col_atg = build_dual_atg(table_text)

# 2. QG-PPR 排序
row_order, col_order = dual_rank(row_atg, col_atg, question)

# 3. 序列化 (列表格式, 供 chain 操作使用)
reranked = rerank_table(table_text, row_order, col_order)

# 或序列化为 Markdown (仅供独立使用)
text = rerank_table_to_text(table_text, row_order, col_order)
```

### 管线集成

- `preprocess_utils/atg_rerank.py` 中的 `ATGReranker` 调用本模块进行双 ATG 重排
- `preprocess_utils/atgo_rerank.py` 中的 `ATGOReranker` / `ATGCReranker` 调用本模块进行单向重排
- `preprocess.py` 和 `preprocess_atgo.py` 作为命令行入口

---

## 对外接口

### graph.py - 数据结构

```python
@dataclass
class CellValueNode:
    value: str
    unique_index: int
    anchor_indices: List[int]

@dataclass
class AnchorNode:
    index: int
    label: str  # "row_i" or "col_j"
    cell_nodes: List[int]

@dataclass
class Triple:
    anchor_idx: int
    edge_attr: str   # 列头 (行中心) or 行索引 (列中心)
    cell_value: str
    row_idx: int
    col_idx: int

class AttributedTableGraph:
    def __init__(self, anchor_type: str = "row")
    def get_triples_by_anchor(self, anchor_idx) -> List[Triple]
    def get_triples_by_edge_attr(self, edge_attr) -> List[Triple]
    def num_triples(self) -> int
    def summary(self) -> str
```

### builder.py - ATG 构建

```python
def build_row_atg(table_text, headers=None) -> AttributedTableGraph
    """构建行中心 ATG: Root -> Row anchors -> Cell nodes (边标注列头)"""

def build_col_atg(table_text, headers=None) -> AttributedTableGraph
    """构建列中心 ATG: Root -> Column anchors -> Cell nodes (边标注行索引)"""

def build_dual_atg(table_text, headers=None) -> Tuple[AttributedTableGraph, AttributedTableGraph]
    """同时构建行中心 + 列中心 ATG"""

def flatten_hierarchical_headers(headers) -> List[str]
    """将多级表头压平为路径字符串"""
```

### qg_ppr.py - QG-PPR 排序

```python
# 默认超参数
DEFAULT_ALPHA = 0.35       # 传送概率
DEFAULT_ITERATIONS = 20   # 幂迭代次数
DEFAULT_V_COL = 1.0       # 列名匹配权重
DEFAULT_V_VAL = 2.0        # 单元格值匹配权重

def qg_ppr(atg, question, alpha=0.35, K=20, ...) -> np.ndarray
    """QG-PPR 完整流程: p0 + A + 幂迭代 -> salience scores"""

def rank_rows(row_atg, question, **kwargs) -> List[int]
    """行重要性排序 (按分数降序)"""

def rank_columns(col_atg, question, **kwargs) -> List[int]
    """列重要性排序 (按分数降序)"""

def dual_rank(row_atg, col_atg, question, **kwargs) -> Tuple[List[int], List[int]]
    """双图 QG-PPR: 同时计算行和列排序"""
```

### serializer.py - 序列化

```python
def rerank_table(table_text, row_order, col_order) -> List
    """按行列排序重构表格, 返回与原始 table_text 相同的列表格式"""

def rerank_table_to_text(table_text, row_order, col_order, caption=None, extra_context=None) -> str
    """按行列排序重构 Markdown 表格文本"""
```

---

## 数据模型

### ATG 双视角架构

```
行中心 ATG (Row-centric):
  Root -> Row anchors -> Cell value nodes
  三元组: <r_i, h_j, c_{i,j}>  (行锚点, 列头边属性, 单元格值)

列中心 ATG (Column-centric):
  Root -> Column anchors -> Cell value nodes
  三元组: <c_j, r_i, c_{i,j}>  (列锚点, 行索引边属性, 单元格值)
```

### QG-PPR 传播权重

| 参数 | 行中心 ATG | 列中心 ATG |
|------|-----------|-----------|
| w_row (同锚点) | 0.6 | 0.4 |
| w_col (同边属性) | 0.4 | 0.6 |

---

## 关键依赖与配置

### 外部依赖

- `numpy` - 矩阵运算 + PageRank 幂迭代
- `pandas` - 表格数据处理 (builder.py)
- `math` - IDF 计算

### 内部依赖

无内部模块依赖，是纯底层计算模块。

---

## 测试与质量

当前无专门测试文件。测试通过 `preprocess_utils/atg_rerank.py` 的集成使用间接验证。

---

## 常见问题 (FAQ)

### Q1: 行中心和列中心 ATG 有什么区别?

**A**: 行中心 ATG 以行为锚点，边属性标注列头，用于计算**行重要性**; 列中心 ATG 以列为锚点，边属性标注行索引，用于计算**列重要性**。两者结合实现双图排序。

### Q2: QG-PPR 的个性化向量如何构建?

**A**: p0(i,j) = v_col * I(h_j in H_q) + v_val * IDF(c_{i,j}) * I(c_{i,j} in C_q)，其中 H_q 是与问题匹配的列名集合，C_q 是与问题匹配的单元格值集合，IDF 基于文档频率计算。

---

## 相关文件清单

- `table_structure/__init__.py` - 模块导出
- `table_structure/graph.py` - ATG 数据结构 (CellValueNode, AnchorNode, Triple, AttributedTableGraph)
- `table_structure/builder.py` - 双 ATG 构建 (build_row_atg, build_col_atg, build_dual_atg)
- `table_structure/qg_ppr.py` - QG-PPR 排序 (qg_ppr, rank_rows, rank_columns, dual_rank)
- `table_structure/serializer.py` - 序列化 (rerank_table, rerank_table_to_text)
