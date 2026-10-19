# Table Structure 模块文档

[根目录](../CLAUDE.md) > **table_structure**

> 最后更新：2026-09-21 19:39:28 CST

---

## 变更记录 (Changelog)

### 2026-09-21 (增量更新)
- 补录 5 个新文件：column_similarity.py、column_smoothing.py、context.py、graph_bias.py、value_similarity.py
- 补录 qg_ppr.py 新 API：`*_with_scores` 系列、`power_iteration`、`rank_row_atg_columns_with_scores`
- 补录 builder.py 的 `merge_similar_values` 相似值合并
- 标注已知问题：`context.build_atg_context` 被外部引用但不存在

### 2026-06-09 (新建)
- 初始化文档，扫描 graph.py、builder.py、qg_ppr.py、serializer.py

---

## 模块职责

Table Structure 模块实现**双 ATG (Attributed Table Graph) + QG-PPR (Question-Guided Personalized PageRank)** 表格结构增强，用于表格行列重排，并支撑主线的 P1/P2/P3 三类实验开关。

参考论文: Beyond Linearization: Attributed Table Graphs for Table Reasoning (arXiv:2601.08444)

### 核心功能

1. **双 ATG 构建**: 行中心 ATG (Row-centric) + 列中心 ATG (Column-centric)，可选相似值节点合并
2. **QG-PPR 重要性排序**: 基于问题的个性化 PageRank 计算行列重要性（带分数输出版本供图提示使用）
3. **P1 gamma 平滑**: `column_smoothing.py` 列间相似度混合传播矩阵
4. **P2 图提示**: `context.py` 生成 Refine 可消费的 `[Graph Hint]` 文本
5. **P3 候选偏置**: `graph_bias.py` 解析 LLM 候选并施加图先验重排
6. **序列化**: 重排后的表格转为列表格式或 Markdown 文本

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

- `preprocess_utils/atgo_rerank.py`：`ATGOReranker` / `ATGCReranker`（主线，run_QA.sh/run_FV.sh 使用）调用行/列单向重排，消费 `rank_rows_with_scores` 等带分数 API 与 `build_smoothed_propagation_matrix`
- `preprocess_utils/atg_rerank.py`：`ATGReranker`（双 ATG 对照版，当前有 ImportError，见 FAQ Q3）
- `preprocess.py` / `preprocess_atgo.py` 为命令行入口
- `refine/*/operations/{select_row,select_column,final_query}.py` 通过 `context.get_graph_hint` 消费 P2 提示

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
def build_row_atg(table_text, headers=None, merge_similar_values=False) -> AttributedTableGraph
    """构建行中心 ATG: Root -> Row anchors -> Cell nodes (边标注列头)"""

def build_col_atg(table_text, headers=None, merge_similar_values=False) -> AttributedTableGraph
    """构建列中心 ATG: Root -> Column anchors -> Cell nodes (边标注行索引)"""

def build_dual_atg(table_text, headers=None, merge_similar_values=False) -> Tuple[ATG, ATG]

def flatten_hierarchical_headers(headers) -> List[str]
    """将多级表头压平为路径字符串"""
```

`merge_similar_values=True` 时借助 `value_similarity.py` 在同列/同行内合并近似值节点（阈值见 `should_merge_values`，默认 0.85）。

### qg_ppr.py - QG-PPR 排序

```python
# 默认超参数
DEFAULT_ALPHA = 0.35       # 传送概率
DEFAULT_ITERATIONS = 20   # 幂迭代次数
DEFAULT_V_COL = 1.0       # 列名匹配权重
DEFAULT_V_VAL = 2.0        # 单元格值匹配权重

def qg_ppr(atg, question, alpha=0.35, K=20, ...) -> np.ndarray
def power_iteration(...)                       # 幂迭代（可独立调用）
def rank_rows(row_atg, question, **kwargs) -> List[int]
def rank_rows_with_scores(...) -> Tuple[List[int], Dict[int, float]]   # ★ P2 用
def rank_columns(col_atg, question, **kwargs) -> List[int]
def rank_columns_with_scores(...) -> Tuple[List[int], Dict]            # ★
def rank_row_atg_columns_with_scores(...) -> Tuple[List[int], Dict]    # ★ 行图上排列
def dual_rank(row_atg, col_atg, question, **kwargs) -> Tuple[List[int], List[int]]
def dual_rank_with_scores(...) -> Tuple[List[int], List[int], Dict, Dict]
```

### column_similarity.py - 列相似度（P1 输入）

```python
class ColumnSimilarityComputer:
    """计算列间相似度矩阵，供 build_smoothed_propagation_matrix 使用"""
```

### column_smoothing.py - P1 gamma 平滑

```python
def build_smoothed_propagation_matrix(
    atg, column_similarity, w_row, w_col, w_val=DEFAULT_W_VAL, gamma=0.1
) -> np.ndarray
    """base 与列间平滑矩阵按 (1-gamma)/gamma 凸组合；gamma=0 时严格退化为 baseline"""
```

### context.py - 图提示（P2）

```python
def build_relevance_context(sample, top_k=3, **_) -> str
    """从 sample 的 row/col_relevance_scores 生成 '[Graph Hint] ... [End Graph Hint]' 文本"""

def get_graph_hint(sample, top_k=3) -> str
    """按 graph_metadata.inject_graph_hint 开关返回缓存的 graph_hint 或现算提示"""
```

### graph_bias.py - 候选偏置（P3）

```python
def parse_column_candidate(candidate: str) -> List[str]   # 解析 LLM 输出的列候选
def parse_row_candidate(candidate: str) -> List[int]      # 解析行候选
def canonicalize_candidate(candidate, candidate_type) -> str
def apply_graph_bias(...)                                  # 用图相关性分数重排候选
```

### serializer.py - 序列化

```python
def rerank_table(table_text, row_order, col_order) -> List
def rerank_table_to_text(table_text, row_order, col_order, caption=None, extra_context=None) -> str
```

### value_similarity.py - 值相似度工具

```python
def normalize_value(value: str) -> str
def value_similarity(left: str, right: str) -> float
def should_merge_values(left, right, threshold=0.85) -> bool
def unique_preserve_order(values) -> List[str]
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

### 样本级图字段（由 atgo_rerank 写入 JSONL）

`graph_metadata`（schema_version/rerank_mode/gamma/inject_graph_hint/bias_weight）、`row_relevance_scores`、`col_relevance_scores`、`graph_hint`、`atgo_reranked_table` / `atgc_reranked_table`、`_atgo_order_changed` / `_atgc_order_changed`。

---

## 关键依赖与配置

- `numpy` - 矩阵运算 + PageRank 幂迭代
- `pandas` - 表格数据处理 (builder.py)
- `math` - IDF 计算
- 内部无对其他项目模块的依赖，是纯底层计算模块（被 preprocess_utils、refine operations、scripts 反向依赖）

---

## 测试与质量

当前无专门测试文件。验证途径：`run_graph_experiment_matrix.py` 消融（baseline / P1 / P2 / P1+P2）+ 端到端准确率对比。

---

## 常见问题 (FAQ)

### Q1: 行中心和列中心 ATG 有什么区别?

**A**: 行中心 ATG 以行为锚点，边属性标注列头，用于计算**行重要性**; 列中心 ATG 以列为锚点，边属性标注行索引，用于计算**列重要性**。两者结合实现双图排序。

### Q2: QG-PPR 的个性化向量如何构建?

**A**: p0(i,j) = v_col * I(h_j in H_q) + v_val * IDF(c_{i,j}) * I(c_{i,j} in C_q)，其中 H_q 是与问题匹配的列名集合，C_q 是与问题匹配的单元格值集合，IDF 基于文档频率计算。

### Q3: 已知问题 - `build_atg_context` 不存在

**A**: `preprocess_utils/atg_rerank.py:20` 与 `scripts/rerun_error_results_with_atg_context.py:29` 执行 `from table_structure.context import build_atg_context`，但当前 `context.py` 只定义了 `build_relevance_context` 与 `get_graph_hint`。这两个入口会 ImportError。主线 `preprocess_atgo.py`（ATGO/ATGC）不引用该符号，不受影响。修复方向：在 context.py 中补回该函数或改调用方为 `build_relevance_context`。

---

## 相关文件清单

- `table_structure/__init__.py` - 模块导出（`__all__` 含全部公共 API）
- `table_structure/graph.py` - ATG 数据结构 (CellValueNode, AnchorNode, Triple, AttributedTableGraph)
- `table_structure/builder.py` - 双 ATG 构建 + 相似值合并
- `table_structure/qg_ppr.py` - QG-PPR 排序（含 `*_with_scores` 变体、power_iteration）
- `table_structure/column_similarity.py` - 列相似度矩阵
- `table_structure/column_smoothing.py` - P1 gamma 平滑传播矩阵
- `table_structure/context.py` - P2 图提示构造
- `table_structure/graph_bias.py` - P3 候选解析与偏置
- `table_structure/value_similarity.py` - 值归一化/相似度/合并判定
- `table_structure/serializer.py` - 序列化 (rerank_table, rerank_table_to_text)
