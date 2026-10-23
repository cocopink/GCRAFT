# Preprocess Utils 模块文档

[← 返回根文档](../CLAUDE.md) > **preprocess_utils**

> 最后更新：2026-09-21 19:39:28 CST

---

## 变更记录 (Changelog)

### 2026-10-20 (P3 下线)
- `atgo_rerank.py` 移除 `bias_weight` 参数与缓存名 bias 段（随 P3 下线；缓存名变更使旧缓存一次性重算，仅 CPU 成本）

### 2026-09-21 (增量更新)
- 补录 atgo_rerank.py（ATGOReranker / ATGCReranker，主线 Stage 0.7）
- 补录 atg_rerank.py（ATGReranker，双 ATG 对照版，含 ImportError 隐患标注）

### 2026-05-21 (精简重构)
- 无内容变更，仅更新时间戳和导航

### 2026-05-10
- 初始化，深度扫描全部 5 个子模块

---

## 模块职责

零 LLM 成本的表格预处理工具包：Stage 0（Flatten 复合表头拆分）、Stage 0.5（TableAnalyzer 分析）、Stage 0.7（ATG 行列重排，主线）。

---

## 对外接口

### detect.py - 复合表头检测

```python
SEPARATOR_PRIORITY = ['/', '\\n', '\n', ' - ', '-', '|', ',']

def has_compound_headers(table: List[List]) -> bool
def find_compound_columns(table: List[List]) -> List[int]
def get_separator(header_cell: str) -> str
```

### flatten.py - 表格拆分

```python
def split_cell_value(value, primary_separator, expected_count) -> List[str]
    """三级降级策略：主分隔符 → 空格 → 兜底复制"""

def flatten_table(table) -> Tuple[List[List], Dict]
    """拆分复合表头表格，返回 (flattened_table, metadata)"""

def flatten_dataset(dataset) -> List[Dict]
    """批量处理，支持 'table_text'(TableQA) 和 'table'(TableFV) 键"""
```

### column_norm.py - 列值标准化

```python
def fuzzy_match_values(question_keyword, column_values, threshold=0.8) -> List[str]
    """基于 pylcs LCS 的模糊匹配"""

def normalize_column(question_keywords, table_text, column_idx) -> Optional[ColumnNormalization]
def extract_numeric_from_mixed(value) -> Optional[float]
def detect_mixed_format_column(values) -> bool
def normalize_column_format(values) -> Dict[str, Optional[float]]
```

### header_tree.py - 复合表头树

```python
class HeaderTree:
    @classmethod
    def build(cls, table_text: List[List]) -> 'HeaderTree'
    def parse_compound_cells(self, table_text) -> Dict[int, List[Dict]]
    def serialize_for_prompt(self, include_cells=True) -> str
```

### cache.py - JSONL 缓存

```python
def check_cache(cache_path) -> bool
def load_cache(cache_path) -> List[Dict]
def save_cache(samples, cache_path) -> None
```

### atgo_rerank.py - ATG 单向重排（★ 主线 Stage 0.7）

```python
GRAPH_SCHEMA_VERSION = 1

class ATGOReranker:
    """仅行重排：只构建行中心 ATG，列序保持原样"""
    def __init__(self, cache_dir=None, gamma=0.0, inject_graph_hint=False)
    def rerank_batch(dataset)   # 写入 atgo_reranked_table / graph_metadata / *_relevance_scores / graph_hint

class ATGCReranker:
    """仅列重排：只构建列中心 ATG，行序保持原样"""
    # 参数与 ATGOReranker 相同，写入 atgc_reranked_table 等
```

- 缓存路径编码实验开关：`{id}.schema-1.mode-row.gamma-0.1.hint-1.pkl`（开关组合变化自动失效）
- 输出统计 `Changed samples: N/M`，并写出 `_changed.jsonl` 精简文件（仅顺序改变的样本）
- 命令行入口：根目录 `preprocess_atgo.py --mode row|col`

### atg_rerank.py - 双 ATG 重排（对照版）

```python
class ATGReranker:
    """双 ATG + QG-PPR 行列同时重排"""
    def __init__(self, cache_dir=None)
    def rerank_batch(dataset)   # 写入 atg_reranked_table；构建时 merge_similar_values=True 附加 atg_context
```

命令行入口：根目录 `preprocess.py`。**注意**：本文件第 20 行 `from table_structure.context import build_atg_context` 引用当前不存在的符号，运行会 ImportError（详见 table_structure/CLAUDE.md FAQ Q3）；主线不经过此文件。

---

## 依赖关系

```
preprocess_utils/__init__.py
  ├── detect.py        (被 flatten.py、header_tree.py 依赖)
  ├── flatten.py       (依赖 detect.py)
  ├── cache.py
  ├── header_tree.py   (依赖 detect.py)
  ├── column_norm.py   (被 agents/table_analyzer.py 依赖)
  ├── atg_rerank.py    (依赖 table_structure: build_dual_atg, dual_rank, rerank_table; context.build_atg_context* )
  └── atgo_rerank.py   (依赖 table_structure: rank_*_with_scores, rerank_table, build_relevance_context,
                        column_smoothing, ColumnSimilarityComputer)
```

外部依赖：`tqdm`、`numpy`（经 table_structure 间接）、`pylcs`。

---

## 测试

历史文档引用的 `tests/test_flatten.py`、`tests/test_column_norm.py`、`tests/test_header_tree.py` 当前不存在（`/tests/` 被 .gitignore 忽略且目录缺失）。现行验证依赖端到端管线与 `run_graph_experiment_matrix.py` 消融对比。

---

## 相关文件

- `preprocess_utils/__init__.py`
- `preprocess_utils/detect.py`
- `preprocess_utils/flatten.py`
- `preprocess_utils/cache.py`
- `preprocess_utils/column_norm.py`
- `preprocess_utils/header_tree.py`
- `preprocess_utils/atg_rerank.py` - 双 ATG 重排（对照）
- `preprocess_utils/atgo_rerank.py` - ATGO/ATGC 单向重排（★ 主线）
- `preprocess.py` - 双 ATG 命令行入口
- `preprocess_atgo.py` - ATGO/ATGC 命令行入口（主线）
