# Refine 模块文档

[← 返回根文档](../CLAUDE.md) > **refine**

> 最后更新：2026-09-21 19:39:28 CST

---

## 变更记录 (Changelog)

### 2026-10-23 (超长降级改进)
- `TableQA/TableFV/utils/llm.py`：上下文超长错误改为「中段裁剪重试一次 → 仍失败才 PLACEHOLDER」，新增事件计数日志（`context_overflow_placeholder` / `moderation_placeholder` 写入 token 日志，`collect_token_usage` 聚合）

### 2026-09-21 (增量更新)
- 补录 P2 图提示消费：operations 三个操作通过 `table_structure.context.get_graph_hint` 注入 `[Graph Hint]`
- 补录 chain.py 对 `atgo_reranked_table` 的表格字段回退链
- 更新 run_QA.sh/run_FV.sh 调用（含 GRAPH_SUFFIX、ENABLE_P1/P2）

### 2026-05-21 (精简重构)
- 仅保留管线实际调用的代码路径

### 2026-05-13
- Stage 0.5 Hint Consumption 文档

---

## 模块职责

Refine 模块是 **Stage 2 修正阶段**，通过 Controller 驱动的 Critic/Judge/Tree 协作，对 Thought 的错误答案进行迭代修正。

### TableQA 和 TableFV 各有一套实现

- `refine/TableQA/` - 含约束模块（constraint_*.py）
- `refine/TableFV/` - 基础版本

---

## 管线入口

### run_QA.sh 调用（当前版本）

```bash
python refine/TableQA/main_tree_based.py \
  --thought_results_dir "${THOUGHT_RESULTS}${GRAPH_SUFFIX}" \
  --refine_results_dir "${REFINE_RESULTS}${GRAPH_SUFFIX}" \
  --base_url $base_url \
  --openai_api_key $openai_api_key \
  --model_name $model_name \
  --first_n $first_n \
  --n_proc $n_proc \
  --chunk_size $chunk_size \
  --use_clarifier $([ "$MODE" = "new" ] && echo "True" || echo "False")
```

- `GRAPH_SUFFIX`：P1+P2 开启时为 `_p12`，否则为空
- `run_FV.sh` 以完全相同的参数调用 `refine/TableFV/main_tree_based.py`

---

## sys.path 设置

```python
# main_tree_based.py
sys.path.append('critic/TableQA')   # 或 'critic/TableFV' → from tools import ...
sys.path.append('refine/TableQA')   # 或 'refine/TableFV' → from utils.xxx / from operations import *
sys.path.append('.')                # 根目录 → from agents / from table_structure import ...
```

---

## Controller 架构（★ 核心）

**文件**：`refine/TableQA/utils/controller.py`（TableFV 也有对应副本）

### 核心组件

```python
@dataclass
class ControllerState:
    sample, question, chain, conclusion
    iteration: int = 0
    error_route: Optional[str] = None
    action_history: List[str]

class ControllerAction(Enum):
    STOP / EXECUTE_TREE / DIAGNOSE_BP / DIAGNOSE_FS / REFINE_CHAIN / REFINE_QUERY / UPDATE_TREE

@dataclass
class Decision:
    action: ControllerAction
    reason: str
    confidence: float
    source: str  # "rule" | "llm"
```

### 主循环

```python
def controller_main_loop(sample, llm, llm_options, max_iterations=5,
                         cache_dir=None, use_clarifier=True, thought_results_dir=None) -> Dict
```

**流程**：
1. 加载 Clarifier 信息
2. Judge 初始评判
3. 主循环：`decide()` → `execute()` → 检查 STOP
4. 一次迭代：`EXECUTE_TREE → DIAGNOSE_BP/FS → REFINE_CHAIN/QUERY`

### ActionExecutor

```python
class ActionExecutor:
    def __init__(self, llm, llm_options):
        from agents import RetrieverAgent  # ★ 唯一使用 agents 的地方
        self.retriever = RetrieverAgent(llm=llm, memory_path=CRITIC_TREE_JSON)

    def execute(self, state, decision) -> ControllerState:
        # EXECUTE_TREE → tree_exec_one_sample → error_route
        # DIAGNOSE_BP/FS → critic_exec_one_sample（支持预检索 few-shot）
        # REFINE_CHAIN → dynamic_chain_exec_one_sample + simple_query_cot_original
        # REFINE_QUERY → simple_query_with_critic
        # UPDATE_TREE → update_error_tree
```

---

## Stage 0.5 Hint Consumption（table_analysis）

当 Stage 0.5 启用时，`sample["table_analysis"]` 通过 JSONL 数据集透传到 Refine。

| Operation | Hint 字段 | 注入位置 |
|-----------|----------|---------|
| `select_row` | `column_normalizations` | `build_prompt()` 函数 |
| `sort_by` | `format_normalizations` | `build_prompt()` 函数 |
| `final_query` | `answer_format_hint` | 4 个 query 函数 |

**向后兼容**：所有注入使用 `sample.get("table_analysis")` 安全访问，Stage 0.5 未启用时自动跳过。

---

## Stage 0.7 P2 Graph Hint Consumption（★ 新增）

当 ATGO 预处理开启 `inject_graph_hint=True`（P2）时，样本携带 `graph_metadata` / `row_relevance_scores` / `col_relevance_scores` / `graph_hint` 字段进入 Refine。

| 消费点 | 方式 |
|--------|------|
| `operations/select_row.py` | `get_graph_hint(sample)` 非空则追加到 prompt（前缀 `\n`） |
| `operations/select_column.py` | 同上（追加到 prompt 尾部） |
| `operations/final_query.py` | 同上（追加空行分隔的提示块） |
| `utils/chain.py` | 表格字段回退链：`table_text → atg_reranked_table → atgo_reranked_table → ...`（约第 109/132/168 行），保证重排表格被链执行使用 |

提示文本由 `table_structure/context.py` 生成，形如：

```
[Graph Hint]
Relevant columns: <top-k 列名>
Relevant rows: row 3, row 7, ...
[End Graph Hint]
```

**向后兼容**：`get_graph_hint` 检查 `graph_metadata.inject_graph_hint` 开关，P2 关闭时返回空串，prompt 不变。

---

## 工具模块（Utils）

### 共有（QA + FV）

| 文件 | 职责 |
|------|------|
| `controller.py` | Controller 状态机 + ActionExecutor + 主循环 |
| `chain.py` | 链执行引擎（含 Critic 反馈引导重执行、重排表格字段回退） |
| `llm.py` | LLM 封装 + Token 统计 |
| `helper.py` | `table2string()`、`table2df()` 等 |
| `evaluate.py` | 准确率评估函数 |
| `extract_step.py` | 错误步骤解析 |
| `read_pkl.py` | pickle 读取 |

### QA 独有

| 文件 | 职责 |
|------|------|
| `validator.py` | 验证器 |
| `verifier.py` | 校验器 |
| `router.py` | 路由变体 |
| `constraint_state.py` | 四类约束管理 |
| `constraint_induction.py` | Critic 反馈 → 约束转换 |
| `constraint_aware_chain.py` | 约束感知链执行 |
| `diff_critic/` | diff 批评子包（diff_bundle / evidence_pack / operation_trace / step_localizer） |

### FV utils

`controller.py`、`chain.py`、`llm.py`、`helper.py`、`evaluate.py`、`extract_step.py`、`read_pkl.py`（与 QA 同名文件结构对齐，无约束模块）。

---

## Operations

6 种操作与 Thought 阶段同名，但 Refine 版本额外支持两类 hint 注入：

- `select_row.py` - 行选择（★ `column_normalizations` + P2 graph hint）
- `sort_by.py` - 排序（★ `format_normalizations`）
- `final_query.py` - 最终查询（★ `answer_format_hint` + P2 graph hint）
- `select_column.py` - 列选择（★ P2 graph hint）
- `group_by.py` - 分组聚合
- `add_column.py` - 添加列

QA 另有入口 `main_constraint_based.py`（约束感知实验入口，主线不使用）。

---

## 相关文件清单

### TableQA

- `refine/TableQA/main_tree_based.py` - 入口（主线）
- `refine/TableQA/main_constraint_based.py` - 约束实验入口
- `refine/TableQA/utils/controller.py` - Controller（★）
- `refine/TableQA/utils/` - chain, llm, helper, evaluate, extract_step, read_pkl, validator, verifier, router, constraint_*, diff_critic/
- `refine/TableQA/operations/` - 6 个操作 + `__init__.py`
- `refine/TableQA/third_party/` - select_column_row_prompts（含 LICENSE）

### TableFV

- `refine/TableFV/main_tree_based.py` - 入口
- `refine/TableFV/utils/` - controller, chain, llm, helper, evaluate, extract_step, read_pkl
- `refine/TableFV/operations/` - 6 个操作 + `__init__.py`
- `refine/TableFV/third_party/` - select_column_row_prompts
