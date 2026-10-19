# Thought 模块文档

[← 返回根文档](../CLAUDE.md) > **thought**

> 最后更新：2026-09-21 19:39:28 CST

---

## 变更记录 (Changelog)

### 2026-09-21 (增量更新)
- 更正 main.py 参数清单：QA 与 FV 现均支持 `--n_proc` / `--chunk_size`
- 补录 ATGO 重排数据（`test_reranked_row.jsonl`）作为主线输入、结果目录 `_p12` 后缀约定
- 补录 token 日志（`set_token_log_dir`）与 clarifier 结果落盘说明

### 2026-05-21 (精简重构)
- 仅保留管线实际调用的代码路径

### 2026-05-13 (深度扫描)
- 6 种 Operations 完整深度扫描

---

## 模块职责

Thought 模块是 **Stage 1 初始推理**，基于 Chain-of-Table 的动态推理链生成第一版答案。

### TableQA 和 TableFV 各有一套完整实现

- `thought/TableQA/` - WikiTableQuestions 任务（数据目录 `data/wikitq/`）
- `thought/TableFV/` - TabFact 任务（数据目录 `data/tabfact/`）

两者结构相同：`main.py` + `operations/`（6 个操作）+ `utils/`（5 个工具）+ `third_party/`

---

## 管线入口

### run_QA.sh 调用（当前版本）

```bash
python thought/TableQA/main.py \
  --dataset_path "$ATGO_ROW_DATA" \          # 主线：test_reranked_row.jsonl（ATGO 行重排后）
  --thought_results_dir "${THOUGHT_RESULTS}${GRAPH_SUFFIX}" \
  --base_url $base_url \
  --openai_api_key $openai_api_key \
  --model_name $model_name \
  --first_n $first_n \
  --n_proc $n_proc \
  --chunk_size $chunk_size \
  --use_clarifier $([ "$MODE" = "new" ] && echo "True" || echo "False")
```

### run_FV.sh 调用

参数结构与 QA 相同，`--dataset_path "$ATGO_ROW_DATA"` 指向 `thought/TableFV/data/tabfact/test_reranked_row.jsonl`。

### main.py 关键签名

```python
def main(
    dataset_path="thought/TableQA/data/wikitq/test_lower.jsonl",
    thought_results_dir="",
    base_url="", openai_api_key="EMPTY", model_name="qwen2.5-72b-instruct",
    first_n=-1, n_proc=8, chunk_size=4,
    use_clarifier: bool = True,
)
```

- `openai_api_key` 支持 `<env:OPENAI_API_KEY>` 占位符自动读环境变量
- `thought_results_dir` 缺省时按 `results/{new|orig}/thought/wikitq` 自动推导
- LLM 初始化后调用 `set_token_log_dir(thought_results_dir, run_tag="wikitq_thought")` 启用 token 计量日志
- `use_clarifier=True` 时：`ClarifierAgent.clarify_batch(dataset)` → 结果保存到 `{thought_results_dir}/clarifier/`

> 结果目录约定：P1+P2 开启时追加 `_p12` 后缀（如 `..._orig_p12`），`MODE=orig` 时追加 `_orig`。

---

## 数据文件（只读）

| 路径 | 说明 |
|------|------|
| `thought/TableQA/data/wikitq/test_lower.jsonl` | WikiTQ 原始测试集（小写化） |
| `thought/TableQA/data/wikitq/tagged_data/*.tagged` | 原始 tagged 表格（pristine-seen/unseen、training） |
| `thought/TableQA/data/wikitq/test_analyzed.jsonl` | Stage 0.5 TableAnalyzer 输出 |
| `thought/TableQA/data/wikitq/test_reranked*.jsonl` | ATG/ATGO 重排输出（`.cache` 为旁路缓存） |
| `thought/TableFV/data/tabfact/test.jsonl` 等 | TabFact 对应文件（raw2clean / test / analyzed / flatten） |

> `.gitignore` 忽略 `*_rerank_cache/`、`test_reranked*.jsonl`、`error_data.jsonl` 等生成物。

---

## 表格操作（Operations）

### 统一接口模式

所有操作遵循**两阶段模式**：
1. `*_func`（规划阶段）：调用 LLM 决定操作参数
2. `*_act`（执行阶段）：对 `table_info` 执行表格变换

**置信度聚合**：多候选概率累加，取 top-1
**失败安全**：`skip_op` + `failure_table_info`，单步失败不崩溃

### 操作状态转换

```
<init> → [add_column, select_row, select_column, group_column, sort_column]
add_column → [select_row, select_column, group_column, sort_column, <END>]
select_row → [select_column, group_column, sort_column, <END>]
select_column → [group_column, sort_column, <END>]
group_column → [sort_column, <END>]
sort_column → [<END>]
```

### 6 种操作

| 操作 | 文件 | 关键特征 |
|------|------|----------|
| **select_column** | `operations/select_column.py` | 列转置 prompt、列并集（`union_num=2`） |
| **select_row** | `operations/select_row.py` | 1-based→0-based 转换、并集策略 |
| **sort_by** | `operations/sort_by.py` | 多类型排序、去重检测 |
| **add_column** | `operations/add_column.py` | 增量逐行填充、5 重安全检查 |
| **group_by** | `operations/group_by.py` | 合理性检查（唯一值 > 0.8 拒绝） |
| **final_query** | `operations/final_query.py` | 3 套 few-shot 模板、无 `*_act` |

### `__init__.py` 导出

```python
from .add_column import add_column_func, add_column_act
from .group_by import group_column_func, group_column_act
from .select_column import select_column_func, select_column_act
from .select_row import select_row_func, select_row_act
from .sort_by import sort_column_func, sort_column_act
from .final_query import simple_query
```

---

## 工具模块（Utils）

| 文件 | 职责 |
|------|------|
| `chain.py` | 推理链执行（`dynamic_chain_exec_with_cache_mp`、`fixed_chain_exec_mp`） |
| `llm.py` | LLM 封装 + Token 统计（`set_token_log_dir`） |
| `helper.py` | `table2string()`、`table2df()`、`NoIndent` 等 |
| `evaluate.py` | `wikitq_match_func_for_samples()`、`tabfact_match_func_for_samples()` |
| `load_data.py` | `load_wikitq_dataset()`、`load_tabfact_dataset()` |

### 第三方

- `third_party/select_column_row_prompts/select_column_row_prompts.py` - `select_column_demo` / `select_row_demo` few-shot 模板（QA 目录含 LICENSE）

---

## sys.path 设置

`main.py` 中通过以下方式设置模块搜索路径：
```python
sys.path.insert(0, 'thought/TableQA')   # 或 'thought/TableFV' → from utils/operations import ...
sys.path.insert(0, '.')                 # 根目录，用于 import agents
sys.path.insert(0, 'critic/TableQA')    # 或 'critic/TableFV' → from tools.read_pkl import ...
```

---

## 数据流

```
输入 JSONL（ATGO 行重排后，可含 table_analysis / atgo_reranked_table / graph 字段）
  ↓
[可选] ClarifierAgent.clarify_batch() → sample['clarifier'] → clarifier/case_dict_{id}.pkl
  ↓
dynamic_chain_exec_with_cache_mp()
  ├── LLM 动态选择操作序列
  ├── 执行 operations（select_row, group_by, ...）
  └── 生成 chain + pred_answer
  ↓
fixed_chain_exec_mp()（Simple query 生成最终答案）
  ↓
输出 pickle 到 thought_results_dir（final_result.pkl / error_results.pkl / cache/）
```

---

## 相关文件清单

### TableQA

- `thought/TableQA/main.py` - 入口
- `thought/TableQA/operations/` - 6 个操作文件 + `__init__.py`
- `thought/TableQA/utils/` - chain.py, llm.py, helper.py, evaluate.py, load_data.py
- `thought/TableQA/third_party/` - select_column_row_prompts（含 LICENSE）
- `thought/TableQA/tools/read_pkl.py` - pickle 读取

### TableFV

- `thought/TableFV/main.py` - 入口
- `thought/TableFV/merge_cache_to_final.py` - 缓存合并辅助
- `thought/TableFV/operations/` - 6 个操作文件 + `__init__.py`
- `thought/TableFV/utils/` - chain.py, llm.py, helper.py, evaluate.py, load_data.py
- `thought/TableFV/third_party/` - select_column_row_prompts（含 LICENSE）
