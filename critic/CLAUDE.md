# Critic 模块文档

[← 返回根文档](../CLAUDE.md) > **critic**

> 最后更新：2026-09-21 19:39:28 CST

---

## 变更记录 (Changelog)

### 2026-09-21 (增量更新)
- 复核无结构变化；更新时间戳与导航
- 确认错误树 JSON 仍为 Refine Stage 2 与 RetrieverAgent 的共享记忆源

### 2026-05-21 (精简重构)
- 仅保留管线实际调用的代码路径
- 移除分析脚本清点（非管线代码）

### 2026-05-13 (深度扫描)
- update_tree.py、get_info.py、multiprocess.py 完整深度扫描

---

## 模块职责

Critic 模块提供**批评知识库**，被 Refine Stage 2 通过 `from tools import ...` 调用。

### 核心功能

1. **错误树管理**：层次化错误分类树（`few_shot_critic.json`）
2. **Critic/Judge/Tree 推理**：多进程推理框架（`multiprocess.py`）
3. **CoT 构建**：为各角色构建 Chain-of-Thought prompt（`get_info.py`）
4. **树更新**：Blueprint 模式的动态知识库更新（`update_tree.py`）
5. **Prompt 模板**：三种角色的标准化指令（`instruction.py`）

---

## 管线入口

Critic 模块**不是独立运行的阶段**，而是被 Refine Stage 2 通过 import 调用：

```python
# refine/TableQA/main_tree_based.py / refine/TableFV/main_tree_based.py
from tools import CRITIC_TREE_JSON, read_pkl, critic_tree_init
from tools import tree_exec_one_sample, critic_exec_one_sample, judge_exec_one_sample, update_error_tree
```

消费方还包括 `agents/retriever_agent.py`（`memory_path=CRITIC_TREE_JSON`）。

---

## 工具模块（QA + FV 各一套）

### TableQA: `critic/TableQA/tools/`

| 文件 | 职责 |
|------|------|
| `__init__.py` | 包入口，统一导出 |
| `instruction.py` | Prompt 模板（critic/judge/tree_instruction） |
| `get_info.py` | CoT 构建、few-shot 检索、表格日志重放 |
| `multiprocess.py` | 多进程 Critic/Judge/Tree 推理 |
| `update_tree.py` | 错误树更新（双模式、垂直/水平扩展） |
| `read_pkl.py` | pickle 读取 |
| `few_shot_critic.json` | 错误树 JSON 数据 |
| `few_shot_critic_json.py` | JSON 工具 |
| `few_shot_judge_json.py` | JSON 工具 |
| `few_shot_tree_json.py` | JSON 工具 |

### TableFV: `critic/TableFV/tools/`

结构与 TableQA 相同，额外包含 `update_tree_orig.py`。

---

## 核心 API

### multiprocess.py - 推理入口

```python
def tree_exec_one_sample(sample, llm, llm_options) -> dict
    """Tree 路由推理，返回 {error_route, conclusion}"""

def critic_exec_one_sample(sample, error_route, llm, llm_options,
                           blueprint_only=False, pre_retrieved_few_shot=None) -> dict
    """Critic 推理，支持 blueprint_only 和预检索 few-shot"""

def judge_exec_one_sample(sample, llm, llm_options) -> dict
    """Judge 判断，返回 {conclusion: '[Correct]'/'[Incorrect]'}"""
```

### get_info.py - 信息检索

```python
def get_act_func(name: str) -> Callable
    """操作名 → 执行函数（如 f_select_row_act）"""

def get_table_log(sample, skip_op=[], first_n_op=None) -> tuple[list[dict], list[str]]
    """重放 chain 生成表格状态日志"""

def get_terminal_nodes(input_dict, selected_blueprint=False) -> list
    """递归提取叶子节点"""

def return_error_shot(error_route, few_shot_dict, selected_blueprint=False) -> list
    """根据错误路由检索最多 5 个 few-shot 示例"""

def get_cot_for_critic/judge/tree(sample) -> str/tuple
    """三种角色的 CoT prompt 构建"""
```

### update_tree.py - 错误树更新

```python
def update_error_tree(sample, error_route, error_tree_json, llm, llm_options,
                      lock, use_blueprint=True) -> None
```

**双模式**：
- `use_blueprint=True`（默认）：生成 blueprint 摘要 + 完整内容
- `use_blueprint=False`：直接存储 critic_template 字符串

**扩展策略**：
- **垂直扩展**：同类别下 LLM 判断是否分裂为子类别
- **水平扩展**：新模板沿树遍历，在找不到处插入新分支

> 错误树容量管理预期挂点见 `memory/CLAUDE.md`（主动遗忘，当前未接线）。

---

## 工具模块依赖关系

```
__init__.py
  ├── read_pkl.py
  ├── get_info.py
  │     └── thought.TableQA.utils.helper.table2string
  │     └── thought.TableQA.operations.*
  ├── multiprocess.py
  │     ├── get_info.py
  │     └── instruction.py
  ├── instruction.py          (纯常量)
  ├── update_tree.py
  │     └── get_info.py
  │     └── thought.TableQA.utils.helper.table2string
  └── few_shot_*_json.py
```

> 注意：`thought.TableQA.utils.helper.table2string` 是跨模块核心依赖，被多个 critic 工具引用。

---

## Prompt 模板（instruction.py）

三种角色各有独立的 instruction 字符串，核心输出格式：
- Critic：`Conclusion: [Incorrect] Step <NUM>`
- Judge：`Conclusion: [Correct]` 或 `Conclusion: [Incorrect]`
- Tree：`Conclusion: [Incorrect] (ERROR ROUTE)` 或 `Conclusion: [Incorrect] (random)`

---

## 测试与质量

无独立测试。验证方式：Refine 阶段端到端准确率（`cal_acc.py`）。

---

## 相关文件清单

### TableQA

- `critic/TableQA/tools/__init__.py` - 包入口
- `critic/TableQA/tools/instruction.py` - Prompt 模板
- `critic/TableQA/tools/get_info.py` - CoT 构建
- `critic/TableQA/tools/multiprocess.py` - 多进程推理
- `critic/TableQA/tools/update_tree.py` - 树更新
- `critic/TableQA/tools/read_pkl.py` - pickle 读取
- `critic/TableQA/tools/few_shot_critic.json` - 错误树数据

### TableFV

- `critic/TableFV/tools/` - 结构同 TableQA，另含 `update_tree_orig.py`
