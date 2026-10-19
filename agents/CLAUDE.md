# Agents 模块文档

[← 返回根文档](../CLAUDE.md) > **agents**

> 最后更新：2026-09-21 19:39:28 CST

---

## 变更记录 (Changelog)

### 2026-09-21 (增量更新)
- 修正测试章节：`tests/test_table_analyzer.py` 当前不存在（/tests/ 被忽略且目录缺失）
- 补充 `merge_cache_to_final` 无关说明移除；确认 9 个未接线 agent 名单不变
- 更新跨模块依赖说明（table_analyzer → preprocess_utils；judge → critic/thought）

### 2026-05-21 (精简重构)
- 标注管线实际使用的 agent（ClarifierAgent + RetrieverAgent + TableAnalyzer）
- 标注 `__init__.py` 加载但管线未使用的 agent
- 移除 DisputeHandler 详细文档（不在管线中）

### 2026-05-10
- 新增 TableAnalyzer 文档

---

## 模块职责

Agents 模块提供多智能体框架基础类和各专业化智能体实现。

### 管线实际使用情况

| Agent | 文件 | 使用阶段 | 管线调用 |
|-------|------|---------|---------|
| **ClarifierAgent** | `clarifier_agent.py` | Stage 1 | `thought/*/main.py` |
| **RetrieverAgent** | `retriever_agent.py` | Stage 2 | `refine/*/utils/controller.py` |
| **TableAnalyzer** | `table_analyzer.py` | Stage 0.5 | `preprocess.py` |
| BaseAgent | `multi_agent_framework.py` | - | 被上述 agent 继承 |

### `__init__.py` 加载但未使用

以下 agent 被 `agents/__init__.py` 导入加载，但在 `run_QA.sh` / `run_FV.sh` 管线中**从未被调用**：

- `reasoner_agent.py` - InitialReasoner
- `judge_agent.py` - JudgeAgent
- `critic_agent.py` - CriticAgent
- `refiner_agent.py` - RefinerAgent
- `validator_agent.py` - ValidatorAgent
- `curator_agent.py` - CuratorAgent
- `dispute_handler.py` - DisputeHandler
- `memory_integration.py` - MemoryEvolutionManager
- `active_forgetting.py` - ActiveForgettingManager（与 `memory/active_forgetting.py` 同名异实现）

> 注意：`__init__.py` 导出所有 agent，因此 import `agents` 包时会加载上述文件。JudgeAgent 等内部引用了 `critic/TableQA/tools/*` 和 `thought/TableQA/utils/helper.py`，形成了额外的跨模块依赖。

---

## 对外接口

### BaseAgent 基类

**文件**：`multi_agent_framework.py`

```python
class BaseAgent:
    def __init__(self, agent_type: AgentType, llm=None): ...
    def process(self, sample, context=None): ...  # 抽象方法

class AgentType(Enum):
    CLARIFIER = "clarifier"
    REASONER = "reasoner"
    JUDGE = "judge"
    CRITIC = "critic"
    REFINER = "refiner"
    VALIDATOR = "validator"
    CURATOR = "curator"
    RETRIEVER = "retriever"
```

---

## 管线使用的 Agent 详解

### 1. ClarifierAgent（Stage 1）

**文件**：`clarifier_agent.py`

**职责**：从表格中提取 Schema 锚点（列名、实体、单位、关键词）

```python
from agents.clarifier_agent import ClarifierAgent

clarifier = ClarifierAgent(llm=llm)
sample = clarifier.clarify_sample(sample)
# sample['clarifier'] = {
#     'headers': [...], 'entities': {...}, 'units': {...}, 'keywords': [...]
# }
```

**管线集成**：
- `thought/TableQA/main.py`：`use_clarifier=True` 时调用 `clarify_batch(dataset)`
- 结果保存到 `{thought_results_dir}/clarifier/case_dict_{id}.pkl`
- Refine 阶段通过 `load_clarifier_info()` 加载

### 2. RetrieverAgent（Stage 2）

**文件**：`retriever_agent.py`

**职责**：从错误树中检索 Blueprint 和 Few-shot 案例

```python
from agents import RetrieverAgent

retriever = RetrieverAgent(llm=llm, memory_path=CRITIC_TREE_JSON)
result = retriever.retrieve(sample, error_route="sub-table error", k=3)
# result = {'blueprint': '...', 'few_shot_examples': [...]}
```

**管线集成**：
- 在 `refine/*/utils/controller.py` 的 `ActionExecutor.__init__()` 中初始化
- Controller 决策 DIAGNOSE_BP/FS 时调用

### 3. TableAnalyzer（Stage 0.5）

**文件**：`table_analyzer.py`

**职责**：零 LLM 成本的表格结构分析，输出 hint 注入 Refine prompt

```python
from agents.table_analyzer import TableAnalyzer

analyzer = TableAnalyzer()
analysis = analyzer.analyze(sample)
# analysis = {
#     'header_tree_text': '...',
#     'answer_format_hint': 'RULE: ...',
#     'column_normalizations': [...],
#     'format_normalizations': {...}
# }
```

**管线集成**：
- `preprocess.py --analysis_only` 批量处理（注：当前 `preprocess.py` 签名未含 `--analysis_only`，分析批量入口以实际代码为准；run_QA.sh 主线经 `test_analyzed.jsonl` 消费）
- 结果写入 JSONL 的 `sample["table_analysis"]` 字段
- Refine operations 通过 `sample.get("table_analysis")` 消费 hints

**覆盖错误模式**：

| 模式 | Hint 消费方 | 依赖 |
|------|-----------|------|
| 答案格式不匹配 | `final_query` → `answer_format_hint` | 无 |
| 列值模糊匹配 | `select_row` → `column_normalizations` | `preprocess_utils/column_norm.py` |
| 混合格式数值 | `sort_by` → `format_normalizations` | `preprocess_utils/column_norm.py` |
| 复合单元格编码 | `header_tree_text` → prompt 注入 | `preprocess_utils/header_tree.py` |

---

## 依赖关系

```
agents/__init__.py
  ├── multi_agent_framework.py  (BaseAgent, AgentType)
  ├── clarifier_agent.py        ★ Stage 1
  ├── retriever_agent.py        ★ Stage 2 (→ multi_agent_framework.py)
  ├── table_analyzer.py         ★ Stage 0.5 (→ preprocess_utils/header_tree, column_norm)
  └── [其他 8 个 agent]         (加载但未使用；active_forgetting 与 memory/ 模块同名异实现)
```

---

## 测试

当前无测试文件。历史文档引用的 `tests/test_table_analyzer.py` 已不存在（`.gitignore` 忽略 `/tests/` 且目录缺失）。TableAnalyzer 的行为目前依赖端到端管线间接验证。

---

## 相关文件

- `agents/__init__.py` - 模块导出
- `agents/multi_agent_framework.py` - BaseAgent 基类
- `agents/clarifier_agent.py` - Clarifier（★）
- `agents/retriever_agent.py` - Retriever（★）
- `agents/table_analyzer.py` - TableAnalyzer（★）
- `agents/{reasoner,judge,critic,refiner,validator,curator}_agent.py` - 未接线 agent
- `agents/dispute_handler.py` / `agents/memory_integration.py` / `agents/active_forgetting.py` - 未接线
- `critic/TableQA/tools/few_shot_critic.json` - 错误树 JSON（Retriever 的 memory_path）
