# Memory 模块文档

[根目录](../CLAUDE.md) > **memory**

> 最后更新：2026-09-21 19:39:28 CST

---

## 变更记录 (Changelog)

### 2026-09-21 (新建)
- 初始化 memory 模块文档（此前无文档）
- 确认该模块当前**未被管线引用**，标注与 `agents/active_forgetting.py` 的同名异实现关系

---

## 模块职责

Memory 模块提供错误树的**主动遗忘（Active Forgetting）机制**：通过置信度跟踪与衰减，剪枝低价值案例，为错误树腾出容量。属论文设计的记忆管理组件，当前处于"备用/未接线"状态。

### 管线使用情况

- `run_QA.sh` / `run_FV.sh` 主线**不调用**本模块
- 全仓库无 `from memory import ...` / `import memory` 的引用
- 注意区分：`agents/active_forgetting.py` 中另有 `ActiveForgettingManager`（供 `agents/memory_integration.py` 使用，同样未接入主线），两者是**同名机制的两套独立实现**，勿混淆

---

## 对外接口

**文件**：`memory/active_forgetting.py`

```python
class ActiveForgetting:
    def __init__(self, min_confidence=0.0000001, max_cases=1000000):
        """min_confidence 极低（几乎从不触发遗忘）；max_cases 极大"""

    def update_confidence(self, case_id, success):
        """成功引导修正 → 权重 +1；失败 → 权重 x0.9；夹在 [min_confidence, 10.0]"""

    def prune_low_confidence_cases(self, error_tree):
        """递归剪掉置信度低于 min_confidence 的案例节点"""

    def _prune_excess_cases(self, error_tree):
        """案例总数超过 max_cases 时按路径优先级裁剪"""

    def get_statistics(self):
        """返回案例统计信息"""
```

**文件**：`memory/__init__.py`

```python
from .active_forgetting import ActiveForgetting
__all__ = ['ActiveForgetting']
```

---

## 关键依赖与配置

- 仅依赖标准库（json / os），无第三方依赖、无项目内依赖
- 文件头含 Apache-2.0 许可声明（Table-Critic contributors）

---

## 数据模型

- `confidence_scores: Dict[case_id, float]` - 内存态置信度表
- `error_tree: Dict` - 与 `critic/*/tools/few_shot_critic.json` 同构的层次化错误树（递归遍历节点）

---

## 测试与质量

无测试文件。当前无任何调用方，属于孤立模块。

---

## 常见问题 (FAQ)

### Q1: 何时会启用本模块？

**A**: 若未来接入，预期挂点在 Refine 阶段错误树更新之后（`critic/*/tools/update_tree.py` 的 `update_error_tree` 调用点附近），用于控制错误树规模。接入前需自行初始化并持久化 `confidence_scores`。

### Q2: 与 agents/active_forgetting.py 什么关系？

**A**: 功能目标相同（置信度衰减 + 剪枝），但实现独立：`memory/` 版是纯函数式的树剪枝工具；`agents/` 版是 Manager 风格（含 CaseRecord、加载错误树、与 CuratorAgent 集成）。演进时应二选一收敛。

---

## 相关文件清单

- `memory/__init__.py` - 模块导出
- `memory/active_forgetting.py` - ActiveForgetting 实现
- `agents/active_forgetting.py` - 同名异实现（ActiveForgettingManager）
- `agents/memory_integration.py` - Manager 与 CuratorAgent 的集成层（未接线）
- `critic/TableQA/tools/few_shot_critic.json` - 被管理的数据结构示例
