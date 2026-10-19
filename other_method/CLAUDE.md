# Other Method 模块文档

[根目录](../CLAUDE.md) > **other_method**

> 最后更新：2026-09-21 19:39:28 CST

---

## 变更记录 (Changelog)

### 2026-09-21 (增量更新)
- 复核无结构变化；更新时间戳与导航

### 2026-05-10
- 初始化 other_method 模块文档

---

## 模块职责

Other Method 模块包含 G-CRAFT (Table-Critic) 的**基线对比方法**实现，用于验证框架相对于现有方法的性能提升。

### 支持的基线方法

| 方法 | 缩写 | 说明 | 对应论文图表 |
|------|------|------|-------------|
| End-to-End QA | `e2e` | 零样本直接问答 | Figure 16 |
| Few-Shot QA | `few_shot` | 2-shot 直接回答 | Figure 17 |
| Chain-of-Thought | `cot` | 2-shot 思维链推理 | Figure 18 |
| CoT-Consistency | `cot_consist` | 多次采样 + 多数投票 | 论文实验部分 |

---

## 入口与启动

### 命令行运行

```bash
# WikiTQ - E2E
python other_method/run_baseline.py \
    --dataset wikitq --method e2e --model gpt-5.4 \
    --base_url https://api.example.com/v1 \
    --first_n 100 --n_proc 8

# TabFact - CoT-Consist
python other_method/run_baseline.py \
    --dataset tabfact --method cot_consist --n_sample 5 \
    --model gpt-5.4 --base_url https://api.example.com/v1
```

### 批量运行

```bash
bash other_method/run_other_method.sh      # 或 run_other_method_2.sh
```

---

## 对外接口

### run_baseline.py

**核心参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集（`wikitq` / `tabfact`） | 必填 |
| `--method` | 方法（`e2e` / `few_shot` / `cot` / `cot_consist`） | 必填 |
| `--model` | 模型名称 | 必填 |
| `--base_url` | API base URL | 必填 |
| `--first_n` | 处理前 N 个样本 | -1（全部） |
| `--n_proc` | 进程数 | 8 |
| `--n_sample` | CoT-Consist 采样次数 | 5 |

### prompts.py

**核心函数**：
```python
def build_prompt(
    dataset: str,
    method: str,
    table_text: str,
    question: str,
    statement: str = None
) -> str:
    """
    构建基线方法的 prompt

    Args:
        dataset: 数据集类型
        method: 方法类型
        table_text: 表格文本
        question: 问题（WikiTQ）
        statement: 陈述（TabFact）

    Returns:
        完整的 prompt 字符串
    """
```

### reextract_wikitq.py

用于重新提取 WikiTQ 数据集的答案，修复数据格式问题。

---

## 关键依赖与配置

### 依赖项

```python
# 内部依赖（复用项目已有模块）
from thought.TableQA.utils.load_data import load_wikitq_dataset
from thought.TableQA.utils.helper import table2string
from thought.TableQA.utils.evaluate import wikitq_match_func_for_samples
from thought.TableFV.utils.load_data import load_tabfact_dataset
from thought.TableFV.utils.evaluate import tabfact_match_func_for_samples

# 外部依赖
import numpy as np
from tqdm import tqdm
```

### 结果目录

```
results/other_method/
├── wikitq/
│   ├── e2e/gpt-5.4/
│   ├── few_shot/gpt-5.4/
│   ├── cot/gpt-5.4/
│   └── cot_consist/gpt-5.4/
└── tabfact/
    ├── e2e/gpt-5.4/
    ├── few_shot/gpt-5.4/
    ├── cot/gpt-5.4/
    └── cot_consist/gpt-5.4/
```

每个结果目录包含 `summary.json`、`extra_ans.json`（WikiTQ）和 `token_logs/` 子目录。（`results/` 被 .gitignore 忽略，仅本地生成。）

---

## 常见问题 (FAQ)

### Q1: 如何添加新的基线方法？

**A**：
1. 在 `prompts.py` 中添加新的 prompt 模板和 `build_prompt` 分支
2. 在 `run_baseline.py` 中添加对应的处理逻辑
3. 在 `run_other_method.sh` 中添加运行命令

### Q2: CoT-Consist 的采样次数如何影响结果？

**A**：`n_sample` 越大，多数投票的稳定性越高，但 API 调用成本也越高。论文中通常使用 5 次采样。

### Q3: 目录中的 nohup 文件是什么？

**A**: `other_method/nohup` 是历史后台运行的输出残留，非代码，可忽略或删除。

---

## 相关文件清单

- `other_method/__init__.py` - 模块初始化
- `other_method/run_baseline.py` - 基线运行器
- `other_method/prompts.py` - Prompt 模板
- `other_method/reextract_wikitq.py` - WikiTQ 答案重提取
- `other_method/run_other_method.sh` / `run_other_method_2.sh` - 批量运行脚本
