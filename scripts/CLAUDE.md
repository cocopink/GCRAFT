# Scripts 模块文档

[根目录](../CLAUDE.md) > **scripts**

> 最后更新：2026-09-21 19:39:28 CST

---

## 变更记录 (Changelog)

### 2026-09-21 (增量更新)
- 补录 7 个新文件：cloud_mainline.sh、run_p1p2_mainline_batch.py、run_graph_experiment_matrix.py、build_error_data.py、rerun_error_results_with_atg_context.py、package_cloud_bundle.sh、cloud_git_sync.sh
- 补充云端实验工作流（vLLM/ollama、16 组主实验）

### 2026-06-09 (新建)
- 初始化文档，收录 3 个脚本

---

## 模块职责

Scripts 目录包含 G-CRAFT 的**实验编排与工程辅助脚本**：pilot 数据集构建、P1/P2 图增强消融矩阵、云端批量主实验（vLLM/ollama）、错误样本重跑、结果重建与汇总、接口烟测、部署打包与 git 同步。

---

## 文件清单

| 文件 | 职责 | 类别 |
|------|------|------|
| `build_refine_pilot_set.py` | 从 thought/baseline 结果构建固定 pilot 索引集 | 实验控制 |
| `summarize_refine_experiment.py` | 基线 vs 实验的成对对比报告 (Markdown) | 报告 |
| `run_graph_experiment_matrix.py` | row-ATGO 图实验矩阵（baseline/P1/P2/P1+P2），默认 dry-run | 实验编排 |
| `run_p1p2_mainline_batch.py` | 主线 P1+P2 批量跑批（模型回退选择 + 烟测前置） | 实验编排 |
| `build_error_data.py` | 从 error_results.pkl 反推 error_data.jsonl（剥离 Thought 阶段字段） | 数据工具 |
| `rerun_error_results_with_atg_context.py` | 仅对 error_results.pkl 重跑 Thought 并注入 ATG context | 数据工具 |
| `rebuild_final_result.py` | 从 cache 目录重建 final_result.pkl | 结果工具 |
| `openai_smoke_test.py` | OpenAI-compatible 接口单模型连通性烟测 | 烟测 |
| `openai_batch_smoke_test.py` | 批量模型连通性烟测（读 configs/llm_providers.json） | 烟测 |
| `cloud_mainline.sh` | 云端一键启动 16 组主实验（4 模型 x 2 方法 x 2 数据集） | 云端 |
| `package_cloud_bundle.sh` | 打包仓库为云端部署 tar.gz（排除 logs/results/cache） | 云端 |
| `cloud_git_sync.sh` | 云端 git 自动初始化/拉取（auto/pull/init/off 四模式） | 云端 |

---

## 入口与使用

### 实验编排

**run_graph_experiment_matrix.py** - 图增强消融矩阵

```bash
# 默认只执行 ATGO 预处理并写出 Thought/Refine 命令清单
python scripts/run_graph_experiment_matrix.py
# 显式传入 --run_llm 才会真正调用 OpenAI-compatible API
python scripts/run_graph_experiment_matrix.py --run_llm
```

内置实验组：`baseline` (gamma=0, 无 hint)、`P1` (gamma=0.1)、`P2` (hint 注入)、`P1+P2`。API key 从 `OPENAI_API_KEY` 环境变量解析（支持 `<env:OPENAI_API_KEY>` 占位符）。

**run_p1p2_mainline_batch.py** - 主线批量

```bash
python scripts/run_p1p2_mainline_batch.py
```

- 默认逻辑模型 `glm-5.1`；`gpt-5.4` 会展开为 4 档回退（gpt-5.4/-high/-medium/-low），逐个烟测取首个可用
- 复用 `openai_smoke_test.RuntimeConfig` 与 `openai_batch_smoke_test.resolve_runtime_config`
- 数据集映射：`wikitq → run_QA.sh`、`tabfact → run_FV.sh`；通过环境变量向脚本透传配置

**cloud_mainline.sh** - 云端 16 组主实验

```bash
bash scripts/cloud_mainline.sh --node all --stage full --backend vllm \
    --base-url http://127.0.0.1:8000/v1 --dataset both --variants baseline,p1p2
```

| 环境变量 | 默认 | 说明 |
|---------|------|------|
| `NODE` | all | 1（0.8B+2B）/ 2（4B+9B）/ all |
| `STAGE` | full | smoke / pilot / full |
| `BACKEND` | vllm | vllm / ollama（vLLM 默认自动拉起 server） |
| `DATASET` | both | wikitq / tabfact / both |
| `VARIANTS` | baseline,p1p2 | 只跑主实验 |
| `RESULTS_BASE` | results/cloud | 结果根目录 |
| `VLLM_*` | - | host/port/max-model-len/dtype/gpu-mem 等服务参数 |

启动前自动 source `.cloud_bundle.env`（若存在）。

### 数据工具

**build_error_data.py** - 错误样本反推

```bash
python scripts/build_error_data.py --tabfact   # 或 --wikitq / --all
```

从 `test/results/thought/<dataset>/<model>/error_results.pkl` 还原原始字段 JSONL，剥离 `chain/clarifier/*_reranked_table/atg_context/graph_relevance/table_analysis` 等 Thought 阶段字段，输出到 `thought/<Task>/data/<dataset>/error_data.jsonl`。

**rerun_error_results_with_atg_context.py** - 错误样本 ATG 重跑

```bash
python scripts/rerun_error_results_with_atg_context.py \
    --dataset wikitq \
    --input-pkl test/results/thought/wikitq/gpt-5.4/error_results.pkl \
    --output-dir test/results/thought/wikitq/gpt-5.4_error_atgctx --mode final
```

注意：该脚本 import `table_structure.context.build_atg_context`，当前该符号不存在（见 table_structure/CLAUDE.md FAQ Q3），运行会 ImportError。

**build_refine_pilot_set.py** - pilot 索引集

```bash
python scripts/build_refine_pilot_set.py \
    --thought-pkl "results/thought/wikitq/gpt-5.4/final_result.pkl" \
    --output "results/pilot/wikitq/pilot_indices_100.json" --size 100 --seed 42
```

优先选取 baseline 错误样本，固定随机种子保证实验可比。

### 报告与结果工具

**summarize_refine_experiment.py**

```bash
python scripts/summarize_refine_experiment.py \
    --baseline-pkl results/refine/baseline/final_result.pkl \
    --experiment-pkl results/refine/experiment/final_result.pkl \
    --experiment-token-json results/refine/experiment/token_usage.json \
    --output results/report.md
```

报告包含 accuracy 对比、changed_count、token usage 统计。

**rebuild_final_result.py**

```bash
python scripts/rebuild_final_result.py --wikitq --n_proc 4
python scripts/rebuild_final_result.py --tabfact --n_proc 4
```

从 cache 目录重建 final_result.pkl（仅执行 fixed_chain = simple_query）。

### 烟测

**openai_smoke_test.py** - 单模型连通性

```bash
python scripts/openai_smoke_test.py --model gpt-5.4
```

- `base_url` 优先读 `HAOMIAO_URL`；`api_key` 优先 `HAOMIAO_AUTH_TOKEN`，兼容 `HAOMIAO_AUTHEN_TOKEN` / `OPENAI_API_KEY`
- 提供 `RuntimeConfig`、`mask_secret`、`run_smoke_test`、`validate_runtime_config`，被 mainline batch 复用

**openai_batch_smoke_test.py** - 批量连通性

```bash
python scripts/openai_batch_smoke_test.py
python scripts/openai_batch_smoke_test.py --model gpt-5.4 --model qwen3.6-plus
python scripts/openai_batch_smoke_test.py --models-file /tmp/models.txt --timeout 20
```

默认 provider `current-gpt54`（读 `configs/llm_providers.json`，该目录被 .gitignore 忽略，属本机配置）；内置 25 个候选模型。

### 云端工程

**package_cloud_bundle.sh**

```bash
bash scripts/package_cloud_bundle.sh              # 输出 dist/table-critic-cloud-<时间戳>.tar.gz
OUT_DIR=/tmp bash scripts/package_cloud_bundle.sh
```

rsync 打包，排除 `.git/`、`__pycache__/`、`logs/`、`results/`、`test/results/`、`dist/` 等。

**cloud_git_sync.sh**

```bash
bash scripts/cloud_git_sync.sh                                          # auto：有仓库则 pull
GIT_REMOTE_URL=<url> GIT_BRANCH=main bash scripts/cloud_git_sync.sh --mode init
```

---

## 关键依赖

```python
from refine.TableQA.utils.read_pkl import read_pkl
from table_structure.context import build_atg_context   # 仅 rerun_error_results_with_atg_context.py（当前缺失）
from scripts.openai_smoke_test import RuntimeConfig, ... # 脚本间互相复用
```

脚本统一以 `sys.path.insert(0, <仓库根>)` 方式加载项目模块，必须从仓库根目录运行。

---

## 相关文件清单

- `scripts/build_refine_pilot_set.py` / `scripts/summarize_refine_experiment.py` / `scripts/rebuild_final_result.py`
- `scripts/run_graph_experiment_matrix.py` / `scripts/run_p1p2_mainline_batch.py`
- `scripts/build_error_data.py` / `scripts/rerun_error_results_with_atg_context.py`
- `scripts/openai_smoke_test.py` / `scripts/openai_batch_smoke_test.py`
- `scripts/cloud_mainline.sh` / `scripts/package_cloud_bundle.sh` / `scripts/cloud_git_sync.sh`
