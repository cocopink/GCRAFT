# 事故调研报告：frozen_memory 接线断裂导致 Refine 修正成果丢失 & critic 知识库并发写防御缺失

> 调研日期：2026-10-19
> 影响范围：2026-09-22 ~ 2026-09-24 云端批量实验（Qwen3.5 系列 × WikiTQ/TabFact × Refine 阶段）
> 知识库本体：**未受损**（写盘从未发生）；受损的是 **Refine 修正成果的留存**

---

## 1. 背景与调研起因

排查课题：「并发对模板知识库（critic 错误树）的影响」。前期排查报告结论为「实际影响：零，UPDATE_TREE 三份 QA 日志中触发 0 次，建议入口显式传 `--frozen_memory True` 拆雷」。

本报告对上述结论逐条复核（代码 + 日志 + 文件 mtime + JSON 完整性），**两项核心结论被推翻**。

## 2. 复核结论总表

### 2.1 原报告属实项 ✅

| 检查项 | 结果 | 证据位置 |
|--------|------|---------|
| 知识库 mtime | `few_shot_critic.json`（QA/FV）均为 2026-09-21 16:44，早于全部实验（9/22-9/24）→ **写盘从未发生** | `critic/Table{QA,FV}/tools/few_shot_critic.json` |
| JSON 完整性 | 两文件均可正常解析，顶层键结构完好 | `python json.load` 验证 |
| decide() 保护规则 | 答案不正确时 UPDATE_TREE 被强制改为 STOP（"Only update tree with correct answers"） | `refine/TableQA/utils/controller.py:800-804`（FV 同位） |
| 云端串行 | 模型维度串行 for 循环，跨模型不并发 | `scripts/cloud_mainline.sh:236` |
| 读路径只读 | Retriever 每次 DIAGNOSE_BP 以只读方式加载 JSON，多进程并发安全 | `agents/retriever_agent.py:245`（`open(path, 'r')`） |
| 入口未传 frozen_memory | `run_QA.sh:141` / `run_FV.sh:141` / `cloud_mainline.sh` 均未传；`main_tree_based.py` 默认 `False` | 各文件 |
| 代码版本一致性 | controller.py mtime = 2026-09-21 16:44:15，即实验期间运行的代码与当前副本一致 | `stat` |

### 2.2 原报告错误项 ❌

#### ❌ 结论一：「UPDATE_TREE 触发 0 次」

真实触发（按日志去重样本数）：

| 运行 | 日志 | 触发（崩溃）样本数 |
|------|------|------------------|
| WikiTQ × Qwen3.5-0.8b（9/22 11:03，有效 run） | `logs/run_QA_vllm-qwen3.5-0.8b-20260922-110321.log` | **539** |
| WikiTQ × Qwen3.5-9b（9/24 13:17） | `logs/run_QA_vllm-qwen3.5-9b-20260924-131729.log` | **189** |
| TabFact × Qwen3.5-0.8b（9/22 12:17，有效 run） | `logs/run_FV_vllm-qwen3.5-0.8b-20260922-121705.log` | **150** |
| 其余 run（2b/4b 全部、FV-2b/4b/9b） | — | 0（见 §4 解释） |

**误判原因**：原报告 grep 的 `[DEBUG UPDATE_TREE]` 打印位于崩溃点**之后**，永远不可达；而 NameError 文本不含 "UPDATE_TREE" 字样，两者未被关联。

**触发证据**（日志原文）：

```
Error refining sample 481: name 'frozen_memory' is not defined
```

#### ❌ 结论二：「实际影响：零」

- 写盘确实为零（知识库未变），但**不是因为没触发**，而是触发后崩溃在写盘之前。
- **实际影响**：每次触发 = 一次「Refine 修正成功（Judge 判 [Correct]）但成果被丢弃」。worker `_refine_one_sample_mp_core` 的 `except` 捕获 NameError 后返回 **thought 阶段原始样本**（`refine/TableQA/main_tree_based.py:44-47`），该样本最终答案回退为已被 Judge 判错的 thought 答案。
- 即：**Refine 对这些样本的全部收益被系统性清零**，实验测得的 Refine 增益被低估。

#### ❌ 结论三：「统一显式传 `--frozen_memory True`（一行改动）即可拆雷」

不成立。`frozen_memory` **接线完全断裂**，CLI 传参后 NameError 照旧发生：

| 断点 | TableQA | TableFV |
|------|---------|---------|
| `execute()` 签名无此参数 | `controller.py:881` | `controller.py:878` |
| 分支处引用了未定义名 | `controller.py:1065` | `controller.py:1030` |
| `ActionExecutor.__init__` 不接收 | `controller.py:860` | `controller.py:857` |
| `controller_main_loop` 构造 executor 不传 | `controller.py:1343` | `controller.py:1237` |
| 模块级无 `global` 声明 | 无 | 无 |

`controller_main_loop` 虽有 `frozen_memory` 形参（QA:1187 / FV:1140），但从未传递到 executor——该参数当前是**安慰剂**。

## 3. 根因链路

```
主循环 REFINE 后 Judge 判 [Correct]
  └─ rule 强制构造 UPDATE_TREE 决策（QA controller.py:1441-1448 / FV:1331-1337，绕过 decide() 保护——保护只拦 is_correct=False）
      └─ executor.execute(state, decision)
          └─ if frozen_memory:            ← NameError！execute() 作用域无此名
              └─（不可达）update_error_tree(...)   ← 写盘代码从未执行
                  └─ 异常沿调用栈上抛
                      └─ worker except: return 原始 sample  ← 修正成果丢弃
```

补充：写盘调用点全仓共 3 处——

1. QA 主线：`refine/TableQA/utils/controller.py:1077-1085`，`lock=nullcontext()` 无互斥
2. FV 主线：`refine/TableFV/utils/controller.py:1041-1049`，同上
3. 旧版管线：`refine/Table{QA,FV}/utils/chain.py`（`judge_critic_refine_with_cache_mp`），锁为 `manager.Lock()` 跨进程有效，**主线不调用**——本次不改动

## 4. 2b/4b 零触发的解释

与已知事故「Qwen3.5 2B/4B 本地权重内容损坏 → 全场景 0 分」吻合：模型全错 → Judge 永不判 [Correct] → UPDATE_TREE 永不触发 → 无 NameError。属数据现象，非代码差异。

## 5. 修复方案（本次实施）

| Phase | 内容 | 关键位置 |
|-------|------|---------|
| 1 | frozen_memory 接线修复 + 知识库路径注入（QA/FV 镜像） | 2 份 controller.py + 2 份 main_tree_based.py |
| 2 | 入口显式传 `--frozen_memory True` | `run_QA.sh:141` / `run_FV.sh:141` |
| 3 | 写盘防御：文件锁下沉至 `update_error_tree` 内部（`fcntl.flock`，`<json>.lock`）+ 临时文件 `os.replace` 原子写 | `critic/Table{QA,FV}/tools/update_tree.py` |
| 4 | 历史实验影响评估脚本 + Markdown 报告（重跑与否另行决策） | `scripts/assess_frozen_memory_impact.py` |

**锁方案说明**：原报告建议「nullcontext → multiprocessing.Lock()」，但调用点 new 的 `multiprocessing.Lock()` 每个 worker 进程各持一把独立实例，**不构成互斥**。正确做法为跨进程传递锁对象（需贯穿 main→Pool→worker 全链）或文件锁；文件锁改动面最小、按知识库路径天然隔离，且下沉到 `update_error_tree` 内部后所有调用方自动受保护。

## 6. 验证方法

1. `python -m py_compile` 全部改动文件
2. frozen 语义烟测：`frozen_memory=True` → skip 分支无异常、知识库 mtime 不变；`frozen_memory=False` + 注入 tmp 路径 → 写盘落在注入路径
3. 并发写烟测：8 进程 × 50 次并发写 tmp 副本 → JSON 始终可完整解析
4. grep 入口脚本确认 `--frozen_memory True`
5. （可选，需 API）pilot `first_n=5` 真跑：日志零 NameError

## 7. 遗留决策

- 3 组受影响实验（§2.2 表）是否重跑：待影响评估报告（§8 Phase 6）产出后决定
- `update_tree_orig.py`、`chain.py` 旧版管线：本次不动（非主线 / 锁已有效）

---

## 8. 演进设计：16 实验独立进化树 + 写盘防御 + 置信度遗忘（2026-10-19 定稿）

> 本节为后续实施的准绳，取代 §5 的旧 Phase 编号。

### 8.0 需求定稿

启动 16 组云端实验时，**每组实验拥有独立错误树**：从原始树播种、仅在本实验运行内进化（n_proc 个 worker 共写本组树），实验间互不影响。默认进化（`frozen_memory=False`），`FROZEN_MEMORY=True` 保留为静态树消融开关。遗忘策略采用置信度衰减（LFU 风格）。

### 8.1 总体架构

```
原始树 critic/Table*/tools/few_shot_critic.json  (只读, 永不被实验触碰)
     │ 启动时播种 (仅当目标不存在 → shutil.copy, 支持断点续跑)
     ▼
results/cloud/<dataset>/<model>/<variant>/memory/few_shot_critic.json   ×16 组各自独立
     ▲└ write: 文件锁内 读→扩展→结算置信度→剪枝→原子写
     └ read: n_proc 个 worker 无锁读 (原子替换保证永不读半截)
```

### 8.2 写盘防御（Phase 3）

`critic/Table{QA,FV}/tools/update_tree.py` 新增两原语（镜像）：

```python
@contextmanager
def error_tree_file_lock(error_tree_json):
    # 锁文件 = <知识库路径>.lock（同目录旁挂），fcntl.flock 独占
    # 互斥对象是"文件"→跨进程天然有效；不同路径（不同实验的树）天然隔离
    # 仅 POSIX；与调用方 lock 参数嵌套时本锁在外层，嵌套顺序固定

def _atomic_write_json(path, data):
    # 同目录 mkstemp → json.dump(indent=4) → os.replace 覆盖
    # 异常时清理临时文件；读方任何时刻都读到完整 JSON
```

`update_error_tree` 内两处替换：`with lock:` → `with error_tree_file_lock(error_tree_json), lock:`；截断写 → `_atomic_write_json(...)`。

**为何不用 `multiprocessing.Lock()`**：调用点 new 的 mp.Lock 每个 worker 各持一把独立实例，不构成互斥；文件锁是跨进程互斥的最小正确实现。锁覆盖读→树扩展（含 LLM 调用）→写全程，与旧 chain.py 路径 `manager.Lock` 语义一致；写仅在判对后发生，频率低。`.gitignore` 补 `*.lock`。

### 8.3 路径注入 + 播种（Phase 4）

| 调用方式 | 读 | 写 |
|----------|----|----|
| `frozen_memory=True` | canonical | 永不写（静态树消融） |
| `frozen_memory=False` 无路径 | canonical | canonical（旧行为兼容） |
| `frozen_memory=False` + `--critic_tree_json` | path（缺失则从 canonical 播种） | path（实验私有树） |

贯通链：`main(critic_tree_json="")` → mp 元组 / 单线程调用点 → `controller_main_loop(error_tree_json="")` → `ActionExecutor(error_tree_json=...)` → `RetrieverAgent(memory_path=...)` + `update_error_tree(error_tree_json=...)`；空串回退 canonical。

播种规则（main() 内）：frozen=False 且路径非空 → 不存在则 `shutil.copy(canonical, path)`；存在则续用（与 cache 断点续跑语义一致）。

`run_QA.sh`/`run_FV.sh`：`FROZEN_MEMORY=${FROZEN_MEMORY:-False}`；树路径 `${REFINE_RESULTS}${GRAPH_SUFFIX}/memory/few_shot_critic.json`。`cloud_mainline.sh` **零改动**——结果目录按 dataset/model/variant 天然隔离 → 16 组 16 棵树。

### 8.4 置信度遗忘（Phase 5，LFU 风格）

1. 新案例结构追加 `"case_id": "<error_route>#<uuid8>"`、`"confidence_score": 1.0`（写入点 `template_dict`）；存量旧案例无 case_id → 结算自然跳过（兼容）
2. executor DIAGNOSE 分支从 `pre_retrieved_few_shot`（案例 dict 列表，`multiprocess.py:82-122`）提取 case_id → `state.hit_case_ids`
3. UPDATE_TREE（必然判对）锁内对命中案例 `confidence_score += 1`；首版不做失败 ×0.9 衰减（失败不写树，无结算时机）
4. 每次写盘锁内 `memory.ActiveForgetting(max_cases=可配,默认 1e6).prune_low_confidence_cases(tree)` 再原子写——机制先通，阈值留实验配置

### 8.5 历史影响评估（Phase 6）

`scripts/assess_frozen_memory_impact.py`（只读）：日志提取 `Error refining sample (\d+)` → 定位 final_result.pkl → 总样本/崩溃数/崩溃样本当前判分 → Markdown 报告；重跑决策归用户。

### 8.6 验证

1. 原语级并发：8 进程 × 50 次（锁→读→改→原子写）同一临时树 → JSON 必可解析、计数无丢失、无 `.tmp` 残留
2. 函数级：mock `get_table_log` + mock llm 真调 `update_error_tree(error_route='random')` → 写盘落注入路径、`.lock` 生成
3. 语义级：frozen=True 读 canonical 不写；frozen=False + 注入路径 → 自动播种；QA/FV 双测
4. `bash -n` run 脚本 + grep 传参；py_compile 全部改动文件；每 Phase 一个 commit
