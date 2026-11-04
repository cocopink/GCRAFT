# 实验 v2：新框架全规模验证 + 原版 token 实证 + 复跑方差

> 创建：2026-11-04 21:15
> 编排：`scripts/run_all_v2.sh`（pid 276490），日志 `logs/run_all_v2e.log`
> 定位：**G-CRAFT 框架 v2 的判定级实验**——框架定义已在交叉消融中确定
> （G thought = raw+Clarifier；refine = 原版 judge/critic/tree 流程）。

---

## 1. 框架定义（v2 定稿）

| 环节 | 实现 | 来源 |
|------|------|------|
| thought | raw 原始表 + Clarifier（E2 C1 口径） | G-CRAFT（交叉消融证明 ≈ 原版且 QA 微优） |
| refine | 原版 judge/critic/tree 流程，**max_loop=5**（用户定稿：全部 5 轮） | Table-Critic 原版（交叉消融证明 +5.26/+2.90 优势） |
| 树 | canonical 原点播种（每模型块 git checkout 恢复），模型间共享顺序进化 | 原版语义 |
| 判定 | match_func（与全部历史批次同口径） | — |

**轮数说明**：曾按 2 轮测试，用户定稿改回 5 轮（原版口径；R1 已证 G 侧轮数无增益，原版侧轮数未单独测但以其原生口径为准）。

## 2. 实验矩阵（16 组 refine + 7 组 thought）

| 模型 | orig_qa | orig_fv | new_qa | new_fv | G thought 补齐 |
|------|---------|---------|--------|--------|----------------|
| 0.8B | ✅ 计划 | ✅ | ✅ | ✅ | tabfact/wikitq |
| 2B | ✅ | ✅ | ✅ | ✅ | tabfact/wikitq |
| 4B | ✅ | ✅ | ✅ | ✅ | 复用 C1_raw |
| 9B | ✅ | ✅ | ✅ | ✅ | tabfact/wikitq |

- **orig_\***：原版 refine × 原版首次 thought（**重跑**，带 TC_TOKEN_LOG token 统计）——回答两次 acc 复跑方差 + 原版 token 实证
- **new_\***：原版 refine × G thought（**新框架全规模**）——回答逐规模是否 ≥ 原版（用户目标）
- 每模型块内 4 组 refine 并发（n_proc 10 × 4），模型间串行切 vLLM

## 3. 记录机制（每数据点可审计）

- `results_v2/<tag>/<model>/manifest.json`：tag/模型/thought 输入路径/max_loop=5/n_proc/backend/**树的 sha256 与 origin**（每模型块 git checkout 恢复原点）/起止时间/最终 acc
- `token_usage.jsonl`：逐调用 token（append 式，mp 下不丢）
- 树状态：每模型块开始快照（git checkout 后的 canonical 原点）；运行后终态在原版仓库（共享树语义，模型间顺序进化——与第一次基线一致，已在报告声明）

## 4. 判定与预期

| 对比 | 判定 |
|------|------|
| orig vs 第一次基线（78.84/93.13 等八个数） | 复跑方差（预期 ±1.5pp 内，E1 口径） |
| new vs orig（同规模同数据集） | **新框架是否 ≥ 原版**（用户核心目标） |
| new(4B) vs 交叉消融（78.95/92.77） | 新框架复现一致性 |

预期（基于交叉消融 4B 实证 78.95/92.77 + 各模型 thought 提升空间）：
- 4B：new ≈ 78.95/92.77（复现）→ QA ✓ 微超原版、FV ≈ 平（−0.36 噪声内）
- 9B：thought 预计 +5.8 提升空间 + 原 refine 正修正 → QA/FV 均有机会超
- 2B：FV 微超机会；QA 结构性难（原版自身负修正）
- 0.8B：QA 追平机会；FV 毁灭式修正难

## 5. 时间线

```
11-04 21:15  0.8B 块（thought ~1h + refine ~5h）
11-05 03:00  2B 块（~9h）
11-05 12:00  4B 块（~10h）
11-05 22:00  9B 块（~11h）
11-06 09:00  16 组全齐 → 最终对比报告（含新框架 vs 原版逐规模表 + token 成本表）
```

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| vLLM 切换窗口 refine 抢跑 | 脚本顺序：服务 ready 确认后才起 refine（v1 教训） |
| 原版 refine 调用 G thought pkl 字段兼容 | 交叉消融已验证零适配通过 |
| mp 下 token 丢失 | append 式逐调用落盘（已修） |
| 树共享进化混杂 | manifest 记录树 sha256；与第一次基线同语义，公平 |
| 单遍方差 | orig 重跑组直接给出复跑方差实证 |
