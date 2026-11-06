# Trick 清单与迭代计划（刷分备选，v3 纯净版先行）

> 创建：2026-11-04。状态：**全部待评估**——v3（G thought + 原 refine 5 轮，纯净版）先行出基准，
> 本清单为 R3b+ 迭代的输入。每个 trick 启用时在对应组 manifest 的 `tricks` 字段记录。

## 基础配置（v3 new 组，已是实证最优）

raw thought + Clarifier + 原 refine（5 轮）+ canonical 树——交叉消融实证（4B 78.95/92.77）。
R1（轮数）/R2（融合树）两个杠杆已判负（详见 data_summary §2）。

## 候选 Trick（按证据强度排序）

| # | Trick | 治什么 | 依据 | 成本 | 风险 | 状态 |
|---|-------|--------|------|------|------|------|
| T-B | **修正回退规则**：全部修正轮判错时，保留 thought 原答案作为 fallback | 0.8B FV 毁灭式修正（原版 −37pp / G −35pp）；2B QA 原版 −10.4pp | E6: critic 修正成功率 95% 但小模型 judge 不可靠——修正失败时无兜底 | 零 LLM（纯规则） | 低（只影响全轮失败样本） | 待评估 |
| T-A | **Judge 复核**：判 [Correct] 后用数值核验视角独立 prompt 二次确认，双对才 STOP | 4B judge FPR ~20%（E7 混淆矩阵）；错误链被误放行提前终止 | E7 实证 | 判对样本 +1 调用 | 中（二次核验引入新噪声，需 smoke） | 待开发+评估 |
| T-C | 双候选修正 + judge 择优 | 单候选修正成功率 | self-consistency 文献 | 修正调用 ×2 | 中 | 备选 |
| T-D | 图提示 raw 变体（不重排只提示） | — | E2 P2≈0（此形态唯一未测） | 低 | 高（大概率无效） | 低优先 |

## 已判负的杠杆（不要重复尝试）

| 杠杆 | 证据 |
|------|------|
| ATGO 重排 | E2：−2.4/−4.8pp |
| P1 γ 平滑 | E2/R1：≈0 |
| P2 图提示（重排表上） | E2：≈0 |
| P3 候选偏置 | E2：≈0（历史两次判负） |
| 融合大树 | R2：FV −3.2pp / QA +0.6 噪声内 |
| 轮数 5 vs 2（G refine） | R1：两数据集 ≈0 |
| Controller 化 refine | 交叉消融：−5.26/−2.90（核心败笔） |

## T-B 实现草图（启用时参考）

位置：原版 `refine/*/utils/chain.py` 的 `_judge_critic_refine_with_cache_mp_core`
——循环退出（`loop_count == max_loop` 仍判错）时，在返回 `proc_sample` 前对比：
- 保留的 refine 链 final answer vs 原 thought 链 final answer
- 规则（二选一，需 smoke 定）：
  a. 修正后 judge 从未判对 → 回退 thought 原链
  b. 两链都保留，由最终 judge 对 thought 链补判一次选对者
- 前置：core 函数需持有原 thought 答案（入口处存 `orig_answer`）

## 与 manifest 的联动

启用 trick 时：manifest 加 `"tricks": ["TB"]` 字段；未启用则无此字段（纯净版标记）。
