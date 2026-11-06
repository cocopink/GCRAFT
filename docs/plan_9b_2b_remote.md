# 9B/2B 远程执行计划（新框架全规模矩阵的剩余部分）

> 创建：2026-11-06。定位：本机（RTX 4080 SUPER 32GB）v4 编排跑 0.8B/4B 块；**9B 与 2B 块转移至第二台服务器执行**（并行加速）。
> 框架定义（已定稿）：G thought（raw+Clarifier）+ 原版 refine（judge/critic/tree，**max_loop=5**）+ canonical 原点播种树。
> 打包：`gcraft_9b2b_package.zip`（部署指南见包内 README.md）。

---

## 1. 实验矩阵（本服务器需跑 12 组 refine）

| 模型 | 组 | thought 输入 | refine | 轮数 | 目的 |
|------|-----|-------------|--------|------|------|
| 9B | new_qa / new_fv | G raw+Clarifier（wikitq/9b 现场补、tabfact/9b 已带） | 原版 | 5 | 新框架 9B |
| 9B | orig_qa / orig_fv | 原版第一次 thought（已带） | 原版 | 5 | 原版 token 实证 + 复跑方差 |
| 2B | new5_qa / new5_fv | G raw+Clarifier（已带） | 原版 | **5** | 2B 轮数消融（对照已有 2 轮版 64.77/8.91） |
| 2B | orig_qa / orig_fv | 原版第一次 thought（已带） | 原版 | 5 | 原版 token 实证 + 复跑方差 |

（2B 的 new_qa/new_fv **2 轮版已在本机完成**：64.77/8.91，均已超原版——保留为轮数消融的 2 轮端。）

## 2. 判定目标（vs Table-Critic 第一次基线）

| 格 | 原版基线 | 新框架预期 | 状态 |
|-----|---------|-----------|------|
| 9B QA | 80.72 | thought 提升空间 + 原 refine 正修正 | 有机会超 |
| 9B FV | 93.77 | 同上 | 有机会超 |
| 2B QA（5 轮） | 8.79 | 2 轮版 8.91 已超；5 轮版预期持平或微降（轮数≈0 效应） | 消融数据 |
| 2B FV（5 轮） | 63.59 | 2 轮版 64.77 已超；5 轮版待测 | 消融数据 |

**已知边界**（勿误判）：0.8B FV 为毁灭式修正格（运行方差实测 +12.2pp），该类格单遍对比不可判定。

## 3. 执行序列（`scripts/run_9b_2b_remote.sh`）

```
Phase 9B: vLLM 9b → G thought wikitq/9b（补齐，~3h）→ new×2 并发（~5h）→ orig×2 并发（~5h）
Phase 2B: vLLM 2b → new5×2 并发（5 轮，~6h）→ orig×2 并发（~4h）
```
总 GPU 时 ≈ 23h。每组产出：result.txt（acc）+ manifest.json（参数/树 sha256/起止）+ token_usage.jsonl（逐调用）。

## 4. 前置条件核查清单（新服务器）

1. **模型权重**：Qwen3.5-9B（~19GB）与 Qwen3.5-2B（~4.3GB），HF 格式（zip 不含，下载/拷贝后改脚本顶部 WEIGHTS 路径）
2. **vLLM ≥0.29**，启动参数：`--max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-num-seqs 128 --served-model-name qwen3.5-<size>`（**served-name 必须小写全小写**，qwen 思考关闭依赖此前缀）
3. **磁盘**：≥10GB（thought pkl + 结果）
4. GPU 显存 ≥24GB（32k 上下文）

## 5. 记录与回传

- 每组 `manifest.json` + `token_usage.jsonl` + `result.txt` 随目录保留
- 完成后回传：`results_v2/` 全目录 + 各组 manifest（本机合并入总报告）
- 树状态：每模型块开始自动 git checkout 恢复 canonical 原点（包内已预置干净树）；**勿手工改动树文件**

## 6. 与本机 v4 编排的分工防冲突

- 本机 v4 排程：0.8B 块（进行中）→ 4B 块 → 9B 块 → 2B new5
- **新服务器先完成 9B/2B 时**：通知本机，v4 到 9B/2B 块前手动跳过（或让 v4 跑完作为复跑——两组独立树，结果不冲突，可当第三个数据点）
- 合并口径：全部对比用**同批**（新服务器批次内部 orig vs new）；与第一次基线的跨批对比仅作参考
