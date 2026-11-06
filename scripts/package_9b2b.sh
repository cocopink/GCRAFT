#!/usr/bin/env bash
# package_9b2b.sh — 组装 9B/2B 远程执行包并 tar
set -u
PKGROOT=/root/autodl-tmp/gcraft_9b2b_package
rm -rf $PKGROOT; mkdir -p $PKGROOT/{docs,scripts,logs}

# ---- 1. Table-Critic 仓库 (原版 refine 代码 + 数据 + 干净树 + 原 thought 产物 2B/9B) ----
rsync -a --exclude '.git' --exclude '__pycache__' --exclude 'results_v2' --exclude 'logs' \
      --exclude 'results/refine' --exclude 'results/cross' --exclude 'results_smoke' \
      --exclude 'results/thought/*/qwen3.5-0.8b' --exclude 'results/thought/*/qwen3.5-4b' \
      /root/code/Table-Critic/ $PKGROOT/Table-Critic/
# canonical 干净树覆盖 (TC 工作区树被 v2 批次污染)
cp /root/code/GCRAFT/critic_tree_archives/canonical_baseline/wikitq.json $PKGROOT/Table-Critic/critic/TableQA/tools/few_shot_critic.json
cp /root/code/GCRAFT/critic_tree_archives/canonical_baseline/tabfact.json $PKGROOT/Table-Critic/critic/TableFV/tools/few_shot_critic.json

# ---- 2. GCRAFT 精简 (thought 管线 + boost_thought 产物 + 文档) ----
rsync -a --exclude '.git' --exclude '__pycache__' \
      --exclude 'results' --exclude 'test' --exclude 'test.tar.gz' --exclude 'results_v2' --exclude 'logs' --exclude 'dist' \
      --exclude 'critic_tree_archives' --exclude 'nohup.out' \
      /root/code/GCRAFT/ $PKGROOT/GCRAFT/
# 补 G thought 产物 (2B×2 已完成 + 9B tabfact 已完成; 9B wikitq 远程补)
mkdir -p $PKGROOT/GCRAFT/results/boost_thought/{wikitq,tabfact}/{2b,9b}
for d in results/boost_thought/tabfact/2b results/boost_thought/wikitq/2b results/boost_thought/tabfact/9b; do
    rsync -a "/root/code/GCRAFT/$d/" "$PKGROOT/GCRAFT/$d/"
done

# ---- 3. 文档与脚本 ----
cp /root/code/GCRAFT/docs/plan_9b_2b_remote.md $PKGROOT/docs/
cp /root/code/GCRAFT/docs/data_summary_2026-11-03.md /root/code/GCRAFT/docs/tricks_log.md /root/code/GCRAFT/docs/ 2>/dev/null || true
cp /root/code/GCRAFT/docs/*.md $PKGROOT/docs/ 2>/dev/null || true
cp /root/code/GCRAFT/scripts/run_9b_2b_remote.sh $PKGROOT/scripts/

# ---- 4. README ----
cat > $PKGROOT/README.md << 'EOF'
# 9B/2B 远程执行包（新框架全规模矩阵剩余部分）

## 内容
- Table-Critic/：原版仓库（refine 代码含 max_loop/token 补丁、数据、canonical 干净树、原 thought 产物 2B/9B）
- GCRAFT/：thought 管线（raw+Clarifier）+ boost_thought 产物（2B×2、9B tabfact）
- scripts/run_9b_2b_remote.sh：一键编排（9B 4 组 + 2B 4 组，manifest/token 全记录）
- docs/plan_9b_2b_remote.md：计划文档（矩阵/判定目标/口径）

## 部署
1. 解包到任意目录（需 ≥10GB 空闲）
2. 模型权重：Qwen3.5-9B（19GB）与 Qwen3.5-2B（4.3GB），HF 格式
   （zip/tar 不含权重；scp 或 huggingface-cli download Qwen/Qwen3.5-9B --local-dir ...）
3. 修改 scripts/run_9b_2b_remote.sh 顶部 WEIGHTS_9B/WEIGHTS_2B 为实际路径
4. 依赖：vllm>=0.29、torch、fire、pandas、tqdm、openai、pylcs（requirements 见 GCRAFT 根）

## 执行
```bash
bash scripts/run_9b_2b_remote.sh
```
（串行 9B 块 → 2B 块，总 GPU 时 ≈ 23h；每组幂等，中断重跑自动续）

## 产出与回传
results_v2/（orig_*/new_*/new5_* 各组）+ 各组 manifest.json + token_usage.jsonl
回传整目录即可（本机合并入总报告）。

## 口径备忘
- 全部 max_loop=5；served-model-name 小写（qwen 思考关闭依赖）；greedy temperature=0
- 树：每模型块自动恢复 canonical 原点（包内预置），判对写入为原版共享语义
- 判定对照基线（第一次）：9B QA 0.8072/FV 0.9377；2B QA 0.0879/FV 0.6359
- 0.8B FV 类"毁灭式修正格"运行方差实测 +12.2pp——该类格子单遍对比不可判定
EOF

# ---- 5. tar ----
cd /root/autodl-tmp && tar -czf gcraft_9b2b_package.tar.gz gcraft_9b2b_package/ && ls -lh gcraft_9b2b_package.tar.gz
echo "包内容概览:"
du -sh $PKGROOT/Table-Critic $PKGROOT/GCRAFT 2>/dev/null
find $PKGROOT -name "final_result.pkl" | wc -l
echo "thought pkl 清单:"; find $PKGROOT -name "final_result.pkl" | sed "s|$PKGROOT/||"
