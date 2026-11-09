#!/usr/bin/env bash
# run_gamma_axis_pilot.sh — Gamma×Axis pilot 重推全矩阵 (24 组)
#
# 4 gamma (0/0.1/0.5/0.7) × 3 轴 (row/col/rowcol) × 2 数据集 (qa/fv), 每组 200 gold 错误样本。
# 前置: scripts/sample_pilot_errors.py 已产出 gamma_pilot_raw.jsonl;
#       /tmp/gen_gamma_data.sh 已产出全部 24 组重排数据 (等待 "DATA ALL DONE")。
#
# 用法: nohup bash scripts/run_gamma_axis_pilot.sh > logs/gamma_axis_pilot_run.log 2>&1 &

set -u
cd /root/code/GCRAFT

GAMMAS="${GAMMAS:-0 0.1 0.5 0.7}"
AXES="row col rowcol"
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="EMPTY"
MODEL="qwen3.5-4b"
N_PROC=24
CHUNK=4

# 等待重排数据生成完毕
while ! grep -q "DATA ALL DONE" /tmp/gen_gamma_data.log 2>/dev/null; do
    sleep 30
done
echo "[$(date -Is)] 重排数据就绪, 开始 24 组重推"

for ds in qa fv; do
  if [ "$ds" = qa ]; then
    D="thought/TableQA/data/wikitq"; TDIR="TableQA"; RES="wikitq"; EXTRA=""
  else
    D="thought/TableFV/data/tabfact"; TDIR="TableFV"; RES="tabfact"
    EXTRA="--raw2clean_path thought/TableFV/data/tabfact/raw2clean.jsonl"
  fi
  for g in $GAMMAS; do
    for m in $AXES; do
      inp="$D/gamma_pilot_${m}_g${g}.jsonl"
      out="test/results/thought/${RES}/qwen3.5-4b_gammapilot_${m}_g${g}"
      if [ ! -f "$inp" ]; then echo "❌ 缺少 $inp, 跳过 [$ds/$m/g$g]"; continue; fi
      echo ""
      echo "[$(date -Is)] === gammapilot $ds / $m / gamma=$g ==="
      python "thought/${TDIR}/main.py" \
        --dataset_path "$inp" \
        --thought_results_dir "$out" \
        --base_url "$BASE_URL" --openai_api_key "$API_KEY" \
        --model_name "$MODEL" --first_n -1 \
        --n_proc "$N_PROC" --chunk_size "$CHUNK" \
        --use_clarifier True $EXTRA
      acc=$(grep -oP 'Accuracy: \K[0-9.]+' "$out/acc.txt" 2>/dev/null || echo "N/A")
      echo "[$(date -Is)] gammapilot $ds/$m/g$g acc: $acc"
    done
  done
done
echo "GAMMA PILOT ALL DONE"
