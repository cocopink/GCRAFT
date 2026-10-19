#!/usr/bin/env bash
# download_model.sh — 一键下载 Qwen3.5-4B（HF 格式，vllm 可调用）
#
# vllm 需要 HF/safetensors 格式（BF16 ~8GB），不是 ollama 的 GGUF。
# 此脚本下载 HF 格式，vllm serve 直接用。
#
# ⚠️ vLLM 兼容性: Qwen/Qwen3.5-4B 有已知 bug #36275（weight naming mismatch，启动失败）。
#    如 vllm serve 启动失败，改用备选: bash download_model.sh --model Qwen/Qwen3-4B-Instruct-2507
#    （Qwen3-4B-Instruct-2507 是 vLLM 兼容的 instruct 版本）
#
# 用法:
#   bash download_model.sh                                       # 默认 Qwen3.5-4B (HF + hf-mirror)
#   SRC=modelscope bash download_model.sh                        # modelscope（国内更快）
#   bash download_model.sh --model Qwen/Qwen3-4B-Instruct-2507   # vLLM 兼容备选
#   bash download_model.sh --model Qwen/Qwen3.5-9B               # 9B
#
# 下载后启动 vllm:
#   vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.9

set -uo pipefail

SRC="${SRC:-hf}"                # hf | modelscope
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)   SRC="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "============================================"
echo "下载 $MODEL （vllm HF 格式，BF16 ~8GB）"
echo "源: $SRC"
echo "============================================"

if [[ "$MODEL" == "Qwen/Qwen3.5-4B" ]]; then
    echo "⚠️  Qwen3.5-4B 在 vLLM 有已知 bug #36275（weight naming mismatch）。"
    echo "   如 vllm serve 启动失败，改用:"
    echo "     bash $0 --model Qwen/Qwen3-4B-Instruct-2507"
    echo ""
fi

case "$SRC" in
    hf)
        # 国内用 hf-mirror 加速；海外可 HF_ENDPOINT=https://huggingface.co
        export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
        pip install -q -U huggingface_hub
        echo "HF_ENDPOINT=$HF_ENDPOINT"
        echo "→ huggingface-cli download $MODEL"
        huggingface-cli download "$MODEL"
        echo "✓ 已下载到 HF cache (~/.cache/huggingface/hub)"
        echo "  启动: vllm serve $MODEL --port 8000 --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.9"
        ;;
    modelscope)
        pip install -q -U modelscope
        LOCAL_DIR="./models/${MODEL##*/}"
        echo "→ modelscope download $MODEL → $LOCAL_DIR"
        modelscope download "$MODEL" --local-dir "$LOCAL_DIR"
        echo "✓ 已下载到 $LOCAL_DIR"
        echo "  启动: vllm serve $LOCAL_DIR --port 8000 --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.9"
        ;;
    *) echo "Error: SRC must be hf|modelscope (got: $SRC)" >&2; exit 2 ;;
esac

echo "============================================"
echo "✓ 下载完成"
echo "============================================"
