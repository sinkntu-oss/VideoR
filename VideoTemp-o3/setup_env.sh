#!/bin/bash
# ============================================================
# VideoTemp-o3 环境安装脚本
# 使用方法: bash setup_env.sh
# ============================================================
set -e

ENV_NAME="Tempo3"

echo "============================================"
echo "  VideoTemp-o3 环境安装 ($ENV_NAME)"
echo "============================================"

# 激活 conda
eval "$(conda shell.bash hook)"

# 如果环境不存在则创建
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "[1/5] 创建 conda 环境 $ENV_NAME (Python 3.12)..."
    conda create -n $ENV_NAME python=3.12 -y
else
    echo "[1/5] conda 环境 $ENV_NAME 已存在，跳过创建"
fi

conda activate $ENV_NAME
echo "  Python: $(python --version)"

# 安装 vLLM (CUDA 12.x)
echo "[2/5] 安装 vLLM v0.11.0..."
pip install https://github.com/vllm-project/vllm/releases/download/v0.11.0/vllm-0.11.0+cu129-cp38-abi3-manylinux1_x86_64.whl

# 安装 ms-swift
echo "[3/5] 安装 ms-swift 3.10.0..."
pip install 'ms-swift[all]==3.10.0' -U

# 安装 flash-attn
echo "[4/5] 安装 flash-attn 2.8.1..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# 安装 DeepSpeed
echo "[5/5] 安装 DeepSpeed 0.16.9..."
pip install deepspeed==0.16.9

echo ""
echo "============================================"
echo "  安装完成！"
echo "  激活环境: conda activate $ENV_NAME"
echo "============================================"
