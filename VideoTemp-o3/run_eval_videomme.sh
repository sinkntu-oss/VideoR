#!/bin/bash
# ============================================================
# Video-MME 单卡 Baseline 评测脚本
# 使用方法:
#   conda activate Tempo3
#   cd /mnt/tidal-alsh01/dataset/eam_ds/VideoR/VideoTemp-o3
#   bash run_eval_videomme.sh
#
# 说明:
#   1. 后台启动 vLLM 推理服务 (单卡, GPU 0)
#   2. 等待服务就绪
#   3. 运行 Video-MME 评测
#   4. 打分输出结果
#   5. 自动关闭 vLLM 服务
# ============================================================
set -e

MODEL_PATH="/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen2.5-VL-7B-Instruct"
PORT=8000

echo "============================================"
echo "  Video-MME Baseline 评测"
echo "  模型: $MODEL_PATH"
echo "  GPU: 单卡 (CUDA_VISIBLE_DEVICES=0)"
echo "============================================"

# -----------------------------------------------------------
# Step 1: 后台启动 vLLM 服务
# -----------------------------------------------------------
echo ""
echo "[Step 1/4] 启动 vLLM 推理服务..."

CUDA_VISIBLE_DEVICES=0 \
DECORD_EOF_RETRY_MAX=20480 \
VIDEO_MIN_PIXELS=50176 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=1024 \
swift deploy \
    --model_type qwen2_5_vl \
    --model $MODEL_PATH \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 81920 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"image": 5, "video": 20}' \
    --temperature 0.1 \
    --vllm_mm_processor_cache_gb 0 \
    --served_model_name Qwen2.5-VL-7B \
    --port $PORT &

VLLM_PID=$!
echo "  vLLM 服务 PID: $VLLM_PID"

# -----------------------------------------------------------
# Step 2: 等待服务就绪
# -----------------------------------------------------------
echo ""
echo "[Step 2/4] 等待 vLLM 服务启动..."

MAX_WAIT=600  # 最多等 10 分钟
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://0.0.0.0:$PORT/v1/models > /dev/null 2>&1; then
        echo "  vLLM 服务已就绪! (等待了 ${WAITED}s)"
        break
    fi
    # 检查 vLLM 进程是否还活着
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  错误: vLLM 服务意外退出"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  已等待 ${WAITED}s..."
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  错误: vLLM 服务启动超时"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# -----------------------------------------------------------
# Step 3: 运行评测
# -----------------------------------------------------------
echo ""
echo "[Step 3/4] 运行 Video-MME 评测..."
python eval/videomme/videomme.py

# -----------------------------------------------------------
# Step 4: 打分
# -----------------------------------------------------------
echo ""
echo "[Step 4/4] 计算评测结果..."
python eval/score.py videomme --return_categories_accuracy --return_task_types_accuracy

# -----------------------------------------------------------
# 清理: 关闭 vLLM 服务
# -----------------------------------------------------------
echo ""
echo "关闭 vLLM 服务 (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "  评测完成!"
echo "  结果文件: eval/videomme/data/videomme/output/test.json"
echo "============================================"
