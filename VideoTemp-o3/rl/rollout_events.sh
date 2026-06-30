#!/bin/bash
# ============================================================
# 事件定位版 Rollout 推理引擎启动脚本
# 使用 event_locating_scheduler 替代 video_processing_scheduler
#
# 支持 5 套 prompt 风格（通过 PROMPT_STYLE 环境变量选择）：
#   baseline (默认) - image 限制=1  (rollout 中只有 tool 返回的 video)
#   b               - 同 baseline
#   c               - 同 baseline
#   d               - image 限制=64 (关键帧版，第一轮就有 2×N 张 image)
#   e               - 同 baseline
#
# 使用:
#   bash rl/rollout_events.sh                  # baseline
#   PROMPT_STYLE=d bash rl/rollout_events.sh   # 方案 D
# ============================================================

PROMPT_STYLE="${PROMPT_STYLE:-baseline}"

# 方案 D 第一轮就有 2×N 张 keyframe（N 通常 3-15，最多覆盖 30+ 张）
# 其他方案第一轮只有 <video>，rollout 中也只有 tool 返回的 <video>，image 用不到
case "$PROMPT_STYLE" in
    baseline|b|c|e) IMAGE_LIMIT=1 ;;
    d)              IMAGE_LIMIT=64 ;;
    *) echo "[ERROR] Unknown PROMPT_STYLE: $PROMPT_STYLE (expected: baseline|b|c|d|e)" >&2; exit 1 ;;
esac

VIDEO_LIMIT="${VIDEO_LIMIT:-10}"
# 默认指向同 PROMPT_STYLE 的 SFT 输出目录；多 checkpoint 时请通过 MODEL=... 显式指定到具体 checkpoint-xxx 子目录
MODEL="${MODEL:-sft/ckpt/test_events_${PROMPT_STYLE}}"

echo "============================================"
echo "  Rollout  PROMPT_STYLE=$PROMPT_STYLE"
echo "  起点模型: $MODEL"
echo "  image 限制: $IMAGE_LIMIT, video 限制: $VIDEO_LIMIT"
echo "  (如需指定具体 checkpoint，请用 MODEL=sft/ckpt/test_events_${PROMPT_STYLE}/checkpoint-xxx bash rl/rollout_events.sh)"
echo "============================================"

# Image 像素参数（方案 D 必需；其他方案无 <image> 输入时无影响）
MAX_PIXELS="${MAX_PIXELS:-501760}"
MIN_PIXELS="${MIN_PIXELS:-50176}"

DECORD_EOF_RETRY_MAX=20480 \
MAX_PIXELS=$MAX_PIXELS \
MIN_PIXELS=$MIN_PIXELS \
VIDEO_MIN_PIXELS=50176 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=512 \
CUDA_VISIBLE_DEVICES=6,7 \
swift rollout \
    --model $MODEL \
    --vllm_use_async_engine true \
    --external_plugins rl/video_event_plugin.py \
    --multi_turn_scheduler event_locating_scheduler \
    --vllm_max_model_len 40960 \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.75 \
    --vllm_mm_processor_cache_gb 0 \
    --max_turns 3 \
    --vllm_limit_mm_per_prompt "{\"image\": $IMAGE_LIMIT, \"video\": $VIDEO_LIMIT}" \
    --port 8100
