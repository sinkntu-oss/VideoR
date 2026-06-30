#!/bin/bash
# ============================================================
# 事件定位版 SFT 训练脚本
# 使用转换后的事件标注数据
#
# 支持 6 套 prompt 风格（通过 PROMPT_STYLE 环境变量选择）：
#   baseline (默认) - 数据集: sft/data_events/
#   b               - 数据集: sft/data_events_b/
#   c               - 数据集: sft/data_events_c/
#   d               - 数据集: sft/data_events_d/  (关键帧版，需 image 像素参数)
#   e               - 数据集: sft/data_events_e/
#   j               - 数据集: sft/data_events_j/  (caption + 1 关键帧/事件，复用 D 的 image 像素参数)
#
# 使用:
#   bash sft/sft_events.sh                  # baseline
#   PROMPT_STYLE=d bash sft/sft_events.sh   # 方案 D
# ============================================================

PROMPT_STYLE="${PROMPT_STYLE:-baseline}"

case "$PROMPT_STYLE" in
    baseline) DATA_DIR="sft/data_events"   ;;
    b)        DATA_DIR="sft/data_events_b" ;;
    c)        DATA_DIR="sft/data_events_c" ;;
    d)        DATA_DIR="sft/data_events_d" ;;
    e)        DATA_DIR="sft/data_events_e" ;;
    j)        DATA_DIR="sft/data_events_j" ;;
    *) echo "[ERROR] Unknown PROMPT_STYLE: $PROMPT_STYLE (expected: baseline|b|c|d|e|j)" >&2; exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-sft/ckpt/test_events_${PROMPT_STYLE}}"

echo "============================================"
echo "  SFT 训练  PROMPT_STYLE=$PROMPT_STYLE"
echo "  数据目录: $DATA_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "============================================"

# Image 像素参数（方案 D / J 必需；其他方案无 <image> 输入时无影响）
# MAX_PIXELS / MIN_PIXELS 控制 Qwen2.5-VL 单张图的像素上下限
MAX_PIXELS="${MAX_PIXELS:-501760}"
MIN_PIXELS="${MIN_PIXELS:-50176}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MAX_PIXELS=$MAX_PIXELS \
MIN_PIXELS=$MIN_PIXELS \
VIDEO_MIN_PIXELS=50176 \
VIDEO_MAX_PIXELS=50176 \
DECORD_EOF_RETRY_MAX=20480 \
FPS_MAX_FRAMES=512 \
swift sft \
    --model /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset $DATA_DIR/wo_tool_call/activitynet.jsonl \
            $DATA_DIR/wo_tool_call/charades.jsonl \
            $DATA_DIR/wo_tool_call/vidchapters.jsonl \
            $DATA_DIR/wo_tool_call/video_r1_image_mc.jsonl \
            $DATA_DIR/wo_tool_call/video_r1_video.jsonl \
            $DATA_DIR/wi_tool_call/activitynet.jsonl \
            $DATA_DIR/wi_tool_call/qvhighlight.jsonl \
            $DATA_DIR/wi_tool_call/longvila.jsonl \
    --torch_dtype bfloat16 \
    --external_plugins sft/loss_scale_plugin.py \
    --loss_scale last_two_rounds \
    --freeze_vit True \
    --freeze_aligner False \
    --freeze_llm False \
    --gradient_checkpointing True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 32 \
    --save_only_model False \
    --save_strategy epoch \
    --save_total_limit 10 \
    --logging_steps 2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --attn_impl flash_attn \
    --deepspeed zero3 \
    --output_dir $OUTPUT_DIR
