#!/bin/bash
# ============================================================
# 事件定位版 GRPO 训练启动脚本
# 使用 video_event_plugin.py 和 event_reward 替代原有版本
#
# 支持 6 套 prompt 风格（通过 PROMPT_STYLE 环境变量选择）：
#   baseline (默认) - 数据集: rl/data_events/
#   b               - 数据集: rl/data_events_b/
#   c               - 数据集: rl/data_events_c/
#   d               - 数据集: rl/data_events_d/   (关键帧版)
#   e               - 数据集: rl/data_events_e/
#   j               - 数据集: rl/data_events_j/   (caption + 1 关键帧/事件)
#
# 使用:
#   bash rl/grpo_events.sh                  # baseline
#   PROMPT_STYLE=d bash rl/grpo_events.sh   # 方案 D
# ============================================================

PROMPT_STYLE="${PROMPT_STYLE:-baseline}"

case "$PROMPT_STYLE" in
    baseline) DATA_DIR="rl/data_events"   ;;
    b)        DATA_DIR="rl/data_events_b" ;;
    c)        DATA_DIR="rl/data_events_c" ;;
    d)        DATA_DIR="rl/data_events_d" ;;
    e)        DATA_DIR="rl/data_events_e" ;;
    j)        DATA_DIR="rl/data_events_j" ;;
    *) echo "[ERROR] Unknown PROMPT_STYLE: $PROMPT_STYLE (expected: baseline|b|c|d|e|j)" >&2; exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-rl/ckpt/test_events_${PROMPT_STYLE}}"
# 默认指向同 PROMPT_STYLE 的 SFT 输出目录；多 checkpoint 时请通过 MODEL=... 显式指定到具体 checkpoint-xxx 子目录
MODEL="${MODEL:-sft/ckpt/test_events_${PROMPT_STYLE}}"

echo "============================================"
echo "  GRPO 训练  PROMPT_STYLE=$PROMPT_STYLE"
echo "  起点模型: $MODEL"
echo "  数据目录: $DATA_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  (如需指定具体 checkpoint，请用 MODEL=sft/ckpt/test_events_${PROMPT_STYLE}/checkpoint-xxx bash rl/grpo_events.sh)"
echo "============================================"

# Image 像素参数（方案 D / J 必需；其他方案无 <image> 输入时无影响）
MAX_PIXELS="${MAX_PIXELS:-501760}"
MIN_PIXELS="${MIN_PIXELS:-50176}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=$MAX_PIXELS \
MIN_PIXELS=$MIN_PIXELS \
DECORD_EOF_RETRY_MAX=20480 \
VIDEO_MIN_PIXELS=50176 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=512 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL \
    --train_type full \
    --external_plugins rl/video_event_plugin.py \
    --reward_funcs acc_reward event_reward tool_penalty format_reward \
    --reward_weights 1.0 0.5 1.0 0.2 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 0.0.0.0 \
    --vllm_server_port 8100 \
    --vllm_server_pass_dataset true \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner false \
    --freeze_llm false \
    --dataset $DATA_DIR/qa.jsonl $DATA_DIR/grounding.jsonl \
    --split_dataset_ratio 0 \
    --max_completion_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 12 \
    --save_only_model false \
    --save_strategy 'steps' \
    --save_steps 50 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --num_generations 8 \
    --temperature 1.0 \
    --log_completions true \
    --log_entropy true \
    --steps_per_generation 12 \
    --num_iterations 1 \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --report_to tensorboard \
    --output_dir $OUTPUT_DIR
