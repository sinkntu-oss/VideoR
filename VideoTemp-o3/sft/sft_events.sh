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
# [M7] 数据集存在性校验：缺失或空文件立刻报错，避免训练启动到一半才崩
DATASET_FILES=(
    "$DATA_DIR/wo_tool_call/activitynet.jsonl"
    "$DATA_DIR/wo_tool_call/charades.jsonl"
    "$DATA_DIR/wo_tool_call/vidchapters.jsonl"
    "$DATA_DIR/wo_tool_call/video_r1_image_mc.jsonl"
    "$DATA_DIR/wo_tool_call/video_r1_video.jsonl"
    "$DATA_DIR/wi_tool_call/activitynet.jsonl"
    "$DATA_DIR/wi_tool_call/qvhighlight.jsonl"
    "$DATA_DIR/wi_tool_call/longvila.jsonl"
)
MISSING=()
for f in "${DATASET_FILES[@]}"; do
    if [ ! -s "$f" ]; then
        MISSING+=("$f")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "[ERROR] 以下数据文件缺失或为空（共 ${#MISSING[@]} 个）:" >&2
    for f in "${MISSING[@]}"; do echo "  - $f" >&2; done
    echo "[ERROR] 请先运行: PROMPT_STYLE=$PROMPT_STYLE bash scripts/prepare_event_data.sh" >&2
    echo "[ERROR] 或检查 $DATA_DIR/ 转换日志找出失败原因" >&2
    exit 1
fi
echo "[OK] 数据集文件校验通过（${#DATASET_FILES[@]} 个 jsonl 全部就绪）"

# [问题 3] D / J 关键帧版：在训练启动前做 chat template 冒烟测试，
#         确认 ms-swift 真的会消费 jsonl 顶层 images 字段（而不是静默丢弃）。
#         若失败，直接退出，避免训完才发现模型完全没看到关键帧。
if [[ "$PROMPT_STYLE" == "d" || "$PROMPT_STYLE" == "j" ]]; then
    if [ -f "scripts/plan_j/verify_sft_template.py" ]; then
        echo "[问题 3] 运行 SFT chat template 冒烟测试..."
        python scripts/plan_j/verify_sft_template.py "$DATA_DIR" --n_samples 3 || {
            echo "[ERROR] chat template 冒烟测试失败：ms-swift 未正确消费 images 字段" >&2
            echo "[ERROR] 详细原因见上方日志；不要启动训练。" >&2
            exit 1
        }
    fi
fi

swift sft \
    --model /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset "${DATASET_FILES[@]}" \
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
