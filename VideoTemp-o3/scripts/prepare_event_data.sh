#!/bin/bash
# ============================================================
# 一键生成事件定位版训练数据
#
# 支持 6 套 prompt 风格（通过 PROMPT_STYLE 环境变量选择）：
#   baseline (默认) - convert_annotations.py                  → sft/data_events    & rl/data_events
#   b               - convert_annotations_b.py                → sft/data_events_b  & rl/data_events_b
#   c               - convert_annotations_c.py                → sft/data_events_c  & rl/data_events_c
#   d               - convert_annotations_d.py                → sft/data_events_d  & rl/data_events_d (关键帧版)
#   e               - convert_annotations_e.py                → sft/data_events_e  & rl/data_events_e
#   j               - plan_j/convert_annotations_j.py         → sft/data_events_j  & rl/data_events_j (caption + 1 关键帧)
#                     需先运行: python scripts/plan_j/generate_event_captions.py
#
# 使用方法:
#   cd VideoR/VideoTemp-o3
#   bash scripts/prepare_event_data.sh                     # baseline
#   PROMPT_STYLE=d bash scripts/prepare_event_data.sh      # 方案 D
#
# 可配置的环境变量:
#   PROMPT_STYLE  - prompt 风格，详见上方 (默认: baseline)
#   CLIP_MODEL    - CLIP 模型路径 (默认: /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/siglip2-so400m-patch16-512)
#   SAMPLE_FPS    - 帧采样率 Hz  (默认: 2.0)
#   KERNEL_SIZE   - 卷积核大小    (默认: 5)
#   BATCH_SIZE    - CLIP 推理批大小 (默认: 64)
#   DEVICE        - 计算设备       (默认: cuda)
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 可配置参数（通过环境变量覆盖）
CLIP_MODEL="${CLIP_MODEL:-/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/siglip2-so400m-patch16-512}"
SAMPLE_FPS="${SAMPLE_FPS:-2.0}"
KERNEL_SIZE="${KERNEL_SIZE:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
PROMPT_STYLE="${PROMPT_STYLE:-baseline}"

# Prompt 风格 → 转换脚本 + 输出目录后缀
case "$PROMPT_STYLE" in
    baseline) CONVERT_SCRIPT="scripts/convert_annotations.py";          SUFFIX="" ;;
    b)        CONVERT_SCRIPT="scripts/convert_annotations_b.py";        SUFFIX="_b" ;;
    c)        CONVERT_SCRIPT="scripts/convert_annotations_c.py";        SUFFIX="_c" ;;
    d)        CONVERT_SCRIPT="scripts/convert_annotations_d.py";        SUFFIX="_d" ;;
    e)        CONVERT_SCRIPT="scripts/convert_annotations_e.py";        SUFFIX="_e" ;;
    j)        CONVERT_SCRIPT="scripts/plan_j/convert_annotations_j.py"; SUFFIX="_j" ;;
    *) echo "[ERROR] Unknown PROMPT_STYLE: $PROMPT_STYLE (expected: baseline|b|c|d|e|j)" >&2; exit 1 ;;
esac

# 方案 J 需要事件级 caption metadata，缺失时给出明确提示
if [ "$PROMPT_STYLE" = "j" ] && [ ! -f "${EVENT_CAPTIONS:-scripts/plan_j/event_captions.json}" ]; then
    echo "[WARN] event_captions.json 不存在: ${EVENT_CAPTIONS:-scripts/plan_j/event_captions.json}"
    echo "       所有事件 caption 将兜底为 '(no description)'。"
    echo "       建议先运行: python scripts/plan_j/generate_event_captions.py --mode auto ..."
fi

SFT_OUTPUT="sft/data_events${SUFFIX}"
RL_OUTPUT="rl/data_events${SUFFIX}"

echo "============================================"
echo "  事件定位版数据准备"
echo "  项目目录:     $PROJECT_DIR"
echo "  PROMPT_STYLE: $PROMPT_STYLE"
echo "  转换脚本:     $CONVERT_SCRIPT"
echo "  SFT 输出:     $SFT_OUTPUT"
echo "  RL  输出:     $RL_OUTPUT"
echo ""
echo "  场景检测算法: Adaptive Event Segmentation"
echo "  CLIP 模型:    $CLIP_MODEL"
echo "  采样帧率:     $SAMPLE_FPS fps"
echo "  卷积核大小:   ${KERNEL_SIZE}×${KERNEL_SIZE}"
echo "  批大小:       $BATCH_SIZE"
echo "  计算设备:     $DEVICE"
echo "============================================"

# -----------------------------------------------------------
# Step 1: 场景预处理 (Adaptive Event Segmentation)
#         所有 prompt 风格共享同一份 scene_metadata.json
# -----------------------------------------------------------
echo ""
echo "[1/3] 场景预处理（CLIP CLS → TSM → 对角差分卷积核 → 事件边界）..."
echo "  这一步需要 GPU 和 CLIP 模型，耗时取决于视频数量..."

if [ -f scripts/scene_metadata.json ]; then
    echo "  scene_metadata.json 已存在，跳过。如需重新生成请先删除。"
else
    python scripts/preprocess_scenes.py \
        --data_dirs sft/data rl/data \
        --output scripts/scene_metadata.json \
        --clip_model "$CLIP_MODEL" \
        --sample_fps "$SAMPLE_FPS" \
        --kernel_size "$KERNEL_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE"
fi

# -----------------------------------------------------------
# Step 2: SFT 标注转换
# -----------------------------------------------------------
echo ""
echo "[2/3] SFT 标注转换（timestamp → event_ids）..."

if [ -d "$SFT_OUTPUT" ]; then
    echo "  $SFT_OUTPUT/ 已存在，跳过。如需重新生成请先删除。"
else
    python "$CONVERT_SCRIPT" \
        --metadata scripts/scene_metadata.json \
        --input_dir sft/data \
        --output_dir "$SFT_OUTPUT" \
        --data_stage sft
fi

# -----------------------------------------------------------
# Step 3: RL 标注转换
# -----------------------------------------------------------
echo ""
echo "[3/3] RL 标注转换（timestamp → event_ids）..."

if [ -d "$RL_OUTPUT" ]; then
    echo "  $RL_OUTPUT/ 已存在，跳过。如需重新生成请先删除。"
else
    python "$CONVERT_SCRIPT" \
        --metadata scripts/scene_metadata.json \
        --input_dir rl/data \
        --output_dir "$RL_OUTPUT" \
        --data_stage rl
fi

# -----------------------------------------------------------
# 完成
# -----------------------------------------------------------
echo ""
echo "============================================"
echo "  事件定位版数据准备完成！(PROMPT_STYLE=$PROMPT_STYLE)"
echo ""
echo "  生成的文件:"
echo "    - scripts/scene_metadata.json  (场景元数据)"
echo "    - $SFT_OUTPUT/                  (SFT 事件标注)"
echo "    - $RL_OUTPUT/                  (RL 事件标注)"
echo ""
echo "  下一步:"
echo "    1. SFT 训练: PROMPT_STYLE=$PROMPT_STYLE bash sft/sft_events.sh"
echo "    2. RL 训练:  PROMPT_STYLE=$PROMPT_STYLE bash rl/rollout_events.sh   (终端1)"
echo "                 PROMPT_STYLE=$PROMPT_STYLE bash rl/grpo_events.sh      (终端2)"
echo "============================================"
