#!/bin/bash
# ============================================================
# 一键生成事件定位版训练数据
#
# 步骤:
# 1. 场景预处理: Adaptive Event Segmentation → scene_metadata.json
#    使用 CLIP [CLS] token 构建 TSM + 对角差分卷积核检测事件边界
# 2. SFT 标注转换: sft/data → sft/data_events
# 3. RL 标注转换:  rl/data  → rl/data_events
#
# 使用方法:
#   cd VideoR/VideoTemp-o3
#   bash scripts/prepare_event_data.sh
#
# 可配置的环境变量:
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

echo "============================================"
echo "  事件定位版数据准备"
echo "  项目目录: $PROJECT_DIR"
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

if [ -d sft/data_events ]; then
    echo "  sft/data_events/ 已存在，跳过。如需重新生成请先删除。"
else
    python scripts/convert_annotations.py \
        --metadata scripts/scene_metadata.json \
        --input_dir sft/data \
        --output_dir sft/data_events \
        --data_stage sft
fi

# -----------------------------------------------------------
# Step 3: RL 标注转换
# -----------------------------------------------------------
echo ""
echo "[3/3] RL 标注转换（timestamp → event_ids）..."

if [ -d rl/data_events ]; then
    echo "  rl/data_events/ 已存在，跳过。如需重新生成请先删除。"
else
    python scripts/convert_annotations.py \
        --metadata scripts/scene_metadata.json \
        --input_dir rl/data \
        --output_dir rl/data_events \
        --data_stage rl
fi

# -----------------------------------------------------------
# 完成
# -----------------------------------------------------------
echo ""
echo "============================================"
echo "  事件定位版数据准备完成！"
echo ""
echo "  生成的文件:"
echo "    - scripts/scene_metadata.json  (场景元数据)"
echo "    - sft/data_events/             (SFT 事件标注)"
echo "    - rl/data_events/              (RL 事件标注)"
echo ""
echo "  下一步:"
echo "    1. SFT 训练: bash sft/sft_events.sh"
echo "    2. RL 训练:  bash rl/rollout_events.sh  (终端1)"
echo "                 bash rl/grpo_events.sh     (终端2)"
echo "============================================"
