#!/bin/bash
# ============================================================
# VideoTemp-o3 数据组织脚本
# 将 datasets/VideoTemp-o3 下的数据通过软链接组织到项目期望的位置
# 使用方法: cd VideoR/VideoTemp-o3 && bash setup_data.sh
# ============================================================
set -e

DATA_SRC="/mnt/tidal-alsh01/dataset/eam_ds/datasets/VideoTemp-o3"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  VideoTemp-o3 数据组织"
echo "  数据源: $DATA_SRC"
echo "  项目目录: $PROJECT_DIR"
echo "============================================"

# -----------------------------------------------------------
# 1. 组织 SFT 数据
# -----------------------------------------------------------
echo ""
echo "[1/3] 组织 SFT 数据..."

# 创建目标目录
mkdir -p "$PROJECT_DIR/sft/data/wo_tool_call"
mkdir -p "$PROJECT_DIR/sft/data/wi_tool_call"

# wo_tool_call (冷启动数据，不含工具调用)
for f in activitynet.jsonl charades.jsonl vidchapters.jsonl video_r1_image_mc.jsonl video_r1_video.jsonl; do
    src="$DATA_SRC/sft/$f"
    dst="$PROJECT_DIR/sft/data/wo_tool_call/$f"
    if [ -f "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
        echo "  链接: sft/data/wo_tool_call/$f"
    elif [ -e "$dst" ]; then
        echo "  已存在: sft/data/wo_tool_call/$f"
    else
        echo "  警告: 源文件不存在 $src"
    fi
done

# wi_tool_call (含工具调用数据)
for f in activitynet.jsonl longvila.jsonl qvhighlight.jsonl; do
    src="$DATA_SRC/sft_tool_call/$f"
    dst="$PROJECT_DIR/sft/data/wi_tool_call/$f"
    if [ -f "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
        echo "  链接: sft/data/wi_tool_call/$f"
    elif [ -e "$dst" ]; then
        echo "  已存在: sft/data/wi_tool_call/$f"
    else
        echo "  警告: 源文件不存在 $src"
    fi
done

# -----------------------------------------------------------
# 2. 组织 RL 数据
# -----------------------------------------------------------
echo ""
echo "[2/3] 组织 RL 数据..."

mkdir -p "$PROJECT_DIR/rl/data"

# qa 数据 (源文件叫 qa-1k.jsonl，项目期望叫 qa.jsonl)
src="$DATA_SRC/rl/qa-1k.jsonl"
dst="$PROJECT_DIR/rl/data/qa.jsonl"
if [ -f "$src" ] && [ ! -e "$dst" ]; then
    ln -s "$src" "$dst"
    echo "  链接: rl/data/qa.jsonl <- qa-1k.jsonl"
elif [ -e "$dst" ]; then
    echo "  已存在: rl/data/qa.jsonl"
else
    echo "  警告: 源文件不存在 $src"
fi

# grounding 数据
src="$DATA_SRC/rl/grounding.jsonl"
dst="$PROJECT_DIR/rl/data/grounding.jsonl"
if [ -f "$src" ] && [ ! -e "$dst" ]; then
    ln -s "$src" "$dst"
    echo "  链接: rl/data/grounding.jsonl"
elif [ -e "$dst" ]; then
    echo "  已存在: rl/data/grounding.jsonl"
else
    echo "  警告: 源文件不存在 $src"
fi

# -----------------------------------------------------------
# 3. 链接视频目录 (SFT 和 RL 视频)
# -----------------------------------------------------------
echo ""
echo "[3/3] 链接视频目录..."

# SFT 视频 - jsonl 中的视频路径是相对路径 (如 ActivityNet/videos/xxx.mp4)
# 这些视频在 sft_videos tar 包中，需要先解压
# 这里先链接 sft_videos 目录，后续解压后视频即可使用
if [ -d "$DATA_SRC/sft_videos" ] && [ ! -e "$PROJECT_DIR/sft_videos" ]; then
    ln -s "$DATA_SRC/sft_videos" "$PROJECT_DIR/sft_videos"
    echo "  链接: sft_videos/"
elif [ -e "$PROJECT_DIR/sft_videos" ]; then
    echo "  已存在: sft_videos/"
fi

# RL 视频
if [ -d "$DATA_SRC/rl_videos" ] && [ ! -e "$PROJECT_DIR/rl_videos" ]; then
    ln -s "$DATA_SRC/rl_videos" "$PROJECT_DIR/rl_videos"
    echo "  链接: rl_videos/"
elif [ -e "$PROJECT_DIR/rl_videos" ]; then
    echo "  已存在: rl_videos/"
fi

echo ""
echo "============================================"
echo "  数据组织完成！"
echo ""
echo "  注意: 视频文件需要先解压 tar.gz 才能使用:"
echo "    cd $PROJECT_DIR"
echo "    # 解压 SFT 视频"
echo "    for f in sft_videos/videos_part_*.tar.gz; do tar xzf \$f; done"
echo "    # 解压 RL 视频"
echo "    for f in rl_videos/videos_part_*.tar.gz; do tar xzf \$f; done"
echo "============================================"
