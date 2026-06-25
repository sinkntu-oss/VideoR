#!/usr/bin/env python3
"""
Adaptive Event Segmentation 场景预处理脚本

基于论文方法：使用视觉编码器 [CLS] token embedding 构建时序相似度矩阵（TSM），
通过对角差分卷积核检测语义事件边界。

算法流程：
  1. 从视频中按 sample_fps 采样帧
  2. 使用 CLIP ViT 提取每帧的 [CLS] token embedding
  3. L2 归一化得到单位范数表示 {c_t}
  4. 计算帧间余弦相似度矩阵 TSM_{i,j} = c_i · c_j / (‖c_i‖·‖c_j‖)
  5. 构造 K×K 对角差分卷积核 K，对零填充 TSM 执行 Conv2D
  6. 提取卷积输出对角元素作为边界分数 s = diag(Conv2D(TSM, K))
  7. 自适应阈值 τ = mean(s)
  8. 边界集合 B = {t | s_{t-1}≤s_t≥s_{t+1}, s_t≥τ} ∪ {1, T}
  9. 将帧序列划分为 N=|B|-1 个变长事件段

使用方法:
    cd VideoR/VideoTemp-o3
    python scripts/preprocess_scenes.py \
        --data_dirs sft/data rl/data \
        --output scripts/scene_metadata.json \
        --clip_model openai/clip-vit-base-patch16 \
        --sample_fps 2.0 \
        --kernel_size 5 \
        --batch_size 64 \
        --device cuda
"""
import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "16" # 抑制mmco警告
import argparse
import json
import os
import sys
import glob
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 对角差分卷积核
# ============================================================

def build_diagonal_kernel(size: int = 5) -> np.ndarray:
    """
    构造对角差分卷积核（Diagonal Difference Convolution Kernel）。

    核结构（以 5×5 为例）：
        +1  +1   0  -1  -1
        +1  +1   0  -1  -1
         0   0   0   0   0
        -1  -1   0  +1  +1
        -1  -1   0  +1  +1

    - 对角块（左上 & 右下）：正权重 → 响应同事件内的高自相似度
    - 反对角块（右上 & 左下）：负权重 → 惩罚跨事件的低互相似度
    - 中心行/列：零权重 → 隔离两组块

    在事件边界处：对角块覆盖的 TSM 区域具有高相似度（同事件），
    反对角块覆盖的区域具有低相似度（跨事件），
    因此卷积响应产生显著的正峰值。
    在事件内部：所有块覆盖的 TSM 区域相似度相近，正负抵消 → 接近零。

    Args:
        size: 核大小（奇数），默认 5

    Returns:
        归一化的卷积核，形状 (size, size)
    """
    assert size % 2 == 1, f"核大小必须为奇数，得到 {size}"
    kernel = np.zeros((size, size), dtype=np.float32)
    half = size // 2

    for i in range(size):
        for j in range(size):
            if i == half or j == half:
                continue  # 中心行/列为零
            if (i < half) == (j < half):
                # 对角块：左上 (i<half, j<half) 和右下 (i>half, j>half)
                kernel[i, j] = 1.0
            else:
                # 反对角块：右上 (i<half, j>half) 和左下 (i>half, j<half)
                kernel[i, j] = -1.0

    # 归一化：除以正权重元素个数，使输出值域稳定
    num_positive = int(np.sum(kernel > 0))
    if num_positive > 0:
        kernel /= num_positive

    return kernel


# ============================================================
# TSM 构建与边界检测
# ============================================================

def build_tsm(features: np.ndarray) -> np.ndarray:
    """
    构建时序相似度矩阵 (Temporal Similarity Matrix)。

    TSM_{i,j} = c_i · c_j / (‖c_i‖ · ‖c_j‖)

    输入特征已 L2 归一化时，内积即余弦相似度。

    Args:
        features: L2 归一化后的特征矩阵，形状 (T, D)

    Returns:
        TSM 矩阵，形状 (T, T)，元素范围 [-1, 1]
    """
    tsm = features @ features.T
    # 数值稳定性：clamp 到 [-1, 1]
    np.clip(tsm, -1.0, 1.0, out=tsm)
    return tsm


def compute_boundary_scores(tsm: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    通过对角差分卷积计算边界分数。

    s = diag(Conv2D(TSM, K))

    优化实现：只计算卷积输出的对角线元素，
    计算复杂度 O(T × K²) 而非完整 Conv2D 的 O(T² × K²)。

    Args:
        tsm: 时序相似度矩阵，形状 (T, T)
        kernel: 对角差分卷积核，形状 (K, K)

    Returns:
        边界分数数组，形状 (T,)，值越高表示边界可能性越大
    """
    T = tsm.shape[0]
    K = kernel.shape[0]
    pad = K // 2

    # 零填充 TSM (论文公式中的 zero-padded TSM)
    tsm_padded = np.pad(tsm, pad, mode='constant', constant_values=0)

    # 只计算对角线位置 (t, t) 处的卷积响应
    scores = np.zeros(T, dtype=np.float32)
    for t in range(T):
        # 提取以 (t+pad, t+pad) 为中心的 K×K 窗口
        patch = tsm_padded[t:t + K, t:t + K]
        scores[t] = np.sum(patch * kernel)

    return scores


def detect_event_boundaries(scores: np.ndarray) -> List[int]:
    """
    使用自适应阈值检测事件边界。

    边界集合 B = {t | s_{t-1} ≤ s_t ≥ s_{t+1}, s_t ≥ τ} ∪ {1, T}
    其中 τ = mean(s)

    Args:
        scores: 边界分数数组，形状 (T,)

    Returns:
        边界帧索引列表（已排序），作为采样帧空间中的索引
    """
    T = len(scores)

    if T < 3:
        # 帧数太少，整段视频作为一个事件
        return list(range(T))

    # 自适应阈值 τ = mean(s)
    tau = np.mean(scores)

    # 检测局部极大值且超过阈值
    boundaries = set()
    boundaries.add(0)        # 首帧始终为边界
    boundaries.add(T - 1)    # 末帧始终为边界

    for t in range(1, T - 1):
        if scores[t] >= tau and scores[t - 1] <= scores[t] and scores[t] >= scores[t + 1]:
            boundaries.add(t)

    return sorted(boundaries)


# ============================================================
# SigLIP2 特征提取
# ============================================================

def load_siglip_model(model_name: str, device: str):
    """
    加载 SigLIP2 视觉编码器。

    Args:
        model_name: HuggingFace 模型名称或本地路径
        device: 计算设备 ('cuda', 'cpu')

    Returns:
        (model, processor) 元组
    """
    import torch
    from transformers import SiglipVisionModel, SiglipImageProcessor

    logger.info(f"加载 SigLIP2 模型: {model_name}")
    logger.info(f"计算设备: {device}")

    model = SiglipVisionModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    )
    processor = SiglipImageProcessor.from_pretrained(model_name)
    model = model.to(device).eval()

    # 获取特征维度
    feature_dim = model.config.hidden_size
    logger.info(f"SigLIP2 特征维度: {feature_dim}")

    return model, processor


def extract_cls_features(
    frames: List[np.ndarray],
    model,
    processor,
    batch_size: int = 64,
    device: str = 'cuda'
) -> np.ndarray:
    """
    使用 SigLIP2 视觉编码器提取 [CLS] token embedding。

    对每帧图像通过 SigLIP2 视觉编码器前向传播，
    取 [CLS] token 的特征作为紧凑的帧级描述符。
    输出经 L2 归一化得到单位范数表示。

    Args:
        frames: 帧列表，每帧为 (H, W, C) uint8 numpy 数组
        model: SigLIP2 模型
        processor: SigLIP2 图像处理器
        batch_size: 每批帧数
        device: 计算设备

    Returns:
        L2 归一化的特征矩阵，形状 (T, D)
    """
    import torch
    from PIL import Image

    all_features = []

    # numpy 帧转 PIL Image
    pil_frames = [Image.fromarray(f) for f in frames]

    for i in range(0, len(pil_frames), batch_size):
        batch = pil_frames[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # SigLIP2: 取 [CLS] token (第一个 token) 的特征
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]

        all_features.append(features.float().cpu().numpy())

    features = np.concatenate(all_features, axis=0)  # (T, D)

    # L2 归一化 → 单位范数表示
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-8)

    return features


# ============================================================
# 视频帧采样
# ============================================================

def sample_video_frames(
    video_path: str,
    sample_fps: float = 2.0,
    use_gpu: bool = True
) -> Tuple[List[np.ndarray], List[int], float, int]:
    """
    从视频中按目标帧率均匀采样帧。

    Args:
        video_path: 视频文件路径
        sample_fps: 采样帧率（Hz）
        use_gpu: 是否使用 GPU 硬件加速解码（NVIDIA GPU 需要 CUDA 支持）

    Returns:
        (frames, sample_indices, orig_fps, total_frames)
        - frames: 采样帧列表，每帧 (H, W, C) uint8
        - sample_indices: 每帧在原始视频中的帧索引
        - orig_fps: 原始视频帧率
        - total_frames: 原始视频总帧数
    """
    from decord import VideoReader, cpu, gpu

    try:
        # 尝试使用 GPU 硬件加速解码（NVIDIA NVDEC）
        if use_gpu:
            try:
                vr = VideoReader(video_path, ctx=gpu(0))
                logger.debug(f"使用 GPU 硬件加速解码: {video_path}")
            except Exception as e:
                logger.debug(f"GPU 解码失败，回退到 CPU: {e}")
                vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        logger.warning(f"打开视频失败 {video_path}: {e}")
        return [], [], 0.0, 0

    orig_fps = vr.get_avg_fps()
    total_frames = len(vr)

    if total_frames < 2 or orig_fps <= 0:
        return [], [], orig_fps, total_frames

    # 按 sample_fps 均匀采样
    sample_interval = max(1, int(round(orig_fps / sample_fps)))
    sample_indices = list(range(0, total_frames, sample_interval))

    # 确保至少 2 帧
    if len(sample_indices) < 2:
        sample_indices = [0, total_frames - 1]

    # 确保最后一帧在列表中（覆盖视频末尾）
    if sample_indices[-1] != total_frames - 1:
        sample_indices.append(total_frames - 1)

    # Decord 批量读取
    frames_batch = vr.get_batch(sample_indices).asnumpy()  # (T, H, W, C)
    frames = [frames_batch[i] for i in range(len(frames_batch))]

    return frames, sample_indices, orig_fps, total_frames


# ============================================================
# 单视频完整处理流程
# ============================================================

def process_single_video(
    video_path: str,
    model,
    processor,
    sample_fps: float = 2.0,
    kernel_size: int = 5,
    batch_size: int = 64,
    device: str = 'cuda',
    use_gpu_decode: bool = True
) -> Optional[Dict]:
    """
    对单个视频执行 Adaptive Event Segmentation。

    完整流程：采样帧 → CLS 特征 → TSM → 卷积核 → 边界检测 → 事件列表

    Args:
        video_path: 视频文件路径
        model: CLIP 模型
        processor: CLIP 图像处理器
        sample_fps: 帧采样率
        kernel_size: 对角差分卷积核大小
        batch_size: CLIP 推理批大小
        device: 计算设备
        use_gpu_decode: 是否使用 GPU 硬件加速解码

    Returns:
        视频事件元数据字典，包含 events 列表
    """
    if not os.path.exists(video_path):
        logger.warning(f"视频不存在: {video_path}")
        return None

    try:
        # 1. 采样帧（使用 GPU 硬件加速解码）
        frames, sample_indices, orig_fps, total_frames = sample_video_frames(
            video_path, sample_fps, use_gpu=use_gpu_decode
        )

        duration = total_frames / orig_fps if orig_fps > 0 else 0.0

        if len(frames) < 3:
            # 帧数太少，整段视频作为一个事件
            return {
                "video_path": video_path,
                "orig_fps": round(orig_fps, 2),
                "total_frames": total_frames,
                "duration": round(duration, 2),
                "sample_fps": sample_fps,
                "num_events": 1,
                "events": [{
                    "event_id": 0,
                    "start_frame": 0,
                    "end_frame": total_frames,
                    "start_time": 0.0,
                    "end_time": round(duration, 2),
                    "num_frames": total_frames
                }]
            }

        # 2. 提取 CLS token embedding
        cls_features = extract_cls_features(
            frames, model, processor,
            batch_size=batch_size, device=device
        )  # (T, D), L2 归一化

        # 3. 构建时序相似度矩阵 TSM
        tsm = build_tsm(cls_features)  # (T, T)

        # 4. 对角差分卷积 → 边界分数
        kernel = build_diagonal_kernel(kernel_size)
        scores = compute_boundary_scores(tsm, kernel)  # (T,)

        # 5. 自适应阈值 + 局部极大值 → 事件边界
        boundary_indices_sampled = detect_event_boundaries(scores)  # 采样帧空间

        # 6. 映射回原始视频帧索引
        boundary_frames_orig = []
        for b_idx in boundary_indices_sampled:
            if b_idx < len(sample_indices):
                boundary_frames_orig.append(sample_indices[b_idx])
            else:
                boundary_frames_orig.append(total_frames)

        # 确保包含首帧 0 和末帧 total_frames
        if boundary_frames_orig[0] != 0:
            boundary_frames_orig.insert(0, 0)
        # 末尾使用 total_frames 作为 exclusive 边界
        if boundary_frames_orig[-1] != total_frames:
            boundary_frames_orig.append(total_frames)

        # 去重并排序
        boundary_frames_orig = sorted(set(boundary_frames_orig))

        # 7. 构建事件列表
        events = []
        for i in range(len(boundary_frames_orig) - 1):
            start_frame = boundary_frames_orig[i]
            end_frame = boundary_frames_orig[i + 1]
            if end_frame <= start_frame:
                continue
            events.append({
                "event_id": len(events),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time": round(start_frame / orig_fps, 2),
                "end_time": round(end_frame / orig_fps, 2),
                "num_frames": int(end_frame - start_frame)
            })

        # 如果没有检测到有效事件，退化为单事件
        if not events:
            events = [{
                "event_id": 0,
                "start_frame": 0,
                "end_frame": total_frames,
                "start_time": 0.0,
                "end_time": round(duration, 2),
                "num_frames": total_frames
            }]

        return {
            "video_path": video_path,
            "orig_fps": round(orig_fps, 2),
            "total_frames": total_frames,
            "duration": round(duration, 2),
            "sample_fps": sample_fps,
            "num_events": len(events),
            "num_sampled_frames": len(frames),
            "boundary_scores_stats": {
                "mean": round(float(np.mean(scores)), 6),
                "std": round(float(np.std(scores)), 6),
                "max": round(float(np.max(scores)), 6),
                "min": round(float(np.min(scores)), 6),
                "threshold": round(float(np.mean(scores)), 6),
            },
            "events": events
        }

    except Exception as e:
        logger.error(f"处理视频失败 {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 数据收集
# ============================================================

def collect_video_paths(data_dirs: List[str], project_root: str) -> set:
    """
    从所有 JSONL 数据文件中收集视频路径。

    Args:
        data_dirs: 数据目录列表
        project_root: 项目根目录

    Returns:
        去重的视频绝对路径集合
    """
    video_paths = set()

    for data_dir in data_dirs:
        jsonl_files = glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)
        for jsonl_file in jsonl_files:
            logger.info(f"扫描数据文件: {jsonl_file}")
            try:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        videos = item.get("videos", [])
                        for vp in videos:
                            # 过滤已裁剪的视频
                            if "cropped_video" in vp:
                                continue
                            abs_path = os.path.join(project_root, vp) if not os.path.isabs(vp) else vp
                            video_paths.add(abs_path)
            except Exception as e:
                logger.error(f"读取 {jsonl_file} 失败: {e}")

    logger.info(f"共收集到 {len(video_paths)} 个唯一视频文件")
    return video_paths


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Event Segmentation 场景预处理 (SigLIP2 + 多线程)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 SigLIP2，2fps 采样，5×5 核，4 线程
  python scripts/preprocess_scenes.py \\
      --data_dirs sft/data rl/data \\
      --output scripts/scene_metadata.json \\
      --clip_model google/siglip-so400m-patch14-384 \\
      --sample_fps 2.0 --kernel_size 5 --num_threads 4

  # 使用本地 SigLIP2 模型路径
  python scripts/preprocess_scenes.py \\
      --clip_model /path/to/siglip-so400m-patch14-384 \\
      --device cuda:0 --num_threads 8

  # CPU 模式（较慢）
  python scripts/preprocess_scenes.py \\
      --device cpu --batch_size 16 --num_threads 2
        """
    )
    parser.add_argument("--data_dirs", nargs="+", default=["sft/data", "rl/data"],
                        help="包含 JSONL 数据的目录列表 (默认: sft/data rl/data)")
    parser.add_argument("--output", type=str, default="scripts/scene_metadata.json",
                        help="输出元数据文件路径 (默认: scripts/scene_metadata.json)")
    parser.add_argument("--project_root", type=str, default=".",
                        help="项目根目录 (默认: 当前目录)")
    parser.add_argument("--clip_model", type=str, default="/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/siglip2-so400m-patch16-512",
                        help="SigLIP2 模型名称或本地路径 (默认: /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/siglip2-so400m-patch16-512)")
    parser.add_argument("--sample_fps", type=float, default=2.0,
                        help="帧采样率 Hz (默认: 2.0)")
    parser.add_argument("--kernel_size", type=int, default=5,
                        help="对角差分卷积核大小，必须为奇数 (默认: 5)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="SigLIP2 推理批大小 (默认: 64)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备 (默认: cuda)")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="视频处理线程数 (默认: 4)")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"算法参数: sample_fps={args.sample_fps}, kernel_size={args.kernel_size}")

    # 验证参数
    assert args.kernel_size % 2 == 1, f"kernel_size 必须为奇数，得到 {args.kernel_size}"
    assert args.sample_fps > 0, f"sample_fps 必须为正数，得到 {args.sample_fps}"
    assert args.num_threads > 0, f"num_threads 必须为正数，得到 {args.num_threads}"

    # 1. 加载 SigLIP2 模型（只加载一次）
    model, processor = load_siglip_model(args.clip_model, args.device)

    # 2. 收集所有视频路径
    video_paths = collect_video_paths(args.data_dirs, project_root)

    if not video_paths:
        logger.error("没有找到任何视频文件")
        sys.exit(1)

    # 3. 多线程并行处理视频
    metadata = {}
    failed = []
    metadata_lock = threading.Lock()

    logger.info(f"开始处理 {len(video_paths)} 个视频...")
    logger.info(f"SigLIP2 模型: {args.clip_model}")
    logger.info(f"设备: {args.device} | 批大小: {args.batch_size} | 线程数: {args.num_threads}")

    def process_video_worker(video_path: str):
        """单个线程处理一个视频"""
        try:
            result = process_single_video(
                video_path=video_path,
                model=model,
                processor=processor,
                sample_fps=args.sample_fps,
                kernel_size=args.kernel_size,
                batch_size=args.batch_size,
                device=args.device,
                use_gpu_decode=True
            )
            
            with metadata_lock:
                if result is not None:
                    rel_path = os.path.relpath(video_path, project_root)
                    metadata[rel_path] = result
                else:
                    failed.append(video_path)
        except Exception as e:
            logger.error(f"处理视频异常 {video_path}: {e}")
            with metadata_lock:
                failed.append(video_path)

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_video_worker, vp) for vp in sorted(video_paths)]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Adaptive Event Segmentation"):
            pass

    # 4. 保存结果
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 5. 统计信息
    total_events = sum(m["num_events"] for m in metadata.values())
    avg_events = total_events / len(metadata) if metadata else 0
    event_counts = [m["num_events"] for m in metadata.values()]

    logger.info(f"\n{'='*60}")
    logger.info(f"Adaptive Event Segmentation 完成！")
    logger.info(f"  算法: SigLIP2 CLS → TSM → {args.kernel_size}×{args.kernel_size} 对角差分卷积核 → 自适应阈值")
    logger.info(f"  SigLIP2 模型: {args.clip_model}")
    logger.info(f"  采样帧率: {args.sample_fps} fps")
    logger.info(f"  处理线程: {args.num_threads}")
    logger.info(f"  成功: {len(metadata)} 个视频")
    logger.info(f"  失败: {len(failed)} 个视频")
    logger.info(f"  总事件数: {total_events}")
    logger.info(f"  平均每视频事件数: {avg_events:.1f}")
    if event_counts:
        logger.info(f"  事件数分布: min={min(event_counts)}, "
                     f"median={sorted(event_counts)[len(event_counts)//2]}, "
                     f"max={max(event_counts)}")
    logger.info(f"  输出文件: {args.output}")
    logger.info(f"{'='*60}")

    if failed:
        logger.warning(f"失败的视频:\n" + "\n".join(f"  - {p}" for p in failed[:20]))
        if len(failed) > 20:
            logger.warning(f"  ... 及其他 {len(failed) - 20} 个")


if __name__ == "__main__":
    main()
