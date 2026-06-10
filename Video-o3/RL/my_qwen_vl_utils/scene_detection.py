"""
基于帧差分析的场景边界检测。
实现对角差分矩阵方法用于检测视频场景转换。
"""

import numpy as np
import torch
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def compute_frame_difference_matrix(video_tensor: torch.Tensor, method: str = 'l2') -> np.ndarray:
    """
    使用对角矩阵方法计算帧对间的差异。

    对于对角线元素（相邻帧），计算相邻帧之间的差异以检测场景转换。

    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
        method: 距离度量 - 'l2'（默认）或 'l1'

    Returns:
        形状为 (T-1,) 的 np.ndarray，包含相邻帧之间的差异大小
    """
    if video_tensor.shape[0] < 2:
        return np.array([])

    # 如果需要转换为浮点数
    if video_tensor.dtype == torch.uint8:
        video_float = video_tensor.float() / 255.0
    else:
        video_float = video_tensor.float()

    # 重塑: (T, C, H, W) -> (T, C*H*W) 便于计算
    T, C, H, W = video_tensor.shape
    video_flat = video_float.reshape(T, -1)

    # 计算相邻帧之间的差异
    diffs = video_flat[1:] - video_flat[:-1]

    if method == 'l2':
        distances = torch.norm(diffs, p=2, dim=1).cpu().numpy()
    elif method == 'l1':
        distances = torch.norm(diffs, p=1, dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown method: {method}")

    return distances


def detect_scene_boundaries(
    video_tensor: torch.Tensor,
    percentile: int = 90,
    min_scene_length: int = 1,
    method: str = 'l2'
) -> List[int]:
    """
    使用自适应阈值对帧差异进行场景边界检测。

    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
        percentile: 自适应阈值计算的百分位数（默认90）
        min_scene_length: 有效场景所需的最少帧数（默认1）
        method: 距离度量 - 'l2'（默认）或 'l1'

    Returns:
        场景边界出现的帧索引列表（新场景第一帧的索引）。
        索引0始终包含在内（第一个场景的开始）。
    """
    if video_tensor.shape[0] < 2:
        return [0]

    # 计算帧差异
    diffs = compute_frame_difference_matrix(video_tensor, method=method)

    if len(diffs) == 0:
        return [0]

    # 从分布计算自适应阈值
    threshold = np.percentile(diffs, percentile)

    logger.info(f"场景检测 - 百分位: {percentile}, 阈值: {threshold:.6f}, "
                f"最小差异: {diffs.min():.6f}, 最大差异: {diffs.max():.6f}, "
                f"平均差异: {diffs.mean():.6f}")

    # 找出差异超过阈值的索引
    boundary_candidates = np.where(diffs > threshold)[0] + 1  # +1 因为 diffs 长度为 T-1

    # 过滤掉距离太近的边界
    boundaries = [0]  # 总是包含第一帧
    for boundary in boundary_candidates:
        if boundary - boundaries[-1] >= min_scene_length:
            boundaries.append(boundary)

    logger.info(f"检测到 {len(boundaries)} 个场景边界，位置: {boundaries}")

    return boundaries


def group_frames_by_scene(
    video_tensor: torch.Tensor,
    boundaries: List[int]
) -> List[Tuple[int, int]]:
    """
    基于检测到的边界将视频帧分组为场景。

    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
        boundaries: 场景开始的帧索引列表

    Returns:
        每个场景的 (start_frame, end_frame) 元组列表。
        结束帧是独占的（标准Python约定）。
    """
    T = video_tensor.shape[0]

    if not boundaries:
        return [(0, T)]

    # 排序边界
    boundaries = sorted(set(boundaries))

    # 确保包含第一帧
    if boundaries[0] != 0:
        boundaries = [0] + boundaries

    # 创建场景间隔
    scenes = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else T
        scenes.append((start, end))

    logger.info(f"分组为 {len(scenes)} 个场景: {scenes}")

    return scenes


def get_scene_statistics(
    video_tensor: torch.Tensor,
    scenes: List[Tuple[int, int]]
) -> dict:
    """
    计算检测到的场景的统计信息。

    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
        scenes: (start_frame, end_frame) 元组列表

    Returns:
        包含场景统计信息的字典
    """
    scene_lengths = [end - start for start, end in scenes]

    stats = {
        'num_scenes': len(scenes),
        'scene_lengths': scene_lengths,
        'avg_scene_length': np.mean(scene_lengths) if scene_lengths else 0,
        'min_scene_length': min(scene_lengths) if scene_lengths else 0,
        'max_scene_length': max(scene_lengths) if scene_lengths else 0,
        'total_frames': video_tensor.shape[0],
    }

    return stats


def map_raw_scenes_to_processed(
    raw_scenes: List[Tuple[int, int]],
    raw_fps: float,
    processed_fps: float,
    total_processed_frames: int
) -> List[Tuple[int, int]]:
    """
    将场景边界从原始视频映射到处理后视频的帧索引。

    处理原始视频和处理后视频由于重新采样而帧数不同的情况。

    Args:
        raw_scenes: 原始视频中 (start_frame, end_frame) 元组列表
        raw_fps: 原始视频帧的FPS
        processed_fps: 处理后视频帧的FPS
        total_processed_frames: 处理后视频的总帧数

    Returns:
        映射到处理后视频索引的 (start_frame, end_frame) 元组列表
    """
    if not raw_scenes or raw_fps == 0:
        return [(0, total_processed_frames)]

    processed_scenes = []

    for raw_start, raw_end in raw_scenes:
        # 将帧索引转换为时间（秒）
        start_time = raw_start / raw_fps
        end_time = raw_end / raw_fps

        # 将时间转换回处理后的帧索引
        # 起始使用floor（包含），结束使用ceil（独占）
        proc_start = max(0, int(np.floor(start_time * processed_fps)))
        proc_end = min(total_processed_frames, int(np.ceil(end_time * processed_fps)))

        # 确保有效的间隔
        if proc_start >= proc_end:
            proc_end = proc_start + 1

        processed_scenes.append((proc_start, proc_end))

    logger.info(f"将 {len(raw_scenes)} 个原始场景映射到 {len(processed_scenes)} 个处理后场景")

    return processed_scenes
