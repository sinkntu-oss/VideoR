#!/usr/bin/env python3
"""
完整视频 TSM 热度图分析
用 1fps 抽帧，计算整个视频的 TSM，并绘制热度图
"""

import os
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '16'

from decord import VideoReader, cpu
import numpy as np
from transformers import SiglipVisionModel, AutoImageProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

print('='*70)
print('  完整视频 TSM 热度图分析 (1fps 采样)')
print('='*70)

# 视频信息
video_path = 'ActivityNet/videos/v_c0Hix_5Vm8I.mp4'
print(f'\n视频: {video_path}')
print(f'总帧数: 6526')
print(f'时长: 217.75 秒')

# 读取 1fps 采样的所有帧
vr = VideoReader(video_path, ctx=cpu(0))
frame_indices = list(range(0, 6526, 30))  # 1fps 采样
print(f'\n采样方式: 1fps (每 30 帧取一帧)')
print(f'采样帧数: {len(frame_indices)}')
print(f'对应时长: {len(frame_indices)/1:.1f} 秒')

# 分批读取帧（避免内存溢出）
batch_size = 100
all_frames = []
for i in range(0, len(frame_indices), batch_size):
    batch_indices = frame_indices[i:i+batch_size]
    batch_frames = vr.get_batch(batch_indices).asnumpy()
    all_frames.extend([batch_frames[j] for j in range(len(batch_frames))])
    print(f'  已读取 {min(i+batch_size, len(frame_indices))}/{len(frame_indices)} 帧')

frames = np.array(all_frames)
print(f'\n帧数据形状: {frames.shape}')

# 加载 SigLIP 模型
print('\n加载 SigLIP 模型...')
model_path = '/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/siglip2-so400m-patch16-512'
model = SiglipVisionModel.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)
model = model.to('cuda').eval()

# 分批编码帧
print('编码帧特征...')
all_feats = []
for i in range(0, len(frames), batch_size):
    batch_frames = frames[i:i+batch_size]
    pil_frames = [Image.fromarray(f) for f in batch_frames]
    inputs = processor(images=pil_frames, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.pooler_output
    all_feats.append(feats.cpu().numpy())
    print(f'  已编码 {min(i+batch_size, len(frames))}/{len(frames)} 帧')

feats = np.concatenate(all_feats, axis=0)
print(f'特征形状: {feats.shape}')

# L2 归一化
print('计算 TSM...')
feats_norm = feats / np.linalg.norm(feats, axis=1, keepdims=True)

# 计算相似度矩阵
tsm = feats_norm @ feats_norm.T
print(f'TSM 形状: {tsm.shape}')

# 统计信息
print(f'\nTSM 统计:')
print(f'  最大值: {tsm.max():.6f}')
print(f'  最小值: {tsm.min():.6f}')
print(f'  平均值: {tsm.mean():.6f}')
print(f'  对角线平均: {np.diag(tsm).mean():.6f}')

# 计算边界分数（使用对角差分卷积核）
print(f'\n计算边界分数...')
kernel_size = 5
kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
half = kernel_size // 2

for i in range(kernel_size):
    for j in range(kernel_size):
        if i == half or j == half:
            continue
        if (i < half) == (j < half):
            kernel[i, j] = 1.0
        else:
            kernel[i, j] = -1.0

num_positive = int(np.sum(kernel > 0))
if num_positive > 0:
    kernel /= num_positive

# 计算对角线上的卷积响应（边界分数）
T = len(tsm)
K = kernel.shape[0]
pad = K // 2

tsm_padded = np.pad(tsm, pad, mode='constant', constant_values=0)

scores = np.zeros(T, dtype=np.float32)
for t in range(T):
    patch = tsm_padded[t:t + K, t:t + K]
    scores[t] = np.sum(patch * kernel)

# 使用分数均值作为阈值（与 preprocess_scenes.py 一致）
threshold = np.mean(scores)
print(f'  边界分数统计:')
print(f'    最大值: {scores.max():.6f}')
print(f'    最小值: {scores.min():.6f}')
print(f'    平均值 (阈值): {threshold:.6f}')
print(f'    标准差: {scores.std():.6f}')

# 检测边界（局部极大值 + 超过阈值）
boundaries = set()
boundaries.add(0)
boundaries.add(T - 1)

for t in range(1, T - 1):
    if scores[t] >= threshold and scores[t - 1] <= scores[t] and scores[t] >= scores[t + 1]:
        boundaries.add(t)

boundaries = sorted(boundaries)
print(f'\n边界检测结果:')
print(f'  检测到 {len(boundaries)} 个边界')
print(f'  边界位置: {boundaries}')
times = [f'{i:.1f}s' for i in boundaries]
print(f'  对应时间: {times}')

# 计算相邻帧的相似度（用于对比）
diag_sim = np.array([tsm[i, i+1] if i < len(tsm)-1 else 0 for i in range(len(tsm))])
print(f'\n相邻帧相似度统计:')
print(f'  最大: {diag_sim.max():.4f}, 最小: {diag_sim.min():.4f}, 平均: {diag_sim.mean():.4f}')

# 绘制热度图
print('\n绘制热度图...')
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# 1. 完整 TSM 热度图
ax = axes[0, 0]
im = ax.imshow(tsm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
ax.set_title('Complete TSM Heatmap (1fps sampling)', fontsize=12, fontweight='bold')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Frame Index')
plt.colorbar(im, ax=ax, label='Similarity')

# 2. 对角线附近的放大图（±50 帧）
ax = axes[0, 1]
window = 100
tsm_window = tsm[:window, :window]
im = ax.imshow(tsm_window, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
ax.set_title(f'Diagonal Zoom (First {window} frames)', fontsize=12, fontweight='bold')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Frame Index')
plt.colorbar(im, ax=ax, label='Similarity')

# 3. 边界分数曲线（使用对角差分卷积核）
ax = axes[1, 0]
ax.plot(scores, linewidth=1.5, color='darkgreen')
ax.fill_between(range(len(scores)), scores, alpha=0.3, color='green')
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (mean={threshold:.4f})')
# 标记检测到的边界
for b in boundaries:
    ax.axvline(x=b, color='orange', linestyle=':', alpha=0.7, linewidth=1)
ax.set_title('Boundary Scores (Diagonal Difference Convolution)', fontsize=12, fontweight='bold')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Score')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 相邻帧相似度曲线
ax = axes[1, 1]
ax.plot(diag_sim, linewidth=1.5, color='darkblue')
ax.fill_between(range(len(diag_sim)), diag_sim, alpha=0.3, color='blue')
ax.set_title('Adjacent Frame Similarity Curve', fontsize=12, fontweight='bold')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Similarity')
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

# 5. 相似度分布直方图
ax = axes[2, 0]
off_diag_indices = np.triu_indices_from(tsm, k=1)
off_diag = tsm[off_diag_indices]
ax.hist(off_diag, bins=50, color='coral', edgecolor='darkred', alpha=0.7)
ax.set_title('Off-diagonal Similarity Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Similarity')
ax.set_ylabel('Frequency')
ax.grid(True, alpha=0.3, axis='y')

# 6. 边界分数分布直方图
ax = axes[2, 1]
ax.hist(scores, bins=50, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (mean={threshold:.4f})')
ax.set_title('Boundary Score Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Score')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('tsm_heatmap.png', dpi=150, bbox_inches='tight')
print('✓ 热度图已保存: tsm_heatmap.png')

# 分析对角线块的大小
print(f'\n对角线块大小分析:')
block_sizes = []
in_block = False
block_start = 0
for i, sim in enumerate(diag_sim):
    if sim >= threshold:
        if not in_block:
            block_start = i
            in_block = True
    else:
        if in_block:
            block_sizes.append(i - block_start)
            in_block = False

if in_block:
    block_sizes.append(len(diag_sim) - block_start)

if block_sizes:
    print(f'  检测到 {len(block_sizes)} 个对角线块')
    print(f'  块大小 - 最大: {max(block_sizes)}, 最小: {min(block_sizes)}, 平均: {np.mean(block_sizes):.1f}')
    print(f'  前 10 个块的大小: {block_sizes[:10]}')
    times = [f'{s:.1f}s' for s in np.array(block_sizes[:10])/1]
    print(f'  对应时长: {times}')

print(f'\n✓ 分析完成！')
