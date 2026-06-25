# 场景事件定位改造方案

> 基于 VideoTemp-o3 的改造实验：将"任意时间戳裁剪"替换为"基于场景的事件定位"。

---

## 1. 改造动机

原始 VideoTemp-o3 中，模型通过自由指定 `start_time` / `end_time` 来裁剪视频片段：

```
模型: <tool_call>{"name":"get_video_clip_frame","arguments":[{"start_time":30,"end_time":45}]}</tool_call>
```

**问题**：
- 模型需要猜测精确的时间戳，探索空间大、学习效率低
- 裁剪的片段可能跨越多个语义场景，或只截取到某个场景的一部分
- 时间戳是连续空间，难以进行离散化的精确奖励

**改造思路**：
- 预先将视频按语义边界划分为若干"事件"（Event）
- 模型从事件列表中选择要查看的事件 ID，而非猜测时间戳
- 标注转换为覆盖原有时间范围的最小事件集合

---

## 2. 新增文件一览

**所有改造均为新增文件，不修改任何原有文件。**

```
VideoTemp-o3/
├── scripts/
│   ├── preprocess_scenes.py       # [新增] Adaptive Event Segmentation 场景预处理
│   ├── convert_annotations.py     # [新增] 标注转换：timestamp → event_ids
│   └── prepare_event_data.sh      # [新增] 一键数据准备
│
├── sft/
│   ├── sft_events.sh              # [新增] 事件版 SFT 训练脚本
│   ├── sft.sh                     # [保留] 原始 SFT 脚本
│   └── loss_scale_plugin.py       # [保留] 原始 loss scale 插件（复用）
│
├── rl/
│   ├── video_event_plugin.py      # [新增] 事件定位插件（调度器 + 奖励函数）
│   ├── grpo_events.sh             # [新增] 事件版 GRPO 训练脚本
│   ├── rollout_events.sh          # [新增] 事件版 Rollout 脚本
│   ├── video_crop_plugin.py       # [保留] 原始裁剪插件
│   ├── grpo.sh                    # [保留] 原始 GRPO 脚本
│   └── rollout.sh                 # [保留] 原始 Rollout 脚本
│
└── README_SCENE.md                # [新增] 本文档
```

---

## 3. 原始方案 vs 事件定位方案

| 维度 | 原始方案 | 事件定位方案 |
|------|---------|------------|
| **视频处理** | 按任意时间戳裁剪 | 预先场景分割，模型选择事件 ID |
| **场景检测算法** | — | CLIP [CLS] → TSM → 对角差分卷积核 → 自适应阈值 |
| **工具调用格式** | `get_video_clip_frame(start_time, end_time)` | `locate_events(event_ids=[0, 2, 5])` |
| **系统提示词** | 仅说明裁剪工具用法 | 列出完整事件列表 + 选择工具用法 |
| **模型动作空间** | 连续（任意时间戳） | 离散（事件 ID 组合） |
| **SFT 标注** | `tool_params: [[136, 147]]` | `tool_event_ids: [3, 4, 5]` |
| **RL 标注** | `timestamp: [[8, 25]]` | `covering_event_ids: [1, 2, 3]` |
| **调度器** | `VideoProcessingScheduler` | `EventLocatingScheduler` |
| **定位奖励** | IoU（时间区间交并比） | F1（事件集合匹配） |
| **惩罚机制** | 隐式 `tool_penalty` | 显式 `ToolPenalty`（多次调用 + 多选惩罚） |
| **插件文件** | `video_crop_plugin.py` | `video_event_plugin.py` |
| **数据目录** | `sft/data/`, `rl/data/` | `sft/data_events/`, `rl/data_events/` |
| **输出目录** | `sft/ckpt/test`, `rl/ckpt/test` | `sft/ckpt/test_events`, `rl/ckpt/test_events` |

---

## 4. Adaptive Event Segmentation 算法

### 4.1 算法概述

场景检测采用 **Adaptive Event Segmentation** 算法，基于视觉编码器的 [CLS] token embedding 构建时序相似度矩阵，通过对角差分卷积核自适应检测语义事件边界。

**核心思想**：
视频中的事件边界对应帧级全局视觉语义的突变。利用视觉编码器的 [CLS] token 作为每帧的紧凑语义描述符，在 TSM 中表现为对角线上的块状高相似度结构，块与块之间的相似度骤降即为事件边界。

### 4.2 算法流程

```
原始视频 (T_orig 帧)
    ↓
Step 1: 按 sample_fps 采样帧 → T 帧
    ↓
Step 2: CLIP ViT 前向传播 → [CLS] token embedding {c_t}, c_t ∈ R^D
    ↓
Step 3: L2 归一化 → 单位范数表示 c_t := c_t / ‖c_t‖
    ↓
Step 4: 构建 TSM_{i,j} = c_i · c_j    (余弦相似度矩阵, T×T)
    ↓
Step 5: 构造 5×5 对角差分卷积核 K
    ↓
Step 6: s = diag(Conv2D(pad(TSM), K))    (边界分数, T 维)
    ↓
Step 7: 自适应阈值 τ = mean(s)
    ↓
Step 8: B = { t | s_{t-1} ≤ s_t ≥ s_{t+1}, s_t ≥ τ } ∪ {1, T}
    ↓
Step 9: 将帧序列划分为 N=|B|-1 个变长事件段
    ↓
输出: scene_metadata.json
```

### 4.3 对角差分卷积核

5×5 卷积核 **K** 的结构：

```
+1  +1   0  -1  -1
+1  +1   0  -1  -1
 0   0   0   0   0
-1  -1   0  +1  +1
-1  -1   0  +1  +1
```

**结构解读**（以核中心 (t,t) 在 TSM 对角线上滑动）：

| 核区域 | TSM 覆盖区域 | 含义 | 权重 |
|-------|------------|------|------|
| 左上 2×2 | TSM[t-2:t, t-2:t] | 事件 A 内部自相似度 | **+1** |
| 右下 2×2 | TSM[t+1:t+3, t+1:t+3] | 事件 B 内部自相似度 | **+1** |
| 右上 2×2 | TSM[t-2:t, t+1:t+3] | A→B 跨事件相似度 | **-1** |
| 左下 2×2 | TSM[t+1:t+3, t-2:t] | B→A 跨事件相似度 | **-1** |
| 中心行列 | 过渡区 | 缓冲 | **0** |

**在事件边界处**：
- 对角块区域（同事件）的 TSM 值很高（帧间语义一致） → +1 × 高值 → 大正值
- 反对角块区域（跨事件）的 TSM 值很低（语义不同） → -1 × 低值 → 小负值
- 合计：**显著的正响应** → 标记为边界候选

**在事件内部**：
- 所有区域的 TSM 值相近（都是同一事件的帧） → 正负抵消 → **接近零**

### 4.4 自适应阈值

```
τ = mean(s)
```

自动适应不同视频的时序动态：
- **内容丰富、场景频繁切换的视频**：τ 升高，过滤弱转场
- **时序均匀的视频**：τ 保持较低，捕获细微语义变化

### 4.5 边界检测规则

同时满足以下条件的帧 t 被识别为事件边界：
1. **局部极大值**：`s_{t-1} ≤ s_t ≥ s_{t+1}`
2. **超过阈值**：`s_t ≥ τ`
3. **始终包含首帧和末帧**，确保完整覆盖

### 4.6 实现优化

计算 `s = diag(Conv2D(TSM, K))` 时，**只需对角线元素**，无需完整 2D 卷积：

```python
# 复杂度: O(T × K²) 而非 O(T² × K²)
for t in range(T):
    patch = TSM_padded[t:t+K, t:t+K]
    scores[t] = sum(patch * kernel)
```

### 4.7 元数据格式

```json
{
  "qa/video_5140.mp4": {
    "orig_fps": 30.0,
    "total_frames": 338,
    "duration": 11.26,
    "sample_fps": 2.0,
    "num_events": 4,
    "num_sampled_frames": 23,
    "boundary_scores_stats": {
      "mean": 0.012345,
      "std": 0.008765,
      "max": 0.045678,
      "min": -0.002345,
      "threshold": 0.012345
    },
    "events": [
      {"event_id": 0, "start_time": 0.0,  "end_time": 3.2,  "num_frames": 96},
      {"event_id": 1, "start_time": 3.2,  "end_time": 6.5,  "num_frames": 99},
      {"event_id": 2, "start_time": 6.5,  "end_time": 9.1,  "num_frames": 78},
      {"event_id": 3, "start_time": 9.1,  "end_time": 11.26, "num_frames": 65}
    ]
  }
}
```

---

## 5. 数据标注转换

### 5.1 最小覆盖集计算

`scripts/convert_annotations.py` 将原有时间戳标注转换为事件定位标注：

```
原始标注: timestamp = [[8, 25]]
事件列表: Event 0: 0-5s, Event 1: 5-12s, Event 2: 12-20s, Event 3: 20-30s

覆盖计算: Event 1 与 [8,25] 重叠 ✓
          Event 2 与 [8,25] 重叠 ✓
          Event 3 与 [8,25] 重叠 ✓

结果: covering_event_ids = [1, 2, 3]
```

### 5.2 转换内容

| 转换项 | 原始 | 转换后 |
|-------|------|--------|
| 系统提示词 | 固定模板 | 包含事件列表的动态模板 |
| assistant 的 tool_call | `get_video_clip_frame` | `locate_events` |
| videos 数组 | 含 cropped_video 路径 | 移除 cropped_video |
| 新增字段 | 无 | `events`, `covering_event_ids`, `tool_event_ids` |

### 5.3 转换后的系统提示词示例

```
You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags.
Enclose your final answer within <answer> and </answer> tags.

The video has been segmented into the following events:
  Event 0: 0.0s - 3.2s
  Event 1: 3.2s - 6.5s
  Event 2: 6.5s - 9.1s
  Event 3: 9.1s - 11.3s

If you need to examine specific events more closely to answer the question,
you may use the following tool to retrieve the video clips for the selected events:

<tool_call>{"name":"locate_events","arguments":{"event_ids":[event_id_1, event_id_2, ...]}}</tool_call>

Use the insights from the selected event clips to inform your reasoning
and construct the final answer.
```

### 5.4 转换后的工具调用示例

```
# 模型输出
<think>根据事件列表，Event 1 和 Event 2 覆盖了提问涉及的时间段...</think>
<tool_call>{"name":"locate_events","arguments":{"event_ids":[1, 2]}}</tool_call>
```

---

## 6. 训练流程

### 6.1 完整多轮对话流程

```
第 1 轮:
  [模型看完整视频 + 事件列表]
    → <think>分析后决定查看 Event 2 和 Event 3</think>
       <tool_call>{"name":"locate_events","arguments":{"event_ids":[2,3]}}</tool_call>

       ↓ 系统裁剪 Event 2 和 Event 3 的视频片段

第 2 轮:
  [模型看事件片段]
    → <think>基于事件片段的内容得出结论</think>
       <answer>C</answer>

  或继续选择更多事件 →

第 3 轮 (最多):
    → <answer>最终答案</answer>
```

### 6.2 奖励函数

`rl/video_event_plugin.py` 注册了 4 个奖励函数：

| 奖励函数 | 名称 | 说明 |
|---------|------|------|
| `Accuracy_Reward` | `acc_reward` | QA: 首字母匹配；Grounding: 预测时间→事件→与 GT 事件集的 F1 |
| `Event_Reward` | `event_reward` | 仅答案正确时：模型选择的事件与目标事件集的 F1 分数 |
| `FormatReward` | `format_reward` | 检查每轮输出是否为合法的 `<think>+<tool_call>` 或 `<think>+<answer>` 格式 |
| `ToolPenalty` | `tool_penalty` | 惩罚多次工具调用(-0.1/次)和过多事件选择(-0.05/个) |

**事件选择 F1 分数计算**：

```
Precision = |选对的事件| / |模型选的事件|   → 惩罚"多选"
Recall    = |选对的事件| / |目标事件|       → 惩罚"漏选"
F1        = 2 × P × R / (P + R)

示例:
  目标事件:   {1, 2, 3}
  模型选择:   {2, 3, 5}
  交集:       {2, 3}
  Precision = 2/3, Recall = 2/3, F1 = 0.667
```

---

## 7. 快速开始

```bash
cd VideoR/VideoTemp-o3

# 前提：已执行 setup_env.sh 和 setup_data.sh

# ===== Step 1: 数据准备 =====
bash scripts/prepare_event_data.sh
# 生成:
#   scripts/scene_metadata.json
#   sft/data_events/
#   rl/data_events/

# ===== Step 2: SFT 训练 =====
bash sft/sft_events.sh
# 输出: sft/ckpt/test_events/

# ===== Step 3: RL 训练 =====
# 终端 1: 启动 Rollout 引擎
bash rl/rollout_events.sh

# 终端 2: 启动 GRPO 训练
bash rl/grpo_events.sh
# 输出: rl/ckpt/test_events/
```

---

## 8. 各脚本参数说明

### 8.1 场景预处理参数

```bash
python scripts/preprocess_scenes.py \
    --data_dirs sft/data rl/data           # JSONL 数据目录
    --output scripts/scene_metadata.json   # 输出文件
    --clip_model openai/clip-vit-base-patch16  # CLIP 模型 (名称或本地路径)
    --sample_fps 2.0                       # 帧采样率 Hz
    --kernel_size 5                        # 对角差分卷积核大小 (奇数)
    --batch_size 64                        # CLIP 推理批大小
    --device cuda                          # 计算设备 (cuda/cpu)
```

**也可通过环境变量配置** (`prepare_event_data.sh`)：
```bash
CLIP_MODEL=/path/to/local/clip SAMPLE_FPS=3.0 KERNEL_SIZE=7 bash scripts/prepare_event_data.sh
```

**调参建议**：
- `sample_fps`：越高检测越精细但越慢。2.0 fps 适合大多数场景。
- `kernel_size`：越大对边界检测越鲁棒但可能遗漏短事件。5 是平衡点。
- `clip_model`：集群环境建议使用本地路径避免下载。`ViT-B/16` 已足够。
- `batch_size`：根据 GPU 显存调整。ViT-B/16 在 A100 上 128 无压力。

### 8.2 标注转换参数

```bash
python scripts/convert_annotations.py \
    --metadata scene_metadata.json   # 场景元数据
    --input_dir sft/data             # 原始数据目录
    --output_dir sft/data_events     # 输出目录
    --data_stage sft                 # sft 或 rl
```

### 8.3 训练参数差异（相对原始方案）

| 参数 | 原始 | 事件版 | 说明 |
|------|------|--------|------|
| `--external_plugins` | `video_crop_plugin.py` | `video_event_plugin.py` | 新插件 |
| `--multi_turn_scheduler` | `video_processing_scheduler` | `event_locating_scheduler` | 新调度器 |
| `--reward_funcs` | `acc iou tool format` | `acc event tool format` | event 替代 iou |
| `--dataset` | `rl/data/*.jsonl` | `rl/data_events/*.jsonl` | 新数据目录 |
| `--output_dir` | `*/ckpt/test` | `*/ckpt/test_events` | 独立输出 |

---

## 9. 技术细节

### 9.1 为什么使用 CLIP [CLS] 而非像素差分

| 维度 | 像素差分 (旧) | CLIP CLS (新) |
|------|-------------|---------------|
| **语义理解** | 无 — 只看像素变化 | 有 — 视觉语义特征 |
| **鲁棒性** | 受光照、摄像机运动影响大 | 对低层变化鲁棒 |
| **阈值设定** | 百分位，需人工调参 | 自适应均值，无需调参 |
| **检测粒度** | 视觉场景切换 | 语义事件转变 |
| **计算成本** | CPU 即可，极快 | 需 GPU + CLIP，中等 |
| **依赖** | numpy only | transformers + torch |

CLIP [CLS] token 捕获的是帧级全局语义，使得检测到的事件边界与视频的**语义节奏**对齐，而非仅仅是视觉外观的变化。

### 9.2 EventLocatingScheduler 工作流

```python
class EventLocatingScheduler(MultiTurnScheduler):
    """
    step() 方法在每轮对话中执行:
    1. 从模型输出解析 event_ids
    2. 从系统提示词中获取事件时间范围
    3. 对每个选中事件，从原始视频裁剪对应片段
    4. 构造包含 <video> 标签的下一轮 user 消息
    5. 将裁剪视频路径追加到 infer_request.videos
    """
```

### 9.3 与原始方案的兼容性

- 所有新文件与原始文件独立，可通过不同的 shell 脚本切换
- `loss_scale_plugin.py` 在两个方案中复用（逻辑相同）
- SFT 数据格式保持 messages 结构一致，仅内容替换
- RL 数据新增 `events` / `covering_event_ids` 字段，不影响原有字段
