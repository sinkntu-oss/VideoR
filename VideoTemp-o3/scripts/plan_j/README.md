# 方案 J：事件级 caption + 视觉锚点

> 通过离线一次性预处理，为每个事件生成 1 句 caption + 1 张代表帧，
> 嵌入 system prompt 作为「视觉 + 文本」双通道事件目录，让模型决策更精准。

详细动机与设计请见 [`../annotation_analysis.md`](../annotation_analysis.md) 第 8.5 节。

---

## 目录文件

| 文件 | 说明 |
|------|------|
| [`generate_event_captions.py`](generate_event_captions.py) | 离线生成 `event_captions.json`（先白嫖原始数据集 caption，再用 VLM 补齐）|
| [`convert_annotations_j.py`](convert_annotations_j.py) | 转换样本：基于方案 D 流水线复用 + 每事件 1 张关键帧 + system prompt 注入 caption |
| `event_captions.json` | 生成产物（首次运行后才会出现）|

---

## 端到端运行流程

```bash
# 0. 项目根目录
cd VideoR/VideoTemp-o3

# 1. 切场（已有，所有方案共用）
bash scripts/prepare_event_data.sh         # 会产出 scripts/scene_metadata.json

# 2. 抽事件 clip / 关键帧（复用方案 D 的产物）
PROMPT_STYLE=d bash scripts/prepare_event_data.sh   # 会产出 sft/data_events_d/event_clips/...

# 3. ⭐ 生成事件 caption（方案 J 独有）
#    模式 1: 只白嫖现成数据集 caption（ActivityNet/VidChapters，零 GPU）
python scripts/plan_j/generate_event_captions.py \
    --scene_metadata scripts/scene_metadata.json \
    --data_dirs sft/data rl/data \
    --output scripts/plan_j/event_captions.json \
    --mode harvest

#    模式 2: 用 VLM 补齐未覆盖的事件（推荐用 Qwen2.5-VL-3B-Instruct）
python scripts/plan_j/generate_event_captions.py \
    --scene_metadata scripts/scene_metadata.json \
    --clips_dir sft/data_events_d/event_clips \
    --output scripts/plan_j/event_captions.json \
    --mode vlm \
    --captioner_model /mnt/.../Qwen2.5-VL-3B-Instruct \
    --captioner_frames 6 \
    --checkpoint_every 1000

#    模式 3: 自动（先 harvest 再 VLM 补齐，最常用）
python scripts/plan_j/generate_event_captions.py \
    --scene_metadata scripts/scene_metadata.json \
    --data_dirs sft/data rl/data \
    --clips_dir sft/data_events_d/event_clips \
    --output scripts/plan_j/event_captions.json \
    --mode auto \
    --captioner_model /mnt/.../Qwen2.5-VL-3B-Instruct

# 4. 转换样本
python scripts/plan_j/convert_annotations_j.py \
    --metadata scripts/scene_metadata.json \
    --input_dir sft/data --output_dir sft/data_events_j --data_stage sft
python scripts/plan_j/convert_annotations_j.py \
    --metadata scripts/scene_metadata.json \
    --input_dir rl/data --output_dir rl/data_events_j --data_stage rl

# 5. 训练 / 推理（如果已集成 PROMPT_STYLE=j 到 4 个 shell 脚本）
PROMPT_STYLE=j bash sft/sft_events.sh
PROMPT_STYLE=j MODEL=sft/ckpt/test_events_j/checkpoint-xxx bash rl/rollout_events.sh
PROMPT_STYLE=j MODEL=sft/ckpt/test_events_j/checkpoint-xxx bash rl/grpo_events.sh
```

---

## `event_captions.json` 数据格式

```json
{
  "videos/v_xxx.mp4": {
    "0": "A man walks into a wooden room and turns on the light.",
    "1": "He opens a drawer and takes out a notebook.",
    "2": "Close-up of his hands flipping through the pages."
  },
  "videos/v_yyy.mp4": {
    "0": "...",
    ...
  }
}
```

- **key**: 视频相对路径（已规范化，与 `scene_metadata.json` 的 key 对齐）
- **value**: `{event_id: caption}` 字典
- 缺失的 event_id 在转换时自动用 `(no description)` 兜底，不会丢样本

---

## 白嫖来源（harvest 模式自动识别）

`generate_event_captions.py` 的 harvest 阶段会扫描原始数据集 jsonl 中的常见字段：

| 字段名 | 说明 | 典型数据集 |
|--------|------|-----------|
| `sentences` + `timestamps` | 每个时间段对应一句 caption | ActivityNet Captions |
| `captions` + `timestamps` | 同上别名 | 部分数据集 |
| `chapters` 或 `chapter_titles` | 章节标题 | VidChapters-7M |

对齐规则：与事件 `[start, end]` 区间正重叠 > 0.5s 的所有原始 sentence 拼接为该事件的 caption。如果某事件覆盖多个原始 sentence，会保留全部。

---

## VLM Captioner 选型建议

| Captioner | 单 clip ≈ | 40 万事件 (8×A100) | 优点 | 缺点 |
|-----------|-----------|---------------------|------|------|
| **Qwen2.5-VL-3B-Instruct** | ~1-2s | ~80 GPU·h | 与训练模型 7B 同源，语义对齐好 | 占显存 |
| InternVL2-2B | ~0.5-1s | ~40 GPU·h | 最快 | 跨模型语义可能漂移 |
| SmolVLM-2.2B | ~0.5-1s | ~40 GPU·h | 轻量 | 描述细节较弱 |
| Qwen2.5-VL-7B (自洽) | ~3-5s | ~200 GPU·h | 与训练模型完全一致 | 慢 |

**推荐 Qwen2.5-VL-3B**，作为训练模型 7B 的"小弟"，保持语义对齐又不会太慢。

---

## 集成到训练流程

把以下 case 分支追加到 4 个 shell 脚本的 `case "$PROMPT_STYLE"` 中：

```bash
# scripts/prepare_event_data.sh
j) CONVERT_SCRIPT="scripts/plan_j/convert_annotations_j.py"; SUFFIX="_j" ;;

# sft/sft_events.sh、rl/grpo_events.sh
j) DATA_DIR="{sft|rl}/data_events_j" ;;

# rl/rollout_events.sh
j) IMAGE_LIMIT=32 ;;   # 每事件 1 张，N 通常 ≤ 30
```

---

## 已知问题与设计权衡

> 以下是 J 方案在实现过程中识别出的潜在问题；P0/P1 已在代码中修复，P2 作为长期改进项保留。

### 🔴 高危（P0，已修复）

#### P0-1 · caption 中的特殊字符破坏 prompt 结构
**风险**：harvest 来源的 sentences 经常含 `\n`、`"`，会让 system prompt 错行或视觉破损。
**修复**：在 [`build_system_prompt`](convert_annotations_j.py) 与 [`harvest_captions`](generate_event_captions.py) 都做一次清洗（`\n→空格`、`"→'`、`\r→空格`）。

#### P0-2 · `EVENT_CAPTIONS` 环境变量在 shell 中不会传给 python 子进程
**风险**：`EVENT_CAPTIONS=... bash xxx.sh` 形式不会自动导出，python 端拿到默认路径，静默退化为「无 caption」。
**修复**：[`scripts/prepare_event_data.sh`](../prepare_event_data.sh) 调用 python 时显式带 `EVENT_CAPTIONS=... python ...`。

#### P0-3 · 重复 import / reload 时 `_orig_lookup` 自我引用
**风险**：测试中 `importlib.reload(convert_annotations_j)` 会让 `_orig_lookup` 指向已 patch 过的 J 版，调用时无限递归。
**修复**：所有 patch（`lookup_events`、`build_system_prompt`、`N_KEYFRAMES_PER_EVENT`）都加 `_j_patched` 标记，幂等执行。

### 🟠 中危（P1，已修复）

#### P1-4 · `assert` 自检在 `-O` 优化模式下被跳过
**风险**：`python -O` / `PYTHONOPTIMIZE=1` 会移除 assert，patch 链路损坏时不报错。
**修复**：5 条自检改为显式 `if ... raise RuntimeError(...)`。

#### P1-5 · `event_clip_path` 抄了 baseline 公共函数，未来易漂移
**修复**：直接 `from convert_annotations import event_clip_rel_path` 复用。

#### P1-6 · VLM 阶段 `clips_dir` 不存在时静默跳过
**风险**：用户没跑 D 时 VLM 阶段会显示「所有事件均已有 caption」却什么都没干。
**修复**：[`vlm_fill_missing`](generate_event_captions.py) 入口检查目录存在性，缺失时打 ERROR 并附操作建议。

#### P1-7 · scene_metadata 重新切场后 caption 会错位
**风险**：caption 按事件 id 索引，scene_metadata 变化（重新切场）后 event id 含义改变，caption 全部错位且无任何告警。
**修复**：caption 文件写入 `_meta.scene_metadata_sha1`；加载时校验，不一致 → RuntimeError，强制用户重跑 caption。

### 🟡 低危（P2，未修复，作为改进项）

| ID | 问题 | 建议方案 |
|---|---|---|
| P1-8 | `events` 字段带 caption 写入 jsonl（每样本多 ~6KB）| 在 `convert_sft/rl_sample` 序列化前剥离 caption；需多包一层 wrapper |
| P2-9 | `harvest` 多 sentence 拼接无长度上限 | harvest 阶段也截断到 200 字符 |
| P2-10 | `OVERLAP_EPS=0.01s` 太松，邻居 sentence 会污染短事件 caption | 用 `min_overlap = 0.3 * min(event_dur, sentence_dur)` |
| P2-12 | `IMAGE_LIMIT=32` 没覆盖 N>32 的视频 | 跑统计确认 max(N)，或在转换时预过滤 |
| P2-13 | 7B + 30 张 image 显存压力 | 第一次跑 `bs=1 + zero3` 验证 |
| P2-14 | `if not events: return events` 让 `None` 与 `[]` 同等处理 | 改为显式分支 |

详细分析见 [`../annotation_analysis.md`](../annotation_analysis.md) 第 8.5 节末尾。
