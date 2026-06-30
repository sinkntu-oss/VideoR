# 事件定位 Prompt 4 个改进方案 (B / C / D / E)

> 关于「事件保存方式与 prompt 拼接」过于冗余/不自然问题的 4 种独立改造版本。
> 每个版本相对 baseline ([`convert_annotations.py`](convert_annotations.py)) 只改动一个语义点，
> 通过 `import + monkey-patch` 实现物理上最小代价的隔离，便于 A/B 实验。

---

## 0. Baseline 的问题（动机）

[`convert_annotations.py`](convert_annotations.py) 当前生成的 system prompt 形态：

```
[system]
You are a helpful assistant.
Think step-by-step ...
Enclose your entire reasoning process within <think> and </think> tags. ...

The video has been segmented into the following events:
  Event 0: 0.0s - 3.2s
  Event 1: 3.2s - 6.5s
  Event 2: 6.5s - 9.1s
  Event 3: 9.1s - 11.3s
  ...                  ← 长视频可能 30+ 行

If you need to examine specific events ...
<tool_call>{"name":"locate_events","arguments":{"event_ids":[event_id_1, ...]}}</tool_call>
...
```

5 个具体痛点：

| # | 问题 | 影响 |
|---|------|------|
| 1 | **事件列表硬塞 system** | 破坏 system 通用性；每条样本不同 → KV cache 失效 |
| 2 | **每事件独占一行 + 完整时间戳** | 长视频 token 浪费严重（30 事件 ≈ 300+ tokens） |
| 3 | **同时给 ID 与时间戳** | tool 只吃 `event_ids`，时间戳冗余 |
| 4 | **think 中时间戳改写为 `Events 1,2,3`** (见 [`rewrite_think()`](convert_annotations.py:181-192)) | 模型反推回时间戳还得查表，认知链路绕 |
| 5 | **tool 说明写在自然语言里** | 与 Qwen2.5-VL 原生 function calling 习惯不一致 |

---

## 1. 总览

| 文件 | 方案 | 改动点（语义） | 是否动 video | 是否动 tool schema |
|------|------|-------------|-------------|------------------|
| [`convert_annotations_b.py`](convert_annotations_b.py) | **B** 信息搬家 | 事件列表从 system 挪到 user 第一轮（紧凑单行） | 否 | 否 |
| [`convert_annotations_c.py`](convert_annotations_c.py) | **C** 剥离时间戳 | system 只暴露事件数量 N；think 中时间戳抹除 | 否 | 否 |
| [`convert_annotations_d.py`](convert_annotations_d.py) | **D** 视觉锚点 | 取消主视频，每个事件用 **2 张关键帧**（image）作为视觉概览 | **是**（主视频 → 2N 张图） | 否 |
| [`convert_annotations_e.py`](convert_annotations_e.py) | **E** 原生 tools schema | system 去掉工具说明；样本顶层新增 `tools` 字段 | 否 | **是** |

四份代码：
- **共享 baseline 所有工具函数**（场景查找、片段裁剪、对齐校验、`main()` 等）
- **通过 monkey-patch `_ca.xxx` 模块级名字**实现替换，原版任意时刻可回退
- **命令行参数与原版 100% 一致**，仅需替换文件名 + 输出目录

---

## 2. 各方案详解

### 2.1 方案 B — 事件列表挪到 user 第一轮

**文件**：[`convert_annotations_b.py`](convert_annotations_b.py)

**改动点（唯一）**：通过 monkey-patch 在两个 convert 函数末尾追加 [`_relocate_event_info()`](convert_annotations_b.py:42-56)：
1. system prompt 替换为 [`SYSTEM_PROMPT_GENERIC`](convert_annotations_b.py:27-37)（不含事件列表 → 完全通用，KV cache 友好）
2. 在第一个含 `<video>` 的 user 消息前 prepend 紧凑单行：
   ```
   Segments (4): [0]0.0-3.2s [1]3.2-6.5s [2]6.5-9.1s [3]9.1-11.3s
   ```

**Prompt 形态示例**：

```text
[system]                       # 通用、固定，可被 KV cache 复用
You are a helpful assistant.
...
If you need to examine specific pre-segmented events of the video more closely ...
<tool_call>{"name":"locate_events","arguments":{"event_ids":[...]}}</tool_call>

[user]
Segments (4): [0]0.0-3.2s [1]3.2-6.5s [2]6.5-9.1s [3]9.1-11.3s
<video>
Question: ...
```

**特点对比**：

| 维度 | Baseline | 方案 B |
|------|---------|--------|
| system 长度（4 事件） | ~150 tokens | ~80 tokens |
| system 唯一性 | 每样本不同 | 全样本相同 |
| 事件列表占用 | 4 行（system） | 1 行（user） |
| 长视频（30 事件）改善 | — | system 长度 ↓ 80% |

---

### 2.2 方案 C — 剥离时间戳，system 只说 N

**文件**：[`convert_annotations_c.py`](convert_annotations_c.py)

**改动点（唯一）**：替换两个模块级函数：
1. [`build_system_prompt()`](convert_annotations_c.py:43-45)：返回 [`SYSTEM_PROMPT_TEMPLATE_C`](convert_annotations_c.py:30-40)，仅插入 `N` 和 `N-1`
2. [`rewrite_think()`](convert_annotations_c.py:52-61)：将 `<think>` 中匹配 [`_RANGE_PAT`](convert_annotations.py:58-60) 的时间区间（如 `"8s-25s"`）一律抹为 `"the relevant segment"`

**Prompt 形态示例**：

```text
[system]
You are a helpful assistant.
...
The video has been pre-segmented into 4 temporally ordered events, indexed 0 to 3.
Identify each event from the visual content itself.
If you need to examine specific events ...
<tool_call>{"name":"locate_events","arguments":{"event_ids":[...]}}</tool_call>

[user] <video>\nQuestion: ...

# Assistant think 改写示例
- 原: <think>Looking at 8s-25s, the man is opening a box.</think>
- 新: <think>Looking at the relevant segment, the man is opening a box.</think>
```

**事件元数据保留位置**：样本顶层 `events` 字段（baseline 已注入）。`rl/video_event_plugin.py` 在 RL 阶段从该字段读取即可。

**特点对比**：

| 维度 | Baseline | 方案 C |
|------|---------|--------|
| system 长度（4 事件） | ~150 tokens | ~75 tokens |
| system 长度（30 事件） | ~600 tokens | **常量** ~75 tokens |
| think 中时间戳泄漏 | 改写为 Events X,Y | 完全无 |
| 模型需视觉理解事件内容 | 否 | **是** |

---

### 2.3 方案 D — 视觉锚点（取消主视频，每事件 2 张关键帧）

**文件**：[`convert_annotations_d.py`](convert_annotations_d.py)

**核心思想**：完整主视频对 prompt 是冗余的 —— 与其让模型在原视频里"找"事件，不如直接给它每个事件的 **2 张代表帧**作为视觉概览。
模型靠这 `2N` 张关键帧就能感知整段视频的内容结构；如需高清细看某个事件，再走 `locate_events` 工具调用拿对应 video clip。

**改动点（唯一）**：在 [`convert_sft_sample()`](convert_annotations_d.py:106-167) 中先调原版逻辑，再在末尾做 5 件事：
1. 用 [`_split_videos()`](convert_annotations_d.py:46-54) 区分主视频与 tool 片段
2. 对每个事件调 [`extract_event_keyframes()`](convert_annotations_d.py:62-101) 抽 `N_KEYFRAMES_PER_EVENT=2` 张关键帧（**等距取 1/3、2/3 位置**，避开转场边界）
3. 第一轮 user 中**首个 `<video>` 直接替换为 `2N` 个 `<image>`**
4. `videos` 字段去掉主视频，**仅保留 tool 调用产生的高清片段**；新增 `images` 字段
5. 重新做 `<image>`/`<video>` 计数 vs `images`/`videos` 长度的双重对齐校验

system prompt 同步改为 [`SYSTEM_PROMPT_TEMPLATE_D`](convert_annotations_d.py:24-37)：

```
You will see {2N} keyframes ({K} per event) sampled from {N} temporally ordered events
(indexed 0 to {N-1}). The keyframes are listed in event order: keyframes [0,1] belong
to event 0, [2,3] to event 1, and so on.
```

**Prompt 形态示例**（4 个事件 × 2 帧 = 8 张图）：

```text
[system]
...
You will see 8 keyframes (2 per event) sampled from 4 temporally ordered events ...
<tool_call>{"name":"locate_events","arguments":{"event_ids":[...]}}</tool_call>

[user]
<image>   # event 0, kf 0 (位于事件 1/3 处)
<image>   # event 0, kf 1 (位于事件 2/3 处)
<image>   # event 1, kf 0
<image>   # event 1, kf 1
<image>   # event 2, kf 0
<image>   # event 2, kf 1
<image>   # event 3, kf 0
<image>   # event 3, kf 1
Question: ...

# 如果模型决定细看 event 1, 2：
[assistant] <think>...</think>
<tool_call>{"name":"locate_events","arguments":{"event_ids":[1, 2]}}</tool_call>
[user] <video><video>Tool execution successful. ...
[assistant] <think>...</think><answer>C</answer>
```

**样本顶层新增 `images` 字段**：

```json
{
  "messages": [...],
  "videos": [
    "sft/data_events_d/event_clips/<safe_name>/event_1.mp4",
    "sft/data_events_d/event_clips/<safe_name>/event_2.mp4"
  ],
  "images": [
    "sft/data_events_d/event_keyframes/<safe_name>/event_0_kf_0.jpg",
    "sft/data_events_d/event_keyframes/<safe_name>/event_0_kf_1.jpg",
    "sft/data_events_d/event_keyframes/<safe_name>/event_1_kf_0.jpg",
    "sft/data_events_d/event_keyframes/<safe_name>/event_1_kf_1.jpg",
    "sft/data_events_d/event_keyframes/<safe_name>/event_2_kf_0.jpg",
    "sft/data_events_d/event_keyframes/<safe_name>/event_2_kf_1.jpg",
    "sft/data_events_d/event_keyframes/<safe_name>/event_3_kf_0.jpg",
    "sft/data_events_d/event_keyframes/<safe_name>/event_3_kf_1.jpg"
  ],
  "events": [...]
}
```

**关键参数**：

- [`N_KEYFRAMES_PER_EVENT`](convert_annotations_d.py:19) = 2（每事件关键帧数，可调）
- 抽帧位置：第 `i` 帧位于事件的 `(i+1)/(K+1)` 处，K=2 时即 1/3 与 2/3，**避开 0% / 100% 的边界帧**（边界往往是转场/模糊）

**注意事项**：

- **token 代价大幅下降**：原版主视频在 Qwen2.5-VL 下通常采样 32-64 帧，方案 D 是固定的 `2N` 张图。中短视频（N≤16）远小于主视频开销
- **抽帧失败处理**：任何一个事件抽帧失败 → 整个样本 `return None` 丢弃（防止训练时遇到空文件）；统计字段 `stats["keyframe_fail"]`
- **跳过条件**：当 base_videos 数量 ≠ 1（多主视频/无主视频）时，原样返回（不做关键帧改造）
- **物理文件位置**：关键帧默认存放在 `<output_dir>/event_keyframes/<safe_video_name>/event_{eid}_kf_{i}.jpg`，与事件视频片段 `<output_dir>/event_clips/` 同级
- **多轮 tool 调用仍走 video**：第一轮看 keyframe，后续工具调用回来仍是高清 `<video>` 片段，混合 image+video 输入，靠 Qwen2.5-VL 原生多模态能力处理

---

### 2.4 方案 E — 原生 tools schema

**文件**：[`convert_annotations_e.py`](convert_annotations_e.py)

**改动点（唯一）**：
1. [`build_system_prompt()`](convert_annotations_e.py:35-39) 返回 [`SYSTEM_PROMPT_TEMPLATE_E`](convert_annotations_e.py:27-32)，移除「工具调用 + 示例」部分，仅保留 think/answer 约束 + 事件列表
2. 通过 monkey-patch 在两个 convert 函数末尾追加 [`_attach_tools()`](convert_annotations_e.py:69-74)，在样本顶层注入 OpenAI / Qwen 兼容的 [`_build_tools_schema()`](convert_annotations_e.py:42-66)

**Prompt + 样本形态示例**：

```text
[system]                    # 仅保留 think/answer + 事件列表，无工具说明
You are a helpful assistant.
...
The video has been segmented into the following events:
  Event 0: 0.0s - 3.2s
  ...

[user] <video>\nQuestion: ...
```

样本顶层新增字段（与 Qwen2.5-VL chat template 兼容）：

```json
{
  "messages": [...],
  "videos": [...],
  "events": [...],
  "covering_event_ids": [...],
  "tools": [{
    "type": "function",
    "function": {
      "name": "locate_events",
      "description": "Retrieve close-up video clips for the specified pre-segmented events. ...",
      "parameters": {
        "type": "object",
        "properties": {
          "event_ids": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 3},
            "minItems": 1,
            "description": "Indices of events to retrieve clips for (0-based)."
          }
        },
        "required": ["event_ids"]
      }
    }
  }]
}
```

`maximum` 字段会**按当前样本的事件数自动推导**，给模型一个强约束。

**注意事项**：

- 需先用一条样本跑通 `swift sft` 的 `tools` 字段支持。ms-swift 在较新版本中已支持 OpenAI 风格 function calling，但具体到多模态多轮场景需 spot check。
- 推理时 [`rl/video_event_plugin.py`](../rl/video_event_plugin.py) 的事件解析逻辑要改为：从 tool_call 的标准 JSON 中读取 `event_ids`（baseline 是字符串 regex，本方案数据下完全兼容）。

---

## 3. 使用方法

每个脚本与 [`prepare_event_data.sh`](prepare_event_data.sh) 中 baseline 命令的参数完全相同，**只需改文件名 + 输出目录**：

```bash
# === 方案 B ===
python scripts/convert_annotations_b.py \
    --metadata scripts/scene_metadata.json \
    --input_dir sft/data --output_dir sft/data_events_b --data_stage sft

# === 方案 C ===
python scripts/convert_annotations_c.py \
    --metadata scripts/scene_metadata.json \
    --input_dir sft/data --output_dir sft/data_events_c --data_stage sft

# === 方案 D ===  (建议先用小数据跑通；会真实裁剪 N 段事件预览)
python scripts/convert_annotations_d.py \
    --metadata scripts/scene_metadata.json \
    --input_dir sft/data --output_dir sft/data_events_d --data_stage sft

# === 方案 E ===
python scripts/convert_annotations_e.py \
    --metadata scripts/scene_metadata.json \
    --input_dir sft/data --output_dir sft/data_events_e --data_stage sft
```

RL 数据生成：把 `--data_stage sft` 改为 `--data_stage rl`，输入输出目录改为 `rl/data` → `rl/data_events_{b,c,d,e}`。

切换到 4 个数据集做 SFT 训练时，只需修改 [`sft/sft_events.sh`](../sft/sft_events.sh:1-45) 中 `--dataset` 路径与 `--output_dir`，其它训练参数完全复用。

---

## 4. 多维度对比

| 维度 | Baseline | B | C | D | E |
|------|---------|---|---|---|---|
| **system 是否通用** | ❌ | ✅ 完全通用 | ❌ 含 N | ❌ 含 N | ❌ 含完整事件列表 |
| **prompt 中是否含时间戳** | ✅ 多行 | ✅ 单行紧凑 | ❌ 完全无 | ❌ 完全无 | ✅ 多行原样 |
| **think 是否含时间戳** | ✅（Events X,Y） | ✅（Events X,Y） | ❌（被抹除） | ✅（Events X,Y） | ✅（Events X,Y） |
| **第一轮视觉输入** | 主视频(`<video>`) | 主视频 | 主视频 | **2N 张关键帧(`<image>`)，主视频被替换** | 主视频 |
| **样本顶层新字段** | events / covering_event_ids | 同左 | 同左 | + **`images`** | + **`tools`** |
| **token 代价（4 事件）** | 主视频 32-64 帧 | 同左 | 同左 | **8 张图（远低于主视频）** | 同左 |
| **token 代价（30 事件）** | 主视频 32-64 帧 | 同左 | 同左 | 60 张图（接近主视频上限） | 同左 |
| **改动复杂度** | — | 低 | 低 | 中 | 低 |
| **训练侧脚本是否需改** | — | 仅 dataset 路径 | 仅 dataset 路径 | 仅 dataset 路径 | dataset 路径（+ 确认 swift 支持 tools） |
| **RL [`video_event_plugin.py`](../rl/video_event_plugin.py) 是否需改** | — | ✅ 解析事件需改 | ✅ 同左 | ✅ 同左 | ✅ tool_call 解析需改 |

---

## 5. 实现细节与代码正确性说明

### 5.1 monkey-patch 链路

所有方案通过 `_ca.xxx = ...` 替换 [`convert_annotations`](convert_annotations.py) 的模块级名字：

```python
import convert_annotations as _ca
_ca.build_system_prompt = ...    # 模块级名字替换
_ca.convert_sft_sample  = ...
```

Python 函数解析全局名字时通过 `__globals__` 查找。`_ca.process_jsonl_file` 内部的 `convert_sft_sample if data_stage == "sft" else convert_rl_sample` 是**调用时**求值的（不是 import 时绑定），因此 monkey-patch 后链路完全连通。`main()` 函数同理。

### 5.2 对齐校验链

baseline 在 [`convert_sft_sample()`](convert_annotations.py:270-276) 中做了：
```python
tag_count = sum(m.content.count("<video>") for m in messages)
assert tag_count == len(s["videos"])
```

- 方案 B：在 user 中插入文本（无 `<video>`），不影响对齐 ✅
- 方案 C：完全不动 messages / videos 结构 ✅
- 方案 D：[`convert_annotations_d.py`](convert_annotations_d.py:153-167) 显式做 `<image>` ↔ `images` 与 `<video>` ↔ `videos` 双重对齐校验；通过 [`_split_videos()`](convert_annotations_d.py:46-54) 防御多主视频边角 case；抽帧失败时整样本丢弃，不会产生坏文件引用 ✅
- 方案 E：仅注入顶层 `tools` 字段，不动 messages / videos ✅

### 5.3 与 RL 阶段的衔接

切到任一新方案后，[`rl/video_event_plugin.py`](../rl/video_event_plugin.py) 的事件解析逻辑需同步调整：

**Baseline 当前实现**：从 system prompt 文本中 regex 解析 `Event N: a.bs - c.ds`。

**推荐改造**：所有 4 个方案的样本顶层都有 `events` 字段。RL scheduler 应直接读取该字段，与 prompt 形态彻底解耦，更鲁棒。

具体来说，[`EventLocatingScheduler._parse_events_from_system()`](../rl/video_event_plugin.py) 改为优先从 `infer_request.metadata.get("events")`（或类似挂载点）读取。这是切换 prompt 方案后唯一必改的训练侧文件。

---

## 6. 实验建议

1. **先跑方案 B**（改动最小、风险最低、可立刻验证 KV cache 收益）
2. **再跑方案 E**（结构最规范、与原生 function calling 对齐，长期可维护性最好）
3. **方案 C** 是「prompt 最干净」的极限形态，适合验证「模型能否纯靠视觉感知事件」的假设
4. **方案 D** 用 `2N` 张关键帧替代主视频，对中短视频（事件数 ≤ 16）token 代价显著低于 baseline；超长视频建议把 `N_KEYFRAMES_PER_EVENT` 降到 1

可以 4 个数据集并行生成，然后用同一份 [`sft_events.sh`](../sft/sft_events.sh) 切换 `--dataset` + `--output_dir` 跑 4 组对照实验，对比指标：
- 训练 loss / 收敛速度
- 工具调用准确率（`tool_event_ids` ∩ `covering_event_ids`）
- 下游评测分（VideoMME / LVBench / MLVU 等）

---

## 7. 与现有训练流程的兼容性诊断

> 把 4 套数据扔进现有训练流程（[`sft/sft_events.sh`](../sft/sft_events.sh) / [`rl/grpo_events.sh`](../rl/grpo_events.sh) / [`rl/video_event_plugin.py`](../rl/video_event_plugin.py)）会发生什么？以下按"环节 × 方案"逐项核对。

### 7.1 总览矩阵（已含修复进度）

| 环节 | 涉及文件 | B | C | D | E |
|------|---------|---|---|---|---|
| SFT swift 训练 | [`sft_events.sh`](../sft/sft_events.sh) | ✅ | ✅ | ⚠️ 需加 image 像素参数 | ⚠️ 需验证 `tools` 字段渲染 |
| Loss scale 插件 | [`loss_scale_plugin.py`](../sft/loss_scale_plugin.py) | ✅ | ✅ | ✅ | ✅ |
| **RL rollout 事件解析** | [`video_event_plugin.py`](../rl/video_event_plugin.py:139-171) | ✅ 已修复 | ✅ 已修复 | ✅ 已修复 | ✅ |
| **RL rollout 主视频路径** | [`video_event_plugin.py`](../rl/video_event_plugin.py:173-201) | ✅ | ✅ | ✅ 已修复 | ✅ |
| **RL 数据转换** | `convert_rl_sample` | ✅ | ✅ | ✅ 已修复 | ✅ |
| RL 奖励计算 | [`video_event_plugin.py`](../rl/video_event_plugin.py) | ✅ | ✅ | ✅ | ✅ |
| 数据准备脚本协调 | [`prepare_event_data.sh`](prepare_event_data.sh) | ⚠️ 需手动跑 | ⚠️ | ⚠️ | ⚠️ |

---

### 7.2 RL rollout 的 system 解析（B/C/D 阻断点 → ✅ 已修复）

**原问题**：[`EventLocatingScheduler._parse_events_from_system()`](../rl/video_event_plugin.py:129-137) 在 rollout 时**用 regex 从 system prompt 文本里抠事件列表**，强依赖 baseline 的 `Event N: a.bs - b.cs` 字面格式。B 移到了 user / C 抹掉时间戳 / D 完全没有此格式 → 返回 `None` → scheduler 收到 `"[Error] No valid event selection."` → tool 调用全部失败。

**修复**：在 [`rl/video_event_plugin.py`](../rl/video_event_plugin.py:139-171) 中：

- 原 `_parse_events_from_system` 改名 `_parse_events_from_system_text`，**仅作兜底**（baseline 数据行为 100% 不变）。
- 新增 `_get_events(infer_request)` 作主入口，4 级 fallback：

  ```
  1. infer_request.events            （属性）
  2. infer_request.data_dict['events']
  3. infer_request['events']         （dict-like）
  4. system 文本 regex 兜底（baseline 行为）
  ```

- 数据格式异常时降级到兜底 + warning，不会让 rollout 崩。
- 调用点 [`step()`](../rl/video_event_plugin.py:178-212) 同步替换。

**效果**：B/C/D 的样本只要顶层保留 `events` 字段，rollout 就能拿到结构化事件列表，prompt 形态完全解耦。

---

### 7.3 方案 D 独有的两个缺口（→ ✅ 已修复）

#### 缺口 1：RL 数据转换函数未 patch（→ ✅ 已修复）

**原问题**：[`convert_annotations_d.py`](convert_annotations_d.py) 旧版**只 patch 了 `convert_sft_sample`，没 patch `convert_rl_sample`**。用方案 D 处理 RL 数据时，`build_system_prompt` 已被换成「2N keyframes...」版本，但 messages 和 videos 完全不变（仍是 `<video>` + 主视频）→ system 描述与实际输入严重错位。

**修复**：

- 把原 SFT 改造逻辑抽成共享函数 [`_apply_keyframe_rewrite()`](convert_annotations_d.py)，返回三态 `"ok" | "skip" | "drop"`。
- 新增 `convert_rl_sample` patch，调用同一函数，RL 强制 `do_extract=True`（关键帧是 D 的硬输入，与 `do_crop` 解耦；已存在的帧文件自动跳过）。
- SFT / RL 两条路径产出的样本结构完全对齐：相同的 system / `<image>` 占位 / `images` / `source_video` 字段。

#### 缺口 2：rollout 时拿不到主视频路径（→ ✅ 已修复）

**原问题**：旧版 [`step():147-148`](../rl/video_event_plugin.py) 用 `current_video_path = infer_request.videos[0]`，方案 D 的 `videos` 为空 → 任何 tool 调用都拿不到原视频路径。

**修复**：

- 转换侧：[`_apply_keyframe_rewrite()`](convert_annotations_d.py) 给样本顶层注入 `source_video`（主视频相对路径）。
- rollout 侧：新增 [`_get_source_video(infer_request)`](../rl/video_event_plugin.py:173-201)，4 级 fallback：

  ```
  1. infer_request.source_video        （属性）
  2. infer_request.data_dict['source_video']
  3. infer_request['source_video']     （dict-like）
  4. infer_request.videos[0]           （baseline 兜底）
  ```

  相对路径自动 `os.path.abspath(...)` 拼成绝对路径。其他方案保持 `videos[0]` 兜底，行为不变。

#### 缺口 3：SFT 训练脚本缺 image 像素参数（⚠️ 待办）

[`sft_events.sh:9-12`](../sft/sft_events.sh:9-12) 只设置了 video 相关像素参数。方案 D 引入大量 `<image>` 输入，需要补充 `MAX_PIXELS` / `MIN_PIXELS`（Qwen2.5-VL 单张图像素上下限），否则用默认值可能与训练目标不一致。**改训练脚本时一并加上即可**。

---

### 7.4 方案 E 的两个待验证点

#### 验证点 1：`tools` 字段与手写 system 的冲突

方案 E 的样本**同时含**：
- 手写 system（含事件列表 + think/answer 约束）
- 顶层 `tools` 字段（OpenAI 风格 schema）

Qwen2.5-VL 的 ChatML chat template 默认会把 `tools` 渲染进 system prompt：

```
# Tools
You may call one or more functions...
<tools>{...json schema...}</tools>
For each function call, return a json object within <tool_call></tool_call>...
```

如果 swift template 把这段**追加**在我们手写 system 之后，最终 system 会变成"手写部分 + 自动注入部分"的拼接 —— 内容上不冲突但**比 baseline 更长**，没达成「剥离工具说明」的初衷。

**验证方法**：跑一条样本，打印渲染后的 system prompt。

**两个清理方向**：
- A. 手写 system 完全交给 swift template（只保留 think/answer 约束），让 tools 自动注入
- B. 不用 `tools` 字段，仍把工具说明写在 system —— 那方案 E 就退化成"把工具说明从内联示例改成自然语言提示词"

#### 验证点 2：RL rollout 端的 tool_call 解析

方案 E 不影响 [`parse_event_ids()`](../rl/video_event_plugin.py:49-62)（仍 regex 抓 `<tool_call>...</tool_call>` 内的 JSON），RL 推理这边兼容 ✅。

---

### 7.5 训练脚本统一参数化（→ ✅ 已完成）

所有训练侧脚本现已支持通过 `PROMPT_STYLE` 环境变量切换 5 套数据，所有方案共用同一份脚本。

| 脚本 | 支持的 PROMPT_STYLE | 关键变更 |
|------|---------------------|---------|
| [`prepare_event_data.sh`](prepare_event_data.sh) | `baseline\|b\|c\|d\|e` | case 分发到 5 个 `convert_annotations*.py`；输出 `{sft,rl}/data_events{_suffix}` |
| [`sft/sft_events.sh`](../sft/sft_events.sh) | 同上 | 数据集路径用 `$DATA_DIR` 拼接；新增 `MAX_PIXELS` / `MIN_PIXELS`（方案 D 必需）；`OUTPUT_DIR` 默认含 PROMPT_STYLE |
| [`rl/grpo_events.sh`](../rl/grpo_events.sh) | 同上 | 同上；新增 `MODEL` 环境变量 |
| [`rl/rollout_events.sh`](../rl/rollout_events.sh) | 同上 | **方案 D 自动放宽** `--vllm_limit_mm_per_prompt` 的 `image` 数（baseline/b/c/e 保持 1，d 调到 64）；其他方案像素参数无影响 |

**使用示例**：

```bash
# 方案 D 的端到端流程
PROMPT_STYLE=d bash scripts/prepare_event_data.sh
PROMPT_STYLE=d bash sft/sft_events.sh
# RL（两个终端）
PROMPT_STYLE=d MODEL=sft/ckpt/test_events_d/checkpoint-xxx bash rl/rollout_events.sh
PROMPT_STYLE=d MODEL=sft/ckpt/test_events_d/checkpoint-xxx bash rl/grpo_events.sh
```

**[`rollout_events.sh`](../rl/rollout_events.sh) 中的隐藏陷阱**：原 `--vllm_limit_mm_per_prompt '{"image": 1, "video": 10}'` 硬限制 1 张 image，方案 D 一条样本最多 2×N≈30+ 张关键帧，超限会被 vllm 拒绝。**改造后按 PROMPT_STYLE 自动调整**，避免踩坑。

---

### 7.6 修复进度

| 方案 | 状态 |
|------|------|
| **B** | ✅ 已通过 `_get_events()` 元数据读取修复 |
| **C** | ✅ 同上 |
| **D** | ✅ events 读取 / `source_video` 注入 / scheduler 读字段 / RL 数据转换 patch / 训练脚本 image 像素参数 全部完成 |
| **E** | ⚠️ 待运行环境验证 `tools` 字段的 swift template 渲染行为 |

**本次涉及的修改文件**：

- [`rl/video_event_plugin.py`](../rl/video_event_plugin.py) — 新增 `_get_events()` + `_get_source_video()`；`step()` 调用点同步替换
- [`scripts/convert_annotations_d.py`](convert_annotations_d.py) — 抽出共享改造函数 `_apply_keyframe_rewrite()`；新增 `convert_rl_sample` patch；样本注入 `source_video` 字段
- [`scripts/prepare_event_data.sh`](prepare_event_data.sh) — 加 `PROMPT_STYLE` 分发
- [`sft/sft_events.sh`](../sft/sft_events.sh) — 加 `PROMPT_STYLE` / `MAX_PIXELS` / `MIN_PIXELS`
- [`rl/grpo_events.sh`](../rl/grpo_events.sh) — 加 `PROMPT_STYLE` / `MAX_PIXELS` / `MIN_PIXELS` / `MODEL`
- [`rl/rollout_events.sh`](../rl/rollout_events.sh) — 加 `PROMPT_STYLE` / 按方案调整 `vllm_limit_mm_per_prompt`

**剩余待办**：

1. 方案 E：跑一条样本打印渲染后的 system，确认 `tools` 字段被 Qwen2.5-VL chat template 正确注入；若 swift 会自动追加 tools 描述，可考虑精简 [`convert_annotations_e.py`](convert_annotations_e.py) 手写 system 中的部分内容避免重复。

---

## 8. 未来优化方向：粗采样信息丢失问题

### 8.1 问题本质：粗采样的不可逆性

baseline / B / C / E 四套方案第一步都是「整段原视频」，由 ms-swift / Qwen2.5-VL VideoProcessor 在线采样。受环境变量约束（[`sft/sft_events.sh:45-50`](../sft/sft_events.sh:45-50) 与 [`rl/rollout_events.sh:41-46`](../rl/rollout_events.sh:41-46)）：

```bash
FPS_MAX_FRAMES=512        # 单视频最多 512 帧
fps=2 (Qwen2.5-VL 默认)   # 等距抽帧
VIDEO_MIN/MAX_PIXELS=50176  # 每帧 ≈ 224×224
```

采样公式：`nframes = floor(min(duration×fps, FPS_MAX_FRAMES, total_frames) / 2) × 2`

→ 长视频会被截断到 512 帧。例如 10 分钟视频 → **每 ~7s 才采 1 帧**。

**致命链路**：

```
原视频 18000 帧 ──采样到 512 帧──▶ 模型 Turn 1 看到稀疏帧
                                          │
                                          ▼
                  ┌─ 看到了的事件 → 可能发 locate_events 工具调用
                  └─ 没看到的事件 → 永远不会被选中（盲区不可见 → 不会被探测）
```

`locate_events` 工具只能**放大已看到的细节**，无法挽回 Turn 1 完全没看到的内容。**System prompt 里只有事件时间列表**（如 `Event 5: 30s-45s`）→ 纯文本元数据无法补偿视觉缺失，模型不知道 Event 5 里发生了什么，也就不知道值不值得 locate。

### 8.2 5 种优化方向（按改动成本排序）

| 方向 | 思路 | 改动量 | Token 额外开销 | 是否结构性消除盲区 |
|------|------|--------|----------------|---------------------|
| **方案 F**（修改 prompt） | 明确告知模型采样稀疏，鼓励主动 tool 调用 | 5 行 prompt | ~50 tokens | ❌ 行为引导而非结构修复 |
| **方案 G**（事件缩略图锚点） | 主视频 + 每事件 1 张代表帧（D 的非破坏版本） | 1 个新转换脚本 (~30 行) | N×64 tokens (~1k) | ✅ 完全消除 |
| **方案 H**（自适应关键节点） | 复用 CLIP CLS 计算抽出 top-K 高变化点帧 | 改 preprocess + 新转换脚本 | K×64 tokens (~2k) | ✅ 概率性消除 |
| **方案 J**（事件级 caption + 代表帧） | 离线 VLM 为每事件生成 1 句 caption，与代表帧一并嵌入 system prompt | 1 个 captioner 脚本 + 1 个转换脚本 | N×64 + N×30 (~2k) | ✅ 完全消除（图+文双通道） |
| **方案 I**（两阶段 zoom） | 拆 `locate_events` 为 `coarse_zoom` + `fine_zoom` 渐进细化 | 重构 plugin + reward | 视轮次而定 | ✅ 主动探测，理论最优 |

### 8.3 方案 F — 视觉盲区提示词增强（最低成本，作为所有方案的通用补丁）

> **核心思想**：让模型知道自己的视觉局限，把"应该调用工具"的判断显式内化到 prompt 中。即使盲区客观存在，至少让模型对每个未被强证据覆盖的事件**默认怀疑**，提高 tool 调用率。

**两条独立改造**（可叠加）：

#### F1：在 system prompt 中明示采样稀疏性

baseline 当前 prompt（[`convert_annotations.py:38-51`](convert_annotations.py:38-51)）只描述了"视频已切成事件"，没有任何关于**采样稀疏**的提示。建议在事件列表后追加：

```
IMPORTANT: The video is sampled at low frame rate. Brief events (especially those
shorter than a few seconds) may be poorly represented or invisible in your initial
view. If you cannot confidently answer the question from the sampled frames alone,
or if you suspect important details in any specific event, you SHOULD use
locate_events to retrieve full clips before answering.
```

预期效果：在 RL 训练中，配合 [`Event_Reward`](../rl/video_event_plugin.py:339-353) 的 F1 信号，鼓励模型形成「不确定 → 调用」的决策习惯，而非「文本元数据足够 → 直接答」的捷径。

#### F2：在 tool 失败/未用时给出反馈式 user 提示

当模型 Turn 1 直接给 `<answer>` 而未调用 tool 时，当前不会有任何反馈。可考虑在 RL rollout 阶段，对**低置信度场景**（如答案中含 "maybe"、"unclear"、"I think" 等）追加一轮 user 提示：

```
You answered without examining specific events in detail. If you are uncertain,
you may use locate_events to retrieve close-up clips for events that might
contain the key information, then revise your answer.
```

这条更复杂（需要置信度检测），属于第二阶段优化。

#### F3：每个事件附带"信息密度提示"（依赖方案 H）

如果做了方案 H（自适应关键节点检测），可以把 CLIP 相似度方差作为「事件复杂度」指标暴露给模型：

```
The video has been segmented into the following events:
  Event 0: 0.0s - 3.2s   [low visual variation]
  Event 1: 3.2s - 27.8s  [HIGH visual variation — likely contains multiple sub-actions]
  Event 2: 27.8s - 30.1s [low visual variation]
  ...
```

让模型优先聚焦高复杂度事件 → tool 调用更有针对性。

### 8.4 方案 F 的实施路径

**复用现有 monkey-patch 模式**，新建 [`scripts/convert_annotations_f.py`](convert_annotations_f.py)：

```python
import convert_annotations as _ca

SYSTEM_PROMPT_TEMPLATE_F = _ca.SYSTEM_PROMPT_TEMPLATE.replace(
    "Use the insights from the selected event clips ",
    """IMPORTANT: The video is sampled at low frame rate. Brief events may be poorly
represented or invisible in your initial view. If you cannot confidently answer the
question from the sampled frames alone, you SHOULD use locate_events to retrieve
full clips before answering.

Use the insights from the selected event clips """
)

def build_system_prompt(events):
    return SYSTEM_PROMPT_TEMPLATE_F.format(event_list=...)  # 同 baseline

_ca.build_system_prompt = build_system_prompt

if __name__ == "__main__":
    _ca.main()
```

**训练侧零改动**：复用现有 [`prepare_event_data.sh`](prepare_event_data.sh) / [`sft_events.sh`](../sft/sft_events.sh) / [`grpo_events.sh`](../rl/grpo_events.sh)，只需在它们的 `case "$PROMPT_STYLE"` 分发表中追加 `f) ...` 分支即可。

**实验对照**：

| 实验组 | PROMPT_STYLE | 验证假设 |
|--------|--------------|---------|
| Baseline | `baseline` | 当前基线 |
| F | `f` | 仅加视觉盲区提示，验证「提示词引导」的纯增益 |
| D | `d` | 取消主视频改 keyframe，验证「结构性消除盲区」的纯增益 |
| D+F | `d` 数据 + F 的 prompt | 是否能进一步叠加（理论上 D 已无盲区，F 收益应趋于 0） |

**核心指标**：
- 第一轮就给 `<answer>` 的样本比例（应下降）
- 平均 tool 调用次数（应上升但不过度）
- 答案准确率 / Event F1（应上升）
- 短事件（< 5s）相关问题的准确率（**该方向最敏感的指标**）

### 8.5 方案 J — 事件粒度离线 caption + 视觉锚点

> **核心思想**：把"模型自己从稀疏视频帧里盲猜"改成"模型读 N 句 caption + 看 N 张代表帧主动决策"。
> 离线一次性预处理生成 caption + keyframe，**训练和推理共享同一份 metadata**，运行时零额外成本。

#### 8.5.1 与方案 G 的关系

J 是 **G 的文本增强版**：样本结构基本一致（主视频 + N 张代表帧），唯一差别在 system prompt 文本：

```
G:  Event 0: 0.0s - 12.3s [see image 0]
J:  Event 0 (0.0s-12.3s): "A man walks into a wooden room and turns on the light." [image 0]
```

模型决策路径从「看图判断」升级为「**看图 + 读 caption 双通道判断**」。caption 提供快速文本过滤：
- 问题问"开门动作" → 模型读 caption 直接锁定 Event 0、Event 2，跳过其他事件的视觉详查
- 问题问"红色物体" → caption 没提颜色 → 模型才需要走 `locate_events` 看视觉

#### 8.5.2 离线 pipeline（与方案 D 高度复用）

```
[一次性离线预处理]
原视频
  ↓ Step 1: preprocess_scenes.py        (CLIP CLS 切场，已有)
  ↓ Step 2: convert_annotations_d.py    (抽 event_clips/*.mp4 + 代表帧，已有)
  ↓ Step 3: ⭐ generate_event_captions.py (新增：VLM 读 clip → 1 句 caption → 写入 metadata)
  ↓ Step 4: convert_annotations_j.py    (新增：组装样本，把 caption 嵌入 system prompt)
sft/data_events_j/  +  scripts/event_captions.json
```

**关键性质**：
- **训练集 + 测试集 + 部署场景** 都用同一份 `event_captions.json` → **零 train-test mismatch**
- **运行时不需要 captioner**：metadata 写死，推理 0 额外延迟
- **与方案 D 流水线天然耦合**：Step 3 输入就是 D 的 `event_clips/*.mp4`，多一道 VLM forward 即可

#### 8.5.3 Captioner 选型与成本估算

假设规模：5 万视频 × 平均 8 事件 = **40 万次 caption 调用**。

| Captioner | 单 clip 时间 | 8×A100 总时长 |
|-----------|--------------|----------------|
| Qwen2.5-VL-7B（自洽，与训练模型一致） | ~3-5s | ~200 GPU·h |
| Qwen2.5-VL-3B | ~1-2s | ~80 GPU·h |
| InternVL2-2B / SmolVLM-2.2B | ~0.5-1s | ~40 GPU·h |

**一次性投入，永久使用**。最便宜方案 ~5 小时跑批完毕。

#### 8.5.4 「白嫖」捷径：复用现成数据集 caption

部分数据集本身就携带事件级 caption，可**零 GPU 成本**注入：

| 数据集 | 是否含事件级 caption | 字段 |
|--------|---------------------|------|
| ActivityNet Captions | ✅ | `sentences` (与 `timestamps` 对齐) |
| VidChapters-7M | ✅ | `chapter_titles` |
| LongVila | ⚠️ 看版本 | — |
| Charades / QVHighlights / Video-R1 | ❌ | — |

在 [`scripts/preprocess_scenes.py`](preprocess_scenes.py) 输出 metadata 时按时间戳对齐：

```python
for ev in events:
    overlapping = [
        s for s, (st, et) in zip(orig_sentences, orig_timestamps)
        if min(et, ev["end_time"]) - max(st, ev["start_time"]) > 0.5
    ]
    ev["caption"] = " ".join(overlapping) or None  # None → 后续由 VLM 补
```

**推荐两步走**：先白嫖 ActivityNet/VidChapters 的 caption 看效果；如果有显著收益，再投资 VLM 补齐 Charades 等没 caption 的数据集。

#### 8.5.5 实施路径

新增两个脚本：

**1) [`scripts/generate_event_captions.py`](generate_event_captions.py)**（待实现）

```python
# 输入：sft/data_events_d/event_clips/<safe>/event_X.mp4 (或代表帧)
# 输出：event_captions.json = { "<safe>/event_X": "A man opens a door." }
# 实现：
#   - 加载 captioner（Qwen2.5-VL-3B / SmolVLM）
#   - 遍历所有 event clip，每条采 4-8 帧调 captioner
#   - prompt: "Describe what happens in this short video clip in one sentence."
#   - 控制输出长度（max_new_tokens=40），过滤空输出
#   - 增量保存：每 1000 条 dump 一次，断点续跑
```

**2) [`scripts/convert_annotations_j.py`](convert_annotations_j.py)**（待实现）

```python
import convert_annotations as _ca
import json

with open("scripts/event_captions.json") as f:
    CAPTIONS = json.load(f)

SYSTEM_PROMPT_TEMPLATE_J = """You are a helpful assistant.
...
The video has been segmented into the following events (each with a brief
description and a representative frame):
{event_list}

If you need to examine specific events more closely, use:
<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[...]}}}}</tool_call>
"""

def build_system_prompt(events, video_key):
    lines = []
    for i, e in enumerate(events):
        cap = CAPTIONS.get(f"{video_key}/event_{e['event_id']}", "(no description)")
        lines.append(f"  Event {e['event_id']} ({e['start_time']:.1f}s-{e['end_time']:.1f}s): \"{cap}\" [image {i}]")
    return SYSTEM_PROMPT_TEMPLATE_J.format(event_list="\n".join(lines))

# 视觉部分：直接复用方案 G 的「主视频 + 每事件 1 张代表帧」改造
# (或基于方案 D 的 keyframe 抽取，N_KEYFRAMES_PER_EVENT=1)
...
```

**训练侧改动**：4 个 shell 脚本的 `case "$PROMPT_STYLE"` 分发表追加 `j) ...` 分支，与方案 D 类似（需要 image 像素参数）。

#### 8.5.6 风险与缓解

| 风险 | 缓解 |
|------|------|
| **VLM caption 幻觉污染训练信号** | (1) 用 Qwen2.5-VL-3B+ 而非 2B 小模型；(2) 抽样 100 条人工质检；(3) caption 模板严格限定"Describe what is visible"避免推断 |
| **caption 长度膨胀挤占视觉 token** | 限制 max_new_tokens=40，平均 ~20-30 tokens/事件，30 事件总开销 ~900 tokens（可接受） |
| **caption 跟问题语义不对齐** | 这是 caption 通用性的固有问题；缓解：先用通用 caption；若效果不足再用 question-conditioned caption（推理时根据问题二次生成，但破坏离线一致性，不推荐） |
| **离线 caption 跟训练时模型理解的不对齐** | 推荐用与训练模型同源的 captioner（Qwen2.5-VL-3B → 训练 Qwen2.5-VL-7B），减少语义偏移 |

#### 8.5.7 与已有方案的可实施性对比

| 维度 | F | G | **J** | D | H | I |
|------|---|---|-------|---|---|---|
| 离线投入 | 0 | 秒级 | **40-200 GPU·h（一次性）** | 同 D 抽帧 | CLIP 已有 | 0 |
| 与 D 流水线复用 | 无 | 高 | **极高（直接复用 event_clip）** | — | 高 | 无 |
| 训练-推理一致性 | ✅ | ✅ | ✅（离线 metadata 写死） | ✅ | ✅ | ✅ |
| 部署侧改动 | 0 | 0 | **0**（不需要部署 captioner） | 0 | 0 | reward 重构 |
| 结构性消除盲区 | ❌ | ✅ | ✅+文本通道 | ✅ | ✅ | ✅ |

**结论**：方案 J 是「**G 的能力上限版**」—— 在 G 已经消除视觉盲区的基础上，再加一个**文本快速过滤通道**，让 tool 调用决策更精准。离线投入比 G 大一个量级，但运行时与 G 等价。

---

### 8.6 推荐落地顺序

```
F (提示词)              → 半小时
  ↓
G (主视频 + 代表帧)      → 1 天
  ↓
J-lite (G + 白嫖 caption) → 1-2 天 (只跑 ActivityNet/VidChapters 子集)
  ↓
J 完整 (J-lite + VLM 补齐其他数据集) → 1 周（含 captioner 跑批）
  ↓
H (自适应关键节点)        → 2 周（如果 J 仍有盲点）
  ↓
I (两阶段 zoom)          → 3-4 周（远期架构升级）
```

**5 组对照实验设计**（推荐先做这一组完整 A/B）：

| 实验组 | 增量 | 验证假设 |
|--------|------|---------|
| baseline | — | 当前基线 |
| F | 仅 prompt 警告 | 「提示词引导」纯增益 |
| G | F + 每事件 1 张代表帧 | 「视觉锚点消除盲区」纯增益 |
| J-lite | G + 白嫖 caption（部分数据） | 「文本目录通道」纯增益 |
| D | 取消主视频改 keyframe | 「全 keyframe」与 G 比，主视频是否必要 |

通过这 5 组可精确归因：prompt 帮的忙、图帮的忙、文本 caption 帮的忙、是否还要保留主视频。

---

## 9. 文件清单

| 文件 | 说明 |
|------|------|
| [`convert_annotations.py`](convert_annotations.py) | Baseline，未改动 |
| [`convert_annotations_b.py`](convert_annotations_b.py) | 方案 B：信息搬家 |
| [`convert_annotations_c.py`](convert_annotations_c.py) | 方案 C：剥离时间戳 |
| [`convert_annotations_d.py`](convert_annotations_d.py) | 方案 D：视觉锚点 |
| [`convert_annotations_e.py`](convert_annotations_e.py) | 方案 E：原生 tools schema |
| [`convert_annotations_f.py`](convert_annotations_f.py) | 方案 F：视觉盲区提示词增强（待实现） |
| [`convert_annotations_j.py`](convert_annotations_j.py) | 方案 J：事件级 caption + 代表帧（待实现） |
| [`generate_event_captions.py`](generate_event_captions.py) | 方案 J 的离线 caption 生成脚本（待实现） |
| [`event_captions.json`](event_captions.json) | 方案 J 的 caption 元数据（待生成） |
| [`annotation_analysis.md`](annotation_analysis.md) | 本文档 |
