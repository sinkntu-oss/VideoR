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

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| [`convert_annotations.py`](convert_annotations.py) | Baseline，未改动 |
| [`convert_annotations_b.py`](convert_annotations_b.py) | 方案 B：信息搬家 |
| [`convert_annotations_c.py`](convert_annotations_c.py) | 方案 C：剥离时间戳 |
| [`convert_annotations_d.py`](convert_annotations_d.py) | 方案 D：视觉锚点 |
| [`convert_annotations_e.py`](convert_annotations_e.py) | 方案 E：原生 tools schema |
| [`annotation_analysis.md`](annotation_analysis.md) | 本文档 |
