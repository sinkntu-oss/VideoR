# 数据集转换方法问题分析

> 分析对象：事件定位改造方案的标注转换脚本 `scripts/convert_annotations.py`
> 日期：2026-06-30

## 背景

事件定位改造把"自由时间戳裁剪"转换为"离散事件 ID 选择"。转换流水线为：
`preprocess_scenes.py`（场景切分）→ `convert_annotations.py`（标注转换）。
本文记录转换方法中发现的问题、描述及解决方向，按严重程度排列。

---

## 🔴 P0 - 问题 1：`<video>` 占位符与 `videos` 数组数量不匹配（必崩）

**问题描述**：
含工具调用的 SFT 样本是多轮结构，第二个 user 消息含 N 个 `<video>`（对应 N 个裁剪片段），
`videos = [主视频, cropped_1, ..., cropped_N]`。
转换时 `convert_sft_sample()`：
- 第 ③ 步只有注释、无代码，第二个 user 消息的 N 个 `<video>` 标签原样保留；
- 第 ⑤ 步却把所有 `cropped_video` 从 videos 数组删除。

结果：messages 里有 `1+N` 个 `<video>` 占位符，但 videos 只剩 `[主视频]`。
ms-swift / Qwen2.5-VL 要求 `<video>` 数量与 videos 严格相等，否则数据加载阶段直接报错。

**解决方法**：
转换时同步重写"多模态对齐层"：根据新的 `event_ids` 数量重写第二个 user 消息的 `<video>`
标签数量，并把对应事件的视频片段路径补回 videos 数组（而非简单删除 cropped_video）。

---

## 🔴 P0 - 问题 2：`<think>` 推理链与 `<tool_call>` 动作不一致

**问题描述**：
`convert_think_text_references()` 明确不修改 think 内容。但新系统提示词已无"时间戳"概念，
tool_call 已改为 `locate_events([...])`，而 think 里仍残留"查看 30-45 秒"这类时间戳推理。
SFT 会学到自相矛盾的"推理→动作"映射，污染思维链质量。

**解决方法**：
重写 think 中的时间戳引用为事件引用（如"查看 Event 2、Event 3"），保证推理链与
动作、系统提示词三者语义一致。

---

## 🔴 P0 - 问题 3：训练数据与运行时插件的视频数量不一致

**问题描述**：
原始 N 个时间段 → N 个片段；转换后 `locate_events` 选 M 个事件（M=最小覆盖集大小，通常 M≠N）。
运行时插件 `EventLocatingScheduler.step()` 为 M 个事件生成 M 个 `<video>`，
但 SFT 转换仍保留 N 个 `<video>`。导致 SFT 看到的片段数与推理/RL 阶段插件产出的片段数
系统性不一致，破坏 train-inference 一致性。

**解决方法**：
让 SFT 的片段数严格等于 event 数（M），并与运行时插件产出对齐；
统一以"选中事件"为单位组织多模态输入。

---

## 🟠 P1 - 问题 4：奖励漏洞——空覆盖集奖励"不作为"

**问题描述**：
当 `covering_event_ids` 为空时，`_compute_event_f1()` 中 `target` 为空则"不选"反而得满分（1.0）。
若 timestamp 超出视频时长、为退化区间或路径错位导致 events 异常，会产生空集，
从而激励模型干脆不调用工具。转换侧未对空覆盖集做校验/过滤。

**解决方法**：
转换时校验并过滤空覆盖集样本；奖励函数对空 target 的退化逻辑加防护
（如空 target 不给正奖励）。

---

## 🟠 P1 - 问题 6：路径匹配脆弱，静默丢弃样本

**问题描述**：
`convert_sft_sample()` 用两次尝试匹配 metadata key，失败即 `return None` 整条丢弃。
preprocess 存的 key 是 `relpath(video, project_root)`，JSONL 路径若有细微差异
（前导 `./`、软链接、绝对/相对混用）即匹配失败，大规模静默丢样本，仅 debug 级日志提示。

**解决方法**：
统一 metadata key 与 JSONL 路径的规范化（normpath / realpath）；
匹配失败时升级为 warning 并统计丢弃数量。

---

## 🟡 P2 - 问题 5：`tool_event_ids` 字段产出后无人消费且可能不自洽

**问题描述**：
第 ⑦ 步产出 `tool_event_ids`，但 SFT 训练只读 messages 和 videos，自定义字段不参与训练；
assistant 真正学到的 event_ids 来自第 ④ 步内嵌进 tool_call 文本的结果。
两者数据来源不同（`tool_params` vs assistant 原文时间戳），可能冗余且自相矛盾。

**解决方法**：
去除冗余 `tool_event_ids`，或在产出时校验其与 tool_call 文本内 event_ids 的自洽性。

---

## 🟡 P2 - 问题 7：浮点边界判定误差

**问题描述**：
scene_metadata 时间被 `round(…, 2)`，timestamp 为原始精度。
重叠判定 `ev_start < target_end and ev_end > target_start` 在边界附近有 ±0.01s 误差，
可能多选/漏选一个边界事件，影响 GT 覆盖集精度及 RL 的 F1 奖励基准。

**解决方法**：
重叠判定加入 epsilon 容差，或统一时间精度后再比较。

---

## 🟡 P2 - 问题 8：单主视频假设

**问题描述**：
SFT 只取第一个非 cropped 视频作主视频，RL 直接取 `videos[0]`。
若一条样本含多个原始视频，其余视频的事件信息全部丢失。

**解决方法**：
支持多视频样本，为每个原始视频分别生成事件列表与覆盖集。

---

## 修复优先级汇总

| 优先级 | 问题 | 核心修复方向 |
|--------|------|------------|
| P0 | 问题 1 | `<video>` 标签数随 event_ids 重写，片段路径补回 videos 数组 |
| P0 | 问题 2 / 3 | think 改写为事件引用；SFT 片段数与 event 数、插件产出对齐 |
| P1 | 问题 4 | 过滤空覆盖集；奖励对空 target 加防护 |
| P1 | 问题 6 | 路径规范化；失败升级为 warning 并统计 |
| P2 | 问题 5 / 7 / 8 | 去冗余字段 / 边界加容差 / 支持多视频 |

---

## 核心结论

转换在"标注语义映射"（最小覆盖集算法）上正确，但在**多轮对话的视频-占位符同步**
与**推理链一致性**上存在结构性缺陷。根因是转换脚本只重写了"文本层"
（tool_call、system prompt），未同步重写"多模态对齐层"
（`<video>` 标签 ↔ videos 数组 ↔ 运行时插件产出）。
其中 **问题 1 会导致事件版 SFT 直接无法运行**，应优先修复。

---

# 修复记录（2026-06-30）

> 修改文件：`scripts/convert_annotations.py`（重写）、`rl/video_event_plugin.py`（奖励防护一处）
> 所有修复均通过语法检查、核心函数自测与端到端对齐测试。

## 🔴 P0 - 问题 1/3：多模态对齐（已修复）

重写 `convert_sft_sample()`，按工具调用轮逐个对齐：
- assistant 的 `tool_call` → `locate_events(event_ids)`；
- 其后 user 消息的 `<video>` 数量重写为 `len(event_ids)`；
- `videos` 数组重建为 `[主视频] + M 个事件片段路径`（移除 cropped_video）；
- 新增 `crop_event_clip()` 按事件边界离线裁剪，采样逻辑与运行时 `EventLocatingScheduler` 完全一致，
  保证 train-inference 一致；提供 `--no_crop_clips` 开关分离裁剪步骤；
- 转换末尾做强一致性校验：`<video>` 总数 ≠ videos 数量则丢弃并计数（`align_mismatch`）。

验证：端到端测试 `video_tags=4 == videos_len=4`，输出 `ALIGN OK`。

## 🔴 P0 - 问题 2：think 推理链一致性（已修复）

新增 `rewrite_think_timestamps()`，将 `<think>` 内带时间单位的区间（如 `8s - 25s`）
改写为事件引用（`Events 1, 2, 3`）。保守匹配（要求时间单位），避免误伤普通数字。

验证：`look at 8s - 25s` → `look at Events 1, 2, 3`，普通数字 `100` 保留不变。

## 🔴 P0 - 问题 3：训练-推理片段数一致（已修复）

随问题 1 一并解决：SFT 片段数 = 选中事件数 M，与运行时插件产出严格对齐；
裁剪采样参数（FPS/帧数）与插件常量一致。

## 🟠 P1 - 问题 4：空覆盖集奖励防护（已修复）

- 运行时：`rl/video_event_plugin.py` 的 `_compute_event_f1()` 空 target 由"满分 1.0"改为返回 `0.0`，
  消除"不调用工具"的捷径激励；
- 转换侧：对空覆盖集样本进行 `empty_cover` 统计告警。

## 🟠 P1 - 问题 6：路径匹配规范化（已修复）

新增 `build_metadata_index()` + `lookup_metadata()`：对路径做 normpath 规范化、
basename 唯一兜底；未命中升级为 warning 并以 `meta_miss` 计数。

## 🟡 P2 - 问题 5：去除冗余 tool_event_ids（已修复）

不再产出 `tool_event_ids`，统一以 tool_call 文本内 event_ids 为准；删除原 `tool_params` 字段。

验证：转换输出 `has_tool_params=False`。

## 🟡 P2 - 问题 7：覆盖集 epsilon 容差（已修复）

`find_covering_events()` 改为基于**正重叠长度 + `OVERLAP_EPS`(1e-2)** 判定，
并对极短目标区间加中点兜底，避免边界相切误判。

验证：`[5,12]` 仅命中 event 1，未误纳相切的 event 0 / event 3。

## 🟡 P2 - 问题 8：单主视频假设（暂未处理）

当前仍以第一个非 cropped 视频为主视频。多视频样本较少，留待后续按需扩展。

## 诊断统计

主流程日志新增统计：元数据未命中(丢弃)、多模态对齐不一致(丢弃)、空覆盖集、事件片段裁剪失败。

## 结论

P0 全部修复，事件版 SFT 数据可正确生成（`<video>` ↔ videos ↔ event_ids 三者一致）；
P1 健壮性与奖励漏洞已闭环；P2 除"多视频支持"外均已处理。
