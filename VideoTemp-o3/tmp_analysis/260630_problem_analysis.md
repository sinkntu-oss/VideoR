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

---

# 奖励设计审查（2026-06-30）

> 审查对象：`rl/video_event_plugin.py` 的 4 个奖励函数（acc / event / format / tool_penalty）。
> 结论：计算正确性基本无 bug，但**激励方向**存在结构性缺陷，部分与论文
> "agentic thinking-with-videos" 初衷相悖。以下按严重程度排列，含问题/后果/修复方向。

## 🔴 P0 - 缺陷 1：激励倒挂，RL 会学会"少用甚至不用工具"

**问题**：四个奖励默认等权相加，但没有任何机制奖励"主动调用工具"，
`ToolPenalty` 只罚多调用；`Event_Reward` 中"未调用工具"(`sl` 空)返回 0(中性)，
"调用但选错"返回 −0.1(负)。

**后果**：以 QA 样本三种轨迹（GRPO 组内对比）为例：
- A. 不调用工具直接答对：acc1 + event0 + format1 + tool0 = **2.0**
- B. 调用工具选对答对：acc1 + event1 + format1 + tool0 = 3.0
- C. 调用工具选错蒙对：acc1 + event(−0.1) + format1 + tool0 = **1.9**

A(2.0) > C(1.9)：模型不确定时"干脆不调用"优于"调用但选错"，
RL 收敛到回避工具调用，退化为"看全片直接答"，违背核心卖点。

**修复方向**：让"调用且选对"始终严格优于"不调用"；对"未调用"对称化处理
（给小负基线），或显式奖励合理调用。

## 🔴 P0 - 缺陷 2：定位奖励被 acc 门控 → 冷启动稀疏（鸡生蛋）

**问题**：`Event_Reward` 要求 `acc >= 0.5` 才算定位 F1。对 QA，acc 是 0/1，
即必须先答对才奖励定位，而答对又依赖看对片段。

**后果**：训练早期答对率低 → event_reward 恒为 0 → 定位无梯度 →
看不到正确片段 → 答对率上不去。定位这一更稠密的早期信号被绑死在最终答案上。

**修复方向**：解耦，答案错时也给降权的定位 F1，让定位先于答案学起来。

## 🟠 P1 - 缺陷 3：grounding 奖励语义重复且混乱

`acc_reward`(用 answer 文本时间戳→事件)与 `event_reward`(用 tool_call 选事件)
对 grounding 都在算事件 F1，复用同一 `covering_list`。名为 Accuracy 实为 F1，
与 QA 的 0/1 语义不同尺度；且事件版删除了原 `normalize_list` 跨任务归一化，加剧尺度不一致。

## 🟠 P1 - 缺陷 5：多选"双重惩罚"且对 grounding 失效

- 双重惩罚：`Event_Reward` 累积所有轮 event_ids 会拉低 F1 precision，
  `ToolPenalty` 又对多选额外扣分，同一行为被罚两次。
- grounding 失效：`ToolPenalty` 只读 `covering_event_ids`，而 grounding GT 存在
  `gt_covering_event_ids`，导致 `cov=[]`，多选惩罚不触发，口径不一致。

## 🟠 P1 - 缺陷 4：F1<0.1 的 −0.1 造成奖励不连续

`f1 if f1>=0.1 else f1-0.1` 在 0.1 处有 0.1 跳变，优势估计在阈值附近突变，
magic number 无依据。修复：用连续塑形替代硬阈值。

## 🟡 P2 - 缺陷 6：QA 答案提取脆弱

`first_letter` 取首个字母，`<answer>The answer is C</answer>` 会取到 'T' 判错。
评测侧 `_extract_videomme_answer` 已处理前缀，奖励侧未处理，RL 探索期会引入标签噪声。

## 🟡 P2 - 缺陷 7：FormatReward 全有全无 + 索引脆弱 + 权重偏大

- 任一轮格式错整条归零，长轨迹信号稀疏；
- `range(2,len(msgs),2)` 硬编码 assistant 在偶数位，结构偏移会误判；
- format∈{0,1} 与 acc 等权，模型可能优先凑格式而非答对。

## 🟡 P2 - 缺陷 8：整体 reward 尺度未平衡

acc∈{0,1}、event∈[−0.1,1]、format∈{0,1}、tool∈[−0.5,0]，等权相加时
format+acc 主导，定位(常被门控为 0)权重被压低；若未配 `reward_weights`，定位信号被淹没。

## 优先级汇总

| 优先级 | 缺陷 | 后果 |
|--------|------|------|
| P0 | 1 激励倒挂 | RL 收敛到"不用工具"，违背论文核心 |
| P0 | 2 acc 门控 | 定位冷启动稀疏，鸡生蛋 |
| P1 | 3 / 5 | 奖励口径不一致、双重惩罚 |
| P1 | 4 / 8 | 优化不稳定、定位信号被淹没 |
| P2 | 6 / 7 | 标签噪声、格式凑数 |

## 结论

奖励在数学上自洽，但把模型推向"少定位、直接答"，与 VideoTemp-o3 想要的
"主动按需定位"相反。最该优先解决缺陷 1（激励倒挂）与缺陷 2（定位门控）。

---

# 奖励设计修复记录（2026-06-30）

> 修改文件：`rl/video_event_plugin.py`（4 个奖励函数）、`rl/grpo_events.sh`（reward_weights）
> 验证：语法检查通过；独立数值模拟确认 B>A（鼓励调用选对）、C≥A（倒挂消除）。

## 🔴 P0 - 缺陷 1/2/4：重构 Event_Reward（已修复）

将 `Event_Reward` 由"acc 门控 + f1<0.1 减 0.1"改为**纯定位 F1**：
```python
return [self._compute_event_f1(sl[i], cl[i]) if (cl[i] and sl[i]) else 0.0
        for i in range(len(tids))]
```
- 缺陷 2：去掉 `acc>=0.5` 门控 → 答案错也给定位 F1，解除冷启动鸡生蛋；
- 缺陷 4：去掉 `-0.1` 负跳变 → 奖励连续；
- 缺陷 1：因去掉 `-0.1`，"调用全错"(0) 不再低于"不调用"(0)，倒挂消除；"调用选对"(>0) 仍严格更高。

验证：A 不调用=2.0 / C 调用选错=2.0 / B 调用选对=3.0 → B>A=C，倒挂消除。

## 🔴 P1 - 缺陷 5：简化 ToolPenalty（已修复）

移除"过度多选 −0.05/个"惩罚，仅保留"多次调用 −0.1/次（下限 −0.5）"：
- 多选已由 Event_Reward 的 F1 precision 自然惩罚，避免双重施压；
- 不再依赖 `covering_event_ids`，消除原先对 grounding(用 `gt_covering_event_ids`)失效的口径不一致。

## 🟠 P2 - 缺陷 7：重构 FormatReward（已修复）

从"全有全无"改为**合格 assistant 轮占比 (ok/总轮数, 0~1)**，更稠密；
并遍历 `role=='assistant'` 的消息替代硬编码 `range(2,len,2)`，消除结构偏移误判。

验证：2/3 轮合格 → 0.67。

## 🟡 P2 - 缺陷 6：增强 QA 答案提取（已修复）

`first_letter` 增加前缀剥离（"The answer is" / "Answer:" 等），
避免 `<answer>The answer is C</answer>` 被误取为 'T'。

## 🟠 P1 - 缺陷 8：配置 reward_weights（已修复）

在 `grpo_events.sh` 增加：
```
--reward_weights 1.0 0.5 1.0 0.2 \
```
对应 `acc_reward event_reward tool_penalty format_reward`：突出答案(1.0)与工具惩罚(1.0)，
定位为辅助稠密信号(0.5)，格式降权(0.2)避免"凑格式压过答对"。

## 🟡 缺陷 3：grounding 语义（设计权衡，未改代码）

`acc_reward`(答案时间戳→事件 F1) 与 `event_reward`(工具选择 F1) 分别衡量
"答案质量"与"中间定位质量"，关注点不同，保留为合理分工；
跨任务尺度差异由 GRPO 组内 advantage 归一化（同 prompt 同任务）缓解。

## 修复后激励结构（covering={1,2}, 权重 1.0/0.5/1.0/0.2）

| 轨迹 | raw | weighted |
|------|-----|----------|
| A 不调用答对 | 2.00 | 1.20 |
| B 调用选对答对 | 3.00 | **1.70** |
| C 调用选错(1次) | 2.00 | 1.20 |
| C2 调用选错(2次) | 1.90 | 1.10 |

B 严格最优（鼓励主动定位且选对），C≥A（不再因"尝试定位"被倒挂惩罚），
C2 因重复调用被合理惩罚。激励方向已与"agentic 按需定位"对齐。
