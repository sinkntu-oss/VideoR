# RL 链路 - 事件定位版

> 基于 `EventLocatingScheduler` + `event_reward` 的多轮强化学习流水线，
> 配套支持 baseline / B / C / D / E / J 等多套 prompt 风格。

---

## 目录文件

| 文件 | 说明 |
|------|------|
| [`video_event_plugin.py`](video_event_plugin.py) | 事件定位调度器 + 4 个 reward 函数（acc/event/format/tool_penalty）|
| [`rollout_events.sh`](rollout_events.sh) | vLLM rollout 推理服务启动脚本 |
| [`grpo_events.sh`](grpo_events.sh) | GRPO 训练启动脚本 |
| [`video_crop_plugin.py`](video_crop_plugin.py) | 旧版（基于时间戳裁剪），保留作历史参考 |
| [`rollout.sh`](rollout.sh) / [`grpo.sh`](grpo.sh) | 旧版启动脚本 |

---

## 两阶段运行流程

```bash
# 终端 1：起 vLLM rollout 服务
PROMPT_STYLE=j MODEL=sft/ckpt/test_events_j/checkpoint-xxx \
    bash rl/rollout_events.sh

# 终端 2：跑 GRPO 训练（连接到上面的 vLLM 端口 8100）
PROMPT_STYLE=j MODEL=sft/ckpt/test_events_j/checkpoint-xxx \
    bash rl/grpo_events.sh
```

数据准备见 [`scripts/prepare_event_data.sh`](../scripts/prepare_event_data.sh)，
方案 J 还需先生成 caption metadata，见 [`scripts/plan_j/README.md`](../scripts/plan_j/README.md)。

---

## 全链路问题分析与修复（2026-06-30 审计）

> 系统审查 SFT + RL 全链路时识别出 10 个潜在问题。
> 🔴 **5 个高危**、🟠 **3 个中危**、本节均已修复；2 个策略性问题作为已知风险记录。

### 🔴 H1 · `EVENT_LINE_PAT` 不匹配方案 J 的 prompt 格式

**位置**：[`video_event_plugin.py`](video_event_plugin.py:54)

**问题**：原 regex `Event\s+(\d+):\s+([\d.]+)s\s*-\s*([\d.]+)s` 只识别 baseline 格式
`Event 0: 0.0s - 3.2s`，而 J 用的 `Event 0 (0.0s-3.2s): "caption"` 因为时间在括号内
且使用紧凑 `s-` 分隔符，无法匹配。

正常情况下 J 走 `infer_request.events` 字段（优先级 1），不会落到 regex 兜底。
但**一旦 ms-swift 字段透传出问题（见 H2），J 会全部 events=空 → 整条 rollout 失败**
且没有任何告警。

**修复**：改为 `Event\s+(\d+)(?:\s*\(|:\s*\(?)\s*([\d.]+)s\s*-\s*([\d.]+)s`，
`(?:\s*\(|:\s*\(?)` 同时允许「冒号+可选括号」与「直接括号」两种分隔。
单元验证两种格式均正确解析（含双位数 event_id）。

---

### 🔴 H2 · ms-swift 顶层非标准字段透传依赖

**位置**：[`scripts/plan_j/verify_fields.py`](../scripts/plan_j/verify_fields.py)（新增）

**问题**：D/J 的 RL 数据严重依赖以下顶层字段被 ms-swift 透传到 `InferRequest`：

| 字段 | 用途 | 依赖位置 |
|---|---|---|
| `events` | reward 计算 + tool 调用对照表 | [`_get_events`](video_event_plugin.py:159), [`_extract_trajectory_data`](video_event_plugin.py:330) |
| `source_video` | rollout 期间裁剪 tool 返回的 clip | [`_get_source_video`](video_event_plugin.py:220) |
| `images` | 第一轮 N 张 keyframe | Qwen2.5-VL 模板 |
| `covering_event_ids` / `gt_covering_event_ids` | event_reward 目标集 | [`_extract_trajectory_data`](video_event_plugin.py:331) |

历史上 `events` / `source_video` 已经做了 4 级 fallback，说明 ms-swift 不同版本对额外
字段的挂载方式不一致；但 `images` / `covering_event_ids` 没有 fallback。如果任一字段
被 ms-swift loader 静默丢弃，训练会**先正常启动后逐渐崩溃**，极难定位。

**修复**：新增 [`scripts/plan_j/verify_fields.py`](../scripts/plan_j/verify_fields.py)
作为**冒烟测试**——扫描转换后的 jsonl，统计字段命中率，并对关键字段做断言：

```bash
# 转换完后立刻校验字段齐全
python scripts/plan_j/verify_fields.py rl/data_events_j --project_root .
python scripts/plan_j/verify_fields.py sft/data_events_j --project_root .
```

校验项包括：`events / images / source_video / covering_event_ids` 存在性，
`<image>` ↔ `images` / `<video>` ↔ `videos` 标签数对齐，
以及随机抽 20 条样本检查 `images / source_video` 文件真实存在。

---

### 🔴 H3 · `_get_source_video` 全 fallback 失败时静默返回 None

**位置**：[`video_event_plugin.py:_get_source_video`](video_event_plugin.py:220)

**问题**：D/J 的 RL 数据 `videos=[]`（主视频被替换成 image keyframe），
完全依赖 `source_video` 字段被 ms-swift 透传。一旦透传失败，4 级 fallback 全部命空，
原代码直接 `return None`，调用方 `step()` 内 `events and src_video` 为 False 后
走 fail 分支，**模型只看到 `"[Error] No valid event selection."`**，无任何 ERROR 日志。

**修复**：在所有 fallback 失败时打一次 `logger.error`（用类级 `_missing_src_warned`
sentinel 防刷屏），提示运维用 verify_fields.py 校验。

---

### 🔴 H4 · `rl/temp_videos/` 无清理逻辑，长跑会耗尽磁盘

**位置**：[`video_event_plugin.py:_crop_event`](video_event_plugin.py:144)

**问题**：原代码 `tmp_dir = "rl/temp_videos/" + strftime("%Y%m%d_%H%M%S")`，
**每秒一个子目录**。长跑 GRPO（几十 GPU·hour）会产出几万个子目录，
每个目录里几十~几百个 mp4 片段，累计可能几十~几百 GB。训练机磁盘满
→ OOM 报错难定位。

**修复**：
1. 目录粒度改为按小时：`strftime("%Y%m%d_%H")`
2. 在 `EventLocatingScheduler.__init__` 中调用 `_cleanup_stale_temp`，
   启动时清理超过 N 小时的子目录（用类级 `_cleanup_done` 防多 worker 重复）
3. 暴露两个环境变量供调整：
   - `EVENT_TEMP_CLIPS_ROOT`（默认 `rl/temp_videos`）
   - `EVENT_TEMP_CLIPS_CLEANUP_HOURS`（默认 24，设 0 关闭）

---

### 🔴 H5 · `current_video_path` 实例属性导致并发串扰

**位置**：[`video_event_plugin.py:step`](video_event_plugin.py:280)

**问题**：原代码 `self.current_video_path = src_video` 把视频路径写到实例属性，
然后几行后 `self._crop_event(self.current_video_path, ...)` 读它。但 vLLM async
+ GRPO `num_generations=8` + `steps_per_generation=12` 会让**一个 scheduler 实例
被几十~上百个 trajectory 共享**。

如果 trajectory A 走到「写 current_video_path」后被切换到 trajectory B 也走 step()，
trajectory B 会覆盖 current_video_path；切回 A 时读到的是 B 的视频
→ **A 的 tool 调用从 B 的视频里裁片段**，模型看到完全错误的内容但程序不报错。

**修复**：完全去掉 `self.current_video_path`，全程用本地变量 `src_video`，
天然 thread/coroutine-safe。判断条件 `hasattr(self, 'current_video_path')` 同步改为 `src_video`。

---

### 🟠 M6 · `max_completion_length=4096` 在 J 多轮下可能截断

**位置**：[`grpo_events.sh:74`](grpo_events.sh:74)

**问题**：J 的 system prompt 在 N=15 个事件时约 500-1500 tokens（caption × 15 + 模板）。
三轮 `<think>+<tool_call>+<answer>` 累计可能突破 4096。一旦截断 → reward=0
→ 训练信号噪声变大。

**修复**：`max_completion_length` 从 4096 提升到 8192。`max_model_len=40960` 仍有
充足余量。如果实测 token 仍偏短可继续调小，反之 J 重数据需要时也容易调大。

---

### 🟠 M7 · SFT 数据集硬编码 8 个 jsonl，缺一即崩

**位置**：[`sft/sft_events.sh`](../sft/sft_events.sh)

**问题**：原脚本硬编码 8 个 jsonl 路径（5 个 wo_tool_call + 3 个 wi_tool_call）。
如果某个数据集（如 `longvila.jsonl`，多轮 + 大视频）转换失败率 100%，
输出目录里没有该文件 → swift sft 加载到中间才 `FileNotFoundError`，
浪费几十秒~几分钟启动时间。

**修复**：用 bash 数组 `DATASET_FILES` 列出所有文件，启动前 `for f in ...; [ -s $f ]`
做存在性 + 非空校验，缺失时立刻打错误清单并退出，提示重跑 `prepare_event_data.sh`。

---

### 🟠 M9 · `source_video` 相对路径依赖 rollout 进程的 cwd

**位置**：[`video_event_plugin.py:_get_source_video`](video_event_plugin.py:261)

**问题**：D/J 写入的 `source_video` 是相对路径（如 `videos/v_xxx.mp4`），
原代码 `os.path.abspath(src)` 基于**调用进程的 cwd** 拼绝对路径。
如果 vLLM rollout 在 systemd / k8s 下启动 cwd ≠ 项目根 → 全部找不到视频。

**修复**：
1. 引入模块级常量 `PROJECT_ROOT_FOR_VIDEOS = os.environ.get("VIDEOTEMP_PROJECT_ROOT", os.getcwd())`
2. 用 `os.path.join(PROJECT_ROOT_FOR_VIDEOS, src)` 替代 `os.path.abspath(src)`
3. 首次遇到相对路径时打 WARNING（含 cwd 值），提示设置 `VIDEOTEMP_PROJECT_ROOT`

---

### 🟡 已知策略性风险（未硬修，作为运行时观察项）

#### M8 · `loss_scale=last_two_rounds` 在 J 第一轮失监督

J 的 `wi_tool_call/*.jsonl` 是多轮数据，第 1 轮 user 含 N 张 keyframe + question。
`last_two_rounds` 只对最后 2 轮 assistant 输出计 loss，等价于让 N 张 keyframe **只在
最后 2 轮承担监督信号** —— 第一轮 think 中"基于 caption + image 路由到 tool"
这一关键能力没有显式 loss。

**建议**：J 单独评估时关注以下指标，若效果不理想再调策略：
- format reward 是否偏低（暗示 think 结构没学透）
- tool_call 是否过度调用（暗示第一轮无监督，模型不自信，瞎调）

可选修法：J 单独跑 `loss_scale=all` 或写 J 专用 loss_scale plugin。

#### M10 · `FormatReward` 与 J caption 引导冲突

J 的 caption 让模型在简单 QA 上倾向跳过 `<think>` 直接 `<answer>` —— 因为 caption
已经把答案揭示了。但 [`FormatReward`](video_event_plugin.py:367) 要求严格
`<think>...</think><answer>...</answer>`，跳过 think 会扣 0.2 分。

**建议**：实测 J 的 format reward 分布，如 p50 < 0.7 则放宽为
`THINK_ANSWER_PAT = (<think>...</think>\s*)?<answer>...</answer>`，
允许 think 可选。需要 PROMPT_STYLE-aware 的 reward 实例化。

---

## 环境变量速查

| 变量 | 默认值 | 说明 | 影响范围 |
|---|---|---|---|
| `PROMPT_STYLE` | `baseline` | 决定数据目录、IMAGE_LIMIT 等 | 全部 shell 脚本 |
| `MODEL` | `sft/ckpt/test_events_${PROMPT_STYLE}` | rollout / grpo 的起点模型 | rollout/grpo |
| `VIDEOTEMP_PROJECT_ROOT` | `cwd` | 主视频相对路径兜底解析根目录 | `video_event_plugin.py` |
| `EVENT_TEMP_CLIPS_ROOT` | `rl/temp_videos` | tool 调用临时片段根目录 | `video_event_plugin.py` |
| `EVENT_TEMP_CLIPS_CLEANUP_HOURS` | `24` | 启动时清理 N 小时前的临时目录（0=关闭）| `video_event_plugin.py` |
| `EVENT_CAPTIONS` | `scripts/plan_j/event_captions.json` | J 方案 caption 文件 | `convert_annotations_j.py` |
| `MAX_PIXELS` / `MIN_PIXELS` | `501760` / `50176` | image 像素上下限（D/J 必需）| 全部 shell 脚本 |

---

## 调试与排障

### 1. 数据准备阶段：校验字段齐全
```bash
python scripts/plan_j/verify_fields.py rl/data_events_j --check_files 20
```
绿色 ✅ 通过；红色 ❌ 失败时报告会列出缺失字段 / 标签对齐错误 / 文件缺失等。

### 2. rollout 阶段：观察日志关键词
- `[_get_source_video] 拿不到主视频路径` → 看 H3，多半是字段透传问题
- `[_get_source_video] source_video 为相对路径` → 看 M9，可能 cwd 不对
- `[temp_clips] 清理过期临时目录: N 个` → H4 自动清理在工作
- `Events [0, 3] from /abs/path/to/video.mp4` → 正常裁剪

### 3. 训练阶段：监控 reward 分布
- `acc_reward` 长期 0 → 答案抽取出问题或数据集解析错
- `event_reward` 长期 0 → 选错事件或 `covering_event_ids` 字段未透传（H2）
- `format_reward` < 0.5 → 看 M10，可能要放宽 J 的格式要求

### 4. 磁盘报警
- `rl/temp_videos/` 仍持续增长 → 检查 `EVENT_TEMP_CLIPS_CLEANUP_HOURS` 是否被设为 0
- 紧急清理：`rm -rf rl/temp_videos/*` 不影响正在跑的 trajectory（裁剪是流式的）

---

## 第二轮端到端审计发现（数据 → SFT → RL → 评估）

本节记录第二轮深度审计（覆盖原始数据 → 预处理 → SFT 训练 → RL rollout
→ GRPO 训练 → 评估）发现的 11 个问题、人工核实结论与修复方案。

### 🔴 真实高危（已修复）

#### 问题 2/11：评估脚本硬编码 baseline prompt，D/J 评估完全失配
**位置**：[`eval/videotemp/videotemp.py`](../eval/videotemp/videotemp.py) +
[`eval/videotemp/videotemp-g.py`](../eval/videotemp/videotemp-g.py) +
[`eval/utils.py`](../eval/utils.py:31-41)

**问题**：`eval/utils.py` 的 `PREFIX_PROMPT` 写死 baseline 的 `get_video_clip_frame`
工具协议，所有 benchmark 评估都用它。D / J 训出的模型用 `locate_events`
+ 关键帧/caption，prompt 失配，分数虚低不可信，方案对比失效。

**修复**：[`eval/utils.py`](../eval/utils.py) 重构为 PROMPT_STYLE 分发：
- `baseline` / `B` / `C` / `E` 走原 `get_video_clip_frame` 流程（默认）
- `D` / `J` 走新 `_run_agent_event_style`：加载 `scene_metadata.json`、
  按事件抽 K=2 或 1 张关键帧、构造对应 system prompt、用 `locate_events`
  工具裁剪
- 视频未在 metadata 中时 warning + 自动退回 baseline（不静默失败）
- `EVAL_EVENT_TOOL=1` 可让 B/C/E 也走事件分支
- 新增环境变量：`EVAL_SCENE_METADATA`、`EVAL_EVENT_CAPTIONS`、`EVAL_KEYFRAME_DIR`
- 兼容历史：未设 `PROMPT_STYLE` 时行为不变；旧代码 `from utils import PREFIX_PROMPT` 仍可用

**使用**：
```bash
# baseline 评估（与历史完全一致）
python eval/videotemp/videotemp.py

# D 方案评估（关键帧）
PROMPT_STYLE=d EVAL_SCENE_METADATA=scripts/scene_metadata.json \
    python eval/videotemp/videotemp.py

# J 方案评估（关键帧 + caption）
PROMPT_STYLE=j EVAL_SCENE_METADATA=scripts/scene_metadata.json \
    EVAL_EVENT_CAPTIONS=scripts/plan_j/event_captions.json \
    python eval/videotemp/videotemp.py
```

#### 问题 3：SFT 中 D/J 的 `images` 字段在 ms-swift 内部可能被静默丢弃
**位置**：[`sft/sft_events.sh`](../sft/sft_events.sh)

**问题**：[`scripts/plan_j/verify_fields.py`](../scripts/plan_j/verify_fields.py)
只能校验 jsonl 写入正确（标签对齐、文件存在），但**无法保证** ms-swift 加载后
chat template 真的把 `images` 传给 Qwen2.5-VL 的 vision encoder。这是 D/J
训练的关键不确定性。

**修复**：新增
[`scripts/plan_j/verify_sft_template.py`](../scripts/plan_j/verify_sft_template.py)
SFT chat template 冒烟测试：
- 加载 N 条含 `images` 字段的样本
- 走 `swift.llm.get_template(...).encode(sample)` 真实渲染流水
- 校验：input_ids 中 image_token (`<|image_pad|>`) 实际出现次数 > 0
- 校验：`pixel_values` / `images` 输出存在
- 任一不通过 exit 1

`sft/sft_events.sh` 在 D/J 启动前自动跑该冒烟测试，失败立刻 exit 1，
避免训完才发现模型完全没看到关键帧。

#### 问题 4：GRPO `vllm_server_pass_dataset true` 字段透传链路不可见
**位置**：[`rl/grpo_events.sh:67`](grpo_events.sh:67) +
[`rl/video_event_plugin.py`](video_event_plugin.py)

**问题**：`--vllm_server_pass_dataset true` 时，GRPO client 把 dataset 推到
vLLM server，但 `events` / `source_video` 等非标准字段是怎么从样本透传到
`InferRequest` 的（属性？data_dict？dict-like？）—— ms-swift 黑盒；不同
版本可能行为不同；失败时 _get_source_video 走 videos[0] fallback 会静默
误用。

**修复**：双管齐下
1. **运行时字段命中追踪**：`_get_events()` / `_get_source_video()` 每种来源
   首次命中时 `logger.info`，输出形如
   `[ms-swift field probe] source_video 字段命中来源 = attr (首次记录...)`
   让运维启动 GRPO 后 5 秒内就能确认字段透传走的哪条路径
2. **启动 sanity check**：`rl/grpo_events.sh` 对 D/J 自动跑
   `verify_fields.py --check_files 5`，缺失 `source_video`/`events` 直接 exit 1

### ⚪ 经核实是子代理误判（无需修复）

#### 问题 1：rollout `infer_request.videos` 累加与消息对齐的隐患（误报）
**核实结论**：[`video_event_plugin.py:298-307`](video_event_plugin.py:298-307)
当前两个 if 条件实际一致：
- L298 写 `<video>×len(processed_paths)` 仅在 `not errors and processed_paths`
- L306 `extend(processed_paths)` 仅在 `not errors`（且 `processed_paths`
  在 errors=空 时自然非空，否则不会有这一支）

errors 非空时既不会写 `<video>` 也不会 extend，对齐保持一致。

#### 问题 8：J 第一轮 `loss_scale=last_two_rounds` 失监督（误报）
**核实结论**：J 数据多轮结构为
```
[system, user(N images + question), assistant(think+tool_call), user(N clips + success), assistant(answer)]
```
仅有 **2 个 assistant turn**，`last_two_rounds` 全部覆盖。第一轮 think+tool_call
有监督，本问题不存在。
（如果未来扩展到 ≥3 个 assistant turn，再考虑 `last_three_rounds`。）

### 🟠 中危（已记录，待用户决定）

| ID | 标题 | 位置 | 状态 |
|----|------|------|------|
| 5 | `OVERLAP_EPS=0.01s` 在短事件上易让邻近 sentence 污染 caption | [`scripts/convert_annotations.py:34`](../scripts/convert_annotations.py:34) | **观察项**：实际数据短事件比例需统计后决定是否改进 |
| 6 | `max_turns=3` 截断行为不明确（第 4 轮怎么处理） | [`rl/rollout_events.sh:62`](rollout_events.sh:62) | **观察项**：日志中统计 `</answer>` 缺失率 |
| 7 | `IMAGE_LIMIT=32`（J）/ `64`（D）未基于 scene_metadata N 分布验证 | [`rl/rollout_events.sh:27`](rollout_events.sh:27) | **观察项**：跑 `jq '.events|length' scene_metadata.json` 统计 |

### 🟡 低危（观察项）

- **问题 9**：[`scripts/convert_annotations_b/c/e.py`](../scripts/) 没有 D/J 的
  `images/source_video` 字段断言。低概率风险（混用错脚本时静默生成不一致样本）。
- **问题 10**：[`scripts/preprocess_scenes.py:391-408,460-469`](../scripts/preprocess_scenes.py:391-408)
  边界事件（帧数<3 / 检测为空）处理缺日志。

### 字段命中追踪日志怎么看

启动 GRPO 后头几条 trajectory 应该看到（典型 J 样本）：
```
[ms-swift field probe] events 字段命中来源 = attr （首次记录；后续命中相同来源不再打印）
[ms-swift field probe] source_video 字段命中来源 = attr （首次记录；后续命中相同来源不再打印）
```
若出现 `system_text_fallback` 或 `videos[0]_fallback`，说明 D/J 字段透传**失败**，
模型实际跑的是 baseline 流程。立刻停训排查。

若出现 `data_dict` 或 `dict_like`，说明 ms-swift 走的是另一种挂载方式，
功能仍正常但需要关注 ms-swift 版本兼容性。
