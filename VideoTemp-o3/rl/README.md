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
