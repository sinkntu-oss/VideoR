#!/usr/bin/env python3
"""方案 J：事件级 caption + 1 张代表帧。

与方案 D 的区别：
  1. 每事件 1 张关键帧（D 是 2 张）
  2. system prompt 中每事件附 1 句 caption（从 event_captions.json 注入）
  3. 其余流程（关键帧抽取 / source_video 注入 / SFT&RL 双 patch）完全复用 D

实现策略 - 三层 monkey-patch：
  - 层 1: 改 `_ca.lookup_events` → 在 events 上挂 caption 字段
  - 层 2: 改 `_ca.build_system_prompt` → 输出含 caption 的 system prompt
  - 层 3: 复用 `_cd.convert_sft_sample / convert_rl_sample` + 改 `_cd.N_KEYFRAMES_PER_EVENT=1`

caption 文件路径通过环境变量 EVENT_CAPTIONS 指定（默认 scripts/plan_j/event_captions.json）。

用法：
    # SFT
    python scripts/plan_j/convert_annotations_j.py \
        --metadata scripts/scene_metadata.json \
        --input_dir sft/data --output_dir sft/data_events_j --data_stage sft

    # RL
    python scripts/plan_j/convert_annotations_j.py \
        --metadata scripts/scene_metadata.json \
        --input_dir rl/data --output_dir rl/data_events_j --data_stage rl
"""
import json
import logging
import os
import sys

# 把上级 scripts/ 加入 sys.path 以便 import 同级的 convert_annotations(_d)
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS_DIR)

import convert_annotations as _ca           # noqa: E402
import convert_annotations_d as _cd         # noqa: E402  关键帧抽取 / source_video 注入
from convert_annotations import main        # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================
# 配置 - 每事件 1 张关键帧
# ============================================================

# 把 D 的 module-level 常量改为 1。convert_annotations_d._apply_keyframe_rewrite
# 等函数内对 N_KEYFRAMES_PER_EVENT 的引用都是运行时查 globals，会拿到新值。
_cd.N_KEYFRAMES_PER_EVENT = 1


# ============================================================
# 加载 caption metadata
# ============================================================

DEFAULT_CAPTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "event_captions.json"
)
CAPTIONS_PATH = os.environ.get("EVENT_CAPTIONS", DEFAULT_CAPTIONS_PATH)


def _load_captions(path: str):
    if not os.path.exists(path):
        logger.warning(
            f"⚠️  event_captions.json 不存在: {path}\n"
            f"   所有事件 caption 将兜底为 '(no description)'。\n"
            f"   请先运行 scripts/plan_j/generate_event_captions.py 生成。"
        )
        return {}
    with open(path) as f:
        data = json.load(f)
    n_videos = len(data)
    n_events = sum(len(v) for v in data.values())
    logger.info(f"[plan-j] 已加载 captions: {n_videos} 视频 / {n_events} 事件 ← {path}")
    return data


CAPTIONS = _load_captions(CAPTIONS_PATH)


# ============================================================
# Patch 1: lookup_events → 在 events 上挂 caption 字段
# ============================================================

_orig_lookup = _ca.lookup_events


def lookup_events_with_caption(index, video_path, project_root):
    events = _orig_lookup(index, video_path, project_root)
    if not events:
        return events
    vkey = _ca.normalize_rel_path(video_path, project_root)
    caps = CAPTIONS.get(vkey, {})
    # 不修改原始 events 引用（_orig_lookup 返回的是 meta["events"] 的直接引用），
    # 所以这里浅拷贝每个 event dict，避免污染共享的 scene_metadata
    return [
        dict(e, caption=caps.get(str(e["event_id"]), "(no description)"))
        for e in events
    ]


_ca.lookup_events = lookup_events_with_caption


# ============================================================
# Patch 2: build_system_prompt → 注入 caption + 描述代表帧
# ============================================================

SYSTEM_PROMPT_TEMPLATE_J = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

The video has been segmented into {n} temporally ordered events (indexed 0 to {last}). Each event is described by a brief summary and accompanied by ONE representative keyframe. The {n} keyframes are listed in event order: keyframe i corresponds to event i.

Events:
{event_list}

The keyframes provide visual evidence; the summaries help you quickly identify which events are relevant to the question. If you need to examine any specific event more closely (e.g., to verify visual details not captured in the summary), you may call:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

The tool will return the full video clips of the selected events for you to refine your answer."""


def build_system_prompt(events):
    """事件列表带 caption；保持与原版同样的 (events,) 单参签名。"""
    lines = []
    for e in events:
        cap = e.get("caption") or "(no description)"
        # 截断过长 caption，避免 prompt 膨胀（保守 200 字符 ≈ 40 tokens）
        if len(cap) > 200:
            cap = cap[:197].rstrip() + "..."
        lines.append(
            f"  Event {e['event_id']} ({e['start_time']:.1f}s-{e['end_time']:.1f}s): \"{cap}\""
        )
    n = len(events)
    return SYSTEM_PROMPT_TEMPLATE_J.format(
        n=n,
        last=max(0, n - 1),
        event_list="\n".join(lines),
    )


# Patch 顺序：必须晚于 import convert_annotations_d（_cd 在 import 时已经把 D 版
# build_system_prompt 写入 _ca，这里覆盖为 J 版即可）
_ca.build_system_prompt = build_system_prompt


# ============================================================
# Patch 3: 复用 D 的 convert_sft_sample / convert_rl_sample
# ============================================================
# _cd 在被 import 时已经把自己的 convert_sft_sample / convert_rl_sample
# 注入到 _ca。这里无需再做事 —— D 的转换逻辑会自动用到我们刚 patch 过的
# lookup_events（带 caption）+ build_system_prompt（J 版），形成完整链路。
#
# 不变量自检（防止 _cd 后续被改动破坏 J 的假设）：
assert _ca.lookup_events is lookup_events_with_caption, "lookup_events patch 失败"
assert _ca.build_system_prompt is build_system_prompt, "build_system_prompt patch 失败"
assert _ca.convert_sft_sample is _cd.convert_sft_sample, "D 的 SFT patch 未生效"
assert _ca.convert_rl_sample is _cd.convert_rl_sample, "D 的 RL patch 未生效"
assert _cd.N_KEYFRAMES_PER_EVENT == 1, "N_KEYFRAMES_PER_EVENT 未被改为 1"


if __name__ == "__main__":
    main()
