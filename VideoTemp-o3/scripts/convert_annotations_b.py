#!/usr/bin/env python3
"""方案 B：事件列表从 system 移到 user 第一轮。

改动点（仅一处）：
  - system prompt 改为「通用模板」（不含事件列表，KV cache 友好）
  - 事件信息以单行紧凑格式 prepend 到第一个含 <video> 的 user 消息

用法：与 convert_annotations.py 完全一致，例如
    python scripts/convert_annotations_b.py \
        --metadata scripts/scene_metadata.json \
        --input_dir sft/data --output_dir sft/data_events_b --data_stage sft
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert_annotations as _ca  # noqa: E402
from convert_annotations import main  # noqa: E402

# ------------------------- 改动点：通用 system + user 端 header -------------------------

SYSTEM_PROMPT_GENERIC = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

If you need to examine specific pre-segmented events of the video more closely, you may use the following tool to retrieve the corresponding clips:

<tool_call>{"name":"locate_events","arguments":{"event_ids":[event_id_1, event_id_2, ...]}}</tool_call>

Use the insights from the selected event clips to inform your reasoning and construct the final answer."""


def _build_user_header(events):
    parts = [f"[{e['event_id']}]{e['start_time']:.1f}-{e['end_time']:.1f}s" for e in events]
    return f"Segments ({len(events)}): " + " ".join(parts)


def _relocate_event_info(out):
    """system 改为通用模板；事件列表压成单行 prepend 到首个含 <video> 的 user。"""
    if not out:
        return out
    msgs = out.get("messages", [])
    events = out.get("events")
    if not msgs or not events:
        return out
    if msgs[0].get("role") == "system":
        msgs[0]["content"] = SYSTEM_PROMPT_GENERIC
    header = _build_user_header(events)
    for m in msgs:
        if m.get("role") == "user" and "<video>" in (m.get("content") or ""):
            m["content"] = header + "\n" + m["content"]
            break
    return out


# ------------------------- monkey-patch -------------------------

_orig_sft = _ca.convert_sft_sample
_orig_rl = _ca.convert_rl_sample
_ca.convert_sft_sample = lambda *a, **kw: _relocate_event_info(_orig_sft(*a, **kw))
_ca.convert_rl_sample = lambda *a, **kw: _relocate_event_info(_orig_rl(*a, **kw))


if __name__ == "__main__":
    main()
