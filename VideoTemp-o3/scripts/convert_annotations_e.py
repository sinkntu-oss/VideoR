#!/usr/bin/env python3
"""方案 E：使用 Qwen2.5-VL / OpenAI 原生 tools schema。

改动点（仅一处）：
  - system prompt 移除「工具调用 + 示例」部分，仅保留 think/answer 约束 + 事件列表
  - 在样本顶层新增 `tools` 字段（OpenAI function-calling JSON schema），
    其中 event_ids 的 minimum / maximum 由当前样本的事件数自动推导

事件列表的展示与原版保持一致，便于跟 baseline 单点对照。

用法：
    python scripts/convert_annotations_e.py \
        --metadata scripts/scene_metadata.json \
        --input_dir sft/data --output_dir sft/data_events_e --data_stage sft
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert_annotations as _ca  # noqa: E402
from convert_annotations import main  # noqa: E402

# ------------------------- 改动点：system 去掉 tool 说明 + tools 字段 -------------------------

SYSTEM_PROMPT_TEMPLATE_E = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

The video has been segmented into the following events:
{event_list}"""


def build_system_prompt(events):
    event_list = "\n".join(
        f"  Event {e['event_id']}: {e['start_time']:.1f}s - {e['end_time']:.1f}s" for e in events
    )
    return SYSTEM_PROMPT_TEMPLATE_E.format(event_list=event_list)


def _build_tools_schema(events):
    n = len(events)
    return [{
        "type": "function",
        "function": {
            "name": "locate_events",
            "description": (
                "Retrieve close-up video clips for the specified pre-segmented events. "
                "Use this when a closer look at certain events would help answer the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_ids": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max(0, n - 1),
                        },
                        "minItems": 1,
                        "description": "Indices of events to retrieve clips for (0-based).",
                    }
                },
                "required": ["event_ids"],
            },
        },
    }]


def _attach_tools(out):
    if not out:
        return out
    events = out.get("events") or []
    if events:
        out["tools"] = _build_tools_schema(events)
    return out


# ------------------------- monkey-patch -------------------------

_ca.build_system_prompt = build_system_prompt
_orig_sft = _ca.convert_sft_sample
_orig_rl = _ca.convert_rl_sample
_ca.convert_sft_sample = lambda *a, **kw: _attach_tools(_orig_sft(*a, **kw))
_ca.convert_rl_sample = lambda *a, **kw: _attach_tools(_orig_rl(*a, **kw))


if __name__ == "__main__":
    main()
