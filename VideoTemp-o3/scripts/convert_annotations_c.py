#!/usr/bin/env python3
"""方案 C：剥离时间戳，system 只暴露事件数量。

改动点（仅一处）：
  - 重写 build_system_prompt：system prompt 不再列举每个事件的时间区间，
    仅告诉模型「视频被分成 N 个时间有序的事件，索引 0..N-1」
  - 配套：rewrite_think 不再将 "8s-25s" 改写为 "Events 1,2,3"，
    而是直接抹除时间戳描述（让模型靠视觉感知，think 中只保留语义推理）

事件元数据仍保留在样本的 events 字段中，供 RL scheduler / 评测使用。

用法：
    python scripts/convert_annotations_c.py \
        --metadata scripts/scene_metadata.json \
        --input_dir sft/data --output_dir sft/data_events_c --data_stage sft
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert_annotations as _ca  # noqa: E402
from convert_annotations import main  # noqa: E402

# ------------------------- 改动点：system 仅暴露 N -------------------------

SYSTEM_PROMPT_TEMPLATE_C = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

The video has been pre-segmented into {n} temporally ordered events, indexed 0 to {last}. Identify each event from the visual content itself.

If you need to examine specific events more closely to answer the question, you may use the following tool to retrieve the corresponding clips:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

Use the insights from the selected event clips to inform your reasoning and construct the final answer."""


def build_system_prompt(events):
    n = len(events)
    return SYSTEM_PROMPT_TEMPLATE_C.format(n=n, last=max(0, n - 1))


# rewrite_think：抹掉 think 中的具体时间戳引用（如 "8s-25s"），由模型基于视觉描述
_THINK_PAT = _ca._THINK_PAT
_RANGE_PAT = _ca._RANGE_PAT


def rewrite_think(text, events):
    """删除 think 中的具体时间戳区间描述，保留其余语义。"""
    def _strip_range(inner_match):
        # 将形如 "between 8s-25s" / "(8s-25s)" 替换为占位词，避免破坏句法
        return "the relevant segment"

    def _rewrite_block(m):
        inner = _RANGE_PAT.sub(_strip_range, m.group(2))
        return m.group(1) + inner + m.group(3)

    return _THINK_PAT.sub(_rewrite_block, text)


# ------------------------- monkey-patch -------------------------

_ca.build_system_prompt = build_system_prompt
_ca.rewrite_think = rewrite_think


if __name__ == "__main__":
    main()
