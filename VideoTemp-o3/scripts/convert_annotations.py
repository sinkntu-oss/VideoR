#!/usr/bin/env python3
"""
标注转换脚本：将原有时间戳标注转换为基于事件的标注。

核心逻辑：
- 对每个样本的 timestamp，找到覆盖该时间范围的最小事件集合
- 将 tool_call 中的 start_time/end_time 替换为 event_ids
- 更新系统提示词和对话格式

输入：
- 原始 JSONL 数据文件
- 场景元数据文件 (scene_metadata.json)

输出：
- 转换后的 JSONL 数据文件（新目录）

使用方法:
    cd VideoR/VideoTemp-o3
    python scripts/convert_annotations.py \
        --metadata scripts/scene_metadata.json \
        --input_dir sft/data \
        --output_dir sft/data_events \
        --data_stage sft

    python scripts/convert_annotations.py \
        --metadata scripts/scene_metadata.json \
        --input_dir rl/data \
        --output_dir rl/data_events \
        --data_stage rl
"""

import argparse
import json
import os
import re
import sys
import glob
import copy
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 新的系统提示词模板
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

The video has been segmented into the following events:
{event_list}

If you need to examine specific events more closely to answer the question, you may use the following tool to retrieve the video clips for the selected events:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

Use the insights from the selected event clips to inform your reasoning and construct the final answer."""


def format_event_list(events: List[Dict]) -> str:
    """将事件列表格式化为可读的文本描述"""
    lines = []
    for ev in events:
        lines.append(f"  Event {ev['event_id']}: {ev['start_time']:.1f}s - {ev['end_time']:.1f}s")
    return "\n".join(lines)


def build_system_prompt(events: List[Dict]) -> str:
    """基于事件列表构建系统提示词"""
    event_list_str = format_event_list(events)
    return SYSTEM_PROMPT_TEMPLATE.format(event_list=event_list_str)


# ============================================================
# 最小事件覆盖集计算
# ============================================================

def find_covering_events(
    events: List[Dict],
    target_start: float,
    target_end: float
) -> List[int]:
    """
    找到覆盖目标时间范围 [target_start, target_end] 的最小事件集合。

    由于事件是连续不重叠的区间，最小覆盖集就是所有与目标区间有重叠的事件。

    Args:
        events: 事件列表，每个事件包含 start_time, end_time, event_id
        target_start: 目标开始时间（秒）
        target_end: 目标结束时间（秒）

    Returns:
        覆盖目标区间的事件 ID 列表
    """
    covering = []
    for ev in events:
        ev_start = ev["start_time"]
        ev_end = ev["end_time"]
        # 检查重叠：事件的起始 < 目标的结束 且 事件的结束 > 目标的起始
        if ev_start < target_end and ev_end > target_start:
            covering.append(ev["event_id"])
    return covering


def find_covering_events_multi(
    events: List[Dict],
    timestamps: List[List[float]]
) -> List[int]:
    """
    对多个时间段，找到覆盖所有时间段的最小事件集合（去重并排序）。

    Args:
        events: 事件列表
        timestamps: 时间段列表 [[start1, end1], [start2, end2], ...]

    Returns:
        覆盖所有时间段的事件 ID 列表（排序去重）
    """
    all_event_ids = set()
    for ts in timestamps:
        if len(ts) >= 2:
            covering = find_covering_events(events, ts[0], ts[1])
            all_event_ids.update(covering)
    return sorted(all_event_ids)


# ============================================================
# 工具调用格式转换
# ============================================================

def convert_tool_call_in_text(text: str, events: List[Dict]) -> str:
    """
    将文本中的旧格式 tool_call 转换为新的事件定位格式。

    旧格式: <tool_call>{"name":"get_video_clip_frame","arguments":[{"start_time":X,"end_time":Y}]}</tool_call>
    新格式: <tool_call>{"name":"locate_events","arguments":{"event_ids":[1,2,3]}}</tool_call>
    """
    pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    def replace_tool_call(match):
        try:
            content = match.group(1).strip()
            tool_call = json.loads(content)
            if tool_call.get("name") == "get_video_clip_frame":
                # 提取时间戳
                timestamps = []
                for arg in tool_call.get("arguments", []):
                    start = float(arg["start_time"])
                    end = float(arg["end_time"])
                    timestamps.append([start, end])
                # 找覆盖事件
                event_ids = find_covering_events_multi(events, timestamps)
                # 构造新 tool_call
                new_call = {
                    "name": "locate_events",
                    "arguments": {"event_ids": event_ids}
                }
                return f'<tool_call>{json.dumps(new_call)}</tool_call>'
            elif tool_call.get("name") == "locate_events":
                # 已经是新格式，保持不变
                return match.group(0)
        except Exception as e:
            logger.warning(f"转换 tool_call 失败: {e}, 原文: {match.group(0)[:100]}")
        return match.group(0)

    return pattern.sub(replace_tool_call, text)


def convert_think_text_references(text: str, events: List[Dict]) -> str:
    """
    将 <think> 中对时间戳的引用转换为事件引用。
    这是一个尽力而为的转换——保留原始推理，但在末尾添加事件映射信息。
    """
    # 不做过多修改 think 内容，保持原始推理
    return text


# ============================================================
# 数据转换主逻辑
# ============================================================

def convert_sft_sample(
    sample: Dict,
    metadata: Dict,
    project_root: str
) -> Optional[Dict]:
    """
    转换一条 SFT 数据样本。

    Args:
        sample: 原始 JSONL 样本
        metadata: 场景元数据（key 为相对路径）
        project_root: 项目根目录

    Returns:
        转换后的样本，或 None（如果无法转换）
    """
    new_sample = copy.deepcopy(sample)
    messages = new_sample.get("messages", [])
    videos = new_sample.get("videos", [])

    if not videos:
        return new_sample

    # 获取主视频的事件列表（第一个非 cropped 视频）
    main_video = None
    for vp in videos:
        if "cropped_video" not in vp:
            main_video = vp
            break
    if main_video is None:
        main_video = videos[0]

    # 查找元数据
    rel_path = main_video  # JSONL 中的路径通常已经是相对路径
    video_meta = metadata.get(rel_path)

    if video_meta is None:
        # 尝试用绝对路径解析
        abs_path = os.path.join(project_root, main_video) if not os.path.isabs(main_video) else main_video
        rel_from_root = os.path.relpath(abs_path, project_root)
        video_meta = metadata.get(rel_from_root)

    if video_meta is None:
        logger.debug(f"找不到视频元数据: {main_video}")
        return None

    events = video_meta["events"]

    # 1. 替换系统提示词
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = build_system_prompt(events)

    # 2. 转换 assistant 消息中的 tool_call
    for msg in messages:
        if msg.get("role") == "assistant":
            msg["content"] = convert_tool_call_in_text(msg["content"], events)

    # 3. 转换 user 消息中的裁剪视频引用
    # wi_tool_call 数据中，第二个 user 消息会引用裁剪后的视频
    # 新流程：不再引用 cropped_video，而是引用对应事件的视频片段
    # 这里保留 <video> 标签，实际视频路径在 videos 数组中处理

    # 4. 处理 videos 数组：移除 cropped_video，
    #    改为在 sample 中记录事件选择信息
    new_videos = [v for v in videos if "cropped_video" not in v]
    new_sample["videos"] = new_videos

    # 5. 添加事件元信息
    new_sample["events"] = events

    # 6. 如果有原始 timestamp，计算对应的事件覆盖
    if "timestamp" in new_sample:
        timestamps = new_sample["timestamp"]
        if isinstance(timestamps, list) and len(timestamps) > 0:
            covering_ids = find_covering_events_multi(events, timestamps)
            new_sample["covering_event_ids"] = covering_ids

    # 7. 如果有 tool_params（wi_tool_call 数据），转换为事件 ID
    if "tool_params" in new_sample:
        tool_timestamps = new_sample["tool_params"]
        if isinstance(tool_timestamps, list):
            tool_event_ids = find_covering_events_multi(events, tool_timestamps)
            new_sample["tool_event_ids"] = tool_event_ids
        del new_sample["tool_params"]

    return new_sample


def convert_rl_sample(
    sample: Dict,
    metadata: Dict,
    project_root: str
) -> Optional[Dict]:
    """
    转换一条 RL 数据样本。

    RL 数据不含 assistant 回复（只有 system + user），
    但需要更新系统提示词并添加事件信息。
    """
    new_sample = copy.deepcopy(sample)
    messages = new_sample.get("messages", [])
    videos = new_sample.get("videos", [])

    if not videos:
        return new_sample

    main_video = videos[0]
    rel_path = main_video
    video_meta = metadata.get(rel_path)

    if video_meta is None:
        abs_path = os.path.join(project_root, main_video) if not os.path.isabs(main_video) else main_video
        rel_from_root = os.path.relpath(abs_path, project_root)
        video_meta = metadata.get(rel_from_root)

    if video_meta is None:
        logger.debug(f"找不到视频元数据: {main_video}")
        return None

    events = video_meta["events"]

    # 1. 替换系统提示词
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = build_system_prompt(events)

    # 2. 添加事件元信息
    new_sample["events"] = events

    # 3. 计算覆盖事件集
    if "timestamp" in new_sample:
        timestamps = new_sample["timestamp"]
        if isinstance(timestamps, list) and len(timestamps) > 0:
            covering_ids = find_covering_events_multi(events, timestamps)
            new_sample["covering_event_ids"] = covering_ids

    if "gt_time_stamp" in new_sample:
        gt_timestamps = new_sample["gt_time_stamp"]
        if isinstance(gt_timestamps, list) and len(gt_timestamps) > 0:
            gt_covering_ids = find_covering_events_multi(events, gt_timestamps)
            new_sample["gt_covering_event_ids"] = gt_covering_ids

    return new_sample


def process_jsonl_file(
    input_path: str,
    output_path: str,
    metadata: Dict,
    project_root: str,
    data_stage: str
) -> Tuple[int, int, int]:
    """
    处理单个 JSONL 文件。

    Returns:
        (total, success, failed) 统计
    """
    total = 0
    success = 0
    failed = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    convert_fn = convert_sft_sample if data_stage == "sft" else convert_rl_sample

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                sample = json.loads(line)
                converted = convert_fn(sample, metadata, project_root)
                if converted is not None:
                    fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"转换样本失败: {e}")
                failed += 1

    return total, success, failed


def main():
    parser = argparse.ArgumentParser(description="将时间戳标注转换为事件定位标注")
    parser.add_argument("--metadata", type=str, required=True,
                        help="场景元数据文件路径 (scene_metadata.json)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="原始数据目录 (如 sft/data 或 rl/data)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出数据目录 (如 sft/data_events 或 rl/data_events)")
    parser.add_argument("--data_stage", type=str, required=True, choices=["sft", "rl"],
                        help="数据阶段: sft 或 rl")
    parser.add_argument("--project_root", type=str, default=".",
                        help="项目根目录")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)

    # 加载元数据
    logger.info(f"加载场景元数据: {args.metadata}")
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    logger.info(f"已加载 {len(metadata)} 个视频的元数据")

    # 扫描所有 JSONL 文件
    jsonl_files = glob.glob(os.path.join(args.input_dir, "**/*.jsonl"), recursive=True)
    if not jsonl_files:
        logger.error(f"目录中没有 JSONL 文件: {args.input_dir}")
        sys.exit(1)

    logger.info(f"找到 {len(jsonl_files)} 个 JSONL 文件")

    # 逐个处理
    total_all = 0
    success_all = 0
    failed_all = 0

    for input_path in jsonl_files:
        # 保持目录结构
        rel = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel)

        logger.info(f"转换: {input_path} -> {output_path}")
        total, success, failed = process_jsonl_file(
            input_path, output_path, metadata, project_root, args.data_stage
        )
        total_all += total
        success_all += success
        failed_all += failed
        logger.info(f"  {success}/{total} 成功, {failed} 失败")

    logger.info(f"\n{'='*60}")
    logger.info(f"标注转换完成！")
    logger.info(f"  总样本: {total_all}")
    logger.info(f"  成功: {success_all}")
    logger.info(f"  失败: {failed_all}")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
