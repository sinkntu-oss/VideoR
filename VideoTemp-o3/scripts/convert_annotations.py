#!/usr/bin/env python3
"""
标注转换脚本：将原有时间戳标注转换为基于事件的标注。

核心逻辑：
- 对每个样本的 timestamp，找到覆盖该时间范围的最小事件集合
- 将 tool_call 中的 start_time/end_time 替换为 event_ids
- 更新系统提示词和对话格式
- 同步重写多模态对齐层：<video> 标签数量 ↔ videos 数组 ↔ 选中事件数量

输入：
- 原始 JSONL 数据文件
- 场景元数据文件 (scene_metadata.json)

输出：
- 转换后的 JSONL 数据文件（新目录）
- （可选）按事件边界裁剪的视频片段（保证 SFT 训练有真实视频文件）

使用方法:
    cd VideoR/VideoTemp-o3
    python scripts/convert_annotations.py \
        --metadata scripts/scene_metadata.json \
        --input_dir sft/data \
        --output_dir sft/data_events \
        --data_stage sft \
        --clip_output_dir sft/data_events/event_clips

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
import math
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

# 覆盖集判定的最小正重叠长度（秒），用于吸收浮点/round 误差（问题 7）
OVERLAP_EPS = 1e-2

# 视频片段采样参数（与 rl/video_event_plugin.py 的 EventLocatingScheduler 保持一致，
# 确保 train-inference 一致性，问题 3）
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 64
FRAME_FACTOR = 2
FPS = 2


# ============================================================
# 新的系统提示词与多轮提示（与运行时插件保持文案一致）
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

The video has been segmented into the following events:
{event_list}

If you need to examine specific events more closely to answer the question, you may use the following tool to retrieve the video clips for the selected events:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

Use the insights from the selected event clips to inform your reasoning and construct the final answer."""

EVENT_SUCCESS_PROMPT = "Tool execution successful. Analyze the visual information from the provided event clips to answer the user's question."


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
# 路径规范化与元数据索引（问题 6）
# ============================================================

def normalize_rel_path(path: str, project_root: str) -> str:
    """
    将任意形式的视频路径规范化为相对 project_root 的 normpath，
    以消除前导 './'、绝对/相对混用、软链接等差异。
    """
    if os.path.isabs(path):
        try:
            path = os.path.relpath(path, project_root)
        except ValueError:
            pass
    return os.path.normpath(path)


def build_metadata_index(metadata: Dict, project_root: str) -> Dict[str, Dict]:
    """
    构建规范化路径 → 元数据 的索引，提升路径匹配鲁棒性。
    同时保留 basename → 元数据 作为兜底（仅当 basename 唯一时）。
    """
    norm_index: Dict[str, Dict] = {}
    basename_index: Dict[str, Optional[Dict]] = {}
    for key, meta in metadata.items():
        norm_key = normalize_rel_path(key, project_root)
        norm_index[norm_key] = meta
        base = os.path.basename(norm_key)
        if base in basename_index:
            basename_index[base] = None  # 冲突，禁用兜底
        else:
            basename_index[base] = meta
    # 合并：basename 兜底放入同一字典的特殊命名空间
    norm_index["__basename__"] = {k: v for k, v in basename_index.items() if v is not None}
    return norm_index


def lookup_metadata(norm_index: Dict, video_path: str, project_root: str) -> Optional[Dict]:
    """根据视频路径在规范化索引中查找元数据，多级兜底。"""
    norm = normalize_rel_path(video_path, project_root)
    if norm in norm_index:
        return norm_index[norm]
    # basename 兜底
    base_map = norm_index.get("__basename__", {})
    return base_map.get(os.path.basename(norm))


# ============================================================
# 最小事件覆盖集计算（问题 7：基于正重叠长度 + epsilon）
# ============================================================

def find_covering_events(
    events: List[Dict],
    target_start: float,
    target_end: float
) -> List[int]:
    """
    找到覆盖目标时间范围 [target_start, target_end] 的最小事件集合。

    由于事件是连续不重叠的区间，最小覆盖集就是所有与目标区间真正有正重叠的事件。
    采用 OVERLAP_EPS 容差，避免边界相切（前一事件的 end == 后一事件的 start）被误判。
    """
    covering = []
    for ev in events:
        ev_start = ev["start_time"]
        ev_end = ev["end_time"]
        overlap = min(ev_end, target_end) - max(ev_start, target_start)
        if overlap > OVERLAP_EPS:
            covering.append(ev["event_id"])
    # 兜底：若因极短目标区间导致无重叠，则取与目标中点所在的事件
    if not covering and events:
        mid = (target_start + target_end) / 2.0
        for ev in events:
            if ev["start_time"] <= mid < ev["end_time"]:
                covering.append(ev["event_id"])
                break
    return covering


def find_covering_events_multi(
    events: List[Dict],
    timestamps: List[List[float]]
) -> List[int]:
    """对多个时间段，找到覆盖所有时间段的最小事件集合（去重并排序）。"""
    all_event_ids = set()
    for ts in timestamps:
        if len(ts) >= 2:
            covering = find_covering_events(events, float(ts[0]), float(ts[1]))
            all_event_ids.update(covering)
    return sorted(all_event_ids)


# ============================================================
# 视频片段裁剪（问题 1/3：保证 SFT 有真实事件片段，且与运行时插件一致）
# ============================================================

def _smart_nframes(total_frames: int, video_fps: float) -> int:
    """与运行时插件一致的帧数计算。"""
    ceil_f = lambda n, f: math.ceil(n / f) * f
    floor_f = lambda n, f: math.floor(n / f) * f
    min_fr = ceil_f(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_fr = floor_f(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
    nframes = total_frames / video_fps * FPS
    nframes = min(min(max(nframes, min_fr), max_fr), total_frames)
    nframes = floor_f(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes {nframes} out of [{FRAME_FACTOR}, {total_frames}]")
    return nframes


def crop_event_clip(main_video_abs: str, start_time: float, end_time: float, out_abs: str) -> bool:
    """
    按事件时间范围从主视频裁剪片段并保存到 out_abs。
    复用与 EventLocatingScheduler._crop_event 一致的采样逻辑，保证训练-推理一致。

    Returns:
        True 表示片段文件已就绪（已存在或裁剪成功），False 表示失败。
    """
    if os.path.exists(out_abs) and os.path.getsize(out_abs) >= 1024:
        return True  # 复用已裁剪片段
    try:
        import cv2
        from decord import VideoReader
    except Exception as e:
        logger.warning(f"裁剪依赖缺失(cv2/decord)，跳过实际裁剪: {e}")
        return False

    try:
        if not os.path.exists(main_video_abs):
            logger.warning(f"主视频不存在，无法裁剪片段: {main_video_abs}")
            return False
        vr = VideoReader(main_video_abs)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        h, w = vr[0].shape[:2]
        duration = total_frames / fps if fps > 0 else 0
        start_time = min(max(0.0, start_time), duration)
        end_time = min(end_time, duration)
        clip_dur = end_time - start_time
        if clip_dur <= 0:
            return False

        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        max_fr = max(FRAME_FACTOR, int(round(clip_dur * fps)))
        nframes = _smart_nframes(max_fr, fps)
        crop_fps = nframes / clip_dur
        interval = max(1, max_fr // nframes)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_abs, fourcc, crop_fps, (w, h))
        cap = cv2.VideoCapture(main_video_abs)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
        idx = 0
        while idx < max_fr:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                out.write(frame)
            idx += 1
        cap.release()
        out.release()
        if not os.path.exists(out_abs) or os.path.getsize(out_abs) < 1024:
            logger.warning(f"裁剪产物无效: {out_abs}")
            return False
        return True
    except Exception as e:
        logger.warning(f"裁剪事件片段失败 {main_video_abs} [{start_time},{end_time}]: {e}")
        return False


def event_clip_rel_path(main_video_rel: str, event_id: int, clip_output_dir: str) -> str:
    """为某视频的某事件生成确定性的片段相对路径（同视频同事件复用同一文件）。"""
    stem = os.path.splitext(main_video_rel)[0]
    safe = stem.replace(os.sep, "_").replace("/", "_")
    return os.path.join(clip_output_dir, safe, f"event_{event_id}.mp4")


# ============================================================
# think 时间戳引用改写（问题 2）
# ============================================================

# 仅匹配“带时间单位”的时间区间，降低误伤普通数字的风险
_THINK_RANGE_PAT = re.compile(
    r'(\d+\.?\d*)\s*(?:s|sec|secs|second|seconds|秒)\s*'
    r'(?:-|–|~|to|至|до|—)\s*'
    r'(\d+\.?\d*)\s*(?:s|sec|secs|second|seconds|秒)?',
    re.IGNORECASE
)


def rewrite_think_timestamps(text: str, events: List[Dict]) -> str:
    """
    将 <think> 中“X秒 - Y秒”形式的时间戳引用替换为对应的事件引用，
    使推理链与 locate_events 动作及系统提示词语义一致（问题 2）。
    仅替换带时间单位的区间，保守处理以避免误伤。
    """
    def _repl(m: re.Match) -> str:
        try:
            a, b = float(m.group(1)), float(m.group(2))
            if b < a:
                a, b = b, a
            ids = find_covering_events(events, a, b)
            if not ids:
                return m.group(0)
            if len(ids) == 1:
                return f"Event {ids[0]}"
            return "Events " + ", ".join(str(i) for i in ids)
        except Exception:
            return m.group(0)

    def _rewrite_segment(seg: str) -> str:
        return _THINK_RANGE_PAT.sub(_repl, seg)

    # 仅改写 <think>...</think> 内部，保留标签外内容
    think_pat = re.compile(r'(<think>)(.*?)(</think>)', re.DOTALL)
    if think_pat.search(text):
        return think_pat.sub(lambda mm: mm.group(1) + _rewrite_segment(mm.group(2)) + mm.group(3), text)
    return text


# ============================================================
# 工具调用解析
# ============================================================

_TOOL_CALL_PAT = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)


def parse_clip_timestamps(text: str) -> Optional[List[List[float]]]:
    """从 assistant 文本中解析旧格式 get_video_clip_frame 的时间戳列表。"""
    m = _TOOL_CALL_PAT.search(text)
    if not m:
        return None
    try:
        tc = json.loads(m.group(1).strip())
        if tc.get("name") == "get_video_clip_frame":
            ts = []
            for arg in tc.get("arguments", []):
                ts.append([float(arg["start_time"]), float(arg["end_time"])])
            return ts
    except Exception:
        return None
    return None


def replace_tool_call_with_events(text: str, event_ids: List[int]) -> str:
    """把 assistant 文本中的旧 tool_call 替换为 locate_events(event_ids)。"""
    new_call = {"name": "locate_events", "arguments": {"event_ids": event_ids}}
    new_str = f'<tool_call>{json.dumps(new_call)}</tool_call>'
    return _TOOL_CALL_PAT.sub(new_str, text, count=1)


def count_video_tags(content: str) -> int:
    return content.count("<video>")


# ============================================================
# SFT 数据转换
# ============================================================

def convert_sft_sample(
    sample: Dict,
    norm_index: Dict,
    project_root: str,
    clip_output_dir: str,
    do_crop: bool,
    stats: Dict,
) -> Optional[Dict]:
    """
    转换一条 SFT 数据样本。

    关键修复：
    - 同步重写每个工具调用轮：assistant 的 tool_call → locate_events(event_ids)，
      其后 user 消息的 <video> 数量重写为 len(event_ids)，并在 videos 数组补回
      对应事件片段路径（可选实际裁剪）。保证 <video> ↔ videos ↔ event_ids 一致。
    - think 中的时间戳引用改写为事件引用。
    """
    new_sample = copy.deepcopy(sample)
    messages = new_sample.get("messages", [])
    videos = new_sample.get("videos", [])

    if not videos:
        return new_sample

    # 主视频：第一个非 cropped 视频
    main_video = None
    for vp in videos:
        if "cropped_video" not in vp:
            main_video = vp
            break
    if main_video is None:
        main_video = videos[0]

    video_meta = lookup_metadata(norm_index, main_video, project_root)
    if video_meta is None:
        stats["meta_miss"] = stats.get("meta_miss", 0) + 1
        logger.warning(f"找不到视频元数据，丢弃样本: {main_video}")
        return None

    events = video_meta["events"]
    main_video_abs = main_video if os.path.isabs(main_video) else os.path.join(project_root, main_video)

    # 1. 替换系统提示词
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = build_system_prompt(events)

    # 2. 遍历对话，逐个处理工具调用轮，保证多模态对齐
    base_videos = [v for v in videos if "cropped_video" not in v]
    clip_paths_in_order: List[str] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant":
            timestamps = parse_clip_timestamps(msg.get("content", ""))
            if timestamps is not None:
                event_ids = find_covering_events_multi(events, timestamps)
                if not event_ids:
                    # 极端：无覆盖事件，退化为选取首事件，避免空选
                    event_ids = [events[0]["event_id"]]
                # 2a. 改写 assistant：tool_call + think
                content = msg["content"]
                content = replace_tool_call_with_events(content, event_ids)
                content = rewrite_think_timestamps(content, events)
                msg["content"] = content

                # 2b. 改写其后的 user 消息：<video> 数量对齐 event_ids 数量
                if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                    next_msg = messages[i + 1]
                    m_count = len(event_ids)
                    next_msg["content"] = "<video>\n" * m_count + EVENT_SUCCESS_PROMPT
                    # 2c. 生成/裁剪对应事件片段，收集片段路径
                    id2ev = {e["event_id"]: e for e in events}
                    for eid in event_ids:
                        ev = id2ev.get(eid)
                        if ev is None:
                            continue
                        rel = event_clip_rel_path(
                            normalize_rel_path(main_video, project_root), eid, clip_output_dir
                        )
                        clip_paths_in_order.append(rel)
                        if do_crop:
                            out_abs = rel if os.path.isabs(rel) else os.path.join(project_root, rel)
                            ok = crop_event_clip(main_video_abs, ev["start_time"], ev["end_time"], out_abs)
                            if not ok:
                                stats["crop_fail"] = stats.get("crop_fail", 0) + 1
        i += 1

    # 3. 重建 videos：基础原始视频 + 按序的事件片段（移除 cropped_video）
    new_sample["videos"] = base_videos + clip_paths_in_order

    # 4. 一致性校验：messages 中 <video> 总数必须等于 videos 数量
    total_video_tags = sum(count_video_tags(m.get("content", "")) for m in messages
                           if isinstance(m.get("content"), str))
    if total_video_tags != len(new_sample["videos"]):
        stats["align_mismatch"] = stats.get("align_mismatch", 0) + 1
        logger.warning(
            f"多模态对齐不一致(已丢弃): <video>={total_video_tags} videos={len(new_sample['videos'])} "
            f"video={main_video}"
        )
        return None

    # 5. 事件元信息
    new_sample["events"] = events

    # 6. GT 覆盖集（基于原始 timestamp）
    if "timestamp" in new_sample:
        timestamps = new_sample["timestamp"]
        if isinstance(timestamps, list) and len(timestamps) > 0:
            covering_ids = find_covering_events_multi(events, timestamps)
            if not covering_ids:
                stats["empty_cover"] = stats.get("empty_cover", 0) + 1
            new_sample["covering_event_ids"] = covering_ids

    # 7. 移除冗余/不自洽的 tool_params（问题 5）：
    #    assistant 学到的 event_ids 已内嵌于 tool_call 文本，不再额外产出 tool_event_ids
    if "tool_params" in new_sample:
        del new_sample["tool_params"]

    return new_sample


# ============================================================
# RL 数据转换
# ============================================================

def convert_rl_sample(
    sample: Dict,
    norm_index: Dict,
    project_root: str,
    clip_output_dir: str,
    do_crop: bool,
    stats: Dict,
) -> Optional[Dict]:
    """
    转换一条 RL 数据样本。
    RL 数据不含 assistant 回复，只需更新系统提示词、注入事件信息与覆盖集。
    """
    new_sample = copy.deepcopy(sample)
    messages = new_sample.get("messages", [])
    videos = new_sample.get("videos", [])

    if not videos:
        return new_sample

    main_video = videos[0]
    video_meta = lookup_metadata(norm_index, main_video, project_root)
    if video_meta is None:
        stats["meta_miss"] = stats.get("meta_miss", 0) + 1
        logger.warning(f"找不到视频元数据，丢弃样本: {main_video}")
        return None

    events = video_meta["events"]

    # 1. 替换系统提示词
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = build_system_prompt(events)

    # 2. 事件元信息
    new_sample["events"] = events

    # 3. 覆盖集
    if "timestamp" in new_sample:
        timestamps = new_sample["timestamp"]
        if isinstance(timestamps, list) and len(timestamps) > 0:
            covering_ids = find_covering_events_multi(events, timestamps)
            if not covering_ids:
                stats["empty_cover"] = stats.get("empty_cover", 0) + 1
            new_sample["covering_event_ids"] = covering_ids

    if "gt_time_stamp" in new_sample:
        gt_timestamps = new_sample["gt_time_stamp"]
        if isinstance(gt_timestamps, list) and len(gt_timestamps) > 0:
            gt_covering_ids = find_covering_events_multi(events, gt_timestamps)
            new_sample["gt_covering_event_ids"] = gt_covering_ids

    return new_sample


# ============================================================
# 文件处理
# ============================================================

def process_jsonl_file(
    input_path: str,
    output_path: str,
    norm_index: Dict,
    project_root: str,
    data_stage: str,
    clip_output_dir: str,
    do_crop: bool,
    stats: Dict,
) -> Tuple[int, int, int]:
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
                converted = convert_fn(
                    sample, norm_index, project_root, clip_output_dir, do_crop, stats
                )
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
    parser.add_argument("--clip_output_dir", type=str, default=None,
                        help="事件片段输出目录 (默认: <output_dir>/event_clips)")
    parser.add_argument("--no_crop_clips", action="store_true",
                        help="仅重写标注与路径，不实际裁剪事件片段（可后续批量裁剪）")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    clip_output_dir = args.clip_output_dir or os.path.join(args.output_dir, "event_clips")
    do_crop = (args.data_stage == "sft") and (not args.no_crop_clips)

    # 加载元数据并建立规范化索引
    logger.info(f"加载场景元数据: {args.metadata}")
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    logger.info(f"已加载 {len(metadata)} 个视频的元数据")
    norm_index = build_metadata_index(metadata, project_root)

    jsonl_files = glob.glob(os.path.join(args.input_dir, "**/*.jsonl"), recursive=True)
    if not jsonl_files:
        logger.error(f"目录中没有 JSONL 文件: {args.input_dir}")
        sys.exit(1)
    logger.info(f"找到 {len(jsonl_files)} 个 JSONL 文件")
    if do_crop:
        logger.info(f"将裁剪事件片段到: {clip_output_dir}")
    else:
        logger.info("跳过事件片段裁剪（--no_crop_clips 或 rl 阶段）")

    total_all = success_all = failed_all = 0
    stats: Dict[str, int] = {}

    for input_path in jsonl_files:
        rel = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel)
        logger.info(f"转换: {input_path} -> {output_path}")
        total, success, failed = process_jsonl_file(
            input_path, output_path, norm_index, project_root,
            args.data_stage, clip_output_dir, do_crop, stats
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
    logger.info(f"  -- 诊断统计 --")
    logger.info(f"  元数据未命中(丢弃): {stats.get('meta_miss', 0)}")
    logger.info(f"  多模态对齐不一致(丢弃): {stats.get('align_mismatch', 0)}")
    logger.info(f"  空覆盖集样本: {stats.get('empty_cover', 0)}")
    logger.info(f"  事件片段裁剪失败: {stats.get('crop_fail', 0)}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
