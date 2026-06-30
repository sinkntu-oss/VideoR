#!/usr/bin/env python3
"""
标注转换脚本：将时间戳标注转换为基于事件的标注。

- timestamp → 覆盖该时间范围的最小事件集合(event_ids)
- tool_call: get_video_clip_frame(start,end) → locate_events(event_ids)
- 同步重写多模态对齐：<video> 标签数 ↔ videos 数组 ↔ 选中事件数
- think 中的时间戳引用改写为事件引用
- (可选) 按事件边界裁剪片段，保证 SFT 有真实视频文件

用法:
    python scripts/convert_annotations.py --metadata scripts/scene_metadata.json \
        --input_dir sft/data --output_dir sft/data_events --data_stage sft
    python scripts/convert_annotations.py --metadata scripts/scene_metadata.json \
        --input_dir rl/data --output_dir rl/data_events --data_stage rl
"""

import argparse
import copy
import glob
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 覆盖集判定的最小正重叠长度(秒)，吸收 round 误差，避免边界相切误判
OVERLAP_EPS = 1e-2
# 片段采样参数，与 rl/video_event_plugin.py 保持一致，确保 train-inference 一致
FPS_MIN_FRAMES, FPS_MAX_FRAMES, FRAME_FACTOR, FPS = 4, 64, 2, 2

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

The video has been segmented into the following events:
{event_list}

If you need to examine specific events more closely to answer the question, you may use the following tool to retrieve the video clips for the selected events:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

Use the insights from the selected event clips to inform your reasoning and construct the final answer."""

EVENT_SUCCESS_PROMPT = "Tool execution successful. Analyze the visual information from the provided event clips to answer the user's question."

_TOOL_CALL_PAT = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
_THINK_PAT = re.compile(r'(<think>)(.*?)(</think>)', re.DOTALL)
# 仅匹配“带时间单位”的区间，避免误伤普通数字
_RANGE_PAT = re.compile(
    r'(\d+\.?\d*)\s*(?:s|sec|secs|second|seconds|秒)\s*(?:-|–|~|to|至|—)\s*'
    r'(\d+\.?\d*)\s*(?:s|sec|secs|second|seconds|秒)?', re.IGNORECASE)


def build_system_prompt(events: List[Dict]) -> str:
    event_list = "\n".join(
        f"  Event {e['event_id']}: {e['start_time']:.1f}s - {e['end_time']:.1f}s" for e in events)
    return SYSTEM_PROMPT_TEMPLATE.format(event_list=event_list)


# ============================================================
# 路径规范化 + 元数据查找 (问题 6)
# ============================================================

def normalize_rel_path(path: str, project_root: str) -> str:
    """规范化为相对 project_root 的 normpath，消除 './'、绝对/相对差异。"""
    if os.path.isabs(path):
        try:
            path = os.path.relpath(path, project_root)
        except ValueError:
            pass
    return os.path.normpath(path)


def build_metadata_index(metadata: Dict, project_root: str) -> Dict[str, Dict]:
    return {normalize_rel_path(k, project_root): v for k, v in metadata.items()}


def lookup_events(index: Dict, video_path: str, project_root: str) -> Optional[List[Dict]]:
    meta = index.get(normalize_rel_path(video_path, project_root))
    return meta["events"] if meta else None


# ============================================================
# 最小事件覆盖集 (问题 7: 基于正重叠长度 + epsilon)
# ============================================================

def find_covering_events(events: List[Dict], start: float, end: float) -> List[int]:
    return [e["event_id"] for e in events
            if min(e["end_time"], end) - max(e["start_time"], start) > OVERLAP_EPS]


def find_covering_events_multi(events: List[Dict], timestamps: List[List[float]]) -> List[int]:
    ids = set()
    for ts in timestamps:
        if len(ts) >= 2:
            ids.update(find_covering_events(events, float(ts[0]), float(ts[1])))
    return sorted(ids)


# ============================================================
# 事件片段裁剪 (问题 1/3: 保证 SFT 有真实片段，且与运行时插件一致)
# ============================================================

def _smart_nframes(total_frames: int, video_fps: float) -> int:
    ceil_f = lambda n, f: math.ceil(n / f) * f
    floor_f = lambda n, f: math.floor(n / f) * f
    min_fr = ceil_f(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_fr = floor_f(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
    nframes = floor_f(min(max(total_frames / video_fps * FPS, min_fr), max_fr, total_frames), FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes {nframes} out of [{FRAME_FACTOR}, {total_frames}]")
    return nframes


def crop_event_clip(video_abs: str, start: float, end: float, out_abs: str) -> bool:
    """按事件边界裁剪片段到 out_abs。已存在则复用。返回是否就绪。"""
    if os.path.exists(out_abs) and os.path.getsize(out_abs) >= 1024:
        return True
    try:
        import cv2
        from decord import VideoReader
    except Exception as e:
        logger.warning(f"裁剪依赖缺失(cv2/decord)，跳过: {e}")
        return False
    try:
        if not os.path.exists(video_abs):
            logger.warning(f"主视频不存在，无法裁剪: {video_abs}")
            return False
        vr = VideoReader(video_abs)
        fps, total_frames = vr.get_avg_fps(), len(vr)
        h, w = vr[0].shape[:2]
        duration = total_frames / fps if fps > 0 else 0
        start, end = min(max(0.0, start), duration), min(end, duration)
        clip_dur = end - start
        if clip_dur <= 0:
            return False

        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        max_fr = max(FRAME_FACTOR, int(round(clip_dur * fps)))
        nframes = _smart_nframes(max_fr, fps)
        interval = max(1, max_fr // nframes)

        out = cv2.VideoWriter(out_abs, cv2.VideoWriter_fourcc(*"mp4v"), nframes / clip_dur, (w, h))
        cap = cv2.VideoCapture(video_abs)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        for idx in range(max_fr):
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                out.write(frame)
        cap.release()
        out.release()
        if not os.path.exists(out_abs) or os.path.getsize(out_abs) < 1024:
            logger.warning(f"裁剪产物无效: {out_abs}")
            return False
        return True
    except Exception as e:
        logger.warning(f"裁剪失败 {video_abs} [{start},{end}]: {e}")
        return False


def event_clip_rel_path(main_video_rel: str, event_id: int, clip_dir: str) -> str:
    safe = os.path.splitext(main_video_rel)[0].replace("/", "_").replace(os.sep, "_")
    return os.path.join(clip_dir, safe, f"event_{event_id}.mp4")


# ============================================================
# 文本改写
# ============================================================

def rewrite_think(text: str, events: List[Dict]) -> str:
    """将 <think> 内带时间单位的区间(如 8s-25s)改写为事件引用 (问题 2)。"""
    def repl(m):
        try:
            a, b = sorted((float(m.group(1)), float(m.group(2))))
            ids = find_covering_events(events, a, b)
            if not ids:
                return m.group(0)
            return f"Event {ids[0]}" if len(ids) == 1 else "Events " + ", ".join(map(str, ids))
        except Exception:
            return m.group(0)
    return _THINK_PAT.sub(lambda mm: mm.group(1) + _RANGE_PAT.sub(repl, mm.group(2)) + mm.group(3), text)


def parse_clip_timestamps(text: str) -> Optional[List[List[float]]]:
    """解析旧格式 get_video_clip_frame 的时间戳列表。"""
    m = _TOOL_CALL_PAT.search(text)
    if not m:
        return None
    try:
        tc = json.loads(m.group(1).strip())
        if tc.get("name") == "get_video_clip_frame":
            return [[float(a["start_time"]), float(a["end_time"])] for a in tc.get("arguments", [])]
    except Exception:
        pass
    return None


def replace_tool_call_with_events(text: str, event_ids: List[int]) -> str:
    new_call = json.dumps({"name": "locate_events", "arguments": {"event_ids": event_ids}})
    return _TOOL_CALL_PAT.sub(f'<tool_call>{new_call}</tool_call>', text, count=1)


# ============================================================
# 样本转换
# ============================================================

def convert_sft_sample(sample, index, project_root, clip_dir, do_crop, stats) -> Optional[Dict]:
    """
    转换 SFT 样本：逐个工具调用轮重写 tool_call/think，并对齐
    <video> 数 ↔ videos 数组 ↔ event_ids 数。
    """
    s = copy.deepcopy(sample)
    messages, videos = s.get("messages", []), s.get("videos", [])
    if not videos:
        return s

    main_video = next((v for v in videos if "cropped_video" not in v), videos[0])
    events = lookup_events(index, main_video, project_root)
    if events is None:
        stats["meta_miss"] += 1
        logger.warning(f"找不到视频元数据，丢弃样本: {main_video}")
        return None

    video_abs = main_video if os.path.isabs(main_video) else os.path.join(project_root, main_video)
    main_rel = normalize_rel_path(main_video, project_root)
    id2ev = {e["event_id"]: e for e in events}

    # 系统提示词
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = build_system_prompt(events)

    # 逐个工具调用轮：改写 assistant，并对齐其后 user 的 <video> 与片段
    base_videos = [v for v in videos if "cropped_video" not in v]
    clip_paths: List[str] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        timestamps = parse_clip_timestamps(msg.get("content", ""))
        if timestamps is None:
            continue
        event_ids = find_covering_events_multi(events, timestamps) or [events[0]["event_id"]]
        msg["content"] = rewrite_think(replace_tool_call_with_events(msg["content"], event_ids), events)

        if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
            messages[i + 1]["content"] = "<video>\n" * len(event_ids) + EVENT_SUCCESS_PROMPT
            for eid in event_ids:
                ev = id2ev.get(eid)
                if ev is None:
                    continue
                rel = event_clip_rel_path(main_rel, eid, clip_dir)
                clip_paths.append(rel)
                if do_crop:
                    out_abs = rel if os.path.isabs(rel) else os.path.join(project_root, rel)
                    if not crop_event_clip(video_abs, ev["start_time"], ev["end_time"], out_abs):
                        stats["crop_fail"] += 1

    s["videos"] = base_videos + clip_paths

    # 一致性校验：<video> 总数必须等于 videos 数量
    tag_count = sum(m.get("content", "").count("<video>")
                    for m in messages if isinstance(m.get("content"), str))
    if tag_count != len(s["videos"]):
        stats["align_mismatch"] += 1
        logger.warning(f"对齐不一致(已丢弃): <video>={tag_count} videos={len(s['videos'])} video={main_video}")
        return None

    s["events"] = events
    if isinstance(s.get("timestamp"), list) and s["timestamp"]:
        s["covering_event_ids"] = find_covering_events_multi(events, s["timestamp"])
        if not s["covering_event_ids"]:
            stats["empty_cover"] += 1
    s.pop("tool_params", None)  # 移除冗余字段 (问题 5)
    return s


def convert_rl_sample(sample, index, project_root, clip_dir, do_crop, stats) -> Optional[Dict]:
    """转换 RL 样本：只更新系统提示词、注入事件信息与覆盖集。"""
    s = copy.deepcopy(sample)
    messages, videos = s.get("messages", []), s.get("videos", [])
    if not videos:
        return s

    events = lookup_events(index, videos[0], project_root)
    if events is None:
        stats["meta_miss"] += 1
        logger.warning(f"找不到视频元数据，丢弃样本: {videos[0]}")
        return None

    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = build_system_prompt(events)
    s["events"] = events
    if isinstance(s.get("timestamp"), list) and s["timestamp"]:
        s["covering_event_ids"] = find_covering_events_multi(events, s["timestamp"])
        if not s["covering_event_ids"]:
            stats["empty_cover"] += 1
    if isinstance(s.get("gt_time_stamp"), list) and s["gt_time_stamp"]:
        s["gt_covering_event_ids"] = find_covering_events_multi(events, s["gt_time_stamp"])
    return s


# ============================================================
# 主流程
# ============================================================

def process_jsonl_file(input_path, output_path, index, project_root,
                       data_stage, clip_dir, do_crop, stats) -> Tuple[int, int, int]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    convert_fn = convert_sft_sample if data_stage == "sft" else convert_rl_sample
    total = success = failed = 0
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                out = convert_fn(json.loads(line), index, project_root, clip_dir, do_crop, stats)
            except Exception as e:
                logger.error(f"转换样本失败: {e}")
                out = None
            if out is not None:
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                success += 1
            else:
                failed += 1
    return total, success, failed


def main():
    parser = argparse.ArgumentParser(description="将时间戳标注转换为事件定位标注")
    parser.add_argument("--metadata", required=True, help="场景元数据文件 (scene_metadata.json)")
    parser.add_argument("--input_dir", required=True, help="原始数据目录 (如 sft/data)")
    parser.add_argument("--output_dir", required=True, help="输出数据目录 (如 sft/data_events)")
    parser.add_argument("--data_stage", required=True, choices=["sft", "rl"], help="数据阶段")
    parser.add_argument("--project_root", default=".", help="项目根目录")
    parser.add_argument("--clip_output_dir", default=None, help="事件片段输出目录 (默认 <output_dir>/event_clips)")
    parser.add_argument("--no_crop_clips", action="store_true", help="只重写标注，不实际裁剪片段")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    clip_dir = args.clip_output_dir or os.path.join(args.output_dir, "event_clips")
    do_crop = (args.data_stage == "sft") and (not args.no_crop_clips)

    with open(args.metadata) as f:
        metadata = json.load(f)
    logger.info(f"已加载 {len(metadata)} 个视频的元数据")
    index = build_metadata_index(metadata, project_root)

    jsonl_files = glob.glob(os.path.join(args.input_dir, "**/*.jsonl"), recursive=True)
    if not jsonl_files:
        logger.error(f"目录中没有 JSONL 文件: {args.input_dir}")
        sys.exit(1)
    logger.info(f"找到 {len(jsonl_files)} 个 JSONL 文件，裁剪={'是' if do_crop else '否'}")

    total_all = success_all = failed_all = 0
    stats: Dict[str, int] = defaultdict(int)
    for input_path in jsonl_files:
        output_path = os.path.join(args.output_dir, os.path.relpath(input_path, args.input_dir))
        logger.info(f"转换: {input_path} -> {output_path}")
        t, s, f = process_jsonl_file(input_path, output_path, index, project_root,
                                     args.data_stage, clip_dir, do_crop, stats)
        total_all, success_all, failed_all = total_all + t, success_all + s, failed_all + f
        logger.info(f"  {s}/{t} 成功, {f} 失败")

    logger.info("=" * 60)
    logger.info(f"完成: 总 {total_all} / 成功 {success_all} / 失败 {failed_all} -> {args.output_dir}")
    logger.info(f"诊断: 元数据未命中={stats['meta_miss']} 对齐不一致={stats['align_mismatch']} "
                f"空覆盖集={stats['empty_cover']} 裁剪失败={stats['crop_fail']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
