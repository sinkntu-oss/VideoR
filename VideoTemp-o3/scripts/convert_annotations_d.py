#!/usr/bin/env python3
"""方案 D：视觉锚点 —— 每个事件用 2 张代表帧（keyframe）替代主视频。

改动点（覆盖 SFT 与 RL 两条转换路径）：
  - 调原版 convert_sft_sample / convert_rl_sample 拿到对齐过的中间结果
  - 用 2 张关键帧/事件（等距取 1/3、2/3 位置，避开边界转场）替换主视频
  - 第一轮 user 中首个 <video> 替换为 2N 个 <image>
  - 新增样本顶层 `images` 字段（2N 条 jpg 路径）
  - 新增样本顶层 `source_video` 字段（主视频相对路径，供 rollout scheduler 拿原视频做 tool 调用裁剪）
  - videos 字段只保留多轮 tool_call 产生的事件片段（高清细看时仍用 video）

system prompt 同步改为「每个事件 2 张关键帧 + 可选 locate_events 细看」。

配套训练侧改动：
  - rl/video_event_plugin.py 的 `_get_events()` / `_get_source_video()` 优先从样本元数据读取
  - sft/sft_events.sh 需补充 MAX_PIXELS / MIN_PIXELS 等 image 像素参数

用法：
    # SFT
    python scripts/convert_annotations_d.py \
        --metadata scripts/scene_metadata.json \
        --input_dir sft/data --output_dir sft/data_events_d --data_stage sft

    # RL（关键帧抽取强制开启，与 SFT 一致）
    python scripts/convert_annotations_d.py \
        --metadata scripts/scene_metadata.json \
        --input_dir rl/data --output_dir rl/data_events_d --data_stage rl
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert_annotations as _ca  # noqa: E402
from convert_annotations import main, normalize_rel_path  # noqa: E402

logger = logging.getLogger(__name__)

# ------------------------- 常量 -------------------------

N_KEYFRAMES_PER_EVENT = 2

SYSTEM_PROMPT_TEMPLATE_D = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

You will see {total} keyframes ({k} per event) sampled from {n} temporally ordered events (indexed 0 to {last}). The keyframes are listed in event order: keyframes [0,1] belong to event 0, [2,3] to event 1, and so on.

If a closer look at any specific event would help, you may call:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

The tool will return the full video clips of the selected events for you to refine your answer."""


def build_system_prompt(events):
    n = len(events)
    return SYSTEM_PROMPT_TEMPLATE_D.format(
        n=n,
        k=N_KEYFRAMES_PER_EVENT,
        total=n * N_KEYFRAMES_PER_EVENT,
        last=max(0, n - 1),
    )


_ca.build_system_prompt = build_system_prompt
_orig_sft = _ca.convert_sft_sample
_orig_rl = _ca.convert_rl_sample


# ------------------------- 工具函数 -------------------------

def _split_videos(videos, clip_dir):
    """按 _orig_sft 的顺序约定 (base_videos + clip_paths) 区分主视频与 tool 片段。"""
    clip_norm = os.path.normpath(clip_dir)
    base_videos, tool_clips = [], []
    for v in videos:
        in_clip_dir = clip_norm in os.path.normpath(v)
        is_pre_cropped = "cropped_video" in v
        (tool_clips if (in_clip_dir or is_pre_cropped) else base_videos).append(v)
    return base_videos, tool_clips


def _keyframe_rel_path(main_video_rel, event_id, kf_idx, kf_root):
    safe = os.path.splitext(main_video_rel)[0].replace("/", "_").replace(os.sep, "_")
    return os.path.join(kf_root, safe, f"event_{event_id}_kf_{kf_idx}.jpg")


def extract_event_keyframes(video_abs, start, end, n_frames, out_paths_abs):
    """从 [start, end] 等距抽 n_frames 帧，等距点 = (i+1)/(n_frames+1)。已存在则跳过。"""
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in out_paths_abs):
        return True
    try:
        import cv2
        from decord import VideoReader
    except Exception as e:
        logger.warning(f"抽帧依赖缺失(cv2/decord)，跳过: {e}")
        return False
    try:
        if not os.path.exists(video_abs):
            logger.warning(f"主视频不存在，无法抽帧: {video_abs}")
            return False
        vr = VideoReader(video_abs)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps if fps > 0 else 0
        start = max(0.0, min(start, duration))
        end = max(start, min(end, duration))
        if end <= start:
            return False

        os.makedirs(os.path.dirname(out_paths_abs[0]), exist_ok=True)
        for i, out_abs in enumerate(out_paths_abs):
            if os.path.exists(out_abs) and os.path.getsize(out_abs) > 0:
                continue
            ratio = (i + 1) / (n_frames + 1)
            t = start + (end - start) * ratio
            frame_idx = max(0, min(int(t * fps), total_frames - 1))
            frame_rgb = vr[frame_idx].asnumpy()
            cv2.imwrite(out_abs, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            if not (os.path.exists(out_abs) and os.path.getsize(out_abs) > 0):
                logger.warning(f"抽帧产物无效: {out_abs}")
                return False
        return True
    except Exception as e:
        logger.warning(f"抽帧失败 {video_abs} [{start},{end}]: {e}")
        return False


# ------------------------- 共享改造逻辑：主视频 → 2N 张关键帧 -------------------------
#
# 返回值三态：
#   "ok"   : 改造成功（out 已就地修改）
#   "skip" : 边角样本（多主视频 / 无 <video> 等），原样保留不改造
#   "drop" : 致命错误（抽帧失败 / 对齐不一致），调用方应丢弃样本
# --------------------------------------------------------------------------------

def _apply_keyframe_rewrite(out, project_root, clip_dir, do_extract, stats):
    events = out.get("events") or []
    videos = out.get("videos") or []
    if not events or not videos:
        return "skip"

    base_videos, tool_clips = _split_videos(videos, clip_dir)
    if len(base_videos) != 1:
        return "skip"  # 多主视频 / 无主视频的边角样本不处理
    main_video = base_videos[0]
    main_abs = main_video if os.path.isabs(main_video) else os.path.join(project_root, main_video)
    main_rel = normalize_rel_path(main_video, project_root)

    # 关键帧输出目录：与 clip_dir 同级
    kf_root = os.path.join(os.path.dirname(clip_dir) or ".", "event_keyframes")

    # 抽 N×K 张关键帧
    images_rel = []
    for ev in events:
        rels = [
            _keyframe_rel_path(main_rel, ev["event_id"], i, kf_root)
            for i in range(N_KEYFRAMES_PER_EVENT)
        ]
        abss = [r if os.path.isabs(r) else os.path.join(project_root, r) for r in rels]
        if do_extract:
            ok = extract_event_keyframes(main_abs, ev["start_time"], ev["end_time"],
                                         N_KEYFRAMES_PER_EVENT, abss)
            if not ok:
                stats["keyframe_fail"] += 1
                return "drop"
        images_rel.extend(rels)

    # 第一轮 user：把首个 <video> 替换为 2N 个 <image>
    msgs = out.get("messages", [])
    replaced = False
    for m in msgs:
        if m.get("role") == "user" and "<video>" in (m.get("content") or ""):
            c = m["content"]
            idx = c.find("<video>")
            tag = "<image>\n" * len(images_rel)
            after = c[idx + len("<video>"):].lstrip("\n")
            m["content"] = c[:idx] + tag + after
            replaced = True
            break
    if not replaced:
        return "skip"  # 找不到主视频 <video>，放弃改造

    # 重组字段：主视频替换为关键帧；后续 tool 调用产生的高清片段仍以 video 保留
    out["videos"] = tool_clips
    out["images"] = images_rel
    # 主视频相对路径：scheduler 通过这个找原视频做 tool 调用裁剪
    out["source_video"] = main_rel

    # 一致性校验
    img_count = sum(
        m.get("content", "").count("<image>")
        for m in msgs
        if isinstance(m.get("content"), str)
    )
    vid_count = sum(
        m.get("content", "").count("<video>")
        for m in msgs
        if isinstance(m.get("content"), str)
    )
    if img_count != len(out["images"]) or vid_count != len(out["videos"]):
        stats["align_mismatch"] += 1
        return "drop"
    return "ok"


# ------------------------- 改动点：SFT / RL 双路径 patch -------------------------

def convert_sft_sample(sample, index, project_root, clip_dir, do_crop, stats):
    out = _orig_sft(sample, index, project_root, clip_dir, do_crop, stats)
    if not out:
        return out
    # SFT 保持原 do_crop 语义：do_crop=False 时不真实抽帧（与 baseline 不裁剪 tool clip 行为一致）
    rc = _apply_keyframe_rewrite(out, project_root, clip_dir, do_extract=do_crop, stats=stats)
    return None if rc == "drop" else out


def convert_rl_sample(sample, index, project_root, clip_dir, do_crop, stats):
    out = _orig_rl(sample, index, project_root, clip_dir, do_crop, stats)
    if not out:
        return out
    # RL 强制抽帧：关键帧是 D 方案的硬输入，与 do_crop 解耦
    # （已存在的帧文件会被自动跳过，重复跑无副作用）
    rc = _apply_keyframe_rewrite(out, project_root, clip_dir, do_extract=True, stats=stats)
    return None if rc == "drop" else out


_ca.convert_sft_sample = convert_sft_sample
_ca.convert_rl_sample = convert_rl_sample


if __name__ == "__main__":
    main()
