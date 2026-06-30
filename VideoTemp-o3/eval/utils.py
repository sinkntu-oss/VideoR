"""
Shared utility functions for all benchmark evaluation scripts.

[问题 2/11 修复] 评估脚本通过 PROMPT_STYLE 环境变量支持 6 套方案（baseline/b/c/d/e/j）。
未设置时默认 baseline，与历史行为完全一致。

D / J 方案需要额外环境变量：
    EVAL_SCENE_METADATA   : scene_metadata.json 路径（默认 scripts/scene_metadata.json）
    EVAL_EVENT_CAPTIONS   : event_captions.json 路径（J 方案需要）
    EVAL_KEYFRAME_DIR     : 关键帧缓存目录（默认 eval/cache_keyframes）
    VIDEOTEMP_PROJECT_ROOT: 视频路径相对的项目根（如视频是绝对路径需此变量配合）

启动评估前必须保证：
  1. 评估集中所有视频在 scene_metadata.json 中有对应条目
     （否则 D/J 会退回 baseline 并打 warning，不静默失败）
  2. J 方案需要预先用 scripts/plan_j/generate_event_captions.py 生成 caption
"""
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import openai
from decord import VideoReader


# ==============================================================================
# Constants
# ==============================================================================
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 64
FRAME_FACTOR = 2
FPS = 2
MAX_ITERATIONS = 3

# ==============================================================================
# [问题 2/11] PROMPT_STYLE 分发与配置
# ==============================================================================
PROMPT_STYLE = os.environ.get("PROMPT_STYLE", "baseline").lower()
EVAL_SCENE_METADATA = os.environ.get("EVAL_SCENE_METADATA", "scripts/scene_metadata.json")
EVAL_EVENT_CAPTIONS = os.environ.get(
    "EVAL_EVENT_CAPTIONS", "scripts/plan_j/event_captions.json"
)
EVAL_KEYFRAME_DIR = os.environ.get("EVAL_KEYFRAME_DIR", "eval/cache_keyframes")
PROJECT_ROOT_FOR_VIDEOS = os.environ.get("VIDEOTEMP_PROJECT_ROOT", "")
MAX_LOCATE_EVENTS = 5  # 与 rl/video_event_plugin.py 一致

_VALID_STYLES = {"baseline", "b", "c", "d", "e", "j"}
if PROMPT_STYLE not in _VALID_STYLES:
    raise ValueError(
        f"[eval/utils] 非法 PROMPT_STYLE={PROMPT_STYLE!r}，"
        f"必须是 {sorted(_VALID_STYLES)} 之一"
    )

# ==============================================================================
# Prompts —— 每套方案一份（与训练时 system prompt 严格对齐）
# ==============================================================================

# baseline / E：原 PREFIX_PROMPT 行为。E 走原生 tools schema，
# 但评估时仍按文本工具调用解析，这里复用 baseline 模板。
PREFIX_PROMPT_BASELINE = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

If analyzing a specific video segment is necessary to answer the question, you may use the following tool to extract a clip from `[start_time]` to `[end_time]`:

<tool_call>{\"name\":\"get_video_clip_frame\",\"arguments\":[{\"start_time\":[start_time],\"end_time\":[end_time]}]}</tool_call>

Use the insights from the clip to inform your reasoning and construct the final answer."""

# 兼容历史代码：旧代码可能直接 import PREFIX_PROMPT。
PREFIX_PROMPT = PREFIX_PROMPT_BASELINE

CROP_SUCCESS_PROMPT = (
    "Tool execution successful. Analyze the visual information from the provided "
    "video clip to answer the user's question."
)
CROP_FAIL_PROMPT = (
    "Tool execution failed. Please continue your analysis based on your existing "
    "knowledge and the information from the conversation so far."
)

# ============================================================
# 事件方案专用 prompt（D/J 与 convert_annotations_*.py 对齐）
# ============================================================

# D（每事件 2 张关键帧）
SYSTEM_PROMPT_TEMPLATE_D = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

You will see {total} keyframes ({k} per event) sampled from {n} temporally ordered events (indexed 0 to {last}). The keyframes are listed in event order: keyframes [0,1] belong to event 0, [2,3] to event 1, and so on.

If a closer look at any specific event would help, you may call:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

The tool will return the full video clips of the selected events for you to refine your answer."""

# J（每事件 1 张关键帧 + caption）
SYSTEM_PROMPT_TEMPLATE_J = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

The video has been segmented into {n} temporally ordered events (indexed 0 to {last}). Each event is described by a brief summary and accompanied by ONE representative keyframe. The {n} keyframes are listed in event order: keyframe i corresponds to event i.

Events:
{event_list}

The keyframes provide visual evidence; the summaries help you quickly identify which events are relevant to the question. If you need to examine any specific event more closely (e.g., to verify visual details not captured in the summary), you may call:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

The tool will return the full video clips of the selected events for you to refine your answer."""

EVENT_SUCCESS_PROMPT = (
    "Tool execution successful. Analyze the visual information from the provided "
    "event clips to answer the user's question."
)
EVENT_FAIL_PROMPT = (
    "Tool execution failed. Please continue your analysis based on your existing "
    "knowledge and the information from the conversation so far."
)


def _is_event_style(style: str) -> bool:
    """D / J 走「事件 + 关键帧」分支；baseline/B/C/E 走时间戳分支。

    注：B/C/E 训练时虽然 prompt 不同，但工具调用名 `locate_events` 与 event_ids 协议一致。
    评估这里 baseline/B/C/E 都退回原 `get_video_clip_frame` + 时间戳协议会失配，
    所以也走事件分支。为了避免破坏现有 baseline 行为，仅 d/j 默认走事件分支；
    b/c/e 默认走 baseline，可通过 EVAL_EVENT_TOOL=1 强制改走事件分支。
    """
    if style in ("d", "j"):
        return True
    if style in ("b", "c", "e") and os.environ.get("EVAL_EVENT_TOOL", "").lower() in ("1", "true", "yes"):
        return True
    return False


# ==============================================================================
# Video Utility Functions
# ==============================================================================
def _get_video_info(video_path: str) -> Tuple[float, int, int, int, float]:
    """Get basic video information."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        frame_shape = vr[0].shape
        height, width = frame_shape[:2]
        total_duration = total_frames / fps if fps > 0 else 0

        if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0 or total_duration <= 0:
            raise ValueError(f"Invalid video metadata for {video_path}")

        return fps, width, height, total_frames, total_duration

    except Exception as e:
        raise RuntimeError(f"Error reading video file {video_path}: {e}")


def smart_nframes(total_frames: int, video_fps: int | float) -> int:
    """Calculate the number of frames for video used for model inputs."""

    def ceil_by_factor(number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
    nframes = total_frames / video_fps * FPS
    if nframes > total_frames:
        print(f"Warning: smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _crop_video(input_path: str, output_dir: str, start_time: float, end_time: float) -> str:
    """Crop a video segment with strict FPS consistency checks."""
    try:
        if start_time < 0 or end_time <= start_time:
            raise ValueError(f"Invalid timestamp: start={start_time}, end={end_time}")

        orig_fps, orig_width, orig_height, total_frames, orig_duration = _get_video_info(input_path)

        start_time = min(max(0, start_time), orig_duration)
        end_time = min(end_time, orig_duration)
        clip_duration = end_time - start_time

        custom_temp_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(custom_temp_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=custom_temp_dir)
        output_path = temp_file.name
        temp_file.close()

        max_frames = int(round(clip_duration * orig_fps))
        nframes = smart_nframes(max_frames, orig_fps)
        crop_video_fps = nframes / clip_duration
        frame_interval = max_frames // nframes

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, crop_video_fps, (orig_width, orig_height))

        cap = cv2.VideoCapture(input_path)
        pos_set_success = cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * orig_fps))

        if not pos_set_success:
            print("Warning: Seeking to start frame failed, reading frame-by-frame...")
            current_pos = 0
            target_pos = int(start_time * orig_fps)
            while current_pos < target_pos and cap.isOpened():
                ret, _ = cap.read()
                if not ret:
                    raise RuntimeError("Unable to reach the starting frame (original video too short).")
                current_pos += 1

        current_frame_in_clip = 0
        while current_frame_in_clip < max_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Reached end of video early. Expected {max_frames} frames, got {current_frame_in_clip}")
                break
            if current_frame_in_clip % frame_interval == 0:
                out.write(frame)
            current_frame_in_clip += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video processing completed. Output: {output_path}")
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file not generated: {output_path}")

        file_size = os.path.getsize(output_path)
        if file_size < 1024:
            raise RuntimeError(f"Output file too small ({file_size} bytes), no valid frame data.")

        return output_path

    except Exception as e:
        return f"Video processing error: {str(e)}"


# ==============================================================================
# [问题 2/11] Event-style helpers（D/J 评估专用）
# ==============================================================================
_SCENE_META_CACHE: Optional[Dict] = None
_CAPTIONS_CACHE: Optional[Dict] = None
_RESOURCE_LOCK = threading.Lock()


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_scene_metadata() -> Dict:
    global _SCENE_META_CACHE
    with _RESOURCE_LOCK:
        if _SCENE_META_CACHE is None:
            if not os.path.exists(EVAL_SCENE_METADATA):
                raise FileNotFoundError(
                    f"[eval/utils] PROMPT_STYLE={PROMPT_STYLE} 需要 scene_metadata.json，"
                    f"但 {EVAL_SCENE_METADATA} 不存在。请先运行 "
                    f"scripts/preprocess_scenes.py 处理评估视频，"
                    f"或设置 EVAL_SCENE_METADATA 指向已生成的文件。"
                )
            with open(EVAL_SCENE_METADATA) as f:
                raw = json.load(f)
            # 规范化 key
            _SCENE_META_CACHE = {os.path.normpath(k): v for k, v in raw.items()}
            print(f"[eval/utils] 已加载 scene_metadata: {len(_SCENE_META_CACHE)} 视频 ← {EVAL_SCENE_METADATA}")
    return _SCENE_META_CACHE


def _load_captions() -> Dict:
    global _CAPTIONS_CACHE
    with _RESOURCE_LOCK:
        if _CAPTIONS_CACHE is None:
            if not os.path.exists(EVAL_EVENT_CAPTIONS):
                print(
                    f"⚠️ [eval/utils] PROMPT_STYLE=j 但 caption 文件不存在: "
                    f"{EVAL_EVENT_CAPTIONS}；所有事件 caption 将兜底为 '(no description)'"
                )
                _CAPTIONS_CACHE = {}
            else:
                with open(EVAL_EVENT_CAPTIONS) as f:
                    data = json.load(f)
                meta = data.pop("_meta", None) if isinstance(data, dict) else None
                _CAPTIONS_CACHE = data
                expected_sha1 = (meta or {}).get("scene_metadata_sha1")
                if expected_sha1 and os.path.exists(EVAL_SCENE_METADATA):
                    actual = _file_sha1(EVAL_SCENE_METADATA)
                    if actual != expected_sha1:
                        raise RuntimeError(
                            "[eval/utils] event_captions ↔ scene_metadata 不一致！\n"
                            f"  scene_metadata.sha1 = {actual}\n"
                            f"  caption._meta.sha1   = {expected_sha1}\n"
                            "  请重新运行 scripts/plan_j/generate_event_captions.py"
                        )
                n_videos = len(_CAPTIONS_CACHE)
                n_events = sum(len(v) for v in _CAPTIONS_CACHE.values() if isinstance(v, dict))
                print(f"[eval/utils] 已加载 captions: {n_videos} 视频 / {n_events} 事件")
    return _CAPTIONS_CACHE


def _normalize_video_key(video_path: str) -> str:
    """与 scripts/convert_annotations.normalize_rel_path 对齐。"""
    p = video_path
    if os.path.isabs(p) and PROJECT_ROOT_FOR_VIDEOS:
        try:
            p = os.path.relpath(p, os.path.normpath(PROJECT_ROOT_FOR_VIDEOS))
        except ValueError:
            pass
    return os.path.normpath(p)


def _lookup_events(video_path: str) -> Optional[List[Dict]]:
    meta = _load_scene_metadata()
    key = _normalize_video_key(video_path)
    item = meta.get(key)
    if not item:
        return None
    return item.get("events", [])


def _sanitize_caption(text: Optional[str]) -> str:
    if not text:
        return "(no description)"
    cleaned = text.replace("\n", " ").replace("\r", " ").replace('"', "'").strip()
    return cleaned or "(no description)"


def _build_event_system_prompt(events: List[Dict], style: str, video_key: str) -> str:
    n = len(events)
    last = max(0, n - 1)
    if style == "d":
        k = 2
        return SYSTEM_PROMPT_TEMPLATE_D.format(n=n, k=k, total=n * k, last=last)
    elif style == "j":
        captions = _load_captions().get(video_key, {})
        lines = []
        for e in events:
            cap = _sanitize_caption(captions.get(str(e["event_id"])))
            if len(cap) > 200:
                cap = cap[:197].rstrip() + "..."
            lines.append(
                f"  Event {e['event_id']} ({e['start_time']:.1f}s-{e['end_time']:.1f}s): \"{cap}\""
            )
        return SYSTEM_PROMPT_TEMPLATE_J.format(n=n, last=last, event_list="\n".join(lines))
    raise ValueError(f"Unsupported event-style: {style}")


def _extract_event_keyframes(
    video_abs: str, events: List[Dict], n_kf: int, video_rel: str
) -> Optional[List[str]]:
    """按事件等距抽 n_kf 张关键帧，返回 N*n_kf 张绝对路径列表。已存在则跳过。"""
    safe = os.path.splitext(video_rel)[0].replace("/", "_").replace(os.sep, "_")
    out_dir = os.path.join(EVAL_KEYFRAME_DIR, safe)
    os.makedirs(out_dir, exist_ok=True)

    paths: List[str] = []
    try:
        vr = VideoReader(video_abs)
    except Exception as e:
        print(f"⚠️ 无法打开主视频抽帧: {video_abs} ({e})")
        return None
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / fps if fps > 0 else 0

    for ev in events:
        start = max(0.0, min(ev["start_time"], duration))
        end = max(start, min(ev["end_time"], duration))
        if end <= start:
            print(f"⚠️ 事件 {ev['event_id']} 时间无效，跳过抽帧 ({video_rel})")
            return None
        for i in range(n_kf):
            out_path = os.path.join(out_dir, f"event_{ev['event_id']}_kf_{i}.jpg")
            paths.append(out_path)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                continue
            ratio = (i + 1) / (n_kf + 1)
            t = start + (end - start) * ratio
            frame_idx = max(0, min(int(t * fps), total_frames - 1))
            try:
                frame_rgb = vr[frame_idx].asnumpy()
                cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"⚠️ 抽帧失败 {video_rel} event={ev['event_id']} kf={i}: {e}")
                return None
            if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
                print(f"⚠️ 抽帧产物无效: {out_path}")
                return None
    return paths


# ==============================================================================
# Agent Core
# ==============================================================================
def _run_agent_baseline(
    client: openai.Client,
    model_name: str,
    user_prompt: str,
    user_video_path: str,
    user_image_path: str = None,
    output_base_dir: str = "eval/agent_runs",
    system_prompt: str = PREFIX_PROMPT_BASELINE,
) -> Tuple[list, str]:
    """Baseline / B / C / E 风格（默认）：模型按时间戳调用 get_video_clip_frame。"""
    run_timestamp = int(time.time())
    output_dir = os.path.join(output_base_dir, f"run_{run_timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Intermediate files saved in: {os.path.abspath(output_dir)}")

    conversation_history = []
    conversation_history.append({"role": "system", "content": system_prompt})

    if user_image_path:
        user_prompt_1, user_prompt_2 = user_prompt.split("<image 1>")
        initial_content = [
            {"type": "video", "video": user_video_path},
            {"type": "text", "text": user_prompt_1},
            {"type": "image", "image": user_image_path},
            {"type": "text", "text": user_prompt_2},
        ]
    else:
        initial_content = [
            {"type": "video", "video": user_video_path},
            {"type": "text", "text": user_prompt},
        ]
    conversation_history.append({"role": "user", "content": initial_content})

    print("\n" + "=" * 20 + " Agent is running " + "=" * 20)
    print(f"🤔 Question: {user_prompt}")
    print(f"🎬 Video: {user_video_path}")

    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i + 1}/{MAX_ITERATIONS} ---")
        print("🧠 Calling model for reasoning...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                temperature=0.1,
                max_tokens=4096,
                stop=["</answer>", "<|im_end|>"],
            )
            generated_text = response.choices[0].message.content
            print(f"🤖 Model Response:\n{generated_text}")
        except Exception as e:
            print(f"❌ API call failed: {e}")
            break

        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": generated_text}],
        })

        if "</answer>" in generated_text:
            print("\n✅ Found final answer, task completed.")
            break

        timestamp_match = re.search(r"<tool_call>(.*?)</tool_call>", generated_text, re.DOTALL)
        if timestamp_match:
            try:
                tool_call = json.loads(timestamp_match.group(1).strip())
                if tool_call["name"] == "get_video_clip_frame":
                    clip_timestamps = [
                        [float(ts["start_time"]), float(ts["end_time"])]
                        for ts in tool_call["arguments"]
                    ]
                else:
                    raise NotImplementedError(f"Unsupported tool call: {tool_call}")
            except Exception as e:
                print(f"⚠️ Timestamp conversion error: {e}")
                break

            print(f"\n🐍 Found video clipping timestamps:\n---\n{clip_timestamps}\n---")

            video_save_dir = os.path.join(output_dir, f"iteration_{i + 1}_videos")
            os.makedirs(video_save_dir, exist_ok=True)

            clipped_videos = []
            error_info = []
            for start_time, end_time in clip_timestamps:
                processed_path = _crop_video(user_video_path, video_save_dir, start_time, end_time)
                if os.path.exists(processed_path):
                    clipped_videos.append(processed_path)
                else:
                    error_info.append(processed_path)

            feedback = []
            if error_info:
                feedback.extend([{"type": "text", "text": error} for error in error_info])
                feedback.append({"type": "text", "text": CROP_FAIL_PROMPT})
            else:
                print("✅ Video clipping executed successfully.")
                feedback.extend([{"type": "video", "video": path} for path in clipped_videos])
                feedback.append({"type": "text", "text": CROP_SUCCESS_PROMPT})

            conversation_history.append({"role": "user", "content": feedback})
        else:
            print("⚠️ Model did not provide answer or clipping timestamps. Terminating loop.")
            break

    print("\n" + "=" * 20 + " Agent run ended " + "=" * 20)
    final_response_text = conversation_history[-1]["content"][0]["text"]
    answer_match = re.search(r"<answer>(.*?)</answer>", final_response_text, re.DOTALL)
    final_answer = answer_match.group(1).strip() if answer_match else final_response_text

    conv_history = [
        msg["content"][0]["text"]
        for msg in conversation_history
        if msg["role"] == "assistant"
    ]

    return conv_history, final_answer


def _run_agent_event_style(
    client: openai.Client,
    model_name: str,
    user_prompt: str,
    user_video_path: str,
    user_image_path: str = None,
    output_base_dir: str = "eval/agent_runs",
    style: str = "d",
) -> Tuple[list, str]:
    """D / J 风格：每事件 K 张关键帧 + locate_events(event_ids) 工具。"""
    n_kf = 2 if style == "d" else 1

    video_key = _normalize_video_key(user_video_path)
    events = _lookup_events(user_video_path)
    if not events:
        print(
            f"⚠️ [eval/utils] {video_key} 不在 scene_metadata 中，"
            f"PROMPT_STYLE={style} 退回 baseline 评估（结果可能不可比，请补全 scene_metadata）"
        )
        return _run_agent_baseline(
            client, model_name, user_prompt, user_video_path,
            user_image_path, output_base_dir, system_prompt=PREFIX_PROMPT_BASELINE,
        )

    video_abs = (
        user_video_path
        if os.path.isabs(user_video_path)
        else (os.path.join(PROJECT_ROOT_FOR_VIDEOS, user_video_path) if PROJECT_ROOT_FOR_VIDEOS else user_video_path)
    )
    if not os.path.exists(video_abs):
        print(f"⚠️ 视频文件不存在，跳过: {video_abs}")
        return [], ""

    keyframe_paths = _extract_event_keyframes(video_abs, events, n_kf, video_key)
    if keyframe_paths is None:
        print(f"⚠️ 关键帧抽取失败 {video_key}，退回 baseline")
        return _run_agent_baseline(
            client, model_name, user_prompt, user_video_path,
            user_image_path, output_base_dir, system_prompt=PREFIX_PROMPT_BASELINE,
        )

    system_prompt = _build_event_system_prompt(events, style, video_key)
    id2ev = {e["event_id"]: e for e in events}

    run_timestamp = int(time.time())
    output_dir = os.path.join(output_base_dir, f"run_{run_timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 [{style}] Intermediate files: {os.path.abspath(output_dir)}")
    print(f"🎬 Video: {video_key} ({len(events)} events, {len(keyframe_paths)} keyframes)")

    conversation_history = [{"role": "system", "content": system_prompt}]
    initial_content: list = [{"type": "image", "image": p} for p in keyframe_paths]
    if user_image_path:
        u1, u2 = user_prompt.split("<image 1>")
        initial_content.append({"type": "text", "text": u1})
        initial_content.append({"type": "image", "image": user_image_path})
        initial_content.append({"type": "text", "text": u2})
    else:
        initial_content.append({"type": "text", "text": user_prompt})
    conversation_history.append({"role": "user", "content": initial_content})

    for i in range(MAX_ITERATIONS):
        print(f"\n--- [{style}] Iteration {i + 1}/{MAX_ITERATIONS} ---")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                temperature=0.1,
                max_tokens=4096,
                stop=["</answer>", "<|im_end|>"],
            )
            generated_text = response.choices[0].message.content
            print(f"🤖 Model Response:\n{generated_text}")
        except Exception as e:
            print(f"❌ API call failed: {e}")
            break

        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": generated_text}],
        })

        if "</answer>" in generated_text:
            print("\n✅ Found final answer, task completed.")
            break

        tc_match = re.search(r"<tool_call>(.*?)</tool_call>", generated_text, re.DOTALL)
        if not tc_match:
            print("⚠️ Model did not provide answer or tool_call. Terminating loop.")
            break

        try:
            tool_call = json.loads(tc_match.group(1).strip())
            if tool_call.get("name") != "locate_events":
                raise NotImplementedError(f"Unsupported tool: {tool_call.get('name')}")
            raw_ids = tool_call.get("arguments", {}).get("event_ids") or []
            selected_ids = [int(x) for x in raw_ids][:MAX_LOCATE_EVENTS]
        except Exception as e:
            print(f"⚠️ tool_call 解析失败: {e}")
            conversation_history.append({
                "role": "user",
                "content": [{"type": "text", "text": f"[Error] Tool call parse error: {e}\n{EVENT_FAIL_PROMPT}"}],
            })
            continue

        chosen = [id2ev[i] for i in selected_ids if i in id2ev]
        if not chosen:
            conversation_history.append({
                "role": "user",
                "content": [{"type": "text", "text": f"[Error] No valid event IDs in {selected_ids}.\n{EVENT_FAIL_PROMPT}"}],
            })
            continue

        clip_dir = os.path.join(output_dir, f"iteration_{i + 1}_clips")
        os.makedirs(clip_dir, exist_ok=True)
        clipped, errors = [], []
        for ev in chosen:
            cp = _crop_video(video_abs, clip_dir, ev["start_time"], ev["end_time"])
            (clipped if os.path.exists(cp) else errors).append(cp)

        if not errors and clipped:
            feedback = [{"type": "video", "video": p} for p in clipped]
            feedback.append({"type": "text", "text": EVENT_SUCCESS_PROMPT})
        else:
            feedback = [{"type": "text", "text": f"{errors}\n{EVENT_FAIL_PROMPT}"}]
        conversation_history.append({"role": "user", "content": feedback})

    print("\n" + "=" * 20 + " Agent run ended " + "=" * 20)
    final_response_text = conversation_history[-1]["content"][0]["text"]
    answer_match = re.search(r"<answer>(.*?)</answer>", final_response_text, re.DOTALL)
    final_answer = answer_match.group(1).strip() if answer_match else final_response_text

    conv_history = [
        msg["content"][0]["text"]
        for msg in conversation_history
        if msg["role"] == "assistant"
    ]
    return conv_history, final_answer


def run_agent_with_sandbox(
    client: openai.Client,
    model_name: str,
    user_prompt: str,
    user_video_path: str,
    user_image_path: str = None,
    output_base_dir: str = "eval/agent_runs",
) -> Tuple[list, str]:
    """主入口 —— 根据 PROMPT_STYLE 分发到对应实现。

    Args:
        client: OpenAI 客户端
        model_name: 模型名
        user_prompt: 问题（含 ``<image 1>`` 占位符时配合 user_image_path）
        user_video_path: 用户视频
        user_image_path: 可选附图（Video-MMMU 等）
        output_base_dir: 中间产物根目录
    """
    if _is_event_style(PROMPT_STYLE):
        return _run_agent_event_style(
            client, model_name, user_prompt, user_video_path,
            user_image_path, output_base_dir, style=PROMPT_STYLE,
        )
    # baseline/B/C/E（评估时统一走 baseline 工具协议）
    return _run_agent_baseline(
        client, model_name, user_prompt, user_video_path,
        user_image_path, output_base_dir, system_prompt=PREFIX_PROMPT_BASELINE,
    )


# ==============================================================================
# VLLM Client Helper
# ==============================================================================
def connect_to_vllm(base_url: str = "http://0.0.0.0:8000/v1") -> Tuple[openai.Client, str]:
    """Connect to a VLLM server and return the client and model name."""
    import sys

    print(f"🚀 Connecting to VLLM server: {base_url}")
    print(f"   PROMPT_STYLE = {PROMPT_STYLE}"
          + (f"  (event-style: keyframes + locate_events)" if _is_event_style(PROMPT_STYLE) else "  (timestamp-style: get_video_clip_frame)"))
    try:
        client = openai.Client(api_key="EMPTY", base_url=base_url)
        models = client.models.list()
        if not models.data:
            raise ValueError("Server did not return any models.")
        model_name = models.data[0].id
        print(f"✅ Connection successful! Using model: {model_name}")
        return client, model_name
    except Exception as e:
        print("\n❌ Unable to connect to VLLM server.")
        print("Please confirm:")
        print("   1. VLLM server is running at the specified address and port.")
        print("   2. Network connection is stable.")
        print(f"   Error details: {e}\n")
        sys.exit(1)
