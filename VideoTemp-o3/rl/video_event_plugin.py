"""
基于事件定位的视频处理插件（替代原有的 video_crop_plugin.py）。

核心变化：
1. 视频预先按场景划分为事件列表，模型通过 event_id 选择要查看的事件
2. 工具调用从 get_video_clip_frame(start,end) → locate_events(event_ids=[...])
3. 奖励函数基于事件集合匹配（F1）而非时间 IoU

注册组件：
- multi_turns['event_locating_scheduler'] → EventLocatingScheduler
- orms['acc_reward']    → Accuracy_Reward
- orms['event_reward']  → Event_Reward
- orms['format_reward'] → FormatReward
- orms['tool_penalty']  → ToolPenalty
"""

import os
import re
import math
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import cv2
from decord import VideoReader
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()

# ============================================================
# 共享常量与正则
# ============================================================

FPS_MIN_FRAMES, FPS_MAX_FRAMES, FRAME_FACTOR, FPS = 4, 64, 2, 2

TOOL_CALL_PAT = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
ANSWER_PAT = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
THINK_TOOL_PAT = re.compile(r"^<think>.*?</think>\s*<tool_call>.*?</tool_call>$", re.DOTALL)
THINK_ANSWER_PAT = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
EVENT_LINE_PAT = re.compile(r'Event\s+(\d+):\s+([\d.]+)s\s*-\s*([\d.]+)s')

EVENT_SUCCESS_PROMPT = "Tool execution successful. Analyze the visual information from the provided event clips to answer the user's question."
EVENT_FAIL_PROMPT = "Tool execution failed. Please continue your analysis based on your existing knowledge and the information from the conversation so far."


def parse_event_ids(text: str, accumulate: bool = False) -> List[int]:
    """解析 locate_events 的 event_ids。accumulate=True 时累积所有 tool_call（用于奖励）。"""
    ids: List[int] = []
    matches = TOOL_CALL_PAT.findall(text)
    if not accumulate:
        matches = matches[:1]
    for content in matches:
        try:
            tc = json.loads(content.strip())
            if tc.get("name") == "locate_events":
                ids.extend(tc["arguments"].get("event_ids", []))
        except Exception:
            continue
    return ids


# ============================================================
# 事件定位调度器
# ============================================================

class EventLocatingScheduler(MultiTurnScheduler):
    """基于事件的多轮视频处理调度器"""

    def _get_video_info(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        vr = VideoReader(video_path)
        fps, total_frames = vr.get_avg_fps(), len(vr)
        h, w = vr[0].shape[:2]
        duration = total_frames / fps if fps > 0 else 0
        return fps, w, h, total_frames, duration

    def smart_nframes(self, total_frames: int, video_fps: float) -> int:
        ceil_f = lambda n, f: math.ceil(n / f) * f
        floor_f = lambda n, f: math.floor(n / f) * f
        min_fr = ceil_f(FPS_MIN_FRAMES, FRAME_FACTOR)
        max_fr = floor_f(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
        nframes = floor_f(min(max(total_frames / video_fps * FPS, min_fr), max_fr, total_frames), FRAME_FACTOR)
        if not (FRAME_FACTOR <= nframes <= total_frames):
            raise ValueError(f"nframes {nframes} out of [{FRAME_FACTOR}, {total_frames}]")
        return nframes

    def _crop_event(self, input_path: str, start_time: float, end_time: float) -> str:
        """裁剪一个事件片段，返回片段路径或 [Error] 字符串。"""
        try:
            if start_time < 0 or end_time <= start_time:
                raise ValueError(f"Invalid: start={start_time}, end={end_time}")
            fps, w, h, total_frames, duration = self._get_video_info(input_path)
            start_time, end_time = min(max(0, start_time), duration), min(end_time, duration)
            clip_dur = end_time - start_time
            if clip_dur <= 0:
                raise ValueError(f"Empty clip: {start_time}-{end_time}")

            tmp_dir = "rl/temp_videos/" + datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=tmp_dir)
            out_path = tmp.name
            tmp.close()

            max_fr = max(FRAME_FACTOR, int(round(clip_dur * fps)))
            nframes = self.smart_nframes(max_fr, fps)
            interval = max(1, max_fr // nframes)

            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), nframes / clip_dur, (w, h))
            cap = cv2.VideoCapture(input_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
            for idx in range(max_fr):
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % interval == 0:
                    out.write(frame)
            cap.release()
            out.release()
            if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
                raise RuntimeError(f"Output invalid: {out_path}")
            return out_path
        except Exception as e:
            return f"[Error] {e}"

    def _parse_events_from_system_text(self, messages: List[Dict]) -> Optional[List[Dict]]:
        """[兜底] 从 system prompt 文本中 regex 解析事件列表（baseline prompt 形态）。"""
        if not messages or messages[0].get("role") != "system":
            return None
        events = [{"event_id": int(m.group(1)),
                   "start_time": float(m.group(2)),
                   "end_time": float(m.group(3))}
                  for m in EVENT_LINE_PAT.finditer(messages[0].get("content", ""))]
        return events or None

    def _get_events(self, infer_request) -> Optional[List[Dict]]:
        """获取事件列表：优先样本元数据 → 兜底 system 文本 regex。

        让 events 数据源与 prompt 形态解耦：B/C/D 等改 prompt 的方案不需要把
        事件列表硬写进 system，只要样本顶层保留 `events` 字段即可。

        访问优先级（兼容不同 ms-swift 版本对额外字段的挂载方式）：
          1. infer_request.events          （属性）
          2. infer_request.data_dict       （dict 字段）
          3. infer_request[...]            （dict-like）
          4. system 文本 regex             （baseline 兜底，行为不变）
        """
        raw = getattr(infer_request, 'events', None)
        if not raw:
            dd = getattr(infer_request, 'data_dict', None)
            if isinstance(dd, dict):
                raw = dd.get('events')
        if not raw:
            try:
                raw = infer_request['events']  # type: ignore[index]
            except (TypeError, KeyError, AttributeError):
                pass
        if not raw:
            return self._parse_events_from_system_text(getattr(infer_request, 'messages', None))

        try:
            events = [{"event_id": int(e["event_id"]),
                       "start_time": float(e["start_time"]),
                       "end_time": float(e["end_time"])} for e in raw]
        except (TypeError, KeyError, ValueError) as ex:
            logger.warning(f"[_get_events] malformed events field, fallback to system text: {ex}")
            return self._parse_events_from_system_text(getattr(infer_request, 'messages', None))
        return events or None

    def _get_source_video(self, infer_request) -> Optional[str]:
        """获取主视频路径：优先样本元数据 source_video → 兜底 videos[0]。

        方案 D 取消了主视频（videos 字段仅留 tool 调用产物甚至为空），故需要从样本
        元数据 `source_video` 字段拿到原视频路径来做 tool 调用裁剪。其他方案下
        videos[0] 即主视频，自动走兜底分支，行为不变。

        访问优先级（与 _get_events 对齐）：
          1. infer_request.source_video    （属性）
          2. infer_request.data_dict       （dict 字段）
          3. infer_request[...]            （dict-like）
          4. infer_request.videos[0]       （baseline 兜底）

        相对路径会基于 cwd 拼成绝对路径（与 ms-swift 加载 videos 字段的行为一致）。
        """
        src = getattr(infer_request, 'source_video', None)
        if not src:
            dd = getattr(infer_request, 'data_dict', None)
            if isinstance(dd, dict):
                src = dd.get('source_video')
        if not src:
            try:
                src = infer_request['source_video']  # type: ignore[index]
            except (TypeError, KeyError, AttributeError):
                pass
        if not src:
            vids = getattr(infer_request, 'videos', None) or []
            return vids[0] if vids else None
        return src if os.path.isabs(src) else os.path.abspath(src)

    def check_finished(self, infer_request, response_choice, current_turn) -> bool:
        if ANSWER_PAT.search(response_choice.message.content):
            return True
        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request, response_choice, current_turn) -> Dict:
        try:
            completion = response_choice.message.content
            src_video = self._get_source_video(infer_request)
            if src_video:
                self.current_video_path = src_video

            events = self._get_events(infer_request)
            selected_ids = parse_event_ids(completion)  # 取首个 tool_call
            processed_paths, errors = [], []

            if selected_ids and events and hasattr(self, 'current_video_path'):
                valid = {e["event_id"]: e for e in events}
                chosen = [valid[i] for i in selected_ids if i in valid][:5]
                if not chosen:
                    errors.append("[Error] No valid event IDs.")
                else:
                    logger.info(f"Events {[e['event_id'] for e in chosen]} from {self.current_video_path}")
                    for ev in chosen:
                        result = self._crop_event(self.current_video_path, ev["start_time"], ev["end_time"])
                        (processed_paths if os.path.exists(result) else errors).append(result)

                if not errors and processed_paths:
                    next_content = "<video>\n" * len(processed_paths) + EVENT_SUCCESS_PROMPT
                else:
                    next_content = str(errors) + EVENT_FAIL_PROMPT
            else:
                next_content = "[Error] No valid event selection." + EVENT_FAIL_PROMPT

            infer_request.messages.append({'role': 'user', 'content': next_content})
            if selected_ids and not errors:
                infer_request.videos.extend(processed_paths)
        except Exception as e:
            logger.error(f"[Error] Event processing: {e}")

        return {'infer_request': infer_request, 'rollout_infos': {'videos': infer_request.videos}}


multi_turns['event_locating_scheduler'] = EventLocatingScheduler


# ============================================================
# 奖励函数
# ============================================================

def _last_traj(global_trajectories: Dict, tid: str) -> Dict:
    return global_trajectories.get(tid, [])[-1]


class BaseEventReward(ORM):
    def _extract_answer(self, text: str) -> Optional[str]:
        if ANSWER_PAT.search(text) and not TOOL_CALL_PAT.search(text):
            return ANSWER_PAT.search(text).group(1).strip()
        return None

    def _compute_event_f1(self, selected: List[int], target: List[int]) -> float:
        """F1 衡量事件选择质量。空目标视为数据异常，统一返回 0（不奖励不作为）。"""
        if not target or not selected:
            return 0.0
        ss, ts = set(selected), set(target)
        inter = ss & ts
        p, r = len(inter) / len(ss), len(inter) / len(ts)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _timestamps_to_event_ids(self, timestamps, events):
        ids = set()
        for ts in timestamps:
            if len(ts) >= 2:
                ids.update(e["event_id"] for e in events
                           if e["start_time"] < ts[1] and e["end_time"] > ts[0])
        return sorted(ids)

    def _compute_answer_reward(self, model_answers, ref_answers, data_types, events_list=None, covering_list=None):
        def first_letter(s):
            # 缺陷6修复：剥离 "The answer is" 等前缀，避免把 'T' 当成选项
            s = (s or "").strip()
            for p in ("the best answer is", "the correct answer is", "the answer is",
                      "the best option is", "the correct option is", "answer:", "option:"):
                if s.lower().startswith(p):
                    s = s[len(p):].strip()
                    break
            return next((ch for ch in s if ch.isalpha()), "")

        rewards = []
        for i, dtype in enumerate(data_types):
            if dtype == "qa":
                m, r = first_letter(model_answers[i]), first_letter(ref_answers[i])
                rewards.append(1.0 if m and r and m == r else 0.0)
            elif dtype == "grounding":
                nums = re.findall(r"-?\d+\.?\d*", model_answers[i]) if events_list and covering_list and i < len(events_list) else []
                if nums and len(nums) % 2 == 0:
                    pred_ts = [[float(nums[j]), float(nums[j + 1])] for j in range(0, len(nums), 2)]
                    pred_ids = self._timestamps_to_event_ids(pred_ts, events_list[i])
                    rewards.append(self._compute_event_f1(pred_ids, covering_list[i]))
                else:
                    rewards.append(0.0)
            else:
                raise NotImplementedError(f"Unsupported: {dtype}")
        return rewards

    def _extract_trajectory_data(self, trajectory_ids, global_trajectories):
        model_answers, ref_answers, data_types = [], [], []
        events_list, covering_list, selected_list = [], [], []
        for tid in trajectory_ids:
            traj = _last_traj(global_trajectories, tid)
            msgs = traj.get('messages', [])
            data_types.append(traj.get('data_type', ''))
            events_list.append(traj.get('events', []))
            covering_list.append(traj.get('gt_covering_event_ids', traj.get('covering_event_ids', [])))

            sel = []
            for msg in msgs:
                if msg.get('role') == 'assistant':
                    sel.extend(parse_event_ids(msg.get('content', ''), accumulate=True))
            selected_list.append(sorted(set(sel)))

            model_answers.append(self._extract_answer(msgs[-1].get('content', '')) or "")
            ref_answers.append(traj.get('solution', ''))
        return model_answers, ref_answers, data_types, events_list, covering_list, selected_list


class Accuracy_Reward(BaseEventReward):
    """答案准确性奖励"""
    def __call__(self, completions, **kwargs):
        tids = kwargs.get('request_id', [])
        ma, ra, dt, el, cl, _ = self._extract_trajectory_data(tids, kwargs.get('trajectory_inputs', {}))
        return self._compute_answer_reward(ma, ra, dt, el, cl)

orms['acc_reward'] = Accuracy_Reward


class Event_Reward(BaseEventReward):
    """事件选择奖励：定位 F1（与答案解耦、连续、无跳变）。

    修复激励倒挂(缺陷1)与 acc 门控(缺陷2)、奖励不连续(缺陷4)：
    - 不再要求答案正确才给定位奖励，答案错也按 F1 给稠密信号（解除冷启动鸡生蛋）；
    - 去掉 f1<0.1 的 -0.1 负跳变，使"调用全错"(0) 不再低于"不调用"(0)，消除激励倒挂；
    - 选对得正分，保证 "调用选对" > "不调用" >= "调用全错"。
    多选由 F1 的 precision 自然惩罚，不再叠加 ToolPenalty(缺陷5)。"""
    def __call__(self, completions, **kwargs):
        tids = kwargs.get('request_id', [])
        _, _, _, _, cl, sl = self._extract_trajectory_data(tids, kwargs.get('trajectory_inputs', {}))
        return [self._compute_event_f1(sl[i], cl[i]) if (cl[i] and sl[i]) else 0.0
                for i in range(len(tids))]

orms['event_reward'] = Event_Reward


class FormatReward(ORM):
    """格式规范性奖励：合格 assistant 轮占比(0~1)。

    修复缺陷7：从"全有全无"改为按比例(更稠密、长轨迹不再因单轮出错全盘归零)；
    遍历 role=='assistant' 的消息而非硬编码偶数索引(消除结构偏移误判)。
    每轮须为 <think>+<tool_call>(且 tool_call 合法) 或 <think>+<answer>。"""
    def __call__(self, completions, **kwargs):
        tids = kwargs.get('request_id', [])
        gt = kwargs.get('trajectory_inputs', {})
        rewards = []
        for tid in tids:
            msgs = _last_traj(gt, tid).get('messages', [])
            responses = [m.get('content', '') for m in msgs if m.get('role') == 'assistant']
            if not responses:
                rewards.append(0.0)
                continue
            ok = 0
            for resp in responses:
                is_tool = bool(THINK_TOOL_PAT.fullmatch(resp))
                if not (is_tool or THINK_ANSWER_PAT.fullmatch(resp)):
                    continue
                if is_tool:
                    try:
                        tc = json.loads(TOOL_CALL_PAT.search(resp).group(1).strip())
                        assert tc['name'] == 'locate_events'
                        assert isinstance(tc['arguments']['event_ids'], list)
                    except Exception:
                        continue
                ok += 1
            rewards.append(ok / len(responses))
        return rewards

orms['format_reward'] = FormatReward


class ToolPenalty(ORM):
    """工具使用惩罚：仅惩罚重复多次调用(-0.1/次, 下限 -0.5)。

    修复缺陷5：移除"过度多选"惩罚——多选已由 Event_Reward 的 F1 precision 惩罚，
    避免双重施压；同时消除原先依赖 covering_event_ids 而对 grounding(用
    gt_covering_event_ids)失效、口径不一致的问题。"""
    def __call__(self, completions, **kwargs):
        tids = kwargs.get('request_id', [])
        gt = kwargs.get('trajectory_inputs', {})
        rewards = []
        for tid in tids:
            msgs = _last_traj(gt, tid).get('messages', [])
            tc_count = sum(len(TOOL_CALL_PAT.findall(m.get('content', '')))
                           for m in msgs if m.get('role') == 'assistant')
            penalty = -0.1 * (tc_count - 1) if tc_count > 1 else 0.0
            rewards.append(max(penalty, -0.5))
        return rewards

orms['tool_penalty'] = ToolPenalty
