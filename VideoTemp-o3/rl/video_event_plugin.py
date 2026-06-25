"""
基于事件定位的视频处理插件（替代原有的 video_crop_plugin.py）。

核心变化：
1. 视频预先按场景划分为事件列表，模型通过 event_id 选择要查看的事件
2. 工具调用从 get_video_clip_frame(start_time, end_time) → locate_events(event_ids=[...])
3. 奖励函数基于事件集合匹配（F1）而非时间 IoU

注册的组件：
- multi_turns['event_locating_scheduler'] → EventLocatingScheduler
- orms['acc_reward']      → Accuracy_Reward
- orms['event_reward']    → Event_Reward (替代原 iou_reward)
- orms['format_reward']   → FormatReward
- orms['tool_penalty']    → ToolPenalty
"""

import re
import cv2
import os
import math
import json
import tempfile
import numpy as np
from datetime import datetime
from decord import VideoReader
from typing import Tuple, Optional, Dict, List, Set
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin import ORM, orms
from swift.llm import RolloutInferRequest
from swift.llm.infer.protocol import ChatCompletionResponseChoice
from swift.utils import get_logger

logger = get_logger()

# ============================================================
# 提示词
# ============================================================

PREFIX_PROMPT = """Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

The video has been segmented into the following events:
{event_list}

If you need to examine specific events more closely to answer the question, you may use the following tool to retrieve the video clips for the selected events:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

Use the insights from the selected event clips to inform your reasoning and construct the final answer.

The question is:
"""

EVENT_SUCCESS_PROMPT = "Tool execution successful. Analyze the visual information from the provided event clips to answer the user's question."
EVENT_FAIL_PROMPT = "Tool execution failed. Please continue your analysis based on your existing knowledge and the information from the conversation so far."


# ============================================================
# 事件定位调度器
# ============================================================

class EventLocatingScheduler(MultiTurnScheduler):
    """基于事件的多轮视频处理调度器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FPS_MIN_FRAMES = 4
        self.FPS_MAX_FRAMES = 64
        self.FRAME_FACTOR = 2
        self.FPS = 2
        self.tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    def _get_video_info(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        h, w = vr[0].shape[:2]
        duration = total_frames / fps if fps > 0 else 0
        return fps, w, h, total_frames, duration

    def smart_nframes(self, total_frames: int, video_fps: float) -> int:
        ceil_f = lambda n, f: math.ceil(n / f) * f
        floor_f = lambda n, f: math.floor(n / f) * f
        min_fr = ceil_f(self.FPS_MIN_FRAMES, self.FRAME_FACTOR)
        max_fr = floor_f(min(self.FPS_MAX_FRAMES, total_frames), self.FRAME_FACTOR)
        nframes = total_frames / video_fps * self.FPS
        nframes = min(min(max(nframes, min_fr), max_fr), total_frames)
        nframes = floor_f(nframes, self.FRAME_FACTOR)
        if not (self.FRAME_FACTOR <= nframes <= total_frames):
            raise ValueError(f"nframes {nframes} out of [{self.FRAME_FACTOR}, {total_frames}]")
        return nframes

    def _crop_event(self, input_path: str, start_time: float, end_time: float) -> str:
        """裁剪一个事件片段"""
        try:
            if start_time < 0 or end_time <= start_time:
                raise ValueError(f"Invalid: start={start_time}, end={end_time}")
            fps, w, h, total_frames, duration = self._get_video_info(input_path)
            start_time = min(max(0, start_time), duration)
            end_time = min(end_time, duration)
            clip_dur = end_time - start_time
            if clip_dur <= 0:
                raise ValueError(f"Empty clip: {start_time}-{end_time}")

            tmp_dir = "rl/temp_videos/" + datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=tmp_dir)
            out_path = tmp.name
            tmp.close()

            max_fr = max(self.FRAME_FACTOR, int(round(clip_dur * fps)))
            nframes = self.smart_nframes(max_fr, fps)
            crop_fps = nframes / clip_dur
            interval = max(1, max_fr // nframes)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, crop_fps, (w, h))
            cap = cv2.VideoCapture(input_path)
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
            if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
                raise RuntimeError(f"Output invalid: {out_path}")
            return out_path
        except Exception as e:
            return f"[Error] {e}"

    def _extract_event_ids(self, text: str) -> Optional[List[int]]:
        """从模型输出中解析 event_ids"""
        m = self.tool_call_pattern.search(text)
        if not m:
            return None
        try:
            tc = json.loads(m.group(1).strip())
            if tc.get("name") == "locate_events":
                ids = tc["arguments"].get("event_ids", [])
                if isinstance(ids, list) and all(isinstance(x, int) for x in ids):
                    return ids
        except Exception:
            pass
        return None

    def _parse_events_from_system(self, messages: List[Dict]) -> Optional[List[Dict]]:
        """从 system prompt 解析事件列表"""
        if not messages or messages[0].get("role") != "system":
            return None
        pat = re.compile(r'Event\s+(\d+):\s+([\d.]+)s\s*-\s*([\d.]+)s')
        events = []
        for m in pat.finditer(messages[0].get("content", "")):
            events.append({
                "event_id": int(m.group(1)),
                "start_time": float(m.group(2)),
                "end_time": float(m.group(3))
            })
        return events if events else None

    def check_finished(self, infer_request, response_choice, current_turn) -> bool:
        completion = response_choice.message.content
        print(completion)
        if re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL):
            return True
        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request, response_choice, current_turn) -> Dict:
        try:
            completion = response_choice.message.content
            if len(infer_request.videos) > 0:
                self.current_video_path = infer_request.videos[0]

            events = self._parse_events_from_system(infer_request.messages)
            selected_ids = self._extract_event_ids(completion)

            processed_paths = []
            errors = []

            if selected_ids and events and hasattr(self, 'current_video_path'):
                valid = {e["event_id"]: e for e in events}
                chosen = [valid[i] for i in selected_ids if i in valid][:5]

                if not chosen:
                    errors.append("[Error] No valid event IDs.")
                else:
                    logger.info(f"Events {[e['event_id'] for e in chosen]} from {self.current_video_path}")
                    for ev in chosen:
                        result = self._crop_event(self.current_video_path, ev["start_time"], ev["end_time"])
                        if os.path.exists(result):
                            processed_paths.append(result)
                        else:
                            errors.append(result)

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

        return {
            'infer_request': infer_request,
            'rollout_infos': {'videos': infer_request.videos}
        }


multi_turns['event_locating_scheduler'] = EventLocatingScheduler


# ============================================================
# 奖励函数基类
# ============================================================

class BaseEventReward(ORM):
    def __init__(self):
        self.tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

    def _parse_selected_event_ids(self, text: str) -> List[int]:
        ids = []
        for content in self.tool_call_pattern.findall(text):
            try:
                tc = json.loads(content.strip())
                if tc.get("name") == "locate_events":
                    ids.extend(tc["arguments"].get("event_ids", []))
            except Exception:
                continue
        return sorted(set(ids))

    def _extract_answer(self, text: str) -> Optional[str]:
        am = self.answer_pattern.search(text)
        cm = self.tool_call_pattern.search(text)
        if am and not cm:
            return am.group(1).strip()
        return None

    def _compute_event_f1(self, selected: List[int], target: List[int]) -> float:
        """F1 分数衡量事件选择质量"""
        if not target:
            return 1.0 if not selected else 0.0
        if not selected:
            return 0.0
        ss, ts = set(selected), set(target)
        inter = ss & ts
        p = len(inter) / len(ss)
        r = len(inter) / len(ts)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _timestamps_to_event_ids(self, timestamps, events):
        ids = set()
        for ts in timestamps:
            if len(ts) >= 2:
                for ev in events:
                    if ev["start_time"] < ts[1] and ev["end_time"] > ts[0]:
                        ids.add(ev["event_id"])
        return sorted(ids)

    def _compute_answer_reward(self, model_answers, ref_answers, data_types, events_list=None, covering_list=None):
        def first_letter(s):
            for ch in (s or "").strip():
                if ch.isalpha():
                    return ch
            return ""

        rewards = []
        for i in range(len(model_answers)):
            if data_types[i] == "qa":
                m, r = first_letter(model_answers[i]), first_letter(ref_answers[i])
                rewards.append(1.0 if m and r and m == r else 0.0)
            elif data_types[i] == "grounding":
                if events_list and covering_list and i < len(events_list):
                    nums = re.findall(r"-?\d+\.?\d*", model_answers[i])
                    if len(nums) % 2 == 0 and len(nums) > 0:
                        pred_ts = [[float(nums[j]), float(nums[j+1])] for j in range(0, len(nums), 2)]
                        pred_ids = self._timestamps_to_event_ids(pred_ts, events_list[i])
                        rewards.append(self._compute_event_f1(pred_ids, covering_list[i]))
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)
            else:
                raise NotImplementedError(f"Unsupported: {data_types[i]}")
        return rewards

    def _extract_trajectory_data(self, trajectory_ids, global_trajectories):
        model_answers, ref_answers, data_types = [], [], []
        events_list, covering_list, selected_list = [], [], []

        for tid in trajectory_ids:
            traj = global_trajectories.get(tid, [])[-1]
            msgs = traj.get('messages', [])
            data_types.append(traj.get('data_type', ''))
            events_list.append(traj.get('events', []))
            gt_cov = traj.get('gt_covering_event_ids', traj.get('covering_event_ids', []))
            covering_list.append(gt_cov)

            sel = []
            for msg in msgs:
                if msg.get('role') == 'assistant':
                    sel.extend(self._parse_selected_event_ids(msg.get('content', '')))
            selected_list.append(sorted(set(sel)))

            ans = self._extract_answer(msgs[-1].get('content', ''))
            model_answers.append(ans if ans else "")
            ref_answers.append(traj.get('solution', ''))

        return model_answers, ref_answers, data_types, events_list, covering_list, selected_list


# ============================================================
# 具体奖励函数
# ============================================================

class Accuracy_Reward(BaseEventReward):
    """答案准确性奖励"""
    def __call__(self, completions, **kwargs):
        tids = kwargs.get('request_id', [])
        gt = kwargs.get('trajectory_inputs', {})
        ma, ra, dt, el, cl, _ = self._extract_trajectory_data(tids, gt)
        return self._compute_answer_reward(ma, ra, dt, el, cl)

orms['acc_reward'] = Accuracy_Reward


class Event_Reward(BaseEventReward):
    """事件选择奖励：答案正确时评估事件选择的 F1"""
    def __call__(self, completions, **kwargs):
        tids = kwargs.get('request_id', [])
        gt = kwargs.get('trajectory_inputs', {})
        ma, ra, dt, el, cl, sl = self._extract_trajectory_data(tids, gt)
        acc = self._compute_answer_reward(ma, ra, dt, el, cl)

        rewards = []
        for i in range(len(tids)):
            if acc[i] >= 0.5:
                if cl[i] and sl[i]:
                    f1 = self._compute_event_f1(sl[i], cl[i])
                    rewards.append(f1 if f1 >= 0.1 else f1 - 0.1)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        return rewards

orms['event_reward'] = Event_Reward


class FormatReward(ORM):
    """格式规范性奖励"""
    def __init__(self):
        self.think_tool = re.compile(r"^<think>.*?</think>\s*<tool_call>.*?</tool_call>$", re.DOTALL)
        self.think_answer = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
        self.tc_pat = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    def __call__(self, completions, **kwargs):
        tids = kwargs.get('request_id', [])
        gt = kwargs.get('trajectory_inputs', {})
        rewards = []
        for tid in tids:
            traj = gt.get(tid, [])[-1]
            msgs = traj.get('messages', [])
            outputs = [msgs[i].get('content', '') for i in range(2, len(msgs), 2)]
            reward = 1.0
            for resp in outputs:
                ok_tool = bool(self.think_tool.fullmatch(resp))
                ok_ans = bool(self.think_answer.fullmatch(resp))
                if not (ok_tool or ok_ans):
                    reward = 0.0
                    break
                if ok_tool:
                    try:
                        tc = json.loads(self.tc_pat.search(resp).group(1).strip())
                        assert tc['name'] == 'locate_events'
                        assert isinstance(tc['arguments']['event_ids'], list)
                    except Exception:
                        reward = 0.0
                        break
            rewards.append(reward)
        return rewards

orms['format_reward'] = FormatReward


class ToolPenalty(ORM):
    """工具使用惩罚"""
    def __init__(self):
        self.tc_pat = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    def __call__(self, completions, **kwargs):
        tids = kwargs.get('request_id', [])
        gt = kwargs.get('trajectory_inputs', {})
        rewards = []
        for tid in tids:
            traj = gt.get(tid, [])[-1]
            msgs = traj.get('messages', [])
            tc_count = 0
            total_sel = 0
            for msg in msgs:
                if msg.get('role') == 'assistant':
                    calls = self.tc_pat.findall(msg.get('content', ''))
                    tc_count += len(calls)
                    for c in calls:
                        try:
                            tc = json.loads(c.strip())
                            if tc.get("name") == "locate_events":
                                total_sel += len(tc["arguments"].get("event_ids", []))
                        except Exception:
                            pass
            penalty = 0.0
            if tc_count > 1:
                penalty -= 0.1 * (tc_count - 1)
            cov = traj.get('covering_event_ids', [])
            if cov and total_sel > len(cov) + 2:
                penalty -= 0.05 * (total_sel - len(cov) - 2)
            rewards.append(max(penalty, -0.5))
        return rewards

orms['tool_penalty'] = ToolPenalty
