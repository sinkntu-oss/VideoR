#!/usr/bin/env python3
"""方案 J - 事件级 caption 生成脚本。

两阶段离线 pipeline：
  Stage 1 (harvest): 从原始数据集 jsonl 的 sentences/timestamps 字段对齐事件 → 零 GPU
  Stage 2 (vlm)    : 对未覆盖事件用 VLM 生成 caption (推荐 Qwen2.5-VL-3B-Instruct)
  auto 模式        : 1 + 2 顺序执行

输出: event_captions.json
  { "<video_rel_path>": { "<event_id>": "...", ... } }

支持增量保存与断点续跑：每 N 条 dump 一次，重跑时自动跳过已生成的事件。

用法见 README.md。
"""
import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 共享 normalize 与覆盖判定（与 convert_annotations.py 对齐，避免 key 不一致）
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS_DIR)
from convert_annotations import normalize_rel_path, OVERLAP_EPS  # noqa: E402


# ============================================================
# 通用 I/O
# ============================================================

def load_existing(path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_captions(path: str, data: Dict[str, Dict[str, str]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def first_video(sample: Dict) -> Optional[str]:
    """从样本中找主视频路径，兼容多种字段约定。"""
    videos = sample.get("videos")
    if videos:
        for v in videos:
            if isinstance(v, str) and "cropped_video" not in v:
                return v
        if isinstance(videos[0], str):
            return videos[0]
    v = sample.get("video")
    return v if isinstance(v, str) else None


def extract_orig_caption_pairs(sample: Dict) -> Optional[Tuple[List[str], List[List[float]]]]:
    """从样本中尝试白嫖原始 caption 与时间戳，返回 (sentences, timestamps) 或 None。

    支持的常见字段名约定（按优先级）：
      1. sentences + timestamps              （ActivityNet Captions）
      2. captions  + timestamps              （部分别名）
      3. chapters  (list of {title,start,end}) （VidChapters-7M）
      4. chapter_titles + chapter_timestamps
    """
    # 1/2: 平行列表
    for sent_key in ("sentences", "captions"):
        sents = sample.get(sent_key)
        ts = sample.get("timestamps")
        if isinstance(sents, list) and isinstance(ts, list) and len(sents) == len(ts) and sents:
            try:
                ts_f = [[float(a), float(b)] for a, b in ts]
                return [str(s).strip() for s in sents], ts_f
            except (ValueError, TypeError):
                continue

    # 3: chapters list of dicts
    chapters = sample.get("chapters")
    if isinstance(chapters, list) and chapters:
        try:
            sents = [str(c.get("title") or c.get("caption") or "").strip() for c in chapters]
            ts_f = [[float(c["start"]), float(c["end"])] for c in chapters]
            if any(sents):
                return sents, ts_f
        except (KeyError, ValueError, TypeError):
            pass

    # 4: chapter_titles + chapter_timestamps
    titles = sample.get("chapter_titles")
    cts = sample.get("chapter_timestamps")
    if isinstance(titles, list) and isinstance(cts, list) and len(titles) == len(cts) and titles:
        try:
            return [str(t).strip() for t in titles], [[float(a), float(b)] for a, b in cts]
        except (ValueError, TypeError):
            pass

    return None


# ============================================================
# Stage 1: Harvest
# ============================================================

def harvest_captions(data_dirs: List[str], scene_metadata: Dict,
                     project_root: str, eps: float = OVERLAP_EPS) -> Dict[str, Dict[str, str]]:
    """扫描所有 jsonl，提取按时间戳对齐到事件的现成 caption。"""
    jsonl_files: List[str] = []
    for d in data_dirs:
        jsonl_files.extend(glob.glob(os.path.join(d, "**/*.jsonl"), recursive=True))
    logger.info(f"[harvest] 扫描 {len(jsonl_files)} 个 jsonl 文件...")

    # 每个视频可能在多份 jsonl 出现，存储所有 (sentences, timestamps) 候选
    by_video: Dict[str, List[Tuple[List[str], List[List[float]]]]] = defaultdict(list)
    n_samples = n_with_caps = 0
    for f in jsonl_files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_samples += 1
                vid = first_video(s)
                if not vid:
                    continue
                pair = extract_orig_caption_pairs(s)
                if not pair:
                    continue
                n_with_caps += 1
                vkey = normalize_rel_path(vid, project_root)
                by_video[vkey].append(pair)
    logger.info(f"[harvest] 共扫描 {n_samples} 样本，含可用 caption 的 {n_with_caps} 条，覆盖 {len(by_video)} 个视频")

    result: Dict[str, Dict[str, str]] = {}
    # 与 scene_metadata 的 key 对齐（也归一化）
    scene_norm = {normalize_rel_path(k, project_root): v for k, v in scene_metadata.items()}
    n_events_covered = n_videos_covered = 0
    for vkey, meta in scene_norm.items():
        if vkey not in by_video:
            continue
        events = meta.get("events", [])
        if not events:
            continue
        # 多份候选中选「覆盖总时长最大」的一份
        best = None
        best_cover = -1.0
        for sents, tss in by_video[vkey]:
            cover = sum(et - st for st, et in tss)
            if cover > best_cover:
                best, best_cover = (sents, tss), cover
        if not best:
            continue
        sents, tss = best
        event_caps: Dict[str, str] = {}
        for ev in events:
            overlapping = []
            for s_text, (st, et) in zip(sents, tss):
                ov = min(et, ev["end_time"]) - max(st, ev["start_time"])
                if ov > eps and s_text:
                    overlapping.append(s_text)
            if overlapping:
                # 去重保序拼接，避免重复 sentence
                seen = set()
                merged = []
                for s_text in overlapping:
                    if s_text not in seen:
                        merged.append(s_text)
                        seen.add(s_text)
                event_caps[str(ev["event_id"])] = " ".join(merged)
        if event_caps:
            result[vkey] = event_caps
            n_videos_covered += 1
            n_events_covered += len(event_caps)
    logger.info(f"[harvest] 命中 {n_videos_covered} 视频 / {n_events_covered} 事件")
    return result


# ============================================================
# Stage 2: VLM
# ============================================================

CAPTION_PROMPT = ("Describe what is visually happening in this short video clip "
                  "in one concise English sentence (max 20 words). "
                  "Focus on actions and objects you can see; do NOT speculate.")


class QwenVLCaptioner:
    """轻量封装 Qwen2.5-VL 系列模型，支持 video 输入。

    依赖：transformers >= 4.49, qwen_vl_utils, torch
    """

    def __init__(self, model_path: str, n_frames: int = 6,
                 max_new_tokens: int = 50, device_map: str = "auto"):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch
        except ImportError as e:
            raise ImportError(
                "需要安装 transformers + torch + qwen_vl_utils: "
                "pip install transformers>=4.49 torch qwen-vl-utils"
            ) from e
        self.torch = torch
        logger.info(f"[vlm] 加载 captioner {model_path} ...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device_map
        ).eval()
        self.n_frames = n_frames
        self.max_new_tokens = max_new_tokens

    def caption(self, video_path: str) -> Optional[str]:
        from qwen_vl_utils import process_vision_info
        if not os.path.exists(video_path):
            return None
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "nframes": self.n_frames,
                 "min_pixels": 50176, "max_pixels": 50176},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                    padding=True, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            trimmed = out[:, inputs.input_ids.shape[1]:]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0].strip()
            decoded = decoded.replace("\n", " ").strip()
            return decoded or None
        except Exception as e:
            logger.warning(f"[vlm] caption 生成失败 {video_path}: {e}")
            return None


def event_clip_path(video_rel_key: str, event_id: int, clips_dir: str) -> str:
    """与 convert_annotations.event_clip_rel_path 的命名对齐。"""
    safe = os.path.splitext(video_rel_key)[0].replace("/", "_").replace(os.sep, "_")
    return os.path.join(clips_dir, safe, f"event_{event_id}.mp4")


def vlm_fill_missing(scene_metadata: Dict, clips_dir: str,
                     existing: Dict[str, Dict[str, str]],
                     captioner: QwenVLCaptioner,
                     project_root: str, out_path: str,
                     checkpoint_every: int = 1000) -> Dict[str, Dict[str, str]]:
    """对缺失 caption 的事件用 VLM 补齐，增量保存。"""
    scene_norm = {normalize_rel_path(k, project_root): v for k, v in scene_metadata.items()}
    out = {k: dict(v) for k, v in existing.items()}

    pending: List[Tuple[str, int, str]] = []
    for vkey, meta in scene_norm.items():
        for ev in meta.get("events", []):
            if str(ev["event_id"]) in out.get(vkey, {}):
                continue
            clip = event_clip_path(vkey, ev["event_id"], clips_dir)
            if not os.path.exists(clip):
                logger.debug(f"[vlm] clip 不存在，跳过: {clip}")
                continue
            pending.append((vkey, ev["event_id"], clip))

    total = len(pending)
    if total == 0:
        logger.info("[vlm] 所有事件均已有 caption，跳过 VLM 阶段")
        return out
    logger.info(f"[vlm] 待生成 caption 事件数: {total}")

    for i, (vkey, eid, clip) in enumerate(pending, 1):
        cap = captioner.caption(clip)
        if cap:
            out.setdefault(vkey, {})[str(eid)] = cap
        if i % 50 == 0 or i == total:
            logger.info(f"[vlm] 进度 {i}/{total} ({100.0 * i / total:.1f}%)")
        if i % checkpoint_every == 0:
            save_captions(out_path, out)
            logger.info(f"[vlm] checkpoint dump @ {i}")
    save_captions(out_path, out)
    logger.info(f"[vlm] 完成，共生成 {total} 条 caption")
    return out


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="生成事件级 caption metadata（方案 J）")
    parser.add_argument("--scene_metadata", required=True, help="切场产物 scene_metadata.json")
    parser.add_argument("--output", required=True, help="输出 event_captions.json")
    parser.add_argument("--mode", choices=["harvest", "vlm", "auto"], default="auto",
                        help="harvest=只白嫖现成 caption; vlm=只跑 captioner; auto=先 harvest 再 VLM 补齐")
    parser.add_argument("--data_dirs", nargs="+", default=["sft/data", "rl/data"],
                        help="harvest 模式扫描的原始数据目录")
    parser.add_argument("--clips_dir", default="sft/data_events_d/event_clips",
                        help="vlm 模式读取的事件 clip 目录（方案 D 的产物）")
    parser.add_argument("--project_root", default=".", help="项目根目录")
    # VLM 参数
    parser.add_argument("--captioner_model", default=None, help="VLM 模型路径（vlm/auto 模式必需）")
    parser.add_argument("--captioner_frames", type=int, default=6, help="每个 clip 采样帧数")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="caption 长度上限")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="VLM 阶段每 N 条 dump 一次")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)

    with open(args.scene_metadata) as f:
        scene_metadata = json.load(f)
    logger.info(f"已加载 {len(scene_metadata)} 视频的 scene_metadata")

    existing = load_existing(args.output)
    if existing:
        n_ev = sum(len(v) for v in existing.values())
        logger.info(f"加载已有 captions: {len(existing)} 视频 / {n_ev} 事件，将增量补齐")

    # Stage 1: Harvest
    if args.mode in ("harvest", "auto"):
        harvested = harvest_captions(args.data_dirs, scene_metadata, project_root)
        # 与 existing 合并：existing 优先（不覆盖已有，保证幂等）
        for vkey, ev_caps in harvested.items():
            target = existing.setdefault(vkey, {})
            for eid, cap in ev_caps.items():
                target.setdefault(eid, cap)
        save_captions(args.output, existing)
        logger.info(f"[harvest] 已保存 → {args.output}")

    # Stage 2: VLM
    if args.mode in ("vlm", "auto"):
        if not args.captioner_model:
            logger.warning("[vlm] 未指定 --captioner_model，跳过 VLM 阶段")
        else:
            captioner = QwenVLCaptioner(
                args.captioner_model,
                n_frames=args.captioner_frames,
                max_new_tokens=args.max_new_tokens,
            )
            existing = vlm_fill_missing(
                scene_metadata, args.clips_dir, existing, captioner,
                project_root, args.output, args.checkpoint_every,
            )

    # 终态统计
    total_videos = len(existing)
    total_events = sum(len(v) for v in existing.values())
    target_events = sum(len(m.get("events", [])) for m in scene_metadata.values())
    logger.info("=" * 60)
    logger.info(f"完成: 覆盖 {total_videos}/{len(scene_metadata)} 视频, "
                f"{total_events}/{target_events} 事件 "
                f"({100.0 * total_events / max(target_events, 1):.1f}%)")
    logger.info(f"输出: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
