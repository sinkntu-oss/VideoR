#!/usr/bin/env python3
"""[H2] 校验方案 D / J 转换后的 jsonl 字段齐全，防止 ms-swift 字段透传问题在训练时才暴露。

检查项：
  - SFT/RL 通用：messages, events, covering_event_ids/gt_covering_event_ids
  - D/J 关键帧版：source_video, images
  - 一致性：<image>/<video> 标签数 == images/videos 列表长度
  - 文件存在性：随机抽 N 条样本检查 images / source_video 真实存在（基于 project_root）

用法：
    python scripts/plan_j/verify_fields.py rl/data_events_j --project_root .
    python scripts/plan_j/verify_fields.py sft/data_events_j --stage sft
"""
import argparse
import glob
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 关键字段定义（按数据形态分类）
COMMON_KEYS = ("messages",)
EVENT_KEYS = ("events",)                                 # convert_*.py 都会写入
GROUNDING_KEYS = ("gt_covering_event_ids",)              # 只对 grounding 数据
QA_KEYS = ("covering_event_ids",)                        # 部分样本有
KEYFRAME_KEYS = ("source_video", "images")               # 仅 D / J


def _count_tag(messages, tag: str) -> int:
    return sum(
        m.get("content", "").count(tag)
        for m in messages
        if isinstance(m.get("content"), str)
    )


def verify_sample(s: Dict, idx: int, project_root: str, errors: List[str], stats: Counter) -> None:
    # 通用
    for k in COMMON_KEYS + EVENT_KEYS:
        if k not in s:
            errors.append(f"[#{idx}] 缺失关键字段: {k}")
            stats[f"missing_{k}"] += 1

    events = s.get("events") or []
    msgs = s.get("messages") or []

    # events 字段元素结构
    if events:
        ev0 = events[0]
        for need in ("event_id", "start_time", "end_time"):
            if need not in ev0:
                errors.append(f"[#{idx}] events[0] 缺 {need}: {ev0}")
                stats[f"events_missing_{need}"] += 1
                break

    # D/J 关键帧版判定：含 images 字段或第一轮 user 含 <image>
    has_image_tag = _count_tag(msgs, "<image>") > 0
    has_images_field = "images" in s and s["images"]
    is_keyframe_variant = has_image_tag or has_images_field

    if is_keyframe_variant:
        for k in KEYFRAME_KEYS:
            if k not in s or not s[k]:
                errors.append(f"[#{idx}] 关键帧版样本缺 {k}（D/J 必需）")
                stats[f"keyframe_missing_{k}"] += 1
        # 标签数对齐
        img_tag = _count_tag(msgs, "<image>")
        if img_tag != len(s.get("images", []) or []):
            errors.append(f"[#{idx}] <image>={img_tag} ≠ images={len(s.get('images', []) or [])}")
            stats["image_align_mismatch"] += 1

    vid_tag = _count_tag(msgs, "<video>")
    vids = s.get("videos", []) or []
    if vid_tag != len(vids):
        errors.append(f"[#{idx}] <video>={vid_tag} ≠ videos={len(vids)}")
        stats["video_align_mismatch"] += 1

    # grounding / qa 二选一标签
    if isinstance(s.get("timestamp"), list) and s["timestamp"]:
        if "covering_event_ids" not in s:
            stats["missing_covering_event_ids"] += 1  # 软警告
    if isinstance(s.get("gt_time_stamp"), list) and s["gt_time_stamp"]:
        if "gt_covering_event_ids" not in s:
            errors.append(f"[#{idx}] grounding 样本缺 gt_covering_event_ids")
            stats["missing_gt_covering_event_ids"] += 1


def check_files_exist(samples: List[Dict], project_root: str, sample_n: int, stats: Counter) -> List[str]:
    """随机抽 N 条检查 source_video / images 真实存在。"""
    errors = []
    if not samples:
        return errors
    picks = random.sample(samples, min(sample_n, len(samples)))
    for i, s in enumerate(picks):
        sv = s.get("source_video")
        if sv:
            p = sv if os.path.isabs(sv) else os.path.join(project_root, sv)
            if not os.path.exists(p):
                errors.append(f"[file#{i}] source_video 不存在: {p}")
                stats["missing_source_video_file"] += 1
        for img in (s.get("images") or [])[:3]:  # 每条只抽前 3 张
            p = img if os.path.isabs(img) else os.path.join(project_root, img)
            if not os.path.exists(p):
                errors.append(f"[file#{i}] image 不存在: {p}")
                stats["missing_image_file"] += 1
                break
        for vid in (s.get("videos") or [])[:1]:
            p = vid if os.path.isabs(vid) else os.path.join(project_root, vid)
            if not os.path.exists(p):
                errors.append(f"[file#{i}] video 不存在: {p}")
                stats["missing_video_file"] += 1
                break
    return errors


def main():
    parser = argparse.ArgumentParser(description="校验 D/J 转换后 jsonl 字段齐全")
    parser.add_argument("data_dir", help="数据目录（如 rl/data_events_j）")
    parser.add_argument("--project_root", default=".", help="项目根目录")
    parser.add_argument("--check_files", type=int, default=20,
                        help="随机抽 N 条样本检查实际文件存在（0=跳过）")
    parser.add_argument("--max_errors", type=int, default=20, help="最多打印 N 条错误")
    parser.add_argument("--stage", choices=["sft", "rl", "auto"], default="auto",
                        help="auto: 根据目录名推断")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    files = sorted(glob.glob(os.path.join(args.data_dir, "**/*.jsonl"), recursive=True))
    if not files:
        logger.error(f"目录中没有 jsonl: {args.data_dir}")
        sys.exit(2)

    logger.info(f"扫描 {len(files)} 个 jsonl, project_root={project_root}")
    all_samples: List[Dict] = []
    field_presence = Counter()
    n_total = 0
    errors: List[str] = []
    stats: Counter = Counter()

    for f in files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                try:
                    s = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"[{f}] JSON 解析失败: {e}")
                    stats["json_decode_error"] += 1
                    continue
                for k in s.keys():
                    field_presence[k] += 1
                verify_sample(s, n_total, project_root, errors, stats)
                all_samples.append(s)

    # 文件存在性抽检
    if args.check_files > 0:
        logger.info(f"随机抽 {args.check_files} 条样本检查实际文件存在...")
        errors.extend(check_files_exist(all_samples, project_root, args.check_files, stats))

    # 报告
    logger.info("=" * 70)
    logger.info(f"扫描完成: {n_total} 条样本，{len(errors)} 条问题")
    logger.info("-" * 70)
    logger.info("字段命中率（顶层字段在多少 % 样本中存在）:")
    for k in sorted(field_presence.keys()):
        pct = 100.0 * field_presence[k] / max(n_total, 1)
        logger.info(f"  {k:30s} {field_presence[k]:8d}  ({pct:5.1f}%)")
    if stats:
        logger.info("-" * 70)
        logger.info("问题统计:")
        for k, v in sorted(stats.items(), key=lambda x: -x[1]):
            logger.info(f"  {k:35s} {v}")
    if errors:
        logger.info("-" * 70)
        logger.info(f"问题样例（前 {args.max_errors} 条）:")
        for e in errors[: args.max_errors]:
            logger.info(f"  {e}")
        logger.info("=" * 70)
        logger.error(f"❌ 校验失败: {len(errors)} 条问题")
        sys.exit(1)
    else:
        logger.info("=" * 70)
        logger.info("✅ 校验通过")
        sys.exit(0)


if __name__ == "__main__":
    main()
