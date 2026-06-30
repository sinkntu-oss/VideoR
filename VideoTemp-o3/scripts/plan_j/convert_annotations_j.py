#!/usr/bin/env python3
"""方案 J：事件级 caption + 1 张代表帧。

与方案 D 的区别：
  1. 每事件 1 张关键帧（D 是 2 张）
  2. system prompt 中每事件附 1 句 caption（从 event_captions.json 注入）
  3. 其余流程（关键帧抽取 / source_video 注入 / SFT&RL 双 patch）完全复用 D

实现策略 - 三层 monkey-patch：
  - 层 1: 改 `_ca.lookup_events` → 在 events 上挂 caption 字段
  - 层 2: 改 `_ca.build_system_prompt` → 输出含 caption 的 system prompt
  - 层 3: 复用 `_cd.convert_sft_sample / convert_rl_sample` + 改 `_cd.N_KEYFRAMES_PER_EVENT=1`

caption 文件路径通过环境变量 EVENT_CAPTIONS 指定（默认 scripts/plan_j/event_captions.json）。

健壮性设计（见 README「已知问题」章节）：
  - P0-1: caption 字符清洗（去除 \\n / \\r / "）
  - P0-3: 所有 patch 加 _j_patched 标记，多次 import / reload 幂等
  - P1-4: 自检改为显式 raise，兼容 python -O
  - P1-7: 加载 caption 时校验 scene_metadata SHA1，避免事件 id 错位

用法：
    # SFT
    python scripts/plan_j/convert_annotations_j.py \\
        --metadata scripts/scene_metadata.json \\
        --input_dir sft/data --output_dir sft/data_events_j --data_stage sft

    # RL
    python scripts/plan_j/convert_annotations_j.py \\
        --metadata scripts/scene_metadata.json \\
        --input_dir rl/data --output_dir rl/data_events_j --data_stage rl
"""
import hashlib
import json
import logging
import os
import sys

# 把上级 scripts/ 加入 sys.path 以便 import 同级的 convert_annotations(_d)
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS_DIR)

import convert_annotations as _ca           # noqa: E402
import convert_annotations_d as _cd         # noqa: E402  关键帧抽取 / source_video 注入
from convert_annotations import main        # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================
# 配置常量
# ============================================================

DEFAULT_CAPTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "event_captions.json"
)
CAPTIONS_PATH = os.environ.get("EVENT_CAPTIONS", DEFAULT_CAPTIONS_PATH)
CAPTION_MAX_CHARS = 200


# ============================================================
# 工具函数
# ============================================================

def _sanitize_caption(text):
    """[P0-1] 清洗 caption 中破坏 system prompt 结构的字符。

    - \\n / \\r → 空格（防止换行错位事件列表）
    - " → '（防止视觉上断行；jsonl 序列化会自动转义不会出错，但 LLM 看到的纯文本会）
    """
    if not text:
        return "(no description)"
    cleaned = text.replace("\n", " ").replace("\r", " ").replace('"', "'").strip()
    return cleaned or "(no description)"


def _file_sha1(path):
    """流式计算大文件 SHA1，用于 caption ↔ scene_metadata 一致性校验。"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# 加载 caption metadata（[P1-7] 含 scene_metadata SHA1）
# ============================================================

def _load_captions(path):
    """返回 (captions_dict, expected_scene_sha1 or None)。"""
    if not os.path.exists(path):
        logger.warning(
            f"⚠️  event_captions.json 不存在: {path}\n"
            f"   所有事件 caption 将兜底为 '(no description)'。\n"
            f"   请先运行 scripts/plan_j/generate_event_captions.py 生成。"
        )
        return {}, None
    with open(path) as f:
        data = json.load(f)
    meta = data.pop("_meta", None) if isinstance(data, dict) else None
    expected_sha1 = (meta or {}).get("scene_metadata_sha1")
    n_videos = len(data)
    n_events = sum(len(v) for v in data.values() if isinstance(v, dict))
    logger.info(f"[plan-j] 已加载 captions: {n_videos} 视频 / {n_events} 事件 ← {path}")
    if expected_sha1:
        logger.info(f"[plan-j] caption 关联 scene_metadata.sha1 = {expected_sha1[:12]}...")
    else:
        logger.warning("[plan-j] caption 文件未记录 scene_metadata.sha1，无法校验一致性")
    return data, expected_sha1


CAPTIONS, _EXPECTED_SCENE_SHA1 = _load_captions(CAPTIONS_PATH)


def _verify_scene_metadata_consistency():
    """[P1-7] main() 启动前校验 caption 与当前 scene_metadata 一致。

    通过预扫 sys.argv 拿到 --metadata 的值（main 内部 argparse 会复用 sys.argv，
    这里只做被动读取，不影响 argparse）。
    """
    if not _EXPECTED_SCENE_SHA1 or not CAPTIONS:
        return
    meta_path = None
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--metadata" and i + 1 < len(argv):
            meta_path = argv[i + 1]
            break
        if arg.startswith("--metadata="):
            meta_path = arg.split("=", 1)[1]
            break
    if not meta_path or not os.path.exists(meta_path):
        return  # main() 会自行报错
    actual = _file_sha1(meta_path)
    if actual != _EXPECTED_SCENE_SHA1:
        raise RuntimeError(
            "scene_metadata.json 与 event_captions.json 不一致！\n"
            f"  scene_metadata.sha1 = {actual}\n"
            f"  caption._meta.sha1   = {_EXPECTED_SCENE_SHA1}\n"
            "  scene_metadata 已重新生成（事件 id 含义可能变化），"
            "  caption 仍是旧的对应关系，会导致全数据集错位。\n"
            "  请重新运行 scripts/plan_j/generate_event_captions.py 重新生成 caption。"
        )
    logger.info("[plan-j] scene_metadata 一致性校验通过 ✓")


# ============================================================
# Patch 1: N_KEYFRAMES_PER_EVENT = 1 ([P0-3] 幂等)
# ============================================================

if not getattr(_cd, "_j_n_kf_patched", False):
    _cd.N_KEYFRAMES_PER_EVENT = 1
    _cd._j_n_kf_patched = True


# ============================================================
# Patch 2: lookup_events → 注入 caption ([P0-3] 幂等)
# ============================================================

if not getattr(_ca.lookup_events, "_j_patched", False):
    _orig_lookup = _ca.lookup_events

    def lookup_events_with_caption(index, video_path, project_root):
        events = _orig_lookup(index, video_path, project_root)
        if events is None:           # baseline 语义：meta 缺失
            return None
        if not events:                # 空事件列表，原样返回
            return events
        vkey = _ca.normalize_rel_path(video_path, project_root)
        caps = CAPTIONS.get(vkey, {})
        # 浅拷贝每个 event dict 避免污染共享的 scene_metadata
        return [
            dict(e, caption=caps.get(str(e["event_id"]), "(no description)"))
            for e in events
        ]

    lookup_events_with_caption._j_patched = True
    _ca.lookup_events = lookup_events_with_caption
else:
    # 已被 patch 过（reload 场景）；复用现存引用
    lookup_events_with_caption = _ca.lookup_events
    logger.debug("[plan-j] lookup_events 已被 patch，跳过重复 patch")


# ============================================================
# Patch 3: build_system_prompt → 注入 caption + 单帧索引说明
# ============================================================

SYSTEM_PROMPT_TEMPLATE_J = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

The video has been segmented into {n} temporally ordered events (indexed 0 to {last}). Each event is described by a brief summary and accompanied by ONE representative keyframe. The {n} keyframes are listed in event order: keyframe i corresponds to event i.

Events:
{event_list}

The keyframes provide visual evidence; the summaries help you quickly identify which events are relevant to the question. If you need to examine any specific event more closely (e.g., to verify visual details not captured in the summary), you may call:

<tool_call>{{"name":"locate_events","arguments":{{"event_ids":[event_id_1, event_id_2, ...]}}}}</tool_call>

The tool will return the full video clips of the selected events for you to refine your answer."""


def build_system_prompt(events):
    """事件列表带 caption；保持与原版同样的 (events,) 单参签名。"""
    lines = []
    for e in events:
        cap = _sanitize_caption(e.get("caption"))
        if len(cap) > CAPTION_MAX_CHARS:
            cap = cap[:CAPTION_MAX_CHARS - 3].rstrip() + "..."
        lines.append(
            f"  Event {e['event_id']} ({e['start_time']:.1f}s-{e['end_time']:.1f}s): \"{cap}\""
        )
    n = len(events)
    return SYSTEM_PROMPT_TEMPLATE_J.format(
        n=n,
        last=max(0, n - 1),
        event_list="\n".join(lines),
    )


build_system_prompt._j_patched = True

# Patch 顺序：必须晚于 import convert_annotations_d（_cd 在 import 时已经把 D 版
# build_system_prompt 写入 _ca，这里覆盖为 J 版即可）；[P0-3] 幂等保护
if not getattr(_ca.build_system_prompt, "_j_patched", False):
    _ca.build_system_prompt = build_system_prompt
else:
    logger.debug("[plan-j] build_system_prompt 已被 patch，跳过重复 patch")


# ============================================================
# [P1-4] 不变量自检：显式 raise（兼容 -O 模式，assert 会被剥离）
# ============================================================

def _selfcheck():
    if _ca.lookup_events is not lookup_events_with_caption:
        raise RuntimeError("J: lookup_events patch 失败")
    if _ca.build_system_prompt is not build_system_prompt:
        raise RuntimeError("J: build_system_prompt patch 失败")
    if _ca.convert_sft_sample is not _cd.convert_sft_sample:
        raise RuntimeError(
            "J: convert_sft_sample 未被 D 接管 —— "
            "可能是 _cd 的 patch 顺序异常或被第三方代码覆盖"
        )
    if _ca.convert_rl_sample is not _cd.convert_rl_sample:
        raise RuntimeError("J: convert_rl_sample 未被 D 接管")
    if _cd.N_KEYFRAMES_PER_EVENT != 1:
        raise RuntimeError(
            f"J: N_KEYFRAMES_PER_EVENT={_cd.N_KEYFRAMES_PER_EVENT}, 应为 1"
        )


_selfcheck()


if __name__ == "__main__":
    _verify_scene_metadata_consistency()
    main()
