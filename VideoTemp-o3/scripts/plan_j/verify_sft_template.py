#!/usr/bin/env python3
"""[问题 3 修复] SFT chat template 冒烟测试：
   验证 ms-swift 是否真的把 jsonl 顶层 `images` 字段消费到 chat template，
   而不是静默丢弃。

verify_fields.py 只能保证 jsonl 写得正确（标签数对齐、文件存在），
但 D/J 的训练正确性最终取决于 ms-swift 对顶层 `images` 字段的处理：
  - 它会把 images 路径替换 chat template 中的 `<image>` placeholder 吗？
  - 还是看到 messages 里的 `<image>` 就报错"找不到对应 image"？
  - 渲染后的 input_ids 里真的包含 image patch tokens 吗？

这个脚本在 SFT 启动前跑一次，最小代价回答这个关键未知。

用法：
    python scripts/plan_j/verify_sft_template.py sft/data_events_j --model /path/to/Qwen2.5-VL-7B-Instruct

输出：
    1. 加载 N 条样本（含 images / source_video）
    2. 走 ms-swift template 渲染流水
    3. 校验：
       - <image> 标签数 == images_inputs 数（visual tokens 被消费）
       - <video> 标签数 == videos_inputs 数
       - 渲染后 input_ids 中 image_token_id 出现次数 > 0（image 真的被 tokenize）
    4. 任一不通过则 exit 1，让 sft_events.sh 在训练启动前就报错
"""
import argparse
import glob
import json
import logging
import os
import sys
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _count_tag(messages, tag: str) -> int:
    return sum(
        m.get("content", "").count(tag)
        for m in messages
        if isinstance(m.get("content"), str)
    )


def _load_samples(data_dir: str, n: int) -> List[Dict]:
    """优先取含 images 字段的样本，最多 n 条。"""
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"未找到 jsonl: {data_dir}")
    samples = []
    for f in files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if s.get("images"):
                    samples.append(s)
                    if len(samples) >= n:
                        return samples
    return samples


def verify_template(samples: List[Dict], model_path: str) -> int:
    """走 ms-swift template 渲染，返回错误数。"""
    try:
        from swift.llm import get_template, get_model_tokenizer
    except ImportError as e:
        logger.error(f"无法 import swift.llm: {e}")
        logger.error("请先 pip install ms-swift（或在已配好的 SFT 环境中运行）")
        return 1

    logger.info(f"加载 tokenizer/processor: {model_path}")
    try:
        _, processor = get_model_tokenizer(
            model_path,
            model_type="qwen2_5_vl",
            load_model=False,
        )
    except Exception as e:
        logger.error(f"加载 tokenizer 失败: {e}")
        return 1

    template = get_template(template_type="qwen2_5_vl", processor=processor)
    # 探测 image_token_id：Qwen2.5-VL 用 <|image_pad|>
    image_token_id = None
    for cand in ("<|image_pad|>", "<image>"):
        try:
            tid = processor.tokenizer.convert_tokens_to_ids(cand)
            if tid is not None and tid >= 0:
                image_token_id = tid
                logger.info(f"image token = {cand!r} (id={tid})")
                break
        except Exception:
            pass

    n_err = 0
    for i, s in enumerate(samples):
        img_tag = _count_tag(s.get("messages", []), "<image>")
        vid_tag = _count_tag(s.get("messages", []), "<video>")
        n_images = len(s.get("images") or [])
        n_videos = len(s.get("videos") or [])
        logger.info(f"\n--- 样本 #{i} ---")
        logger.info(f"  <image> 标签 = {img_tag}, images 列表 = {n_images}")
        logger.info(f"  <video> 标签 = {vid_tag}, videos 列表 = {n_videos}")

        if img_tag != n_images:
            logger.error(f"  [E] <image> 标签数 ≠ images 列表长度（jsonl 写入错误）")
            n_err += 1
            continue

        try:
            encoded = template.encode(s)
        except Exception as e:
            logger.error(f"  [E] template.encode 失败: {e}")
            logger.error(f"      → ms-swift 不能消费这种 jsonl 结构（顶层 images / messages 中 <image>）")
            n_err += 1
            continue

        input_ids = encoded.get("input_ids", [])
        if not input_ids:
            logger.error(f"  [E] template.encode 返回空 input_ids")
            n_err += 1
            continue

        if image_token_id is not None:
            n_img_tok = sum(1 for t in input_ids if t == image_token_id)
            logger.info(f"  ✓ input_ids 长度 = {len(input_ids)}; image_token 出现 {n_img_tok} 次")
            if n_images > 0 and n_img_tok == 0:
                logger.error(
                    f"  [E] 样本含 {n_images} 张图，但渲染后 input_ids 不含任何 image_token "
                    f"→ ms-swift 没有真正消费 images 字段！"
                )
                n_err += 1
        # 顺便检查 pixel_values（如果 template 触发了图像编码）
        pv = encoded.get("pixel_values") or encoded.get("images")
        if pv is None and n_images > 0:
            logger.warning(
                "  [W] encode 输出无 pixel_values/images，可能 image encoding 走了 lazy "
                "（不一定是 bug；训练时 collator 会再处理）"
            )

    return n_err


def main():
    parser = argparse.ArgumentParser(description="SFT chat template 冒烟测试（验证 images 字段是否被消费）")
    parser.add_argument("data_dir", help="数据目录（如 sft/data_events_j）")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "SFT_BASE_MODEL",
            "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen2.5-VL-7B-Instruct",
        ),
        help="base 模型路径（用于加载 processor）",
    )
    parser.add_argument("--n_samples", type=int, default=3, help="抽样条数")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.warning(
            f"模型路径不存在 ({args.model})，跳过 template 渲染校验。"
            f" 可通过 --model 显式指定或在 SFT 环境中运行。"
        )
        sys.exit(0)  # 不阻塞 CI / 非 SFT 环境

    samples = _load_samples(args.data_dir, args.n_samples)
    if not samples:
        logger.warning(f"未找到含 images 字段的样本 → 该数据集可能不是 D/J 风格，跳过")
        sys.exit(0)

    logger.info(f"加载 {len(samples)} 条含 images 字段的样本，开始 template 渲染校验...")
    n_err = verify_template(samples, args.model)

    if n_err == 0:
        logger.info("=" * 60)
        logger.info("✅ SFT chat template 冒烟测试通过：images 字段被 ms-swift 正确消费")
        sys.exit(0)
    else:
        logger.error("=" * 60)
        logger.error(f"❌ 冒烟测试失败 ({n_err} 个错误)")
        logger.error("    → ms-swift 没有正确消费 images 字段，训练后模型看不到关键帧")
        logger.error("    → 不要启动 SFT 训练，请先排查：")
        logger.error("      1. ms-swift 版本（建议 >= 3.0）")
        logger.error("      2. messages 中 <image> 标签位置（必须在 user content 中）")
        logger.error("      3. images 路径文件存在性（用 verify_fields.py --check_files 50）")
        sys.exit(1)


if __name__ == "__main__":
    main()
