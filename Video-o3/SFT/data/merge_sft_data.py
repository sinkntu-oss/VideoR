"""
Merge Seeker-173K SFT json files into sft_data_stage1.json and sft_data_stage2.json.

Stage 1: wo_tool + single_w_tool  (basic format alignment)
Stage 2: multi_w_tool + longvt    (TDAM multi-video attention masking)
"""

import json
import os

SFT_DIR = "/mnt/tidal-alsh01/dataset/eam_ds/VideoR/dataset/Seeker-173K/SFT"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

STAGE1_FILES = [
    "sft_llava-video_youtube_qa_mc_2_3_m_clue_single_wo_tool_9946.json",
    "sft_llava-video_youtube_qa_mc_2_3_m_clue_multi_wo_tool_29474.json",
    "sft_llava-video_youtube_qa_mc_2_3_m_clue_single_w_tool_79241.json",
    "sft_selfbuilt_1_qa_clue_single_w_tool_2574.json",
    "sft_selfbuilt_2_qa_f180to600_clue_single_w_tool_2502.json",
]

STAGE2_FILES = [
    "sft_llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_same_9886.json",
    "sft_llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_diff_2790.json",
    "sft_selfbuilt_2_qa_f180to600_clue_multi_w_tool_1606.json",
    "sft_longvt_longvideo_reflection_2000.json",
]


def merge(file_list, out_path):
    merged = []
    for fname in file_list:
        fpath = os.path.join(SFT_DIR, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        merged.extend(data)
        print(f"  {fname}: {len(data)} samples")
    with open(out_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False)
    print(f"-> {out_path} ({len(merged)} samples total)\n")


print("=== Stage 1 ===")
merge(STAGE1_FILES, os.path.join(OUT_DIR, "sft_data_stage1.json"))

print("=== Stage 2 ===")
merge(STAGE2_FILES, os.path.join(OUT_DIR, "sft_data_stage2.json"))
