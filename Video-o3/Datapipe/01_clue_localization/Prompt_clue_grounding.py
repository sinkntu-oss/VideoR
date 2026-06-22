rewrite_prompt = """Your task is to identify the precise timestamp intervals (clues) in the video that support the correct answer to the provided question. You will be provided with a single question, its options, and the correct answer. Your goal is to find the video evidence.

Instructions:
1. Analyze the Question and Answer: Understand what specific visual information is needed to justify the answer.

2. Locate Evidence in Video: Scan the video to find the exact time intervals where this information appears.

3. Define Timestamp Intervals (Critical Merging Step):
   - Record the start and end time for each relevant segment.
   - The `timestamp` field should represent the **Macro-Interval** (Start of the first relevant event -> End of the last continuous relevant event).
   - Merge continuous segments: If multiple relevant shots or events occur sequentially (or are separated by only a few seconds), combine them into a single, continuous Macro-Interval.
   - Do not break them into fragmented clips (e.g., a clip from 0:00-0:20 and another from 0:21-0:33 must be merged into 0:00-0:33).
   - Ensure the interval is accurate enough and long enough to contain the full context of the evidence.
   - Prefer longer, cohesive blocks over many short snippets.

4. Provide Explanations with Internal Details:
   - For each merged interval, write a comprehensive description of the visual content.
   - Use Internal Timestamps: You are advised to reference specific timestamps within the text description to point out details.

5. Verification: Double-check that the selected timestamps definitely contain the clue and that the description is accurate. Avoid incorrect or irrelevant timestamps.

Output Format (json):
{
    "clues": [
        {"timestamp": ["start_time", "end_time"], "text": "Description of the evidence block. You can describe the flow of events and mention specific moments (e.g., 'At M:SS...', 'From M:SS to N:SS...', etc.) inside this text."}
        // Add more clue objects if multiple segments are required
    ]
}

Here is the question information:"""
# ============ 中文翻译 ============
# 你的任务是识别视频中支持所提问题正确答案的精确时间戳区间（线索）。
# 你将获得一道题目、对应的选项以及正确答案，你的目标是从视频中找到支持该答案的证据。
#
# 操作说明：
# 1. 分析题目与答案：理解需要哪些具体的视觉信息来论证该答案。
#
# 2. 在视频中定位证据：扫描视频，找到该信息出现的精确时间区间。
#
# 3. 定义时间戳区间（关键合并步骤）：
#    - 记录每个相关片段的开始和结束时间。
#    - `timestamp` 字段应表示**宏观区间**（第一个相关事件的开始 -> 最后一个连续相关事件的结束）。
#    - 合并连续片段：如果多个相关镜头或事件按顺序出现（或仅相隔几秒），则将其合并为一个连续的宏观区间。
#    - 不要将其拆分为碎片化的片段（例如，0:00-0:20 的片段和 0:21-0:33 的片段必须合并为 0:00-0:33）。
#    - 确保区间足够准确且足够长，以包含证据的完整上下文。
#    - 优先选择较长的连贯片段，而非多个短片段。
#
# 4. 提供包含内部细节的说明：
#    - 对每个合并区间，撰写对视觉内容的全面描述。
#    - 使用内部时间戳：建议在文字描述中引用具体时间戳，以指出关键细节。
#
# 5. 核实：仔细确认所选时间戳确实包含线索，且描述准确无误。避免使用不正确或无关的时间戳。
#
# 输出格式（JSON）：
# {
#     "clues": [
#         {"timestamp": ["start_time", "end_time"], "text": "证据块的描述。你可以描述事件的发展过程，并在文字中提及具体时刻（例如，"在 M:SS 时……"、"从 M:SS 到 N:SS……"等）。"}
#         // 如需多个时间段，可添加更多线索对象
#     ]
# }
#
# 以下是题目信息：
# =================================
