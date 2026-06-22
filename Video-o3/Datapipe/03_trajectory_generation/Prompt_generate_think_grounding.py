rewrite_prompt = """You are an intelligent video analysis assistant.
Your task is to generate a structured "Chain of Thought" (CoT) explanation for a video question, based on provided ground-truth visual clues.

I will provide you with a JSON object containing:
1. "question": The question to answer.
2. "options": Multiple choice options (if applicable).
3. "grounding": A time interval containing the relevant clues for answering the question.
4. "think_answer": The thinking process of answering the question by observing only the "grounding" segment of the video.
5. "answer": The correct answer of the question.

Your job is to rewrite the sample into a tool-grounding-style multi-round reasoning format: Your need to generate a thinking process based on the provided information, progressing from the question's content to locating clues, observing those clues, and arriving at the answer.
 - Hint: The temporal segments ("grounding") are tool calls. A "grounding" step chooses a time interval to carefully inspect.

Specifically, you need to generate two forms of this thinking process:
[think_grounding]: Analyze the question and describe the process of locating the evidence: 
 - You should first analyze the question to determine what clues you need to find. 
 - Next, analyze the video content based on the given visual information to assess areas where clues may exist (if necessary, use timestamps and concise video clip descriptions).
 - Finally, determine the range of clues that require further investigation.
 - Please note: During this process, you should avoid mentioning anything like "based on the given clues". You should derive the correct clues through your own reasoning, not by directly reading the given clues.
[think_answer]: Carefully examine the clues found and determine the answer:
 - Describe the visual content at the grounded timestamps in detail. 
 - Explain how this visual evidence leads directly to the correct answer.
 - Please Note: You should use the original given "think_answer" as much as possible, only add extra reasoning or modify existing content when necessary. It is recommended to make appropriate language modifications to make the entire reasoning process more coherent.

Output the reasoning process in the following JSON format:
[
  {
    "think_grounding": "Your generated think_grounding",
    "grounding": ["start_time", "end_time"] 
  },
  {
    "think_answer": "Your generated think_answer",
    "answer": "The uppercase letter option of the correct answer."
  }
]

The "grounding" field must strictly match the timestamps from the provided "clues".
The "answer" field must be the content of the correct option.
Input JSON:
"""
# ============ 中文翻译 ============
# 你是一位智能视频分析助手。
# 你的任务是基于所提供的真实视觉线索，为视频问题生成结构化的"思维链"（CoT）解释。
#
# 我将提供给你一个 JSON 对象，其中包含：
# 1. "question"：需要回答的问题。
# 2. "options"：多项选择选项（如适用）。
# 3. "grounding"：包含回答该问题所需相关线索的时间区间。
# 4. "think_answer"：仅通过观看视频中 "grounding" 指定片段来回答该问题的思维过程。
# 5. "answer"：该问题的正确答案。
#
# 你的工作是将样本改写为工具定位式（tool-grounding-style）的多轮推理格式：你需要根据所提供的信息，
# 生成一套从问题出发，逐步定位线索、观察线索，最终得出答案的思维过程。
#  - 提示：时间片段（"grounding"）是工具调用。一次 "grounding" 步骤会选择一个时间区间进行仔细检视。
#
# 具体来说，你需要生成两种形式的思维过程：
# [think_grounding]：分析问题并描述定位证据的过程：
#  - 首先分析问题，确定需要找到哪些线索。
#  - 接下来，根据给定的视觉信息分析视频内容，评估线索可能存在的区域
#    （如有必要，使用时间戳和简洁的视频片段描述）。
#  - 最后，确定需要进一步调查的线索范围。
#  - 请注意：在此过程中，避免提及"根据给定线索"之类的表述。你应该通过自己的推理推导出正确的线索，
#    而非直接读取已提供的线索。
# [think_answer]：仔细检视找到的线索并确定答案：
#  - 详细描述定位时间戳处的视觉内容。
#  - 解释这些视觉证据如何直接导出正确答案。
#  - 请注意：尽可能使用原始给定的 "think_answer"，仅在必要时添加额外推理或修改已有内容。
#    建议进行适当的语言调整，使整体推理过程更加连贯。
#
# 以下列 JSON 格式输出推理过程：
# [
#   {
#     "think_grounding": "你生成的 think_grounding",
#     "grounding": ["start_time", "end_time"]
#   },
#   {
#     "think_answer": "你生成的 think_answer",
#     "answer": "正确答案的大写字母选项。"
#   }
# ]
#
# "grounding" 字段必须与所提供的 "clues" 中的时间戳严格匹配。
# "answer" 字段必须是正确选项的内容。
# 输入 JSON：
# =================================
