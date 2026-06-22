rewrite_prompt = """Your task is to generate a detailed solution for a given [QUESTION], which may involve understanding and analyzing video content as well as related text information. Follow these instructions specifically for video question answering:

1. Use the same language as the original question (if the question is in English, answer in English; if the question is in another language, answer in that language).
2. Solution Crafting:
- Provide a comprehensive response that includes each intermediate step, as well as explanations of your reasoning throughout.
- Ensure your solution is well-structured, logically ordered, and incorporates all relevant details from both the video and any accompanying text or multimodal elements.
3. Content Requirements:
- The solution must be complete, with each step thoroughly explained and justified so that the reasoning is transparent and logically sound.
- If necessary, provide concrete visual evidence using timestamped descriptions of the visual content. All timestamps must be formatted as M:SS.
4. Clarification of Problem Understanding:
- If there are any necessary assumptions or interpretations based on the video context (such as scene changes, actions, events), note these early in the answer.
5. Analysis of Video:
- Carefully analyze the video content, including key frames, motion, events, interactions, temporal sequences.
- Highlight and explain how video elements, sequences, or timestamps contribute to the reasoning and the answer.
- Incorporate and interpret any supplementary data (text overlays, subtitles, or related information).
6. Logical Flow:
- Ensure every step follows naturally from the previous one, with clear connections and no logical leaps, making your reasoning easy to understand.
7. Error Checking:
- Carefully review each reasoning step. If there are different possible interpretations or uncertainties in the video, state which you choose and why, and clarify if any assumptions or approximations are made.
8. Please do not include irrelevant information; address only what is required for the video question answering.

**Output Format (json)**:
{
    "think": "The reasoning process you generated.",
    "answer": "The answer options you provided."
}

Here is the specific [QUESTION]:
"""
# ============ 中文翻译 ============
# 你的任务是为给定的 [QUESTION]（问题）生成一套详细的解题过程，该过程可能涉及对视频内容以及相关文本信息的理解与分析。
# 请严格按照以下针对视频问答的操作说明进行：
#
# 1. 使用与原始问题相同的语言（若问题为中文，则用中文作答；若问题为其他语言，则用该语言作答）。
# 2. 解题过程要求：
# - 提供包含每个中间步骤的完整解答，并在整个过程中对推理过程加以解释。
# - 确保解题过程结构清晰、逻辑有序，并融合来自视频及任何附加文本或多模态元素的所有相关细节。
# 3. 内容要求：
# - 解题过程必须完整，每一步骤均须详细解释并说明理由，使推理过程透明且逻辑严密。
# - 如有必要，请使用带有时间戳的视觉内容描述作为具体视觉证据。所有时间戳必须格式化为 M:SS。
# 4. 问题理解说明：
# - 如果存在基于视频背景所必需的假设或解读（例如场景切换、动作、事件），请在答案开头处注明。
# 5. 视频分析：
# - 仔细分析视频内容，包括关键帧、运动、事件、交互、时间序列。
# - 突出并解释视频元素、序列或时间戳如何对推理和答案产生贡献。
# - 融合并解读所有补充数据（文字叠加、字幕或相关信息）。
# 6. 逻辑流畅性：
# - 确保每一步骤自然地衔接上一步骤，连接清晰、无逻辑跳跃，使推理过程易于理解。
# 7. 错误核查：
# - 仔细审查每个推理步骤。如果视频中存在不同的可能解读或不确定性，请说明你选择的解读及原因，
#   并阐明所做的任何假设或近似处理。
# 8. 请不要包含无关信息；仅针对视频问答所需内容进行作答。
#
# **输出格式（JSON）**：
# {
#     "think": "你生成的推理过程。",
#     "answer": "你给出的答案选项。"
# }
#
# 以下是具体的 [QUESTION]：
# =================================
