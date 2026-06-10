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