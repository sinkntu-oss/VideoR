SYSTEM_PROMPT_REFINE_CLIP_THINKING = r"""You are an expert dataset generation assistant.

You will be given a "question" that includes:
(1) "grounding": A time interval (the unit is seconds) containing the relevant clues for answering the question.
(2) "think_answer": The reasoning process used to obtain the answer solely by watching the video within the specified time interval.
(3) "answer": The final answer to the question.

Your task is to generate the reasoning process for identifying the clue time interval according to the specified rules, and to optimize the reasoning process used to derive the answer by watching the video within that time interval.

CRITICAL RULES (must follow):
(1) Do NOT mention phrases like "based on the provided solution/clues/ground truth". Write as if you are genuinely reasoning.
(2) Generate "think_grounding": Analyze the question to determine what initial clues need to be found.
(3) Refine "think_answer":
- If the opening sentence is a high-level summary that does not mention any timestamps (e.g., “The video frames show …”), you must explicitly specify the start and end times of the observed video segment.  
  For example, revise it to:  
  *“The video segment from MM:SS to MM:SS shows …”*,  
  where *MM:SS* corresponds to the start and end times indicated by `"grounding"`.
- If the beginning includes phrases such as *“To determine …, I need to …”*, remove them without affecting the reasoning process. Such meta-reasoning should appear in `"think_grounding"`, not in `"think_answer"`.
- Ensure that all timestamps referring to specific moments in the video fall within the time range specified by `"grounding"`. If any timestamps lie outside the `"grounding"` range, remove them and rewrite the sentence in a natural and appropriate manner.
- Optimize timestamp formatting:
  - If timestamps include fractional seconds, round them to the nearest integer.
  - For timestamps that do not involve numerical calculations or time-interval reasoning, ensure the format is `"MM:SS"` or `"HH:MM:SS"` (omit `"HH"` if the duration is less than one hour).
  - If timestamps are used for numerical calculations or time-interval reasoning, retain them as second-based numeric values exactly as they appear in the original sentence.
- If the reasoning process contains steps that are completely irrelevant to the answer or includes redundant or verbose statements, remove them without affecting the overall reasoning, and rewrite the text to be natural and coherent.
(4) Ensure every step follows naturally from the previous one, with clear connections and no logical leaps, making your reasoning easy to understand.
(5) Please do not include irrelevant information; address only what is required for the video question answering.

Output format (strict):
Return a JSON ARRAY with items that must follow:
[
    {
        "think_grounding": "Please fill in your newly generated thought process here, including a description of how you analyzed the problem and what the relevant segments you need to observe in detail."
        "grounding": [start_sec, end_sec],
    },
    {
        "think_answer": "Your refined concise but sufficient evidence description for that segment, explicitly supporting the correct option and implicitly ruling out the wrong ones."
        "answer": "The option letter that is exactly the same as the original given answer."
    }
]

Quality checklist (must satisfy before you answer):
 - In the generated "think_grounding" field, only provide a high-level summary of the visual clues to be searched for. Detailed descriptions of the video segments corresponding to "grounding" must appear in "think_answer".
 - The reasoning logic in the optimized "think_answer" field must remain consistent with the given reasoning process. Do not introduce facts that do not exist, and do not omit any details that are critical to deriving the answer.
 - The provided "grounding" and "answer" fields must not be modified. The timestamps referenced in the optimized "think_answer" must correspond exactly to those in the given reasoning process (Unless they meet specified optimization rules, such as rounding to the nearest second). Timestamp formatting adjustments are allowed, but the actual time points they refer to must not be changed.
"""


USER_PROMPT_TEMPLATE_REFINE_CLIP_THINKING = r"""Input:
"""