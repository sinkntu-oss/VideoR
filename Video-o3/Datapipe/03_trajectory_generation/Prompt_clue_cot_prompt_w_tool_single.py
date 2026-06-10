rewrite_prompt = """You are an intelligent video analysis assistant.
Your task is to generate a structured "Chain of Thought" (CoT) explanation for a video question, based on provided ground-truth visual clues.

I will provide you with a JSON object containing:
1. "question": The question to answer.
2. "options": Multiple choice options (if applicable).
3. "answer": The correct answer.
4. "clues": A list of visual clues, where each clue has a "timestamp" (start, end) and "text" (description).

Your need to generate a thinking process based on the provided information, progressing from the question's content to locating clues, observing those clues, and arriving at the answer.

Specifically, you need to generate two forms of this thinking process:
[THINKING_1]: Analyze the question and describe the process of locating the evidence: 
 - You should first analyze the question to determine what clues you need to find. 
 - Next, analyze the video content based on the given visual information to assess areas where clues may exist (if necessary, use timestamps and concise video clip descriptions).
 - Finally, determine the range of clues that require further investigation.
 - Please note: During this process, you should avoid mentioning anything like "based on the given clues". You should derive the correct clues through your own reasoning, not by directly reading the given clues.
[THINKING_2]: Carefully examine the clues found and determine the answer:
 - Describe the visual content at the grounded timestamps in detail, using the descriptions from the 'clues'. 
 - Explain how this visual evidence leads directly to the correct answer.

Output the reasoning process in the following JSON format:
[
  {
    "think": "[THINKING_1]",
    "grounding": ["start_time", "end_time"] 
  },
  {
    "think": "[THINKING_2]",
    "answer": "The uppercase letter option of the correct answer."
  }
]

The "grounding" field must strictly match the timestamps from the provided "clues".
The "answer" field must be the content of the correct option.
Input JSON:
"""
