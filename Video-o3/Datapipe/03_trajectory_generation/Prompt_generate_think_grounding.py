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
