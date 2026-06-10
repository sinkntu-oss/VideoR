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