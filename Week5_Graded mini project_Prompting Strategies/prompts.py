"""
Prompt templates and builder functions for all extraction strategies.
"""

def zero_shot_prompt(snippet: str) -> str:
    return f"""Extract the following details from job posting below and return ONLY a valid JSON object:
    {{
      "company": "<name>",
      "role": "<title>",
      "years_experience_required": <integer or null>
    }}

    Job Posting:
    {snippet}
    """

def few_shot_prompt(snippet: str) -> str:
    return f"""Extract the following details from job posting below. Return ONLY a valid JSON object.

Example 1:
Job posting: "Walmart is hiring a senior AI ML engineer. Applicants should have at least 3 years of experience in AI development."
Output:
{{"company": "Walmart", "role": "Senior AI ML engineer", "years_experience_required": 3}}

Example 2:
Job posting: "Meta is looking for a data scientist with 2+ years of hands on ML experience to join our AI team."
Output:
{{"company": "Meta", "role": "Data scientist", "years_experience_required": 2}}

Now extract from job posting:
{snippet}
"""

def structured_prompt(snippet: str) -> str:
    return f"""You are an information extraction assistant.
From the job posting below, extract exactly three fields and return them as a valid JSON object.
Required format:
{{
  "company": "<company name string>",
  "role": "<job title string>",
  "years_experience_required": <min years as integer or null>
}}
Do not include any explanation or extra text - return ONLY the JSON object.

Job posting:
{snippet}
"""

def cot_prompt(snippet: str) -> str:
    return f"""Read the job posting below and extract the key details by thinking through each step.
Step 1: Identify the company name.
Step 2: Identify the job title or role.
Step 3: Find any mention of years of experience required. Convert phrases like "3-5 years" to 3. If none, return null.

Provide your final JSON answer in this exact structure:
{{
  "company": "<name>",
  "role": "<title>",
  "years_experience_required": <integer or null>
}}

Job posting:
{snippet}
"""

def judge_prompt_template(snippet: str, llm_extracted: dict, golden_item: dict) -> str:
    return f"""You are an evaluator checking the quality of an information extraction result.
Job posting:
{snippet}

Extracted results:
Company: {llm_extracted.get("company")}
Role: {llm_extracted.get("role")}
Years of experience required: {llm_extracted.get("years_experience_required")}

Expected results:
Company: {golden_item.get("company")}
Role: {golden_item.get("role")}
Years of experience required: {golden_item.get("years_experience_required")}

Rate the output quality on a scale of 1 to 5 (5 = perfect match, 1 = completely wrong).
Respond in this format only:
Score: <1 to 5>
Reason: <Your in justification one sentence>
"""