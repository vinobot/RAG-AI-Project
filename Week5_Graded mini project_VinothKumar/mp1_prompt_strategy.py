import json
import time
import re
import os

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Step 1: Loading env file and assigning OpenAI model details
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

MODEL = "gpt-3.5-turbo"
JUDGE_MODEL = "gpt-4o-2024-05-13"

# Pricing reference for cost calculation
COST_PER_1K_INPUT_TOKENS = 0.0005
COST_PER_1K_OUTPUT_TOKENS = 0.0015
print("Initial setup complete...!")
print("   Model:          JudgeModel: \n", MODEL, JUDGE_MODEL)

# Step 2: Loading dataset
def load_data(snippets_path="data/job_snippets.jsonl", goldendata_path="data/golden_set.jsonl"):
    with open(snippets_path, "r", encoding="utf-8") as f:
        snippets = [json.loads(line.strip()) for line in f if line.strip()]

    with open(goldendata_path, "r", encoding="utf-8") as f:
        golden = [json.loads(line.strip()) for line in f if line.strip()]

    return snippets, golden

# Step 3: Building prompts (Updated key to years_experience_required)

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

# Step 4: Call LLM function
def call_llm(prompt: str) -> dict:
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    elapsed = round(time.time() - start, 3)
    return {
        "text": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency_in_sec": elapsed
    }

# Step 5: Parse LLM responses
def parse_response(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
        # Normalize dictionary keys to handle variations robustly
        normalized_data = {str(k).lower().replace(" ", "_"): v for k, v in data.items()}
        return {
            "company": normalized_data.get("company"),
            "role": normalized_data.get("role"),
            "years_experience_required": (
                normalized_data.get("years_experience_required")
                or normalized_data.get("years_of_experience")
                or normalized_data.get("years_of_exp")
            )
        }
    except Exception:
        return {"company": None, "role": None, "years_experience_required": None}

# Step 6: Check accuracy
def normalise(value) -> str:
    return str(value).lower().strip() if value is not None else ""

def check_accuracy(llm_extracted: dict, golden_item: dict) -> dict:
    company_status = normalise(llm_extracted.get("company")) == normalise(golden_item.get("company"))
    role_status = normalise(llm_extracted.get("role")) == normalise(golden_item.get("role"))
    experience_status = normalise(llm_extracted.get("years_experience_required")) == normalise(golden_item.get("years_experience_required"))
    
    score = round(sum([company_status, role_status, experience_status]) / 3, 4)
    return {
        "company_match": company_status,
        "role_match": role_status,
        "experience_match": experience_status,
        "accuracy_score": score
    }

# Step 7: Call LLM Judge
def call_llm_judge(snippet: str, llm_extracted: dict, golden_item: dict) -> dict:
    judge_prompt = f"""You are an evaluator checking the quality of an information extraction result.
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
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        )
        text = response.choices[0].message.content
        score = re.search(r"Score:\s*(\d)", text)
        reason = re.search(r"Reason:\s*(.+)", text)
        return {
            "judge_score": int(score.group(1)) if score else 0,
            "judge_reason": reason.group(1).strip() if reason else text.strip()
        }
    except Exception as e:
        return {"judge_score": 0, "judge_reason": f"Judge API error: {e}"}

# Step 8: Execution Engine

STRATEGIES = {
    "zero_shot": (zero_shot_prompt, parse_response),
    "few_shot": (few_shot_prompt, parse_response),
    "structured_prompt": (structured_prompt, parse_response),
    "cot_prompt": (cot_prompt, parse_response)
}

def run_prompt_strategy(prompt_name: str, snippets: list, golden_data: dict) -> list:
    build_prompt, parse_func = STRATEGIES[prompt_name]
    results = []

    print(f"\n{'='*50}")
    print(f"Running strategy: {prompt_name}")
    print(f"{'='*50}")

    for idx, item in enumerate(snippets):
        item_id = item.get("id", idx + 1)
        snippet = item.get("snippet", "")
        golden_item = golden_data.get(item_id, {})

        prompt = build_prompt(snippet)
        llm_out = call_llm(prompt)

        extracted = parse_func(llm_out["text"])
        accuracy = check_accuracy(extracted, golden_item)
        judge_out = call_llm_judge(snippet, extracted, golden_item)

        cost = round(
            (llm_out["input_tokens"] / 1000 * COST_PER_1K_INPUT_TOKENS) +
            (llm_out["output_tokens"] / 1000 * COST_PER_1K_OUTPUT_TOKENS), 6
        )

        print(
            f"[{item_id}] accuracy = {accuracy['accuracy_score']:.2f} | "
            f"judge = {judge_out['judge_score']}/5 | "
            f"latency = {llm_out['latency_in_sec']}s | "
            f"cost = ${cost:.6f}"
        )

        results.append({
            "id": item_id,
            "strategy": prompt_name,
            "company_extracted": extracted.get("company"), 
            "role_extracted": extracted.get("role"), 
            "yoe_extracted": extracted.get("years_experience_required"),
            "company_golden": golden_item.get("company"), 
            "role_golden": golden_item.get("role"), 
            "yoe_golden": golden_item.get("years_experience_required"),
            "company_match": accuracy["company_match"], 
            "role_match": accuracy["role_match"], 
            "yoe_match": accuracy["experience_match"],
            "accuracy": accuracy["accuracy_score"],
            "latency": llm_out["latency_in_sec"],
            "input_tokens": llm_out["input_tokens"], 
            "output_tokens": llm_out["output_tokens"],
            "cost": cost,
            "judge_score": judge_out["judge_score"],
            "judge_reason": judge_out["judge_reason"]
        })
    return results

def run_all_strategies(snippets: list, golden_data: dict) -> pd.DataFrame:
    all_results = []
    for name in STRATEGIES:
        strategy_res = run_prompt_strategy(name, snippets, golden_data)
        all_results.extend(strategy_res)
    return pd.DataFrame(all_results)

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("strategy").agg({
        "accuracy": "mean",
        "judge_score": "mean",
        "latency": "mean",
        "cost": "sum"
    }).reset_index()
    return summary

def load_data_in_csv(df: pd.DataFrame, summary: pd.DataFrame):
    df.to_csv("results_details.csv", index=False)  
    summary.to_csv("results.summary.csv", index=False)
    print("Saved details... Please find the data loaded in results_details.csv and results.summary.csv") 

# Entry Point
if __name__ == "__main__":
    snippets, golden = load_data()
    golden_data = {g.get("id", i + 1): g for i, g in enumerate(golden)}

    print(f"Snippets loaded: {len(snippets)}")
    print(f"Golden records loaded: {len(golden_data)}")

    df = run_all_strategies(snippets, golden_data)

    print(f"\n{'='*50}")
    print("FINAL STRATEGY PERFORMANCE SUMMARY")
    print(f"{'='*50}")

    summary = build_summary(df)
    print(summary.to_string(index=False))
    load_data_in_csv(df, summary)