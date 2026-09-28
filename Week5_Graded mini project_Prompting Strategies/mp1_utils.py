"""
Utility functions for environment initialization, data I/O, API calls, parsing, scoring, and data export.
"""

import json
import time
import re
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from prompts import judge_prompt_template

# Constants & Setup
MODEL = "gpt-3.5-turbo"
JUDGE_MODEL = "gpt-4o-2024-05-13"
COST_PER_1K_INPUT_TOKENS = 0.0005
COST_PER_1K_OUTPUT_TOKENS = 0.0015

def init_client() -> OpenAI:
    """Initialize and return the OpenAI client."""
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    print("Initial setup complete...!")
    print(f"Model: {MODEL} | Judge Model: {JUDGE_MODEL}")
    return client

def load_data(snippets_path="data/job_snippets.jsonl", goldendata_path="data/golden_set.jsonl") -> tuple[list, dict]:
    """Load snippets list and golden records mapped by ID."""
    with open(snippets_path, "r", encoding="utf-8") as f:
        snippets = [json.loads(line.strip()) for line in f if line.strip()]

    with open(goldendata_path, "r", encoding="utf-8") as f:
        golden = [json.loads(line.strip()) for line in f if line.strip()]

    golden_data = {g.get("id", i + 1): g for i, g in enumerate(golden)}
    return snippets, golden_data

def call_llm(client: OpenAI, prompt: str) -> dict:
    """Call completion endpoint and return output with usage stats."""
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

def parse_response(text: str) -> dict:
    """Parse JSON string output cleanly with field normalization."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
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

def normalise(value) -> str:
    """Normalize string values for exact-matching."""
    return str(value).lower().strip() if value is not None else ""

def check_accuracy(llm_extracted: dict, golden_item: dict) -> dict:
    """Compare extracted dictionary against ground truth."""
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

def call_llm_judge(client: OpenAI, snippet: str, llm_extracted: dict, golden_item: dict) -> dict:
    """Invoke LLM judge to evaluate semantic extraction quality."""
    prompt = judge_prompt_template(snippet, llm_extracted, golden_item)
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
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

def compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Compute prompt run cost based on token counters."""
    return round(
        (input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS) +
        (output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS), 6
    )

def export_results(df: pd.DataFrame, summary: pd.DataFrame):
    """Save detailed and summarized result DataFrames to CSV."""
    df.to_csv("results_details.csv", index=False)
    summary.to_csv("results.summary.csv", index=False)
    print("\nFiles saved successfully: 'results_details.csv' and 'results.summary.csv'")