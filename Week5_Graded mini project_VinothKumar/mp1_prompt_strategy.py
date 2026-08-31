"""
Main pipeline execution script.
"""

import pandas as pd
from prompts import (
    zero_shot_prompt,
    few_shot_prompt,
    structured_prompt,
    cot_prompt
)
from mp1_utils import (
    init_client,
    load_data,
    call_llm,
    parse_response,
    check_accuracy,
    call_llm_judge,
    compute_cost,
    export_results
)

# Strategy mappings
STRATEGIES = {
    "zero_shot": (zero_shot_prompt, parse_response),
    "few_shot": (few_shot_prompt, parse_response),
    "structured_prompt": (structured_prompt, parse_response),
    "cot_prompt": (cot_prompt, parse_response)
}

def run_prompt_strategy(client, prompt_name: str, snippets: list, golden_data: dict) -> list:
    """Run a single strategy across all dataset snippets."""
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
        llm_out = call_llm(client, prompt)

        extracted = parse_func(llm_out["text"])
        accuracy = check_accuracy(extracted, golden_item)
        judge_out = call_llm_judge(client, snippet, extracted, golden_item)
        cost = compute_cost(llm_out["input_tokens"], llm_out["output_tokens"])

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

def run_all_strategies(client, snippets: list, golden_data: dict) -> pd.DataFrame:
    """Run all strategy evaluations sequentially."""
    all_results = []
    for name in STRATEGIES:
        strategy_res = run_prompt_strategy(client, name, snippets, golden_data)
        all_results.extend(strategy_res)
    return pd.DataFrame(all_results)

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-strategy performance metrics."""
    return df.groupby("strategy").agg({
        "accuracy": "mean",
        "judge_score": "mean",
        "latency": "mean",
        "cost": "sum"
    }).reset_index()

if __name__ == "__main__":
    client = init_client()
    snippets, golden_data = load_data()

    print(f"Snippets loaded: {len(snippets)}")
    print(f"Golden records loaded: {len(golden_data)}")

    df = run_all_strategies(client, snippets, golden_data)

    print(f"\n{'='*50}")
    print("FINAL STRATEGY PERFORMANCE SUMMARY")
    print(f"{'='*50}")

    summary = build_summary(df)
    print(summary.to_string(index=False))
    
    export_results(df, summary)