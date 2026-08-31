# Prompting stategies (Mini-Project 1)

This project evaluates four distinct prompting strategies using `gpt-3.5-turbo` for structured information extraction from raw job postings. Results are benchmarked against a golden reference dataset using exact field-matching and an LLM-as-a-Judge (`gpt-4o`).

---

## 1. Project Architecture

* **`mp1_prompting_strategies.py`**: Main execution engine running all 4 strategies, evaluation, and CSV exports.
* **`data/job_snippets.jsonl`**: Input job posting text snippets.
* **`data/golden_set.jsonl`**: Ground truth extraction records.
* **`results_details.csv`**: Record-level output containing extractions, exact matches, latencies, and token costs.
* **`results.summary.csv`**: Aggregated performance metrics grouped by strategy.

---

## 2. Setup Instructions

### Prerequisites
* Python 3.9+ installed
* OpenAI API credentials

### Step-by-Step Setup

1. **Clone repository and setup environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install pandas openai python-dotenv

2. **Script Execution Flow**

* Loads job snippets and ground-truth golden dataset.

* Runs all 4 strategies sequentially (Zero-Shot, Few-Shot, Structured, Chain-of-Thought).

* Parses JSON responses dynamically and normalizes key variations.

* Computes field-level accuracy and calls gpt-4o as an evaluator.

* Displays a live summary table in the terminal and exports full logs to CSV.   