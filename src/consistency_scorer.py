"""
Consistency Scorer for Clinical LLM Consistency Study.

For each question, calculates:
- Consistency Score: proportion of prompt variations that agree
- Majority Answer: most common answer across variations
- Is Accurate: whether majority answer matches ground truth
- Per-style accuracy: which prompt style is most reliable
"""

import json
import os
import pandas as pd
import numpy as np
from collections import Counter

os.makedirs("results/scored", exist_ok=True)
os.makedirs("results/summary", exist_ok=True)

PROMPT_STYLES = ["original", "formal", "simplified", "roleplay", "direct"]


def consistency_score(responses: dict) -> float:
    """
    Given a dict of {style: extracted_answer},
    returns the proportion of responses that match the majority answer.

    Example:
      {"original": "A", "formal": "B", "simplified": "B",
       "roleplay": "B", "direct": "A"}
      majority = "B" (3 votes), score = 3/5 = 0.6
    """
    answers = [v for v in responses.values() if v != "UNKNOWN"]
    if not answers:
        return 0.0
    majority = Counter(answers).most_common(1)[0][0]
    return sum(1 for a in answers if a == majority) / len(responses)


def majority_answer(responses: dict) -> str:
    """Returns the most common answer across all prompt styles."""
    answers = [v for v in responses.values() if v != "UNKNOWN"]
    if not answers:
        return "UNKNOWN"
    return Counter(answers).most_common(1)[0][0]


def unknown_rate(responses: dict) -> float:
    """Proportion of responses that were UNKNOWN."""
    unknowns = sum(1 for v in responses.values() if v == "UNKNOWN")
    return unknowns / len(responses)


def score_dataset(dataset_name: str, model_key: str) -> pd.DataFrame:
    """
    Load raw inference results and compute consistency metrics
    for every question. Returns a scored DataFrame.
    """
    input_file = f"results/raw/{dataset_name}_{model_key}.json"

    with open(input_file) as f:
        data = json.load(f)

    records = []

    for item in data:
        # extract just the answer letter/word per style
        responses = {
            style: item["responses"][style]["extracted"]
            for style in PROMPT_STYLES
            if style in item["responses"]
        }

        maj = majority_answer(responses)
        cons_score = consistency_score(responses)
        unk_rate = unknown_rate(responses)
        correct = item["correct_answer"].strip().upper()

        # is the majority answer correct?
        is_accurate = (maj.upper() == correct)

        record = {
            "id": item["id"],
            "question": item["question"][:80],
            "correct_answer": correct,
            "majority_answer": maj,
            "is_accurate": is_accurate,
            "consistency_score": round(cons_score, 3),
            "unknown_rate": round(unk_rate, 3),
            "dataset": dataset_name,
            "model": model_key,
        }

        # add per-style answers and correctness
        for style in PROMPT_STYLES:
            ans = responses.get(style, "UNKNOWN")
            record[f"ans_{style}"] = ans
            record[f"correct_{style}"] = (ans.upper() == correct)

        records.append(record)

    df = pd.DataFrame(records)

    # save scored results
    out_file = f"results/scored/{dataset_name}_{model_key}.csv"
    df.to_csv(out_file, index=False)
    print(f"  Saved scored results to {out_file}")

    return df


def summarize(df: pd.DataFrame, dataset_name: str, model_key: str) -> dict:
    """
    Compute summary statistics from scored DataFrame.
    This becomes your results tables in the paper.
    """
    summary = {
        "dataset": dataset_name,
        "model": model_key,
        "n_questions": len(df),

        # consistency metrics
        "mean_consistency": round(df["consistency_score"].mean(), 3),
        "std_consistency": round(df["consistency_score"].std(), 3),
        "fully_consistent": int((df["consistency_score"] == 1.0).sum()),
        "fully_consistent_pct": round(
            (df["consistency_score"] == 1.0).mean() * 100, 1),

        # accuracy metrics
        "overall_accuracy": round(df["is_accurate"].mean() * 100, 1),
        "unknown_rate": round(df["unknown_rate"].mean() * 100, 1),

        # per style accuracy
        **{f"acc_{s}": round(df[f"correct_{s}"].mean() * 100, 1)
           for s in PROMPT_STYLES}
    }
    return summary


if __name__ == "__main__":
    datasets = ["medqa", "medmcqa", "pubmedqa"]
    models = ["phi3_mini","llama3.2"]

    all_summaries = []

    for model_key in models:
        print(f"\n=== Scoring {model_key} ===")
        for dataset_name in datasets:
            print(f"\n  Dataset: {dataset_name}")
            df = score_dataset(dataset_name, model_key)
            summary = summarize(df, dataset_name, model_key)
            all_summaries.append(summary)

            # print key metrics immediately
            print(f"    Mean Consistency:     {summary['mean_consistency']}")
            print(f"    Fully Consistent:     {summary['fully_consistent_pct']}%")
            print(f"    Overall Accuracy:     {summary['overall_accuracy']}%")
            print(f"    Unknown Rate:         {summary['unknown_rate']}%")
            print(f"    Accuracy by style:")
            for s in PROMPT_STYLES:
                print(f"      {s:12}: {summary[f'acc_{s}']}%")

    # save master summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("results/summary/master_summary.csv", index=False)
    print("\n✅ Master summary saved to results/summary/master_summary.csv")
    print("\n=== MASTER SUMMARY ===")
    print(summary_df.to_string(index=False))