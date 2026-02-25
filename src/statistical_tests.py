"""
Statistical significance tests for consistency and accuracy differences.
Uses Wilcoxon signed-rank test for paired comparisons between models.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs("results/stats", exist_ok=True)

MODELS = ["phi3_mini", "llama3.2", "gemma2", "mistral"]
DATASETS = ["medqa", "medmcqa", "pubmedqa"]

def load_scored(dataset, model):
    return pd.read_csv(f"results/scored/{dataset}_{model}.csv")

def run_tests():
    results = []

    for dataset in DATASETS:
        print(f"\n=== {dataset.upper()} ===")

        scores = {}
        for model in MODELS:
            df = load_scored(dataset, model)
            scores[model] = df

        print("\nConsistency Score Comparisons (Wilcoxon signed-rank test):")
        pairs = [
            ("llama3.2", "phi3_mini"),
            ("gemma2", "phi3_mini"),
            ("mistral", "phi3_mini"),
            ("llama3.2", "gemma2"),
        ]

        for m1, m2 in pairs:
            s1 = scores[m1]["consistency_score"].values
            s2 = scores[m2]["consistency_score"].values
            stat, p = stats.wilcoxon(s1, s2)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else \
                  "*" if p < 0.05 else "ns"
            print(f"  {m1} vs {m2}: W={stat:.1f}, p={p:.4f} {sig}")

            results.append({
                "dataset": dataset,
                "model1": m1,
                "model2": m2,
                "metric": "consistency",
                "statistic": round(stat, 3),
                "p_value": round(p, 4),
                "significance": sig
            })

        print("\nAccuracy Comparisons (McNemar test):")
        for m1, m2 in pairs:
            a1 = scores[m1]["is_accurate"].astype(int).values
            a2 = scores[m2]["is_accurate"].astype(int).values

            m1_only = np.sum((a1 == 1) & (a2 == 0))
            m2_only = np.sum((a1 == 0) & (a2 == 1))

            if m1_only + m2_only > 0:
                chi2 = (abs(m1_only - m2_only) - 1)**2 / \
                       (m1_only + m2_only)
                p = stats.chi2.sf(chi2, df=1)
            else:
                chi2, p = 0, 1.0

            sig = "***" if p < 0.001 else "**" if p < 0.01 else \
                  "*" if p < 0.05 else "ns"
            print(f"  {m1} vs {m2}: chi2={chi2:.3f}, p={p:.4f} {sig}")

            results.append({
                "dataset": dataset,
                "model1": m1,
                "model2": m2,
                "metric": "accuracy",
                "statistic": round(chi2, 3),
                "p_value": round(p, 4),
                "significance": sig
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/stats/significance_tests.csv", index=False)
    print("\n✅ Saved to results/stats/significance_tests.csv")
    print("\nSignificance key: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")

if __name__ == "__main__":
    run_tests()