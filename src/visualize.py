"""
Visualization script for Clinical LLM Consistency Study.
Generates all publication-ready figures for the paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# ── Load master summary ──────────────────────────────────────────────────────

df = pd.read_csv("results/summary/master_summary.csv")

MODELS = ["phi3_mini", "llama3.2", "gemma2", "mistral", "meditron"]
DATASETS = ["medqa", "medmcqa", "pubmedqa"]
STYLES = ["original", "formal", "simplified", "roleplay", "direct"]

MODEL_LABELS = {
    "phi3_mini": "Phi-3 Mini\n(3.8B)",
    "llama3.2":  "Llama 3.2\n(3B)",
    "gemma2":    "Gemma 2\n(2B)",
    "mistral":   "Mistral\n(7B)",
    "meditron":  "Meditron\n(7B)*"
}

DATASET_LABELS = {
    "medqa":    "MedQA\n(USMLE)",
    "medmcqa":  "MedMCQA\n(AIIMS)",
    "pubmedqa": "PubMedQA"
}

COLORS = {
    "phi3_mini": "#4C72B0",
    "llama3.2":  "#DD8452",
    "gemma2":    "#55A868",
    "mistral":   "#C44E52",
    "meditron":  "#8172B2"
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150
})


# ── Figure 1: Consistency heatmap ────────────────────────────────────────────

def fig1_consistency_heatmap():
    matrix = []
    for model in MODELS:
        row = []
        for dataset in DATASETS:
            val = df[(df.model == model) & (df.dataset == dataset)
                     ]["mean_consistency"].values[0]
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_yticks(range(len(MODELS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])

    for i in range(len(MODELS)):
        for j in range(len(DATASETS)):
            ax.text(j, i, f"{matrix[i,j]:.3f}",
                    ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean Consistency Score")
    ax.set_title("Figure 1: Mean Consistency Scores Across Models and Datasets",
                 fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("figures/fig1_consistency_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig1_consistency_heatmap.png")


# ── Figure 2: Accuracy heatmap ───────────────────────────────────────────────

def fig2_accuracy_heatmap():
    matrix = []
    for model in MODELS:
        row = []
        for dataset in DATASETS:
            val = df[(df.model == model) & (df.dataset == dataset)
                     ]["overall_accuracy"].values[0]
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=20, vmax=70)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_yticks(range(len(MODELS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])

    for i in range(len(MODELS)):
        for j in range(len(DATASETS)):
            ax.text(j, i, f"{matrix[i,j]:.1f}%",
                    ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Overall Accuracy (%)")
    ax.set_title("Figure 2: Overall Accuracy Across Models and Datasets",
                 fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("figures/fig2_accuracy_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig2_accuracy_heatmap.png")


# ── Figure 3: Consistency vs Accuracy scatter ────────────────────────────────

def fig3_consistency_vs_accuracy():
    fig, (ax_main, ax_inset) = plt.subplots(1, 2, 
        figsize=(16, 7), 
        gridspec_kw={'width_ratios': [3, 1]})

    for model in MODELS:
        mdf = df[df.model == model]
        
        # Main plot — exclude Meditron PubMedQA outlier
        for _, row in mdf.iterrows():
            if model == "meditron" and row["dataset"] == "pubmedqa":
                continue
            ax_main.scatter(row["mean_consistency"], 
                          row["overall_accuracy"],
                          color=COLORS[model], s=150, zorder=5)
            ax_main.annotate(
                DATASET_LABELS[row["dataset"]].replace("\n", " "),
                (row["mean_consistency"], row["overall_accuracy"]),
                textcoords="offset points", xytext=(8, 5),
                fontsize=8, color=COLORS[model])

        # Inset plot — only Meditron
        if model == "meditron":
            for _, row in mdf.iterrows():
                ax_inset.scatter(row["mean_consistency"],
                               row["overall_accuracy"],
                               color=COLORS[model], s=150, zorder=5)
                ax_inset.annotate(
                    DATASET_LABELS[row["dataset"]].replace("\n"," "),
                    (row["mean_consistency"], row["overall_accuracy"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color=COLORS[model])

    # Main plot formatting
    ax_main.set_xlabel("Mean Consistency Score", fontsize=12)
    ax_main.set_ylabel("Overall Accuracy (%)", fontsize=12)
    ax_main.set_title("Instruction-Tuned Models", fontweight="bold")
    ax_main.axvline(x=0.8, color="gray", linestyle="--", alpha=0.5)
    ax_main.set_xlim(0.65, 0.95)
    ax_main.set_ylim(25, 70)
    ax_main.grid(True, alpha=0.3)

    # Add legend to main plot
    handles = [plt.scatter([], [], color=COLORS[m], s=100,
               label=MODEL_LABELS[m].replace("\n", " "))
               for m in MODELS]
    ax_main.legend(handles=handles, loc="upper left", fontsize=9)

    # Inset plot formatting
    ax_inset.set_xlabel("Mean Consistency Score", fontsize=10)
    ax_inset.set_ylabel("Overall Accuracy (%)", fontsize=10)
    ax_inset.set_title("Meditron (7B)*\n(Not Instruction-Tuned)",
                       fontweight="bold", fontsize=10)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.set_xlim(-0.05, 0.9)
    ax_inset.set_ylim(-2, 40)

    # Add note
    ax_inset.text(0.05, -1.5,
                  "* Near-complete instruction\n"
                  "  following failure on PubMedQA\n"
                  "  (99% UNKNOWN rate)",
                  fontsize=7, color="#8172B2",
                  style="italic")

    fig.suptitle("Figure 3: Consistency vs Accuracy — Are They Correlated?",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/fig3_consistency_vs_accuracy.png",
                bbox_inches="tight")
    plt.close()
    print("  Saved fig3_consistency_vs_accuracy.png")


# ── Figure 4: Accuracy by prompt style ──────────────────────────────────────

def fig4_accuracy_by_style():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, dataset in zip(axes, DATASETS):
        x = np.arange(len(STYLES))
        width = 0.2

        for i, model in enumerate(MODELS):
            mdf = df[(df.model == model) & (df.dataset == dataset)]
            vals = [mdf[f"acc_{s}"].values[0] for s in STYLES]
            ax.bar(x + i * width, vals, width,
                   label=MODEL_LABELS[model].replace("\n", " "),
                   color=COLORS[model], alpha=0.85)

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([s.capitalize() for s in STYLES],
                           rotation=20, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 80)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.suptitle("Figure 4: Accuracy by Prompt Style Across Models and Datasets",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/fig4_accuracy_by_style.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig4_accuracy_by_style.png")


# ── Figure 5: Unknown rate comparison ───────────────────────────────────────

def fig5_unknown_rate():
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DATASETS))
    width = 0.2

    for i, model in enumerate(MODELS):
        vals = []
        for dataset in DATASETS:
            val = df[(df.model == model) & (df.dataset == dataset)
                     ]["unknown_rate"].values[0]
            vals.append(val)
        ax.bar(x + i * width, vals, width,
               label=MODEL_LABELS[model].replace("\n", " "),
               color=COLORS[model], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylabel("Unknown Rate (%)")
    ax.set_title("Figure 5: Instruction-Following Failure Rate (Unknown Responses)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("figures/fig5_unknown_rate.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig5_unknown_rate.png")


# ── Figure 6: Fully consistent questions ────────────────────────────────────

def fig6_fully_consistent():
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DATASETS))
    width = 0.2

    for i, model in enumerate(MODELS):
        vals = []
        for dataset in DATASETS:
            val = df[(df.model == model) & (df.dataset == dataset)
                     ]["fully_consistent_pct"].values[0]
            vals.append(val)
        ax.bar(x + i * width, vals, width,
               label=MODEL_LABELS[model].replace("\n", " "),
               color=COLORS[model], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylabel("Fully Consistent Questions (%)")
    ax.set_title("Figure 6: Percentage of Questions with Perfect Consistency",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("figures/fig6_fully_consistent.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig6_fully_consistent.png")
    

# ── Figure 7: Consistency score distribution (box plots) ─────────────────────

def fig7_consistency_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, dataset in zip(axes, DATASETS):
        data_to_plot = []
        labels = []

        for model in MODELS:
            scored_file = f"results/scored/{dataset}_{model}.csv"
            sdf = pd.read_csv(scored_file)
            data_to_plot.append(sdf["consistency_score"].values)
            labels.append(MODEL_LABELS[model].replace("\n", " "))

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))

        for patch, model in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.7)

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_ylabel("Consistency Score")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Figure 7: Distribution of Consistency Scores Across Models and Datasets",
        fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/fig7_consistency_distribution.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig7_consistency_distribution.png")


# ── Figure 8: Roleplay vs best style performance gap ────────────────────────

def fig8_roleplay_gap():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, dataset in zip(axes, DATASETS):
        models_list = []
        roleplay_acc = []
        best_acc = []
        gaps = []

        for model in MODELS:
            mdf = df[(df.model == model) & (df.dataset == dataset)]
            rp = mdf["acc_roleplay"].values[0]
            best = max(
                mdf["acc_original"].values[0],
                mdf["acc_formal"].values[0],
                mdf["acc_simplified"].values[0],
                mdf["acc_direct"].values[0]
            )
            models_list.append(MODEL_LABELS[model].replace("\n", " "))
            roleplay_acc.append(rp)
            best_acc.append(best)
            gaps.append(best - rp)

        x = np.arange(len(MODELS))
        width = 0.35

        ax.bar(x - width/2, best_acc, width,
               label="Best non-roleplay style",
               color="#4C72B0", alpha=0.85)
        ax.bar(x + width/2, roleplay_acc, width,
               label="Roleplay style",
               color="#C44E52", alpha=0.85)

        # annotate gap
        for i, (b, r, g) in enumerate(zip(best_acc, roleplay_acc, gaps)):
            ax.annotate(f"↓{g:.1f}%",
                        xy=(x[i] + width/2, r),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", fontsize=9,
                        color="#C44E52", fontweight="bold")

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, rotation=15, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 80)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Figure 8: Roleplay Prompt Accuracy vs Best Performing Style",
        fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/fig8_roleplay_gap.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig8_roleplay_gap.png")


# ── Run all figures ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    fig1_consistency_heatmap()
    fig2_accuracy_heatmap()
    fig3_consistency_vs_accuracy()
    fig4_accuracy_by_style()
    fig5_unknown_rate()
    fig6_fully_consistent()
    fig7_consistency_distribution()
    fig8_roleplay_gap()
    print("\n✅ All figures saved to figures/")
    print("\n✅ All figures saved to figures/")