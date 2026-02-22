"""
Dataset loader for clinical LLM consistency study.
Loads MedQA, MedMCQA, and PubMedQA from HuggingFace datasets.
"""

from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def load_medqa(n_samples=200):
    """
    MedQA - USMLE style 4-option MCQ
    """
    print("Loading MedQA...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    df = pd.DataFrame(dataset["test"])
    df = df.sample(n=min(n_samples, len(df)), random_state=42)
    print(f"  Loaded {len(df)} samples")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample:\n{df.iloc[0]}\n")
    return df

def load_medmcqa(n_samples=200):
    """
    MedMCQA - Indian medical entrance exam (AIIMS/NEET)
    """
    print("Loading MedMCQA...")
    dataset = load_dataset("medmcqa")
    df = pd.DataFrame(dataset["validation"])
    df = df.sample(n=min(n_samples, len(df)), random_state=42)
    print(f"  Loaded {len(df)} samples")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample:\n{df.iloc[0]}\n")
    return df

def load_pubmedqa(n_samples=200):
    """
    PubMedQA - Biomedical research QA (yes/no/maybe)
    """
    print("Loading PubMedQA...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    df = pd.DataFrame(dataset["train"])
    df = df.sample(n=min(n_samples, len(df)), random_state=42)
    print(f"  Loaded {len(df)} samples")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample:\n{df.iloc[0]}\n")
    return df

if __name__ == "__main__":
    medqa_df = load_medqa()
    medmcqa_df = load_medmcqa()
    pubmedqa_df = load_pubmedqa()

    medqa_df.to_csv("data/processed/medqa_sample.csv", index=False)
    medmcqa_df.to_csv("data/processed/medmcqa_sample.csv", index=False)
    pubmedqa_df.to_csv("data/processed/pubmedqa_sample.csv", index=False)

    print("All datasets loaded and saved to data/processed/")
    print(f"  MedQA:     {len(medqa_df)} samples")
    print(f"  MedMCQA:   {len(medmcqa_df)} samples")
    print(f"  PubMedQA:  {len(pubmedqa_df)} samples")