"""
Inference Engine for Clinical LLM Consistency Study.

For each question, runs all 5 prompt variations through the model
and collects answers. Saves results for consistency scoring.
"""

import json
import os
import time
import re
import requests
import pandas as pd
from tqdm import tqdm

os.makedirs("results/raw", exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = {
    "phi3_mini": "phi3:mini",
    "llama3.2": "llama3.2:3b",
    "gemma2":"gemma2:2b",
    "mistral":"mistral:7b"
}

def query_ollama(model: str, prompt: str, timeout: int = 60) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,      # deterministic — important for research
            "num_predict": 10,     # we only need a letter or yes/no/maybe
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def extract_answer_mcq(raw: str) -> str:
    """
    Extract A/B/C/D from model response.
    Handles cases like 'A', 'A.', 'The answer is A', 'Option A', etc.
    """
    raw = raw.upper().strip()
    # look for standalone A B C or D
    match = re.search(r'\b([ABCD])\b', raw)
    if match:
        return match.group(1)
    # fallback — first character if it's a letter
    if raw and raw[0] in "ABCD":
        return raw[0]
    return "UNKNOWN"


def extract_answer_pubmedqa(raw: str) -> str:
    """
    Extract yes/no/maybe from model response.
    """
    raw = raw.lower().strip()
    if "yes" in raw:
        return "yes"
    if "no" in raw:
        return "no"
    if "maybe" in raw:
        return "maybe"
    return "UNKNOWN"


def run_inference(dataset_name: str, model_key: str, model_name: str,
                  max_samples: int = None):
    """
    Run all 5 prompt variations for each question through the model.
    Saves results to results/raw/{dataset}_{model}.json
    """
    prompt_file = f"data/prompts/{dataset_name}_prompts.json"
    output_file = f"results/raw/{dataset_name}_{model_key}.json"

    # skip if already done
    if os.path.exists(output_file):
        print(f"  Already exists: {output_file} — skipping")
        return

    with open(prompt_file) as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    is_pubmedqa = dataset_name == "pubmedqa"
    results = []

    print(f"\nRunning {model_key} on {dataset_name} ({len(data)} questions)...")

    for item in tqdm(data):
        record = {
            "id": item["id"],
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "responses": {}
        }

        for style, prompt in item["prompts"].items():
            raw = query_ollama(model_name, prompt)
            if is_pubmedqa:
                answer = extract_answer_pubmedqa(raw)
            else:
                answer = extract_answer_mcq(raw)

            record["responses"][style] = {
                "raw": raw,
                "extracted": answer
            }
            time.sleep(0.1)  # small pause to avoid overwhelming Ollama

        results.append(record)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✅ Saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    # First run: phi3_mini on all 3 datasets
    # We use max_samples=10 first to verify pipeline works
    # Then remove max_samples to run full 200

    # for dataset in ["medqa", "medmcqa", "pubmedqa"]:
    #     run_inference(
    #         dataset_name=dataset,
    #         model_key="phi3_mini",
    #         model_name=MODELS["phi3_mini"],
    #         # max_samples=10      # ← test run first, change to None for full run
    #         max_samples=None      # ← full run
    #     )
        
    for model_key, model_name in MODELS.items():
        for dataset in ["medqa", "medmcqa", "pubmedqa"]:
            run_inference(
                dataset_name=dataset,
                model_key=model_key,
                model_name=model_name,
                max_samples=None
            )

    print("\n✅ Test inference complete. Check results/raw/ for output.")