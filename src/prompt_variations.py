"""
Prompt Variation Engine for Clinical LLM Consistency Study.

For each clinical question, we generate 5 prompt variations:
1. Original - question as-is
2. Formal - clinical/professional rephrasing
3. Simplified - plain language rephrasing
4. Roleplay - asking as a medical professional
5. Inverse - negation or opposite framing

This tests whether small LLMs give CONSISTENT answers regardless
of how the question is asked — a key clinical safety requirement.
"""

import pandas as pd
import json
import os

os.makedirs("data/prompts", exist_ok=True)


# ── MedQA prompt templates ──────────────────────────────────────────────────

def make_medqa_prompts(question: str, options: dict) -> dict:
    """
    Generate 5 prompt variations for a MedQA (USMLE) question.
    Options is a dict like {"A": "...", "B": "...", "C": "...", "D": "..."}
    """
    # Format options as a clean string
    opts = "\n".join([f"{k}: {v}" for k, v in options.items()])

    prompts = {

        "original": (
            f"Answer the following medical question by choosing the correct option.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{opts}\n\n"
            f"Answer with only the option letter (A, B, C, or D)."
        ),

        "formal": (
            f"You are a medical expert. Based on your clinical knowledge, "
            f"select the most appropriate answer to the following question.\n\n"
            f"Clinical Question: {question}\n\n"
            f"Choices:\n{opts}\n\n"
            f"Respond with only the letter of the correct choice (A, B, C, or D)."
        ),

        "simplified": (
            f"Read this medical question carefully and pick the best answer.\n\n"
            f"{question}\n\n"
            f"{opts}\n\n"
            f"Which letter is correct? Reply with just A, B, C, or D."
        ),

        "roleplay": (
            f"You are a physician taking a medical licensing exam. "
            f"Answer this question as you would on the exam.\n\n"
            f"Q: {question}\n\n"
            f"{opts}\n\n"
            f"Your answer (A, B, C, or D):"
        ),

        "direct": (
            f"Medical question: {question}\n\n"
            f"Option A: {options.get('A', '')}\n"
            f"Option B: {options.get('B', '')}\n"
            f"Option C: {options.get('C', '')}\n"
            f"Option D: {options.get('D', '')}\n\n"
            f"What is the correct option? State only the letter."
        ),
    }
    return prompts


# ── MedMCQA prompt templates ────────────────────────────────────────────────

def make_medmcqa_prompts(question: str, opa: str, opb: str,
                          opc: str, opd: str) -> dict:
    """
    Generate 5 prompt variations for a MedMCQA question.
    """
    opts = f"A: {opa}\nB: {opb}\nC: {opc}\nD: {opd}"

    prompts = {

        "original": (
            f"Answer the following medical question by choosing the correct option.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{opts}\n\n"
            f"Answer with only the option letter (A, B, C, or D)."
        ),

        "formal": (
            f"You are a medical expert. Select the most appropriate answer "
            f"to the following clinical question.\n\n"
            f"Question: {question}\n\n"
            f"Choices:\n{opts}\n\n"
            f"Respond with only the letter of the correct choice (A, B, C, or D)."
        ),

        "simplified": (
            f"Read this question and pick the best answer.\n\n"
            f"{question}\n\n"
            f"{opts}\n\n"
            f"Which letter is correct? Reply with just A, B, C, or D."
        ),

        "roleplay": (
            f"You are a doctor taking a medical entrance exam. "
            f"Answer this question as you would on the exam.\n\n"
            f"Q: {question}\n\n"
            f"{opts}\n\n"
            f"Your answer (A, B, C, or D):"
        ),

        "direct": (
            f"Medical question: {question}\n\n"
            f"Option A: {opa}\n"
            f"Option B: {opb}\n"
            f"Option C: {opc}\n"
            f"Option D: {opd}\n\n"
            f"What is the correct option? State only the letter."
        ),
    }
    return prompts


# ── PubMedQA prompt templates ───────────────────────────────────────────────

def make_pubmedqa_prompts(question: str, context: str) -> dict:
    """
    Generate 5 prompt variations for a PubMedQA question.
    Answer is yes / no / maybe.
    """
    # context is a dict with key 'contexts' (list of strings)
    if isinstance(context, dict):
        ctx_text = " ".join(context.get("contexts", []))[:1000]
    else:
        ctx_text = str(context)[:1000]

    prompts = {

        "original": (
            f"Based on the following research context, answer the question "
            f"with yes, no, or maybe.\n\n"
            f"Context: {ctx_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer with only: yes, no, or maybe."
        ),

        "formal": (
            f"You are a biomedical researcher. Based on the provided abstract, "
            f"determine whether the answer to the question is yes, no, or maybe.\n\n"
            f"Abstract: {ctx_text}\n\n"
            f"Research Question: {question}\n\n"
            f"Respond with only: yes, no, or maybe."
        ),

        "simplified": (
            f"Read the text below and answer the question.\n\n"
            f"Text: {ctx_text}\n\n"
            f"Question: {question}\n\n"
            f"Reply with only: yes, no, or maybe."
        ),

        "roleplay": (
            f"You are a doctor reviewing a research paper. "
            f"Based on this excerpt, answer the clinical question.\n\n"
            f"Excerpt: {ctx_text}\n\n"
            f"Question: {question}\n\n"
            f"Your answer (yes, no, or maybe):"
        ),

        "direct": (
            f"Context: {ctx_text}\n\n"
            f"Q: {question}\n"
            f"A (yes/no/maybe):"
        ),
    }
    return prompts


# ── Build full prompt dataset ───────────────────────────────────────────────

def build_prompt_dataset():
    """
    Load processed CSVs and generate all prompt variations.
    Saves a JSON file for each dataset.
    """

    # ── MedQA ──
    print("Building MedQA prompts...")
    medqa_df = pd.read_csv("data/processed/medqa_sample.csv")
    medqa_prompts = []

    for idx, row in medqa_df.iterrows():
        # options column is a string repr of dict — parse it
        try:
            options = json.loads(row["options"].replace("'", '"'))
        except Exception:
            options = {"A": "", "B": "", "C": "", "D": ""}

        entry = {
            "id": idx,
            "question": row["question"],
            "correct_answer": row["answer_idx"],
            "prompts": make_medqa_prompts(row["question"], options)
        }
        medqa_prompts.append(entry)

    with open("data/prompts/medqa_prompts.json", "w") as f:
        json.dump(medqa_prompts, f, indent=2)
    print(f"  Saved {len(medqa_prompts)} prompt sets for MedQA")

    # ── MedMCQA ──
    print("Building MedMCQA prompts...")
    medmcqa_df = pd.read_csv("data/processed/medmcqa_sample.csv")

    # Map numeric correct option to letter
    cop_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    medmcqa_prompts = []

    for idx, row in medmcqa_df.iterrows():
        entry = {
            "id": idx,
            "question": row["question"],
            "correct_answer": cop_map.get(int(row["cop"]), "A"),
            "subject": row.get("subject_name", ""),
            "prompts": make_medmcqa_prompts(
                row["question"],
                str(row["opa"]), str(row["opb"]),
                str(row["opc"]), str(row["opd"])
            )
        }
        medmcqa_prompts.append(entry)

    with open("data/prompts/medmcqa_prompts.json", "w") as f:
        json.dump(medmcqa_prompts, f, indent=2)
    print(f"  Saved {len(medmcqa_prompts)} prompt sets for MedMCQA")

    # ── PubMedQA ──
    print("Building PubMedQA prompts...")
    pubmedqa_df = pd.read_csv("data/processed/pubmedqa_sample.csv")
    pubmedqa_prompts = []

    for idx, row in pubmedqa_df.iterrows():
        # context was saved as string — parse it back
        try:
            context = json.loads(row["context"].replace("'", '"'))
        except Exception:
            context = {"contexts": [str(row["context"])]}

        entry = {
            "id": idx,
            "question": row["question"],
            "correct_answer": row["final_decision"],
            "prompts": make_pubmedqa_prompts(row["question"], context)
        }
        pubmedqa_prompts.append(entry)

    with open("data/prompts/pubmedqa_prompts.json", "w") as f:
        json.dump(pubmedqa_prompts, f, indent=2)
    print(f"  Saved {len(pubmedqa_prompts)} prompt sets for PubMedQA")

    print("\n✅ All prompt datasets built and saved to data/prompts/")


if __name__ == "__main__":
    build_prompt_dataset()