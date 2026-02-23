# Clinical LLM Consistency Study — Research Log

## Project Title
Prompt Sensitivity and Answer Consistency of Small Open-Source LLMs on Clinical 
Question Answering: Implications for Low-Resource Healthcare Deployment

## Target Journal
JMIR AI — https://ai.jmir.org

## Research Question
Do small open-source LLMs (≤7B parameters) produce consistent answers when the 
same clinical question is paraphrased in different ways, and are they reliable 
enough for low-resource healthcare deployment?

## Hypothesis
Small LLMs will show significant answer inconsistency across prompt variations, 
with consistency scores varying meaningfully across model families and sizes — 
suggesting that accuracy alone is insufficient to evaluate clinical LLM safety.

## Models We Will Evaluate
- Llama 3.2 3B (Meta)
- Phi-3 Mini 3.8B (Microsoft)
- Mistral 7B (Mistral AI)
- Gemma 2B (Google)

## Datasets
- MedQA (USMLE-style multiple choice)
- MedMCQA (Indian medical entrance exam)
- PubMedQA (biomedical research QA)

## Key Metrics
- Answer Consistency Score (same answer across N prompt variations)
- Accuracy (against ground truth)
- Prompt Sensitivity Index (how much answer changes with rephrasing)
- Confidence Proxy (answer entropy across variations)

---

## Log

### 02/20/2026 — Day 1
- Set up project folder structure
- Initialized virtual environment
- Installed dependencies
- Created GitHub repo
- Defined research question and hypothesis

### 02/21/2026 — Day 2
- Wrote dataset loader (src/load_datasets.py)
- Successfully loaded 3 datasets from HuggingFace:
  - MedQA (USMLE): 200 samples, columns: question, options, answer
  - MedMCQA (AIIMS/NEET): 200 samples, columns: question, opa/b/c/d, cop, subject
  - PubMedQA: 200 samples, columns: question, context, long_answer, final_decision
- Fixed bigbio/med_qa loading script error, switched to GBaker/MedQA-USMLE-4-options
- Data saved to data/processed/

### 02/22/2026 — Day 3
- Built prompt variation engine (src/prompt_variations.py)
- Generated 5 prompt styles per question: original, formal, simplified, roleplay, direct
- Output: 200 prompt sets per dataset = 1000 prompts per dataset = 3000 total prompts
- Saved to data/prompts/ as JSON files
- This is the core novelty of the paper — testing consistency across prompt styles


### 02/23/2026 — Day 4
- Built inference engine (src/inference.py)
- Uses Ollama with temperature=0 for deterministic outputs
- Test run: 10 questions per dataset, all 3 datasets completed
- Key early finding: Phi-3 Mini shows clear prompt sensitivity
  - MedQA Q1: answered A on original/roleplay, B on formal/simplified/direct
  - MedMCQA Q1: consistent on 4/5 styles but returned UNKNOWN on roleplay
  - PubMedQA Q1: split between maybe and no across styles, all incorrect
- Pipeline validated — ready for full 200 sample run

### 02/24/2026 — Day 5
- Completed full inference run for phi3_mini
- 200 questions × 5 prompt styles × 3 datasets = 3000 prompts processed
- Timing:
  - MedQA: 20 mins (6.2s/question)
  - MedMCQA: 11 mins (3.3s/question)
  - PubMedQA: 23 mins (6.9s/question)
- Results saved to results/raw/
- Ready for consistency scoring
```
---

Here's where we stand and what's coming next so you can see the full picture:
```
✅ Day 1 — Environment setup
✅ Day 2 — Datasets loaded (600 questions)
✅ Day 3 — Prompt variations built (3000 prompts total)
⬜ Day 4 — Model inference engine (feed prompts to LLMs, collect answers)
⬜ Day 5 — Consistency scorer (calculate your core metric)
⬜ Day 6 — Run first model (Phi-3 Mini — fastest on CPU)
⬜ Day 7 — Analysis + visualizations