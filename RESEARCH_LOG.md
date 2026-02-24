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

### 02/25/2026 — Day 6
- Fixed MedQA correct answer format bug (was using full text, switched to answer_idx)
- Completed consistency scoring for phi3_mini across all 3 datasets
- Key findings:
  - Mean consistency: MedQA=0.698, MedMCQA=0.730, PubMedQA=0.830
  - Fully consistent questions: only 19.5-41% across datasets
  - Roleplay prompt consistently worst style across all datasets
  - Direct/original prompts most reliable
  - MedQA unknown rate 10.5% — instruction following failures
  - Accuracy ranges: 48-53% across datasets

### 02/26/2026 — Day 8
- Completed llama3.2 inference and scoring across all 3 datasets
- Key comparisons vs phi3_mini:
  - Llama3.2 better consistency on MedQA (0.776 vs 0.698)
  - Llama3.2 dramatically lower unknown rate on MedQA (0.8% vs 10.5%)
  - Llama3.2 higher accuracy on PubMedQA (65% vs 48%)
  - Roleplay no longer worst style for llama3.2 — finding is model specific
  - Key insight: consistency and accuracy are independent — a model can be
    consistently wrong, which is dangerous in clinical settings

### 02/27/2026 — Day 9
- Completed Mistral 7B inference and scoring
- All 4 models now complete across all 3 datasets
- Full experimental results ready for analysis and visualization

-------------------------------------------------------------------------

Key findings summary:
- Consistency ranking: Gemma2 > Mistral > Llama3.2 > Phi3_mini
- Accuracy ranking: Llama3.2 > Phi3_mini > Mistral > Gemma2
- No model wins on both metrics — consistency-accuracy tradeoff confirmed
- Roleplay prompt worst performing style across ALL 4 models
- Mistral unknown rate surprisingly high (4.7-7.2%) despite being largest model
- Gemma consistently wrong — most dangerous failure mode identified

-------------------------------------------------------------------------

## Literature Review

### Paper 1: CLEVER (JMIR AI, 2025)
**What they did:** Proposed a physician preference-based evaluation 
framework comparing GPT-4o against smaller medical LLMs on clinical 
tasks including summarization and QA.

**What they found:** Domain-specific medical LLMs are often preferred 
over larger general models on clinical tasks. Automated metrics alone 
are insufficient for clinical evaluation.

**How our work differs:** CLEVER evaluates output quality through human 
expert judgment on large models. Our work evaluates answer consistency 
through automated prompt variation testing on small CPU-runnable models 
specifically for low-resource deployment. We ask not "is the answer good" 
but "is the answer stable."

**Citation relevance:** Cite in related work to position our automated 
consistency framework as complementary to human preference evaluation.
-------------------------------------------------------------------------

### Paper 2: SLM Healthcare Survey (Garg et al.)
**Type:** Literature review — no new experiments

**What they did:** Surveyed existing work on small language models 
in healthcare. Defined SLMs as up to ~7B parameters. Covered clinical 
NLP tasks, optimization techniques, and deployment challenges.

**What they found:** SLMs are increasingly viable for low-resource 
clinical settings. Key challenges remain in evaluation, trustworthiness 
and reliability. Prompt sensitivity not investigated.

**How our work differs:** We run actual experiments. We directly measure 
prompt sensitivity and consistency — a gap this survey explicitly 
identifies as needed but doesn't address.

**Citation-ready sentence:** "While prior work reviews the architecture 
and task performance of small language models in clinical NLP, empirical 
investigation of answer stability under prompt variation remains absent. 
Our study fills this gap by introducing consistency as a reliability 
metric under prompt variation."

**Where to cite:** Introduction (background on SLMs in healthcare) 
and Related Work (to establish the gap).

-------------------------------------------------------------------------

### Paper 3: MedQA Dataset Paper
**Citation:** Jin et al., 2021. "What Disease does this Patient Have? 
A Large-scale Open Domain Question Answering Dataset from Medical Exams"

**What it is:** Original paper introducing the MedQA dataset derived 
from USMLE medical licensing exam questions. Contains 4-option MCQ 
questions covering clinical medicine.

**Why we cite it:** We use this dataset directly in our experiments. 
Any paper using MedQA must cite the original source.

**Where to cite:** Methods section — Dataset Description.

-------------------------------------------------------------------------
### Paper 4: MedMCQA Dataset Paper
**Citation:** Pal et al., 2022. "MedMCQA: A Large-scale Multi-Subject 
Multi-Choice Dataset for Medical Exam Comprehension"

**What it is:** Introduced the MedMCQA dataset from Indian medical 
entrance exams (AIIMS/NEET). Contains over 180,000 questions across 
medical subjects with subject labels.

**Why we cite it:** We use this dataset directly in our experiments.

**Where to cite:** Methods section — Dataset Description.

-------------------------------------------------------------------------

### Paper 5: PubMedQA Dataset Paper
**Citation:** Jin et al., 2019. "PubMedQA: A Dataset for Biomedical 
Research Question Answering"

**What it is:** Introduced PubMedQA — a dataset of yes/no/maybe 
questions derived from PubMed research abstracts. Tests reasoning 
from provided biomedical evidence rather than memorized knowledge.

**Why we cite it:** We use this dataset directly in our experiments.

**Where to cite:** Methods section — Dataset Description.

-------------------------------------------------------------------------

### Paper 6: Prompt Robustness Paper
**Citation:** Ngweta et al. "Towards LLMs Robustness to Changes 
in Prompt Format Styles" — RPI / IBM Research

**What they did:** Empirical study measuring prompt brittleness 
in LLMs using spread metric (max-min accuracy across prompt formats). 
Proposed Mixture of Formats (MOF) to reduce brittleness. Tested on 
general NLP benchmarks with large models (Falcon-11B, Llama-2-13B, 
Llama-3-70B).

**What they found:** LLMs show systematic sensitivity to prompt 
format changes. MOF reduces brittleness in most but not all cases. 
Sensitivity is model and task specific.

**How our work differs:**
- They test general NLP tasks — we test clinical QA specifically
- They use spread metric — we use majority-based consistency score
- They don't address clinical safety implications — we do
- They test large models only — we focus on sub-7B CPU-runnable models
- They propose mitigation (MOF) — we quantify the gap first

**Citation-ready sentence:** "While Ngweta et al. quantify prompt 
brittleness in general language tasks and propose mitigation strategies, 
the clinical safety implications of prompt sensitivity in small 
open-source models remain unexamined — a gap our study addresses."

**Where to cite:** Related Work — directly before introducing 
our consistency metric as a clinical extension of prompt brittleness.

-------------------------------------------------------------------------

### Paper 7: Medical Hallucinations Paper
**Citation:** Kim et al. 2025. "Medical Hallucinations in Foundation 
Models and Their Impact on Healthcare"

**What they did:** Defined taxonomy of medical hallucinations. 
Benchmarked general and medical LLMs on clinical tasks using 
physician annotations. Surveyed clinicians globally about 
hallucination experiences. Evaluated mitigation techniques.

**What they found:** Hallucinations occur frequently across all 
models including medically fine-tuned ones. Clinicians consider 
them capable of causing patient harm. Reasoning errors not just 
factual gaps drive hallucinations. Models present wrong information 
with high confidence.

**Key connection to our work:** Their warning about confident 
wrong outputs directly maps to our Gemma finding — highest 
consistency (0.888) but lowest accuracy (40%). A model that 
consistently gives wrong answers is more dangerous than one 
that shows uncertainty, because it hides its unreliability.

**Prompt sensitivity:** They do not study prompt variation — 
that gap is ours to fill.

**Citation-ready sentence:** "Kim et al. demonstrate that 
hallucinations occur frequently across foundation models and 
that clinicians consider confidently wrong outputs particularly 
dangerous — a risk compounded when models exhibit high 
consistency in incorrect answers across varied prompt formulations."

**Where to cite:** Introduction — to motivate why reliability 
and consistency matter beyond accuracy alone.

## Literature Review Status
- [x] CLEVER — JMIR AI 2025
- [x] SLM Healthcare Survey — Garg et al.
- [x] MedQA dataset paper — Jin et al. 2021
- [x] MedMCQA dataset paper — Pal et al. 2022
- [x] PubMedQA dataset paper — Jin et al. 2019
- [x] Prompt sensitivity — Ngweta et al.
- [x] Clinical LLM safety — Kim et al. 2025
✅ Literature review complete — ready to write the paper

7 papers covering every angle this paper needs:
target journal — CLEVER
background — SLM survey
datasets — MedQA, MedMCQA, PubMedQA papers
methodological precedent — Ngweta et al.
clinical motivation — Kim et al.
```
---

To do list:
```
✅ Day 1 — Environment setup
✅ Day 2 — Datasets loaded (600 questions)
✅ Day 3 — Prompt variations built (3000 prompts total)
✅ Day 4 — Model inference engine (feed prompts to LLMs, collect answers)
✅ Day 5 — Consistency scorer (calculate your core metric)
✅ Day 6 — Run first model (Phi-3 Mini — fastest on CPU)
✅ Day 7 — Analysis + visualizations