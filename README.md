# Prompt Sensitivity and Answer Consistency of Small Open-Source LLMs on Clinical Question Answering

## Overview
This repository contains the code, data processing scripts, and evaluation 
pipeline for the paper:

**"Prompt Sensitivity and Answer Consistency of Small Open-Source Large 
Language Models on Clinical Question Answering: Implications for 
Low-Resource Healthcare Deployment"**

> Shravani Hariprasad, Independent Researcher, 2026

📄 **arXiv preprint:** [link coming soon]  
📊 **Target journal:** JMIR AI

---

## Research Summary

This study systematically evaluates how small open-source language models 
(2B-7B parameters) respond to differently worded clinical questions. 
We introduce a **consistency score** metric that measures whether a model 
gives the same answer when the same question is phrased in five different 
ways — a critical safety requirement for clinical AI deployment.

### Key Findings
- Consistency and accuracy are **independent metrics** — models can be 
  highly consistent but consistently wrong (Gemma 2: 0.888 consistency, 
  33% accuracy)
- **Roleplay prompts** consistently underperform across all models and 
  datasets — persona-based framing reduces clinical reliability
- **Instruction following** varies independently of model size — larger 
  models do not guarantee safer outputs
- **Domain pretraining alone** (Meditron-7B) is insufficient for structured 
  clinical QA without instruction tuning

---

## Models Evaluated
| Model | Parameters | Developer | Type |
|-------|-----------|-----------|------|
| Phi-3 Mini | 3.8B | Microsoft | Instruction-tuned |
| Llama 3.2 | 3B | Meta | Instruction-tuned |
| Gemma 2 | 2B | Google | Instruction-tuned |
| Mistral 7B | 7B | Mistral AI | Instruction-tuned |
| Meditron-7B | 7B | EPFL | Domain-pretrained |

---

## Datasets
| Dataset | Source | Format | Samples |
|---------|--------|--------|---------|
| MedQA | USMLE licensing exam | 4-option MCQ | 200 |
| MedMCQA | AIIMS/NEET entrance exam | 4-option MCQ | 200 |
| PubMedQA | PubMed research abstracts | Yes/No/Maybe | 200 |

---

## Prompt Styles
Five semantically equivalent but stylistically distinct prompt formulations:
1. **Original** — question as-is from dataset
2. **Formal** — clinical academic language
3. **Simplified** — plain everyday language
4. **Roleplay** — model instructed to respond as a physician
5. **Direct** — bare question with minimal framing

---

## Project Structure
```
clinical-llm-eval/
├── src/
│   ├── load_datasets.py        # Load MedQA, MedMCQA, PubMedQA
│   ├── prompt_variations.py    # Generate 5 prompt styles per question
│   ├── inference.py            # Run models via Ollama
│   ├── consistency_scorer.py   # Calculate consistency and accuracy metrics
│   ├── statistical_tests.py    # Wilcoxon and McNemar significance tests
│   └── visualize.py            # Generate all figures
├── data/
│   ├── processed/              # Sampled datasets (CSV)
│   └── prompts/                # Generated prompt variations (JSON)
├── results/
│   ├── raw/                    # Model responses (JSON)
│   ├── scored/                 # Consistency scores (CSV)
│   ├── summary/                # Master summary (CSV)
│   └── stats/                  # Statistical test results (CSV)
├── figures/                    # Publication-ready figures (PNG)
├── paper/                      # LaTeX source files
└── RESEARCH_LOG.md             # Complete research diary
```

---

## Requirements
- Python 3.12+
- Ollama (for local model inference)
- See `requirements.txt` for Python dependencies

---

## Setup and Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/shravani-01/clinical-llm-eval.git
cd clinical-llm-eval
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Pull models via Ollama
```bash
ollama pull phi3:mini
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull mistral:7b
ollama pull meditron
```

### 4. Run the pipeline
```bash
# Load datasets
python src/load_datasets.py

# Generate prompt variations
python src/prompt_variations.py

# Run inference (runs all models)
python src/inference.py

# Score consistency
python src/consistency_scorer.py

# Statistical tests
python src/statistical_tests.py

# Generate figures
python src/visualize.py
```

---

## Results Summary

| Model | Mean Consistency | Overall Accuracy | UNKNOWN Rate |
|-------|-----------------|-----------------|--------------|
| Phi-3 Mini (3.8B) | 0.753 | 49.7% | 5.0% |
| Llama 3.2 (3B) | 0.786 | 56.5% | 2.0% |
| Gemma 2 (2B) | 0.861 | 38.8% | 1.4% |
| Mistral 7B | 0.812 | 44.3% | 5.5% |
| Meditron-7B* | 0.511 | 21.8% | 41.4% |

*Meditron-7B is not instruction-tuned. High UNKNOWN rates reflect 
instruction-following failure, not lack of domain knowledge.

---

## Citation
If you use this code or findings in your research, please cite:
```bibtex
@article{hariprasad2026prompt,
  title={Prompt Sensitivity and Answer Consistency of Small Open-Source 
  Large Language Models on Clinical Question Answering: Implications for 
  Low-Resource Healthcare Deployment},
  author={Hariprasad, Shravani},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License
This project is licensed under the MIT License.

## Contact
Shravani Hariprasad — Independent Researcher  
GitHub: [@shravani-01](https://github.com/shravani-01)
