# **Ghost in the Machine**

### Precog NLP Recruitment Task 2026

A three-class text classification study on distinguishing **Human**, **Generic AI**, and **Style-mimicking AI** (instructed to imitate Mary Shelley) texts.

Built end-to-end: Dataset Construction, Exploratory Data Analysis, Three-tier Classification, Adversarial Testing, and a Genetic Algorithm attacking the final model.

---

## If you are here to read the work

**Go straight to**  [`notebooks/`](notebooks/).** Start at `task0.ipynb` and work through in order:

```
task0.ipynb  →  task1.ipynb  →  task2.ipynb  →  task3.ipynb  →  task4.ipynb
```

Every notebook is written to be read (not just run). Each section has markdown cells explaining the rationale behind every decision, what I expected, what actually happened, and why. The code cells are also commented. You do not need to re-run anything — all outputs and figures are already saved.

---

## Overview

| Task        | Name                       | What it does                                                                                                                |
| ----------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **0** | The Library of Babel       | Builds the dataset: scrapes and cleans Victorian novels, generates AI paragraphs via Gemini API, splits into train/val/test |
| **1** | The Fingerprint            | EDA: 15 stylometric features engineered, visualised with PCA and UMAP, statistically tested                                 |
| **2** | The Multi-Tiered Detective | Three classifiers: XGBoost (Tier A), GloVe FFNN (Tier B), DistilBERT+LoRA (Tier C)                                          |
| **3** | The Smoking Gun            | Interpretability: Integrated Gradients saliency, near-miss analysis, adversarial probing                                    |
| **4** | The Turing Test            | Genetic Algorithm trying to evolve AI text to fool Tier C; personal essay test                                              |

---

## Codebase:

```
notebooks/
    feature_extraction.py   # Shared feature pipeline imported by task1.ipynb and task2.ipynb
    task0.ipynb             # Dataset construction
    task1.ipynb             # EDA and feature engineering
    task2.ipynb             # Three-tier classifier training
    task3.ipynb             # Interpretability and adversarial probing
    task4.ipynb             # Genetic Algorithm and personal essay test

data/
    raw/                    # Original Project Gutenberg text files
    processed/              # Cleaned novel texts
    class1/                 # Human paragraphs (Shelley + Dickens), JSONL
    class2/                 # Generic AI paragraphs (Gemini), JSONL
    class3/                 # Mimic AI paragraphs (Gemini, Shelley-style), JSONL
    splits/                 # train / val / test JSONL (70/15/15, stratified)
    features/               # Pre-computed stylometric feature CSVs
    glove/                  # GloVe embeddings (not in repo — see step 3 below)
    checkpoints/            # Per-topic generation checkpoints from Task 0
    task4/                  # GA run outputs (initial population, best-per-gen, final pop)
    topics.json             # The 10 shared themes used across all three classes

models/
    distilbert_lora_final/  # Saved DistilBERT+LoRA adapter (Tier C) + tokenizer
    distilbert_lora/        # Training checkpoints (epochs 1-4)

figures/
    task1/                  # PCA scatter, UMAP scatter, PCA loadings, punctuation heatmap
    task2/                  # Confusion matrices and training curves for all three tiers
    task3/                  # Token attribution plots, near-miss plots, AI-isms bar charts
    task4/                  # GA fitness curve, essay vs AI-rewrite comparison
```

---

## How to reproduce my work?

### 1. Clone the repository

```bash
git clone https://github.com/DevashishXO/Precog-Recruitment-Task.git
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Download GloVe embeddings

Task 2 (Tier B) uses GloVe 6B 300d embeddings. These are too large for GitHub so you need to download them manually.

1. Download `glove.6B.zip` from the [Stanford NLP GloVe page](https://nlp.stanford.edu/data/glove.6B.zip)
2. Extract `glove.6B.300d.txt`
3. Place it at:

```
data/
└── glove/
    └── glove.6B.300d.txt   ← file goes here
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up your `.env` file *(only needed if re-generating the dataset)*

Task 0 uses the Gemini API to generate AI paragraphs. If you want to re-run the generation cells, create a `.env` file at the root:

```
api_key=YOUR_KEY_1,YOUR_KEY_2,...
```

The dataset is already generated and committed to the repo, so you can skip this step if you just want to run Tasks 1–4.

### 6. Run the notebooks in order

Open Jupyter and run the notebooks in sequence:

```
notebooks/task0.ipynb  →  task1.ipynb  →  task2.ipynb  →  task3.ipynb  →  task4.ipynb
```

Run cells top to bottom. Each notebook is self-contained and reads from the outputs of the previous one.

---

## Dependencies

Key packages (full list in `requirements.txt`):

| Package                                            | Purpose                                                |
| -------------------------------------------------- | ------------------------------------------------------ |
| `spacy` (+ `en_core_web_sm`)                   | Sentence segmentation, POS tagging, dependency parsing |
| `google-genai`                                   | Gemini API calls (Task 0 and Task 4)                   |
| `xgboost`                                        | Tier A classifier                                      |
| `torch`, `transformers`, `peft`              | Tier B (FFNN) and Tier C (DistilBERT + LoRA)           |
| `captum`                                         | Integrated Gradients saliency analysis (Task 3)        |
| `umap-learn`                                     | UMAP visualisation (Task 1)                            |
| `textstat`                                       | Flesch-Kincaid readability features                    |
| `scikit-learn`                                   | Train/val/test splitting, Grid Search, metrics         |
| `datasets`, `evaluate`                         | HuggingFace training pipeline (Tier C)                 |
| `pandas`, `numpy`, `matplotlib`, `seaborn` | General data handling and visualisation                |

---

## A note on reproducibility

All random seeds are fixed at `42` throughout. The dataset, model weights, and all figures are already committed to the repository. You do not need to re-run anything to verify the results — but if you do re-run, outputs should match.

The one non-deterministic element is Gemini API generation (Tasks 0 and 4). Re-running those cells will produce different paragraphs.
