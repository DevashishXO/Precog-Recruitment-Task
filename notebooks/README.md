# notebooks/

This is core where my work lives. Each notebook corresponds to one task and is meant to be read in order. I have used markdown cells  to explain every decision, expectation, and observation before and after each code cell.

Start at `task0.ipynb` and move forward in order.

```
notebooks/
├── task0.ipynb               # Task 0: The Library of Babel - dataset construction
│                             #   Cleans Victorian novels, generates AI paragraphs via
│                             #   Gemini API, builds and splits the three-class dataset.
│
├── task1.ipynb               # Task 1: The Fingerprint - EDA and feature engineering
│                             #   15 stylometric features computed, visualised with
│                             #   PCA and UMAP, statistically compared across classes.
│
├── task2.ipynb               # Task 2: The Multi-Tiered Detective - classifiers
│                             #   Tier A: XGBoost on stylometric features
│                             #   Tier B: Feedforward NN on averaged GloVe embeddings
│                             #   Tier C: DistilBERT fine-tuned with LoRA
│
├── task3.ipynb               # Task 3: The Smoking Gun - interpretations
│                             #   Integrated Gradients saliency analysis, near-miss
│                             #   analysis, adversarial probing of the Tier C model.
│
├── task4.ipynb               # Task 4: The Turing Test - evolution of texts
│                             #   Genetic Algorithm evolving AI text to fool the
│                             #   Tier C detector; personal essay classification test.
│
└── feature_extraction.py     # Shared feature pipeline (Helper file)
                              #   Contains extract_all_features() and FEATURE_COLUMNS
                              #   imported by task1.ipynb and task2.ipynb so the same
                              #   feature logic runs on train, val, and test splits.
```
