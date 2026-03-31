# figures/

I have organized all visualisations generated across Tasks 1–4.

Figures are saved as `.png` at 150 dpi and are embedded in the corresponding notebook cell outputs.

```
figures/
│
├── task1/
│   ├── pca_scatter.png               # PCA of 15 stylometric features (3-class scatter plot)
│   ├── pca_loadings.png              # PC1 and PC2 feature loadings as horizontal bar charts
│   ├── umap_scatter.png              # UMAP of 15 stylometric features (cleaner than PCA)
│   └── punctuation heatmap.png       # Mean punctuation rates per 1000 words, all 6 kinds x 3 classes
│
├── task2/
│   ├── tier_a_confusion.png          # Tier A (XGBoost) confusion matrix on test set
│   ├── tier_a_feature_importance.png # Gain-based feature importance bar chart (XGBoost)
│   ├── tier_b_loss.png               # Tier B (GloVe FFNN) train and val loss curves
│   ├── tier_b_confusion.png          # Tier B confusion matrix on test set
│   ├── tier_c_curves.png             # Tier C (DistilBERT+LoRA) train loss, val loss, val macro-F1
│   └── tier_c_confusion.png          # Tier C confusion matrix on test set
│
├── task3/
│   ├── attribution_class0.png        # Integrated Gradients token attribution — Human sample
│   ├── attribution_class1.png        # Integrated Gradients token attribution — Generic AI sample
│   ├── attribution_class2.png        # Integrated Gradients token attribution — Mimic AI sample
│   ├── top_tokens_per_class.png      # Aggregated top-attribution tokens across 5 samples per class
│   ├── boundary_human_1.png          # Near-miss Human sample 1 attribution plot
│   ├── boundary_human_2.png          # Near-miss Human sample 2 attribution plot
│   ├── boundary_human_3.png          # Near-miss Human sample 3 attribution plot
│   ├── boundary_ai_1.png             # Near-miss AI sample 1 attribution plot
│   ├── boundary_ai_2.png             # Near-miss AI sample 2 attribution plot
│   ├── boundary_ai_3.png             # Near-miss AI sample 3 attribution plot
│   ├── test_past_tense_attribution.png  # Adversarial test 1: modern prose in past tense
│   └── test_victorian_attribution.png   # Adversarial test 2: Victorian vocabulary in present tense
│
└── task4/
    ├── ga_fitness_curve.png          # P(Human) best and mean across 7 GA generations
    └── essay_comparison.png          # Bar chart: my essay vs AI-style rewrite class probabilities
```
