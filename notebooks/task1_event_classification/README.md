# Task 1 — Information Extraction (event-type classification)

Classifies an ACLED event description (the free-text `NOTES` field) into one of six event types: Battles, Explosions/Remote violence, Protests, Riots, Strategic developments, Violence against civilians.

**Notebook:** [`Task1Classifier.ipynb`](Task1Classifier.ipynb)

**Approach:** TF-IDF features (sublinear TF, 1–2 grams, English stop words) benchmarked across LinearSVC, Logistic Regression, Multinomial Naive Bayes, and Random Forest with 5-fold cross-validation, followed by a confusion-matrix and misclassification analysis of the LinearSVC model.

**Results (saved 2022 run, ~100-row sample):** Logistic Regression 0.59 · Multinomial NB 0.59 · LinearSVC 0.57 · Random Forest 0.50 (CV accuracy). The report cites ~65% for Multinomial NB on a fuller run.

## Running

1. Obtain `task_1_information_extraction_train.tsv` (see [`../../data/README.md`](../../data/README.md)) and adjust the path in the loading cell (the notebook was written for Colab and reads `/content/...`).
2. Run the cells sequentially.

## Provenance

- The Milestone 2 and Milestone 3 submissions of this notebook were **byte-identical**, so a single copy is kept (the original duplicate layout is in git history).
- The report describes a BERT-based tagging upgrade for Milestone 3; that code was never checked into the repository.
- See the provenance cell at the top of the notebook for the exact post-hoc fixes applied in the 2026 cleanup (missing seaborn import, `plt.show()` call, one empty cell removed).
