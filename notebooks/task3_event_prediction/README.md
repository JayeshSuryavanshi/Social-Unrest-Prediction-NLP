# Task 3 — Event Prediction

Predicts whether a social-unrest event will occur, from news text. Two genuinely different approaches were built across the course's milestones, so both are kept:

## Milestone 2 — LSTM over daily event sequences ([`milestone2_lstm/`](milestone2_lstm/))

`Task3_milestone2.ipynb`: filters ACLED data to one high-activity location (Jammu), builds per-day sentence vectors (GloVe word embeddings averaged via spaCy tokenization), windows them into 5-day sequences, and trains a small Keras LSTM to predict whether an event occurs in a lookahead window.

**Status:** the saved run's GloVe load errored (traceback preserved in the notebook) and the training/evaluation outputs were never captured, so no metrics survive from this approach; the report discusses it qualitatively.

## Milestone 3 — news-matching pipeline ([`milestone3_news_pipeline/`](milestone3_news_pipeline/))

Three notebooks, run in order:

1. **`DownloadBingNews.ipynb`** — pulls unrest-related news (protests, battles, explosions, riots, …) via the Azure Bing News Search v7 API and extracts full article text/summary/keywords with `newspaper3k` → `getBingNews.csv`, `getBingNewsProcessed.csv`.
   ⚠️ **This stage can no longer be run:** Microsoft retired the Bing Search APIs in August 2025. A modern re-run would substitute another news source (e.g. GDELT); the downstream notebooks are source-agnostic.
2. **`Labelling_Event_with_ACLED.ipynb`** — labels each article by TF-IDF cosine similarity (threshold ≥ 0.3) against ACLED event notes from the same period (`ACLED2019data.csv`, Jan–Feb 2019), recording the news-to-event date gap → `bingACLEDNewsMatch.csv`.
3. **`Training_and_Prediction.ipynb`** — trains binary classifiers on the labeled articles (1,065 rows, keywords as features).

**Results (saved run, 10% held-out test set, n=107):** SGDClassifier 0.85 · Logistic Regression 0.85 · Multinomial NB 0.83 accuracy. Among matched pairs, news preceded its ACLED event by **≈5 days on average** — the pipeline's effective warning lead time. These numbers match the 2022 report's screenshots, but note the saved outputs of notebooks 1 and 3 were produced in a pre-2022 environment (details in their provenance cells).

## Data

See [`../../data/README.md`](../../data/README.md) for every file these notebooks expect and how to obtain what's still obtainable. Each notebook's top cell documents the exact post-hoc fixes applied in the 2026 cleanup (a page-fetch loop-indentation bug, a dead-code validation-features bug, a removed-in-sklearn-0.20 import, junk-output stripping).
