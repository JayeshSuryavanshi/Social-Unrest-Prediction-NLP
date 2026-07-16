# Monitoring and Predicting Social Unrest

NLP course project — University at Buffalo, 2022.
**Team Tesla:** Jayesh Suryavanshi · Sumeet Aher · Krati Sharma
Course taught by Dr. Rohini K. Srihari (TAs: Sougata Saha, Souvik Das). Full write-up: [`docs/report.pdf`](docs/report.pdf).

The project builds an early-warning pipeline around [ACLED](https://acleddata.com/) (Armed Conflict Location & Event Data): extract structure from conflict-event descriptions, generate ACLED-style event summaries, and predict upcoming unrest events from news streams.

Built in spring 2022 — before ChatGPT — with the then-standard NLP toolkit: TF-IDF + classical scikit-learn models, GloVe embeddings + a Keras LSTM, and a fine-tuned T5 transformer. The notebooks are preserved as a faithful record of that work; see [Reproducibility](#reproducibility--environment) and the provenance note at the top of each notebook for what was (and was not) touched in the 2026 cleanup.

## The three tasks

| | What it does | Approach | Where |
|---|---|---|---|
| **Task 1 — Information extraction** | Classify an ACLED event description (`NOTES`) into one of 6 event types (Battles, Explosions/Remote violence, Protests, Riots, Strategic developments, Violence against civilians) | TF-IDF (1–2 grams) → LinearSVC / Logistic Regression / Multinomial NB / Random Forest | [`notebooks/task1_event_classification/`](notebooks/task1_event_classification/) |
| **Task 2 — Event summarization** | Generate an ACLED-style narrative summary from structured event fields (date, actors, location, fatalities, …) | T5-base fine-tuned on ~90k ACLED rows (structured-to-text) | [`notebooks/task2_event_summarization/`](notebooks/task2_event_summarization/) |
| **Task 3 — Event prediction** | Predict whether an unrest event will occur, from news text | M2: GloVe + LSTM over daily event sequences · M3: Bing News ↔ ACLED matching + SGD/LogReg/MNB classifiers | [`notebooks/task3_event_prediction/`](notebooks/task3_event_prediction/) |

### What exists where (tasks × milestones)

The course ran in three milestones. Not everything described in the report has code in this repository — this matrix is the honest inventory:

| | Milestone 1 | Milestone 2 | Milestone 3 |
|---|---|---|---|
| **Task 1** | report only | ✅ `Task1Classifier.ipynb` | same notebook (M2/M3 submissions were byte-identical; the report's BERT tagging upgrade was never committed) |
| **Task 2** | report only | — (never implemented) | ✅ `Tesla_SUP_Subtask2_2.ipynb` (T5) |
| **Task 3** | report only | ✅ `Task3_milestone2.ipynb` (LSTM) | ✅ 3-notebook news pipeline |

## Repository map

```
├── docs/report.pdf                            # final project report
├── data/README.md                             # how to obtain the datasets (not versioned)
└── notebooks/
    ├── task1_event_classification/
    │   └── Task1Classifier.ipynb              # TF-IDF + 4 classical models, confusion-matrix analysis
    ├── task2_event_summarization/
    │   └── Tesla_SUP_Subtask2_2.ipynb         # T5-base fine-tuning + ROUGE evaluation
    └── task3_event_prediction/
        ├── milestone2_lstm/
        │   └── Task3_milestone2.ipynb         # GloVe + spaCy day-vectors → LSTM
        └── milestone3_news_pipeline/
            ├── DownloadBingNews.ipynb         # 1) pull news via Bing News Search v7 (API retired 2025)
            ├── Labelling_Event_with_ACLED.ipynb  # 2) label news by cosine-matching ACLED events
            └── Training_and_Prediction.ipynb  # 3) train/evaluate event-prediction classifiers
```

## Results

All numbers below are from the saved notebook outputs of the original 2022 runs (see each notebook for the full classification reports).

**Task 1 — event-type classification** (5-fold CV accuracy; run on a ~100-row sample of the training set, hence the modest numbers — the report cites ~65% for Multinomial NB on a fuller run):

| Model | CV accuracy |
|---|---|
| Logistic Regression | 0.59 |
| Multinomial Naive Bayes | 0.59 |
| LinearSVC | 0.57 |
| Random Forest | 0.50 |

**Task 2 — T5 summarization** (first 100 validation examples):

| Metric | F1 | Precision | Recall |
|---|---|---|---|
| ROUGE-1 | 0.375 | 0.648 | 0.272 |
| ROUGE-2 | 0.184 | 0.351 | 0.129 |
| ROUGE-L | 0.359 | 0.619 | 0.261 |

**Task 3 — event prediction, Milestone 3 pipeline** (1,065 news articles labeled by ACLED matching; 10% held-out test set, n=107):

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| SGDClassifier (hinge) | 0.85 | 0.85 |
| Logistic Regression | 0.85 | 0.85 |
| Multinomial Naive Bayes | 0.83 | 0.83 |

Among matched news–event pairs, the news article preceded its ACLED event by **≈5 days on average** — the pipeline's effective warning lead time.

(Task 3 Milestone 2, the LSTM approach: training/evaluation outputs were not captured in the saved run, so no metrics survive; the report discusses it qualitatively.)

## Data

None of the datasets are versioned here (ACLED's terms don't permit redistribution, and the files total ~130 MB). [`data/README.md`](data/README.md) lists every file the notebooks expect and how to obtain it:

- the seven course-provided ACLED-derived TSVs (train/valid/test per task),
- an ACLED export (Jan–Feb 2019) for the news-labeling step,
- GloVe 300-d embeddings,
- the Bing News CSVs — **these can no longer be regenerated**: Microsoft retired the Bing News Search v7 API in August 2025.

## Reproducibility & environment

These notebooks are a **historical record**, not a maintained pipeline. They were written for Google Colab (GPU for Tasks 2/3) and still contain `/content/...` and Google Drive paths. [`requirements.txt`](requirements.txt) documents the stack; re-execution end-to-end is not guaranteed and was deliberately not attempted during cleanup — every saved output you see is from the original 2022 runs.

## Known gaps & limitations

- **Milestone 1** produced reports only; no code exists for it.
- **Task 2 had no Milestone 2 implementation**, and the report's BERT-based tagging upgrade for Task 1 Milestone 3 was never committed.
- The Task 2 **fine-tuned T5 model was never saved**, so its ROUGE numbers can't be reproduced without retraining.
- Task 1's saved metrics come from a ~100-row sample; Task 2's ROUGE from 100 validation examples.
- The Task 3 M2 LSTM run had a **GloVe loading error** (traceback preserved in the notebook) and its metrics were never captured.
- Several small 2022 bugs were found and fixed during the 2026 cleanup (a validation-features copy-paste bug, a page-fetch loop-indentation bug, a removed-in-sklearn-0.20 import, a missing seaborn import). Each notebook's provenance cell states exactly what changed; all saved outputs predate the fixes.

## Repository history notes

**2026 cleanup.** The repo was restructured from the original `Team Tesla/codebase/Milestone N/Task M/` layout into the task-centric tree above; a byte-identical duplicate notebook was deduplicated, 1.7 MB of junk cell output was stripped, the bugs above were fixed, and this README replaced a generic placeholder. The full original layout is preserved in git history.

**Security disclosure.** The 2022 code hardcoded an Azure Bing News Search v7 subscription key, which therefore exists in this repository's git history. It was scrubbed from the working tree in the 2026 cleanup and is inert regardless: Microsoft retired the Bing Search APIs in August 2025, and the endpoint no longer exists. History was deliberately not rewritten — the key appears in nearly every historical commit, so a rewrite would invalidate all existing commit URLs/forks while GitHub would still retain cached views; the working-tree scrub plus this disclosure is the honest fix.

## License

[MIT](LICENSE). ACLED data is subject to [ACLED's terms of use](https://acleddata.com/terms-of-use/) and is not distributed with this repository.
