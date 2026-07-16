# Task 2 — Event Summarization (T5)

Generates an ACLED-style narrative event summary from structured event fields — a data-to-text NLG task: `EVENT_DATE | SOURCE | FATALITIES | EVENT_TYPE | SUB_EVENT_TYPE | ACTOR1 | ACTOR2 | LOCATION | INTERACTION` → the free-text `NOTES` description.

**Notebook:** [`Tesla_SUP_Subtask2_2.ipynb`](Tesla_SUP_Subtask2_2.ipynb)

**Approach:** T5-base (Hugging Face `transformers`) fine-tuned with the Adafactor optimizer on ~90k training rows; ACLED's numeric `INTERACTION` codes are first decoded into readable actor-pair phrases. The saved run trained for a **single epoch** (~11,250 batches — about an hour on a Colab GPU per the report; a later cell aspirationally sets 8 epochs, but the saved outputs show one).

**Results (saved 2022 run — 1 epoch, first 100 validation examples):**

| Metric | F1 | Precision | Recall |
|---|---|---|---|
| ROUGE-1 | 0.375 | 0.648 | 0.272 |
| ROUGE-2 | 0.184 | 0.351 | 0.129 |
| ROUGE-L | 0.359 | 0.619 | 0.261 |

## Running

1. Install the dependencies from the first cell (`transformers`, `sentencepiece`, `nltk`, plus `rouge` for evaluation).
2. Obtain `task_2_event_summarization_train.tsv` and `..._valid.tsv` (see [`../../data/README.md`](../../data/README.md)); the notebook was written for Colab and reads them from a mounted Google Drive — adjust the paths for your environment.
3. Make sure a GPU runtime is allocated, then run the cells sequentially.

## Caveats

- This was the **only** implementation of Task 2 — no Milestone 2 version ever existed.
- The fine-tuned model was **never saved to disk**, so the ROUGE numbers can't be reproduced without retraining (~11k batches per epoch on a Colab GPU).
- See the provenance cell at the top of the notebook for the exact post-hoc edits from the 2026 cleanup (section headers added, one empty cell removed).
