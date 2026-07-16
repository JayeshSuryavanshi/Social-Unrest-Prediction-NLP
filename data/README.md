# Data

No datasets are versioned in this repository — ACLED's [terms of use](https://acleddata.com/terms-of-use/) don't permit redistribution, and the files totaled ~130 MB (they were removed from version control in 2026; the notebooks' saved outputs still reflect runs on the original files).

## Course-provided TSVs

The course distributed seven ACLED-derived TSV files ("Project 2 data"). The notebooks expect them with these names:

| File | Rows | Size | Used by |
|---|---|---|---|
| `task_1_information_extraction_train.tsv` | 100,000 | 29 MB | Task 1 classifier |
| `task_1_information_extraction_valid.tsv` | 12,712 | 4 MB | Task 1 classifier |
| `task_1_information_extraction_test.tsv` | 12,712 | 3 MB | Task 1 classifier |
| `task_2_event_summarization_train.tsv` | 90,000 | 31 MB | Task 2 (T5) |
| `task_2_event_summarization_valid.tsv` | 11,704 | 4 MB | Task 2 (T5) |
| `task_2_event_summarization_test.tsv` | 11,704 | 2 MB | (not referenced by the notebooks) |
| `task_3_event_prediction.tsv` | 115,664 | 57 MB | Task 3 M2 (LSTM) |

Columns include ACLED's standard fields: `NOTES`, `EVENT_DATE`, `SOURCE`, `FATALITIES`, `EVENT_TYPE`, `SUB_EVENT_TYPE`, `ACTOR1`, `INTER1`, `ACTOR2`, `INTER2`, `INTERACTION`, `LOCATION`, ….

To rebuild equivalents, register at [acleddata.com](https://acleddata.com/) and use the ACLED export tool, then split per task. Exact reproduction of the course splits is not possible — the original TSVs were course-provided.

**Filename quirk:** the Milestone 2 LSTM notebook reads `task_1_information_extraction_training_set.tsv` — a name that matches none of the files above. Its saved output shows shape `(115668, 30)`, which is far closer to `task_3_event_prediction.tsv` (115,664 rows) than to any Task 1 file, so it was most likely the same ACLED extract in a marginally different cut. Kept as-is; point it at your equivalent file if you re-run.

## Other inputs

- **`ACLED2019data.csv`** — an ACLED export covering **1 Jan – 28 Feb 2019**, used by `Labelling_Event_with_ACLED.ipynb` to label news articles. Pull the same date range from the ACLED export tool.
- **`glove.6B.300d.txt`** — GloVe 300-d embeddings from the [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/) (`glove.6B.zip`), used by the Milestone 2 LSTM notebook.

## Pipeline-generated files (cannot be regenerated)

The Milestone 3 news pipeline wrote these intermediates: `getBingNews.csv`, `getBingNewsProcessed.csv` (from `DownloadBingNews.ipynb`) and `bingACLEDNewsMatch.csv` (from `Labelling_Event_with_ACLED.ipynb`, consumed by `Training_and_Prediction.ipynb`).

Microsoft **retired the Bing News Search v7 API in August 2025**, so the download step no longer works and these files cannot be recreated as written. A modern re-implementation would substitute another news source (e.g. [GDELT](https://www.gdeltproject.org/) or NewsAPI) in `DownloadBingNews.ipynb`; the rest of the pipeline is source-agnostic.

## Paths

The notebooks were written for Google Colab and read from `/content/...` (files uploaded to the session) or a mounted Google Drive (`/content/drive/MyDrive/project_2_data/...`). Adjust paths to wherever you place the files locally.
