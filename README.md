# Synthetic Data Sandbox

A local app where a non-technical user (for example a physician) uploads a sensitive tabular dataset,
trains a **Variational Autoencoder on their own machine**, gets a **synthetic dataset** that keeps the
statistical properties of the original, and asks questions about it in natural language.

> **Status**: work in progress. This repository currently contains the **Data Engine** (schema
> inference, cleaning, encoding, schema contract, decoding) and the sample datasets downloader.
> The generative core (VAE), the validation reports, the FastAPI backend, the Streamlit UI and the
> natural-language analyst will follow.

## Privacy model

- The real data never leave the machine: reading, cleaning, training and validation all run locally.
- The language model only receives the **schema** of the synthetic dataset (column names, types and
  category levels) and the user's question: never a row, real or synthetic.
- The session is disposable: real data and the trained model live in memory or in a temporary folder
  and are deleted on reset. The only thing the user takes away is the synthetic CSV.

## Installation

Python 3.12 (tested on Linux). A GPU is optional: the VAE trains on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # the app plus tests; just the app: pip install -e .
```

The editable install (`-e`) is the supported way: the app finds its `data/` folder from its own location.

## Sample datasets

The sample datasets are **not** included in the repository: they are only needed to develop, test and
evaluate the app. After installing with the `dev` extra, download them with:

```bash
sds-download-samples                    # all six, into ./data
sds-download-samples --only adult pima  # a subset
```

Each file is verified with a SHA-256 checksum. They are kept as raw as possible (only a header row is
added where the source has none), because messy files are exactly what the Data Engine must handle.
The destination is `data/` at the project root, or the folder named by the `SDS_DATA_DIR` environment
variable.

| Name | Source | Rows | License |
|---|---|---|---|
| `adult` | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) | 32,561 | CC BY 4.0 |
| `breast_cancer` | [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | CC BY 4.0 |
| `bank_marketing` | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) | 45,211 | CC BY 4.0 |
| `cdc_diabetes` | [UCI CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) | 253,680 | not stated by UCI |
| `diabetes_130` | [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) | 101,766 | CC BY 4.0 |
| `pima` | Pima Indians Diabetes ([mirror](https://github.com/jbrownlee/Datasets)) | 768 | unclear |

## Tests

```bash
pytest
```

The regression tests run the Data Engine on the six sample datasets and are **skipped** (with the
command to run in the reason) when the files have not been downloaded.

## Data Engine

`sds.data_engine` turns any CSV into numbers a neural network can learn from, and back:

1. **Read and clean** (`inspect_csv`): encoding and separator detection, null tokens (`?`, `NA`, empty
   cells...), numbers with a decimal comma, ISO dates, case unification of text levels.
2. **Propose a role** for every column (numeric, categorical, date, identifier, dropped) with a reason
   in plain language; the user can confirm or change it (`UserChoices`, `apply_choices`).
3. **Prepare** (`prepare`): keep the chosen columns, drop the rows with nulls in them, split into a
   training set and a holdout. The holdout never enters training: it is the reference for the privacy check.
4. **Encode** (`fit_encoder`): Min-Max scaling of numbers and dates, One-Hot encoding of categories (rare
   levels merged into `Other`). Parameters are fitted on the training set only.
5. **Schema contract** (`build_contract`): a JSON document describing every column, enough to decode the
   model output without the real data. Its `llm_schema()` projection is all the language model will see.
6. **Decode** (`decode`): from model vectors back to a table with the original column names, types and
   levels, repairing pairs of categories never seen together in the training set.

## Known limitations

- **Dates**: only the ISO format `YYYY-MM-DD` is recognised; `dd/mm/yyyy` and `mm/dd/yyyy` cannot be told
  apart, so such columns are read as text.
- **Split**: the training/holdout split is random. If several rows belong to the same person, that person
  can end up in both sets, which makes the privacy check more optimistic.
- **Rare whole values merged into `Other`** are chosen after the category repair, so they can form rare
  combinations never seen in the real data (0.09% of the rows on `diabetes_130`).

## License

[MIT](LICENSE). The sample datasets are not part of this repository and keep their own licenses.
