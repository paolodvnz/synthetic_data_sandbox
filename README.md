# Synthetic Data Sandbox

A local app where a non-technical user (for example a physician) uploads a sensitive tabular dataset,
trains a **Variational Autoencoder on their own machine**, gets a **synthetic dataset** that keeps the
statistical properties of the original, and asks questions about it in natural language.

> **Status**: work in progress. The app runs end to end on the local side: **Upload → Train → Validate**,
> with the download of the synthetic CSV behind a privacy gate. The fourth step, **Ask** (questions in
> natural language answered with charts by a language model), is shown but not available yet.

## Privacy model

- The real data never leave the machine: reading, cleaning, training and validation all run locally.
- The language model will only receive the **schema** of the synthetic dataset (column names, types and
  category levels) and the user's question: never a row, real or synthetic.
- The session is disposable: the real data, the trained model and the synthetic data live **in memory only**,
  in the API process, and are dropped on "Start over" or when the app stops (even after a crash nothing is
  left on disk). The only thing the user takes away is the synthetic CSV.
- **Privacy gate**: the download is unlocked only when the synthetic rows pass the privacy check (exact
  copies and distance to the closest record, against real rows the model never saw).

## Installation

Python 3.12 (tested on Linux). A GPU is optional: the VAE trains on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # the app plus tests; just the app: pip install -e .
```

The editable install (`-e`) is the supported way: the app finds its `api/`, `ui/` and `data/` folders from
its own location.

## Running the app

```bash
sds-app          # starts the API and the UI on this machine and opens the browser
sds-app --dev    # developer mode: extra panels on how the engine works, and the sample datasets
```

Both processes listen on the local machine only (`127.0.0.1`). Stop the app with `Ctrl+C` or with the
**Close** button of the page: in both cases the data in memory are discarded. `--api-port` and `--ui-port`
change the ports (default 8000 and 8501).

The guided path:

1. **Upload**: a CSV file (up to 200 MB). The app proposes which columns to use and how to read them, and
   explains why; the user can untick columns (for example identifiers) or change an ambiguous reading.
2. **Train**: one button. The VAE trains with settings scaled to the dataset (there are no knobs), then
   generates as many synthetic rows as were used for learning and validates them. A progress bar follows
   the training, which can be stopped.
3. **Validate**: the outcome of the privacy gate, non-blocking warnings on fidelity, the download, a preview
   of the synthetic rows, association heatmaps (real vs synthetic) and a histogram per column.

The developer mode **shows more and never removes a protection**: the schema contract, a decoding check,
the training plan and curves, the full validation reports. The privacy gate stays blocking and no real row
ever reaches the page.

The two parts can also be started by hand, from the project root:

```bash
uvicorn api.main:app --host 127.0.0.1
streamlit run ui/app.py
```

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
variable. In developer mode the Upload step offers them in a "Try a sample dataset" menu.

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

The regression tests run the engine on the sample datasets and are **skipped** (with the command to run in
the reason) when the files have not been downloaded. The few GPU tests are skipped on a machine without CUDA.

## How it works

### Data Engine

`sds.data_engine` turns any CSV into numbers a neural network can learn from, and back:

1. **Read and clean** (`inspect_csv`): encoding and separator detection, null tokens (`?`, `NA`, empty
   cells...), numbers with a decimal comma, ISO dates, case unification of text levels.
2. **Propose a role** for every column (numeric, categorical, date, identifier, dropped) with a reason
   in plain language; the user can confirm or change it (`UserChoices`, `apply_choices`). Numbers with few
   distinct values, whole or not, are read as categories by default.
3. **Prepare** (`prepare`): keep the chosen columns, drop the rows with nulls in them, split into a
   training set and a holdout. The holdout never enters training: it is the reference for the validation.
4. **Encode** (`fit_encoder`): Min-Max scaling of numbers and dates, One-Hot encoding of categories (rare
   levels merged into `Other`). Parameters are fitted on the training set only.
5. **Schema contract** (`build_contract`): a JSON document describing every column, enough to decode the
   model output without the real data. Its `llm_schema()` projection is all the language model will see.
6. **Decode** (`decode`): from model vectors back to a table with the original column names, types, levels
   and decimals, repairing pairs of categories never seen together in the training set.

### Generative core

`sds.vae` is a tabular VAE in PyTorch: dense encoder → (μ, log σ²), reparameterization trick, mirrored
decoder. The network size follows the width of the encoded vector. The loss is `mixed_sigma`: a Gaussian
likelihood with a learned σ per numeric column plus a cross-entropy per categorical group, and the KL term
with a warm-up. The training length is set in epochs (capped in steps) with early stopping on a validation
part of the training set. Sampling starts from the prior, adds the learned noise to the numeric columns
and replaces the few rows the coherence repair cannot fix. The assignment's baseline loss (MSE on every
position) is kept for comparison.

### Validation

`sds.validation` compares the synthetic rows with the **holdout**, real people the model never saw:

- **Privacy** (`privacy_check`, blocking): exact copies of training rows beyond the genuine duplicates of
  the real data, and the distance to the closest record (Gower distance), whose low percentile must not be
  much lower than the one of the holdout. A synthetic row should be no closer to the training data than a new real
  person is.
- **Fidelity** (`fidelity_report`, non-blocking warnings): Wasserstein distance per numeric column, total
  variation per categorical column and the difference of the association matrices, each measured against
  the differences between groups of real people, so that only differences beyond chance are reported.

### API and UI

A FastAPI backend (`api/`) holds the single in-memory session and runs the training job in the background;
the Streamlit UI (`ui/`) talks only to the API, over HTTP. Every error reaches the page as a plain-language
message, never as a stack trace.

## Known limitations

- **Dates**: only the ISO format `YYYY-MM-DD` is recognised; `dd/mm/yyyy` and `mm/dd/yyyy` cannot be told
  apart, so such columns are read as text.
- **Split**: the training/holdout split is random. If several rows belong to the same person, that person
  can end up in both sets, which makes the privacy check more optimistic.
- **Rare whole values merged into `Other`** are chosen after the category repair, so they can form rare
  combinations never seen in the real data (0.09% of the rows on `diabetes_130`).
- **Near-copies on dense data**: when many real people have identical answers (for example `cdc_diabetes`:
  few columns, almost all yes/no), new people are often identical to a training row, so no distance-based
  check can tell a near-copy from a real person. Exact copies are still checked, and the page shows a
  non-blocking warning.
- **Point masses** (a value shared by many rows, such as a capital gain of 0) are only partly rebuilt: the
  synthetic data spread part of them over nearby values.
- **Very small datasets**: the fidelity baseline is estimated from few rows, so a fidelity warning can
  occasionally appear by chance.

## License

[MIT](LICENSE). The sample datasets are not part of this repository and keep their own licenses.
