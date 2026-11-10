"""Generate the column-level "torture" CSV used by the Data Engine tests.

Each column exercises one rule of the schema contract; its name says which. The expected outcome
of every column lives in ``tests/test_data_engine_reading.py``. The output is deterministic, so the
committed CSV can be checked against a fresh generation.

Usage:
    python tests/fixtures/make_torture_csv.py
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np

SEED = 42
N_ROWS = 200
OUTPUT_PATH = Path(__file__).resolve().parent / "torture_columns.csv"

N_QMARKS = 10           # "?" in num_with_qmark
N_ND = 5                # "n.d." in num_with_nd (2.5% -> above the 95% "mostly numeric" bar)
N_PER_NULL_TOKEN = 10   # per token in null_tokens_mix (5 tokens -> 25% < MAX_NULL_FRACTION)
NULL_TOKENS = ["?", "NA", "", "null", " N/A "]
N_CODE_POOL = 80        # numeric-looking codes in codes_mixed (ICD-9 style: "250.83")
LETTER_CODES = ["V57", "V58", "V45", "E878", "V10", "E888", "V57", "V45"]  # 4%, 6 distinct
DATE_START = dt.date(2024, 1, 1)
DATE_SPAN_DAYS = 365


def _with_tokens(rng: np.random.Generator, values: list[str], tokens: list[str]) -> list[str]:
    """Overwrite random positions of ``values`` with ``tokens`` (one position per token)."""
    out = list(values)
    for pos, token in zip(rng.choice(len(out), size=len(tokens), replace=False), tokens, strict=True):
        out[pos] = token
    return out


def build_columns(seed: int = SEED, n_rows: int = N_ROWS) -> dict[str, list[str]]:
    """Build every torture column as a list of raw strings (exactly as they will appear in the CSV)."""
    rng = np.random.default_rng(seed)

    def floats(mean: float, std: float) -> list[str]:
        return [f"{v:.2f}" for v in rng.normal(mean, std, n_rows)]

    cols: dict[str, list[str]] = {}
    cols["numeric_float"] = floats(120, 15)
    cols["numeric_int_wide"] = [str(v) for v in rng.integers(20, 80, n_rows)]
    cols["cat_str"] = rng.choice(["low", "medium", "high", "very high"], n_rows).tolist()
    cols["cat_int_codes"] = [str(v) for v in rng.integers(1, 6, n_rows)]
    cols["id_int"] = [str(v) for v in rng.permutation(np.arange(1000, 1000 + n_rows))]
    cols["id_str"] = [f"P{i:05d}" for i in rng.permutation(n_rows)]
    cols["high_card_str"] = [f"city_{k}" for k in rng.integers(0, 100, n_rows)]
    cols["constant"] = ["yes"] * n_rows
    cols["all_null"] = [""] * n_rows
    cols["mostly_null"] = _with_tokens(
        rng, [str(v) for v in rng.integers(0, 50, n_rows)], ["?"] * (n_rows // 2)
    )
    cols["num_with_qmark"] = _with_tokens(rng, floats(80, 10), ["?"] * N_QMARKS)
    cols["null_tokens_mix"] = _with_tokens(
        rng, floats(5, 1), [t for t in NULL_TOKENS for _ in range(N_PER_NULL_TOKEN)]
    )
    cols["spaced_levels"] = rng.choice([" Male", "Male ", "Male", " Female ", "Female"], n_rows).tolist()
    cols["case_levels"] = rng.choice(
        ["Male", "male", "MALE", "Female", "female"], n_rows, p=[0.4, 0.07, 0.03, 0.4, 0.1]
    ).tolist()
    cols["num_with_nd"] = _with_tokens(rng, floats(30, 5), ["n.d."] * N_ND)
    cols["visit_date"] = [
        (DATE_START + dt.timedelta(days=int(d))).isoformat()
        for d in rng.integers(0, DATE_SPAN_DAYS, n_rows)
    ]
    # Diagnosis-like codes: ~96% look numeric, but the rest are many *different* words, so they
    # are labels, not numbers with a missing-value marker (real case: diag_1 of Diabetes 130).
    code_pool = [f"{v:.2f}" for v in rng.uniform(1, 999, N_CODE_POOL)]
    cols["codes_mixed"] = _with_tokens(rng, rng.choice(code_pool, n_rows).tolist(), LETTER_CODES)
    return cols


def write_csv(path: Path = OUTPUT_PATH, seed: int = SEED) -> Path:
    """Write the torture CSV to ``path`` and return it."""
    cols = build_columns(seed)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(cols.keys())
        writer.writerows(zip(*cols.values(), strict=True))
    return path


if __name__ == "__main__":
    write_csv()
