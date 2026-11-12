"""The Validate step without Streamlit: data of the charts and tables of the developer panel (pure functions)."""

from __future__ import annotations

import pandas as pd

RANGE_DIGITS = 3                   # significant digits of a bin edge on the axis
DATE_CHARS = len("YYYY-MM-DD")     # bin edges of a date column: the day is enough
SOURCES = ("Real", "Synthetic")


def edge_text(value, role: str) -> str:
    """One bin edge as the axis shows it."""
    if role == "date":
        return str(value)[:DATE_CHARS]
    return f"{value:.{RANGE_DIGITS}g}"


def bin_labels(histogram: dict) -> list[str]:
    """Labels of the bars: ``"a – b"`` ranges for numbers and dates (``labels`` are edges), levels otherwise."""
    if histogram["role"] == "categorical":
        return [str(level) for level in histogram["labels"]]
    edges = [edge_text(value, histogram["role"]) for value in histogram["labels"]]
    return [f"{low} – {high}" for low, high in zip(edges, edges[1:])]


def histogram_frame(histogram: dict) -> pd.DataFrame:
    """Long table for grouped bars: one row per bin and source, with the bin order (``order``)."""
    labels = bin_labels(histogram)
    rows = [{"bin": label, "data": source, "share": share, "order": position}
            for source, shares in zip(SOURCES, (histogram["real"], histogram["synthetic"]))
            for position, (label, share) in enumerate(zip(labels, shares))]
    return pd.DataFrame(rows, columns=["bin", "data", "share", "order"])


def heatmap_frame(columns: list[str], matrix: list[list[float | None]]) -> pd.DataFrame:
    """Long table of an association matrix: one row per pair of columns (``None`` stays missing)."""
    rows = [{"row": row, "column": column, "association": value}
            for row, values in zip(columns, matrix) for column, value in zip(columns, values)]
    return pd.DataFrame(rows, columns=["row", "column", "association"])     # None becomes NaN: a float column


def fidelity_table(fidelity: dict) -> pd.DataFrame:
    """Developer panel: distance, real-vs-real baseline and spread, z and excess of every column."""
    return pd.DataFrame(fidelity["columns"], columns=["name", "role", "distance", "baseline", "spread", "z", "excess"])


def privacy_table(privacy: dict) -> pd.DataFrame:
    """Developer panel: the privacy measures, synthetic next to the holdout (new real people)."""
    rows = [
        ("Distance to closest record, percentile", privacy["synthetic_percentile"], privacy["holdout_percentile"]),
        ("Exact copies of train rows (share)", privacy["synthetic_exact_share"], privacy["holdout_exact_share"]),
        ("Closer to train than to holdout (share)", privacy["train_closer_share"],
         privacy["expected_train_closer_share"]),
    ]
    return pd.DataFrame(rows, columns=["measure", "synthetic", "holdout / expected"])
