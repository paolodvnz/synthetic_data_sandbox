"""Preparation: remove rows with missing values, then split them into train and holdout.

The holdout is the reference of the privacy check (DCR): real rows the VAE never sees. The split is
purely random. Declared limit: when the same person appears in several rows, some of their rows end
up on both sides and the privacy check becomes too lenient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sds.config import RANDOM_STATE
from sds.data_engine.choices import FinalSchema
from sds.data_engine.thresholds_config import (
    HIGH_ROW_LOSS_FRACTION,
    HOLDOUT_FRACTION,
    MAX_HOLDOUT_ROWS,
    MAX_LISTED_COLUMNS,
    MIN_ROWS,
    MIN_USABLE_ROWS,
)
from sds.data_engine.types import CsvInspection, DataEngineError, Issue


@dataclass
class PreparedData:
    """Rows ready for encoding.

    Attributes:
        train: Rows the VAE learns from (schema columns only, file order of the rows).
        holdout: Rows kept apart for the privacy check, never trained on.
        rows_read: Rows in the inspected file.
        rows_removed: Rows removed because a kept column was missing.
        issues: Warnings about the rows removed and the size of the dataset.
    """

    train: pd.DataFrame
    holdout: pd.DataFrame
    rows_read: int
    rows_removed: int
    issues: list[Issue] = field(default_factory=list)


def removal_issue(frame: pd.DataFrame, rows_removed: int) -> Issue | None:
    """Warning about the rows removed, naming the emptiest columns when most rows are lost."""
    if rows_removed == 0:
        return None
    share = rows_removed / len(frame)
    if share <= HIGH_ROW_LOSS_FRACTION:
        return Issue(
            "rows_removed",
            f"{rows_removed:,} rows with missing values were removed ({share:.0%} of the file).",
        )
    emptiest = frame.isna().sum().sort_values(ascending=False)
    names = ", ".join(f"'{name}'" for name in emptiest[emptiest > 0].index[:MAX_LISTED_COLUMNS])
    return Issue(
        "many_rows_removed",
        f"{rows_removed:,} rows with missing values were removed ({share:.0%} of the file). "
        f"Most missing values are in {names}: removing these columns would keep more rows.",
    )


def holdout_size(n_rows: int) -> int:
    """Number of holdout rows: a fixed share, capped, and never zero."""
    return max(1, min(round(n_rows * HOLDOUT_FRACTION), MAX_HOLDOUT_ROWS))


def prepare(inspection: CsvInspection, schema: FinalSchema, *, seed: int = RANDOM_STATE) -> PreparedData:
    """Keep the schema columns, remove rows with missing values and split train / holdout.

    Args:
        inspection: Result of ``inspect_csv``.
        schema: Result of ``apply_choices`` on the same inspection.
        seed: Seed of the random split (same seed, same split).

    Returns:
        Train and holdout rows, with the warnings to show to the user.

    Raises:
        DataEngineError: ``too_few_rows`` when fewer than ``MIN_USABLE_ROWS`` rows are left.
    """
    frame = inspection.frame[list(schema.columns)]
    complete = frame.dropna()
    rows_removed = len(frame) - len(complete)
    issues = [issue] if (issue := removal_issue(frame, rows_removed)) else []

    if len(complete) < MIN_USABLE_ROWS:
        raise DataEngineError(
            "too_few_rows",
            f"Only {len(complete):,} complete rows are left (at least {MIN_USABLE_ROWS} are needed). "
            "Removing the columns with many missing values may help.",
        )

    shuffled = np.random.default_rng(seed).permutation(len(complete))
    in_holdout = np.zeros(len(complete), dtype=bool)
    in_holdout[shuffled[: holdout_size(len(complete))]] = True
    train, holdout = complete[~in_holdout], complete[in_holdout]

    if len(train) < MIN_ROWS:
        issues.append(Issue(
            "few_rows",
            f"Only {len(train):,} rows are used for learning: the synthetic data may be less faithful "
            f"(more than {MIN_ROWS:,} rows are recommended).",
        ))
    return PreparedData(train, holdout, rows_read=len(frame), rows_removed=rows_removed, issues=issues)
