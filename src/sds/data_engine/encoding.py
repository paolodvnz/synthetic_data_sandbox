"""Encoding: turn the train / holdout tables into the vectors the VAE learns from.

Fit on the train rows only (no leakage), with scikit-learn: one transformer per column, placed one
after the other in the vector; the positions of each column form the index map needed by the mixed
loss. No ``ColumnTransformer``: its ``fit`` builds and discards the whole float64 table, and its
``transform`` stacks float64 copies before the cast (8.6 GB for a 200 MB CSV). Here each column is
fit without producing output, and rows are encoded in blocks straight into one float32 array.
- numeric: ``MinMaxScaler``;
- date: days since the first train date, then ``MinMaxScaler``;
- categorical: ``OneHotEncoder``. Levels with fewer than ``MIN_LEVEL_ROWS`` train rows are merged
  into one "Other" slot (they are learned poorly and could identify someone), and a column has at
  most ``MAX_CATEGORIES`` slots. A holdout level never seen in train goes to "Other" when the
  column has one, else to all zeros. Holdout numbers outside the train range are not clipped.
Columns that cannot be learned from the train rows (one value only, or only rare levels) get no
positions in the vector. The encoder lives in memory for the session; the JSON contract is built
from its fitted parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sds.data_engine.choices import FinalSchema
from sds.data_engine.thresholds_config import (
    ENCODE_BLOCK_ROWS,
    MAX_CATEGORIES,
    MAX_LISTED_VALUES,
    MIN_LEVEL_ROWS,
)
from sds.data_engine.types import DataEngineError, Issue


@dataclass(frozen=True)
class FeatureGroup:
    """Positions of one column in the encoded vector.

    Attributes:
        name: Column name.
        role: ``numeric``, ``categorical`` or ``date``.
        start: First position in the vector.
        stop: Position after the last one (``vector[start:stop]`` is the column).
        levels: Kept levels of a categorical column, in slot order and original type (int stays
            int); the "Other" slot is not included.
        has_other: True when the group ends with the "Other" slot.
        merged: Levels merged into "Other" with their number of train rows (decoding of integer
            columns picks among them with these proportions).
    """

    name: str
    role: str
    start: int
    stop: int
    levels: tuple = ()
    has_other: bool = False
    merged: dict = field(default_factory=dict)


@dataclass
class TabularEncoder:
    """Encoder fit on the train rows: same parameters for train, holdout and decoding.

    Attributes:
        groups: Index map, one group per encoded column, in schema order.
        width: Length of the encoded vector.
        issues: Warnings about rare levels and columns that could not be learned.
        transformers: The fitted scikit-learn transformer of every group, in the same order.
        date_origins: First train date of every date column (day 0).
        integer_codes: Categorical columns made of whole numbers (encoded as int).
    """

    groups: list[FeatureGroup]
    width: int
    issues: list[Issue]
    transformers: list[OneHotEncoder | MinMaxScaler]
    date_origins: dict[str, pd.Timestamp]
    integer_codes: frozenset[str]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode rows (train or holdout) with the train parameters.

        Rows go by blocks of ``ENCODE_BLOCK_ROWS``, written straight into the result: memory is the
        float32 output plus one block, whatever the number of rows.

        Args:
            frame: Rows with at least the encoded columns, without missing values.

        Returns:
            Array of shape ``(len(frame), width)``, float32.
        """
        vectors = np.empty((len(frame), self.width), dtype=np.float32)
        with warnings.catch_warnings():
            # Unknown holdout levels without an "Other" slot become all zeros: expected, not a problem.
            warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
            for start in range(0, len(frame), ENCODE_BLOCK_ROWS):
                block = frame.iloc[start:start + ENCODE_BLOCK_ROWS]
                rows = slice(start, start + len(block))
                for group, transformer in zip(self.groups, self.transformers, strict=True):
                    values = model_input(block[group.name], self.date_origins, self.integer_codes)
                    vectors[rows, group.start:group.stop] = transformer.transform(values)
        return vectors

    def category_slots(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """One-Hot slot of every categorical column, without encoding the whole vector.

        Args:
            frame: Rows with at least the encoded columns, without missing values (train rows: every
                row has exactly one slot set).

        Returns:
            Column name -> slot index per row (int64), for the categorical groups in vector order.
        """
        slots = {}
        for group, transformer in zip(self.groups, self.transformers, strict=True):
            if group.role != "categorical":
                continue
            column = np.empty(len(frame), dtype=np.int64)
            for start in range(0, len(frame), ENCODE_BLOCK_ROWS):
                block = frame[group.name].iloc[start:start + ENCODE_BLOCK_ROWS]
                values = model_input(block, self.date_origins, self.integer_codes)
                column[start:start + len(block)] = transformer.transform(values).argmax(axis=1)
            slots[group.name] = column
        return slots

    def value_range(self, name: str) -> tuple[float, float]:
        """Train minimum and maximum of a numeric or date column (dates in days), as scaled.

        Args:
            name: Name of an encoded numeric or date column.

        Returns:
            ``(min, max)`` used by the MinMax scaling.
        """
        index = next(i for i, g in enumerate(self.groups) if g.name == name)
        scaler = self.transformers[index]
        return float(scaler.data_min_[0]), float(scaler.data_max_[0])


def model_input(column: pd.Series, date_origins: dict[str, pd.Timestamp],
                integer_codes: frozenset[str]) -> pd.DataFrame:
    """One column as its scikit-learn transformer expects it: dates as days, codes as int.

    A one-column frame, so the transformer keeps the column name it was fit with.
    """
    if column.name in date_origins:
        column = (pd.to_datetime(column) - date_origins[column.name]) / pd.Timedelta(days=1)
    elif column.name in integer_codes:
        column = column.astype("int64")
    return column.to_frame()


def python_value(value: object) -> object:
    """NumPy scalar -> plain Python value (ints stay ints): the contract must be JSON."""
    return value.item() if isinstance(value, np.generic) else value


def unlearnable_issue(name: str, role: str, counts: pd.Series) -> Issue | None:
    """Why a column cannot be learned from the train rows, or None."""
    if len(counts) == 1:
        return Issue(
            "constant_in_train",
            f"Column '{name}' has the same value in every training row: it will not be generated.",
            name,
        )
    if role == "categorical" and counts.max() < MIN_LEVEL_ROWS:
        return Issue(
            "all_levels_rare",
            f"Every value of '{name}' appears in fewer than {MIN_LEVEL_ROWS} rows: it cannot be "
            "learned and will not be generated.",
            name,
        )
    return None


def merged_issue(name: str, merged: dict, integer: bool) -> Issue:
    """Plain-language warning about the rare levels merged into "Other"."""
    shown = ", ".join(f"'{level}'" for level in list(merged)[:MAX_LISTED_VALUES])
    more = f" and {len(merged) - MAX_LISTED_VALUES} more" if len(merged) > MAX_LISTED_VALUES else ""
    if len(merged) == 1:
        outcome = ("it is generated without its link to the other columns" if integer
                   else "it appears as 'Other' in the synthetic data")
        message = f"{shown} is rare in '{name}' (fewer than {MIN_LEVEL_ROWS} rows): {outcome}."
    else:
        outcome = ("the synthetic data picks among them at random" if integer
                   else "they appear as 'Other' in the synthetic data")
        message = f"{len(merged)} rare values of '{name}' ({shown}{more}) are learned together: {outcome}."
    return Issue("levels_merged", message, name)


def nothing_to_learn_error(unlearnable: list[Issue]) -> DataEngineError:
    """Blocking error when no column of the schema can be learned, naming each one and why."""
    reasons = {
        "constant_in_train": "the same value in every training row",
        "all_levels_rare": f"every value appears in fewer than {MIN_LEVEL_ROWS} rows",
    }
    listed = "; ".join(f"'{issue.column}' ({reasons[issue.code]})" for issue in unlearnable)
    return DataEngineError(
        "nothing_to_learn",
        f"None of the chosen columns can be learned from the training rows: {listed}. Keep more columns.",
    )


def fit_encoder(train: pd.DataFrame, schema: FinalSchema) -> TabularEncoder:
    """Fit the encoder on the train rows.

    Args:
        train: Train rows from ``prepare`` (schema columns, no missing values).
        schema: Final schema from ``apply_choices``: which columns, with which role.

    Returns:
        The fitted encoder, its index map and the warnings to show to the user.

    Raises:
        DataEngineError: ``nothing_to_learn`` when no column can be learned from the train rows
            (an empty vector would break the VAE).
    """
    issues: list[Issue] = []
    kept = []
    for name, column in schema.columns.items():
        issue = unlearnable_issue(name, column.role, train[name].value_counts())
        if issue:
            issues.append(issue)
        else:
            kept.append(column)
    if not kept:
        raise nothing_to_learn_error(issues)

    date_origins = {c.name: train[c.name].min() for c in kept if c.role == "date"}
    integer_codes = frozenset(
        c.name for c in kept if c.role == "categorical" and pd.api.types.is_numeric_dtype(train[c.name])
    )
    groups, transformers, start = [], [], 0
    for column in kept:
        values = model_input(train[column.name], date_origins, integer_codes)
        transformer = column_transformer(column.role).fit(values)
        stop = start + len(transformer.get_feature_names_out())
        levels, merged = (), {}
        if column.role == "categorical":
            infrequent = transformer.infrequent_categories_[0]
            rare = set() if infrequent is None else {python_value(v) for v in infrequent}
            levels = tuple(python_value(v) for v in transformer.categories_[0] if python_value(v) not in rare)
            if rare:
                counts = values[column.name].value_counts()
                merged = {python_value(level): int(n) for level, n in counts.items() if python_value(level) in rare}
                issues.append(merged_issue(column.name, merged, column.name in integer_codes))
        groups.append(FeatureGroup(
            name=column.name,
            role=column.role,
            start=start,
            stop=stop,
            levels=levels,
            has_other=bool(merged),
            merged=merged,
        ))
        transformers.append(transformer)
        start = stop

    return TabularEncoder(groups, start, issues, transformers, date_origins, integer_codes)


def column_transformer(role: str) -> OneHotEncoder | MinMaxScaler:
    """Unfitted scikit-learn transformer of one column: One-Hot for categorical, MinMax otherwise."""
    if role == "categorical":
        return OneHotEncoder(
            min_frequency=MIN_LEVEL_ROWS,
            max_categories=MAX_CATEGORIES,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )
    return MinMaxScaler()
