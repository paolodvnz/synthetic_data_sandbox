"""Column level of the reading pipeline: strings -> typed values.

Order matters and follows the schema contract: strip -> null tokens -> numeric conversion (with
decimal comma) -> date detection -> case unification (text columns only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from sds.data_engine.thresholds_config import (
    MAX_LISTED_VALUES,
    MAX_NULL_MARKERS,
    NULL_TOKENS,
    NUMERIC_MAJORITY,
)
from sds.data_engine.types import Issue

Kind = Literal["numeric", "text", "date", "empty"]


@dataclass
class CleanColumn:
    """One column after cleaning.

    Attributes:
        values: Cleaned values, same length as the input; nulls are NaN/NaT.
        kind: ``numeric``, ``text``, ``date`` or ``empty`` (all values null).
        is_integer: True for numeric columns whose values are all whole numbers.
        issues: Warnings raised while cleaning.
    """

    values: pd.Series
    kind: Kind
    is_integer: bool = False
    issues: list[Issue] = field(default_factory=list)


def strip_and_nullify(raw: pd.Series) -> pd.Series:
    """Strip spaces and turn the null tokens into NaN."""
    stripped = raw.str.strip()
    return stripped.mask(stripped.str.lower().isin(NULL_TOKENS))


def unify_case(values: pd.Series, column: str) -> tuple[pd.Series, list[Issue]]:
    """Merge levels that differ only by letter case into their most frequent spelling.

    Ties are broken alphabetically, so the result is deterministic.
    """
    counts = values.value_counts()
    mapping: dict[str, str] = {}
    for _, group in counts.groupby(counts.index.str.lower()):
        if len(group) > 1:
            canonical = sorted(group.items(), key=lambda item: (-item[1], item[0]))[0][0]
            mapping |= {variant: canonical for variant in group.index if variant != canonical}
    if not mapping:
        return values, []
    merged = ", ".join(f"'{a}' -> '{b}'" for a, b in list(mapping.items())[:MAX_LISTED_VALUES])
    issue = Issue("case_unified", f"Values differing only by upper/lower case were merged: {merged}.", column)
    return values.replace(mapping), [issue]


def parse_numbers(values: pd.Series, decimal_comma: bool) -> pd.Series:
    """Parse to float (NaN where impossible); optionally accept a comma as decimal mark."""
    text = values.astype(object)
    if decimal_comma:
        text = text.where(text.isna(), text.astype(str).str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce").astype(float)


def to_numeric(values: pd.Series, column: str, decimal_comma: bool) -> tuple[pd.Series | None, list[Issue]]:
    """Convert to numbers if (almost) every non-null value parses.

    - All values parse -> numeric.
    - At least ``NUMERIC_MAJORITY`` parse and the others are at most ``MAX_NULL_MARKERS``
      different words -> numeric; those words are missing-value markers ("n.d.") and become NaN.
    - At least ``NUMERIC_MAJORITY`` parse but the others are many different words -> the column
      stays text: these are labels that mostly look like numbers (ICD-9 codes "250.83", "V57").
    - Otherwise -> text, silently.
    Both "mostly numeric" cases raise a warning listing the non-numeric values.

    Returns:
        (numbers or ``None`` when the column stays text, warnings).
    """
    present = values.notna()
    if not present.any():
        return None, []
    parsed = parse_numbers(values, decimal_comma)
    failed = present & parsed.isna()
    share_ok = 1 - failed.sum() / present.sum()
    if share_ok < NUMERIC_MAJORITY:
        return None, []
    if not failed.any():
        return parsed, []

    offenders = values[failed].unique()
    listed = ", ".join(f"'{v}'" for v in offenders[:MAX_LISTED_VALUES])
    n_failed = int(failed.sum())
    if len(offenders) <= MAX_NULL_MARKERS:
        message = f"Mostly numbers, but {n_failed} values are not ({listed}); they are treated as missing."
        return parsed, [Issue("mostly_numeric", message, column)]
    message = (
        f"Mostly numbers, but {n_failed} values are {len(offenders)} different words ({listed}, ...): "
        "the column is treated as text (codes), not as numbers."
    )
    return None, [Issue("mostly_numeric", message, column)]


def to_dates(values: pd.Series) -> pd.Series | None:
    """Convert to dates if every non-null value is an ISO 8601 date; otherwise ``None``."""
    present = values.notna()
    parsed = pd.to_datetime(values.astype(object), format="ISO8601", errors="coerce")
    if (present & parsed.isna()).any():
        return None
    return parsed


def clean_column(raw: pd.Series, *, decimal_comma: bool, unify_case_levels: bool) -> CleanColumn:
    """Run the column-level pipeline on one column of strings.

    Args:
        raw: Column as read from the file (all strings).
        decimal_comma: Also accept ``1,5`` as a number (only when the separator is not a comma).
        unify_case_levels: Merge levels differing only by case (``Male``/``male``).

    Returns:
        The cleaned column with its kind and warnings.
    """
    column = str(raw.name)
    values = strip_and_nullify(raw)
    if values.isna().all():
        return CleanColumn(values, "empty")

    numbers, issues = to_numeric(values, column, decimal_comma)
    if numbers is not None:
        present = numbers.dropna()
        is_integer = bool(np.all(present == np.floor(present)))
        return CleanColumn(numbers, "numeric", is_integer, issues)

    dates = to_dates(values)
    if dates is not None:
        return CleanColumn(dates, "date", issues=issues)

    if unify_case_levels:
        values, case_issues = unify_case(values, column)
        issues += case_issues
    return CleanColumn(values, "text", issues=issues)
