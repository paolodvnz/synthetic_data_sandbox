"""Column level of the reading pipeline: strings -> typed values.

Order matters and follows the schema contract: strip -> null tokens -> numeric conversion (with
decimal comma) -> date detection -> case unification (text columns only).

Dates: ISO, or one of ``DATE_FORMATS`` (day/month/year or month/day/year, ``/``, ``-`` or ``.``, four-digit year). A
column is a date column when its values are real dates in **one** format; a few marker words become missing, as
for the numbers. A column of date-shaped values that is not one format (mixed formats, impossible dates, or every
value readable both as day/month and as month/day) stays text and says why (``date_problem``): the role inference
drops it with a clear warning instead of treating it as ordinary text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from sds.data_engine.thresholds_config import (
    DATE_FORMATS,
    DATE_SHAPE,
    ISO_DATE_FORMAT,
    MAX_LISTED_VALUES,
    MAX_NULL_MARKERS,
    NULL_TOKENS,
    NUMERIC_MAJORITY,
)
from sds.data_engine.types import Issue

Kind = Literal["numeric", "text", "date", "empty"]
DateProblem = Literal["date_ambiguous", "date_unreadable"]


@dataclass
class CleanColumn:
    """One column after cleaning.

    Attributes:
        values: Cleaned values, same length as the input; nulls are NaN/NaT.
        kind: ``numeric``, ``text``, ``date`` or ``empty`` (all values null).
        is_integer: True for numeric columns whose values are all whole numbers.
        issues: Warnings raised while cleaning.
        date_format: Format of a date column (``strftime``), else ``None``.
        date_problem: Why a text column of date-shaped values is not a date column, else ``None``.
    """

    values: pd.Series
    kind: Kind
    is_integer: bool = False
    issues: list[Issue] = field(default_factory=list)
    date_format: str | None = None
    date_problem: DateProblem | None = None


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
      stays text, silently: these are labels that mostly look like numbers (ICD-9 codes "250.83", "V57").
    - Otherwise -> text, silently.
    The warning (listing the non-numeric values) is raised only when values are changed, i.e. turned into missing.

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
    if len(offenders) > MAX_NULL_MARKERS:
        return None, []
    listed = ", ".join(f"'{v}'" for v in offenders[:MAX_LISTED_VALUES])
    message = f"Mostly numbers, but {int(failed.sum())} values are not ({listed}); they are treated as missing."
    return parsed, [Issue("mostly_numeric", message, column)]


@dataclass
class DateReading:
    """What the date detection found in a text column.

    Attributes:
        dates: The parsed dates (markers and missing values as NaT), or ``None`` when it is not a date column.
        date_format: Their format, or ``None``.
        problem: Why date-shaped values are not a date column, or ``None``.
        issues: The ``mostly_dates`` warning, if any.
    """

    dates: pd.Series | None = None
    date_format: str | None = None
    problem: DateProblem | None = None
    issues: list[Issue] = field(default_factory=list)


def parse_dates(values: pd.Series, date_format: str) -> pd.Series:
    """Dates in one format (``ISO_DATE_FORMAT`` accepts every ISO 8601 form, as before); NaT where a value does not fit."""
    fmt = "ISO8601" if date_format == ISO_DATE_FORMAT else date_format
    return pd.to_datetime(values.astype(object), format=fmt, errors="coerce")


def one_format(values: pd.Series, present: pd.Series) -> tuple[pd.Series | None, str | None, DateProblem | None]:
    """The only format every present value fits: (dates, format, None), or (None, None, the problem)."""
    fits = {}
    for date_format in (ISO_DATE_FORMAT, *DATE_FORMATS):
        parsed = parse_dates(values, date_format)
        if not (present & parsed.isna()).any():
            fits[date_format] = parsed
    readings = list(fits.items())
    if not readings:
        return None, None, "date_unreadable"
    first_format, first = readings[0]
    if any(not parsed.equals(first) for _, parsed in readings[1:]):     # day/month and month/day differ
        return None, None, "date_ambiguous"
    return first, first_format, None


def to_dates(values: pd.Series, column: str) -> DateReading:
    """Read a text column as dates, the way ``to_numeric`` reads numbers.

    - At least ``NUMERIC_MAJORITY`` of the values have the shape of a date (ISO, or ``DATE_SHAPE``) and the others
      are at most ``MAX_NULL_MARKERS`` different words -> the words are missing-value markers (``mostly_dates``);
    - more different words, or fewer date-shaped values -> text, silently (no value is changed).
    Then the date-shaped values must be real dates in one format: otherwise the problem is returned.
    """
    present = values.notna()
    if not present.any():
        return DateReading()
    shaped = present & (parse_dates(values, ISO_DATE_FORMAT).notna()
                        | values.astype("string").str.fullmatch(DATE_SHAPE).fillna(False).astype(bool))
    share = shaped.sum() / present.sum()
    if share < NUMERIC_MAJORITY:
        return DateReading()
    issues = []
    others = present & ~shaped
    if others.any():
        offenders = values[others].unique()
        if len(offenders) > MAX_NULL_MARKERS:
            return DateReading()
        listed = ", ".join(f"'{v}'" for v in offenders[:MAX_LISTED_VALUES])
        message = f"Mostly dates, but {int(others.sum())} values are not ({listed}); they are treated as missing."
        issues.append(Issue("mostly_dates", message, column))
        present = shaped                     # the markers are not dates: they parse as NaT, i.e. missing
    dates, date_format, problem = one_format(values, present)
    return DateReading(dates, date_format, problem, issues)


def time_issues(dates: pd.Series, column: str) -> list[Issue]:
    """The ``time_dropped`` warning when some date has a time of day: the Data Engine works in whole days."""
    present = dates.dropna()
    if (present == present.dt.normalize()).all():
        return []
    message = f"Column '{column}' also has times of day: only the date is kept, the synthetic data have no times."
    return [Issue("time_dropped", message, column)]


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

    reading = to_dates(values, column)
    issues += reading.issues
    if reading.dates is not None:
        issues += time_issues(reading.dates, column)
        return CleanColumn(reading.dates, "date", issues=issues, date_format=reading.date_format)

    if unify_case_levels:
        values, case_issues = unify_case(values, column)
        issues += case_issues
    return CleanColumn(values, "text", issues=issues, date_problem=reading.problem)
