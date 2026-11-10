"""Role inference: decide how each cleaned column will be used (or why it is not).

Rules are applied in order; the first that matches wins (schema contract, "dtype + cardinality").
"""

from __future__ import annotations

from sds.data_engine.cleaning import CleanColumn
from sds.data_engine.thresholds_config import (
    ID_UNIQUENESS_RATIO,
    MAX_CATEGORIES,
    MAX_DISCRETE_LEVELS,
    MAX_NULL_FRACTION,
    MIN_LEVEL_ROWS,
)
from sds.data_engine.types import ColumnInfo, Issue


def base_role(kind: str, is_integer: bool, n_distinct: int) -> str:
    """Role of a usable column from its values alone (no drop or ID rules).

    Used for columns the user brings back: a re-included ID or dropped column follows the usual
    type rules but is never dropped again.
    """
    if kind == "date":
        return "date"
    if kind == "text":
        return "categorical"
    if is_integer and n_distinct <= MAX_DISCRETE_LEVELS:
        return "categorical"
    return "numeric"


def infer_role(name: str, column: CleanColumn) -> tuple[ColumnInfo, list[Issue]]:
    """Propose the role of one column and the warnings that explain it.

    Order: all null / constant / near constant (fewer than ``MIN_LEVEL_ROWS`` rows differ from the
    most frequent value: learned poorly, and those few rows could identify someone) / too many
    nulls -> dropped; date -> date (before the ID rule, since daily dates can be almost unique);
    integer or text almost unique -> suspected ID; text with too many levels -> dropped; integer
    with few values -> categorical (codes are always whole numbers); other numbers -> numeric;
    other text -> categorical.

    Args:
        name: Column name.
        column: The cleaned column.

    Returns:
        The column role and the related warnings.
    """
    values = column.values
    n_present = int(values.notna().sum())
    n_distinct = int(values.nunique())
    null_fraction = 1 - n_present / len(values) if len(values) else 1.0
    uniqueness = n_distinct / n_present if n_present else 0.0
    n_minority = n_present - int(values.value_counts().iloc[0]) if n_present else 0

    def info(role: str, reason: str) -> ColumnInfo:
        return ColumnInfo(name, role, reason, column.kind, column.is_integer, n_distinct,
                          round(null_fraction, 4), n_minority)

    def dropped(reason: str) -> tuple[ColumnInfo, list[Issue]]:
        issue = Issue("column_dropped", f"Column '{name}' will not be used: {reason}.", name)
        return info("dropped", reason), [issue]

    if column.kind == "empty":
        return dropped("all values are missing")
    if n_distinct == 1:
        return dropped("it has the same value in every row")
    if n_minority < MIN_LEVEL_ROWS:
        reason = f"only {n_minority} rows differ from the most common value"
        issue = Issue(
            "near_constant",
            f"Column '{name}' will not be used: {reason}. So few rows are learned poorly and could "
            "identify someone.",
            name,
        )
        return info("dropped", reason), [issue]
    if null_fraction > MAX_NULL_FRACTION:
        return dropped(f"{null_fraction:.0%} of its values are missing")
    if column.kind == "date":
        issue = Issue(
            "date_privacy",
            f"Column '{name}' contains dates, which can help identify people (e.g. date of birth). "
            "Consider removing it in Advanced.",
            name,
        )
        return info("date", "dates"), [issue]
    if (column.kind == "text" or column.is_integer) and uniqueness >= ID_UNIQUENESS_RATIO:
        reason = "almost every row has a different value, like an identifier"
        issue = Issue(
            "suspected_id",
            f"Column '{name}' looks like an identifier and is excluded. Include it in Advanced if it is not.",
            name,
        )
        return info("id", reason), [issue]
    if column.kind == "text":
        if n_distinct > MAX_CATEGORIES:
            return dropped(f"it has {n_distinct} different values (more than {MAX_CATEGORIES})")
        return info("categorical", "text with few distinct values"), []
    if column.is_integer and n_distinct <= MAX_DISCRETE_LEVELS:
        return info("categorical", f"whole numbers with only {n_distinct} distinct values"), []
    return info("numeric", "numbers"), []
