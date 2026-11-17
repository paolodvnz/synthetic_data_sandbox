"""Public data types of the Data Engine: issues, column roles, inspection result, errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Role = Literal["numeric", "categorical", "date", "id", "dropped"]
USABLE_ROLES: frozenset[str] = frozenset({"numeric", "categorical", "date"})
MASS_SLOTS = 2        # one-hot indicator of a point mass: [yes, no] after the continuous position
MASS_YES_SLOT = 0    # slot of "the row has the mass value"


@dataclass(frozen=True)
class Issue:
    """A non-blocking warning shown to the user.

    Attributes:
        code: Stable machine-readable identifier (tests and the UI rely on it).
        message: Plain-language explanation for a non-technical user.
        column: The column concerned, or ``None`` for file-level issues.
    """

    code: str
    message: str
    column: str | None = None


@dataclass(frozen=True)
class ColumnInfo:
    """Role proposed for one column, why, and the facts the decision was based on.

    Attributes:
        name: Column name.
        role: One of ``numeric``, ``categorical``, ``date``, ``id``, ``dropped``.
        reason: Short plain-language reason for the role.
        kind: What the cleaned values are: ``numeric``, ``text``, ``date`` or ``empty``.
        is_integer: True for numeric columns made only of whole numbers.
        n_distinct: Number of distinct non-null values.
        null_fraction: Share of missing values.
        n_minority: Rows (non-null) that differ from the most frequent value.
        date_format: How the dates of a date column are written in the file (``strftime``; ISO: ``%Y-%m-%d``),
            ``None`` for the other columns.
    """

    name: str
    role: Role
    reason: str
    kind: str
    is_integer: bool
    n_distinct: int
    null_fraction: float
    n_minority: int = 0
    date_format: str | None = None


@dataclass
class CsvInspection:
    """Result of reading a CSV and inferring its schema.

    No row is removed and every column is kept in ``frame``: dropped columns and suspected IDs are
    only *roles*, so the user can still change them before training.

    Attributes:
        frame: Cleaned values (stripped, nulls as NaN, numeric/date columns converted).
        columns: Role of every column, keyed by name, in file order.
        issues: Warnings collected along the pipeline.
        separator: Detected field separator.
        encoding: Encoding used to decode the file.
        unify_case: Whether case variants were merged while cleaning.
    """

    frame: pd.DataFrame
    columns: dict[str, ColumnInfo]
    issues: list[Issue] = field(default_factory=list)
    separator: str = ","
    encoding: str = "utf-8"
    unify_case: bool = True


class DataEngineError(Exception):
    """Blocking error: the file cannot be used. Carries a stable ``code`` and a user message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
