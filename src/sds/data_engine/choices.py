"""User choices: the engine proposes, the user confirms.

The user decides what a column *means* (is it an identifier? are these numbers codes? is the date
needed?); the engine decides how it is *distributed* and explains its proposals. Any proposal can be
overridden, except including a column with no values at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from sds.data_engine.inference import base_role
from sds.data_engine.thresholds_config import MAX_CATEGORIES, MAX_NULL_FRACTION, MIN_LEVEL_ROWS, YES_NO_LEVELS
from sds.data_engine.types import USABLE_ROLES, ColumnInfo, CsvInspection, DataEngineError, Issue

_SET_FIELDS = ("excluded", "included", "as_categorical", "as_numeric")


@dataclass
class UserChoices:
    """What the user changed with respect to the engine's proposal.

    Attributes:
        excluded: Columns the user unchecked.
        included: Suspected IDs or engine-dropped columns the user wants to keep.
        as_categorical: Numeric columns to treat as codes/categories.
        as_numeric: Numeric categorical columns (codes) to treat as quantities.
        unify_case: Merge text values differing only by case (needs a new inspection if changed).
    """

    excluded: set[str] = field(default_factory=set)
    included: set[str] = field(default_factory=set)
    as_categorical: set[str] = field(default_factory=set)
    as_numeric: set[str] = field(default_factory=set)
    unify_case: bool = True

    def to_dict(self) -> dict:
        """JSON-ready representation (sets become sorted lists, for a stable output)."""
        return {**{name: sorted(getattr(self, name)) for name in _SET_FIELDS}, "unify_case": self.unify_case}

    @classmethod
    def from_dict(cls, data: dict) -> UserChoices:
        """Build from the JSON representation, rejecting unknown fields and wrong types."""
        known = {f.name for f in fields(cls)}
        problems = [f"unknown field '{key}'" for key in data if key not in known]
        for name in _SET_FIELDS:
            value = data.get(name, [])
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                problems.append(f"'{name}' must be a list of column names")
        if not isinstance(data.get("unify_case", True), bool):
            problems.append("'unify_case' must be true or false")
        if problems:
            raise DataEngineError("invalid_choices", "Invalid choices: " + "; ".join(problems) + ".")
        return cls(**{name: set(data.get(name, [])) for name in _SET_FIELDS},
                   unify_case=data.get("unify_case", True))


@dataclass(frozen=True)
class FinalColumn:
    """A column that will be used, with its final role.

    Attributes:
        name: Column name.
        role: ``numeric``, ``categorical`` or ``date``.
        changed_by_user: True when the user brought the column back or changed its role.
    """

    name: str
    role: str
    changed_by_user: bool


@dataclass
class FinalSchema:
    """Columns to use (file order) and the consequences of the user's choices."""

    columns: dict[str, FinalColumn]
    issues: list[Issue] = field(default_factory=list)


def reinclusion_consequence(info: ColumnInfo) -> str:
    """Plain-language consequence of keeping a column the engine proposed to leave out."""
    name = info.name
    if info.role == "id":
        return f"If '{name}' really identifies people, the synthetic data may contain real identifiers."
    if info.n_distinct == 1:
        return f"'{name}' has the same value in every row: it adds no information."
    if info.n_minority < MIN_LEVEL_ROWS:
        rows = "1 row differs" if info.n_minority == 1 else f"{info.n_minority} rows differ"
        return f"Only {rows} in '{name}': they could identify someone."
    if info.null_fraction > MAX_NULL_FRACTION:
        return f"Keeping '{name}' removes at least {info.null_fraction:.0%} of the rows (those where it is missing)."
    if info.kind == "text" and info.n_distinct > MAX_CATEGORIES:
        return (f"'{name}' has {info.n_distinct} different values: rare ones are learned poorly and could "
                "identify someone.")
    return f"'{name}' was proposed for removal ({info.reason})."


def check_names(inspection: CsvInspection, choices: UserChoices) -> list[str]:
    """Problems with column names and contradictory choices."""
    problems = []
    for name in _SET_FIELDS:
        for column in sorted(getattr(choices, name) - set(inspection.columns)):
            problems.append(f"there is no column '{column}'")
    for column in sorted(choices.excluded & choices.included):
        problems.append(f"'{column}' cannot be both removed and kept")
    for column in sorted(choices.as_categorical & choices.as_numeric):
        problems.append(f"'{column}' cannot be both categories and quantities")
    for column in sorted(choices.included):
        info = inspection.columns.get(column)
        if info is not None and info.kind == "empty":
            problems.append(f"'{column}' has no values at all, so it cannot be kept")
    return problems


def fixed_reason(info: ColumnInfo) -> str | None:
    """Why the reading of a column cannot change, or None if the user may switch it.

    Only numbers (whole or not) with 3 to ``MAX_CATEGORIES`` values are ambiguous (codes or
    quantities?). With two values both readings decode to the same yes/no column; with more values
    One-Hot becomes too large. The message is the plain-language reason shown to the user.
    """
    if info.kind == "empty":
        return "it has no values"
    if info.kind == "date":
        return "it contains dates: it can only be kept or removed"
    if info.kind == "text":
        return "it contains text, so it cannot be treated as a quantity"
    if info.n_distinct <= YES_NO_LEVELS:
        return "it has only two values (like yes/no): both readings give the same result"
    if info.n_distinct > MAX_CATEGORIES:
        return f"it has {info.n_distinct} different values, too many to be categories (max {MAX_CATEGORIES})"
    return None


def corrected_role(info: ColumnInfo, role: str, choices: UserChoices) -> tuple[str, str | None]:
    """Apply a role correction to a kept column. Returns (role, problem or None).

    Asking for the role the column already has is always accepted (the UI may send it back).
    """
    if info.name in choices.as_categorical:
        target = "categorical"
    elif info.name in choices.as_numeric:
        target = "numeric"
    else:
        return role, None
    if target == role:
        return role, None
    reason = fixed_reason(info)
    if reason:
        return role, f"'{info.name}' cannot change: {reason}"
    return target, None


def apply_choices(inspection: CsvInspection, choices: UserChoices) -> FinalSchema:
    """Combine the engine's proposal with the user's choices into the final schema.

    Args:
        inspection: Result of ``inspect_csv``, made with the same ``unify_case`` as ``choices``.
        choices: What the user changed.

    Returns:
        The columns to use, in file order, and the consequences of bringing columns back.

    Raises:
        DataEngineError: ``stale_inspection`` if ``unify_case`` differs from the inspection;
            ``invalid_choices`` listing every problem; ``no_usable_columns`` if nothing is left.
    """
    if choices.unify_case != inspection.unify_case:
        raise DataEngineError(
            "stale_inspection", "The case option changed: the file must be inspected again."
        )

    problems = check_names(inspection, choices)
    columns: dict[str, FinalColumn] = {}
    issues: list[Issue] = []
    for name, info in inspection.columns.items():
        if name in choices.excluded:
            continue
        brought_back = name in choices.included and info.role not in USABLE_ROLES
        if info.role in USABLE_ROLES:
            role = info.role
        elif brought_back and info.kind != "empty":
            role = base_role(info.kind, info.n_distinct)
            issues.append(Issue("column_reincluded", reinclusion_consequence(info), name))
        else:
            continue
        new_role, problem = corrected_role(info, role, choices)
        if problem:
            problems.append(problem)
        columns[name] = FinalColumn(name, new_role, changed_by_user=brought_back or new_role != info.role)

    if problems:
        raise DataEngineError(
            "invalid_choices", "Some choices cannot be applied: " + "; ".join(problems) + "."
        )
    if not columns:
        raise DataEngineError("no_usable_columns", "No column is left: keep at least one column.")
    return FinalSchema(columns=columns, issues=issues)
