"""The Upload step without Streamlit: user choices from the table and Advanced, warnings grouped (pure functions)."""

from __future__ import annotations

import pandas as pd

USABLE_ROLES = frozenset({"numeric", "categorical", "date"})       # roles the engine keeps by default
ROLE_LABELS = {"numeric": "quantity", "categorical": "category", "date": "date", "id": "identifier?",
               "dropped": "left out"}
READINGS = {"quantity": "numeric", "category": "categorical"}      # label of the switch -> role
CORRECTION_FIELD = {"numeric": "as_numeric", "categorical": "as_categorical"}

# Issues shown in view: they may need the user to act (an identifier, too few rows, a file read with
# difficulty...). Every other issue is a note, listed under "Details".
ATTENTION_CODES = frozenset({"suspected_id", "date_privacy", "few_rows", "many_rows_removed", "missing_header",
                             "encoding_fallback", "bad_lines_skipped", "constant_in_train", "all_levels_rare"})
PHASES = ("file", "choices", "rows", "encoding")
MISSING_DECIMALS = 1                                               # "12.3%" of missing values in the table


def kept_by_engine(column: dict) -> bool:
    """Whether the engine uses ``column`` unless the user says otherwise."""
    return column["role"] in USABLE_ROLES


def current_readings(columns: list[dict]) -> dict[str, str]:
    """The role each column is read as now: the applied one if kept, else the one it would have if kept."""
    return {c["name"]: c["final_role"] or c["kept_role"] for c in columns}


def desired_choices(columns: list[dict], keep: dict[str, bool], readings: dict[str, str], unify_case: bool) -> dict:
    """The ``UserChoices.to_dict`` form of what the user sees on the page.

    Args:
        columns: ``columns`` of the dataset view.
        keep: Column name -> ticked in the table.
        readings: Column name -> role picked in Advanced (only columns without a ``fixed_reason`` count).
        unify_case: The case option.
    """
    choices = {"excluded": [], "included": [], "as_categorical": [], "as_numeric": [], "unify_case": unify_case}
    for column in columns:
        name = column["name"]
        if kept_by_engine(column) and not keep[name]:
            choices["excluded"].append(name)
        if not kept_by_engine(column) and keep[name]:
            choices["included"].append(name)
        if keep[name] and column["fixed_reason"] is None and readings[name] != column["kept_role"]:
            choices[CORRECTION_FIELD[readings[name]]].append(name)
    return {key: sorted(value) if isinstance(value, list) else value for key, value in choices.items()}


def all_issues(issues: dict) -> list[dict]:
    """The issues of every phase, in the order of the phases."""
    return [issue for phase in PHASES for issue in issues.get(phase, [])]


def attention_issues(issues: dict) -> list[dict]:
    """The issues shown in view."""
    return [issue for issue in all_issues(issues) if issue["code"] in ATTENTION_CODES]


def detail_issues(issues: dict) -> list[dict]:
    """The notes, listed under "Details"."""
    return [issue for issue in all_issues(issues) if issue["code"] not in ATTENTION_CODES]


def notes_by_column(issues: dict) -> dict[str, list[str]]:
    """Column name -> messages about it (shown in the "Notes" column of the table)."""
    notes: dict[str, list[str]] = {}
    for issue in all_issues(issues):
        if issue["column"] is not None:
            notes.setdefault(issue["column"], []).append(issue["message"])
    return notes


def missing_label(fraction: float) -> str:
    """Share of missing values as the table shows it: a few never read "0.0%", nor almost all "100.0%"."""
    smallest = 10 ** -MISSING_DECIMALS                             # in percent
    percent = 100 * fraction
    if percent == 0 or percent == 100:
        return f"{percent:.0f}%"
    if percent < smallest:
        return f"<{smallest:g}%"
    if percent > 100 - smallest:
        return f">{100 - smallest:g}%"
    return f"{percent:.{MISSING_DECIMALS}f}%"


def columns_frame(data: dict) -> pd.DataFrame:
    """The table of columns of the Upload step: a "Keep" tick, the reading, the missing values, why, examples, notes."""
    notes = notes_by_column(data["issues"])
    return pd.DataFrame([{
        "Keep": c["final_role"] is not None,
        "Column": c["name"],
        "We read it as": ROLE_LABELS[c["final_role"] or c["role"]],
        "Missing": missing_label(c["null_fraction"]),
        "Why": c["reason"],
        "Examples": ", ".join(c["examples"]),
        "Notes": " ".join(notes.get(c["name"], [])),
    } for c in data["columns"]])
