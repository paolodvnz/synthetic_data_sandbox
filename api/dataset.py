"""Upload and user choices: from the CSV bytes to a dataset ready for training.

Pure functions on a ``Session`` (no HTTP), so they can be tested directly. The view sent to the UI describes
the columns, the warnings and the row counts: never a row of the real data.
"""

from __future__ import annotations

import re

from api.api_config import DEFAULT_SOURCE_NAME, MAX_SOURCE_NAME_CHARS, N_EXAMPLES
from api.serialization import to_jsonable
from api.session import Session, Stage
from sds.data_engine import (
    DataEngineError,
    UserChoices,
    apply_choices,
    build_contract,
    fit_encoder,
    fixed_reason,
    inspect_csv,
    prepare,
)
from sds.data_engine.inference import base_role
from sds.data_engine.types import USABLE_ROLES

CASE_MERGE_CODE = "case_unified"


def source_name(name: str | None) -> str:
    """The file name alone (no folders, Windows or Unix), at most ``MAX_SOURCE_NAME_CHARS`` characters."""
    plain = re.split(r"[\\/]", name or "")[-1].strip()
    return plain[-MAX_SOURCE_NAME_CHARS:] if plain else DEFAULT_SOURCE_NAME


def fit(session: Session) -> None:
    """Prepare and encode with the current schema; on a blocking problem, stay at ``Stage.INSPECTED``.

    The problem is kept in ``session.problem`` for the view (it is not an HTTP error: the user fixes it by
    changing the choices).
    """
    session.clear_from(Stage.PREPARED)
    session.problem = None
    try:
        prepared = prepare(session.inspection, session.schema)
        encoder = fit_encoder(prepared.train, session.schema)
        contract = build_contract(session.inspection, session.schema, prepared, encoder)
    except DataEngineError as error:
        session.problem = {"code": error.code, "message": error.message}
        return
    session.prepared, session.encoder, session.contract = prepared, encoder, contract


def load_dataset(session: Session, data: bytes, name: str | None) -> None:
    """Load a CSV into the empty session, inspected and prepared with the default choices.

    Raises:
        ApiError: 409 ``training_running`` or ``dataset_loaded`` (one dataset per session: Start over first);
            the session is then unchanged.
        DataEngineError: When the file cannot be used; the session is then empty.
    """
    with session.lock:
        session.require_idle()
        session.require_empty()
        session.reset()
        inspection = inspect_csv(data)
        choices = UserChoices()
        session.source_name, session.source_bytes = source_name(name), data
        session.inspection, session.choices = inspection, choices
        session.case_merges = [i for i in inspection.issues if i.code == CASE_MERGE_CODE]
        session.schema = apply_choices(inspection, choices)
        fit(session)


def change_choices(session: Session, data: dict) -> None:
    """Apply the user's choices (``UserChoices.to_dict`` form) and prepare again.

    The inspection is redone from the stored bytes only when the case option changes.

    Raises:
        ApiError: 409 ``training_running`` or ``no_dataset``.
        DataEngineError: ``invalid_choices`` / ``no_usable_columns``; the session is then unchanged.
    """
    with session.lock:
        session.require_idle()
        session.require(Stage.INSPECTED)
        choices = UserChoices.from_dict(data)
        inspection = session.inspection
        if choices.unify_case != inspection.unify_case:
            inspection = inspect_csv(session.source_bytes, unify_case=choices.unify_case)
        schema = apply_choices(inspection, choices)
        session.inspection, session.choices, session.schema = inspection, choices, schema
        fit(session)


def example_text(value) -> str:
    """A value as the user wrote it: whole numbers without ``.0`` (a missing value makes pandas read them as floats)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def column_view(session: Session, name: str) -> dict:
    """What the UI shows about one column (its values only as a few distinct examples)."""
    info = session.inspection.columns[name]
    final = session.schema.columns.get(name)
    examples = session.inspection.frame[name].dropna().unique()[:N_EXAMPLES]
    kept_role = info.role if info.role in USABLE_ROLES else base_role(info.kind, info.n_distinct)
    return {
        "name": name,
        "role": info.role,
        "reason": info.reason,
        "kept_role": kept_role,
        "fixed_reason": fixed_reason(info),
        "null_fraction": info.null_fraction,
        "examples": [example_text(value) for value in examples],
        "final_role": final.role if final else None,
        "changed_by_user": bool(final and final.changed_by_user),
    }


def dataset_view(session: Session) -> dict:
    """JSON view of the current dataset: file, columns, choices, warnings, row counts, readiness.

    Raises:
        ApiError: 409 ``no_dataset``.
    """
    with session.lock:
        session.require(Stage.INSPECTED)
        inspection, prepared, ready = session.inspection, session.prepared, session.stage >= Stage.PREPARED
        view = {
            "file": {"name": session.source_name, "separator": inspection.separator,
                     "encoding": inspection.encoding, "n_rows": len(inspection.frame)},
            "columns": [column_view(session, name) for name in inspection.columns],
            "choices": session.choices.to_dict(),
            "case_merges": session.case_merges,
            "issues": {"file": inspection.issues, "choices": session.schema.issues,
                       "rows": prepared.issues if ready else [], "encoding": session.encoder.issues if ready else []},
            "rows": {"read": prepared.rows_read, "removed": prepared.rows_removed, "train": len(prepared.train),
                     "holdout": len(prepared.holdout)} if ready else None,
            "ready": ready,
            "problem": session.problem,
        }
        return to_jsonable(view)
