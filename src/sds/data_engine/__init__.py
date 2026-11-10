"""Data Engine: read a CSV, infer its schema, encode/decode it for the VAE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sds.data_engine.choices import FinalColumn, FinalSchema, UserChoices, apply_choices, fixed_reason
from sds.data_engine.cleaning import clean_column
from sds.data_engine.contract import SchemaContract, build_contract
from sds.data_engine.decoding import decode
from sds.data_engine.encoding import FeatureGroup, TabularEncoder, fit_encoder
from sds.data_engine.inference import infer_role
from sds.data_engine.preparation import PreparedData, prepare
from sds.data_engine.reading import read_table
from sds.data_engine.types import USABLE_ROLES, ColumnInfo, CsvInspection, DataEngineError, Issue

__all__ = [
    "ColumnInfo",
    "CsvInspection",
    "DataEngineError",
    "FeatureGroup",
    "FinalColumn",
    "FinalSchema",
    "Issue",
    "PreparedData",
    "SchemaContract",
    "TabularEncoder",
    "UserChoices",
    "apply_choices",
    "build_contract",
    "decode",
    "fit_encoder",
    "fixed_reason",
    "inspect_csv",
    "prepare",
]


def inspect_csv(source: str | Path | bytes, *, unify_case: bool = True) -> CsvInspection:
    """Read a CSV, clean every column and propose its role.

    Pipeline (schema contract): encoding -> separator -> malformed lines -> strip -> null tokens
    -> numbers (decimal comma if the separator is not ',') -> dates -> case unification -> roles.
    No row is removed: dropping rows with nulls happens at fit time, after the user has confirmed
    which columns to keep.

    Args:
        source: Path of the CSV file, or its raw bytes.
        unify_case: Merge text levels that differ only by case (default; can be turned off in Advanced).

    Returns:
        The cleaned frame, the role of every column and all warnings.

    Raises:
        DataEngineError: When the file cannot be used (see ``reading.read_table``), or with code
            ``no_usable_columns`` when every column is dropped or looks like an identifier.
    """
    raw = read_table(source)
    issues = list(raw.issues)
    decimal_comma = raw.separator != ","

    cleaned: dict[str, pd.Series] = {}
    columns: dict[str, ColumnInfo] = {}
    for name in raw.frame.columns:
        column = clean_column(raw.frame[name], decimal_comma=decimal_comma, unify_case_levels=unify_case)
        info, role_issues = infer_role(name, column)
        cleaned[name] = column.values
        columns[name] = info
        issues += column.issues + role_issues

    if not any(info.role in USABLE_ROLES for info in columns.values()):
        raise DataEngineError(
            "no_usable_columns",
            "No column can be used: they are all empty, constant, too incomplete or identifiers.",
        )

    return CsvInspection(
        frame=pd.DataFrame(cleaned),
        columns=columns,
        issues=issues,
        separator=raw.separator,
        encoding=raw.encoding,
        unify_case=unify_case,
    )
