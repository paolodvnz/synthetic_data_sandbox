"""Specification tests for the reading + type-inference stage of ``sds.data_engine``.

They describe the schema contract on the torture fixture and on tiny inline files.

Public API under test:
    inspect_csv(source: Path | bytes, *, unify_case: bool = True) -> CsvInspection
        .frame    pandas DataFrame after the reading pipeline (stripped, nulls as NaN, typed)
        .columns  dict[str, ColumnInfo]  with ColumnInfo.role in
                  {"numeric", "categorical", "date", "id", "dropped"} and ColumnInfo.reason
        .issues   list[Issue]            with Issue.code, Issue.column (or None), Issue.message
    DataEngineError(code, message)       raised for blocking errors

Tests assert on stable codes, never on the user-facing messages.
"""

from pathlib import Path

import pandas as pd
import pytest

import make_torture_csv
from sds.data_engine import DataEngineError, inspect_csv
from sds.data_engine.thresholds_config import MIN_LEVEL_ROWS

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TORTURE_CSV = FIXTURES / "torture_columns.csv"


def issue_codes(inspection, column=None) -> set[str]:
    """Codes of the issues raised (optionally only those about ``column``)."""
    return {i.code for i in inspection.issues if column is None or i.column == column}


# --------------------------------------------------------------------------------------------
# Fixture integrity
# --------------------------------------------------------------------------------------------

def test_torture_csv_is_up_to_date(tmp_path):
    fresh = make_torture_csv.write_csv(tmp_path / "fresh.csv")
    assert fresh.read_bytes() == TORTURE_CSV.read_bytes(), "run tests/fixtures/make_torture_csv.py"


# --------------------------------------------------------------------------------------------
# Column level: one column, one expected outcome
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def torture():
    return inspect_csv(TORTURE_CSV)


EXPECTED_ROLES = {
    "numeric_float": "numeric",
    "numeric_int_wide": "numeric",
    "cat_str": "categorical",
    "cat_int_codes": "categorical",
    "id_int": "id",
    "id_str": "id",
    "high_card_str": "dropped",
    "constant": "dropped",
    "all_null": "dropped",
    "mostly_null": "dropped",
    "num_with_qmark": "numeric",
    "null_tokens_mix": "numeric",
    "spaced_levels": "categorical",
    "case_levels": "categorical",
    "num_with_nd": "numeric",
    "visit_date": "date",
    "codes_mixed": "dropped",
}


def test_every_torture_column_has_an_expectation(torture):
    assert set(torture.columns) == set(EXPECTED_ROLES)


@pytest.mark.parametrize(("column", "role"), EXPECTED_ROLES.items())
def test_column_role(torture, column, role):
    assert torture.columns[column].role == role


def test_dropped_and_id_columns_are_reported(torture):
    for column, role in EXPECTED_ROLES.items():
        if role in {"dropped", "id"}:
            assert issue_codes(torture, column), f"no issue reported for {column}"


def test_question_mark_does_not_turn_a_number_into_text(torture):
    col = torture.frame["num_with_qmark"]
    assert pd.api.types.is_numeric_dtype(col)
    assert col.isna().sum() == make_torture_csv.N_QMARKS


def test_all_null_tokens_become_nan(torture):
    expected = len(make_torture_csv.NULL_TOKENS) * make_torture_csv.N_PER_NULL_TOKEN
    assert torture.frame["null_tokens_mix"].isna().sum() == expected


def test_spaces_are_stripped_from_levels(torture):
    assert set(torture.frame["spaced_levels"].dropna()) == {"Male", "Female"}


def test_case_variants_are_unified_to_the_most_frequent(torture):
    assert set(torture.frame["case_levels"].dropna()) == {"Male", "Female"}
    assert "case_unified" in issue_codes(torture, "case_levels")


def test_case_unification_can_be_disabled():
    inspection = inspect_csv(TORTURE_CSV, unify_case=False)
    assert set(inspection.frame["case_levels"].dropna()) == {"Male", "male", "MALE", "Female", "female"}
    assert "case_unified" not in issue_codes(inspection)


def test_mostly_numeric_column_warns_and_treats_offenders_as_missing(torture):
    assert "mostly_numeric" in issue_codes(torture, "num_with_nd")
    assert torture.frame["num_with_nd"].isna().sum() == make_torture_csv.N_ND


def test_mostly_numeric_codes_with_many_different_words_stay_text(torture):
    # Regression from the real data (diag_1 of Diabetes 130): ICD-9-like codes are labels.
    assert not pd.api.types.is_numeric_dtype(torture.frame["codes_mixed"])
    assert "mostly_numeric" in issue_codes(torture, "codes_mixed")


def test_date_column_is_kept_with_privacy_warning(torture):
    assert pd.api.types.is_datetime64_any_dtype(torture.frame["visit_date"])
    assert "date_privacy" in issue_codes(torture, "visit_date")


# --------------------------------------------------------------------------------------------
# File level: tiny CSVs written inline
# --------------------------------------------------------------------------------------------

REPEAT = 10  # rows of the tiny inline files repeated, so that no column is near-constant


def test_semicolon_separator_and_decimal_comma():
    data = ("dose;arm\n" + "1,5;A\n2,25;B\n1,5;A\n3;B\n2,25;A\n3;B\n" * REPEAT).encode()
    inspection = inspect_csv(data)
    assert inspection.columns["dose"].kind == "numeric"     # read as numbers (3 values: a category by default)
    assert inspection.frame["dose"].tolist() == [1.5, 2.25, 1.5, 3.0, 2.25, 3.0] * REPEAT


def test_utf8_bom_is_removed_from_first_column_name():
    data = ("﻿age,sex\n" + "30,M\n30,F\n40,M\n40,F\n" * REPEAT).encode("utf-8")
    assert list(inspect_csv(data).frame.columns) == ["age", "sex"]


def test_latin1_file_is_read_with_a_warning():
    data = ("city,n\n" + "città,1\nperò,2\ncittà,1\nperò,2\n" * REPEAT).encode("latin-1")
    inspection = inspect_csv(data)
    assert "encoding_fallback" in issue_codes(inspection)
    assert set(inspection.frame["city"]) == {"città", "però"}


def _rows(n: int) -> str:
    return "".join(f"{i % 4},{'ab'[i % 2]}\n" for i in range(n))


def test_few_malformed_lines_are_skipped_with_a_warning():
    # 40 good rows, one with too many fields, one with too few: 2/42 < 5%
    data = ("a,b\n" + _rows(20) + "1,x,EXTRA\n" + _rows(10) + "2\n" + _rows(10)).encode()
    inspection = inspect_csv(data)
    assert "bad_lines_skipped" in issue_codes(inspection)
    assert len(inspection.frame) == 40


def test_too_many_malformed_lines_is_an_error():
    data = ("a,b\n" + _rows(10) + "1,x,EXTRA\n2,y,EXTRA\n").encode()
    with pytest.raises(DataEngineError) as error:
        inspect_csv(data)
    assert error.value.code == "too_many_bad_lines"


def test_numeric_column_names_suggest_a_missing_header():
    data = ("6,148,72\n" + "1,85,66\n8,183,64\n1,89,66\n6,148,72\n1,85,66\n" * REPEAT).encode()
    assert "missing_header" in issue_codes(inspect_csv(data))


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"", "empty_file"),
        (b"a,b\n", "empty_file"),
        (b"id,flag\n1,x\n2,x\n3,x\n4,x\n", "no_usable_columns"),
        (b"PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00", "not_csv"),
        (b"This is not a table.\nJust some words here\n", "unknown_separator"),
    ],
    ids=["empty", "header_only", "nothing_usable", "xlsx_bytes", "free_text"],
)
def test_blocking_errors(data, code):
    with pytest.raises(DataEngineError) as error:
        inspect_csv(data)
    assert error.value.code == code


# --------------------------------------------------------------------------------------------
# Near-constant columns: few rows differ from the most frequent value -> proposed for removal.
# Absolute count, measured on the whole file.
# --------------------------------------------------------------------------------------------

def near_constant_csv(n_different: int, n_rows: int = 200) -> bytes:
    """'flag' is 'no' except in ``n_different`` rows; 'value' is an ordinary numeric column."""
    lines = ["value,flag"] + [f"{i}.5,{'yes' if i < n_different else 'no'}" for i in range(n_rows)]
    return ("\n".join(lines) + "\n").encode()


def test_near_constant_column_is_proposed_for_removal():
    inspection = inspect_csv(near_constant_csv(MIN_LEVEL_ROWS - 1))
    assert inspection.columns["flag"].role == "dropped"
    assert inspection.columns["flag"].n_minority == MIN_LEVEL_ROWS - 1
    assert "near_constant" in issue_codes(inspection, "flag")


def test_column_with_enough_different_rows_is_kept():
    inspection = inspect_csv(near_constant_csv(MIN_LEVEL_ROWS))
    assert inspection.columns["flag"].role == "categorical"
    assert "near_constant" not in issue_codes(inspection)


def test_minority_rows_ignore_missing_values():
    # 3 'yes', 40 missing, the rest 'no': missing values are not "different rows"
    lines = ["value,flag"] + [f"{i}.5,{'yes' if i < 3 else '?' if i < 43 else 'no'}" for i in range(200)]
    inspection = inspect_csv(("\n".join(lines) + "\n").encode())
    assert inspection.columns["flag"].n_minority == 3
    assert inspection.columns["flag"].role == "dropped"
