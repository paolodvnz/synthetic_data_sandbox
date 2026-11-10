"""Specification tests for the preparation step.

Rows with missing values in the kept columns are removed, then the rows are split at random into a
train set (for the VAE) and a holdout set (only for the privacy check). Public API under test:
    prepare(inspection, schema, *, seed=RANDOM_STATE) -> PreparedData
        .train, .holdout  DataFrames with the schema columns only (schema order), disjoint rows
        .rows_read        rows in the inspected file
        .rows_removed     rows removed because of missing values
        .issues           list[Issue]: rows_removed | many_rows_removed, few_rows
    Fewer than MIN_USABLE_ROWS rows left -> DataEngineError("too_few_rows").
"""

import pytest

from sds.data_engine import DataEngineError, UserChoices, apply_choices, inspect_csv, prepare
from sds.data_engine import preparation
from sds.data_engine.thresholds_config import HOLDOUT_FRACTION, MIN_ROWS, MIN_USABLE_ROWS


def make_csv(n_rows: int, *, extra_nulls: int = 0) -> bytes:
    """CSV with a numeric column, a categorical one and an 'extra' column with missing values.

    ``extra`` is missing ('?') in its first ``extra_nulls`` rows.
    """
    lines = ["value,group,extra"]
    for i in range(n_rows):
        extra = "?" if i < extra_nulls else f"{i}.25"
        lines.append(f"{i}.5,{'abc'[i % 3]},{extra}")
    return ("\n".join(lines) + "\n").encode()


def prepared(csv: bytes, choices: UserChoices | None = None, **kwargs):
    inspection = inspect_csv(csv)
    schema = apply_choices(inspection, choices or UserChoices())
    return prepare(inspection, schema, **kwargs)


def codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


# --------------------------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------------------------

def test_holdout_is_a_fifth_of_the_rows():
    result = prepared(make_csv(2000))
    assert len(result.holdout) == round(2000 * HOLDOUT_FRACTION) == 400
    assert len(result.train) == 1600


def test_holdout_is_capped(monkeypatch):
    monkeypatch.setattr(preparation, "MAX_HOLDOUT_ROWS", 100)
    result = prepared(make_csv(2000))
    assert len(result.holdout) == 100
    assert len(result.train) == 1900


def test_train_and_holdout_are_disjoint_and_cover_every_row():
    result = prepared(make_csv(500))
    train, holdout = set(result.train["value"]), set(result.holdout["value"])
    assert not train & holdout
    assert len(train | holdout) == 500


def test_split_is_reproducible():
    first, second = prepared(make_csv(500), seed=7), prepared(make_csv(500), seed=7)
    assert first.holdout.equals(second.holdout)
    assert first.train.equals(second.train)


def test_a_different_seed_gives_a_different_split():
    first, second = prepared(make_csv(500), seed=1), prepared(make_csv(500), seed=2)
    assert set(first.holdout["value"]) != set(second.holdout["value"])


def test_only_kept_columns_in_schema_order():
    result = prepared(make_csv(500), UserChoices(excluded={"group"}))
    assert list(result.train.columns) == list(result.holdout.columns) == ["value", "extra"]


# --------------------------------------------------------------------------------------------
# Rows with missing values
# --------------------------------------------------------------------------------------------

def test_no_missing_values_no_rows_removed():
    result = prepared(make_csv(500))
    assert result.rows_read == 500
    assert result.rows_removed == 0
    assert not codes(result) & {"rows_removed", "many_rows_removed"}


def test_rows_with_missing_values_are_removed_and_reported():
    result = prepared(make_csv(500, extra_nulls=50))
    assert result.rows_removed == 50
    assert len(result.train) + len(result.holdout) == 450
    assert not result.train["extra"].isna().any() and not result.holdout["extra"].isna().any()
    (issue,) = [i for i in result.issues if i.code == "rows_removed"]
    assert "50" in issue.message


def test_missing_values_in_removed_columns_do_not_remove_rows():
    result = prepared(make_csv(500, extra_nulls=50), UserChoices(excluded={"extra"}))
    assert result.rows_removed == 0
    assert len(result.train) + len(result.holdout) == 500


def test_losing_most_rows_names_the_column_responsible():
    # 'extra' is 60% empty: the engine drops it, the user brings it back.
    result = prepared(make_csv(500, extra_nulls=300), UserChoices(included={"extra"}))
    assert result.rows_removed == 300
    (issue,) = [i for i in result.issues if i.code == "many_rows_removed"]
    assert "'extra'" in issue.message
    assert "rows_removed" not in codes(result)  # one message, not two


# --------------------------------------------------------------------------------------------
# How many rows are enough
# --------------------------------------------------------------------------------------------

def test_small_dataset_gets_a_warning():
    result = prepared(make_csv(MIN_ROWS))  # train = 80% < MIN_ROWS
    assert "few_rows" in codes(result)


def test_large_enough_dataset_gets_no_warning():
    n_rows = round(MIN_ROWS / (1 - HOLDOUT_FRACTION)) + 10
    assert "few_rows" not in codes(prepared(make_csv(n_rows)))


def test_too_few_rows_is_an_error():
    with pytest.raises(DataEngineError) as error:
        prepared(make_csv(MIN_USABLE_ROWS - 1))
    assert error.value.code == "too_few_rows"


def test_too_few_rows_after_removing_missing_values_is_an_error():
    with pytest.raises(DataEngineError) as error:
        prepared(make_csv(200, extra_nulls=200 - MIN_USABLE_ROWS + 1), UserChoices(included={"extra"}))
    assert error.value.code == "too_few_rows"


def test_smallest_usable_dataset_has_a_holdout():
    result = prepared(make_csv(MIN_USABLE_ROWS))
    assert len(result.holdout) >= 1 and len(result.train) >= 1
