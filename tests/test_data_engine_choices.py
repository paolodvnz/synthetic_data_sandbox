"""Specification tests for the user choices.

The engine proposes, the user confirms. Public API under test:
    UserChoices(excluded, included, as_categorical, as_numeric, unify_case)  + to_dict / from_dict
    apply_choices(inspection, choices) -> FinalSchema
        .columns  dict[str, FinalColumn]  kept columns only, file order; FinalColumn.role in
                  {"numeric", "categorical", "date"}, FinalColumn.changed_by_user: bool
        .issues   list[Issue]             consequences of the user's choices
    Invalid choices -> DataEngineError("invalid_choices") listing every problem.
"""

import json
from pathlib import Path

import pytest

from sds.data_engine import DataEngineError, UserChoices, apply_choices, fixed_reason, inspect_csv

TORTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "torture_columns.csv"

DEFAULT_KEPT = {
    "numeric_float": "numeric",
    "numeric_int_wide": "numeric",
    "cat_str": "categorical",
    "cat_int_codes": "categorical",
    "num_with_qmark": "numeric",
    "null_tokens_mix": "numeric",
    "spaced_levels": "categorical",
    "case_levels": "categorical",
    "num_with_nd": "numeric",
    "visit_date": "date",
}


@pytest.fixture(scope="module")
def torture():
    return inspect_csv(TORTURE_CSV)


def roles(schema) -> dict[str, str]:
    return {name: column.role for name, column in schema.columns.items()}


def problems(inspection, choices) -> str:
    """Message of the invalid_choices error raised by ``apply_choices``."""
    with pytest.raises(DataEngineError) as error:
        apply_choices(inspection, choices)
    assert error.value.code == "invalid_choices"
    return error.value.message


# --------------------------------------------------------------------------------------------
# Defaults and simple choices
# --------------------------------------------------------------------------------------------

def test_default_choices_keep_the_engine_proposal(torture):
    schema = apply_choices(torture, UserChoices())
    assert roles(schema) == DEFAULT_KEPT
    assert list(schema.columns) == [c for c in torture.columns if c in DEFAULT_KEPT]  # file order
    assert not any(column.changed_by_user for column in schema.columns.values())


def test_excluding_a_date(torture):
    schema = apply_choices(torture, UserChoices(excluded={"visit_date"}))
    assert "visit_date" not in schema.columns


def test_including_suspected_ids_uses_the_usual_rules(torture):
    schema = apply_choices(torture, UserChoices(included={"id_int", "id_str"}))
    assert schema.columns["id_int"].role == "numeric"      # 200 distinct whole numbers
    assert schema.columns["id_str"].role == "categorical"  # text
    assert schema.columns["id_int"].changed_by_user


def test_reincluding_a_dropped_column_shows_the_consequence(torture):
    schema = apply_choices(torture, UserChoices(included={"mostly_null"}))
    assert "mostly_null" in schema.columns
    messages = [i.message for i in schema.issues if i.code == "column_reincluded" and i.column == "mostly_null"]
    assert messages and "50%" in messages[0]


def test_reincluding_a_constant_column_is_allowed(torture):
    schema = apply_choices(torture, UserChoices(included={"constant"}))
    assert schema.columns["constant"].role == "categorical"


def test_reincluding_a_near_constant_column_says_how_many_rows_differ():
    lines = ["value,flag"] + [f"{i}.5,{'yes' if i < 3 else 'no'}" for i in range(200)]
    inspection = inspect_csv(("\n".join(lines) + "\n").encode())
    schema = apply_choices(inspection, UserChoices(included={"flag"}))
    assert schema.columns["flag"].role == "categorical"
    messages = [i.message for i in schema.issues if i.code == "column_reincluded" and i.column == "flag"]
    assert messages and "3 rows" in messages[0]


def test_an_all_null_column_cannot_be_included(torture):
    assert "all_null" in problems(torture, UserChoices(included={"all_null"}))


# --------------------------------------------------------------------------------------------
# Role corrections
# --------------------------------------------------------------------------------------------

def test_codes_can_be_treated_as_quantities(torture):
    schema = apply_choices(torture, UserChoices(as_numeric={"cat_int_codes"}))
    assert schema.columns["cat_int_codes"].role == "numeric"
    assert schema.columns["cat_int_codes"].changed_by_user


def test_whole_numbers_can_be_treated_as_codes():
    # 26 codes: above MAX_DISCRETE_LEVELS, so inferred as numeric (like discharge_disposition_id)
    data = ("code,arm\n" + "".join(f"{i % 26 + 1},{'ab'[i % 2]}\n" for i in range(130))).encode()
    inspection = inspect_csv(data)
    assert inspection.columns["code"].role == "numeric"
    schema = apply_choices(inspection, UserChoices(as_categorical={"code"}))
    assert schema.columns["code"].role == "categorical"


def test_too_many_values_cannot_become_categories(torture):
    assert "numeric_int_wide" in problems(torture, UserChoices(as_categorical={"numeric_int_wide"}))


def test_decimals_cannot_become_categories(torture):
    assert "numeric_float" in problems(torture, UserChoices(as_categorical={"numeric_float"}))


def test_text_cannot_become_a_quantity(torture):
    assert "cat_str" in problems(torture, UserChoices(as_numeric={"cat_str"}))


def test_dates_cannot_change_role(torture):
    message = problems(torture, UserChoices(as_numeric={"visit_date"}))
    assert "visit_date" in message


def test_yes_no_codes_cannot_become_quantities():
    # Two values: codes and quantities decode to the same thing, so the reading is fixed.
    data = ("smoker,arm\n" + "".join(f"{i % 2},{'abc'[i % 3]}\n" for i in range(60))).encode()
    inspection = inspect_csv(data)
    assert "smoker" in problems(inspection, UserChoices(as_numeric={"smoker"}))
    assert fixed_reason(inspection.columns["smoker"]) is not None


def test_confirming_the_engine_reading_is_always_allowed(torture):
    # The UI may send the reading back unchanged, even for fixed columns.
    schema = apply_choices(torture, UserChoices(as_categorical={"cat_str"}, as_numeric={"numeric_float"}))
    assert roles(schema) == DEFAULT_KEPT
    assert not any(column.changed_by_user for column in schema.columns.values())


def test_only_whole_numbers_with_few_values_can_change_reading(torture):
    # Rule for the two groups of the Advanced panel: changeable = whole numbers, 3 to MAX_CATEGORIES values.
    changeable = {name for name in DEFAULT_KEPT if fixed_reason(torture.columns[name]) is None}
    assert changeable == {"cat_int_codes"}


def test_corrections_on_excluded_columns_are_ignored(torture):
    schema = apply_choices(torture, UserChoices(excluded={"cat_int_codes"}, as_numeric={"cat_int_codes"}))
    assert "cat_int_codes" not in schema.columns


# --------------------------------------------------------------------------------------------
# Invalid combinations
# --------------------------------------------------------------------------------------------

def test_unknown_column(torture):
    assert "nonexistent" in problems(torture, UserChoices(excluded={"nonexistent"}))


def test_excluded_and_included_at_once(torture):
    assert "id_int" in problems(torture, UserChoices(excluded={"id_int"}, included={"id_int"}))


def test_categorical_and_numeric_at_once(torture):
    message = problems(torture, UserChoices(as_numeric={"cat_int_codes"}, as_categorical={"cat_int_codes"}))
    assert "cat_int_codes" in message


def test_every_problem_is_reported_together(torture):
    message = problems(torture, UserChoices(excluded={"nonexistent"}, as_numeric={"cat_str"}))
    assert "nonexistent" in message and "cat_str" in message


def test_excluding_everything_leaves_no_usable_column(torture):
    with pytest.raises(DataEngineError) as error:
        apply_choices(torture, UserChoices(excluded=set(DEFAULT_KEPT)))
    assert error.value.code == "no_usable_columns"


def test_changing_case_unification_requires_a_new_inspection(torture):
    with pytest.raises(DataEngineError) as error:
        apply_choices(torture, UserChoices(unify_case=False))
    assert error.value.code == "stale_inspection"


# --------------------------------------------------------------------------------------------
# Serialisation (the choices travel between UI and API)
# --------------------------------------------------------------------------------------------

def test_choices_round_trip_through_json():
    choices = UserChoices(excluded={"b", "a"}, included={"c"}, as_categorical={"d"}, as_numeric=set(), unify_case=False)
    payload = json.dumps(choices.to_dict())
    assert UserChoices.from_dict(json.loads(payload)) == choices
    assert json.loads(payload)["excluded"] == ["a", "b"]  # sorted: stable output


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(DataEngineError) as error:
        UserChoices.from_dict({"exclude": ["a"]})
    assert error.value.code == "invalid_choices"
