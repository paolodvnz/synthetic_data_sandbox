"""Specification tests for api.dataset: upload and user choices, without HTTP."""

import json

import pytest

from api.dataset import DEFAULT_SOURCE_NAME, N_EXAMPLES, change_choices, dataset_view, load_dataset
from api.errors import ApiError
from api.session import Session, Stage
from api_csv import all_cells, make_csv
from sds.data_engine import DataEngineError

MARKER = object()
TOO_FEW_ROWS = 30          # below MIN_USABLE_ROWS = 50: the inspection works, the preparation does not


@pytest.fixture
def loaded() -> Session:
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    return session


def view_of(session: Session) -> dict:
    """The view, after a strict JSON round trip (no NaN, only JSON types)."""
    return json.loads(json.dumps(dataset_view(session), allow_nan=False))


# --- upload ---------------------------------------------------------------------------------

def test_upload_prepares_the_dataset(loaded):
    assert loaded.stage is Stage.PREPARED
    view = view_of(loaded)
    assert view["ready"] is True and view["problem"] is None
    assert view["file"]["name"] == "patients.csv" and view["file"]["n_rows"] == 120
    rows = view["rows"]
    assert rows["train"] + rows["holdout"] == rows["read"] - rows["removed"]
    assert rows["holdout"] > 0


def test_view_describes_every_column(loaded):
    columns = {c["name"]: c for c in view_of(loaded)["columns"]}
    assert list(columns) == ["age", "weight", "sex", "city", "visits"]
    for column in columns.values():
        assert set(column) == {"name", "role", "reason", "kept_role", "fixed_reason", "null_fraction", "examples",
                               "final_role", "changed_by_user"}
        assert column["reason"]
        assert 1 <= len(column["examples"]) <= N_EXAMPLES
        assert all(isinstance(v, str) for v in column["examples"])
    assert columns["weight"]["fixed_reason"]                  # decimals: the reading cannot change
    assert columns["visits"]["fixed_reason"] is None          # 0-5 whole numbers: codes or quantities


def test_view_contains_no_row(loaded):
    text = json.dumps(view_of(loaded))
    for cells in all_cells(make_csv()):
        assert ",".join(cells) not in text and json.dumps(cells) not in text


def test_view_groups_the_issues_by_phase(loaded):
    issues = view_of(loaded)["issues"]
    assert set(issues) == {"file", "choices", "rows", "encoding"}
    assert any(i["code"] == "case_unified" for i in issues["file"])
    assert any(i["code"] == "few_rows" for i in issues["rows"])        # 96 training rows < MIN_ROWS


def test_case_merges_come_from_the_default_inspection(loaded):
    merges = view_of(loaded)["case_merges"]
    assert [m["column"] for m in merges] == ["sex"]


def test_view_echoes_the_default_choices(loaded):
    assert view_of(loaded)["choices"] == {"excluded": [], "included": [], "as_categorical": [], "as_numeric": [],
                                          "unify_case": True}


def test_a_second_upload_is_refused_and_changes_nothing(loaded):
    # One dataset per session: another file only after Start over (/reset), never replacing work silently.
    loaded.trained = MARKER
    with pytest.raises(ApiError) as caught:
        load_dataset(loaded, make_csv(seed=1), "other.csv")
    assert caught.value.status == 409 and caught.value.code == "dataset_loaded"
    assert "Start over" in caught.value.message
    assert loaded.trained is MARKER and view_of(loaded)["file"]["name"] == "patients.csv"


def test_after_a_reset_another_file_can_be_loaded(loaded):
    loaded.reset()
    load_dataset(loaded, make_csv(seed=1), "other.csv")
    assert view_of(loaded)["file"]["name"] == "other.csv"


def test_a_file_that_cannot_be_prepared_also_blocks_a_new_upload():
    # The columns stay visible (the user can fix the choices); another file still needs Start over.
    session = Session()
    load_dataset(session, make_csv(TOO_FEW_ROWS), "small.csv")
    with pytest.raises(ApiError) as caught:
        load_dataset(session, make_csv(), "patients.csv")
    assert caught.value.code == "dataset_loaded"


def test_failed_upload_leaves_an_empty_session():
    session = Session()
    with pytest.raises(DataEngineError) as caught:
        load_dataset(session, b"", "empty.csv")
    assert caught.value.code == "empty_file"
    assert session.stage is Stage.EMPTY
    load_dataset(session, make_csv(), "patients.csv")          # nothing loaded: another file is welcome
    assert session.stage is not Stage.EMPTY


def test_too_few_rows_keeps_the_columns_visible():
    session = Session()
    load_dataset(session, make_csv(TOO_FEW_ROWS), "small.csv")
    assert session.stage is Stage.INSPECTED
    view = view_of(session)
    assert view["ready"] is False and view["rows"] is None
    assert view["problem"]["code"] == "too_few_rows" and view["problem"]["message"]
    assert len(view["columns"]) == 5


@pytest.mark.parametrize(("given", "kept"), [
    ("patients.csv", "patients.csv"),
    ("../../etc/passwd.csv", "passwd.csv"),
    ("C:\\Users\\me\\data.csv", "data.csv"),
    ("", DEFAULT_SOURCE_NAME),
    (None, DEFAULT_SOURCE_NAME),
    ("x" * 300 + ".csv", ("x" * 300 + ".csv")[-255:]),
])
def test_source_name_is_a_plain_file_name(given, kept):
    session = Session()
    load_dataset(session, make_csv(), given)
    assert session.source_name == kept


def test_bytes_are_kept_in_memory(loaded):
    assert loaded.source_bytes == make_csv()


# --- choices --------------------------------------------------------------------------------

def test_excluding_a_column(loaded):
    width = loaded.contract.width
    change_choices(loaded, {"excluded": ["city"]})
    view = view_of(loaded)
    columns = {c["name"]: c for c in view["columns"]}
    assert columns["city"]["final_role"] is None and columns["age"]["final_role"] == "numeric"
    assert view["choices"]["excluded"] == ["city"]
    assert loaded.stage is Stage.PREPARED and loaded.contract.width < width


def test_reading_a_column_as_quantities(loaded):
    # visits (0-5) is proposed as categories: the user may read it as a quantity
    assert {c["name"]: c for c in view_of(loaded)["columns"]}["visits"]["final_role"] == "categorical"
    change_choices(loaded, {"as_numeric": ["visits"]})
    columns = {c["name"]: c for c in view_of(loaded)["columns"]}
    assert columns["visits"]["final_role"] == "numeric" and columns["visits"]["changed_by_user"] is True


def test_changing_choices_drops_the_later_stages(loaded):
    loaded.trained, loaded.synthetic = MARKER, MARKER
    change_choices(loaded, {"excluded": ["city"]})
    assert loaded.trained is None and loaded.synthetic is None


def test_turning_off_case_unification_inspects_again(loaded):
    merges = view_of(loaded)["case_merges"]
    change_choices(loaded, {"unify_case": False})
    assert loaded.inspection.unify_case is False
    view = view_of(loaded)
    assert view["choices"]["unify_case"] is False
    assert view["case_merges"] == merges                          # still shows what the option would do
    assert not any(i["code"] == "case_unified" for i in view["issues"]["file"])


@pytest.mark.parametrize("choices", [{"excluded": ["no_such_column"]}, {"colour": []}, {"excluded": "city"},
                                     {"as_categorical": ["weight"]}])
def test_invalid_choices_leave_the_session_unchanged(loaded, choices):
    contract, kept_choices = loaded.contract, loaded.choices
    with pytest.raises(DataEngineError) as caught:
        change_choices(loaded, choices)
    assert caught.value.code == "invalid_choices"
    assert loaded.contract is contract and loaded.choices is kept_choices


def test_excluding_everything_is_refused(loaded):
    with pytest.raises(DataEngineError) as caught:
        change_choices(loaded, {"excluded": ["age", "weight", "sex", "city", "visits"]})
    assert caught.value.code == "no_usable_columns"
    assert loaded.stage is Stage.PREPARED


def test_choices_need_a_dataset():
    with pytest.raises(ApiError) as caught:
        change_choices(Session(), {})
    assert caught.value.status == 409 and caught.value.code == "no_dataset"


def test_view_needs_a_dataset():
    with pytest.raises(ApiError) as caught:
        dataset_view(Session())
    assert caught.value.code == "no_dataset"


def test_choices_can_fix_a_dataset_that_was_not_ready():
    session = Session()
    load_dataset(session, make_csv(TOO_FEW_ROWS), "small.csv")
    change_choices(session, {})                                   # same choices: still not ready, no error
    assert session.stage is Stage.INSPECTED and view_of(session)["problem"]["code"] == "too_few_rows"


def test_choices_that_fix_the_problem_make_it_ready():
    # 60 rows, 'notes' empty in 1 row of 4: 45 complete rows (< 50). Excluding 'notes' leaves 60.
    lines = ["age,sex,notes"] + [f"{20 + i},{'MF'[i % 2]},{'' if i % 4 == 0 else 'n' + str(i % 3)}" for i in range(60)]
    session = Session()
    load_dataset(session, ("\n".join(lines) + "\n").encode(), "notes.csv")
    assert view_of(session)["problem"]["code"] == "too_few_rows"
    change_choices(session, {"excluded": ["notes"]})
    view = view_of(session)
    assert session.stage is Stage.PREPARED and view["ready"] is True and view["problem"] is None


def test_examples_of_whole_numbers_have_no_decimals():
    # A missing value turns the column into floats when read: 39 must still be shown as "39", not "39.0".
    data = b"age,weight,sex\n" + b"".join(f"{18 + i % 60},{70 + i % 7}.5,{'MF'[i % 2]}\n".encode() for i in range(80))
    data += b",71.5,M\n"
    session = Session()
    load_dataset(session, data, "p.csv")
    columns = {c["name"]: c for c in dataset_view(session)["columns"]}
    assert columns["age"]["examples"] == ["18", "19", "20"]
    assert columns["weight"]["examples"] == ["70.5", "71.5", "72.5"]
