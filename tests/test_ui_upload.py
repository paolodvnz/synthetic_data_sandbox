"""Specification tests for the Upload step: send the file once, the columns, Advanced, the warnings, the rows.

The page runs with ``AppTest`` (no browser) against a fake API (``ui_fake``). The table of columns
(``st.data_editor``) cannot be edited in ``AppTest``: its translation into choices is tested on the pure functions.
"""

import pytest
from streamlit.testing.v1 import AppTest

from sds.config import PROJECT_ROOT
from ui.api_client import ApiRefused
from ui.upload_view import (
    attention_issues,
    columns_frame,
    current_readings,
    desired_choices,
    detail_issues,
    missing_label,
    notes_by_column,
)
from ui_fake import FakeApi, dataset, install, job

APP = str(PROJECT_ROOT / "ui" / "app.py")
WAIT_SECONDS = 30
CSV = ("patients.csv", b"age,sex\n40,F\n", "text/csv")


# --- pure functions -------------------------------------------------------------------------

def keep_all_used(data: dict) -> dict:
    return {c["name"]: c["final_role"] is not None for c in data["columns"]}


def test_nothing_changed_gives_the_current_choices():
    data = dataset()
    readings = current_readings(data["columns"])
    assert desired_choices(data["columns"], keep_all_used(data), readings, True) == data["choices"]


def test_unticking_a_used_column_excludes_it_and_ticking_a_left_out_one_includes_it():
    data = dataset()
    keep = keep_all_used(data) | {"age": False, "id": True}
    choices = desired_choices(data["columns"], keep, current_readings(data["columns"]), True)
    assert choices["excluded"] == ["age"] and choices["included"] == ["id"]


def test_a_changed_reading_becomes_a_correction():
    data = dataset()
    readings = current_readings(data["columns"]) | {"visits": "categorical"}
    choices = desired_choices(data["columns"], keep_all_used(data), readings, True)
    assert choices["as_categorical"] == ["visits"] and choices["as_numeric"] == []


def test_readings_of_columns_not_kept_are_ignored():
    data = dataset()
    readings = current_readings(data["columns"]) | {"visits": "categorical"}
    choices = desired_choices(data["columns"], keep_all_used(data) | {"visits": False}, readings, True)
    assert choices["as_categorical"] == [] and choices["excluded"] == ["visits"]


def test_the_case_option_is_carried():
    data = dataset()
    assert desired_choices(data["columns"], keep_all_used(data), current_readings(data["columns"]),
                           False)["unify_case"] is False


def test_current_readings_follow_an_applied_correction():
    data = dataset()
    data["columns"][2]["final_role"] = "categorical"              # visits, corrected earlier by the user
    assert current_readings(data["columns"])["visits"] == "categorical"


def test_only_some_issues_need_attention():
    issues = dataset()["issues"]
    assert [i["code"] for i in attention_issues(issues)] == ["suspected_id", "few_rows"]
    assert [i["code"] for i in detail_issues(issues)] == ["rows_removed", "levels_merged"]


def test_notes_by_column():
    notes = notes_by_column(dataset()["issues"])
    assert set(notes) == {"id", "sex"} and "identifier" in notes["id"][0]


@pytest.mark.parametrize("fraction, label", [(0.0, "0%"), (0.0004, "<0.1%"), (0.001, "0.1%"), (0.1234, "12.3%"),
                                             (0.999, "99.9%"),
                                             (0.9996, ">99.9%"), (1.0, "100%")])
def test_missing_label(fraction, label):
    # A few missing values never read "0.0%", nor almost all of them "100.0%".
    assert missing_label(fraction) == label


def test_the_columns_table_shows_the_share_of_missing_values():
    data = dataset()
    data["columns"][1]["null_fraction"] = 0.25
    table = columns_frame(data)
    assert list(table.columns) == ["Keep", "Column", "We read it as", "Missing", "Why", "Examples", "Notes"]
    assert table["Missing"].tolist() == ["0%", "25.0%", "0%", "0%"]
    assert table["Keep"].tolist() == [False, True, True, True]           # the identifier is left out
    assert "identifier" in table.loc[0, "Notes"]


# --- the page ---------------------------------------------------------------------------------

@pytest.fixture
def fake(monkeypatch):
    def make(view=None, data=None, dev_mode=False) -> FakeApi:
        view = view or ({"stage": "prepared", "job": None} if data else {"stage": "empty", "job": None})
        return install(monkeypatch, FakeApi(view, dev_mode=dev_mode, data=data))
    return make


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=WAIT_SECONDS).run()
    assert not app.exception
    return app


def texts(elements) -> str:
    return " | ".join(str(e.value) for e in elements)


def test_empty_session_asks_for_a_file(fake):
    fake()
    app = run_app()
    assert len(app.get("file_uploader")) == 1
    assert "next_train" not in [b.key for b in app.button]


def test_the_file_is_sent_once_and_then_dropped_by_the_ui(fake):
    api = fake()
    app = run_app()
    generation = app.session_state["uploader_generation"]
    app.get("file_uploader")[0].set_value(CSV).run()
    assert api.uploads == [("patients.csv", b"age,sex\n40,F\n")]
    assert app.session_state["uploader_generation"] == generation + 1      # new key: the uploader is empty
    assert "patients.csv" in texts(app.markdown)
    app.run()
    assert len(api.uploads) == 1


def test_a_refused_file_explains_why(fake):
    api = fake()
    api.refuse_upload = ApiRefused(413, "file_too_large", "The file is larger than 200 MB.")
    app = run_app()
    app.get("file_uploader")[0].set_value(CSV).run()
    assert "larger than 200 MB" in texts(app.error)
    assert "next_train" not in [b.key for b in app.button]


def test_with_a_dataset_another_file_needs_start_over(fake):
    fake(data=dataset(), dev_mode=True)
    app = run_app()
    assert not app.get("file_uploader") and "load_sample" not in [b.key for b in app.button]
    assert "Start over" in texts(app.caption)
    app.button(key="start_over").click().run()
    app.button(key="start_over_confirm").click().run()
    assert len(app.get("file_uploader")) == 1 and "load_sample" in [b.key for b in app.button]


def test_after_a_refresh_the_dataset_comes_from_the_api(fake):
    api = fake(data=dataset())
    app = run_app()
    assert api.dataset_reads == 1 and not api.uploads
    assert "patients.csv" in texts(app.markdown)
    assert [m.value for m in app.metric][:3] == ["790", "198", "12"]


def test_the_dataset_is_read_once_per_ui_session(fake):
    api = fake(data=dataset())
    app = run_app()
    app.run()
    assert api.dataset_reads == 1


def test_warnings_that_need_attention_are_shown_the_rest_in_details(fake):
    fake(data=dataset())
    app = run_app()
    warnings = texts(app.warning)
    assert "identifier" in warnings and "Only 790 rows" in warnings
    assert "rare in 'sex'" not in warnings and "rows with missing values" not in warnings
    details = next(e for e in app.expander if e.label.startswith("Details"))
    assert details.label == "Details (2 notes)"
    assert "rare in 'sex'" in texts(details.markdown)


def test_a_dataset_not_ready_shows_the_problem_and_no_next(fake):
    fake(view={"stage": "inspected", "job": None}, data=dataset(ready=False))
    app = run_app()
    assert "Too few rows" in texts(app.error)
    assert "next_train" not in [b.key for b in app.button]


def test_next_opens_train(fake):
    fake(data=dataset())
    app = run_app()
    app.button(key="next_train").click().run()
    assert app.session_state["step"] == "train"


def test_changing_a_reading_then_apply(fake):
    api = fake(data=dataset())
    app = run_app()
    assert "apply_choices" not in [b.key for b in app.button]
    app.radio(key=f"reading_{app.session_state['editor_token']}_visits").set_value("category").run()
    assert app.button(key="next_train").disabled                           # apply (or discard) first
    app.button(key="apply_choices").click().run()
    assert api.sent_choices[-1]["as_categorical"] == ["visits"]
    assert "apply_choices" not in [b.key for b in app.button]
    assert not app.button(key="next_train").disabled


def test_discard_goes_back_to_the_applied_choices(fake):
    api = fake(data=dataset())
    app = run_app()
    app.radio(key=f"reading_{app.session_state['editor_token']}_visits").set_value("category").run()
    app.button(key="discard_choices").click().run()
    assert not api.sent_choices and "apply_choices" not in [b.key for b in app.button]
    assert app.radio(key=f"reading_{app.session_state['editor_token']}_visits").value == "quantity"


def test_only_changeable_columns_have_a_reading(fake):
    fake(data=dataset())
    app = run_app()
    assert [r.key.rsplit("_", 1)[1] for r in app.radio] == ["visits"]


def test_case_option_only_when_the_file_has_case_merges(fake):
    fake(data=dataset())
    assert not run_app().checkbox
    api = fake(data=dataset(case_merges=True))
    app = run_app()
    app.checkbox(key=f"case_{app.session_state['editor_token']}").uncheck().run()
    app.button(key="apply_choices").click().run()
    assert api.sent_choices[-1]["unify_case"] is False


def test_a_refused_choice_keeps_the_page(fake, monkeypatch):
    api = fake(data=dataset())

    def refuse(choices):
        raise ApiRefused(422, "no_usable_columns", "Keep at least one column.")

    monkeypatch.setattr("ui.api_client.set_choices", refuse)
    app = run_app()
    app.radio(key=f"reading_{app.session_state['editor_token']}_visits").set_value("category").run()
    app.button(key="apply_choices").click().run()
    assert "Keep at least one column." in texts(app.error)
    assert not api.sent_choices


def test_while_training_nothing_can_change(fake):
    fake(view={"stage": "prepared", "job": job("running")}, data=dataset())
    app = run_app()
    app.button(key="step_upload").click().run()
    assert not app.get("file_uploader")                                   # a dataset is loaded: Start over first
    assert all(r.disabled for r in app.radio)
    assert "training is running" in texts(app.info)


def test_samples_menu_only_in_developer_mode(fake):
    fake()
    assert "load_sample" not in [b.key for b in run_app().button]
    api = fake(dev_mode=True)
    app = run_app()
    assert app.selectbox(key="sample_name").options == ["adult"]           # only the downloaded ones
    app.button(key="load_sample").click().run()
    assert api.samples_loaded == ["adult"] and "patients.csv" in texts(app.markdown)


def test_developer_panels_of_the_upload_step(fake):
    fake(data=dataset(), dev_mode=True)
    app = run_app()
    app.toggle(key="dev_panels").set_value(True).run()
    assert not app.exception
    random_rows = app.dataframe[-1].value
    assert list(random_rows.columns) == ["age", "sex"] and len(random_rows) == 3
    labels = [e.label for e in app.expander]
    assert any("contract" in label for label in labels) and any("Decoding check" in label for label in labels)
    assert "latin-1" in texts(app.markdown) + texts(app.json)


def test_an_emptied_api_drops_the_kept_dataset(fake):
    api = fake(data=dataset())
    app = run_app()
    api.view, api.data = {"stage": "empty", "job": None}, None          # e.g. the API was restarted
    app.run()
    other = dataset()
    other["file"]["name"] = "other.csv"
    api.loaded(other)                                                   # a new file, loaded from elsewhere
    app.run()
    assert "other.csv" in texts(app.markdown)
