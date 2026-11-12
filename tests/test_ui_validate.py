"""Specification tests for the Validate step: gate, warnings, download, preview, heatmaps, histograms, dev panels.

The page runs with ``AppTest`` against a fake API (``ui_fake``).
"""

import pytest
from streamlit.testing.v1 import AppTest

from sds.config import PROJECT_ROOT
from ui.validate_view import bin_labels, heatmap_frame, histogram_frame
from ui_fake import ADVICE, FakeApi, dataset, install, job, validation

APP = str(PROJECT_ROOT / "ui" / "app.py")
WAIT_SECONDS = 30


# --- pure functions -------------------------------------------------------------------------

def test_bin_labels_of_a_numeric_column_are_ranges():
    histogram = {"role": "numeric", "labels": [18.0, 42.5, 67.0, 91.5]}
    assert bin_labels(histogram) == ["18 – 42.5", "42.5 – 67", "67 – 91.5"]


def test_bin_labels_round_long_numbers():
    histogram = {"role": "numeric", "labels": [0.0, 28.650000000000002, 1234567.0]}
    assert bin_labels(histogram) == ["0 – 28.7", "28.7 – 1.23e+06"]


def test_bin_labels_of_dates_keep_the_day():
    histogram = {"role": "date", "labels": ["2020-01-01T00:00:00", "2020-06-30T12:00:00", "2021-01-01T00:00:00"]}
    assert bin_labels(histogram) == ["2020-01-01 – 2020-06-30", "2020-06-30 – 2021-01-01"]


def test_bin_labels_of_a_categorical_column_are_its_levels():
    assert bin_labels({"role": "categorical", "labels": ["F", "M", 3]}) == ["F", "M", "3"]


def test_histogram_frame_is_long_with_both_sources_in_order():
    frame = histogram_frame(validation()["histograms"][0])
    assert list(frame.columns) == ["bin", "data", "share", "order"]
    assert list(frame["data"]) == ["Real", "Real", "Real", "Synthetic", "Synthetic", "Synthetic"]
    assert list(frame["share"]) == [0.4, 0.4, 0.2, 0.35, 0.45, 0.2]
    assert list(frame["order"]) == [0, 1, 2, 0, 1, 2]


def test_heatmap_frame_has_one_cell_per_pair():
    associations = validation()["associations"]
    frame = heatmap_frame(associations["columns"], associations["synthetic"])
    assert len(frame) == 9 and list(frame.columns) == ["row", "column", "association"]
    cell = frame[(frame["row"] == "age") & (frame["column"] == "visits")]
    assert cell["association"].item() == 0.45


def test_heatmap_frame_keeps_missing_values_missing():
    frame = heatmap_frame(["a", "b"], [[1.0, None], [None, 1.0]])
    assert frame["association"].isna().sum() == 2


# --- the page ---------------------------------------------------------------------------------

@pytest.fixture
def fake(monkeypatch):
    def make(*, passed: bool = True, dev_mode: bool = False) -> FakeApi:
        api = FakeApi({"stage": "validated", "job": job("done", step=100)}, dev_mode=dev_mode, data=dataset())
        api.passed = passed
        return install(monkeypatch, api)
    return make


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=WAIT_SECONDS).run()      # a validated session opens on Validate
    assert not app.exception and app.session_state["step"] == "validate"
    return app


def texts(elements) -> str:
    return " | ".join(str(e.value) for e in elements)


def download_buttons(app: AppTest) -> list:
    return app.get("download_button")


def test_gate_passed(fake):
    fake()
    app = run_app()
    assert "Privacy check passed" in texts(app.success) and "790" in texts(app.success)
    assert len(download_buttons(app)) == 1
    preview = app.dataframe[0].value
    assert list(preview.columns) == ["age", "sex", "visits"] and len(preview) == 2


def test_gate_failed_hides_preview_and_download(fake):
    api = fake(passed=False)
    app = run_app()
    errors = texts(app.error)
    assert "much closer to the real records" in errors and ADVICE in errors
    assert not download_buttons(app) and not app.success
    assert api.downloads == 0


def test_the_csv_is_asked_only_on_click(fake):
    api = fake()
    app = run_app()
    app.run()
    assert api.downloads == 0                     # the button holds a function, not the file


def test_download_name_follows_the_uploaded_file(fake):
    fake()
    app = run_app()
    assert download_buttons(app)[0].proto.label == "Download the synthetic data (CSV)"
    assert "patients_synthetic.csv" in texts(app.caption)


def test_warnings_do_not_block(fake):
    fake()
    app = run_app()
    warnings = texts(app.warning)
    assert "near copies cannot be checked" in warnings and "spread differently" in warnings
    assert len(download_buttons(app)) == 1


def test_heatmaps_and_one_histogram_at_a_time(fake):
    fake()
    app = run_app()
    assert app.selectbox(key="histogram_column").options == ["age", "sex"]
    charts = app.get("vega_lite_chart")
    assert len(charts) == 2                       # heatmaps side by side (one concatenated chart) + histogram
    app.selectbox(key="histogram_column").set_value("sex").run()
    assert not app.exception


def test_ask_stays_disabled(fake):
    fake()
    assert run_app().button(key="step_ask").disabled


def test_developer_panels_of_the_validate_step(fake):
    fake(dev_mode=True)
    app = run_app()
    app.toggle(key="dev_panels").set_value(True).run()
    assert not app.exception and not app.error
    labels = [e.label for e in app.expander]
    assert any("fidelity" in label.lower() for label in labels) and any("privacy" in label.lower() for label in labels)
    assert len(download_buttons(app)) == 2        # the CSV and the full reports in JSON
    tables = [d.value for d in app.dataframe]
    assert any({"z", "excess"} <= set(t.columns) for t in tables)


def test_validation_not_ready_explains(fake, monkeypatch):
    api = fake()
    api.view = {"stage": "prepared", "job": None}
    app = AppTest.from_file(APP, default_timeout=WAIT_SECONDS).run()
    assert not app.exception and app.session_state["step"] != "validate"


def chart_tables(chart) -> list:
    """The data of a chart drawn by the page (Arrow tables sent with the Vega-Lite spec)."""
    import pyarrow as pa
    return [pa.ipc.open_stream(dataset.data.data).read_all().to_pandas() for dataset in chart.proto.datasets]


def test_the_heatmaps_draw_real_and_synthetic(fake):
    fake()
    app = run_app()
    tables = chart_tables(app.get("vega_lite_chart")[0])
    pairs = [t[(t["row"] == "age") & (t["column"] == "visits")]["association"].item() for t in tables]
    assert sorted(pairs) == [0.45, 0.5]                      # synthetic and real


def test_heatmap_values_are_numbers_even_with_gaps():
    frame = heatmap_frame(["a", "b"], [[1.0, None], [None, 1.0]])
    assert frame["association"].dtype == float


def test_the_downloaded_file_gets_the_name_from_the_api(fake, monkeypatch):
    import streamlit as st
    fake()
    seen = {}
    real_button = st.download_button

    def spy(*args, **kwargs):
        seen.setdefault(kwargs.get("key"), kwargs)
        return real_button(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", spy)
    run_app()
    assert seen["download_csv"]["file_name"] == "patients_synthetic.csv"
    assert seen["download_csv"]["data"]() == b"age;sex;visits\n40;F;2\n"
