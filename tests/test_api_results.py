"""Specification tests for the validation results and the download (``/validation``, ``/download``).

The privacy gate is enforced here: nothing of the synthetic table (preview or file) before it passes.
"""

import dataclasses
import io
import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.api_config import GATE_ADVICE, PREVIEW_ROWS
from api.dataset import change_choices, load_dataset
from api.errors import ApiError
from api.main import create_app
from api.pipeline import start_training
from api.results import content_disposition, download_name, synthetic_csv, validation_view
from api.session import Session, Stage
from api_csv import make_csv
from sds.data_engine import Issue, UserChoices
from sds.vae.model import Architecture

WAIT_SECONDS = 60
SMALL = Architecture(hidden_dims=(16,), latent_dim=2)
FAST = {"max_steps": 20, "architecture": SMALL, "device": "cpu"}
FAILURE = Issue("copied_rows", "Many synthetic rows copy real records.")


def strict(value) -> None:
    """Fail on NaN or infinities, which are not JSON."""
    json.dumps(value, allow_nan=False)


def validate(session: Session) -> Session:
    job = start_training(session, **FAST)
    job.thread.join(WAIT_SECONDS)
    assert not job.thread.is_alive() and session.stage is Stage.VALIDATED, job.error
    return session


def validated(data: bytes | None = None, name: str = "patients.csv") -> Session:
    session = Session()
    load_dataset(session, make_csv() if data is None else data, name)
    return validate(session)


def pass_gate(session: Session) -> Session:
    """The tiny test model may fail the gate: tests on the format must not depend on it."""
    session.privacy = dataclasses.replace(session.privacy, passed=True, failures=[])
    return session


def fail_gate(session: Session) -> None:
    session.privacy = dataclasses.replace(session.privacy, passed=False, failures=[FAILURE])


def italian_csv() -> bytes:
    """``;``-separated, Latin-1, with accented city names (as an Excel export on an Italian PC)."""
    text = make_csv().decode("utf-8").replace(",", ";").replace("Rome", "Forlì").replace("Milan", "Cantù")
    return text.encode("latin-1")


# --- /validation view -----------------------------------------------------------------------

def test_view_needs_a_validation():
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    with pytest.raises(ApiError) as error:
        validation_view(session)
    assert error.value.status == 409 and error.value.code == "not_trained"          # first missing stage


def test_view_when_the_gate_passes():
    session = pass_gate(validated())
    view = validation_view(session)
    strict(view)
    assert view["gate"] == {"passed": True, "failures": [], "advice": None}
    assert view["n_rows"] == len(session.synthetic) == len(session.prepared.train)
    names = [c.name for c in session.contract.columns]
    assert view["preview"]["columns"] == names
    assert len(view["preview"]["rows"]) == PREVIEW_ROWS
    expected = session.synthetic.head(PREVIEW_ROWS)
    assert view["preview"]["rows"][0] == [None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
                                          for v in expected.iloc[0]]


def test_view_when_the_gate_fails_hides_the_synthetic_rows():
    session = validated()
    fail_gate(session)
    view = validation_view(session)
    assert view["gate"]["passed"] is False
    assert view["gate"]["failures"] == [{"code": "copied_rows", "message": FAILURE.message, "column": None}]
    assert view["gate"]["advice"] == GATE_ADVICE
    assert view["preview"] is None
    assert view["histograms"] and view["associations"]       # the charts stay: they explain the result


def test_advice_says_where_and_how_to_act():
    assert "step 1" in GATE_ADVICE.lower() and "untick" in GATE_ADVICE.lower()
    assert "more rows" in GATE_ADVICE and "same result" in GATE_ADVICE


def test_warnings_privacy_first_then_fidelity_with_their_source():
    session = validated()
    session.privacy = dataclasses.replace(session.privacy, warnings=[Issue("near_copies_unchecked", "p")])
    session.fidelity = dataclasses.replace(session.fidelity, warnings=[Issue("numeric_drift", "f", "age")])
    view = validation_view(session)
    assert view["warnings"] == [
        {"source": "privacy", "code": "near_copies_unchecked", "message": "p", "column": None},
        {"source": "fidelity", "code": "numeric_drift", "message": "f", "column": "age"},
    ]


def test_associations_for_the_two_heatmaps():
    session = validated()
    view = validation_view(session)["associations"]
    real, synthetic = session.fidelity.association_real, session.fidelity.association_synthetic
    assert view["columns"] == list(real.columns)
    assert np.allclose(np.array(view["real"], dtype=float), real.to_numpy(), equal_nan=True)
    assert np.allclose(np.array(view["synthetic"], dtype=float), synthetic.to_numpy(), equal_nan=True)


def test_histograms_are_shares_not_rows():
    session = validated()
    histograms = validation_view(session)["histograms"]
    assert [h["name"] for h in histograms] == [h.name for h in session.fidelity.histograms]
    for h in histograms:
        assert set(h) == {"name", "role", "labels", "real", "synthetic"}
        edges = 1 if h["role"] in ("numeric", "date") else 0           # numeric labels are the bin edges
        assert len(h["labels"]) == len(h["real"]) + edges and len(h["real"]) == len(h["synthetic"])
        for shares in (h["real"], h["synthetic"]):                       # values outside the train range fall out
            assert all(0 <= s <= 1 for s in shares) and sum(shares) <= 1 + 1e-6


def test_no_technical_numbers_in_the_view():
    view = validation_view(validated())
    assert set(view) == {"gate", "warnings", "associations", "histograms", "preview", "n_rows", "download_name"}


def test_view_names_the_download():
    # The UI's download button needs the name before the click: the rule stays in one place, here.
    assert validation_view(validated(name="visite 2026.csv"))["download_name"] == "visite 2026_synthetic.csv"


# --- the CSV --------------------------------------------------------------------------------

def test_csv_is_the_synthetic_table():
    session = pass_gate(validated())
    data = synthetic_csv(session)
    back = pd.read_csv(io.BytesIO(data), sep=",")
    assert list(back.columns) == [c.name for c in session.contract.columns]
    assert len(back) == len(session.synthetic)
    assert not data.decode("utf-8").splitlines()[0].startswith(",")          # no index column
    pd.testing.assert_frame_equal(back, pd.read_csv(io.StringIO(session.synthetic.to_csv(index=False))))


def test_csv_keeps_the_separator_and_is_utf8():
    session = pass_gate(validated(italian_csv(), "pazienti.csv"))
    assert session.inspection.separator == ";"
    text = synthetic_csv(session).decode("utf-8")                              # UTF-8 even from a Latin-1 file
    assert text.splitlines()[0] == ";".join(c.name for c in session.contract.columns)
    assert "Forlì" in text or "Cantù" in text


def test_csv_leaves_out_the_excluded_columns():
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    change_choices(session, UserChoices(excluded={"city"}).to_dict())
    pass_gate(validate(session))
    header = synthetic_csv(session).decode("utf-8").splitlines()[0]
    assert "city" not in header.split(",")


def test_csv_empty_cells_for_missing_values():
    session = pass_gate(validated())
    session.synthetic = session.synthetic.copy()
    session.synthetic.iloc[0, 0] = np.nan
    first_row = synthetic_csv(session).decode("utf-8").splitlines()[1]
    assert first_row.startswith(",") and "nan" not in first_row.lower()


def test_csv_is_refused_when_the_gate_fails():
    session = validated()
    fail_gate(session)
    with pytest.raises(ApiError) as error:
        synthetic_csv(session)
    assert error.value.status == 403 and error.value.code == "privacy_gate"
    assert FAILURE.message in error.value.message and GATE_ADVICE in error.value.message


def test_csv_needs_a_validation():
    with pytest.raises(ApiError) as error:
        synthetic_csv(Session())
    assert error.value.code == "no_dataset"


@pytest.mark.parametrize("source, expected", [
    ("patients.csv", "patients_synthetic.csv"),
    ("export.2024.csv", "export.2024_synthetic.csv"),
    ("data.TSV", "data_synthetic.csv"),
    ("notes", "notes_synthetic.csv"),
    (".csv", "dataset_synthetic.csv"),
])
def test_download_name(source, expected):
    assert download_name(source) == expected


# --- HTTP -----------------------------------------------------------------------------------

def http_validated(data: bytes | None = None, name: str = "patients.csv") -> TestClient:
    app = create_app(dev_mode=False)
    app.state.train_options = dict(FAST)
    c = TestClient(app, raise_server_exceptions=False)
    body = make_csv() if data is None else data
    assert c.post("/upload", params={"name": name}, content=body, headers={"Content-Type": "text/csv"}).status_code == 200
    job_id = c.post("/train").json()["job_id"]
    deadline = time.monotonic() + WAIT_SECONDS
    while c.get(f"/status/{job_id}").json()["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert app.state.session.stage is Stage.VALIDATED
    return c


def test_http_validation_view():
    c = http_validated()
    response = c.get("/validation")
    assert response.status_code == 200 and "gate" in response.json()


def test_http_validation_before_training_is_409():
    c = TestClient(create_app(dev_mode=False), raise_server_exceptions=False)
    c.post("/upload", content=make_csv(), headers={"Content-Type": "text/csv"})
    response = c.get("/validation")
    assert response.status_code == 409 and response.json()["code"] == "not_trained"


def test_http_download_file():
    c = http_validated(italian_csv(), "pazienti forlì.csv")
    pass_gate(c.app.state.session)
    response = c.get("/download")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "charset=utf-8" in response.headers["content-type"]
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert "filename*=UTF-8''pazienti%20forl%C3%AC_synthetic.csv" in disposition
    assert 'filename="pazienti forl_synthetic.csv"' in disposition                # ASCII fallback
    assert response.content == synthetic_csv(c.app.state.session)


def test_http_download_refused_by_the_gate():
    c = http_validated()
    fail_gate(c.app.state.session)
    response = c.get("/download")
    assert response.status_code == 403
    assert response.json()["code"] == "privacy_gate" and GATE_ADVICE in response.json()["message"]


def test_http_download_during_a_new_training_is_409():
    c = http_validated()
    c.app.state.train_options = {**FAST, "max_steps": 10**7, "eval_every": 5}
    c.post("/train")
    try:
        response = c.get("/download")                 # the new training dropped the old results
        assert response.status_code == 409 and response.json()["code"] == "not_trained"
    finally:
        c.post("/train/stop")


def test_content_disposition_survives_quotes_and_backslashes():
    header = content_disposition('a"b\\c.csv')
    fallback = header.split('filename="', 1)[1].split('"', 1)[0]
    assert fallback == "abc.csv"
    assert "filename*=UTF-8''a%22b%5Cc.csv" in header
