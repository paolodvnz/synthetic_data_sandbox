"""Tests of the UI's HTTP client and of the page against a real HTTP server (Streamlit AppTest, no browser)."""

import json
import socket
import sys

import pytest
from streamlit.testing.v1 import AppTest

from sds.config import PROJECT_ROOT
from sds.launcher import Command, start, stop, wait_until_ready
from ui.api_client import ApiUnavailable, api_url, health
from ui.ui_config import API_URL_ENV, DEFAULT_API_URL

WAIT_SECONDS = 30
APP = str(PROJECT_ROOT / "ui" / "app.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    """A static server whose /health is the API's answer (dev mode on)."""
    (tmp_path / "health").write_text(json.dumps({"status": "ok", "dev_mode": True, "max_upload_mb": 200, "closable": False}))
    (tmp_path / "session").write_text(json.dumps({"stage": "empty", "job": None}))
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "samples").write_text(json.dumps([{"name": "adult", "available": True}]))
    port = free_port()
    server = start(Command("fake", [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
                                    "--directory", str(tmp_path)], {}, tmp_path))
    url = f"http://127.0.0.1:{port}"
    assert wait_until_ready(f"{url}/health", server, WAIT_SECONDS)
    monkeypatch.setenv(API_URL_ENV, url)
    yield url
    stop([server], WAIT_SECONDS)


def test_api_url(monkeypatch):
    monkeypatch.delenv(API_URL_ENV, raising=False)
    assert api_url() == DEFAULT_API_URL
    monkeypatch.setenv(API_URL_ENV, "http://127.0.0.1:9999")
    assert api_url() == "http://127.0.0.1:9999"


def test_health_when_the_api_is_down(monkeypatch):
    monkeypatch.setenv(API_URL_ENV, f"http://127.0.0.1:{free_port()}")
    with pytest.raises(ApiUnavailable):
        health()


def test_health_when_the_api_answers_with_an_error(tmp_path, monkeypatch):
    port = free_port()                                   # a server with no /health: 404, an HTML page
    server = start(Command("fake", [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
                                    "--directory", str(tmp_path)], {}, tmp_path))
    try:
        assert wait_until_ready(f"http://127.0.0.1:{port}/", server, WAIT_SECONDS)
        monkeypatch.setenv(API_URL_ENV, f"http://127.0.0.1:{port}")
        with pytest.raises(ApiUnavailable):
            health()
    finally:
        stop([server], WAIT_SECONDS)


def test_app_talks_to_a_real_http_server(fake_api):
    app = AppTest.from_file(APP, default_timeout=WAIT_SECONDS).run()
    assert not app.exception and not app.error
    assert app.session_state["step"] == "upload"
    assert app.toggle(key="dev_panels").label == "Show developer panels"     # dev mode read from /health


def test_app_explains_when_the_api_is_down(monkeypatch):
    monkeypatch.setenv(API_URL_ENV, f"http://127.0.0.1:{free_port()}")
    app = AppTest.from_file(APP, default_timeout=WAIT_SECONDS).run()
    assert not app.exception
    assert "sds-app" in app.error[0].value and "Traceback" not in app.error[0].value


# --- the fake API of the UI tests answers like the real one ------------------------------------

def shape(value):
    """The structure of a JSON value: keys and types, not the values (lists: the shape of their first item)."""
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else type(value).__name__


def same_shape(fake, real) -> bool:
    """``fake`` has the structure of ``real`` (``None`` matches anything: a value that may be missing)."""
    if fake is None or real is None:
        return True
    if isinstance(real, dict):
        return isinstance(fake, dict) and set(fake) == set(real) and all(same_shape(fake[k], real[k]) for k in real)
    if isinstance(real, list):
        return isinstance(fake, list) and (not fake or not real or same_shape(fake[0], real[0]))
    return shape(fake) == shape(real)


def test_the_fake_api_answers_like_the_real_one():
    from fastapi.testclient import TestClient

    from api.main import create_app
    from api_csv import make_csv
    from ui_fake import FakeApi, dataset

    real = TestClient(create_app(dev_mode=True))
    fake = FakeApi({"stage": "empty", "job": None})
    assert same_shape(fake.health(), real.get("/health").json())
    assert same_shape(fake.get_session(), real.get("/session").json())
    assert same_shape(fake.list_samples(), real.get("/dev/samples").json())
    view = real.post("/upload", params={"name": "p.csv"}, content=make_csv(),
                     headers={"Content-Type": "text/csv"}).json()
    ours = dataset()
    for key in ("file", "choices", "rows", "problem", "ready"):
        assert same_shape(ours[key], view[key]), key
    assert same_shape(ours["columns"][0], view["columns"][0])
    assert set(ours["issues"]) == set(view["issues"])
    check, real_check = fake.dev_decoding_check(), real.get("/dev/decoding-check").json()
    assert set(check) == set(real_check)                       # identical_share: keys are column names (data)
    assert set(check["random_rows"]) == set(real_check["random_rows"])
    assert set(fake.dev_contract()) == set(real.get("/dev/contract").json())


def test_the_fake_training_answers_like_the_real_one():
    import time

    from fastapi.testclient import TestClient

    from api.main import create_app
    from api_csv import make_csv
    from sds.vae.model import Architecture
    from ui_fake import FakeApi, dataset, job

    real = TestClient(create_app(dev_mode=True))
    real.app.state.train_options = {"max_steps": 40, "eval_every": 10, "device": "cpu",
                                    "architecture": Architecture(hidden_dims=(16,), latent_dim=2)}
    real.post("/upload", params={"name": "p.csv"}, content=make_csv(), headers={"Content-Type": "text/csv"})
    fake = FakeApi({"stage": "prepared", "job": None}, data=dataset())
    assert same_shape(fake.dev_hyperparameters(), real.get("/dev/hyperparameters").json())
    started = real.post("/train").json()
    assert same_shape(fake.start_training(), started)
    deadline = time.monotonic() + WAIT_SECONDS
    while real.get(f"/status/{started['job_id']}").json()["state"] == "running":
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)
    fake.view["job"] = job("done", step=100)
    assert same_shape(fake.get_status("abc"), real.get(f"/status/{started['job_id']}").json())
    assert same_shape(fake.dev_training(), real.get("/dev/training").json())
    fake.view["stage"] = "validated"
    real_validation = real.get("/validation").json()
    ours = fake.get_validation()
    assert set(ours) == set(real_validation)
    for key in ("gate", "associations", "n_rows", "download_name", "preview"):
        assert same_shape(ours[key], real_validation[key]), key
    assert same_shape(ours["histograms"][0], real_validation["histograms"][0])
    assert same_shape(ours["warnings"][1], real_validation["warnings"][0])
    details, real_details = fake.dev_validation(), real.get("/dev/validation").json()
    assert set(details["fidelity"]) == set(real_details["fidelity"])
    for key in ("columns", "association_real", "warnings"):
        assert same_shape(details["fidelity"][key], real_details["fidelity"][key]), key
    assert same_shape(details["privacy"], real_details["privacy"])
    assert same_shape(fake.stop_training(), real.post("/train/stop").json())
    assert same_shape(fake.get_session(), real.get("/session").json())
