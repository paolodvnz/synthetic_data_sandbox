"""Tests of the UI's HTTP client and of the page against a real HTTP server (Streamlit AppTest, no browser)."""

import inspect
import json
import socket
import sys

import pytest
from streamlit.testing.v1 import AppTest

from sds.config import PROJECT_ROOT
from sds.launcher import Command, start, stop, wait_until_ready
from ui.api_client import ApiRefused, ApiUnavailable, api_url, health
from ui.ui_config import API_URL_ENV, DEFAULT_API_URL
from ui_fake import PAGE_FUNCTIONS

WAIT_SECONDS = 30
APP = str(PROJECT_ROOT / "ui" / "app.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    """A static server whose /health is the API's answer (dev mode on)."""
    (tmp_path / "health").write_text(json.dumps({"status": "ok", "dev_mode": True, "max_upload_mb": 200, "closable": False,
                                                       "llm_configured": False, "max_question_chars": 500}))
    (tmp_path / "session").write_text(json.dumps({"stage": "empty", "job": None, "gate_passed": None}))
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


def test_a_refusal_on_health_means_the_api_is_not_there(monkeypatch):
    import ui.api_client

    def refused(method, path, **kwargs):
        raise ApiRefused(404, "not_found", "Not here.")       # another program on the port: the API never refuses

    monkeypatch.setattr(ui.api_client, "call", refused)
    with pytest.raises(ApiUnavailable):
        health()


# --- every function of the client calls a route the API has (the UI tests use the fake, which has no paths) ---------

CLIENT_ARGUMENTS = {"content": b"a,b\n1,2\n", "name": "p.csv", "choices": {}, "job_id": "abc", "question": "Why?"}


def test_the_fake_stands_for_every_client_function():
    import ui.api_client

    plumbing = {"api_url", "call", "request"}
    public = {name for name, f in inspect.getmembers(ui.api_client, inspect.isfunction)
              if f.__module__ == "ui.api_client"} - plumbing
    assert public == set(PAGE_FUNCTIONS)


@pytest.mark.parametrize("name", PAGE_FUNCTIONS)
def test_every_client_function_calls_a_route_of_the_api(monkeypatch, name):
    from starlette.routing import Match

    import ui.api_client
    from api.main import create_app

    sent = []

    class Answer:
        content = b""

        def json(self):
            return {}

    monkeypatch.setattr(ui.api_client, "request", lambda method, path, **kwargs: sent.append((method, path)) or Answer())
    function = getattr(ui.api_client, name)
    function(**{p: CLIENT_ARGUMENTS[p] for p in inspect.signature(function).parameters})
    [(method, path)] = sent
    scope = {"type": "http", "method": method, "path": path, "root_path": ""}
    assert any(route.matches(scope)[0] is Match.FULL for route in create_app(dev_mode=True).routes), f"{method} {path}"


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


def test_the_fake_ask_step_answers_like_the_real_one():
    import dataclasses

    from fastapi.testclient import TestClient

    from api.ask import LlmAccess
    from api.dataset import load_dataset
    from api.main import create_app
    from api.pipeline import start_training
    from api.session import Session
    from api_csv import make_csv
    from llm_fake import FakeClient
    from sds.llm_analyst.settings import LlmSettings
    from sds.vae.model import Architecture
    from ui_fake import FakeApi, dataset, job, turn

    session = Session()
    load_dataset(session, make_csv(), "p.csv")
    start_training(session, max_steps=20, device="cpu",
                   architecture=Architecture(hidden_dims=(16,), latent_dim=2)).thread.join(WAIT_SECONDS)
    session.privacy = dataclasses.replace(session.privacy, passed=True, failures=[])
    chart = "```python\nfig, ax = plt.subplots()\ndf['age'].plot.hist(ax=ax)\nprint(1)\n```\nAges."
    app = create_app(dev_mode=True)
    app.state.session = session
    app.state.llm = LlmAccess(settings=LlmSettings(api_key="sk-test", model="m"), client=FakeClient(chart),
                              problem=None)
    real = TestClient(app)
    fake = FakeApi({"stage": "validated", "job": job("done")}, dev_mode=True, data=dataset())
    fake.llm_configured, fake.answers = True, [turn()]
    assert same_shape(fake.health(), real.get("/health").json())
    assert same_shape(fake.ask("q"), real.post("/ask", json={"question": "q"}).json())
    assert same_shape(fake.get_conversation(), real.get("/conversation").json())
    assert same_shape(fake.new_conversation(), real.post("/conversation/new").json())
    ours, theirs = fake.dev_llm(), real.get("/dev/llm").json()
    assert set(ours) == set(theirs)
    assert same_shape(ours["last"], theirs["last"]) and same_shape(ours["last"]["calls"][0], theirs["last"]["calls"][0])


def test_the_fake_privacy_check_answers_like_the_real_one(monkeypatch):
    """While the real job is held in the privacy check, the fake's views have the same shape."""
    import threading

    from fastapi.testclient import TestClient

    import api.pipeline
    from api.main import create_app
    from api_csv import make_csv
    from sds.vae.model import Architecture
    from ui_fake import FakeApi, dataset, job

    entered, release = threading.Event(), threading.Event()
    real_check = api.pipeline.privacy_check

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(WAIT_SECONDS)
        return real_check(*args, **kwargs)

    monkeypatch.setattr(api.pipeline, "privacy_check", held)
    real = TestClient(create_app(dev_mode=True))
    real.app.state.train_options = {"max_steps": 20, "device": "cpu",
                                    "architecture": Architecture(hidden_dims=(16,), latent_dim=2)}
    real.post("/upload", params={"name": "p.csv"}, content=make_csv(), headers={"Content-Type": "text/csv"})
    started = real.post("/train").json()
    try:
        assert entered.wait(WAIT_SECONDS)
        fake = FakeApi({"stage": "generated", "job": job("running", phase="checking_privacy")}, data=dataset())
        assert fake.get_session() == fake.get_session() | {"stage": real.get("/session").json()["stage"]}
        assert same_shape(fake.get_session(), real.get("/session").json())
        assert real.get(f"/status/{started['job_id']}").json()["phase"] == fake.get_status("abc")["phase"]
        ours, theirs = fake.get_validation(), real.get("/validation").json()
        assert set(ours) == set(theirs) and ours["gate"] is theirs["gate"] is None
        assert ours["preview"] is theirs["preview"] is None
        assert {w["source"] for w in theirs["warnings"]} <= {w["source"] for w in ours["warnings"]} == {"fidelity"}
        assert fake.dev_validation()["privacy"] is real.get("/dev/validation").json()["privacy"] is None
    finally:
        release.set()
        real.post("/train/stop")
