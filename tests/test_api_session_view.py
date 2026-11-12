"""Specification tests for ``/session``: the stage and the latest job, so the UI can resume after a page refresh.

Every wait has a timeout: a concurrency bug must fail a test, not hang the suite.
"""

import time

import pytest
from fastapi.testclient import TestClient

from api.dataset import load_dataset
from api.main import create_app
from api.pipeline import job_view, reset_session, session_view, start_training, stop_training
from api.session import Session
from api_csv import make_csv
from sds.vae.model import Architecture

WAIT_SECONDS = 60
SMALL = Architecture(hidden_dims=(16,), latent_dim=2)
FAST = {"max_steps": 20, "architecture": SMALL, "device": "cpu"}
ENDLESS = {"max_steps": 10**7, "eval_every": 5, "architecture": SMALL, "device": "cpu"}
CSV_HEADERS = {"Content-Type": "text/csv"}


def prepared_session() -> Session:
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    return session


@pytest.fixture
def session():
    """A prepared session whose running job, if any, is stopped at the end of the test."""
    s = prepared_session()
    yield s
    if s.job is not None and s.job.thread is not None:
        s.job.stop_event.set()
        s.job.thread.join(WAIT_SECONDS)


def test_empty_session():
    assert session_view(Session()) == {"stage": "empty", "job": None}


def test_prepared_session_without_a_job(session):
    assert session_view(session) == {"stage": "prepared", "job": None}


def test_running_job_is_shown_with_the_prepared_stage(session):
    job = start_training(session, **ENDLESS)
    view = session_view(session)
    assert view["stage"] == "prepared"
    assert view["job"]["job_id"] == job.job_id and view["job"]["state"] == "running"


def test_job_is_the_same_view_as_status(session):
    job = start_training(session, **FAST)
    job.thread.join(WAIT_SECONDS)
    view = session_view(session)
    assert view["stage"] == "validated"
    assert view["job"] == job_view(session, job.job_id) | {"elapsed_seconds": view["job"]["elapsed_seconds"]}
    assert view["job"]["state"] == "done"


def test_stopped_job_stays_visible(session):
    start_training(session, **ENDLESS)
    stop_training(session, WAIT_SECONDS)
    view = session_view(session)
    assert view["stage"] == "prepared" and view["job"]["state"] == "stopped"


def test_after_reset_nothing_is_left(session):
    start_training(session, **FAST).thread.join(WAIT_SECONDS)
    reset_session(session)
    assert session_view(session) == {"stage": "empty", "job": None}


def test_http_session():
    c = TestClient(create_app(dev_mode=False), raise_server_exceptions=False)
    c.app.state.train_options = dict(FAST)
    assert c.get("/session").json() == {"stage": "empty", "job": None}
    assert c.post("/upload", params={"name": "p.csv"}, content=make_csv(), headers=CSV_HEADERS).status_code == 200
    job_id = c.post("/train").json()["job_id"]
    deadline = time.monotonic() + WAIT_SECONDS
    while c.get("/session").json()["job"]["state"] == "running":
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)
    body = c.get("/session").json()
    assert body["stage"] == "validated" and body["job"]["job_id"] == job_id
