"""Specification tests for ``/reset``: stop the job, forget the dataset, free the GPU memory.

Every wait has a timeout: a concurrency bug must fail a test, not hang the suite.
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

import api.pipeline
from api.dataset import load_dataset
from api.errors import ApiError
from api.main import create_app
from api.pipeline import JobState, job_view, reset_session, start_training, stop_training
from api.session import STAGE_FIELDS, Session, Stage
from api_csv import make_csv
from sds.vae.model import Architecture

WAIT_SECONDS = 60
SMALL = Architecture(hidden_dims=(16,), latent_dim=2)
FAST = {"max_steps": 20, "architecture": SMALL, "device": "cpu"}
ENDLESS = {"max_steps": 10**7, "eval_every": 5, "architecture": SMALL, "device": "cpu"}
CSV_HEADERS = {"Content-Type": "text/csv"}


def wait_until(condition, timeout: float = WAIT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def prepared_session() -> Session:
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    return session


def assert_empty(session: Session) -> None:
    assert session.stage is Stage.EMPTY and session.job is None and session.problem is None
    for names in STAGE_FIELDS.values():
        for name in names:
            assert getattr(session, name) is None, name


def test_reset_an_empty_session_is_fine():
    session = Session()
    reset_session(session)
    assert_empty(session)


def test_reset_after_a_finished_job_forgets_everything():
    session = prepared_session()
    job = start_training(session, **FAST)
    job.thread.join(WAIT_SECONDS)
    assert session.stage is Stage.VALIDATED
    reset_session(session)
    assert_empty(session)
    with pytest.raises(ApiError) as error:
        job_view(session, job.job_id)
    assert error.value.code == "unknown_job"
    with pytest.raises(ApiError) as error:
        stop_training(session)
    assert error.value.code == "no_training"


def test_reset_stops_a_running_job_and_waits_for_it():
    session = prepared_session()
    job = start_training(session, **ENDLESS)
    wait_until(lambda: job.latest is not None)
    reset_session(session)
    assert job.stop_event.is_set() and not job.thread.is_alive() and job.state is JobState.STOPPED
    assert_empty(session)


def test_reset_with_a_stuck_job_still_empties_and_the_orphan_writes_nothing(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = api.pipeline.privacy_check

    def stuck(*args, **kwargs):
        entered.set()
        assert release.wait(WAIT_SECONDS)
        return real(*args, **kwargs)

    monkeypatch.setattr(api.pipeline, "privacy_check", stuck)
    session = prepared_session()
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    reset_session(session, timeout=0.05)
    assert_empty(session)                                       # emptied although the thread is still there
    load_dataset(session, make_csv(seed=2), "new.csv")          # a new dataset meanwhile
    release.set()
    job.thread.join(WAIT_SECONDS)
    assert not job.thread.is_alive()
    assert session.stage is Stage.PREPARED and session.source_name == "new.csv" and session.trained is None


def test_reset_does_not_hold_the_lock_while_waiting(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = api.pipeline.privacy_check

    def stuck(*args, **kwargs):
        entered.set()
        assert release.wait(WAIT_SECONDS)
        return real(*args, **kwargs)

    monkeypatch.setattr(api.pipeline, "privacy_check", stuck)
    session = prepared_session()
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    resetter = threading.Thread(target=reset_session, args=(session,), kwargs={"timeout": WAIT_SECONDS})
    resetter.start()
    wait_until(lambda: job.stop_event.is_set())
    assert session.lock.acquire(timeout=1)                      # /status answers while Reset waits
    session.lock.release()
    release.set()
    resetter.join(WAIT_SECONDS)
    assert not resetter.is_alive()
    assert_empty(session)


def test_reset_frees_the_gpu_memory(monkeypatch):
    calls = []
    monkeypatch.setattr(api.pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api.pipeline.torch.cuda, "empty_cache", lambda: calls.append("empty"))
    reset_session(prepared_session())
    assert calls == ["empty"]


def test_reset_without_cuda_does_not_touch_it(monkeypatch):
    monkeypatch.setattr(api.pipeline.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(api.pipeline.torch.cuda, "empty_cache", lambda: pytest.fail("no CUDA here"))
    reset_session(prepared_session())


# --- HTTP -----------------------------------------------------------------------------------

def client(options: dict = FAST) -> TestClient:
    app = create_app(dev_mode=False)
    app.state.train_options = dict(options)
    return TestClient(app, raise_server_exceptions=False)


def test_http_reset_then_start_over():
    c = client(ENDLESS)
    assert c.post("/upload", params={"name": "p.csv"}, content=make_csv(), headers=CSV_HEADERS).status_code == 200
    job_id = c.post("/train").json()["job_id"]
    response = c.post("/reset")
    assert response.status_code == 200 and response.json() == {"stage": "empty"}
    assert c.get(f"/status/{job_id}").json()["code"] == "unknown_job"
    assert c.get("/dataset").json()["code"] == "no_dataset"
    assert c.post("/upload", params={"name": "q.csv"}, content=make_csv(seed=4), headers=CSV_HEADERS).status_code == 200


def test_http_reset_twice():
    c = client()
    assert c.post("/reset").status_code == 200
    assert c.post("/reset").json() == {"stage": "empty"}
