"""Specification tests for the training job: train → generate → validate in a thread, status, stop.

Every wait has a timeout (``join(timeout)``, loops with a deadline): a concurrency bug must fail a test,
not hang the suite.
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

import api.pipeline
from api.api_config import INTERNAL_ERROR_MESSAGE
from api.dataset import change_choices, load_dataset
from api.errors import ApiError
from api.main import create_app
from api.pipeline import Job, JobState, Phase, job_view, start_training, stop_training
from api.session import Session, Stage
from api_csv import make_csv
from sds.data_engine import UserChoices
from sds.samples import sample_path
from sds.vae.layout import MIXED_SIGMA_LOSS, VaeError
from sds.vae.model import Architecture
from sds.vae.training import step_budget

WAIT_SECONDS = 60                     # longest wait for a job (a short one takes ~1 s)
POLL_SECONDS = 0.02
SMALL = Architecture(hidden_dims=(16,), latent_dim=2)
FAST = {"max_steps": 20, "architecture": SMALL, "device": "cpu"}
# Runs until stopped: early stopping never starts (its warm-up is 10% of the steps), an evaluation every 5 steps.
ENDLESS = {"max_steps": 10**7, "eval_every": 5, "architecture": SMALL, "device": "cpu"}
CSV_HEADERS = {"Content-Type": "text/csv"}


def wait_until(condition, timeout: float = WAIT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the job"
        time.sleep(POLL_SECONDS)


def finished(job: Job) -> Job:
    """Wait for the job's thread to end; fail (not hang) if it does not."""
    job.thread.join(WAIT_SECONDS)
    assert not job.thread.is_alive(), "the job thread did not end"
    return job


def prepared_session() -> Session:
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    assert session.stage is Stage.PREPARED
    return session


def blocking(monkeypatch, name: str):
    """Replace ``api.pipeline.<name>`` with a wrapper that waits for ``release`` before calling it."""
    entered, release = threading.Event(), threading.Event()
    real = getattr(api.pipeline, name)

    def wrapper(*args, **kwargs):
        entered.set()
        assert release.wait(WAIT_SECONDS)
        return real(*args, **kwargs)

    monkeypatch.setattr(api.pipeline, name, wrapper)
    return entered, release


@pytest.fixture
def session():
    """A prepared session whose running job, if any, is stopped at the end of the test."""
    s = prepared_session()
    yield s
    if s.job is not None and s.job.thread is not None:
        s.job.stop_event.set()
        s.job.thread.join(WAIT_SECONDS)


# --- start ----------------------------------------------------------------------------------

def test_start_needs_a_dataset():
    with pytest.raises(ApiError) as error:
        start_training(Session(), **FAST)
    assert error.value.status == 409 and error.value.code == "no_dataset"


def test_start_needs_a_prepared_dataset():
    session = prepared_session()
    session.clear_from(Stage.PREPARED)               # as after a blocking preparation problem
    with pytest.raises(ApiError) as error:
        start_training(session, **FAST)
    assert error.value.code == "not_prepared"


def test_full_pipeline_reaches_validated(session):
    job = finished(start_training(session, **FAST))
    assert job.state is JobState.DONE and job.error is None
    assert session.stage is Stage.VALIDATED
    assert len(session.synthetic) == len(session.prepared.train)
    assert job.max_steps == FAST["max_steps"] and job.latest.step == FAST["max_steps"]
    assert session.trained.loss == MIXED_SIGMA_LOSS and not session.trained.cancelled


def test_default_budget_and_loss(session, monkeypatch):
    seen = {}

    def fake_train(x, layout, **kwargs):
        seen.update(kwargs, n_rows=len(x))
        raise VaeError("fake", "stop here")

    monkeypatch.setattr(api.pipeline, "train_vae", fake_train)
    job = finished(start_training(session))
    expected = step_budget(len(session.prepared.train)).max_steps
    assert seen["max_steps"] == expected == job.max_steps
    assert seen["loss"] == MIXED_SIGMA_LOSS and seen["n_rows"] == len(session.prepared.train)
    assert callable(seen["on_progress"]) and seen["stop_event"] is job.stop_event


def test_results_appear_only_at_the_end(session, monkeypatch):
    entered, release = blocking(monkeypatch, "privacy_check")
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    assert job.phase is Phase.VALIDATING and job.state is JobState.RUNNING
    assert session.stage is Stage.PREPARED and session.trained is None and session.synthetic is None
    release.set()
    assert finished(job).state is JobState.DONE and session.stage is Stage.VALIDATED


def test_phases_in_order(session, monkeypatch):
    phases = []
    for name in ("generate", "fidelity_report", "privacy_check"):
        real = getattr(api.pipeline, name)
        monkeypatch.setattr(api.pipeline, name,
                            lambda *a, _real=real, **k: phases.append(session.job.phase) or _real(*a, **k))
    finished(start_training(session, **FAST))
    assert phases == [Phase.GENERATING, Phase.VALIDATING, Phase.VALIDATING]


def test_a_new_training_starts_from_scratch(session):
    finished(start_training(session, **FAST))
    assert session.stage is Stage.VALIDATED
    job = start_training(session, **ENDLESS)
    assert session.stage is Stage.PREPARED          # old model, synthetic table and reports dropped
    stop_training(session)
    assert job.state is JobState.STOPPED


# --- one job at a time: nothing else changes the dataset while it runs ------------------------

def test_second_start_while_running_is_409(session):
    start_training(session, **ENDLESS)
    with pytest.raises(ApiError) as error:
        start_training(session, **FAST)
    assert error.value.status == 409 and error.value.code == "training_running"


def test_choices_and_upload_are_refused_while_running(session):
    start_training(session, **ENDLESS)
    before = session.choices
    with pytest.raises(ApiError) as error:
        change_choices(session, UserChoices(excluded={"city"}).to_dict())
    assert error.value.code == "training_running"
    with pytest.raises(ApiError) as error:
        load_dataset(session, make_csv(seed=1), "other.csv")
    assert error.value.code == "training_running"
    assert session.choices is before and session.source_name == "patients.csv"


# --- progress -------------------------------------------------------------------------------

def test_progress_is_reported_while_training(session):
    job = start_training(session, **ENDLESS)
    wait_until(lambda: job.latest is not None and job.latest.step >= 2 * ENDLESS["eval_every"])
    view = job_view(session, job.job_id)
    assert view["state"] == "running" and view["phase"] == "training"
    assert view["step"] >= 2 * ENDLESS["eval_every"] and view["max_steps"] == ENDLESS["max_steps"]
    assert view["stop_requested"] is False and view["error"] is None


def test_view_shows_both_losses_of_the_latest_evaluation(session):
    job = finished(start_training(session, **FAST))
    view = job_view(session, job.job_id)
    assert view["train_loss"] == job.latest.train.total and view["validation_loss"] == job.latest.validation.total
    assert view["train_loss"] != view["validation_loss"]


def test_view_before_the_first_evaluation(session):
    job = Job(job_id="abc", max_steps=100)
    session.job = job
    view = job_view(session, "abc")
    assert view == {"job_id": "abc", "state": "running", "phase": "training", "step": 0, "max_steps": 100,
                    "train_loss": None, "validation_loss": None, "elapsed_seconds": view["elapsed_seconds"],
                    "stop_requested": False, "error": None}
    assert view["elapsed_seconds"] >= 0


def test_elapsed_time_stops_at_the_end(session):
    job = finished(start_training(session, **FAST))
    first = job_view(session, job.job_id)["elapsed_seconds"]
    time.sleep(0.05)
    assert job_view(session, job.job_id)["elapsed_seconds"] == first > 0


def test_unknown_job_is_404(session):
    with pytest.raises(ApiError) as error:
        job_view(session, "nope")
    assert error.value.status == 404 and error.value.code == "unknown_job"
    old = finished(start_training(session, **FAST))
    finished(start_training(session, **FAST))
    with pytest.raises(ApiError):                    # a replaced job is gone
        job_view(session, old.job_id)


# --- stop -----------------------------------------------------------------------------------

def test_stop_during_training_then_change_the_choices_and_train_again(session):
    """The user's case: a useless column noticed during training → Stop → untick it → train again."""
    job = start_training(session, **ENDLESS)
    wait_until(lambda: job.latest is not None)
    stopped = stop_training(session)
    assert stopped is job and job.state is JobState.STOPPED and not job.thread.is_alive()
    assert session.stage is Stage.PREPARED and session.trained is None
    change_choices(session, UserChoices(excluded={"city"}).to_dict())
    assert session.stage is Stage.PREPARED and "city" not in session.schema.columns
    again = finished(start_training(session, **FAST))
    assert again.state is JobState.DONE and session.stage is Stage.VALIDATED


def test_stop_between_phases_writes_nothing(session, monkeypatch):
    real = api.pipeline.generate

    def generate_then_stop(*args, **kwargs):
        table = real(*args, **kwargs)
        session.job.stop_event.set()                 # the user pressed Stop while the table was generated
        return table

    calls = []
    monkeypatch.setattr(api.pipeline, "generate", generate_then_stop)
    monkeypatch.setattr(api.pipeline, "fidelity_report", lambda *a, **k: calls.append("fidelity"))
    job = finished(start_training(session, **FAST))
    assert job.state is JobState.STOPPED and session.stage is Stage.PREPARED
    assert calls == []                               # no time wasted on the next phases


def test_stop_during_training_skips_the_next_phases(session, monkeypatch):
    calls = []
    monkeypatch.setattr(api.pipeline, "generate", lambda *a, **k: calls.append("generate"))
    job = start_training(session, **ENDLESS)
    wait_until(lambda: job.latest is not None)
    stop_training(session)
    assert job.state is JobState.STOPPED and calls == []


def test_stop_during_fidelity_skips_the_privacy_check(session, monkeypatch):
    calls = []
    entered, release = blocking(monkeypatch, "fidelity_report")
    monkeypatch.setattr(api.pipeline, "privacy_check", lambda *a, **k: calls.append("privacy"))
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    job.stop_event.set()
    release.set()
    assert finished(job).state is JobState.STOPPED and calls == []


def test_stop_requested_during_the_last_phase_wins(session, monkeypatch):
    entered, release = blocking(monkeypatch, "privacy_check")
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    job.stop_event.set()
    release.set()
    assert finished(job).state is JobState.STOPPED and session.stage is Stage.PREPARED


def test_stop_waits_for_the_thread(session, monkeypatch):
    entered, release = blocking(monkeypatch, "privacy_check")
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    threading.Timer(0.3, release.set).start()
    stop_training(session, timeout=WAIT_SECONDS)
    assert not job.thread.is_alive() and job.state is JobState.STOPPED


def test_stop_after_the_timeout_still_running(session, monkeypatch):
    entered, release = blocking(monkeypatch, "privacy_check")
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    stop_training(session, timeout=0.05)
    view = job_view(session, job.job_id)
    assert view["state"] == "running" and view["stop_requested"] is True
    release.set()
    assert finished(job).state is JobState.STOPPED


def test_stop_does_not_hold_the_lock_while_waiting(session, monkeypatch):
    entered, release = blocking(monkeypatch, "privacy_check")
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    stopper = threading.Thread(target=stop_training, args=(session,), kwargs={"timeout": WAIT_SECONDS})
    stopper.start()
    wait_until(lambda: job.stop_event.is_set())
    got = session.lock.acquire(timeout=1)            # /status must answer while Stop waits
    assert got
    session.lock.release()
    release.set()
    stopper.join(WAIT_SECONDS)
    assert not stopper.is_alive()


def test_stop_without_a_job_is_409(session):
    with pytest.raises(ApiError) as error:
        stop_training(session)
    assert error.value.status == 409 and error.value.code == "no_training"


def test_stop_after_the_end_changes_nothing(session):
    job = finished(start_training(session, **FAST))
    assert stop_training(session).state is JobState.DONE and session.stage is Stage.VALIDATED
    assert job.stop_event.is_set() is False


# --- failures and orphan threads ------------------------------------------------------------

def test_known_error_fails_the_job_with_its_message(session, monkeypatch):
    def broken(*args, **kwargs):
        raise VaeError("too_wide", "The table is too wide.")

    monkeypatch.setattr(api.pipeline, "generate", broken)
    job = finished(start_training(session, **FAST))
    assert job.state is JobState.FAILED and session.stage is Stage.PREPARED
    assert job.error == {"code": "too_wide", "message": "The table is too wide."}


def test_unexpected_error_hides_the_details(session, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api.pipeline, "fidelity_report", broken)
    job = finished(start_training(session, **FAST))
    assert job.state is JobState.FAILED and session.stage is Stage.PREPARED
    assert job.error == {"code": "internal_error", "message": INTERNAL_ERROR_MESSAGE}
    # A failed job is not running: the user can change the choices.
    change_choices(session, UserChoices(excluded={"city"}).to_dict())


def test_orphan_thread_writes_nothing(session, monkeypatch):
    entered, release = blocking(monkeypatch, "privacy_check")
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    session.job = Job(job_id="newer", max_steps=1, state=JobState.STOPPED)   # e.g. after a reset
    release.set()
    finished(job)
    assert session.stage is Stage.PREPARED and session.trained is None


# --- HTTP -----------------------------------------------------------------------------------

def client(*, dev_mode: bool = False, options: dict = FAST) -> TestClient:
    app = create_app(dev_mode=dev_mode)
    app.state.train_options = dict(options)
    return TestClient(app, raise_server_exceptions=False)


def stop_all(c: TestClient) -> None:
    job = c.app.state.session.job
    if job is not None and job.thread is not None:
        job.stop_event.set()
        job.thread.join(WAIT_SECONDS)


def uploaded(**kwargs) -> TestClient:
    c = client(**kwargs)
    assert c.post("/upload", params={"name": "p.csv"}, content=make_csv(), headers=CSV_HEADERS).status_code == 200
    return c


def test_app_trains_with_the_defaults():
    assert create_app(dev_mode=False).state.train_options == {}


def test_http_train_then_status_until_done():
    c = uploaded()
    response = c.post("/train")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "running" and body["max_steps"] == FAST["max_steps"]
    wait_until(lambda: c.get(f"/status/{body['job_id']}").json()["state"] != "running")
    final = c.get(f"/status/{body['job_id']}").json()
    assert final["state"] == "done" and final["step"] == FAST["max_steps"]
    assert c.app.state.session.stage is Stage.VALIDATED


def test_http_train_without_a_dataset_is_409():
    response = client().post("/train")
    assert response.status_code == 409 and response.json()["code"] == "no_dataset"


def test_http_unknown_status_is_404():
    response = uploaded().get("/status/nope")
    assert response.status_code == 404 and response.json()["code"] == "unknown_job"


def test_http_stop_and_retrain():
    c = uploaded(options=ENDLESS)
    try:
        job_id = c.post("/train").json()["job_id"]
        assert c.post("/train").json()["code"] == "training_running"
        refused = c.post("/choices", json=UserChoices(excluded={"city"}).to_dict())
        assert refused.status_code == 409 and refused.json()["code"] == "training_running"
        upload = c.post("/upload", params={"name": "q.csv"}, content=make_csv(seed=3), headers=CSV_HEADERS)
        assert upload.status_code == 409
        stopped = c.post("/train/stop")
        assert stopped.status_code == 200 and stopped.json()["state"] == "stopped"
        assert stopped.json()["job_id"] == job_id
        assert c.post("/choices", json=UserChoices(excluded={"city"}).to_dict()).status_code == 200
    finally:
        stop_all(c)


def test_http_stop_without_a_job_is_409():
    response = uploaded().post("/train/stop")
    assert response.status_code == 409 and response.json()["code"] == "no_training"


def test_http_failed_job_shows_a_plain_message(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api.pipeline, "generate", broken)
    c = uploaded()
    job_id = c.post("/train").json()["job_id"]
    wait_until(lambda: c.get(f"/status/{job_id}").json()["state"] != "running")
    body = c.get(f"/status/{job_id}").json()
    assert body["state"] == "failed" and "secret" not in str(body)
    assert body["error"]["code"] == "internal_error"


def test_http_sample_is_refused_while_running():
    if not sample_path("adult").is_file():
        pytest.skip("sample CSVs not downloaded: run sds-download-samples")
    c = uploaded(dev_mode=True, options=ENDLESS)
    try:
        c.post("/train")
        response = c.post("/dev/samples/adult")
        assert response.status_code == 409 and response.json()["code"] == "training_running"
    finally:
        stop_all(c)
