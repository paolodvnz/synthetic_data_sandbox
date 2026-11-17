"""Specification tests for the two deliveries of the job: synthetic table and fidelity first, privacy after.

While the privacy check runs the Validate step can already show the comparison, but nothing of the synthetic table
leaves the app (download, preview, questions) until the gate passes. The check is held by a wrapper of
``api.pipeline.privacy_check`` that waits for an event, so "the check is running" is a state the tests control.
Every wait has a timeout: a concurrency bug must fail a test, not hang the suite.
"""

import threading

import pytest
from fastapi.testclient import TestClient

import api.pipeline
from api.api_config import INTERNAL_ERROR_MESSAGE
from api.ask import LlmAccess
from api.dataset import load_dataset
from api.dev_views import validation_details_view
from api.errors import ApiError
from api.main import create_app
from api.pipeline import Job, JobState, Phase, session_view, start_training
from api.results import require_gate, synthetic_csv, validation_view
from api.session import Session, Stage
from api_csv import make_csv
from llm_fake import FakeClient
from sds.llm_analyst.settings import LlmSettings
from sds.validation.privacy import PrivacyCheckStopped
from sds.vae.model import Architecture

WAIT_SECONDS = 60
SMALL = Architecture(hidden_dims=(16,), latent_dim=2)
FAST = {"max_steps": 20, "architecture": SMALL, "device": "cpu"}
SETTINGS = LlmSettings(api_key="sk-test-not-a-real-key", model="test-model", base_url="https://provider.test/v1")


def held(monkeypatch, name: str):
    """Replace ``api.pipeline.<name>`` with a wrapper that waits for ``release`` before calling it."""
    entered, release = threading.Event(), threading.Event()
    real = getattr(api.pipeline, name)

    def wrapper(*args, **kwargs):
        entered.set()
        assert release.wait(WAIT_SECONDS)
        return real(*args, **kwargs)

    monkeypatch.setattr(api.pipeline, name, wrapper)
    return entered, release


def finished(job: Job) -> Job:
    job.thread.join(WAIT_SECONDS)
    assert not job.thread.is_alive(), "the job thread did not end"
    return job


@pytest.fixture
def checking(monkeypatch):
    """A session whose job has delivered the synthetic table and is held in the privacy check.

    Yields ``(session, job, release)``; the job is released and joined at the end of the test.
    """
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    entered, release = held(monkeypatch, "privacy_check")
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS), job.error
    yield session, job, release
    job.stop_event.set()
    release.set()
    job.thread.join(WAIT_SECONDS)


def http(session: Session) -> TestClient:
    app = create_app(dev_mode=True)
    app.state.session = session
    app.state.llm = LlmAccess(settings=SETTINGS, client=FakeClient("Not asked."), problem=None)
    return TestClient(app, raise_server_exceptions=False)


# --- the two deliveries ----------------------------------------------------------------------------------------

def test_the_synthetic_table_comes_before_the_privacy_check(checking):
    session, job, release = checking
    assert job.state is JobState.RUNNING and job.phase is Phase.CHECKING_PRIVACY
    assert session.stage is Stage.GENERATED
    assert session.trained is not None and session.synthetic is not None and session.fidelity is not None
    assert session.privacy is None
    release.set()
    assert finished(job).state is JobState.DONE and session.stage is Stage.VALIDATED


def test_phases_in_order(monkeypatch):
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    phases = []
    for name in ("generate", "fidelity_report", "privacy_check"):
        real = getattr(api.pipeline, name)
        monkeypatch.setattr(api.pipeline, name,
                            lambda *a, _real=real, **k: phases.append(session.job.phase) or _real(*a, **k))
    finished(start_training(session, **FAST))
    assert phases == [Phase.GENERATING, Phase.COMPARING, Phase.CHECKING_PRIVACY]


def test_the_check_can_be_stopped(monkeypatch):
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    seen = {}
    real = api.pipeline.privacy_check

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(api.pipeline, "privacy_check", spy)
    job = finished(start_training(session, **FAST))
    assert seen["stop_event"] is job.stop_event


# --- stop, failure and orphans after the first delivery ----------------------------------------------------------

def test_stop_during_the_check_drops_everything(checking):
    """A synthetic table without a privacy report would keep the gate closed for ever: train again instead."""
    session, job, release = checking
    session.conversation = ["a question asked by mistake"]
    job.stop_event.set()
    release.set()
    assert finished(job).state is JobState.STOPPED and job.error is None
    assert session.stage is Stage.PREPARED
    assert session.trained is None and session.synthetic is None and session.fidelity is None
    assert session.conversation == []


def test_a_check_that_stops_early_ends_the_job_as_stopped(monkeypatch):
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")

    def stopped(*args, stop_event, **kwargs):
        stop_event.set()                                 # the user pressed Stop between two blocks
        raise PrivacyCheckStopped()

    monkeypatch.setattr(api.pipeline, "privacy_check", stopped)
    job = finished(start_training(session, **FAST))
    assert job.state is JobState.STOPPED and job.error is None and session.stage is Stage.PREPARED


def test_a_stop_after_the_last_block_of_the_check_still_drops_everything(monkeypatch):
    """The report is ready when Stop arrives: Stop wins, nothing of this training stays."""
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    real = api.pipeline.privacy_check

    def stopped_at_the_end(*args, stop_event, **kwargs):
        report = real(*args, stop_event=stop_event, **kwargs)
        stop_event.set()                                 # the user pressed Stop after the last block
        return report

    monkeypatch.setattr(api.pipeline, "privacy_check", stopped_at_the_end)
    job = finished(start_training(session, **FAST))
    assert job.state is JobState.STOPPED and job.error is None
    assert session.stage is Stage.PREPARED and session.privacy is None and session.synthetic is None


def test_a_failed_check_drops_everything(monkeypatch):
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")

    def broken(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api.pipeline, "privacy_check", broken)
    job = finished(start_training(session, **FAST))
    assert job.state is JobState.FAILED and session.stage is Stage.PREPARED and session.synthetic is None
    assert job.error == {"code": "internal_error", "message": INTERNAL_ERROR_MESSAGE}


def test_an_orphan_writes_no_first_delivery(monkeypatch):
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    entered, release = held(monkeypatch, "fidelity_report")
    job = start_training(session, **FAST)
    assert entered.wait(WAIT_SECONDS)
    session.job = Job(job_id="newer", max_steps=1, state=JobState.STOPPED)     # e.g. after a reset
    release.set()
    finished(job)
    assert session.stage is Stage.PREPARED and session.trained is None and session.synthetic is None


def test_an_orphan_neither_writes_its_privacy_nor_clears_a_newer_job(checking):
    """After a reset past the timeout a newer job owns the session: the old thread must leave it alone."""
    session, job, release = checking
    newer = Job(job_id="newer", max_steps=1, state=JobState.RUNNING)
    job.stop_event.set()                                 # what the reset did before giving up waiting
    session.job = newer                                  # the fields now stand for the newer job's results
    release.set()
    assert finished(job).state is JobState.STOPPED
    assert session.job is newer and session.stage is Stage.GENERATED and session.privacy is None


# --- the gate while the check runs -------------------------------------------------------------------------------

def test_the_gate_is_pending(checking):
    session, _, _ = checking
    for refused in (require_gate, synthetic_csv):
        with pytest.raises(ApiError) as error:
            refused(session)
        assert error.value.status == 409 and error.value.code == "privacy_pending"
        assert "still running" in error.value.message


def test_http_download_and_ask_are_pending(checking):
    session, _, _ = checking
    client = http(session)
    for response in (client.get("/download"), client.post("/ask", json={"question": "How are ages spread?"})):
        assert response.status_code == 409 and response.json()["code"] == "privacy_pending"
    assert client.app.state.llm.client.requests == []


def test_the_comparison_is_shown_while_the_check_runs(checking):
    session, _, _ = checking
    view = validation_view(session)
    assert view["gate"] is None and view["preview"] is None
    assert view["associations"]["columns"] and view["histograms"]
    assert {w["source"] for w in view["warnings"]} <= {"fidelity"}
    assert view["n_rows"] == len(session.synthetic)


def test_http_validation_while_the_check_runs(checking):
    session, _, _ = checking
    response = http(session).get("/validation")
    assert response.status_code == 200 and response.json()["gate"] is None


def test_session_view_while_the_check_runs(checking):
    session, job, _ = checking
    view = session_view(session)
    assert view["stage"] == "generated" and view["gate_passed"] is None
    assert view["job"]["state"] == "running" and view["job"]["phase"] == "checking_privacy"


def test_developer_view_shows_the_fidelity_first(checking):
    session, _, _ = checking
    view = validation_details_view(session)
    assert view["privacy"] is None and view["fidelity"]["columns"]


def test_nothing_changes_the_dataset_while_the_check_runs(checking):
    session, _, _ = checking
    with pytest.raises(ApiError) as error:
        session.require_idle()
    assert error.value.code == "training_running"
    assert error.value.message == ("The app is still working on the synthetic data: wait for it to finish or "
                                   "press Stop first.")
