"""Specification tests for the Train step: start, progress bar and phases, stop, outcome, developer panels.

The page runs with ``AppTest`` against a fake API (``ui_fake``). The bar refreshes itself in the browser
(``st.fragment(run_every=...)``); here every ``run()`` draws it again from the job the fake returns.
"""

import pytest
from streamlit.testing.v1 import AppTest

from sds.config import PROJECT_ROOT
from ui.api_client import ApiRefused
from ui.train_view import elapsed_text, ended_early, history_frames, progress
from ui_fake import FakeApi, dataset, evaluation, install, job

APP = str(PROJECT_ROOT / "ui" / "app.py")
WAIT_SECONDS = 30


# --- pure functions -------------------------------------------------------------------------

def test_progress_follows_the_steps_then_the_phases():
    assert progress(job(step=25)) == (0.25, "Learning from your data… 25%")
    assert progress(job(step=0)) == (0.0, "Learning from your data… 0%")
    assert progress(job(phase="generating", step=60)) == (1.0, "Creating the synthetic data…")
    assert progress(job(phase="validating", step=60)) == (1.0, "Checking privacy and quality…")


def test_progress_never_passes_the_end():
    assert progress(job(step=150))[0] == 1.0


def test_ended_early_only_for_a_finished_job_short_of_its_steps():
    assert ended_early(job("done", step=60))
    assert not ended_early(job("done", step=100))
    assert not ended_early(job("stopped", step=60)) and not ended_early(job("failed", step=60))


@pytest.mark.parametrize("seconds, text", [(0.4, "0 s"), (42.6, "43 s"), (60, "1 min 0 s"), (102.0, "1 min 42 s"),
                                           (3725, "62 min 5 s")])
def test_elapsed_text(seconds, text):
    assert elapsed_text(seconds) == text


def test_history_frames():
    frames = history_frames([evaluation(10), evaluation(20)])
    assert list(frames["total"].columns) == ["train", "validation"]
    assert list(frames["total"].index) == [10, 20]
    assert list(frames["components"].columns) == ["numeric", "categorical", "kl"]
    assert list(frames["schedule"].columns) == ["kl_weight", "active_dims"]
    assert frames["total"].loc[20, "train"] == pytest.approx(0.15)
    assert frames["total"].loc[20, "validation"] == pytest.approx(0.2)
    assert frames["components"].loc[20, "numeric"] == pytest.approx(0.075)        # the validation's


# --- the page ---------------------------------------------------------------------------------

@pytest.fixture
def fake(monkeypatch):
    def make(job_view=None, *, stage=None, dev_mode=False) -> FakeApi:
        stage = stage or ("validated" if job_view and job_view["state"] == "done" else "prepared")
        return install(monkeypatch, FakeApi({"stage": stage, "job": job_view}, dev_mode=dev_mode, data=dataset()))
    return make


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=WAIT_SECONDS).run()
    assert not app.exception
    return app


def on_train(app: AppTest) -> AppTest:
    app.button(key="step_train").click().run()
    assert not app.exception and app.session_state["step"] == "train"
    return app


def texts(elements) -> str:
    return " | ".join(str(e.value) for e in elements)


def keys(app: AppTest) -> list[str]:
    return [b.key for b in app.button]


def test_a_prepared_dataset_can_be_trained(fake):
    api = fake()
    app = on_train(run_app())
    assert "start_training" in keys(app) and "stop_training" not in keys(app)
    app.button(key="start_training").click().run()
    assert not app.exception and api.trainings == 1
    assert "stop_training" in keys(app) and "start_training" not in keys(app)


def test_a_running_job_shows_the_bar_and_the_time(fake):
    fake(job("running", step=25))
    app = run_app()                                           # a running job opens on Train
    assert app.get("progress")[0].proto.value == 25
    assert "25%" in app.get("progress")[0].proto.text
    assert "1 min 42 s" in texts(app.caption)


def test_the_bar_moves_between_runs(fake):
    api = fake(job("running", step=25))
    app = run_app()
    api.view["job"] = job("running", step=70)
    app.run()
    assert app.get("progress")[0].proto.value == 70


def test_the_phases_after_the_training(fake):
    api = fake(job("running", step=60, phase="generating"))
    app = run_app()
    assert "Creating the synthetic data" in app.get("progress")[0].proto.text
    api.view["job"] = job("running", step=60, phase="validating")
    app.run()
    assert "Checking privacy and quality" in app.get("progress")[0].proto.text


def test_stop(fake):
    api = fake(job("running", step=25))
    app = run_app()
    app.button(key="stop_training").click().run()
    assert not app.exception and api.stops == 1
    assert "Training stopped" in texts(app.info) and "start_training" in keys(app)


def test_a_stop_still_waiting_keeps_the_bar(fake):
    fake(job("running", step=25, stop_requested=True))
    app = run_app()
    assert app.button(key="stop_training").disabled and app.button(key="stop_training").label == "Stopping…"
    assert app.get("progress")


def test_the_end_of_the_job_redraws_the_whole_page(fake):
    api = fake(job("running", step=90))
    app = run_app()
    api.view = {"stage": "validated", "job": job("done", step=100)}
    app.run()
    assert "next_validate" in keys(app) and not app.get("progress")
    assert not app.button(key="step_validate").disabled


def test_finished_shows_the_outcome_and_next(fake):
    fake(job("done", step=100))
    app = on_train(run_app())
    assert "ready" in texts(app.success) and "1 min 42 s" in texts(app.success)
    assert "stopped improving" not in texts(app.info)
    assert "start_training" not in keys(app)                 # training again the same way gives the same result
    app.button(key="next_validate").click().run()
    assert app.session_state["step"] == "validate"


def test_early_stopping_is_explained(fake):
    fake(job("done", step=60))
    app = on_train(run_app())
    assert "stopped improving, so training ended early" in texts(app.info)


def test_a_failed_job_shows_its_message_and_can_start_again(fake):
    fake(job("failed", error={"code": "internal_error", "message": "Something went wrong."}))
    app = on_train(run_app())
    assert "Something went wrong." in texts(app.error) and "start_training" in keys(app)


def test_a_refused_start_explains_why(fake, monkeypatch):
    fake()

    def refuse():
        raise ApiRefused(409, "training_running", "A training is running.")

    monkeypatch.setattr("ui.api_client.start_training", refuse)
    app = on_train(run_app())
    app.button(key="start_training").click().run()
    assert "A training is running." in texts(app.error)


def test_a_job_that_disappeared_redraws_the_page(fake):
    api = fake(job("running", step=25))
    app = run_app()
    api.view = {"stage": "empty", "job": None}               # e.g. Start over from another tab
    app.run()
    assert not app.exception and app.session_state["step"] == "upload"


def test_developer_panels_before_training(fake):
    fake(dev_mode=True)
    app = on_train(run_app())
    app.toggle(key="dev_panels").set_value(True).run()
    assert not app.exception and not app.error
    assert any("hyperparameters" in e.label.lower() for e in app.expander)
    assert not any("curves" in e.label.lower() for e in app.expander)


def test_developer_panels_during_and_after_training(fake):
    api = fake(job("running", step=25), dev_mode=True)
    app = run_app()
    app.toggle(key="dev_panels").set_value(True).run()
    assert not app.exception and any("curves" in e.label.lower() for e in app.expander)
    assert len(app.get("vega_lite_chart")) == 3
    api.view = {"stage": "validated", "job": job("done", step=60)}
    app.run()
    assert not app.exception and "stopped_early" in texts(app.json)
