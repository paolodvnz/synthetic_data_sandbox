"""Specification tests for the UI skeleton: the bar of the 4 steps, resuming from ``/session``, "Start over", "Close".

The page runs with ``AppTest`` (no browser) against a fake ``ui.api_client``: no API is started.
"""

import pytest
from streamlit.testing.v1 import AppTest

import ui.api_client
from sds.config import PROJECT_ROOT
from ui.api_client import ApiRefused, ApiUnavailable
from ui.navigation import Step, default_step, reachable_steps
from ui_fake import FakeApi, dataset, install, job

APP = str(PROJECT_ROOT / "ui" / "app.py")
WAIT_SECONDS = 30


# --- navigation (pure) ----------------------------------------------------------------------

@pytest.mark.parametrize("stage, running, expected", [
    ("empty", False, {Step.UPLOAD}),
    ("inspected", False, {Step.UPLOAD}),                  # not ready: the choices must be fixed first
    ("prepared", False, {Step.UPLOAD, Step.TRAIN}),
    ("prepared", True, {Step.UPLOAD, Step.TRAIN}),
    ("validated", False, {Step.UPLOAD, Step.TRAIN, Step.VALIDATE}),
])
def test_reachable_steps(stage, running, expected):
    assert reachable_steps({"stage": stage, "job": job() if running else None}) == expected


def test_ask_is_never_reachable_yet():
    assert Step.ASK not in reachable_steps({"stage": "validated", "job": job("done")})


@pytest.mark.parametrize("stage, job_state, expected", [
    ("empty", None, Step.UPLOAD),
    ("prepared", None, Step.UPLOAD),
    ("prepared", "running", Step.TRAIN),                  # a refresh during the training goes back to the bar
    ("prepared", "stopped", Step.TRAIN),                  # the outcome of the last job is shown where it ran
    ("prepared", "failed", Step.TRAIN),
    ("validated", "done", Step.VALIDATE),
])
def test_default_step(stage, job_state, expected):
    view = {"stage": stage, "job": job(job_state) if job_state else None}
    assert default_step(view) is expected


# --- the page ---------------------------------------------------------------------------------

@pytest.fixture
def fake(monkeypatch):
    def make(view: dict, dev_mode: bool = False) -> FakeApi:
        data = dataset() if view["stage"] != "empty" else None
        return install(monkeypatch, FakeApi(view, dev_mode=dev_mode, data=data))
    return make


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=WAIT_SECONDS).run()
    assert not app.exception
    return app


def step_button(app: AppTest, step: Step):
    return app.button(key=f"step_{step.value}")


def test_empty_session_opens_upload_and_locks_the_rest(fake):
    fake({"stage": "empty", "job": None})
    app = run_app()
    assert not step_button(app, Step.UPLOAD).disabled
    assert all(step_button(app, s).disabled for s in (Step.TRAIN, Step.VALIDATE, Step.ASK))
    assert app.session_state["step"] == Step.UPLOAD.value


def test_running_job_resumes_on_train(fake):
    fake({"stage": "prepared", "job": job("running")})
    app = run_app()
    assert app.session_state["step"] == Step.TRAIN.value


def test_validated_session_resumes_on_validate(fake):
    fake({"stage": "validated", "job": job("done")})
    app = run_app()
    assert app.session_state["step"] == Step.VALIDATE.value
    assert step_button(app, Step.ASK).disabled


def test_clicking_a_reachable_step_opens_it(fake):
    fake({"stage": "validated", "job": job("done")})
    app = run_app()
    step_button(app, Step.UPLOAD).click().run()
    assert app.session_state["step"] == Step.UPLOAD.value


def test_a_step_no_longer_reachable_falls_back(fake):
    api = fake({"stage": "validated", "job": job("done")})
    app = run_app()
    api.view = {"stage": "empty", "job": None}          # e.g. the API was restarted in the meantime
    app.run()
    assert app.session_state["step"] == Step.UPLOAD.value


def test_start_over_asks_first(fake):
    api = fake({"stage": "validated", "job": job("done")})
    app = run_app()
    app.button(key="start_over").click().run()
    assert api.resets == 0                                # only the confirmation dialog opened
    app.button(key="start_over_confirm").click().run()
    assert api.resets == 1
    assert app.session_state["step"] == Step.UPLOAD.value


def test_start_over_forgets_the_ui_state_and_the_uploaded_file(fake):
    fake({"stage": "validated", "job": job("done")})
    app = run_app()
    app.session_state["leftover"] = "something the UI kept"
    generation = app.session_state["uploader_generation"]
    app.button(key="start_over").click().run()
    app.button(key="start_over_confirm").click().run()
    assert "leftover" not in app.session_state
    assert app.session_state["uploader_generation"] == generation + 1   # a new key empties the file uploader


def test_start_over_can_be_cancelled(fake):
    api = fake({"stage": "validated", "job": job("done")})
    app = run_app()
    app.button(key="start_over").click().run()
    app.button(key="start_over_cancel").click().run()
    assert api.resets == 0 and app.session_state["step"] == Step.VALIDATE.value


def test_developer_toggle_only_in_developer_mode(fake):
    fake({"stage": "empty", "job": None}, dev_mode=False)
    assert not run_app().toggle
    fake({"stage": "empty", "job": None}, dev_mode=True)
    app = run_app()
    assert app.toggle(key="dev_panels").label == "Show developer panels"


def test_developer_panel_shows_the_session_view(fake):
    fake({"stage": "prepared", "job": None}, dev_mode=True)
    app = run_app()
    assert not app.json
    app.toggle(key="dev_panels").set_value(True).run()
    assert '{"stage": "prepared", "job": null}' in [j.value for j in app.json]


def test_api_going_down_gives_a_plain_message(fake, monkeypatch):
    fake({"stage": "empty", "job": None})

    def down():
        raise ApiUnavailable("connection refused")

    monkeypatch.setattr(ui.api_client, "get_session", down)
    app = run_app()
    assert "sds-app" in app.error[0].value and "Traceback" not in app.error[0].value


def test_reset_waits_as_long_as_the_api_stop(monkeypatch):
    from api.api_config import STOP_TIMEOUT_SECONDS
    from ui.ui_config import STOP_REQUEST_SECONDS
    assert STOP_REQUEST_SECONDS > STOP_TIMEOUT_SECONDS
    seen = {}
    monkeypatch.setattr(ui.api_client, "call", lambda method, path, **kwargs: seen.update(kwargs) or {})
    ui.api_client.reset()
    assert seen["timeout"] == STOP_REQUEST_SECONDS


def test_api_refusal_keeps_its_message(monkeypatch):
    class Response:
        ok, status_code = False, 409

        def json(self):
            return {"code": "no_dataset", "message": "Upload a dataset first."}

    monkeypatch.setattr(ui.api_client.requests, "request", lambda *a, **k: Response())
    with pytest.raises(ui.api_client.ApiRefused) as error:
        ui.api_client.get_session()
    assert (error.value.status, error.value.code, error.value.message) == (409, "no_dataset", "Upload a dataset first.")


def test_start_over_is_off_when_there_is_nothing_to_forget(fake):
    fake({"stage": "empty", "job": None})
    assert run_app().button(key="start_over").disabled
    fake({"stage": "inspected", "job": None})
    assert not run_app().button(key="start_over").disabled


def test_start_over_keeps_the_developer_toggle(fake):
    fake({"stage": "prepared", "job": None}, dev_mode=True)
    app = run_app()
    app.toggle(key="dev_panels").set_value(True).run()
    app.button(key="start_over").click().run()
    app.button(key="start_over_confirm").click().run()
    assert app.toggle(key="dev_panels").value
    assert '{"stage": "empty", "job": null}' in [j.value for j in app.json]


# --- Close: shut the whole app down from the page (only when sds-app started it) ------------------

def test_close_only_when_the_app_can_close(fake):
    fake({"stage": "empty", "job": None}).closable = False
    assert "close" not in [b.key for b in run_app().button]
    fake({"stage": "empty", "job": None}).closable = True
    assert not run_app().button(key="close").disabled               # nothing to lose, closing is still possible


def test_close_asks_first_then_closes(fake):
    api = fake({"stage": "validated", "job": job("done")})
    api.closable = True
    app = run_app()
    app.button(key="close").click().run()
    assert api.shutdowns == 0                                       # only the confirmation dialog opened
    app.button(key="close_confirm").click().run()
    assert api.shutdowns == 1
    page = " ".join(str(m.value) for m in app.markdown)
    assert "closed" in page.lower() and "close this tab" in page.lower()
    assert not app.button                                          # nothing left to press: the app is gone
    api.view = None                                                # the API is gone: no call on the next run
    app.run()
    assert not app.exception and api.shutdowns == 1               # asked once
    assert "close this tab" in " ".join(str(m.value) for m in app.markdown).lower()


def test_the_goodbye_is_on_the_page_before_the_app_closes(fake, monkeypatch):
    # The API ends right after its answer and sds-app then stops the UI: a page drawn after the call would never
    # reach the browser. The message must already be on the page when the API is asked to close.
    api = fake({"stage": "validated", "job": job("done")})
    api.closable = True
    calls = []
    monkeypatch.setattr(ui.api_client, "shutdown", lambda: calls.append("shutdown") or {"closing": True})
    import streamlit
    real_markdown = streamlit.markdown
    monkeypatch.setattr(streamlit, "markdown", lambda body, *a, **k: calls.append(body) or real_markdown(body, *a, **k))
    app = run_app()
    app.button(key="close").click().run()
    app.button(key="close_confirm").click().run()
    goodbye = [i for i, c in enumerate(calls) if "close this tab" in str(c).lower()]
    assert goodbye and goodbye[0] < calls.index("shutdown")


def test_close_can_be_cancelled(fake):
    api = fake({"stage": "validated", "job": job("done")})
    api.closable = True
    app = run_app()
    app.button(key="close").click().run()
    app.button(key="close_cancel").click().run()
    assert api.shutdowns == 0 and app.session_state["step"] == Step.VALIDATE.value


def test_a_refused_close_is_shown_and_the_app_stays(fake):
    api = fake({"stage": "validated", "job": job("done")})
    api.closable = True
    api.refuse_shutdown = ApiRefused(409, "not_closable", "Stop it from the terminal.")
    app = run_app()
    app.button(key="close").click().run()
    app.button(key="close_confirm").click().run()
    assert "Stop it from the terminal." in [e.value for e in app.error]
    assert app.button(key="close")                                 # still running
    assert "close this tab" not in " ".join(str(m.value) for m in app.markdown).lower()
