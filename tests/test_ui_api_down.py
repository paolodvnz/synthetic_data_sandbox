"""Specification tests: the API may stop answering at any moment, and the page never shows an exception.

Going down is modelled as it happens: the first calls are answered, every later one fails with ``ApiUnavailable``.
Each interaction is replayed with the API going down at every moment of its last run (before the first call, after
each one), so a call added to a step is covered without touching these tests. A button of a step reruns the whole
page, inside the handler of ``app.py``; the exceptions are the dialogs (Start over, Close), drawn outside it, and the
fragments: Streamlit keeps their errors inside them, also on the page's first run, and the browser reruns them by
themselves.
"""

import pytest
from streamlit.testing.v1 import AppTest

import ui.api_client
from sds.config import PROJECT_ROOT
from ui.api_client import ApiUnavailable
from ui.ui_config import API_DOWN_MESSAGE
from ui_fake import PAGE_FUNCTIONS, FakeApi, dataset, job, turn

APP = str(PROJECT_ROOT / "ui" / "app.py")
WAIT_SECONDS = 30


class Outage:
    """Stands for ``ui.api_client``: answers with ``api`` for the first ``working`` calls (``None``: all), then down."""

    def __init__(self, monkeypatch, api: FakeApi, working: int | None = None) -> None:
        self.calls: list[str] = []
        self.working = working
        for name in PAGE_FUNCTIONS:
            monkeypatch.setattr(ui.api_client, name, self.counted(name, getattr(api, name)))

    def counted(self, name: str, answer):
        def call(*args, **kwargs):
            self.calls.append(name)
            if self.working is not None and len(self.calls) > self.working:
                raise ApiUnavailable("connection refused")
            return answer(*args, **kwargs)
        return call



def errors(app: AppTest) -> list[str]:
    return [e.value for e in app.error]


def assert_survives(monkeypatch, make_api, before, last) -> None:
    """``before()`` drives the page with the API up; ``last(app)`` is the action whose run sees the API go down.

    First a normal replay (no error: the interaction is real), then one replay per call ``last`` makes, the API
    going down just before it: no exception, and every error on the page is the plain message.
    """
    outage = Outage(monkeypatch, make_api())
    app = before()
    start = len(outage.calls)
    app = last(app)
    assert not app.exception and not errors(app)
    moments = outage.calls[start:]
    assert moments, "the action makes no call"
    for k, name in enumerate(moments):
        outage = Outage(monkeypatch, make_api())
        app = before()
        outage.working = len(outage.calls) + k
        app = last(app)
        moment = f"down at call {k} of the action ({name})"
        assert not app.exception, moment
        assert errors(app) and set(errors(app)) == {API_DOWN_MESSAGE}, moment      # only the plain message


# --- the page of every step, the API going down at any moment ------------------------------------------------------

VALIDATED = {"stage": "validated", "job": job("done", step=100)}
SCENARIOS = {                                                     # name: (session view, step that opens)
    "upload-empty": ({"stage": "empty", "job": None}, "upload"),
    "upload-dataset": ({"stage": "prepared", "job": None}, "upload"),
    "train-ready": ({"stage": "prepared", "job": None}, "train"),
    "train-running": ({"stage": "prepared", "job": job("running")}, "train"),
    "validate-checking": ({"stage": "generated", "job": job("running", phase="checking_privacy")}, "validate"),
    "validate-done": (VALIDATED, "validate"),
    "ask": (VALIDATED, "ask"),
}


def fake_api(view: dict) -> FakeApi:
    """Developer mode on (its panels make calls too), the language model configured, one answered question."""
    api = FakeApi(dict(view), dev_mode=True, data=None if view["stage"] == "empty" else dataset())
    api.llm_configured, api.turns = True, [turn()]
    return api


def opening(step: str):
    """``before``: the page not run yet, on ``step`` with the developer panels on."""
    def before() -> AppTest:
        app = AppTest.from_file(APP, default_timeout=WAIT_SECONDS)
        app.session_state["step"], app.session_state["dev_panels"] = step, True
        return app
    return before


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_page_survives_the_api_going_down_at_any_moment(monkeypatch, scenario):
    view, step = SCENARIOS[scenario]
    assert_survives(monkeypatch, lambda: fake_api(view), opening(step), lambda app: app.run())


# --- outside the page's handler ------------------------------------------------------------------------------------

def fragment_script(module: str, function: str, *more) -> None:
    """Runs one fragment by itself, as the browser does every POLL_SECONDS or when one of its buttons is pressed."""
    import importlib

    getattr(importlib.import_module(module), function)("abc", *more)


@pytest.mark.parametrize("module, function, more, stage, phase, button", [
    ("ui.train_step", "live_job", (False,), "prepared", "training", "stop_training"),
    ("ui.validate_step", "live_check", (), "generated", "checking_privacy", "stop_check"),
], ids=["train", "validate"])
def test_stop_pressed_when_the_api_goes_down_gives_a_plain_message(monkeypatch, module, function, more, stage, phase,
                                                                   button):
    import streamlit

    monkeypatch.setattr(streamlit, "rerun", streamlit.stop)       # alone, a rerun would draw the fragment for ever

    def before() -> AppTest:
        return AppTest.from_function(fragment_script, args=(module, function, *more),
                                     default_timeout=WAIT_SECONDS).run()

    assert_survives(monkeypatch, lambda: FakeApi({"stage": stage, "job": job("running", phase=phase)}, data=dataset()),
                    before, lambda app: app.button(key=button).click().run())


@pytest.mark.parametrize("opener, confirm", [("start_over", "start_over_confirm"), ("close", "close_confirm")])
def test_a_dialog_confirmed_when_the_api_goes_down_gives_a_plain_message(monkeypatch, opener, confirm):
    def before() -> AppTest:
        app = AppTest.from_file(APP, default_timeout=WAIT_SECONDS).run()
        return app.button(key=opener).click().run()

    assert_survives(monkeypatch, lambda: FakeApi(dict(VALIDATED), data=dataset(), closable=True),
                    before, lambda app: app.button(key=confirm).click().run())
