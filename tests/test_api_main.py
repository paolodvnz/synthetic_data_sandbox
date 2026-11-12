"""Specification tests for api.main: the app factory, /health, /shutdown and the error format.

/shutdown (the "Close" button): only when ``sds-app`` started the API (``SDS_CLOSABLE``, shown by /health as
``closable``); the API answers first, then ends with ``CLOSED_BY_USER_EXIT_CODE``, which ``sds-app`` reads as a
wanted stop. Without ``sds-app`` nobody would stop the UI: 409 ``not_closable``.
"""

import logging

import pytest

import api.main
from fastapi.testclient import TestClient

from api.api_config import DEV_MODE_ENV
from api.errors import ApiError
from api.main import close_process, create_app
from sds.config import CLOSABLE_ENV, CLOSED_BY_USER_EXIT_CODE, MAX_UPLOAD_MB
from sds.data_engine import DataEngineError
from sds.vae.layout import VaeError


def client(app=None) -> TestClient:
    return TestClient(app or create_app(dev_mode=False), raise_server_exceptions=False)


def test_health(monkeypatch):
    monkeypatch.delenv(CLOSABLE_ENV, raising=False)
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "dev_mode": False, "max_upload_mb": MAX_UPLOAD_MB, "closable": False}


# --- /shutdown ------------------------------------------------------------------------------

@pytest.fixture
def exits(monkeypatch) -> list[int]:
    """The exit codes the API would end with (the process is not really ended)."""
    codes = []
    monkeypatch.setattr(api.main, "close_process", codes.append)
    return codes


def test_closable_only_when_sds_app_says_so(monkeypatch):
    monkeypatch.setenv(CLOSABLE_ENV, "1")
    assert client(create_app(dev_mode=False)).get("/health").json()["closable"] is True
    monkeypatch.setenv(CLOSABLE_ENV, "0")
    assert client(create_app(dev_mode=False)).get("/health").json()["closable"] is False
    assert client(create_app(dev_mode=False, closable=True)).get("/health").json()["closable"] is True


def test_shutdown_answers_then_ends_the_api_with_the_closing_code(exits):
    response = client(create_app(dev_mode=False, closable=True)).post("/shutdown")
    assert response.status_code == 200 and response.json() == {"closing": True}
    assert exits == [CLOSED_BY_USER_EXIT_CODE]          # after the answer (a background task of the response)


def test_shutdown_is_refused_without_sds_app(exits):
    response = client(create_app(dev_mode=False, closable=False)).post("/shutdown")
    assert response.status_code == 409 and response.json()["code"] == "not_closable"
    assert "terminal" in response.json()["message"]
    assert exits == []


def test_close_process_ends_the_process_at_once(monkeypatch):
    # os._exit, not sys.exit: from a worker thread sys.exit would end only the thread, and uvicorn's own
    # shutdown would wait for a running training. Everything in memory goes with the process.
    codes = []
    monkeypatch.setattr(api.main.os, "_exit", codes.append)
    close_process(CLOSED_BY_USER_EXIT_CODE)
    assert codes == [CLOSED_BY_USER_EXIT_CODE]


def test_the_closing_code_is_not_one_of_the_usual_ones():
    # 0 success, 1 error, 2 usage, 3 uvicorn startup failure, 126-128+ shells and signals.
    assert CLOSED_BY_USER_EXIT_CODE not in {0, 1, 2, 3} and 3 < CLOSED_BY_USER_EXIT_CODE < 126


def test_upload_limit_is_200_mb():
    assert MAX_UPLOAD_MB == 200


def test_dev_mode_argument():
    assert client(create_app(dev_mode=True)).get("/health").json()["dev_mode"] is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_dev_mode_on_from_the_environment(monkeypatch, value):
    monkeypatch.setenv(DEV_MODE_ENV, value)
    assert client(create_app()).get("/health").json()["dev_mode"] is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "dev"])
def test_dev_mode_off_from_the_environment(monkeypatch, value):
    monkeypatch.setenv(DEV_MODE_ENV, value)
    assert client(create_app()).get("/health").json()["dev_mode"] is False


def test_dev_mode_off_without_the_variable(monkeypatch):
    monkeypatch.delenv(DEV_MODE_ENV, raising=False)
    assert client(create_app()).get("/health").json()["dev_mode"] is False


def test_each_app_has_its_own_session():
    first, second = create_app(dev_mode=False), create_app(dev_mode=False)
    assert first.state.session is not second.state.session


def app_raising(error: Exception):
    """An app with one extra route that raises ``error``."""
    app = create_app(dev_mode=False)

    @app.get("/boom")
    def boom():
        raise error

    return app


@pytest.mark.parametrize(("error", "status", "code"), [
    (ApiError(409, "not_trained", "Train the model first."), 409, "not_trained"),
    (DataEngineError("empty_file", "The file is empty."), 422, "empty_file"),
    (VaeError("invalid_contract", "The description of the data cannot be used."), 422, "invalid_contract"),
])
def test_known_errors_become_code_and_message(error, status, code):
    response = client(app_raising(error)).get("/boom")
    assert response.status_code == status
    assert response.json() == {"code": code, "message": error.message}


def test_unexpected_errors_hide_the_details(caplog):
    with caplog.at_level(logging.ERROR):
        response = client(app_raising(RuntimeError("secret detail at line 42"))).get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert "secret" not in response.text and "Traceback" not in response.text
    assert "secret detail at line 42" in caplog.text            # the developer still finds it in the log


def test_unknown_route_keeps_the_same_format():
    response = client().get("/nowhere")
    assert response.status_code == 404
    assert set(response.json()) == {"code", "message"}
