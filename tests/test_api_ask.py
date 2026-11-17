"""Specification tests for the Ask step of the API: ``/ask``, ``/conversation``, ``/health``, ``/dev/llm``.

A scripted fake client stands in for the provider (no test calls a real model); the code of the answers runs in
the real sandbox, on the synthetic table of a tiny model trained once for the whole file.
"""

import base64
import dataclasses
import threading

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from api.ask import LlmAccess, status_of
from api.dataset import load_dataset
from api.main import create_app
from api.pipeline import start_training
from api.session import STAGE_FIELDS, Session, Stage
from api_csv import dated_csv, make_csv
from llm_fake import FakeClient, reply
from sds.data_engine import Issue
from sds.llm_analyst import errors
from sds.llm_analyst.prompt import system_prompt
from sds.llm_analyst.sandbox import RUNTIME_MESSAGE
from sds.llm_analyst.settings import LlmNotConfiguredError, LlmSettings
from sds.vae.model import Architecture
from sds.validation.fidelity import ColumnFidelity, difference_phrase

KEY = "sk-test-not-a-real-key"
SETTINGS = LlmSettings(api_key=KEY, model="test-model", base_url="https://provider.test/v1")
FAST = {"max_steps": 20, "architecture": Architecture(hidden_dims=(16,), latent_dim=2), "device": "cpu"}
WAIT_SECONDS = 60

CHART = ("```python\nfig, ax = plt.subplots()\ndf['age'].plot.hist(ax=ax)\nsummary = pd.DataFrame({'columns': [len(df.columns)]})\n```\n"
         "The histogram shows how ages are spread.")
REFUSAL = "I cannot name the patients: the data are synthetic."
BROKEN = "```python\nfig, ax = plt.subplots()\ndf['glucose'].plot.hist(ax=ax)\n```\nGlucose."
QUESTION = "How are ages spread?"


def trained(data: bytes, name: str) -> Session:
    session = Session()
    load_dataset(session, data, name)
    job = start_training(session, **FAST)
    job.thread.join(WAIT_SECONDS)
    assert session.stage is Stage.VALIDATED, job.error
    return session


@pytest.fixture(scope="module")
def template() -> Session:
    return trained(make_csv(), "patients.csv")


def copy_of(template: Session, passed: bool = True) -> Session:
    """A fresh session holding the template's objects (the model is trained once), with the gate set."""
    session = Session()
    for names in STAGE_FIELDS.values():
        for name in names:
            setattr(session, name, getattr(template, name))
    failures = [] if passed else [Issue("copied_rows", "Many synthetic rows copy real records.")]
    session.privacy = dataclasses.replace(template.privacy, passed=passed, failures=failures)
    return session


def http(session: Session | None, fake: FakeClient | None = None, *, dev_mode: bool = False,
         configured: bool = True) -> TestClient:
    app = create_app(dev_mode=dev_mode)
    if session is not None:
        app.state.session = session
    if configured:
        app.state.llm = LlmAccess(settings=SETTINGS, client=fake or FakeClient(), problem=None)
    return TestClient(app, raise_server_exceptions=False)


def ask(client: TestClient, question: str = QUESTION) -> httpx.Response:
    return client.post("/ask", json={"question": question})


# --- a question, an answer ---------------------------------------------------------------------------------------


def test_a_chart_answer(template):
    fake = FakeClient(CHART)
    response = ask(http(copy_of(template), fake))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"question", "kind", "text", "image_png", "summary", "code", "corrected", "notes"}
    assert (body["question"], body["kind"], body["corrected"]) == (QUESTION, "chart", False)
    assert body["text"] == "The histogram shows how ages are spread."
    assert base64.b64decode(body["image_png"]).startswith(b"\x89PNG")
    assert "plot.hist" in body["code"]


def test_a_refusal(template):
    body = ask(http(copy_of(template), FakeClient(REFUSAL)), "Who are the patients?").json()
    assert (body["kind"], body["text"], body["image_png"], body["code"], body["summary"]) == (
        "text", REFUSAL, None, None, None)


def test_the_provider_receives_the_schema_of_the_synthetic_data_and_the_question(template):
    session = copy_of(template)
    fake = FakeClient(CHART)
    client = http(session, fake)
    assert ask(client, f"  {QUESTION}  ").json()["question"] == QUESTION
    assert client.get("/conversation").json()["turns"][0]["question"] == QUESTION
    (request,) = fake.requests
    assert request["model"] == "test-model"
    assert request["messages"] == [{"role": "system", "content": system_prompt(session.contract.llm_schema())},
                                   {"role": "user", "content": QUESTION}]


@pytest.mark.parametrize("separator", [",", ";"])
def test_the_sandbox_reads_the_synthetic_data_with_commas(template, separator):
    """Whatever the separator of the uploaded file, the code sees all the columns."""
    session = copy_of(template)
    session.inspection = dataclasses.replace(session.inspection, separator=separator)
    body = ask(http(session, FakeClient(CHART))).json()
    assert body["summary"] == {"columns": ["columns"], "rows": [[len(session.synthetic.columns)]], "truncated": False}


FIRST_VISIT = ("```python\nfig, ax = plt.subplots()\ndf['age'].plot.hist(ax=ax)\n"
               "summary = pd.DataFrame({'first': [df['visit'].iloc[0]]})\n```\nAges.")


def test_the_sandbox_reads_the_dates_in_iso_format():
    """Whatever the format of the uploaded file, the generated code sees ISO dates (one reading only)."""
    session = copy_of(trained(dated_csv("%d/%m/%Y"), "visits.csv"))
    body = ask(http(session, FakeClient(FIRST_VISIT))).json()
    assert body["summary"]["rows"] == [[session.synthetic["visit"].iloc[0].strftime("%Y-%m-%d")]]


# --- reliability notes: the columns of the chart that the Validate step found less faithful ----------------------

def less_faithful(session: Session, *names: str) -> Session:
    """Mark some columns less faithful, as the fidelity report does (z and excess beyond the limits)."""
    columns = [ColumnFidelity(n, "numeric", distance=0.1, baseline=0.02, spread=0.005, z=16.0, excess=0.08)
               for n in names]
    session.fidelity = dataclasses.replace(session.fidelity, less_faithful=columns)
    return session


def test_a_chart_on_a_less_faithful_column_gets_a_note(template):
    session = less_faithful(copy_of(template), "age")
    body = ask(http(session, FakeClient(CHART))).json()
    (note,) = body["notes"]
    assert "'age'" in note and difference_phrase(session.fidelity.less_faithful[0]) in note


def test_a_less_faithful_column_the_chart_does_not_use_gets_no_note(template):
    session = less_faithful(copy_of(template), "city")
    assert ask(http(session, FakeClient(CHART))).json()["notes"] == []


def test_a_refusal_has_no_note(template):
    session = less_faithful(copy_of(template), "age")
    assert ask(http(session, FakeClient(REFUSAL))).json()["notes"] == []


def test_the_notes_stay_in_the_conversation(template):
    session = less_faithful(copy_of(template), "age")
    client = http(session, FakeClient(CHART))
    notes = ask(client).json()["notes"]
    assert client.get("/conversation").json()["turns"][0]["notes"] == notes and notes


def test_the_provider_does_not_learn_which_columns_are_less_faithful(template):
    session = less_faithful(copy_of(template), "age")
    fake = FakeClient(CHART)
    ask(http(session, fake))
    (request,) = fake.requests
    assert request["messages"][0]["content"] == system_prompt(session.contract.llm_schema())


def test_a_broken_answer_is_fixed(template):
    fake = FakeClient(BROKEN, CHART)
    body = ask(http(copy_of(template), fake)).json()
    assert (body["kind"], body["corrected"]) == ("chart", True)
    assert len(fake.requests) == 2


# --- the conversation --------------------------------------------------------------------------------------------


def test_the_conversation_keeps_the_answers_in_order(template):
    client = http(copy_of(template), FakeClient(CHART, REFUSAL))
    first, second = ask(client).json(), ask(client, "Who are they?").json()
    assert client.get("/conversation").json() == {"turns": [first, second]}


def test_the_next_question_carries_the_history(template):
    fake = FakeClient(CHART, CHART)
    client = http(copy_of(template), fake)
    ask(client)
    ask(client, "And for weight?")
    assert fake.requests[1]["messages"][1:] == [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": CHART},
        {"role": "user", "content": "And for weight?"},
    ]


def test_errors_do_not_enter_the_conversation(template):
    fake = FakeClient(BROKEN, BROKEN, CHART)
    client = http(copy_of(template), fake)
    failed = ask(client)
    assert failed.status_code == 502
    assert failed.json() == {"code": errors.RUNTIME_ERROR, "message": RUNTIME_MESSAGE}
    assert "glucose" not in failed.text  # the detail (KeyError: 'glucose') stays in the developer panel
    ask(client, "And for weight?")
    assert [turn["question"] for turn in client.get("/conversation").json()["turns"]] == ["And for weight?"]
    assert fake.requests[2]["messages"][1:] == [{"role": "user", "content": "And for weight?"}]


def test_a_new_conversation_keeps_the_model(template):
    fake = FakeClient(CHART, CHART)
    client = http(copy_of(template), fake)
    ask(client)
    assert client.post("/conversation/new").json() == {"turns": []}
    assert client.get("/conversation").json() == {"turns": []}
    assert client.get("/session").json()["stage"] == "validated"
    ask(client, "And for weight?")
    assert len(fake.requests[1]["messages"]) == 2  # system prompt and question: no history


def test_the_conversation_is_empty_at_first():
    assert http(None).get("/conversation").json() == {"turns": []}


@pytest.mark.parametrize("change", ["reset", "clear_from"])
def test_a_change_of_the_synthetic_data_drops_the_conversation(template, change):
    session = copy_of(template)
    client = http(session, FakeClient(CHART))
    ask(client)
    if change == "reset":
        client.post("/reset")
    else:
        session.clear_from(Stage.TRAINED)  # what a new training or new choices do
    assert client.get("/conversation").json() == {"turns": []}


# --- what is checked before the question -------------------------------------------------------------------------


def test_the_privacy_gate_comes_first(template):
    fake = FakeClient(CHART)
    response = ask(http(copy_of(template, passed=False), fake))
    assert response.status_code == 403 and response.json()["code"] == "privacy_gate"
    assert fake.requests == []


def test_no_dataset_no_question():
    fake = FakeClient(CHART)
    response = ask(http(None, fake))
    assert response.status_code == 409 and response.json()["code"] == "no_dataset"
    assert fake.requests == []


def test_the_gate_is_checked_before_the_configuration():
    response = ask(http(None, configured=False))
    assert response.json()["code"] == "no_dataset"


def test_an_llm_not_configured(template):
    response = ask(http(copy_of(template), configured=False))
    assert response.status_code == 409 and response.json()["code"] == errors.LLM_NOT_CONFIGURED
    assert "LLM_API_KEY" in response.json()["message"]


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_an_invalid_question_is_422(template, question):
    fake = FakeClient()
    response = ask(http(copy_of(template), fake), question)
    assert response.status_code == 422 and response.json()["code"] == errors.INVALID_QUESTION
    assert fake.requests == []


def test_a_body_without_a_question_is_an_invalid_request(template):
    response = http(copy_of(template)).post("/ask", json={"text": "hello"})
    assert response.status_code == 422 and response.json()["code"] == "invalid_request"


# --- status of the analyst's errors ------------------------------------------------------------------------------

REQUEST = httpx.Request("POST", "https://provider.test/v1/chat/completions")


def status_error(kind, status):
    return kind(f"Error code: {status}", response=httpx.Response(status, request=REQUEST), body=None)


@pytest.mark.parametrize(
    ("script", "status", "code"),
    [
        ((status_error(openai.RateLimitError, 429),), 503, errors.PROVIDER_BUSY),
        ((openai.APIConnectionError(request=REQUEST),), 503, errors.PROVIDER_UNREACHABLE),
        ((status_error(openai.AuthenticationError, 401),), 502, errors.PROVIDER_AUTH),
        ((status_error(openai.NotFoundError, 404),), 502, errors.PROVIDER_MODEL),
        ((status_error(openai.InternalServerError, 500),), 502, errors.PROVIDER_ERROR),
        ((reply(CHART, finish_reason="length"),), 502, errors.INCOMPLETE_ANSWER),
        (("```python\nimport os\n```", "```python\nimport os\n```"), 502, errors.UNSAFE_CODE),
        (("```python\nprint(1)\n```", "```python\nprint(1)\n```"), 502, errors.NO_CHART),
    ],
    ids=["busy", "unreachable", "auth", "model", "provider", "incomplete", "unsafe", "no-chart"],
)
def test_analyst_errors_have_their_status(template, script, status, code):
    response = ask(http(copy_of(template), FakeClient(*script)))
    assert response.status_code == status
    assert set(response.json()) == {"code", "message"} and response.json()["code"] == code
    assert KEY not in response.text


def test_every_analyst_code_has_a_status():
    codes = {value for name, value in vars(errors).items() if name.isupper()}
    assert {code: status_of(code) for code in codes} == {
        **{code: 502 for code in codes},
        errors.INVALID_QUESTION: 422,
        errors.LLM_NOT_CONFIGURED: 409,
        errors.PROVIDER_BUSY: 503,
        errors.PROVIDER_UNREACHABLE: 503,
    }


# --- one question at a time, outside the session lock ------------------------------------------------------------


def ask_in_background(client: TestClient, results: list, question: str = QUESTION) -> threading.Thread:
    thread = threading.Thread(target=lambda: results.append(ask(client, question)))
    thread.start()
    return thread


def test_one_question_at_a_time_and_the_rest_of_the_app_answers(template):
    fake = FakeClient(CHART, hold=True)
    client = http(copy_of(template), fake)
    results = []
    thread = ask_in_background(client, results)
    try:
        assert fake.entered.wait(WAIT_SECONDS)
        second = ask(client, "Another one?")
        assert second.status_code == 409 and second.json()["code"] == "question_running"
        assert client.get("/session").status_code == 200  # the session lock is free while the model thinks
    finally:
        fake.release()
        thread.join(WAIT_SECONDS)
    assert results[0].status_code == 200
    assert len(fake.requests) == 1
    assert len(client.get("/conversation").json()["turns"]) == 1


def test_the_next_question_is_accepted_after_an_error(template):
    client = http(copy_of(template), FakeClient(status_error(openai.RateLimitError, 429), CHART))
    assert ask(client).status_code == 503
    assert ask(client).status_code == 200


@pytest.mark.parametrize("change", ["/conversation/new", "/reset"])
def test_an_answer_overtaken_by_a_change_is_not_kept(template, change):
    """The answer reaches who asked, but it belongs to a conversation (or data) that no longer exists."""
    fake = FakeClient(CHART, hold=True)
    client = http(copy_of(template), fake)
    results = []
    thread = ask_in_background(client, results)
    try:
        assert fake.entered.wait(WAIT_SECONDS)
        client.post(change)
    finally:
        fake.release()
        thread.join(WAIT_SECONDS)
    assert results[0].status_code == 200
    assert client.get("/conversation").json() == {"turns": []}


# --- /health ------------------------------------------------------------------------------------------------------


def test_health_tells_whether_the_llm_is_configured(template):
    assert http(None).get("/health").json()["llm_configured"] is True
    assert http(None, configured=False).get("/health").json()["llm_configured"] is False


def test_the_tests_never_see_a_real_key():
    """``conftest`` empties LLM_API_KEY: the .env of the developer never reaches a test app."""
    access = create_app(dev_mode=False).state.llm
    assert access.settings is None and access.client is None
    assert isinstance(access.problem, LlmNotConfiguredError)


def test_llm_access_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", KEY)
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.test/v1")
    access = LlmAccess.from_environment()
    assert access.settings.model == "env-model" and access.problem is None
    assert access.client.max_retries == 0


# --- /dev/llm ----------------------------------------------------------------------------------------------------


def test_dev_llm_is_absent_without_the_developer_mode():
    assert http(None).get("/dev/llm").status_code == 404


def test_dev_llm_before_any_dataset():
    view = http(None, dev_mode=True).get("/dev/llm").json()
    assert view == {"configured": True, "model": "test-model", "base_url": "https://provider.test/v1",
                    "system_prompt": None, "max_history_turns": 3, "last": None}


def test_dev_llm_shows_the_prompt_as_soon_as_the_dataset_is_prepared():
    session = Session()
    load_dataset(session, make_csv(), "patients.csv")
    assert session.stage is Stage.PREPARED
    view = http(session, dev_mode=True).get("/dev/llm").json()
    assert view["system_prompt"] == system_prompt(session.contract.llm_schema())


def test_dev_llm_not_configured():
    view = http(None, dev_mode=True, configured=False).get("/dev/llm").json()
    assert (view["configured"], view["model"], view["base_url"]) == (False, None, None)


def test_dev_llm_after_a_fixed_answer(template):
    session = copy_of(template)
    client = http(session, FakeClient(reply(BROKEN, prompt_tokens=700, completion_tokens=90), CHART), dev_mode=True)
    ask(client)
    view = client.get("/dev/llm").json()
    assert view["system_prompt"] == system_prompt(session.contract.llm_schema())
    last = view["last"]
    assert (last["question"], last["outcome"], last["corrected"], last["error"]) == (QUESTION, "chart", True, None)
    assert [call["content"] for call in last["calls"]] == [BROKEN, CHART]
    assert (last["calls"][0]["prompt_tokens"], last["calls"][0]["completion_tokens"]) == (700, 90)
    assert last["seconds"] >= sum(call["seconds"] for call in last["calls"])


def test_dev_llm_after_a_failure_shows_the_detail(template):
    client = http(copy_of(template), FakeClient(BROKEN, BROKEN), dev_mode=True)
    ask(client)
    last = client.get("/dev/llm").json()["last"]
    assert last["outcome"] == errors.RUNTIME_ERROR
    assert last["error"]["code"] == errors.RUNTIME_ERROR
    assert last["error"]["detail"].startswith("KeyError: 'glucose'")
    assert [call["content"] for call in last["calls"]] == [BROKEN, BROKEN]


def test_dev_llm_never_shows_the_key(template):
    raised = openai.AuthenticationError(f"Error code: 401 - Incorrect API key provided: {KEY}",
                                        response=httpx.Response(401, request=REQUEST), body=None)
    client = http(copy_of(template), FakeClient(raised), dev_mode=True)
    ask(client)
    assert KEY not in client.get("/dev/llm").text


def test_an_invalid_question_is_shown_in_the_developer_panel(template):
    client = http(copy_of(template), dev_mode=True)
    ask(client, "")
    last = client.get("/dev/llm").json()["last"]
    assert (last["outcome"], last["calls"]) == (errors.INVALID_QUESTION, [])
