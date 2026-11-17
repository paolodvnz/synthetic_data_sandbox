"""Tests for the analyst: history, one fix attempt, provider errors, question check.

A scripted fake client stands in for the provider: no test calls a real model. The code of the answers runs in the
real sandbox, on a small synthetic CSV.
"""

import io
from types import SimpleNamespace

import httpx
import openai
import pandas as pd
import pytest

from llm_fake import FakeClient, reply
from sds.llm_analyst import analyst
from sds.llm_analyst.analyst import AskResult, Exchange, ask, make_client
from sds.llm_analyst.errors import (
    INCOMPLETE_ANSWER,
    INVALID_QUESTION,
    NO_CHART,
    PROVIDER_AUTH,
    PROVIDER_BUSY,
    PROVIDER_ERROR,
    PROVIDER_MODEL,
    PROVIDER_QUOTA,
    PROVIDER_UNREACHABLE,
    RUNTIME_ERROR,
    TIMEOUT,
    UNSAFE_CODE,
    AnalystError,
)
from sds.llm_analyst.llm_config import (
    ANSWER_TOKENS,
    MAX_ERROR_CHARS,
    MAX_HISTORY_TURNS,
    MAX_QUESTION_CHARS,
    REQUEST_TIMEOUT_SECONDS,
)
from sds.llm_analyst.prompt import system_prompt
from sds.llm_analyst.sandbox import ChartResult
from sds.llm_analyst.settings import LlmSettings

KEY = "sk-test-not-a-real-key"
SETTINGS = LlmSettings(api_key=KEY, model="test-model")
SCHEMA = [
    {"name": "age", "type": "integer"},
    {"name": "bmi", "type": "float"},
    {"name": "smoker", "type": "categorical", "values": ["no", "yes"]},
]
SENTINEL = 987.654  # a value that exists only in the rows: it must never reach the provider
CSV = pd.DataFrame({
    "age": [30, 41, 52, 63],
    "bmi": [22.5, SENTINEL, 27.25, 31.0],
    "smoker": ["no", "yes", "no", "yes"],
}).to_csv(index=False)

GOOD = ("```python\nfig, ax = plt.subplots()\ndf['bmi'].plot.hist(ax=ax)\n"
        "summary = pd.DataFrame({'mean BMI': [round(df['bmi'].mean(), 2)]})\n```\n"
        "The histogram shows how BMI values are spread.")
MEAN_BMI = round(pd.read_csv(io.StringIO(CSV))["bmi"].mean(), 2)  # computed locally, from the sentinel row too
GOOD_NO_TEXT = "```python\nfig, ax = plt.subplots()\ndf['age'].plot.hist(ax=ax)\n```"
REFUSAL = "I cannot tell you who the patients are: the data are synthetic."
KEY_ERROR = "```python\nfig, ax = plt.subplots()\ndf['glucose'].plot.hist(ax=ax)\n```\nGlucose spread."
FORBIDDEN = "```python\nimport seaborn as sns\nfig, ax = plt.subplots()\nsns.histplot(df['bmi'], ax=ax)\n```\nBMI."
NO_FIGURE = "```python\nprint(df['bmi'].mean())\n```\nThe mean BMI."
ENDLESS = "```python\nwhile True:\n    pass\n```\nNever ends."
LONG_ERROR = "```python\nraise ValueError('x' * 1000)\n```\nFails with a long message."


def run(client, question="Show the distribution of BMI.", history=()):
    return ask(question, history, schema=SCHEMA, synthetic_csv=CSV, client=client, settings=SETTINGS)


def failure(client, **kwargs):
    with pytest.raises(AnalystError) as error:
        run(client, **kwargs)
    return error.value


# --- a question, a chart -----------------------------------------------------------------------------------------


def test_a_chart_answer():
    result = run(FakeClient(GOOD))
    assert isinstance(result, AskResult)
    assert result.kind == "chart"
    assert result.png.startswith(b"\x89PNG")
    assert result.text == "The histogram shows how BMI values are spread."
    assert result.summary == {"columns": ["mean BMI"], "rows": [[MEAN_BMI]], "truncated": False}
    assert "plot.hist" in result.code
    assert result.corrected is False
    assert result.exchange == Exchange("Show the distribution of BMI.", GOOD)


def test_a_refusal_is_a_text_answer():
    result = run(FakeClient(REFUSAL), question="Who are the patients?")
    assert (result.kind, result.text, result.png, result.code, result.summary) == ("text", REFUSAL, None, None, None)
    assert result.exchange == Exchange("Who are the patients?", REFUSAL)


def test_the_request_is_portable():
    """Only model, messages and the token limit: other parameters are refused by some models."""
    client = FakeClient(GOOD)
    run(client)
    (request,) = client.requests
    assert set(request) == {"model", "messages", "max_completion_tokens"}
    assert request["model"] == "test-model"
    assert request["max_completion_tokens"] == ANSWER_TOKENS


def test_the_first_message_is_the_system_prompt_and_the_last_the_question():
    client = FakeClient(GOOD)
    run(client, question="  Show the distribution of BMI.  ")
    messages = client.requests[0]["messages"]
    assert messages == [
        {"role": "system", "content": system_prompt(SCHEMA)},
        {"role": "user", "content": "Show the distribution of BMI."},
    ]


def test_each_call_is_traced():
    client = FakeClient(reply(GOOD, prompt_tokens=700, completion_tokens=300))
    (call,) = run(client).calls
    assert (call.content, call.finish_reason, call.prompt_tokens, call.completion_tokens) == (GOOD, "stop", 700, 300)
    assert call.seconds >= 0


def test_missing_usage_is_traced_as_unknown():
    """Usage is optional in the API: a provider without it must not break the answer."""
    response = reply(GOOD)
    response.usage = None
    (call,) = run(FakeClient(response)).calls
    assert (call.prompt_tokens, call.completion_tokens) == (None, None)


# --- history -----------------------------------------------------------------------------------------------------


def exchanges(count):
    return [Exchange(f"question {n}", f"answer {n}") for n in range(1, count + 1)]


def test_only_the_last_turns_of_the_history_are_sent():
    client = FakeClient(GOOD)
    run(client, question="And for age?", history=exchanges(MAX_HISTORY_TURNS + 2))
    messages = client.requests[0]["messages"]
    first = 3  # questions 1 and 2 are left out
    expected = []
    for n in range(first, MAX_HISTORY_TURNS + 3):
        expected += [{"role": "user", "content": f"question {n}"}, {"role": "assistant", "content": f"answer {n}"}]
    assert messages[1:-1] == expected
    assert messages[-1] == {"role": "user", "content": "And for age?"}


def test_a_short_history_is_sent_whole():
    client = FakeClient(GOOD)
    run(client, history=exchanges(1))
    assert [m["content"] for m in client.requests[0]["messages"][1:]] == [
        "question 1", "answer 1", "Show the distribution of BMI."]


def test_the_history_is_not_changed():
    history = exchanges(2)
    run(FakeClient(GOOD), history=history)
    assert history == exchanges(2)


def test_no_row_and_no_summary_number_reaches_the_provider():
    """Privacy: the provider receives the schema, the questions and its own answers, never the data or results."""
    client = FakeClient(GOOD, GOOD)
    first = run(client)
    run(client, question="And for age?", history=[first.exchange])
    sent = repr(client.requests)
    assert str(SENTINEL) not in sent
    assert first.summary["rows"] == [[MEAN_BMI]] and str(MEAN_BMI) not in sent  # the table stays here
    assert "iVBOR" not in sent  # no chart, in base64


# --- one fix attempt ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("broken", "error_start"),
    [
        (KEY_ERROR, "KeyError: 'glucose'"),
        (FORBIDDEN, "import of 'seaborn' is not allowed"),
        (NO_FIGURE, "no chart was drawn"),
    ],
    ids=["runtime", "unsafe", "no-chart"],
)
def test_a_broken_answer_is_fixed_once(broken, error_start):
    client = FakeClient(broken, GOOD)
    result = run(client)
    assert result.kind == "chart" and result.corrected is True
    assert len(client.requests) == 2
    second = client.requests[1]["messages"]
    assert second[:-2] == client.requests[0]["messages"]
    assert second[-2] == {"role": "assistant", "content": broken}
    assert second[-1]["role"] == "user"
    assert second[-1]["content"].startswith(f"Your code could not be run. Error: {error_start}")
    assert result.exchange == Exchange("Show the distribution of BMI.", GOOD)  # the fixed version
    assert [call.content for call in result.calls] == [broken, GOOD]


def test_a_fixed_answer_may_come_without_explanation():
    """Seen once during development: the chart is kept, no sentence is invented."""
    result = run(FakeClient(KEY_ERROR, GOOD_NO_TEXT))
    assert (result.kind, result.text, result.corrected) == ("chart", "", True)


def test_a_fix_answered_in_text_is_a_text_answer():
    result = run(FakeClient(KEY_ERROR, REFUSAL))
    assert (result.kind, result.text, result.corrected) == ("text", REFUSAL, True)
    assert result.exchange.answer == REFUSAL


@pytest.mark.parametrize(
    ("second", "code"),
    [(KEY_ERROR, RUNTIME_ERROR), (FORBIDDEN, UNSAFE_CODE), (NO_FIGURE, NO_CHART)],
    ids=["runtime", "unsafe", "no-chart"],
)
def test_broken_twice_is_the_error_of_the_second_attempt(second, code):
    client = FakeClient(KEY_ERROR, second)
    error = failure(client)
    assert error.code == code
    assert len(client.requests) == 2  # never a third call
    assert [call.content for call in error.calls] == [KEY_ERROR, second]


def test_the_fix_request_carries_no_stack_trace_and_is_cut():
    client = FakeClient(LONG_ERROR, GOOD)
    run(client)
    request = client.requests[1]["messages"][-1]["content"]
    assert "Traceback" not in request and 'File "' not in request
    error = request.removeprefix("Your code could not be run. Error: ").split("\nFix the code")[0]
    assert error.startswith("ValueError: xxx") and len(error) == MAX_ERROR_CHARS


def test_an_endless_answer_is_not_retried(monkeypatch):
    """The user has already waited for the timeout: a second attempt could double the wait."""
    monkeypatch.setattr(analyst, "RUN_TIMEOUT_SECONDS", 2)
    client = FakeClient(ENDLESS, GOOD)
    assert failure(client).code == TIMEOUT
    assert len(client.requests) == 1


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_an_incomplete_answer_is_not_retried(finish_reason):
    client = FakeClient(reply(GOOD, finish_reason=finish_reason), GOOD)
    error = failure(client)
    assert error.code == INCOMPLETE_ANSWER
    assert len(client.requests) == 1
    assert [call.finish_reason for call in error.calls] == [finish_reason]


def test_an_incomplete_fix_is_an_incomplete_answer():
    client = FakeClient(KEY_ERROR, reply("", finish_reason="length"))
    assert failure(client).code == INCOMPLETE_ANSWER


def test_a_response_without_choices_is_incomplete():
    assert failure(FakeClient(SimpleNamespace(choices=[], usage=None))).code == INCOMPLETE_ANSWER


def test_a_refusal_is_not_retried():
    client = FakeClient(REFUSAL, GOOD)
    run(client)
    assert len(client.requests) == 1


# --- provider errors ---------------------------------------------------------------------------------------------

URL = "https://provider.test/v1/chat/completions"
REQUEST = httpx.Request("POST", URL)


def status_error(kind, status, code=None, message="provider message"):
    body = {"message": message, "code": code}
    response = httpx.Response(status, request=REQUEST, json={"error": body})
    return kind(f"Error code: {status} - {message}", response=response, body=body)


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (status_error(openai.AuthenticationError, 401), PROVIDER_AUTH),
        (status_error(openai.PermissionDeniedError, 403), PROVIDER_AUTH),
        (status_error(openai.NotFoundError, 404), PROVIDER_MODEL),
        (status_error(openai.RateLimitError, 429, code="insufficient_quota"), PROVIDER_QUOTA),
        (status_error(openai.RateLimitError, 429, code="rate_limit_exceeded"), PROVIDER_BUSY),
        (status_error(openai.RateLimitError, 429), PROVIDER_BUSY),
        (openai.APIConnectionError(request=REQUEST), PROVIDER_UNREACHABLE),
        (openai.APITimeoutError(request=REQUEST), PROVIDER_UNREACHABLE),
        (status_error(openai.BadRequestError, 400), PROVIDER_ERROR),
        (status_error(openai.InternalServerError, 500), PROVIDER_ERROR),
        (status_error(openai.APIStatusError, 418), PROVIDER_ERROR),
    ],
    ids=["401", "403", "404", "quota", "rate-limit", "429-no-code", "connection", "timeout", "400", "500", "418"],
)
def test_provider_errors_become_plain_language(raised, code):
    client = FakeClient(raised, GOOD)
    error = failure(client)
    assert error.code == code
    assert error.message and "Error code" not in error.message and "Traceback" not in error.message
    assert error.detail.startswith(type(raised).__name__)
    assert len(error.detail) <= MAX_ERROR_CHARS
    assert len(client.requests) == 1  # never retried


def test_the_provider_messages_name_what_to_check():
    assert "LLM_API_KEY" in failure(FakeClient(status_error(openai.AuthenticationError, 401))).message
    assert "LLM_MODEL" in failure(FakeClient(status_error(openai.NotFoundError, 404))).message
    assert "minute" in failure(FakeClient(status_error(openai.RateLimitError, 429))).message


def test_a_provider_error_during_the_fix_is_a_provider_error():
    client = FakeClient(KEY_ERROR, status_error(openai.RateLimitError, 429))
    error = failure(client)
    assert error.code == PROVIDER_BUSY
    assert [call.content for call in error.calls] == [KEY_ERROR]


def test_the_key_never_appears_in_an_error():
    raised = status_error(openai.AuthenticationError, 401, message=f"Incorrect API key provided: {KEY}")
    error = failure(FakeClient(raised))
    assert KEY not in error.message and KEY not in error.detail


# --- the question ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t", "x" * (MAX_QUESTION_CHARS + 1)],
                         ids=["empty", "spaces", "blank", "too-long"])
def test_an_invalid_question_never_reaches_the_provider(question):
    client = FakeClient()
    error = failure(client, question=question)
    assert error.code == INVALID_QUESTION and error.message
    assert client.requests == []


def test_a_question_of_the_maximum_length_is_accepted():
    client = FakeClient(GOOD)
    run(client, question="x" * MAX_QUESTION_CHARS + "   ")
    assert client.requests[0]["messages"][-1]["content"] == "x" * MAX_QUESTION_CHARS


# --- the client ----------------------------------------------------------------------------------------------------


def test_make_client_never_retries_and_has_a_timeout():
    client = make_client(LlmSettings(api_key=KEY, model="m", base_url="https://provider.test/v1"))
    assert isinstance(client, openai.OpenAI)
    assert client.api_key == KEY
    assert str(client.base_url).rstrip("/") == "https://provider.test/v1"
    assert client.max_retries == 0
    assert client.timeout == REQUEST_TIMEOUT_SECONDS


def test_make_client_without_base_url_uses_the_default_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = make_client(SETTINGS)
    assert str(client.base_url).startswith("https://api.openai.com/")


TABLE = {"columns": ["n"], "rows": [[1]], "truncated": False}


def test_the_sandbox_result_is_used_as_it_is(monkeypatch):
    """ask hands the code to run_code with the synthetic CSV and keeps what it returns."""
    seen = {}

    def fake_run_code(code, csv_text, *, timeout):
        seen.update(code=code, csv=csv_text, timeout=timeout)
        return ChartResult(png=b"\x89PNG-fake", summary=TABLE, figures=1)

    monkeypatch.setattr(analyst, "run_code", fake_run_code)
    result = run(FakeClient(GOOD))
    assert seen["csv"] == CSV and "plot.hist" in seen["code"]
    assert (result.png, result.summary) == (b"\x89PNG-fake", TABLE)
