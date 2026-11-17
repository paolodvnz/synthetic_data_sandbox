"""The analyst: one question of the user in, one chart (or a refusal in words) out.

What goes to the provider: the system prompt (built from the schema), the last ``MAX_HISTORY_TURNS`` questions with
the model's own answers, and the new question. Never a row, a number of the summary table or a chart. The code of the answer runs
locally, in the sandbox, on the synthetic data only.

The analyst keeps no state: the caller (the API) keeps the conversation and passes it back with every question.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import openai

from sds.llm_analyst.errors import (
    INVALID_QUESTION,
    NO_CHART,
    PROVIDER_AUTH,
    PROVIDER_BUSY,
    PROVIDER_ERROR,
    PROVIDER_MODEL,
    PROVIDER_QUOTA,
    PROVIDER_UNREACHABLE,
    RUNTIME_ERROR,
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
from sds.llm_analyst.prompt import fix_request, parse_answer, system_prompt
from sds.llm_analyst.sandbox import run_code
from sds.llm_analyst.sandbox_config import RUN_TIMEOUT_SECONDS
from sds.llm_analyst.settings import LlmSettings

CHART, TEXT = "chart", "text"

# Failures the model can fix by writing other code. Not a timeout: the user has already waited for it.
FIXABLE = frozenset({UNSAFE_CODE, RUNTIME_ERROR, NO_CHART})
QUOTA_CODE = "insufficient_quota"  # OpenAI's code for a 429 that means "no credit left", not "too fast"

EMPTY_QUESTION_MESSAGE = "Please write a question."
LONG_QUESTION_MESSAGE = f"The question is too long: please keep it under {MAX_QUESTION_CHARS} characters."
AUTH_MESSAGE = "The language model provider refused the key: check LLM_API_KEY in .env and restart the app."
MODEL_MESSAGE = "The language model was not found: check LLM_MODEL and LLM_BASE_URL in .env and restart the app."
QUOTA_MESSAGE = "The account of the language model provider has no credit left: top it up, or use another key."
BUSY_MESSAGE = "Too many questions in a short time: please wait a minute and ask again."
UNREACHABLE_MESSAGE = "The language model could not be reached: check the internet connection and ask again."
PROVIDER_ERROR_MESSAGE = "The language model provider had a problem: please ask again in a moment."


@dataclass(frozen=True)
class Exchange:
    """One question and the model's answer, as it goes back to the model in the history.

    Attributes:
        question: The question, without surrounding spaces.
        answer: The raw answer of the model (code block and explanation, or the refusal).
    """

    question: str
    answer: str


@dataclass(frozen=True)
class CallTrace:
    """One call to the provider, for the developer panel.

    Attributes:
        content: The raw answer (``""`` when there was none).
        finish_reason: Why the model stopped (``"stop"`` for a complete answer).
        prompt_tokens: Tokens of the request, ``None`` if the provider did not say.
        completion_tokens: Tokens of the answer (reasoning included), ``None`` if the provider did not say.
        seconds: Duration of the call.
    """

    content: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float


@dataclass(frozen=True)
class AskResult:
    """The answer to one question.

    Attributes:
        kind: ``"chart"``, or ``"text"`` for an answer in words only (a refusal).
        text: The explanation; may be empty after a fix (the model does not always repeat it).
        png: The chart as PNG bytes, ``None`` for a text answer.
        summary: The numbers behind the chart, computed locally (see ``ChartResult``); ``None`` for a text answer
            or when the code left none. Never sent to the provider.
        code: The code that drew the chart, ``None`` for a text answer.
        corrected: Whether the answer needed the fix attempt.
        exchange: What the caller adds to the history.
        calls: The calls made to the provider (one, or two with the fix attempt).
    """

    kind: str
    text: str
    png: bytes | None
    summary: dict | None
    code: str | None
    corrected: bool
    exchange: Exchange
    calls: tuple[CallTrace, ...]


def make_client(settings: LlmSettings) -> openai.OpenAI:
    """The provider client: no automatic retry, so the wait is predictable and a 429 is reported at once.

    Args:
        settings: Key, model and address of the provider.

    Returns:
        A client for any provider compatible with the OpenAI API.
    """
    return openai.OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                         timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)


def checked_question(question: str) -> str:
    """The question without surrounding spaces; refused when empty or too long."""
    question = question.strip()
    if not question:
        raise AnalystError(INVALID_QUESTION, EMPTY_QUESTION_MESSAGE)
    if len(question) > MAX_QUESTION_CHARS:
        raise AnalystError(INVALID_QUESTION, LONG_QUESTION_MESSAGE)
    return question


def conversation(question: str, history: Sequence[Exchange], schema: list[dict]) -> list[dict]:
    """The messages of a question: system prompt, the last turns of the history, the question."""
    messages = [{"role": "system", "content": system_prompt(schema)}]
    for exchange in history[-MAX_HISTORY_TURNS:]:
        messages += [{"role": "user", "content": exchange.question},
                     {"role": "assistant", "content": exchange.answer}]
    return messages + [{"role": "user", "content": question}]


def provider_error(error: openai.OpenAIError, settings: LlmSettings) -> AnalystError:
    """A provider failure in plain language, with the technical reason (never the key) in ``detail``."""
    if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
        code, message = PROVIDER_AUTH, AUTH_MESSAGE
    elif isinstance(error, openai.NotFoundError):
        code, message = PROVIDER_MODEL, MODEL_MESSAGE
    elif isinstance(error, openai.RateLimitError) and error.code == QUOTA_CODE:
        code, message = PROVIDER_QUOTA, QUOTA_MESSAGE
    elif isinstance(error, openai.RateLimitError):
        code, message = PROVIDER_BUSY, BUSY_MESSAGE
    elif isinstance(error, openai.APIConnectionError):  # a timeout too
        code, message = PROVIDER_UNREACHABLE, UNREACHABLE_MESSAGE
    else:
        code, message = PROVIDER_ERROR, PROVIDER_ERROR_MESSAGE
    detail = f"{type(error).__name__}: {error}".replace(settings.api_key, "***")[:MAX_ERROR_CHARS]
    return AnalystError(code, message, detail)


def call(client, settings: LlmSettings, messages: list[dict]) -> CallTrace:
    """One call to the provider, with the portable parameters only."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(model=settings.model, messages=messages,
                                                  max_completion_tokens=ANSWER_TOKENS)
    except openai.OpenAIError as error:
        raise provider_error(error, settings) from None
    seconds = time.perf_counter() - start
    choice = response.choices[0] if response.choices else None
    usage = response.usage
    return CallTrace(
        content=(choice.message.content or "") if choice else "",
        finish_reason=choice.finish_reason if choice else None,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        seconds=round(seconds, 2),
    )


def finish(question: str, calls: list[CallTrace], synthetic_csv: str, corrected: bool) -> AskResult:
    """Read the last answer and, if it has code, draw its chart.

    Raises:
        AnalystError: ``incomplete_answer``, or the sandbox error of the code.
    """
    trace = calls[-1]
    answer = parse_answer(trace.content, trace.finish_reason)
    exchange = Exchange(question, trace.content)
    if answer.code is None:
        return AskResult(kind=TEXT, text=answer.text, png=None, summary=None, code=None, corrected=corrected,
                         exchange=exchange, calls=tuple(calls))
    chart = run_code(answer.code, synthetic_csv, timeout=RUN_TIMEOUT_SECONDS)
    return AskResult(kind=CHART, text=answer.text, png=chart.png, summary=chart.summary, code=answer.code,
                     corrected=corrected, exchange=exchange, calls=tuple(calls))


def ask(question: str, history: Sequence[Exchange], *, schema: list[dict], synthetic_csv: str, client,
        settings: LlmSettings) -> AskResult:
    """Answer one question with a chart drawn on the synthetic data, fixing broken code once.

    Args:
        question: The question of the user.
        history: The previous exchanges of the conversation (only the last ``MAX_HISTORY_TURNS`` are sent).
        schema: The columns of the synthetic data, as given by ``SchemaContract.llm_schema()``.
        synthetic_csv: The synthetic data, comma-separated, with a header (never sent to the provider).
        client: An OpenAI-compatible client (see ``make_client``).
        settings: The model to ask, and the key to keep out of error messages.

    Returns:
        The chart and its explanation, or a text answer.

    Raises:
        AnalystError: ``invalid_question`` (no call made), ``incomplete_answer``, a ``provider_*`` code, or the
            sandbox error of the second attempt (``unsafe_code``, ``runtime_error``, ``no_chart``), or ``timeout``.
            ``calls`` holds the calls made before the error.
    """
    question = checked_question(question)
    messages = conversation(question, history, schema)
    calls: list[CallTrace] = []
    try:
        calls.append(call(client, settings, messages))
        try:
            return finish(question, calls, synthetic_csv, corrected=False)
        except AnalystError as error:
            if error.code not in FIXABLE:
                raise
            fix = [{"role": "assistant", "content": calls[-1].content},
                   {"role": "user", "content": fix_request(error.detail or error.message)}]
            calls.append(call(client, settings, messages + fix))
            return finish(question, calls, synthetic_csv, corrected=True)
    except AnalystError as error:
        error.calls = tuple(calls)
        raise
