"""The Ask step: questions about the synthetic table, behind the privacy gate, one at a time.

The provider receives only the schema of the synthetic table and the questions (``sds.llm_analyst``); the code it
writes runs in the sandbox on the synthetic CSV. The conversation lives in the session, in memory, and is dropped
with the synthetic table it talks about.

A question can take minutes, so it is answered **outside** the session lock: the rest of the app keeps answering.
Under a chart, a **note** for every column it uses that the Validate step found less faithful (the fidelity report's
``less_faithful``): a local reading of the code, the provider learns nothing about it.

Like an orphan training job, an answer whose conversation was replaced meanwhile (new conversation, new training,
reset) reaches who asked but is not kept: it is added to the list it started from, which nobody reads any more.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

from api.api_config import (
    CHART_NOTE,
    QUESTION_RUNNING_MESSAGE,
    STATUS_BAD_GATEWAY,
    STATUS_CONFLICT,
    STATUS_UNAVAILABLE,
    STATUS_UNPROCESSABLE,
)
from api.errors import ApiError
from api.results import require_gate
from api.session import Session, Stage
from sds.llm_analyst.analyst import AskResult, CallTrace, ask, make_client
from sds.llm_analyst.errors import (
    INVALID_QUESTION,
    LLM_NOT_CONFIGURED,
    PROVIDER_BUSY,
    PROVIDER_UNREACHABLE,
    AnalystError,
)
from sds.llm_analyst.llm_config import MAX_HISTORY_TURNS
from sds.llm_analyst.prompt import system_prompt
from sds.llm_analyst.sandbox import columns_used
from sds.llm_analyst.settings import LlmNotConfiguredError, LlmSettings, load_settings
from sds.validation.fidelity import ColumnFidelity, difference_phrase

# HTTP status of the analyst's errors; any other code (the answer could not be used) is STATUS_BAD_GATEWAY.
STATUS_OF_CODE = {
    INVALID_QUESTION: STATUS_UNPROCESSABLE,
    LLM_NOT_CONFIGURED: STATUS_CONFLICT,
    PROVIDER_BUSY: STATUS_UNAVAILABLE,
    PROVIDER_UNREACHABLE: STATUS_UNAVAILABLE,
}


def status_of(code: str) -> int:
    """HTTP status of an ``AnalystError`` code."""
    return STATUS_OF_CODE.get(code, STATUS_BAD_GATEWAY)


@dataclass(frozen=True)
class LlmAccess:
    """How the API reaches the provider, read once when the app is created.

    Attributes:
        settings: Key, model and address; ``None`` when not configured.
        client: The provider client; ``None`` when not configured (tests put a fake one here).
        problem: Why it is not configured (names the missing variables), else ``None``.
    """

    settings: LlmSettings | None
    client: object | None
    problem: AnalystError | None

    @classmethod
    def from_environment(cls) -> LlmAccess:
        """Read the ``LLM_*`` settings (environment, then ``.env``) and build the client."""
        try:
            settings = load_settings()
        except LlmNotConfiguredError as error:
            return cls(settings=None, client=None, problem=error)
        return cls(settings=settings, client=make_client(settings), problem=None)

    @property
    def configured(self) -> bool:
        """Whether questions can be asked."""
        return self.settings is not None


@dataclass(frozen=True)
class Turn:
    """One answered question of the conversation, with the reliability notes of its chart."""

    question: str
    result: AskResult
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LastQuestion:
    """What happened to the latest question, for the developer panel.

    Attributes:
        question: As typed.
        outcome: ``"chart"``, ``"text"`` or the code of the error.
        corrected: Whether the fix attempt was used.
        error: The ``AnalystError``, else ``None``.
        calls: The calls made to the provider.
        seconds: Duration of the whole question (calls and runs).
    """

    question: str
    outcome: str
    corrected: bool
    error: AnalystError | None
    calls: tuple[CallTrace, ...]
    seconds: float


def chart_notes(result: AskResult, less_faithful: list[ColumnFidelity]) -> tuple[str, ...]:
    """A note for every less faithful column the chart's code names (none for an answer without code)."""
    if result.code is None:
        return ()
    by_name = {c.name: c for c in less_faithful}
    return tuple(CHART_NOTE.format(name=name, difference=difference_phrase(by_name[name]))
                 for name in columns_used(result.code, list(by_name)))


def turn_view(turn: Turn) -> dict:
    """A turn as the UI shows it: chart in base64, explanation, summary table, code, reliability notes."""
    result = turn.result
    return {
        "question": turn.question,
        "kind": result.kind,
        "text": result.text,
        "image_png": base64.b64encode(result.png).decode("ascii") if result.png is not None else None,
        "summary": result.summary,
        "code": result.code,
        "corrected": result.corrected,
        "notes": list(turn.notes),
    }


def ask_question(session: Session, question: str, llm: LlmAccess) -> dict:
    """Answer a question with a chart drawn on the synthetic table, and keep it in the conversation.

    Args:
        session: The current session.
        question: The question as typed.
        llm: How to reach the provider.

    Returns:
        The turn, as ``turn_view`` shows it.

    Raises:
        ApiError: 409 with the code of the first missing stage, 403 ``privacy_gate``, 409 ``question_running``.
        AnalystError: ``llm_not_configured`` and every error of ``sds.llm_analyst.analyst.ask``.
    """
    with session.lock:
        require_gate(session)
        if not llm.configured:
            raise llm.problem
        if not session.ask_lock.acquire(blocking=False):
            raise ApiError(STATUS_CONFLICT, "question_running", QUESTION_RUNNING_MESSAGE)
        conversation = session.conversation
        history = [turn.result.exchange for turn in conversation]
        schema = session.contract.llm_schema()
        synthetic_csv = session.synthetic.to_csv(index=False)   # commas: the sandbox reads it with the defaults
        less_faithful = list(session.fidelity.less_faithful)
    start = time.perf_counter()
    try:
        result = ask(question, history, schema=schema, synthetic_csv=synthetic_csv, client=llm.client,
                     settings=llm.settings)
    except AnalystError as error:
        with session.lock:
            session.last_question = LastQuestion(question, error.code, len(error.calls) > 1, error, error.calls,
                                                 round(time.perf_counter() - start, 2))
        raise
    finally:
        session.ask_lock.release()
    turn = Turn(result.exchange.question, result, chart_notes(result, less_faithful))
    with session.lock:
        session.last_question = LastQuestion(question, result.kind, result.corrected, None, result.calls,
                                             round(time.perf_counter() - start, 2))
        conversation.append(turn)   # the list of the start: if it was replaced meanwhile, the turn goes with it
    return turn_view(turn)


def conversation_view(session: Session) -> dict:
    """The turns of the current conversation (the UI rebuilds the chat from it after a refresh)."""
    with session.lock:
        return {"turns": [turn_view(turn) for turn in session.conversation]}


def new_conversation(session: Session) -> dict:
    """Forget the questions, keep the model and the synthetic table."""
    with session.lock:
        session.conversation = []
        return conversation_view(session)


def call_view(call: CallTrace) -> dict:
    """One call to the provider, as the developer panel shows it."""
    return {"content": call.content, "finish_reason": call.finish_reason, "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens, "seconds": call.seconds}


def llm_view(session: Session, llm: LlmAccess) -> dict:
    """The developer view of the Ask step: provider, full system prompt and the latest question.

    Never the key: the provider's messages in ``detail`` already have it replaced.
    """
    settings = llm.settings
    with session.lock:
        prompt = system_prompt(session.contract.llm_schema()) if session.stage >= Stage.PREPARED else None
        last = session.last_question
        last_view = None
        if last is not None:
            error = None if last.error is None else {"code": last.error.code, "message": last.error.message,
                                                     "detail": last.error.detail}
            last_view = {"question": last.question, "outcome": last.outcome, "corrected": last.corrected,
                         "error": error, "calls": [call_view(call) for call in last.calls], "seconds": last.seconds}
        return {
            "configured": llm.configured,
            "model": settings.model if settings else None,
            "base_url": settings.base_url if settings else None,
            "system_prompt": prompt,
            "max_history_turns": MAX_HISTORY_TURNS,
            "last": last_view,
        }
