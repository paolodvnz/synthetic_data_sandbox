"""Errors of the question step, each with a stable code and a plain-language message.

The API turns every ``AnalystError`` into ``{"code", "message"}`` with one rule, as it does for the Data Engine.
"""

from __future__ import annotations

LLM_NOT_CONFIGURED = "llm_not_configured"
INCOMPLETE_ANSWER = "incomplete_answer"
UNSAFE_CODE = "unsafe_code"
RUNTIME_ERROR = "runtime_error"
TIMEOUT = "timeout"
NO_CHART = "no_chart"
INVALID_QUESTION = "invalid_question"
PROVIDER_AUTH = "provider_auth"
PROVIDER_MODEL = "provider_model"
PROVIDER_QUOTA = "provider_quota"
PROVIDER_BUSY = "provider_busy"
PROVIDER_UNREACHABLE = "provider_unreachable"
PROVIDER_ERROR = "provider_error"


class AnalystError(Exception):
    """An error of the question step that the UI can show as it is.

    Attributes:
        code: Stable machine-readable identifier (tests and the API rely on it).
        message: Plain-language explanation for a non-technical user, never a stack trace.
        detail: Technical reason (e.g. ``"KeyError: 'glucose'"``) for the fix request and the developer panel;
            ``None`` when there is nothing to add.
        calls: The calls to the provider made before the error, set by the analyst for the developer panel.
    """

    calls: tuple = ()

    def __init__(self, code: str, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
