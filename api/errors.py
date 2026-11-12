"""Errors the API returns on purpose, with a stable code and a plain-language message."""

from __future__ import annotations


class ApiError(Exception):
    """An error the UI can show as it is.

    Attributes:
        status: HTTP status of the response.
        code: Stable machine-readable identifier (tests and the UI rely on it).
        message: Plain-language explanation for a non-technical user.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
