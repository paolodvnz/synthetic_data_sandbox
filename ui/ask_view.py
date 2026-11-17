"""The Ask step without Streamlit: chart bytes, summary table, cache key of the conversation, safe text (pure functions)."""

from __future__ import annotations

import base64
import re

import pandas as pd

MARKDOWN_SPECIALS = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<$])")


def image_bytes(turn: dict) -> bytes | None:
    """The chart of a turn as PNG bytes, ``None`` for an answer in words."""
    encoded = turn["image_png"]
    return base64.b64decode(encoded) if encoded is not None else None


def summary_frame(turn: dict) -> pd.DataFrame | None:
    """The numbers behind the chart of a turn as a table, ``None`` when there are none."""
    summary = turn["summary"]
    return pd.DataFrame(summary["rows"], columns=summary["columns"]) if summary is not None else None


def conversation_key(view: dict) -> str | None:
    """What the cached conversation belongs to: the job of the session (a new training drops the conversation)."""
    return view["job"]["job_id"] if view["job"] is not None else None


def escape_markdown(text: str) -> str:
    """Text typed by the user, shown as typed (Streamlit would read it as markdown)."""
    return MARKDOWN_SPECIALS.sub(r"\\\1", text)


def safe_text(text: str) -> str:
    """The model's explanation: markdown kept (bold, lists), but a dollar sign never starts a formula."""
    return text.replace("$", r"\$")
