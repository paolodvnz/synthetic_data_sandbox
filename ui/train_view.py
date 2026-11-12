"""The Train step without Streamlit: bar, early stopping, time, curves of the developer panel (pure functions)."""

from __future__ import annotations

import pandas as pd

PHASE_TEXT = {"generating": "Creating the synthetic data…", "validating": "Checking privacy and quality…"}
LOSS_COMPONENTS = ("numeric", "categorical", "kl")
SECONDS_PER_MINUTE = 60


def is_running(job: dict | None) -> bool:
    """Whether ``job`` (a ``/status`` view, or ``None``) is still running."""
    return job is not None and job["state"] == "running"


def progress(job: dict) -> tuple[float, str]:
    """Fraction of the bar and its text: the steps while training, then full with the phase."""
    if job["phase"] in PHASE_TEXT:
        return 1.0, PHASE_TEXT[job["phase"]]
    fraction = min(1.0, job["step"] / job["max_steps"])
    return fraction, f"Learning from your data… {round(fraction * 100)}%"


def ended_early(job: dict) -> bool:
    """A finished job short of its steps: early stopping (``train_vae`` always evaluates the last step)."""
    return job["state"] == "done" and job["step"] < job["max_steps"]


def elapsed_text(seconds: float) -> str:
    """``"43 s"`` or ``"1 min 42 s"``."""
    minutes, rest = divmod(round(seconds), SECONDS_PER_MINUTE)
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def history_frames(history: list[dict]) -> dict[str, pd.DataFrame]:
    """The ``/dev/training`` history as three tables indexed by step, ready for ``st.line_chart``.

    Returns:
        ``total`` (train and validation loss), ``components`` (validation loss per part), ``schedule`` (KL
        weight and active latent dimensions).
    """
    steps = [e["step"] for e in history]
    total = pd.DataFrame({"train": [e["train"]["total"] for e in history],
                          "validation": [e["validation"]["total"] for e in history]}, index=steps)
    components = pd.DataFrame({part: [e["validation"][part] for e in history] for part in LOSS_COMPONENTS},
                              index=steps)
    schedule = pd.DataFrame({"kl_weight": [e["kl_weight"] for e in history],
                             "active_dims": [e["active_dims"] for e in history]}, index=steps)
    for frame in (total, components, schedule):
        frame.index.name = "step"
    return {"total": total, "components": components, "schedule": schedule}
