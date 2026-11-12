"""Which steps of the guided path can be opened, decided from the API's ``/session`` view (pure functions)."""

from __future__ import annotations

from enum import Enum


class Step(str, Enum):
    """The four steps, in order."""

    UPLOAD = "upload"
    TRAIN = "train"
    VALIDATE = "validate"
    ASK = "ask"


STEP_LABELS: dict[Step, str] = {
    Step.UPLOAD: "1 · Upload",
    Step.TRAIN: "2 · Train",
    Step.VALIDATE: "3 · Validate",
    Step.ASK: "4 · Ask (coming soon)",
}

# Stages of the API from which each step opens (Ask opens once the language model is added: never in this version).
OPENS_FROM: dict[Step, frozenset[str]] = {
    Step.UPLOAD: frozenset({"empty", "inspected", "prepared", "trained", "generated", "validated"}),
    Step.TRAIN: frozenset({"prepared", "trained", "generated", "validated"}),
    Step.VALIDATE: frozenset({"validated"}),
    Step.ASK: frozenset(),
}


def reachable_steps(view: dict) -> set[Step]:
    """The steps that can be opened in the session ``view`` (``{"stage", "job"}``)."""
    return {step for step, stages in OPENS_FROM.items() if view["stage"] in stages}


def default_step(view: dict) -> Step:
    """The step to open when the page is (re)loaded: where the user was most likely working.

    A job (running or ended) brings back to Train, where its bar or its outcome is shown; a validated session
    to Validate; otherwise Upload.
    """
    if view["stage"] == "validated":
        return Step.VALIDATE
    if view["job"] is not None and Step.TRAIN in reachable_steps(view):
        return Step.TRAIN
    return Step.UPLOAD
