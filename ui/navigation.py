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
    Step.ASK: "4 · Ask",
}

# Stages of the API from which each step opens; Ask also needs the privacy gate passed (as the API enforces).
OPENS_FROM: dict[Step, frozenset[str]] = {
    Step.UPLOAD: frozenset({"empty", "inspected", "prepared", "trained", "generated", "validated"}),
    Step.TRAIN: frozenset({"prepared", "trained", "generated", "validated"}),
    Step.VALIDATE: frozenset({"generated", "validated"}),     # "generated": the comparison, privacy being checked
    Step.ASK: frozenset({"validated"}),
}
ASK_LOCKED_HELP = "Opens when the privacy check passes (step 3, Validate)."


def checking_privacy(job: dict | None) -> bool:
    """Whether ``job`` (a ``/status`` view, or ``None``) delivered the synthetic table and is checking its privacy."""
    return job is not None and job["state"] == "running" and job["phase"] == "checking_privacy"


def reachable_steps(view: dict) -> set[Step]:
    """The steps that can be opened in the session ``view`` (``{"stage", "job", "gate_passed"}``)."""
    steps = {step for step, stages in OPENS_FROM.items() if view["stage"] in stages}
    if view["gate_passed"] is not True:
        steps.discard(Step.ASK)        # a failed gate would lead to a dead end: Validate explains why
    return steps


def default_step(view: dict) -> Step:
    """The step to open when the page is (re)loaded: where the user was most likely working.

    A synthetic table (privacy checked or being checked) brings to Validate; a job (running or ended) back to Train,
    where its bar or its outcome is shown; otherwise Upload.
    """
    if view["stage"] in {"generated", "validated"}:
        return Step.VALIDATE
    if view["job"] is not None and Step.TRAIN in reachable_steps(view):
        return Step.TRAIN
    return Step.UPLOAD
