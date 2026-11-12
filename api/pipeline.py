"""The training job: train the VAE, generate the synthetic table and validate it, in one background thread.

One job at a time. The results are written to the session only at the end, all together, and only if the
session still belongs to this job and nobody asked to stop: a stopped, failed or orphan job leaves the session
at ``Stage.PREPARED``, so the user can change the choices and train again from scratch.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import torch

from api.api_config import INTERNAL_ERROR_MESSAGE, STATUS_CONFLICT, STATUS_NOT_FOUND, STOP_TIMEOUT_SECONDS
from api.errors import ApiError
from api.session import Session, Stage
from sds.data_engine import DataEngineError
from sds.vae.hyperparameters_config import BATCH_SIZE, PATIENCE
from sds.vae.layout import MIXED_SIGMA_LOSS, OutputLayout, VaeError
from sds.vae.losses import warmup_steps
from sds.vae.model import Architecture, choose_architecture
from sds.vae.runtime import get_device
from sds.vae.sampling import generate
from sds.vae.training import Evaluation, evaluation_interval, split_validation, step_budget, train_vae
from sds.validation.fidelity import fidelity_report
from sds.validation.privacy import privacy_check

logger = logging.getLogger("sds.api")


class JobState(str, Enum):
    """Life of a job: ``running`` until it ends one of the three other ways."""

    RUNNING = "running"
    DONE = "done"
    STOPPED = "stopped"
    FAILED = "failed"


class Phase(str, Enum):
    """What a running job is doing."""

    TRAINING = "training"
    GENERATING = "generating"
    VALIDATING = "validating"


@dataclass(eq=False)
class Job:
    """One run of the pipeline.

    Attributes:
        job_id: Random identifier the UI polls with.
        max_steps: Gradient steps at most (the end of the progress bar).
        stop_event: Set by ``stop_training``; checked by ``train_vae`` at every step and between the phases.
        state, phase: Where the job is.
        latest: The last evaluation of the training (written by the training thread only).
        history: Every evaluation so far, in order (appended by the training thread; copy before reading).
        started, ended: ``time.monotonic()`` at the start and at the end (``None`` while running).
        error: ``{"code", "message"}`` of a failed job.
        thread: The thread running ``run_pipeline``.
    """

    job_id: str
    max_steps: int
    stop_event: threading.Event = field(default_factory=threading.Event)
    state: JobState = JobState.RUNNING
    phase: Phase = Phase.TRAINING
    latest: Evaluation | None = None
    history: list[Evaluation] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    error: dict | None = None
    thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        """True while the job runs."""
        return self.state is JobState.RUNNING


@dataclass(frozen=True)
class TrainingPlan:
    """Everything the training will use, with the sizes it comes from: shown by ``/dev/hyperparameters``.

    The job passes these values to ``train_vae`` explicitly, so the panel and the training cannot differ.
    """

    n_rows: int
    n_training: int
    n_validation: int
    width: int
    architecture: Architecture
    steps_per_epoch: int
    max_steps: int
    eval_every: int
    warmup_steps: int
    batch_size: int
    patience: int
    device: str

    def train_kwargs(self) -> dict:
        """Keyword arguments of ``train_vae`` (``batch_size`` nominal: ``train_vae`` caps it at the rows)."""
        return {"architecture": self.architecture, "max_steps": self.max_steps, "eval_every": self.eval_every,
                "batch_size": BATCH_SIZE, "patience": self.patience, "device": self.device}


def training_plan(n_rows: int, width: int, *, architecture: Architecture | None = None, max_steps: int | None = None,
                  eval_every: int | None = None, device: str | None = None) -> TrainingPlan:
    """The training of ``n_rows`` train rows of ``width`` values, by the engine's rules.

    The keyword arguments replace a rule (tests only: the app uses the rules).
    """
    n_training, n_validation = (len(part) for part in split_validation(n_rows))
    max_steps = max_steps or step_budget(n_rows).max_steps
    return TrainingPlan(
        n_rows=n_rows,
        n_training=n_training,
        n_validation=n_validation,
        width=width,
        architecture=architecture or choose_architecture(width),
        steps_per_epoch=-(-n_training // BATCH_SIZE),
        max_steps=max_steps,
        eval_every=eval_every or evaluation_interval(max_steps),
        warmup_steps=warmup_steps(max_steps),
        batch_size=min(BATCH_SIZE, n_training),
        patience=PATIENCE,
        device=device or str(get_device()),
    )


def record(job: Job, evaluation: Evaluation) -> None:
    """``on_progress`` of the job: keep the evaluation (no session lock: the stop waits without it)."""
    job.history.append(evaluation)
    job.latest = evaluation


def start_training(session: Session, **train_options) -> Job:
    """Start a new job on the prepared dataset, from scratch.

    Args:
        session: A session at ``Stage.PREPARED`` or later, with no job running.
        **train_options: Passed to ``training_plan`` (tests only: the app uses the engine's rules).

    Returns:
        The running job.

    Raises:
        ApiError: 409 ``training_running`` or the code of the first missing stage.
    """
    with session.lock:
        session.require_idle()
        session.require(Stage.PREPARED)
        session.clear_from(Stage.TRAINED)
        plan = training_plan(len(session.prepared.train), session.contract.width, **train_options)
        job = Job(job_id=uuid.uuid4().hex, max_steps=plan.max_steps)
        job.thread = threading.Thread(target=run_pipeline, name=f"sds-job-{job.job_id[:8]}", daemon=True,
                                      args=(session, job, session.prepared, session.encoder, session.contract,
                                            plan))
        session.job = job
        job.thread.start()
        return job


def run_pipeline(session: Session, job: Job, prepared, encoder, contract, plan: TrainingPlan) -> None:
    """Body of the job thread: train → generate → validate, then write the outcome to the session."""
    try:
        x_train = encoder.transform(prepared.train)
        trained = train_vae(x_train, OutputLayout.from_contract(contract), loss=MIXED_SIGMA_LOSS,
                            on_progress=lambda evaluation: record(job, evaluation), stop_event=job.stop_event,
                            **plan.train_kwargs())
        if job.stop_event.is_set():
            return finish(session, job, JobState.STOPPED)
        job.phase = Phase.GENERATING
        synthetic = generate(trained, contract, len(prepared.train))
        if job.stop_event.is_set():
            return finish(session, job, JobState.STOPPED)
        job.phase = Phase.VALIDATING
        fidelity = fidelity_report(synthetic, prepared.holdout, prepared.train, contract)
        if job.stop_event.is_set():
            return finish(session, job, JobState.STOPPED)
        privacy = privacy_check(encoder.transform(synthetic), x_train, encoder.transform(prepared.holdout), contract)
        finish(session, job, JobState.DONE, results=(trained, synthetic, fidelity, privacy))
    except (VaeError, DataEngineError) as error:
        finish(session, job, JobState.FAILED, error={"code": error.code, "message": error.message})
    except Exception:
        logger.exception("training job %s failed", job.job_id)
        finish(session, job, JobState.FAILED, error={"code": "internal_error", "message": INTERNAL_ERROR_MESSAGE})


def finish(session: Session, job: Job, state: JobState, *, results: tuple | None = None,
           error: dict | None = None) -> None:
    """End the job; write the results only if it is still the session's job and no stop was asked."""
    with session.lock:
        if state is JobState.DONE and job.stop_event.is_set():
            state = JobState.STOPPED                  # Stop pressed during the last phase: the results are dropped
        if state is JobState.DONE and session.job is job:
            session.trained, session.synthetic, session.fidelity, session.privacy = results
        job.error, job.ended = error, time.monotonic()
        job.state = state                             # last: whoever sees "done" finds the results


def stop_training(session: Session, timeout: float = STOP_TIMEOUT_SECONDS) -> Job:
    """Ask the job to stop and wait for its thread (without the lock, which the thread needs to finish).

    A job that already ended is returned unchanged (a late click is not an error).

    Returns:
        The job: ``stopped``, or still ``running`` if the timeout passed first.

    Raises:
        ApiError: 409 ``no_training`` when no job was ever started.
    """
    with session.lock:
        job = session.job
        if job is None:
            raise ApiError(STATUS_CONFLICT, "no_training", "There is no training to stop.")
        if not job.active:
            return job
        job.stop_event.set()
    job.thread.join(timeout)
    return job


def reset_session(session: Session, timeout: float = STOP_TIMEOUT_SECONDS) -> None:
    """Forget everything: stop a running job (waiting for it without the lock), drop the dataset and the job.

    Past the timeout the session is emptied anyway: the old thread is then an orphan, which cannot write
    (``finish`` checks ``session.job is job``) and ends by itself. Resetting an empty session is fine.
    """
    with session.lock:
        job = session.job
        if job is not None and job.active:
            job.stop_event.set()
        else:
            job = None
    if job is not None:
        job.thread.join(timeout)
    with session.lock:
        session.reset()
        session.job = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()                  # PyTorch keeps the memory of the dropped model reserved


def job_view(session: Session, job_id: str) -> dict:
    """JSON view of the job for the progress bar.

    Raises:
        ApiError: 404 ``unknown_job`` (never started, or replaced by a newer job).
    """
    with session.lock:
        job = session.job
        if job is None or job.job_id != job_id:
            raise ApiError(STATUS_NOT_FOUND, "unknown_job", "This training is not known: start a new one.")
        return describe_job(job)


def session_view(session: Session) -> dict:
    """The stage and the latest job (running or ended, else ``None``): the UI resumes from it after a refresh.

    A page refresh opens a new Streamlit session, which has lost the ``job_id``, while the job goes on here.
    """
    with session.lock:
        job = session.job
        return {"stage": session.stage.name.lower(), "job": describe_job(job) if job is not None else None}


def describe_job(job: Job) -> dict:
    """JSON view of ``job`` (the caller holds the session lock)."""
    latest = job.latest
    end = job.ended if job.ended is not None else time.monotonic()
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "phase": job.phase.value,
        "step": latest.step if latest else 0,
        "max_steps": job.max_steps,
        "train_loss": float(latest.train.total) if latest else None,
        "validation_loss": float(latest.validation.total) if latest else None,
        "elapsed_seconds": end - job.started,
        "stop_requested": job.stop_event.is_set(),
        "error": job.error,
    }
