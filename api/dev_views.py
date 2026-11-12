"""Developer views: how the engine works on the current dataset (contract, decoder, training plan, history, reports).

Computed on request from the session. They show more, never weaken a protection, and never return a row of
the real data: only the contract (whose min and max stay local), shares, losses and synthetic or random rows.
"""

from __future__ import annotations

import numpy as np

from api.api_config import DECODING_CHECK_ROWS, DECODING_CHECK_TOLERANCE, N_RANDOM_ROWS, STATUS_CONFLICT
from api.errors import ApiError
from api.pipeline import job_view, training_plan
from api.serialization import to_jsonable
from api.session import Session, Stage
from sds.config import RANDOM_STATE
from sds.data_engine import decode
from sds.vae.hyperparameters_config import (
    BETA,
    FEATURES_PER_LATENT,
    HIDDEN_PER_FEATURE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_HIDDEN,
    MAX_LATENT,
    MAX_STEPS,
    MIN_HIDDEN,
    MIN_LATENT,
    MIN_STEPS,
    WEIGHT_DECAY,
)
from sds.vae.layout import MIXED_SIGMA_LOSS


def contract_view(session: Session) -> dict:
    """The full contract (it stays local) next to what the LLM would receive.

    Raises:
        ApiError: 409 with the code of the first missing stage.
    """
    with session.lock:
        session.require(Stage.PREPARED)
        return {"contract": to_jsonable(session.contract.to_dict()), "llm_schema": to_jsonable(session.contract.llm_schema())}


def decoding_check_view(session: Session) -> dict:
    """Check the decoder before any model: train rows encoded and decoded back, and random vectors decoded.

    At most ``DECODING_CHECK_ROWS`` train rows, a seeded sample beyond.

    Raises:
        ApiError: 409 with the code of the first missing stage.
    """
    with session.lock:
        session.require(Stage.PREPARED)
        train, encoder, contract = session.prepared.train, session.encoder, session.contract
    if len(train) > DECODING_CHECK_ROWS:
        train = train.sample(DECODING_CHECK_ROWS, random_state=RANDOM_STATE).sort_index()
    original = train.reset_index(drop=True)
    decoded = decode(encoder.transform(train), contract)
    identical = {}
    for column in contract.columns:
        before, after = original[column.name], decoded[column.name]
        if column.role == "numeric" and not column.integer:
            tolerance = (column.max - column.min) * DECODING_CHECK_TOLERANCE
            same = np.isclose(after.to_numpy(dtype=float), before.to_numpy(dtype=float), rtol=0, atol=tolerance)
        else:
            same = after.to_numpy() == (before.astype("int64") if column.integer else before).to_numpy()
        identical[column.name] = float(np.mean(same))
    noise = np.random.default_rng(RANDOM_STATE).random((N_RANDOM_ROWS, contract.width))
    random_rows = decode(noise, contract)
    return {
        "n_checked": len(train),
        "identical_share": identical,
        "random_rows": {"columns": to_jsonable(list(random_rows.columns)),
                        "rows": to_jsonable(random_rows.to_numpy(dtype=object))},
    }


def hyperparameters_view(session: Session, train_options: dict) -> dict:
    """The training plan the job will use (``training_plan`` with the app's options), with its rules.

    Raises:
        ApiError: 409 with the code of the first missing stage.
    """
    with session.lock:
        session.require(Stage.PREPARED)
        plan = training_plan(len(session.prepared.train), session.contract.width, **train_options)
    return {
        "inputs": {"train_rows": plan.n_rows, "training_rows": plan.n_training,
                   "validation_rows": plan.n_validation, "width": plan.width},
        "architecture": {
            "hidden_dims": list(plan.architecture.hidden_dims),
            "latent_dim": plan.architecture.latent_dim,
            "rules": [f"hidden = {HIDDEN_PER_FEATURE} x W within [{MIN_HIDDEN}, {MAX_HIDDEN}]",
                      f"latent = W / {FEATURES_PER_LATENT} within [{MIN_LATENT}, {MAX_LATENT}]"],
        },
        "budget": {"epochs": MAX_EPOCHS, "steps_per_epoch": plan.steps_per_epoch, "max_steps": plan.max_steps,
                   "limits": [MIN_STEPS, MAX_STEPS], "eval_every": plan.eval_every,
                   "evaluations": -(-plan.max_steps // plan.eval_every)},
        "kl": {"warmup_steps": plan.warmup_steps, "beta": BETA},
        "optimizer": {"learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": plan.batch_size},
        "early_stopping": {"patience": plan.patience, "from_step": plan.warmup_steps},
        "loss": MIXED_SIGMA_LOSS,
        "device": plan.device,
    }


def training_view(session: Session) -> dict:
    """The training history, also while the job runs, and how it ended.

    Raises:
        ApiError: 409 ``no_training`` before the first training.
    """
    with session.lock:
        job = session.job
        if job is None:
            raise ApiError(STATUS_CONFLICT, "no_training", "No training has been started yet.")
        history = list(job.history)                     # the training thread keeps appending
        trained = session.trained                       # present only after a job that ended well
        status = job_view(session, job.job_id)
    result = None
    if trained is not None:
        result = {"best_step": trained.best_step, "stopped_early": trained.stopped_early,
                  "cancelled": trained.cancelled}
    return {"state": status["state"], "phase": status["phase"], "max_steps": status["max_steps"],
            "elapsed_seconds": status["elapsed_seconds"], "history": to_jsonable(history), "result": result}


def validation_details_view(session: Session) -> dict:
    """The full fidelity and privacy reports (distances, baselines, z, excesses, DCR percentiles).

    Raises:
        ApiError: 409 with the code of the first missing stage.
    """
    with session.lock:
        session.require(Stage.VALIDATED)
        return {"fidelity": to_jsonable(session.fidelity), "privacy": to_jsonable(session.privacy)}
