"""Specification tests for the developer views (``/dev/*``): more of how the engine works, never a real row."""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.pipeline
from api.api_config import DECODING_CHECK_ROWS, N_RANDOM_ROWS
from api.dataset import load_dataset
from api.dev_views import (
    contract_view,
    decoding_check_view,
    hyperparameters_view,
    training_view,
    validation_details_view,
)
from api.errors import ApiError
from api.main import create_app
from api.pipeline import JobState, start_training, stop_training, training_plan
from api.session import Session, Stage
from api_csv import make_csv
from sds.vae.hyperparameters_config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
)
from sds.vae.losses import warmup_steps
from sds.vae.model import Architecture, choose_architecture
from sds.vae.runtime import get_device
from sds.vae.training import evaluation_interval, split_validation, step_budget

WAIT_SECONDS = 60
SMALL = Architecture(hidden_dims=(16,), latent_dim=2)
FAST = {"max_steps": 20, "architecture": SMALL, "device": "cpu"}
ENDLESS = {"max_steps": 10**7, "eval_every": 5, "architecture": SMALL, "device": "cpu"}
CSV_HEADERS = {"Content-Type": "text/csv"}
MARKER_ROWS = 300


def marker_csv() -> tuple[bytes, list[float]]:
    """The test CSV plus ``marker``: one distinct value per row, 6 decimals over a range of 10^6.

    A chance collision with a synthetic value is then negligible: a marker value in a response was sent by us.
    """
    rng = np.random.default_rng(11)
    lines = make_csv(MARKER_ROWS).decode("utf-8").splitlines()
    markers = [round(float(v), 6) for v in rng.uniform(0, 1e6, MARKER_ROWS)]
    rows = [lines[0] + ",marker"] + [f"{line},{m:.6f}" for line, m in zip(lines[1:], markers, strict=True)]
    return ("\n".join(rows) + "\n").encode("utf-8"), markers


def numbers_in(value) -> set[float]:
    found = set()
    if isinstance(value, dict):
        for item in value.values():
            found |= numbers_in(item)
    elif isinstance(value, list):
        for item in value:
            found |= numbers_in(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        found.add(round(float(value), 6))
    return found


def prepared_session(data: bytes | None = None) -> Session:
    session = Session()
    load_dataset(session, make_csv() if data is None else data, "patients.csv")
    assert session.stage is Stage.PREPARED
    return session


def finished(job):
    job.thread.join(WAIT_SECONDS)
    assert not job.thread.is_alive()
    return job


def strict(value) -> None:
    json.dumps(value, allow_nan=False)


# --- training_plan: one source for the panel and the job -------------------------------------

@pytest.mark.parametrize("n_rows, width", [(96, 21), (614, 21), (24_130, 119), (89_493, 151)])
def test_plan_follows_the_engine_rules(n_rows, width):
    plan = training_plan(n_rows, width)
    n_training, n_validation = (len(part) for part in split_validation(n_rows))
    budget = step_budget(n_rows)
    assert (plan.n_rows, plan.n_training, plan.n_validation, plan.width) == (n_rows, n_training, n_validation, width)
    assert plan.architecture == choose_architecture(width)
    assert plan.max_steps == budget.max_steps and plan.eval_every == budget.eval_every
    assert plan.warmup_steps == warmup_steps(budget.max_steps)
    assert plan.batch_size == min(BATCH_SIZE, n_training)
    assert plan.steps_per_epoch == -(-n_training // BATCH_SIZE)
    assert plan.patience == PATIENCE and plan.device == str(get_device())


def test_plan_options_replace_the_defaults():
    plan = training_plan(614, 21, **ENDLESS)
    assert plan.architecture == SMALL and plan.max_steps == 10**7 and plan.eval_every == 5 and plan.device == "cpu"
    assert plan.warmup_steps == warmup_steps(10**7)
    assert training_plan(614, 21, max_steps=1000).eval_every == evaluation_interval(1000)


def test_the_job_trains_with_the_plan_shown_in_the_panel(monkeypatch):
    seen = {}

    def fake_train(x, layout, **kwargs):
        seen.update(kwargs)
        raise api.pipeline.VaeError("fake", "stop here")

    monkeypatch.setattr(api.pipeline, "train_vae", fake_train)
    session = prepared_session()
    shown = hyperparameters_view(session, {})
    finished(start_training(session))
    assert seen["max_steps"] == shown["budget"]["max_steps"]
    assert seen["eval_every"] == shown["budget"]["eval_every"]
    assert seen["patience"] == shown["early_stopping"]["patience"]
    assert str(seen["device"]) == shown["device"]
    assert list(seen["architecture"].hidden_dims) == shown["architecture"]["hidden_dims"]
    assert seen["architecture"].latent_dim == shown["architecture"]["latent_dim"]
    assert seen["batch_size"] == BATCH_SIZE


# --- /dev/hyperparameters -------------------------------------------------------------------

def test_hyperparameters_view():
    session = prepared_session()
    view = hyperparameters_view(session, {})
    strict(view)
    n_rows, width = len(session.prepared.train), session.contract.width
    assert view["inputs"] == {"train_rows": n_rows, "training_rows": training_plan(n_rows, width).n_training,
                              "validation_rows": training_plan(n_rows, width).n_validation, "width": width}
    assert view["budget"]["epochs"] == MAX_EPOCHS and view["budget"]["limits"] == [200, 20_000]
    assert view["architecture"]["rules"] and view["loss"] == "mixed_sigma"
    assert view["optimizer"] == {"learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                                 "batch_size": min(BATCH_SIZE, view["inputs"]["training_rows"])}
    assert view["kl"]["warmup_steps"] == view["early_stopping"]["from_step"]


def test_number_of_evaluations():
    session = prepared_session()
    for options, expected in [({"max_steps": 1000}, 100), ({"max_steps": 1000, "eval_every": 3}, 334)]:
        assert hyperparameters_view(session, options)["budget"]["evaluations"] == expected


def test_hyperparameters_need_a_prepared_dataset():
    with pytest.raises(ApiError) as error:
        hyperparameters_view(Session(), {})
    assert error.value.code == "no_dataset"


# --- /dev/contract --------------------------------------------------------------------------

def test_contract_view():
    session = prepared_session()
    view = contract_view(session)
    strict(view)
    assert view["contract"] == json.loads(json.dumps(session.contract.to_dict()))
    assert view["llm_schema"] == json.loads(json.dumps(session.contract.llm_schema()))


def test_contract_view_needs_a_prepared_dataset():
    with pytest.raises(ApiError):
        contract_view(Session())


# --- /dev/decoding-check --------------------------------------------------------------------

def test_decoding_check():
    session = prepared_session()
    view = decoding_check_view(session)
    strict(view)
    names = [c.name for c in session.contract.columns]
    assert list(view["identical_share"]) == names
    assert all(0 <= share <= 1 for share in view["identical_share"].values())
    assert view["identical_share"]["age"] == 1.0                    # whole numbers come back exactly
    assert view["identical_share"]["weight"] == 1.0                 # one decimal, rounded back exactly
    assert view["n_checked"] == len(session.prepared.train)
    assert view["random_rows"]["columns"] == names and len(view["random_rows"]["rows"]) == N_RANDOM_ROWS


def test_decoding_check_sees_the_values_merged_into_other():
    lines = make_csv().decode("utf-8").splitlines()
    rows, rare = [lines[0]], 6                                       # fewer rows than MIN_LEVEL_ROWS -> "Other"
    for line in lines[1:]:
        if rare and "Turin" in line:
            line, rare = line.replace("Turin", "Bari"), rare - 1
        rows.append(line)
    data = ("\n".join(rows) + "\n").encode("utf-8")
    session = prepared_session(data)
    n_bari = int((session.prepared.train["city"] == "Bari").sum())
    view = decoding_check_view(session)
    assert view["identical_share"]["city"] == pytest.approx(1 - n_bari / len(session.prepared.train))
    assert n_bari > 0 and view["identical_share"]["city"] < 1


def test_decoding_check_is_capped(monkeypatch):
    monkeypatch.setattr(api.dev_views, "DECODING_CHECK_ROWS", 40)
    session = prepared_session()
    view = decoding_check_view(session)
    assert view["n_checked"] == 40 < len(session.prepared.train)
    assert DECODING_CHECK_ROWS == 10_000


def test_decoding_check_is_deterministic():
    session = prepared_session()
    assert decoding_check_view(session) == decoding_check_view(session)


# --- /dev/training --------------------------------------------------------------------------

def test_training_view_needs_a_job():
    with pytest.raises(ApiError) as error:
        training_view(prepared_session())
    assert error.value.code == "no_training"


def test_training_history_grows_during_the_job():
    session = prepared_session()
    job = start_training(session, **ENDLESS)
    try:
        deadline = time.monotonic() + WAIT_SECONDS
        while len(job.history) < 3:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        view = training_view(session)
        strict(view)
        assert view["state"] == "running" and view["result"] is None
        assert len(view["history"]) >= 3
        first = view["history"][0]
        assert set(first) == {"step", "kl_weight", "train", "validation", "active_dims"}
        assert set(first["train"]) == {"total", "numeric", "categorical", "kl"}
        assert [h["step"] for h in view["history"]] == sorted(h["step"] for h in view["history"])
    finally:
        stop_training(session)


def test_training_view_after_the_job():
    session = prepared_session()
    job = finished(start_training(session, **FAST))
    view = training_view(session)
    assert view["state"] == "done" and len(view["history"]) == len(session.trained.history) == len(job.history)
    assert view["result"] == {"best_step": session.trained.best_step, "stopped_early": session.trained.stopped_early,
                              "cancelled": False}
    assert view["elapsed_seconds"] > 0 and view["max_steps"] == FAST["max_steps"]


def test_training_view_of_a_stopped_job_has_no_result():
    session = prepared_session()
    job = start_training(session, **ENDLESS)
    while not job.history:
        time.sleep(0.02)
    stop_training(session)
    view = training_view(session)
    assert view["state"] == "stopped" and view["result"] is None and view["history"]


# --- /dev/validation ------------------------------------------------------------------------

def test_validation_details():
    session = prepared_session()
    finished(start_training(session, **FAST))
    view = validation_details_view(session)
    strict(view)
    assert set(view) == {"fidelity", "privacy"}
    assert view["privacy"]["passed"] == session.privacy.passed
    assert view["privacy"]["synthetic_percentile"] == pytest.approx(session.privacy.synthetic_percentile)
    assert [c["name"] for c in view["fidelity"]["columns"]] == [c.name for c in session.fidelity.columns]


def test_validation_details_need_a_validation():
    with pytest.raises(ApiError) as error:
        validation_details_view(prepared_session())
    assert error.value.code == "not_trained"


# --- never a real row -----------------------------------------------------------------------

def test_no_real_value_leaves_through_the_developer_views():
    data, markers = marker_csv()
    session = prepared_session(data)
    assert "marker" in session.schema.columns and session.schema.columns["marker"].role == "numeric"
    finished(start_training(session, **FAST))
    train_markers = {round(v, 6) for v in session.prepared.train["marker"]}
    allowed = {round(float(session.prepared.train["marker"].min()), 6),
               round(float(session.prepared.train["marker"].max()), 6)}    # contract statistics, local only
    views = {"contract": contract_view(session), "decoding": decoding_check_view(session),
             "hyperparameters": hyperparameters_view(session, {}), "training": training_view(session),
             "validation": validation_details_view(session)}
    for name, view in views.items():
        leaked = (numbers_in(view) & train_markers) - allowed
        assert not leaked, f"{name} returned real values: {sorted(leaked)[:3]}"
    assert len(markers) == MARKER_ROWS


# --- HTTP -----------------------------------------------------------------------------------

DEV_PATHS = ["/dev/contract", "/dev/decoding-check", "/dev/hyperparameters", "/dev/training", "/dev/validation"]


def client(dev_mode: bool) -> TestClient:
    app = create_app(dev_mode=dev_mode)
    app.state.train_options = dict(FAST)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("path", DEV_PATHS)
def test_developer_views_are_absent_without_the_flag(path):
    c = client(False)
    c.post("/upload", content=make_csv(), headers=CSV_HEADERS)
    assert c.get(path).status_code == 404


def test_developer_views_over_http():
    c = client(True)
    c.post("/upload", content=make_csv(), headers=CSV_HEADERS)
    for path in ("/dev/contract", "/dev/decoding-check", "/dev/hyperparameters"):
        assert c.get(path).status_code == 200, path
    assert c.get("/dev/training").json()["code"] == "no_training"
    assert c.get("/dev/hyperparameters").json()["budget"]["max_steps"] == FAST["max_steps"]   # the app's options
    job_id = c.post("/train").json()["job_id"]
    deadline = time.monotonic() + WAIT_SECONDS
    while c.get(f"/status/{job_id}").json()["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert c.app.state.session.job.state is JobState.DONE
    assert c.get("/dev/training").status_code == 200 and c.get("/dev/validation").status_code == 200
