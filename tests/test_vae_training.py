"""Specification tests for the VAE training loop.

The holdout of the Data Engine never reaches this module: a validation part is cut from the train
rows. Data stay on the CPU and reach the device one fixed-size batch at a time (DataLoader), for
training and evaluation alike. Public API under test:
    split_validation(n_rows, *, fraction=VALIDATION_FRACTION, seed=RANDOM_STATE) -> (train_index, validation_index)
        disjoint, covering, at least 1 row each; fewer than 2 rows -> ValueError
    EarlyStopping(patience, start_step): update(step, value) -> bool (new best), best_step, best_value,
        should_stop; values before start_step are ignored
    LossValues(total, numeric, categorical, kl): floats
    evaluate(model, x, layout, *, loss, beta=BETA, batch_size=BATCH_SIZE) -> (LossValues, active_dims)
        eval mode (z = mu), full KL weight beta, batched: the result does not depend on batch_size
    step_budget(n_rows, *, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS) -> StepBudget(max_steps, eval_every)
        the training length is set in epochs over the rows left after the validation cut:
        max_steps = clamp(max_epochs * steps_per_epoch, MIN_STEPS, MAX_STEPS), so the KL warm-up (a share of
        max_steps) and early stopping scale with the data; eval_every = max(1, max_steps // EVALUATIONS)
    train_vae(x, layout, *, loss, architecture=None, beta=BETA, max_steps=None, batch_size=BATCH_SIZE,
              eval_every=None, patience=PATIENCE, seed=RANDOM_STATE, device=None,
              on_progress=None, stop_event=None) -> TrainedVae
        max_steps=None -> step_budget(len(x), batch_size=batch_size); eval_every=None -> max(1, max_steps // EVALUATIONS)
        Evaluation(step, kl_weight, train, validation, active_dims) every eval_every steps and at the end;
        kl_weight = kl_schedule(step, max_steps, beta=beta); validation at the full weight beta;
        early stopping only after the warm-up, best weights restored;
        stop_event set -> stops cleanly, cancelled=True
    TrainedVae(model, layout, loss, history, best_step, stopped_early, cancelled, train_index, validation_index)
        model in eval mode, on the training device
"""

import math
import threading

import numpy as np
import pytest
import torch

from sds.data_engine import SchemaContract
from sds.data_engine.contract import ContractColumn
from sds.vae.hyperparameters_config import (
    EVALUATIONS,
    INITIAL_SIGMA,
    KL_WARMUP_FRACTION,
    MAX_EPOCHS,
    MAX_STEPS,
    MIN_STEPS,
)
from sds.vae.layout import BASELINE_LOSS, LOSS_NAMES, MIXED_LOSS, MIXED_SIGMA_LOSS, OutputLayout, VaeError
from sds.vae.losses import kl_schedule
from sds.vae.model import Architecture, TabularVae, choose_architecture
from sds.vae.training import EarlyStopping, LossValues, StepBudget, evaluate, split_validation, step_budget, train_vae

CPU = torch.device("cpu")
SMALL = Architecture(hidden_dims=(32, 32), latent_dim=4)
TOLERANCE = 1e-5

LAYOUT = OutputLayout.from_contract(SchemaContract(version=1, width=7, columns=[
    ContractColumn("age", "numeric", 0, 1, min=0.0, max=1.0),
    ContractColumn("sex", "categorical", 1, 3, levels=("F", "M")),
    ContractColumn("bmi", "numeric", 3, 4, min=0.0, max=1.0),
    ContractColumn("grade", "categorical", 4, 7, levels=("a", "b", "c")),
], dropped=[], rows={}, issues=[]))


def encoded_rows(n_rows: int, seed: int = 0) -> np.ndarray:
    """Rows shaped like the encoder output for LAYOUT: numeric in [0, 1], one 1 per group."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_rows, LAYOUT.width), dtype=np.float32)
    x[:, [0, 3]] = rng.random((n_rows, 2))
    x[np.arange(n_rows), 1 + rng.integers(0, 2, n_rows)] = 1
    x[np.arange(n_rows), 4 + rng.integers(0, 3, n_rows)] = 1
    return x


def train(x=None, **kwargs):
    options = {"loss": MIXED_LOSS, "architecture": SMALL, "max_steps": 100, "eval_every": 20,
               "batch_size": 32, "device": CPU} | kwargs
    return train_vae(encoded_rows(100) if x is None else x, LAYOUT, **options)


def reconstruction(values: LossValues) -> float:
    return values.numeric + values.categorical


# --------------------------------------------------------------------------------------------
# Validation split (cut from the train rows; the holdout never comes here)
# --------------------------------------------------------------------------------------------

def test_split_sizes_and_coverage():
    train_index, validation_index = split_validation(100)
    assert len(validation_index) == 10 and len(train_index) == 90
    assert set(train_index).isdisjoint(validation_index)
    assert set(train_index) | set(validation_index) == set(range(100))


def test_split_is_seeded():
    assert np.array_equal(split_validation(100, seed=1)[1], split_validation(100, seed=1)[1])
    assert not np.array_equal(split_validation(100, seed=1)[1], split_validation(100, seed=2)[1])


def test_split_keeps_one_row_on_each_side():
    train_index, validation_index = split_validation(5)
    assert len(validation_index) == 1 and len(train_index) == 4
    assert [len(part) for part in split_validation(2)] == [1, 1]
    assert [len(part) for part in split_validation(2, fraction=0.9)] == [1, 1]     # never empties training


def test_split_needs_two_rows():
    with pytest.raises(ValueError):
        split_validation(1)


# --------------------------------------------------------------------------------------------
# Early stopping (pure logic)
# --------------------------------------------------------------------------------------------

def test_early_stopping_ignores_the_warm_up():
    stopping = EarlyStopping(patience=1, start_step=100)
    assert not stopping.update(50, 1.0)
    assert not stopping.update(60, 5.0)
    assert not stopping.should_stop and stopping.best_step is None


def test_early_stopping_counts_evaluations_without_improvement():
    stopping = EarlyStopping(patience=2, start_step=0)
    assert stopping.update(10, 3.0)
    assert stopping.update(20, 2.0)
    assert not stopping.update(30, 2.5)
    assert not stopping.should_stop
    assert stopping.update(40, 1.0)                   # an improvement resets the count
    assert not stopping.update(50, 1.0)                # equal is not better
    assert not stopping.update(60, 4.0)
    assert stopping.should_stop
    assert stopping.best_step == 40 and stopping.best_value == 1.0


# --------------------------------------------------------------------------------------------
# Evaluation: deterministic, at the full KL weight, batched
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("loss", LOSS_NAMES)
def test_evaluation_does_not_depend_on_the_batch_size(loss):
    torch.manual_seed(0)
    model = TabularVae(LAYOUT.width, SMALL, n_sigma=2 if loss == MIXED_SIGMA_LOSS else None)
    x = encoded_rows(50)
    small, dims_small = evaluate(model, x, LAYOUT, loss=loss, batch_size=7)
    whole, dims_whole = evaluate(model, x, LAYOUT, loss=loss, batch_size=50)
    for part in ("total", "numeric", "categorical", "kl"):
        assert getattr(small, part) == pytest.approx(getattr(whole, part), rel=TOLERANCE)
    assert dims_small == dims_whole


def test_evaluation_is_deterministic_and_uses_the_full_weight():
    torch.manual_seed(0)
    model = TabularVae(LAYOUT.width, SMALL).train()
    x = encoded_rows(50)
    first, _ = evaluate(model, x, LAYOUT, loss=MIXED_LOSS, beta=0.5)
    second, _ = evaluate(model, x, LAYOUT, loss=MIXED_LOSS, beta=0.5)
    assert first == second
    assert first.total == pytest.approx(first.numeric + first.categorical + 0.5 * first.kl, rel=TOLERANCE)


# --------------------------------------------------------------------------------------------
# Sanity check: the network can memorise a few rows (beta = 0)
# --------------------------------------------------------------------------------------------

def test_sanity_check_memorises_64_rows():
    x = encoded_rows(64)
    result = train(x, beta=0.0, max_steps=1500, eval_every=100, patience=1_000, batch_size=64)
    torch.manual_seed(0)
    untrained, _ = evaluate(TabularVae(LAYOUT.width, SMALL), x[result.train_index], LAYOUT, loss=MIXED_LOSS)
    assert reconstruction(result.history[-1].train) < 0.1 * reconstruction(untrained)


# --------------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------------

def test_history_and_progress_callback():
    seen = []
    result = train(max_steps=90, eval_every=20, beta=0.5, on_progress=seen.append)
    assert [e.step for e in result.history] == [20, 40, 60, 80, 90]
    assert seen == result.history
    for evaluation in result.history:
        assert evaluation.kl_weight == pytest.approx(kl_schedule(evaluation.step, 90, beta=0.5))
        v = evaluation.validation
        assert v.total == pytest.approx(v.numeric + v.categorical + 0.5 * v.kl, rel=TOLERANCE)
        assert 0 <= evaluation.active_dims <= SMALL.latent_dim


def test_indices_split_the_rows():
    result = train(encoded_rows(100))
    assert set(result.train_index).isdisjoint(result.validation_index)
    assert len(result.train_index) + len(result.validation_index) == 100
    assert len(result.validation_index) == 10


def test_early_stopping_restores_the_best_weights():
    x = encoded_rows(40)
    max_steps = 3000
    result = train(x, max_steps=max_steps, eval_every=10, patience=2, batch_size=16)
    assert result.stopped_early and not result.cancelled
    assert result.history[-1].step < max_steps
    warmup = round(KL_WARMUP_FRACTION * max_steps)
    counted = [e for e in result.history if e.step >= warmup]
    best = min(counted, key=lambda e: e.validation.total)
    assert result.best_step == best.step
    restored, _ = evaluate(result.model, x[result.validation_index], LAYOUT, loss=MIXED_LOSS)
    assert restored.total == pytest.approx(best.validation.total, rel=TOLERANCE)


def test_runs_to_the_end_without_early_stopping():
    result = train(max_steps=100, patience=1_000)
    assert not result.stopped_early and not result.cancelled
    assert result.history[-1].step == 100


def test_same_seed_same_training():
    first, second = train(seed=3), train(seed=3)
    assert first.history == second.history
    assert first.history != train(seed=4).history


def test_cancel_during_training():
    stop = threading.Event()
    result = train(max_steps=1_000, eval_every=20, stop_event=stop, on_progress=lambda e: stop.set())
    assert result.cancelled and not result.stopped_early
    assert [e.step for e in result.history] == [20]


def test_cancel_before_the_start():
    stop = threading.Event()
    stop.set()
    result = train(stop_event=stop)
    assert result.cancelled and result.history == []


@pytest.mark.parametrize("loss", LOSS_NAMES)
def test_every_loss_trains(loss):
    result = train(loss=loss, max_steps=60)
    assert result.loss == loss
    assert np.isfinite(result.history[-1].validation.total)
    if loss == MIXED_SIGMA_LOSS:
        assert result.model.log_sigma is not None
        start = torch.full_like(result.model.log_sigma, math.log(INITIAL_SIGMA))
        assert not torch.allclose(result.model.log_sigma, start)          # sigma is learned
    else:
        assert result.model.log_sigma is None


def test_batch_larger_than_the_data():
    result = train(encoded_rows(20), batch_size=256, max_steps=40)
    assert len(result.history) == 2


def test_default_architecture_follows_the_rule():
    x = encoded_rows(100)
    result = train_vae(x, LAYOUT, loss=BASELINE_LOSS, max_steps=20, eval_every=20, device=CPU)
    assert result.model.architecture == choose_architecture(LAYOUT.width)


def test_model_is_returned_in_eval_mode_on_the_device():
    result = train()
    assert not result.model.training
    assert next(result.model.parameters()).device.type == "cpu"
    assert result.layout is LAYOUT


def test_wrong_width():
    with pytest.raises(VaeError) as error:
        train_vae(np.zeros((20, LAYOUT.width + 1), dtype=np.float32), LAYOUT, loss=MIXED_LOSS, device=CPU)
    assert error.value.code == "wrong_width"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_training_on_cuda():
    result = train(device=torch.device("cuda"), max_steps=40)
    assert next(result.model.parameters()).device.type == "cuda"
    assert np.isfinite(result.history[-1].validation.total)


# --------------------------------------------------------------------------------------------
# Training length in epochs (capacity experiment: fixed steps over-trained the small datasets)
# --------------------------------------------------------------------------------------------

def test_budget_by_hand():
    # 500 rows -> 50 for validation, 450 for training -> 2 batches of 256 per epoch
    assert step_budget(500, batch_size=256, max_epochs=1_000) == StepBudget(max_steps=2_000,
                                                                            eval_every=2_000 // EVALUATIONS)


@pytest.mark.parametrize("n_rows, steps_per_epoch", [(284, 1), (286, 2)])
def test_epochs_count_the_rows_left_after_the_validation_cut(n_rows, steps_per_epoch):
    # 284 rows -> 28 validation, 256 training (1 batch); 286 rows -> 29 validation, 257 training (2 batches)
    assert step_budget(n_rows, batch_size=256, max_epochs=1_000).max_steps == steps_per_epoch * 1_000


def test_batch_size_sets_the_steps_per_epoch():
    assert step_budget(500, batch_size=64, max_epochs=1_000).max_steps == 8 * 1_000      # ceil(450 / 64)


def test_batch_larger_than_the_rows_is_one_step_per_epoch():
    assert step_budget(40, batch_size=256, max_epochs=1_000).max_steps == 1_000


def test_large_datasets_are_capped_at_max_steps():
    assert step_budget(24_130) == StepBudget(max_steps=MAX_STEPS, eval_every=MAX_STEPS // EVALUATIONS)


def test_tiny_budgets_are_raised_to_min_steps():
    assert step_budget(500, max_epochs=1).max_steps == MIN_STEPS


def test_budget_grows_with_the_rows_and_stays_within_bounds():
    budgets = [step_budget(n) for n in range(2, 60_000, 97)]
    steps = [b.max_steps for b in budgets]
    assert steps == sorted(steps)
    assert all(MIN_STEPS <= s <= MAX_STEPS for s in steps)
    for budget in budgets:
        assert budget.eval_every >= 1
        assert EVALUATIONS <= budget.max_steps // budget.eval_every < 2 * EVALUATIONS


def test_budget_needs_two_rows():
    with pytest.raises(ValueError):
        step_budget(1)


def test_calibrated_defaults():
    assert MAX_EPOCHS == 1_000                 # chosen by the capacity experiment
    assert 0 < MIN_STEPS < MAX_STEPS


def test_default_length_follows_the_budget():
    x = encoded_rows(20)                       # 18 training rows, 3 batches of 8 per epoch
    budget = step_budget(len(x), batch_size=8)
    assert budget.max_steps == 3 * MAX_EPOCHS
    result = train_vae(x, LAYOUT, loss=MIXED_LOSS, architecture=SMALL, batch_size=8, patience=10**6, device=CPU)
    assert result.history[-1].step == budget.max_steps
    assert [e.step for e in result.history[:-1]] == list(range(budget.eval_every, budget.max_steps, budget.eval_every))
    warmup = round(KL_WARMUP_FRACTION * budget.max_steps)
    assert result.history[-1].kl_weight == pytest.approx(kl_schedule(budget.max_steps, budget.max_steps))
    assert all(e.kl_weight < 1 for e in result.history if e.step < warmup)


def test_explicit_max_steps_sets_the_evaluation_interval():
    result = train(max_steps=300, eval_every=None, patience=10**6)
    expected = max(1, 300 // EVALUATIONS)
    assert [e.step for e in result.history] == list(range(expected, 301, expected))
