"""Specification tests for the VAE losses.

Three variants on the same network; every reconstruction term is a sum over the features of a row,
averaged over the batch, like the KL term (the default mean of ``F.mse_loss`` would divide the
reconstruction by the vector width and let the KL win). Public API under test:
    vae_loss(logits, x, mu, logvar, layout, *, loss, kl_weight=BETA, log_sigma=None) -> LossParts
        LossParts(total, numeric, categorical, kl); kl is unweighted,
        total = numeric + categorical + kl_weight * kl
        loss="baseline":    MSE after a sigmoid on every position (split into numeric / group positions)
        loss="mixed":       MSE after a sigmoid on numeric positions + cross-entropy per one-hot group
        loss="mixed_sigma": Gaussian NLL per numeric column, (x - x_hat)^2 / (2 sigma^2) + log sigma,
                            sigma learned and clamped to [MIN_SIGMA, MAX_SIGMA] + cross-entropy per group
        log_sigma (one per numeric position) is required by "mixed_sigma" and refused by the others.
    kl_divergence(mu, logvar) -> KL(q(z|x) || N(0, I)), summed over latent dims, averaged over rows
    kl_schedule(step, total_steps) -> KL weight: linear warm-up from 0 to BETA over the first
        KL_WARMUP_FRACTION of the steps, then BETA
    active_latent_dims(kl_per_dim(mu, logvar)) -> latent dims whose mean KL is above ACTIVE_DIM_KL
"""

import math

import pytest
import torch
import torch.nn.functional as F

from sds.data_engine import SchemaContract
from sds.data_engine.contract import ContractColumn
from sds.vae.hyperparameters_config import ACTIVE_DIM_KL, BETA, KL_WARMUP_FRACTION, MAX_SIGMA, MIN_SIGMA
from sds.vae.layout import BASELINE_LOSS, MIXED_LOSS, MIXED_SIGMA_LOSS, OutputLayout, activate
from sds.vae.losses import LossParts, active_latent_dims, kl_divergence, kl_per_dim, kl_schedule, vae_loss

ALL_LOSSES = (BASELINE_LOSS, MIXED_LOSS, MIXED_SIGMA_LOSS)
LATENT = 2
TOLERANCE = 1e-5
CONFIDENT_LOGIT = 30.0          # softmax of (30, 0, 0) is one-hot up to float precision


def make_layout(columns: list[ContractColumn], width: int) -> OutputLayout:
    contract = SchemaContract(version=1, width=width, columns=columns, dropped=[], rows={}, issues=[])
    return OutputLayout.from_contract(contract)


# numeric at 0, a 3-slot group at 1..3, numeric at 4
LAYOUT = make_layout([
    ContractColumn("a", "numeric", 0, 1, min=0.0, max=1.0),
    ContractColumn("c", "categorical", 1, 4, levels=("x", "y", "z")),
    ContractColumn("b", "numeric", 4, 5, min=0.0, max=1.0),
], width=5)
X = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, 1.0, 0.0]])
ZERO_LOGITS = torch.zeros_like(X)                       # sigmoid -> 0.5, softmax -> uniform
MU0 = torch.zeros(len(X), LATENT)
LOGVAR0 = torch.zeros(len(X), LATENT)


def log_sigma(value: float) -> torch.Tensor:
    return torch.full((len(LAYOUT.numeric_index),), value)


def loss_of(name: str, logits=ZERO_LOGITS, x=X, mu=MU0, logvar=LOGVAR0, layout=LAYOUT, **kwargs) -> LossParts:
    if name == MIXED_SIGMA_LOSS and "log_sigma" not in kwargs:
        kwargs["log_sigma"] = log_sigma(0.0)
    return vae_loss(logits, x, mu, logvar, layout, loss=name, **kwargs)


def close(a: torch.Tensor, b: float) -> bool:
    return math.isclose(float(a), b, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


# --------------------------------------------------------------------------------------------
# KL divergence
# --------------------------------------------------------------------------------------------

def test_kl_is_zero_for_the_prior():
    assert close(kl_divergence(MU0, LOGVAR0), 0.0)


def test_kl_closed_form():
    mu = torch.tensor([[1.0, 2.0]])
    logvar = torch.tensor([[0.0, math.log(4.0)]])
    expected = 0.5 * 1.0 + (-0.5 * (1 + math.log(4.0) - 4.0 - 4.0))   # -1/2 (1 + log s2 - mu2 - s2) per dim
    assert close(kl_divergence(mu, logvar), expected)


def test_kl_is_averaged_over_rows():
    mu = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    assert close(kl_divergence(mu, torch.zeros_like(mu)), (0.5 + 4.5) / 2)


# --------------------------------------------------------------------------------------------
# Reduction: sum over the features of a row, mean over the batch
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_LOSSES)
def test_doubling_the_batch_does_not_change_the_loss(name):
    single = loss_of(name)
    double = loss_of(name, logits=ZERO_LOGITS.repeat(2, 1), x=X.repeat(2, 1),
                     mu=MU0.repeat(2, 1), logvar=LOGVAR0.repeat(2, 1))
    for part in ("total", "numeric", "categorical", "kl"):
        assert close(getattr(double, part), float(getattr(single, part))), part


def test_numeric_part_is_a_sum_over_columns():
    # sigmoid(0) = 0.5 against targets 0/1: 0.25 per numeric column, 2 columns per row
    assert close(loss_of(BASELINE_LOSS).numeric, 0.5)
    wider = make_layout([
        ContractColumn("a", "numeric", 0, 1), ContractColumn("b", "numeric", 1, 2),
        ContractColumn("d", "numeric", 2, 3),
    ], width=3)
    x = torch.tensor([[0.0, 1.0, 0.0]])
    parts = vae_loss(torch.zeros_like(x), x, MU0[:1], LOGVAR0[:1], wider, loss=BASELINE_LOSS)
    assert close(parts.numeric, 0.75)


# --------------------------------------------------------------------------------------------
# Baseline: MSE after a sigmoid everywhere
# --------------------------------------------------------------------------------------------

def test_baseline_categorical_part_is_mse_on_group_positions():
    assert close(loss_of(BASELINE_LOSS).categorical, 3 * 0.25)


def test_baseline_reconstruction_is_the_summed_mse():
    logits = torch.randn(X.shape, generator=torch.Generator().manual_seed(0))
    parts = loss_of(BASELINE_LOSS, logits=logits)
    expected = F.mse_loss(torch.sigmoid(logits), X, reduction="sum") / len(X)
    assert close(parts.numeric + parts.categorical, float(expected))


def test_baseline_uses_the_baseline_activation():
    logits = torch.randn(X.shape, generator=torch.Generator().manual_seed(1))
    out = activate(logits, LAYOUT, loss=BASELINE_LOSS)
    parts = loss_of(BASELINE_LOSS, logits=logits)
    assert close(parts.numeric + parts.categorical, float(((out - X) ** 2).sum() / len(X)))


# --------------------------------------------------------------------------------------------
# Mixed (a): MSE on numeric positions + cross-entropy per group
# --------------------------------------------------------------------------------------------

def test_mixed_numeric_part_equals_baseline():
    logits = torch.randn(X.shape, generator=torch.Generator().manual_seed(2))
    assert close(loss_of(MIXED_LOSS, logits=logits).numeric, float(loss_of(BASELINE_LOSS, logits=logits).numeric))


def test_mixed_cross_entropy_of_uniform_logits():
    assert close(loss_of(MIXED_LOSS).categorical, math.log(3))


@pytest.mark.parametrize("name", [MIXED_LOSS, MIXED_SIGMA_LOSS])
def test_cross_entropy_matches_manual_computation(name):
    logits = torch.randn(X.shape, generator=torch.Generator().manual_seed(3))
    target = X[:, 1:4].argmax(dim=1)
    expected = -torch.log_softmax(logits[:, 1:4], dim=1)[torch.arange(len(X)), target].mean()
    assert close(loss_of(name, logits=logits).categorical, float(expected))


def test_confident_right_logits_give_almost_zero_cross_entropy():
    logits = ZERO_LOGITS.clone()
    logits[:, 1:4] = X[:, 1:4] * CONFIDENT_LOGIT
    assert float(loss_of(MIXED_LOSS, logits=logits).categorical) < TOLERANCE


def test_mixed_without_categorical_columns():
    layout = make_layout([ContractColumn("a", "numeric", 0, 1)], width=1)
    x = torch.tensor([[1.0]])
    parts = vae_loss(torch.zeros_like(x), x, MU0[:1], LOGVAR0[:1], layout, loss=MIXED_LOSS)
    assert close(parts.categorical, 0.0)
    assert close(parts.numeric, 0.25)


# --------------------------------------------------------------------------------------------
# Mixed (c): Gaussian NLL with a learned sigma per numeric column
# --------------------------------------------------------------------------------------------

def test_sigma_nll_formula():
    # sigma = 1/sqrt(2): (x - x_hat)^2 / (2 sigma^2) = squared error, plus log sigma per column
    value = math.log(1 / math.sqrt(2))
    parts = loss_of(MIXED_SIGMA_LOSS, log_sigma=log_sigma(value))
    assert close(parts.numeric, float(loss_of(MIXED_LOSS).numeric) + 2 * value)


def test_sigma_nll_with_unit_sigma_is_half_the_mse():
    assert close(loss_of(MIXED_SIGMA_LOSS, log_sigma=log_sigma(0.0)).numeric, 0.5 * 0.5)


def test_sigma_is_clamped():
    low = loss_of(MIXED_SIGMA_LOSS, log_sigma=log_sigma(-100.0)).numeric
    high = loss_of(MIXED_SIGMA_LOSS, log_sigma=log_sigma(100.0)).numeric
    assert close(low, float(loss_of(MIXED_SIGMA_LOSS, log_sigma=log_sigma(math.log(MIN_SIGMA))).numeric))
    assert close(high, float(loss_of(MIXED_SIGMA_LOSS, log_sigma=log_sigma(math.log(MAX_SIGMA))).numeric))
    assert torch.isfinite(low)


def test_sigma_categorical_part_equals_mixed():
    logits = torch.randn(X.shape, generator=torch.Generator().manual_seed(4))
    assert close(loss_of(MIXED_SIGMA_LOSS, logits=logits).categorical,
                 float(loss_of(MIXED_LOSS, logits=logits).categorical))


def test_sigma_without_numeric_columns():
    layout = make_layout([ContractColumn("c", "categorical", 0, 2, levels=("x", "y"))], width=2)
    x = torch.tensor([[1.0, 0.0]])
    parts = vae_loss(torch.zeros_like(x), x, MU0[:1], LOGVAR0[:1], layout, loss=MIXED_SIGMA_LOSS,
                     log_sigma=torch.zeros(0))
    assert close(parts.numeric, 0.0)
    assert close(parts.categorical, math.log(2))


def test_log_sigma_is_required_by_mixed_sigma():
    with pytest.raises(ValueError):
        vae_loss(ZERO_LOGITS, X, MU0, LOGVAR0, LAYOUT, loss=MIXED_SIGMA_LOSS)


@pytest.mark.parametrize("name", [BASELINE_LOSS, MIXED_LOSS])
def test_log_sigma_is_refused_by_the_other_losses(name):
    with pytest.raises(ValueError):
        vae_loss(ZERO_LOGITS, X, MU0, LOGVAR0, LAYOUT, loss=name, log_sigma=log_sigma(0.0))


def test_log_sigma_must_have_one_value_per_numeric_column():
    with pytest.raises(ValueError):
        loss_of(MIXED_SIGMA_LOSS, log_sigma=torch.zeros(3))


def test_sigma_uses_the_mixed_activation():
    logits = torch.randn(X.shape, generator=torch.Generator().manual_seed(5))
    assert torch.equal(activate(logits, LAYOUT, loss=MIXED_SIGMA_LOSS), activate(logits, LAYOUT, loss=MIXED_LOSS))


# --------------------------------------------------------------------------------------------
# Total, KL weight, gradients, errors
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_LOSSES)
def test_total_is_the_sum_of_the_parts(name):
    mu = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    parts = loss_of(name, mu=mu, kl_weight=0.3)
    assert close(parts.total, float(parts.numeric + parts.categorical + 0.3 * parts.kl))


@pytest.mark.parametrize("name", ALL_LOSSES)
def test_kl_weight_scales_only_the_kl(name):
    mu = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    off, on = loss_of(name, mu=mu, kl_weight=0.0), loss_of(name, mu=mu, kl_weight=2.0)
    assert close(on.kl, float(off.kl))
    assert close(on.total - off.total, 2.0 * float(off.kl))


def test_default_kl_weight_is_beta():
    mu = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    assert close(loss_of(MIXED_LOSS, mu=mu).total, float(loss_of(MIXED_LOSS, mu=mu, kl_weight=BETA).total))


@pytest.mark.parametrize("name", ALL_LOSSES)
def test_gradients_flow(name):
    logits = torch.randn(X.shape, generator=torch.Generator().manual_seed(6)).requires_grad_()
    mu, logvar = MU0.clone().requires_grad_(), LOGVAR0.clone().requires_grad_()
    kwargs = {"log_sigma": log_sigma(-1.0).requires_grad_()} if name == MIXED_SIGMA_LOSS else {}
    vae_loss(logits, X, mu + 0.1, logvar, LAYOUT, loss=name, **kwargs).total.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert mu.grad is not None and logvar.grad is not None
    if kwargs:
        assert kwargs["log_sigma"].grad.abs().sum() > 0


def test_parts_are_scalars():
    parts = loss_of(MIXED_LOSS)
    for part in (parts.total, parts.numeric, parts.categorical, parts.kl):
        assert part.ndim == 0


def test_unknown_loss():
    with pytest.raises(ValueError):
        vae_loss(ZERO_LOGITS, X, MU0, LOGVAR0, LAYOUT, loss="mse")


def test_x_and_logits_must_match():
    with pytest.raises(ValueError):
        vae_loss(ZERO_LOGITS, X[:, :4], MU0, LOGVAR0, LAYOUT, loss=MIXED_LOSS)


# --------------------------------------------------------------------------------------------
# KL warm-up and collapse diagnostic
# --------------------------------------------------------------------------------------------

def test_kl_schedule_warm_up():
    total = 1000
    warmup = round(KL_WARMUP_FRACTION * total)
    assert kl_schedule(0, total) == 0.0
    assert math.isclose(kl_schedule(warmup // 2, total), BETA * (warmup // 2) / warmup)
    assert kl_schedule(warmup, total) == BETA
    assert kl_schedule(total - 1, total) == BETA


def test_kl_schedule_is_monotonic():
    values = [kl_schedule(step, 200) for step in range(200)]
    assert values == sorted(values)


def test_kl_schedule_with_an_explicit_beta():
    assert kl_schedule(1000, 1000, beta=0.0) == 0.0            # sanity check: no KL at all
    assert kl_schedule(50, 1000, beta=0.5) == pytest.approx(0.5 * 50 / round(KL_WARMUP_FRACTION * 1000))


def test_kl_schedule_without_warm_up_steps():
    assert kl_schedule(0, 1) == BETA          # too few steps for a warm-up: full weight at once


def test_active_latent_dims():
    mu = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    logvar = torch.zeros_like(mu)
    logvar[:, 1] = -1.0                       # KL of dim 1: -1/2 (1 - 1 - e^-1) ~ 0.18 > threshold
    assert ACTIVE_DIM_KL < 0.18
    assert active_latent_dims(kl_per_dim(mu, logvar)) == 2
    assert active_latent_dims(kl_per_dim(torch.zeros(4, 3), torch.zeros(4, 3))) == 0
