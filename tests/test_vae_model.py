"""Specification tests for the VAE network and the choice of its size.

The size follows the width W of the encoded vector, not the number of rows. Public API under test:
    Architecture(hidden_dims, latent_dim)
    choose_architecture(width) -> Architecture
        2 hidden layers of clamp(2 W, MIN_HIDDEN, MAX_HIDDEN), latent clamp(round(W / 4), MIN_LATENT, MAX_LATENT)
    TabularVae(width, architecture, *, n_sigma=None)
        encode(x) -> (mu, logvar); decode(z) -> logits; forward(x) -> (logits, mu, logvar)
        train mode: z = mu + sigma * eps; eval mode: z = mu (deterministic)
        log_sigma: nn.Parameter of n_sigma values starting at log(INITIAL_SIGMA), or None
        dense layers + ReLU only: no dropout, no batch norm
    build_vae(layout, *, loss, architecture=None) -> TabularVae (log_sigma only for "mixed_sigma")
"""

import math

import pytest
import torch
from torch import nn

from sds.data_engine import SchemaContract
from sds.data_engine.contract import ContractColumn
from sds.vae.hyperparameters_config import INITIAL_SIGMA, MAX_HIDDEN, MAX_LATENT, MIN_HIDDEN, MIN_LATENT
from sds.vae.layout import BASELINE_LOSS, MIXED_LOSS, MIXED_SIGMA_LOSS, OutputLayout
from sds.vae.model import Architecture, TabularVae, build_vae, choose_architecture

N_ROWS = 16
ARCHITECTURE = Architecture(hidden_dims=(64, 64), latent_dim=8)
WIDTH = 9

LAYOUT = OutputLayout.from_contract(SchemaContract(version=1, width=WIDTH, columns=[
    ContractColumn("age", "numeric", 0, 1, min=0.0, max=1.0),
    ContractColumn("sex", "categorical", 1, 3, levels=("F", "M")),
    ContractColumn("bmi", "numeric", 3, 4, min=0.0, max=1.0),
    ContractColumn("visit", "date", 4, 5, min=0.0, max=1.0, origin="2020-01-01"),
    ContractColumn("grade", "categorical", 5, 9, levels=("a", "b", "c", "d")),
], dropped=[], rows={}, issues=[]))


def batch(width: int = WIDTH) -> torch.Tensor:
    return torch.rand(N_ROWS, width, generator=torch.Generator().manual_seed(0))


# --------------------------------------------------------------------------------------------
# Size from the width
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("width, hidden, latent", [
    (21, 64, 8),        # Pima
    (32, 64, 8),        # Breast Cancer
    (53, 106, 13),      # Bank Marketing
    (65, 130, 16),      # CDC Diabetes
    (119, 238, 30),     # Adult
    (151, 302, 32),     # Diabetes 130
])
def test_architecture_on_the_benchmark_widths(width, hidden, latent):
    assert choose_architecture(width) == Architecture((hidden, hidden), latent)


def test_architecture_is_clamped():
    small, large = choose_architecture(2), choose_architecture(5_000)
    assert small == Architecture((MIN_HIDDEN, MIN_HIDDEN), MIN_LATENT)
    assert large == Architecture((MAX_HIDDEN, MAX_HIDDEN), MAX_LATENT)


# --------------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------------

def test_output_shapes():
    model = TabularVae(WIDTH, ARCHITECTURE)
    logits, mu, logvar = model(batch())
    assert logits.shape == (N_ROWS, WIDTH)
    assert mu.shape == logvar.shape == (N_ROWS, ARCHITECTURE.latent_dim)
    assert model.decode(torch.zeros(3, ARCHITECTURE.latent_dim)).shape == (3, WIDTH)


def test_dense_layers_follow_the_architecture():
    model = TabularVae(WIDTH, ARCHITECTURE)
    sizes = sorted((m.in_features, m.out_features) for m in model.modules() if isinstance(m, nn.Linear))
    expected = sorted([
        (WIDTH, 64), (64, 64), (64, 8), (64, 8),        # encoder + heads for mu and logvar
        (8, 64), (64, 64), (64, WIDTH),                 # decoder, mirrored
    ])
    assert sizes == expected


def test_every_hidden_layer_has_a_relu():
    model = TabularVae(WIDTH, ARCHITECTURE)
    n_hidden = 2 * len(ARCHITECTURE.hidden_dims)               # encoder + decoder; heads and output stay linear
    assert sum(isinstance(m, nn.ReLU) for m in model.modules()) == n_hidden


def test_no_dropout_and_no_batch_norm():
    model = TabularVae(WIDTH, ARCHITECTURE)
    banned = (nn.Dropout, nn.BatchNorm1d, nn.LayerNorm)
    assert not any(isinstance(m, banned) for m in model.modules())


def test_eval_mode_is_deterministic():
    model = TabularVae(WIDTH, ARCHITECTURE).eval()
    x = batch()
    first, mu, _ = model(x)
    assert torch.equal(first, model(x)[0])
    assert torch.allclose(first, model.decode(mu))          # eval: z = mu


def test_train_mode_samples_the_latent_code():
    torch.manual_seed(0)
    model = TabularVae(WIDTH, ARCHITECTURE).train()
    x = batch()
    assert not torch.equal(model(x)[0], model(x)[0])


def test_no_sigma_by_default():
    model = TabularVae(WIDTH, ARCHITECTURE)
    assert model.log_sigma is None


def test_sigma_is_a_learned_parameter():
    model = TabularVae(WIDTH, ARCHITECTURE, n_sigma=3)
    assert isinstance(model.log_sigma, nn.Parameter)
    assert model.log_sigma.shape == (3,)
    assert torch.allclose(model.log_sigma, torch.full((3,), math.log(INITIAL_SIGMA)))
    assert any(p is model.log_sigma for p in model.parameters())


def test_gradients_reach_every_parameter():
    model = TabularVae(WIDTH, ARCHITECTURE, n_sigma=3).train()
    logits, mu, logvar = model(batch())
    (logits.sum() + mu.sum() + logvar.sum() + model.log_sigma.sum()).backward()
    assert all(p.grad is not None for p in model.parameters())


# --------------------------------------------------------------------------------------------
# Model for a layout and a loss
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("loss", [BASELINE_LOSS, MIXED_LOSS])
def test_build_without_sigma(loss):
    model = build_vae(LAYOUT, loss=loss)
    assert model.log_sigma is None
    assert model.width == WIDTH
    assert model.architecture == choose_architecture(WIDTH)


def test_build_with_a_given_architecture():
    assert build_vae(LAYOUT, loss=MIXED_LOSS, architecture=ARCHITECTURE).architecture == ARCHITECTURE


def test_build_with_one_sigma_per_numeric_column():
    model = build_vae(LAYOUT, loss=MIXED_SIGMA_LOSS)
    assert model.log_sigma.shape == (len(LAYOUT.numeric_index),)      # age, bmi, visit


def test_build_unknown_loss():
    with pytest.raises(ValueError):
        build_vae(LAYOUT, loss="mse")
