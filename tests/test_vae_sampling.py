"""Specification tests for sampling synthetic rows from a trained VAE.

z ~ N(0, I) -> decoder -> activation of the loss the model was trained with -> decode. Batched like
training, with its own seeded generator. Public API under test:
    sample(trained, n_rows, *, seed=RANDOM_STATE, numeric_noise=None, batch_size=BATCH_SIZE) -> np.ndarray
        shape (n_rows, width); numeric_noise=None -> noise x_hat + sigma * eps only for "mixed_sigma";
        numeric_noise=True without a learned sigma -> ValueError; n_rows <= 0 -> ValueError;
        the model is left in eval mode
    generate(trained, contract, n_rows, *, seed=RANDOM_STATE, numeric_noise=None) -> pd.DataFrame
        decode(sample(...), contract, seed=seed) when every row is coherent; rows still pairing categorical
        values never seen in train (decode_checked) are replaced by new rows, for at most
        MAX_GENERATION_ROUNDS rounds; always exactly n_rows rows
"""

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from sds.data_engine import SchemaContract, decode
from sds.data_engine.contract import ContractColumn, SeenPairs
from sds.vae.hyperparameters_config import MAX_SIGMA, MIN_SIGMA
from sds.vae.layout import BASELINE_LOSS, MIXED_LOSS, MIXED_SIGMA_LOSS, OutputLayout
from sds.vae.model import Architecture, TabularVae
from sds.vae.sampling import generate, sample
from sds.vae.training import TrainedVae, train_vae

CPU = torch.device("cpu")
SMALL = Architecture(hidden_dims=(32, 32), latent_dim=4)
N_TRAIN = 600
N_SAMPLES = 4000
SHARE_F = 0.9                    # share of "F" in the toy data
SHARE_TOLERANCE = 0.05
SUM_TOLERANCE = 1e-5

CONTRACT = SchemaContract(version=1, width=7, columns=[
    ContractColumn("age", "numeric", 0, 1, min=17.0, max=90.0, integer=True),
    ContractColumn("sex", "categorical", 1, 3, levels=("F", "M")),
    ContractColumn("bmi", "numeric", 3, 4, min=15.0, max=60.0, integer=False),
    ContractColumn("grade", "categorical", 4, 7, levels=("a", "b", "c")),
], dropped=[], rows={}, issues=[])
LAYOUT = OutputLayout.from_contract(CONTRACT)
NUMERIC = [0, 3]
GROUPS = [slice(1, 3), slice(4, 7)]


def toy_rows(n_rows: int = N_TRAIN, seed: int = 0) -> np.ndarray:
    """Encoded rows with a clear structure: 90% "F", age around 0.7, grade uniform."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_rows, LAYOUT.width), dtype=np.float32)
    x[:, 0] = np.clip(rng.normal(0.7, 0.1, n_rows), 0, 1)
    x[:, 3] = rng.random(n_rows)
    is_f = rng.random(n_rows) < SHARE_F
    x[is_f, 1], x[~is_f, 2] = 1, 1
    x[np.arange(n_rows), 4 + rng.integers(0, 3, n_rows)] = 1
    return x


def trained(loss: str, max_steps: int) -> TrainedVae:
    return train_vae(toy_rows(), LAYOUT, loss=loss, architecture=SMALL, max_steps=max_steps,
                     eval_every=100, patience=1_000, batch_size=64, device=CPU)


@pytest.fixture(scope="module")
def mixed() -> TrainedVae:
    return trained(MIXED_LOSS, max_steps=1500)


@pytest.fixture(scope="module")
def sigma() -> TrainedVae:
    return trained(MIXED_SIGMA_LOSS, max_steps=300)


@pytest.fixture(scope="module")
def baseline() -> TrainedVae:
    return trained(BASELINE_LOSS, max_steps=100)


def untrained(loss: str = MIXED_LOSS) -> TrainedVae:
    torch.manual_seed(0)
    return TrainedVae(model=TabularVae(LAYOUT.width, SMALL).eval(), layout=LAYOUT, loss=loss, history=[],
                      best_step=None, stopped_early=False, cancelled=False,
                      train_index=np.arange(N_TRAIN), validation_index=np.array([], dtype=int))


# --------------------------------------------------------------------------------------------
# Shape and values
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_rows, batch_size", [(1, 64), (1000, 64), (64, 64), (5, 256)])
def test_shape(mixed, n_rows, batch_size):
    vectors = sample(mixed, n_rows, batch_size=batch_size)
    assert vectors.shape == (n_rows, LAYOUT.width)
    assert vectors.dtype == np.float32


@pytest.mark.parametrize("name", ["mixed", "sigma"])
def test_groups_are_distributions(name, request):
    vectors = sample(request.getfixturevalue(name), 500, numeric_noise=False)
    for group in GROUPS:
        assert np.allclose(vectors[:, group].sum(axis=1), 1, atol=SUM_TOLERANCE)


@pytest.mark.parametrize("name", ["mixed", "sigma", "baseline"])
def test_values_without_noise_are_in_the_unit_interval(name, request):
    vectors = sample(request.getfixturevalue(name), 500, numeric_noise=False)
    assert (vectors >= 0).all() and (vectors <= 1).all()


def test_baseline_groups_are_not_constrained(baseline):
    sums = sample(baseline, 500)[:, GROUPS[1]].sum(axis=1)
    assert not np.allclose(sums, 1, atol=SUM_TOLERANCE)        # sigmoid scores: decode cleans them


# --------------------------------------------------------------------------------------------
# Seed
# --------------------------------------------------------------------------------------------

def test_same_seed_same_rows_whatever_the_global_state(mixed):
    torch.manual_seed(1)
    first = sample(mixed, 300, seed=7)
    torch.manual_seed(2)
    np.random.seed(2)
    assert np.array_equal(first, sample(mixed, 300, seed=7))


def test_different_seed_different_rows(mixed):
    assert not np.array_equal(sample(mixed, 300, seed=7), sample(mixed, 300, seed=8))


# --------------------------------------------------------------------------------------------
# Noise on numeric columns (mixed_sigma only)
# --------------------------------------------------------------------------------------------

def test_noise_is_on_by_default_for_mixed_sigma(sigma):
    assert np.array_equal(sample(sigma, 500, seed=3), sample(sigma, 500, seed=3, numeric_noise=True))
    assert not np.array_equal(sample(sigma, 500, seed=3), sample(sigma, 500, seed=3, numeric_noise=False))


def test_noise_widens_the_numeric_columns(sigma):
    noisy = sample(sigma, N_SAMPLES, seed=3, numeric_noise=True)
    plain = sample(sigma, N_SAMPLES, seed=3, numeric_noise=False)
    assert (noisy[:, NUMERIC].var(axis=0) > plain[:, NUMERIC].var(axis=0)).all()


def test_noise_leaves_the_groups_alone(sigma):
    noisy = sample(sigma, 500, seed=3, numeric_noise=True)
    plain = sample(sigma, 500, seed=3, numeric_noise=False)
    for group in GROUPS:
        assert np.allclose(noisy[:, group], plain[:, group])


@pytest.mark.parametrize("log_sigma, bound", [(100.0, MAX_SIGMA), (-100.0, MIN_SIGMA)])
def test_noise_uses_the_clamped_sigma(sigma, log_sigma, bound):
    extreme = copy.deepcopy(sigma)
    with torch.no_grad():
        extreme.model.log_sigma.fill_(log_sigma)
    noise = sample(extreme, N_SAMPLES, seed=3) - sample(extreme, N_SAMPLES, seed=3, numeric_noise=False)
    assert np.allclose(noise[:, NUMERIC].std(axis=0), bound, rtol=0.1)      # same bounds as the loss


@pytest.mark.parametrize("name", ["mixed", "baseline"])
def test_no_noise_without_a_learned_sigma(name, request):
    model = request.getfixturevalue(name)
    assert np.array_equal(sample(model, 200, seed=3), sample(model, 200, seed=3, numeric_noise=False))
    with pytest.raises(ValueError):
        sample(model, 200, numeric_noise=True)


# --------------------------------------------------------------------------------------------
# Errors and model state
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_rows", [0, -3])
def test_rows_must_be_positive(mixed, n_rows):
    with pytest.raises(ValueError):
        sample(mixed, n_rows)


def test_model_is_left_in_eval_mode(mixed):
    mixed.model.train()
    first = sample(mixed, 100, seed=5)
    assert not mixed.model.training
    assert np.array_equal(first, sample(mixed, 100, seed=5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_sampling_from_a_model_on_cuda():
    on_gpu = train_vae(toy_rows(), LAYOUT, loss=MIXED_SIGMA_LOSS, architecture=SMALL, max_steps=40,
                       eval_every=20, device=torch.device("cuda"))
    vectors = sample(on_gpu, 300)
    assert isinstance(vectors, np.ndarray) and vectors.shape == (300, LAYOUT.width)


# --------------------------------------------------------------------------------------------
# generate: the synthetic table
# --------------------------------------------------------------------------------------------

def test_generate_returns_the_contract_columns(mixed):
    table = generate(mixed, CONTRACT, 250, seed=4)
    assert list(table.columns) == ["age", "sex", "bmi", "grade"]
    assert len(table) == 250
    assert pd.api.types.is_integer_dtype(table["age"])
    assert pd.api.types.is_float_dtype(table["bmi"])
    assert set(table["sex"]) <= {"F", "M"} and set(table["grade"]) <= {"a", "b", "c"}
    assert table["age"].between(17, 90).all() and table["bmi"].between(15, 60).all()


def test_generate_decodes_with_the_same_seed(mixed):
    seed = 4
    expected = decode(sample(mixed, 300, seed=seed), CONTRACT, seed=seed)
    pd.testing.assert_frame_equal(generate(mixed, CONTRACT, 300, seed=seed), expected)


def test_generate_is_seeded(sigma):
    pd.testing.assert_frame_equal(generate(sigma, CONTRACT, 200, seed=4), generate(sigma, CONTRACT, 200, seed=4))
    assert not generate(sigma, CONTRACT, 200, seed=4).equals(generate(sigma, CONTRACT, 200, seed=5))


# --------------------------------------------------------------------------------------------
# generate: rows the coherence repair could not fix are regenerated
# --------------------------------------------------------------------------------------------
# "sex" slots (F, M), "grade" slots (a, b, c). With only (F, a) seen, a row (M, b) has no fix in either
# column (M meets no grade, b meets no sex): it can only be replaced. (M, a) and (F, b) are repaired.

ONLY_F_A = replace(CONTRACT, seen_pairs=[SeenPairs("sex", "grade", ((0, 0),))])
NOTHING_SEEN = replace(CONTRACT, seen_pairs=[SeenPairs("sex", "grade", ())])


def test_unfixable_rows_are_regenerated(mixed):
    table = generate(mixed, ONLY_F_A, 500, seed=4)
    assert len(table) == 500
    assert set(zip(table["sex"], table["grade"], strict=True)) == {("F", "a")}


def test_regeneration_is_seeded(mixed):
    pd.testing.assert_frame_equal(generate(mixed, ONLY_F_A, 300, seed=4), generate(mixed, ONLY_F_A, 300, seed=4))


def test_regeneration_gives_up_with_the_requested_rows(mixed):
    # No row can ever be coherent: after the last round the incoherent rows are kept, the size is right.
    table = generate(mixed, NOTHING_SEEN, 200, seed=4)
    assert len(table) == 200
    assert list(table.index) == list(range(200))


def test_a_coherent_table_is_not_regenerated(mixed):
    # Every row coherent: exactly decode(sample(...)), as without seen pairs.
    every = replace(CONTRACT, seen_pairs=[SeenPairs("sex", "grade", tuple((i, j) for i in range(2) for j in range(3)))])
    pd.testing.assert_frame_equal(generate(mixed, every, 300, seed=4), generate(mixed, CONTRACT, 300, seed=4))


# --------------------------------------------------------------------------------------------
# End to end: the trained model learned something an untrained one did not
# --------------------------------------------------------------------------------------------

def test_trained_model_reproduces_the_share_of_f(mixed):
    share = (generate(mixed, CONTRACT, N_SAMPLES, seed=6)["sex"] == "F").mean()
    assert abs(share - SHARE_F) < SHARE_TOLERANCE


def test_untrained_model_does_not():
    share = (generate(untrained(), CONTRACT, N_SAMPLES, seed=6)["sex"] == "F").mean()
    assert abs(share - SHARE_F) > 2 * SHARE_TOLERANCE


def test_trained_model_reproduces_the_mean_age(mixed):
    real_mean = 17 + 0.7 * (90 - 17)
    synthetic = generate(mixed, CONTRACT, N_SAMPLES, seed=6)["age"].mean()
    assert abs(synthetic - real_mean) < 0.05 * (90 - 17)
