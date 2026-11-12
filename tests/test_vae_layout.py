"""Specification tests for the output layout and the activation of the VAE outputs.

The VAE reads the index map only from the schema contract. The model returns raw logits; the
activation is tied to the loss the model was trained with. Public API under test:
    OutputLayout.from_contract(contract) -> OutputLayout(width, numeric_index, categorical_slices)
        numeric_index: positions of numeric and date columns (one each), as a long tensor
        categorical_slices: one slice(start, stop) per one-hot group, in contract order
        .to(device) -> OutputLayout on that device
    Invalid contract (gap, overlap, wrong width, group with fewer than 2 slots, unknown role)
        -> VaeError("invalid_contract").
    activate(logits, layout, *, loss) -> torch.Tensor
        loss="baseline": sigmoid on every position (one-hot groups included)
        loss="mixed":    sigmoid on numeric_index, softmax on every categorical group
    Logits of the wrong width -> VaeError("wrong_width"); unknown loss -> ValueError.
"""

import pandas as pd
import pytest
import torch

from sds.data_engine import SchemaContract, decode
from sds.data_engine.contract import ContractColumn
from sds.vae.layout import BASELINE_LOSS, MIXED_LOSS, OutputLayout, VaeError, activate

N_ROWS = 64
LOGIT_SCALE = 5.0           # wide logits, so sigmoid and softmax are not trivially near 0.5
SUM_TOLERANCE = 1e-5


def make_contract(columns: list[ContractColumn], width: int) -> SchemaContract:
    return SchemaContract(version=1, width=width, columns=columns, dropped=[], rows={}, issues=[])


def handmade_contract() -> SchemaContract:
    """Mixed contract: 2 numeric, 1 date, a 2-slot and a 4-slot categorical group, interleaved."""
    columns = [
        ContractColumn("age", "numeric", 0, 1, min=17.0, max=90.0, integer=True),
        ContractColumn("sex", "categorical", 1, 3, levels=("F", "M")),
        ContractColumn("bmi", "numeric", 3, 4, min=15.0, max=60.0, integer=False),
        ContractColumn("visit", "date", 4, 5, min=0.0, max=365.0, origin="2020-01-01"),
        ContractColumn("grade", "categorical", 5, 9, integer=True, levels=(1, 2, 3), has_other=True,
                       merged=((4, 5), (5, 3))),
    ]
    return make_contract(columns, width=9)


CONTRACT = handmade_contract()
NUMERIC_POSITIONS = [0, 3, 4]
GROUP_SLICES = [slice(1, 3), slice(5, 9)]


def random_logits(width: int = CONTRACT.width, n_rows: int = N_ROWS) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(n_rows, width, generator=generator) * LOGIT_SCALE


# --------------------------------------------------------------------------------------------
# Layout from the contract
# --------------------------------------------------------------------------------------------

def test_layout_positions():
    layout = OutputLayout.from_contract(CONTRACT)
    assert layout.width == CONTRACT.width
    assert layout.numeric_index.dtype == torch.long
    assert layout.numeric_index.tolist() == NUMERIC_POSITIONS
    assert list(layout.categorical_slices) == GROUP_SLICES


def test_layout_without_categorical_columns():
    contract = make_contract([ContractColumn("x", "numeric", 0, 1, min=0.0, max=1.0),
                              ContractColumn("y", "numeric", 1, 2, min=0.0, max=1.0)], width=2)
    layout = OutputLayout.from_contract(contract)
    assert layout.numeric_index.tolist() == [0, 1]
    assert list(layout.categorical_slices) == []


def test_layout_without_numeric_columns():
    contract = make_contract([ContractColumn("c", "categorical", 0, 3, levels=("a", "b", "c"))], width=3)
    layout = OutputLayout.from_contract(contract)
    assert layout.numeric_index.tolist() == []
    assert list(layout.categorical_slices) == [slice(0, 3)]


@pytest.mark.parametrize("columns, width", [
    pytest.param([ContractColumn("x", "numeric", 0, 1), ContractColumn("y", "numeric", 2, 3)], 3, id="gap"),
    pytest.param([ContractColumn("c", "categorical", 0, 3, levels=("a", "b", "c")),
                  ContractColumn("y", "numeric", 2, 3)], 3, id="overlap"),
    pytest.param([ContractColumn("x", "numeric", 0, 1)], 2, id="width_too_large"),
    pytest.param([ContractColumn("x", "numeric", 0, 1), ContractColumn("y", "numeric", 1, 2)], 1,
                 id="width_too_small"),
    pytest.param([ContractColumn("x", "numeric", 1, 2), ContractColumn("y", "numeric", 0, 1)], 2,
                 id="out_of_order"),
    pytest.param([ContractColumn("c", "categorical", 0, 1, levels=("a",))], 1, id="one_slot_group"),
    pytest.param([ContractColumn("x", "numeric", 0, 2)], 2, id="two_slot_numeric"),
    pytest.param([ContractColumn("t", "text", 0, 1)], 1, id="unknown_role"),
])
def test_invalid_contract(columns, width):
    with pytest.raises(VaeError) as error:
        OutputLayout.from_contract(make_contract(columns, width))
    assert error.value.code == "invalid_contract"


# --------------------------------------------------------------------------------------------
# Activation
# --------------------------------------------------------------------------------------------

def test_mixed_activation_groups_are_distributions():
    layout = OutputLayout.from_contract(CONTRACT)
    out = activate(random_logits(), layout, loss=MIXED_LOSS)
    assert out.shape == (N_ROWS, CONTRACT.width)
    for group in GROUP_SLICES:
        block = out[:, group]
        assert (block >= 0).all()
        assert torch.allclose(block.sum(dim=1), torch.ones(N_ROWS), atol=SUM_TOLERANCE)


def test_mixed_activation_numeric_is_sigmoid():
    layout = OutputLayout.from_contract(CONTRACT)
    logits = random_logits()
    out = activate(logits, layout, loss=MIXED_LOSS)
    assert torch.allclose(out[:, NUMERIC_POSITIONS], torch.sigmoid(logits[:, NUMERIC_POSITIONS]))


def test_mixed_activation_group_is_softmax_of_its_own_logits():
    layout = OutputLayout.from_contract(CONTRACT)
    logits = random_logits()
    out = activate(logits, layout, loss=MIXED_LOSS)
    for group in GROUP_SLICES:
        assert torch.allclose(out[:, group], torch.softmax(logits[:, group], dim=1))


def test_baseline_activation_is_sigmoid_everywhere():
    layout = OutputLayout.from_contract(CONTRACT)
    logits = random_logits()
    out = activate(logits, layout, loss=BASELINE_LOSS)
    assert torch.allclose(out, torch.sigmoid(logits))
    group_sums = out[:, GROUP_SLICES[1]].sum(dim=1)
    assert not torch.allclose(group_sums, torch.ones(N_ROWS), atol=SUM_TOLERANCE)   # not constrained


@pytest.mark.parametrize("loss", [BASELINE_LOSS, MIXED_LOSS])
def test_activation_does_not_modify_logits(loss):
    layout = OutputLayout.from_contract(CONTRACT)
    logits = random_logits()
    before = logits.clone()
    activate(logits, layout, loss=loss)
    assert torch.equal(logits, before)


@pytest.mark.parametrize("loss", [BASELINE_LOSS, MIXED_LOSS])
def test_activation_keeps_gradient(loss):
    layout = OutputLayout.from_contract(CONTRACT)
    logits = random_logits().requires_grad_()
    activate(logits, layout, loss=loss).sum().backward()
    assert logits.grad is not None


def test_wrong_logit_width():
    layout = OutputLayout.from_contract(CONTRACT)
    with pytest.raises(VaeError) as error:
        activate(random_logits(width=CONTRACT.width + 1), layout, loss=MIXED_LOSS)
    assert error.value.code == "wrong_width"


def test_unknown_loss():
    layout = OutputLayout.from_contract(CONTRACT)
    with pytest.raises(ValueError):
        activate(random_logits(), layout, loss="mse")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("loss", [BASELINE_LOSS, MIXED_LOSS])
def test_activation_on_cuda(loss):
    device = torch.device("cuda")
    layout = OutputLayout.from_contract(CONTRACT).to(device)
    assert layout.numeric_index.device.type == "cuda"
    out = activate(random_logits().to(device), layout, loss=loss)
    assert out.device.type == "cuda"
    expected = activate(random_logits(), OutputLayout.from_contract(CONTRACT), loss=loss)
    assert torch.allclose(out.cpu(), expected, atol=SUM_TOLERANCE)


# --------------------------------------------------------------------------------------------
# Bridge with the Data Engine: the activated output is what decode() expects
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("loss", [BASELINE_LOSS, MIXED_LOSS])
def test_decode_of_activated_logits(loss):
    layout = OutputLayout.from_contract(CONTRACT)
    out = activate(random_logits(), layout, loss=loss)
    table = decode(out.detach().cpu().numpy(), CONTRACT)
    assert list(table.columns) == ["age", "sex", "bmi", "visit", "grade"]
    assert len(table) == N_ROWS
    assert pd.api.types.is_integer_dtype(table["age"])
    assert pd.api.types.is_string_dtype(table["sex"])
    assert pd.api.types.is_float_dtype(table["bmi"])
    assert pd.api.types.is_datetime64_any_dtype(table["visit"])
    assert pd.api.types.is_integer_dtype(table["grade"])
    assert set(table["sex"]) <= {"F", "M"}
    assert set(table["grade"]) <= {1, 2, 3, 4, 5}
