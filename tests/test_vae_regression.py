"""Regression test of the VAE output layout and losses on the real contracts of the sample datasets.

The contract is the boundary between the Data Engine and the VAE: this module checks that every real
contract (after a JSON round trip, as the API will store it) gives a layout of the right width, that the
activated outputs decode into tables with the contract's columns, and that every loss is finite.
Uses the shared fixtures of ``conftest.py`` (skipped when the sample CSVs are missing).
"""

import json

import pytest

from sds.data_engine import SchemaContract, UserChoices, apply_choices, build_contract, decode

torch = pytest.importorskip("torch")

from sds.vae.layout import (  # noqa: E402  (after the torch check)
    BASELINE_LOSS,
    LOSS_NAMES,
    MIXED_LOSS,
    MIXED_SIGMA_LOSS,
    OutputLayout,
    activate,
)
from sds.vae.losses import vae_loss  # noqa: E402

N_ROWS = 50
N_ENCODED_ROWS = 100
LATENT_DIM = 2
SEED = 0


def test_vae_layout_on_real_contracts(inspections, preparations, encoders):
    generator = torch.Generator().manual_seed(SEED)
    for ds, inspection in inspections.items():
        schema = apply_choices(inspection, UserChoices())
        contract = build_contract(inspection, schema, preparations[ds], encoders[ds])
        layout = OutputLayout.from_contract(SchemaContract.from_dict(json.loads(json.dumps(contract.to_dict()))))
        assert layout.width == encoders[ds].width, ds
        assert len(layout.numeric_index) + len(layout.categorical_slices) == len(contract.columns), ds
        for group in layout.categorical_slices:          # encoded rows: one 1 per group
            block = encoders[ds].transform(preparations[ds].train.head(N_ENCODED_ROWS))[:, group]
            assert (block.sum(axis=1) == 1).all(), ds
        logits = torch.randn(N_ROWS, layout.width, generator=generator)
        for loss in (BASELINE_LOSS, MIXED_LOSS):
            table = decode(activate(logits, layout, loss=loss).numpy(), contract)
            assert list(table.columns) == [c.name for c in contract.columns], (ds, loss)
            assert len(table) == N_ROWS, (ds, loss)
        x = torch.from_numpy(encoders[ds].transform(preparations[ds].train.head(N_ROWS)))
        latent = torch.zeros(N_ROWS, LATENT_DIM)
        for loss in LOSS_NAMES:
            log_sigma = torch.zeros(len(layout.numeric_index)) if loss == MIXED_SIGMA_LOSS else None
            parts = vae_loss(logits, x, latent, latent, layout, loss=loss, log_sigma=log_sigma)
            assert torch.isfinite(parts.total), (ds, loss)
