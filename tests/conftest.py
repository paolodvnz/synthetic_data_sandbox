"""Shared fixtures: the Data Engine run once on the six sample datasets.

Used by the regression tests of every block (``test_data_engine_regression.py``,
``test_vae_regression.py``). They are session-scoped because ``inspect_csv`` on all files takes ~11 s.
Tests that use them are skipped when the CSVs are missing (download them with ``sds-download-samples``).
"""

from pathlib import Path

import pytest

from sds.data_engine import UserChoices, apply_choices, fit_encoder, inspect_csv, prepare
from sds.samples import SAMPLE_NAMES, data_dir, missing_samples


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    """Folder holding all sample CSVs; skips the requesting test when any is missing."""
    folder = data_dir()
    if missing := missing_samples(SAMPLE_NAMES, folder):
        pytest.skip(f"sample CSVs not found in {folder} ({', '.join(missing)}): run sds-download-samples")
    return folder


@pytest.fixture(scope="session")
def inspections(samples_dir) -> dict:
    """``inspect_csv`` on every sample dataset."""
    return {ds: inspect_csv(samples_dir / f"{ds}.csv") for ds in SAMPLE_NAMES}


@pytest.fixture(scope="session")
def preparations(inspections) -> dict:
    """``prepare`` with the default choices on every sample dataset."""
    return {ds: prepare(inspection, apply_choices(inspection, UserChoices()))
            for ds, inspection in inspections.items()}


@pytest.fixture(scope="session")
def encoders(inspections, preparations) -> dict:
    """Encoder fitted on the train split of every sample dataset, with the default choices."""
    return {ds: fit_encoder(p.train, apply_choices(inspections[ds], UserChoices()))
            for ds, p in preparations.items()}
