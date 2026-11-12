"""Tests for the training runtime helpers: seeding and device selection."""

import torch

from sds.vae.runtime import get_device, set_seed


def test_set_seed_is_reproducible():
    set_seed(123)
    first = torch.rand(3)
    set_seed(123)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_get_device_returns_torch_device():
    assert isinstance(get_device(), torch.device)
