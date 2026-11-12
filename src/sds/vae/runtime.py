"""Runtime helpers for training and sampling: reproducible seeds and device selection."""

import random

import numpy as np
import torch

from sds.config import RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Args:
        seed: Seed value shared by all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Return the CUDA device when available, otherwise the CPU.

    Returns:
        The torch device to use for training and sampling.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
