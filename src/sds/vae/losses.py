"""Losses of the tabular VAE: three reconstruction variants on the same network, plus the KL term.

Every term is a sum over the features of a row, averaged over the batch. The KL term is per row
(summed over the latent dimensions), so the reconstruction must be too: the default mean of
``F.mse_loss`` would divide it by the vector width and let the KL win (latent collapse).

- ``baseline`` (the assignment): MSE after a sigmoid on every position, one-hot groups included.
- ``mixed``: MSE after a sigmoid on numeric positions + cross-entropy on the logits of every group.
- ``mixed_sigma``: Gaussian NLL with a learned sigma per numeric column + cross-entropy per group.
  The best sigma is the column's root mean squared error, so numeric columns and groups balance
  themselves; sigma is clamped (see ``hyperparameters_config``). The constant 1/2 log(2 pi) is left out.

Loss *values* are not comparable across variants (``mixed_sigma`` can be negative): compare the
synthetic data they produce instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sds.vae.hyperparameters_config import ACTIVE_DIM_KL, BETA, KL_WARMUP_FRACTION, MAX_SIGMA, MIN_SIGMA
from sds.vae.layout import BASELINE_LOSS, LOSS_NAMES, MIXED_SIGMA_LOSS, OutputLayout

LOG_MIN_SIGMA = math.log(MIN_SIGMA)
LOG_MAX_SIGMA = math.log(MAX_SIGMA)


@dataclass(frozen=True)
class LossParts:
    """One loss evaluation and its components (scalar tensors, averaged over the batch).

    Attributes:
        total: What the optimiser minimises: ``numeric + categorical + kl_weight * kl``.
        numeric: Reconstruction of the numeric and date columns.
        categorical: Reconstruction of the one-hot groups.
        kl: KL divergence from the prior, unweighted.
    """

    total: torch.Tensor
    numeric: torch.Tensor
    categorical: torch.Tensor
    kl: torch.Tensor


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) of every latent dimension, averaged over the rows (closed form)."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)), summed over the latent dimensions and averaged over the rows."""
    return kl_per_dim(mu, logvar).sum()


def warmup_steps(total_steps: int) -> int:
    """Number of steps of the KL warm-up: ``KL_WARMUP_FRACTION`` of ``total_steps``."""
    return round(KL_WARMUP_FRACTION * total_steps)


def kl_schedule(step: int, total_steps: int, *, beta: float = BETA) -> float:
    """Weight of the KL term at ``step``: linear warm-up from 0 to ``beta``, then ``beta``.

    With too few steps for a warm-up the weight is ``beta`` from the start. A warm-up lets the decoder
    learn to use the latent code before the KL pulls every row towards the prior.

    Args:
        step: Steps already done.
        total_steps: Steps of the whole training.
        beta: Final weight (``BETA``; 0 for the sanity check, where the network must memorise).
    """
    steps = warmup_steps(total_steps)
    if steps == 0:
        return beta
    return beta * min(1.0, step / steps)


def active_latent_dims(kl_dims: torch.Tensor) -> int:
    """Latent dimensions whose mean KL (``kl_per_dim``) is above ``ACTIVE_DIM_KL``: the others carry no information."""
    return int((kl_dims > ACTIVE_DIM_KL).sum())


def check_log_sigma(log_sigma: torch.Tensor | None, layout: OutputLayout, loss: str) -> None:
    """``log_sigma`` must come with ``mixed_sigma`` (one value per numeric column) and only with it."""
    if loss != MIXED_SIGMA_LOSS:
        if log_sigma is not None:
            raise ValueError(f"log_sigma is only used by the {MIXED_SIGMA_LOSS!r} loss, not by {loss!r}.")
        return
    if log_sigma is None:
        raise ValueError(f"The {MIXED_SIGMA_LOSS!r} loss needs log_sigma (one value per numeric column).")
    expected = (len(layout.numeric_index),)
    if tuple(log_sigma.shape) != expected:
        raise ValueError(f"log_sigma has shape {tuple(log_sigma.shape)}; expected {expected}.")


def group_cross_entropy(logits: torch.Tensor, x: torch.Tensor, layout: OutputLayout) -> torch.Tensor:
    """Cross-entropy of every one-hot group on its own logits, summed over groups, mean over rows."""
    total = logits.new_zeros(())
    for group in layout.categorical_slices:
        total = total + F.cross_entropy(logits[:, group], x[:, group].argmax(dim=1), reduction="sum")
    return total / len(x)


def vae_loss(logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             layout: OutputLayout, *, loss: str, kl_weight: float = BETA,
             log_sigma: torch.Tensor | None = None) -> LossParts:
    """Negative ELBO of a batch, with the reconstruction of ``loss``.

    Args:
        logits: Raw decoder output, shape ``(n_rows, layout.width)``.
        x: The encoded rows, same shape (numeric in [0, 1], one-hot groups).
        mu: Mean of q(z|x), shape ``(n_rows, latent_dim)``.
        logvar: Log-variance of q(z|x), same shape.
        layout: Layout of the contract.
        loss: One of ``LOSS_NAMES``.
        kl_weight: Weight of the KL term (``kl_schedule`` during training).
        log_sigma: Learned log sigma of every numeric column; ``mixed_sigma`` only.

    Returns:
        The total and its components.

    Raises:
        ValueError: unknown ``loss``, ``x`` and ``logits`` of different shapes, or ``log_sigma``
            missing, unexpected or of the wrong shape.
    """
    if loss not in LOSS_NAMES:
        raise ValueError(f"Unknown loss {loss!r}; expected one of {LOSS_NAMES}.")
    if x.shape != logits.shape or logits.ndim != 2 or logits.shape[1] != layout.width:
        raise ValueError(f"logits {tuple(logits.shape)} and x {tuple(x.shape)} must both be (n_rows, {layout.width}).")
    check_log_sigma(log_sigma, layout, loss)

    n_rows = len(x)
    numeric_index = layout.numeric_index
    if loss == BASELINE_LOSS:
        squared = (torch.sigmoid(logits) - x).pow(2)
        numeric = squared[:, numeric_index].sum() / n_rows
        categorical = sum((squared[:, g].sum() for g in layout.categorical_slices), logits.new_zeros(())) / n_rows
    else:
        squared = (torch.sigmoid(logits[:, numeric_index]) - x[:, numeric_index]).pow(2)
        if loss == MIXED_SIGMA_LOSS:
            log_sigma = log_sigma.clamp(LOG_MIN_SIGMA, LOG_MAX_SIGMA)
            squared = squared / (2 * torch.exp(2 * log_sigma)) + log_sigma
        numeric = squared.sum() / n_rows
        categorical = group_cross_entropy(logits, x, layout)

    kl = kl_divergence(mu, logvar)
    return LossParts(total=numeric + categorical + kl_weight * kl, numeric=numeric, categorical=categorical, kl=kl)
