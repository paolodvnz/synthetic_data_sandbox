"""The VAE network and the choice of its size.

Dense layers with ReLU: encoder x -> h -> (mu, log sigma^2), reparameterization, mirrored decoder
z -> h -> logits. The decoder returns raw logits; ``layout.activate`` turns them into decoder-ready
values with the activation of the loss. No dropout (in a VAE it adds noise to the reconstruction)
and no batch norm (medical datasets are often small).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import nn

from sds.vae.hyperparameters_config import (
    FEATURES_PER_LATENT,
    HIDDEN_PER_FEATURE,
    INITIAL_SIGMA,
    MAX_HIDDEN,
    MAX_LATENT,
    MIN_HIDDEN,
    MIN_LATENT,
    N_HIDDEN_LAYERS,
)
from sds.vae.layout import LOSS_NAMES, MIXED_SIGMA_LOSS, OutputLayout


@dataclass(frozen=True)
class Architecture:
    """Sizes of the network.

    Attributes:
        hidden_dims: Widths of the encoder's hidden layers (the decoder uses them reversed).
        latent_dim: Size of the latent code z.
    """

    hidden_dims: tuple[int, ...]
    latent_dim: int


def clamp(value: int, low: int, high: int) -> int:
    """``value`` limited to ``[low, high]``."""
    return max(low, min(high, value))


def choose_architecture(width: int) -> Architecture:
    """Network sizes for a dataset: they follow the width of the encoded vector, not the number of rows.

    Hidden layers: ``HIDDEN_PER_FEATURE * width``; latent code: ``width / FEATURES_PER_LATENT``, so the
    bottleneck stays narrower than the input. Both within named bounds. Memorisation on small datasets is
    handled by the training length (in epochs) and early stopping, not by a smaller network.

    Args:
        width: Length of the encoded vector.

    Returns:
        The architecture.
    """
    hidden = clamp(HIDDEN_PER_FEATURE * width, MIN_HIDDEN, MAX_HIDDEN)
    latent = clamp(round(width / FEATURES_PER_LATENT), MIN_LATENT, MAX_LATENT)
    return Architecture(hidden_dims=(hidden,) * N_HIDDEN_LAYERS, latent_dim=latent)


def dense_stack(sizes: tuple[int, ...]) -> nn.Sequential:
    """Linear + ReLU for every consecutive pair of ``sizes``."""
    layers: list[nn.Module] = []
    for n_in, n_out in pairwise(sizes):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
    return nn.Sequential(*layers)


class TabularVae(nn.Module):
    """Variational autoencoder for encoded tabular rows.

    Attributes:
        width: Length of the encoded vector (input and output).
        architecture: Sizes of the network.
        log_sigma: Learned log sigma of every numeric column (``mixed_sigma`` loss), else ``None``.
    """

    def __init__(self, width: int, architecture: Architecture, *, n_sigma: int | None = None) -> None:
        """Build the network.

        Args:
            width: Length of the encoded vector.
            architecture: Sizes of the network.
            n_sigma: Number of learned sigmas (one per numeric column), or ``None`` for none.
        """
        super().__init__()
        self.width = width
        self.architecture = architecture
        hidden, latent = architecture.hidden_dims, architecture.latent_dim
        self.encoder = dense_stack((width, *hidden))
        self.mu_head = nn.Linear(hidden[-1], latent)
        self.logvar_head = nn.Linear(hidden[-1], latent)
        self.decoder = nn.Sequential(dense_stack((latent, *reversed(hidden))), nn.Linear(hidden[0], width))
        if n_sigma is None:
            self.register_parameter("log_sigma", None)
        else:
            self.log_sigma = nn.Parameter(torch.full((n_sigma,), math.log(INITIAL_SIGMA)))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log-variance of q(z|x)."""
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Raw logits of the encoded vector for latent codes ``z``."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, sample z and decode.

        In training mode z = mu + sigma * eps (reparameterization trick: the randomness is in eps, so
        gradients flow through mu and sigma); in eval mode z = mu, a deterministic estimate.

        Returns:
            ``(logits, mu, logvar)``.
        """
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.decode(z), mu, logvar


def build_vae(layout: OutputLayout, *, loss: str, architecture: Architecture | None = None) -> TabularVae:
    """The network for a contract layout and a loss (``log_sigma`` only for ``mixed_sigma``).

    Args:
        layout: Layout of the contract.
        loss: One of ``LOSS_NAMES``.
        architecture: Network sizes; default ``choose_architecture(layout.width)``.

    Raises:
        ValueError: when ``loss`` is not one of ``LOSS_NAMES``.
    """
    if loss not in LOSS_NAMES:
        raise ValueError(f"Unknown loss {loss!r}; expected one of {LOSS_NAMES}.")
    n_sigma = len(layout.numeric_index) if loss == MIXED_SIGMA_LOSS else None
    return TabularVae(layout.width, architecture or choose_architecture(layout.width), n_sigma=n_sigma)
