"""Sampling synthetic rows from a trained VAE.

z ~ N(0, I) -> decoder -> activation of the loss the model was trained with -> ``decode``. The
encoder is not used: generation starts from the prior. Batched like training (GPU memory does not
grow with the number of rows), with its own seeded generators: the same model and seed give the same
table whatever the global random state.

The decoder returns the *mean* of every numeric column, so synthetic numeric columns would be
narrower than the real ones. The ``mixed_sigma`` loss learned a sigma per numeric column: sampling
adds ``sigma * eps`` to them (what that loss models). The other losses have no sigma and add nothing.
Values pushed outside [0, 1] by the noise are clipped to the train range by ``decode``.

``decode`` repairs the rows that pair categorical values never seen together in train; the few it
cannot repair (Adult: 0.1%) are replaced by new rows, drawn with seeds derived from ``seed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sds.config import RANDOM_STATE
from sds.data_engine.decoding import decode_checked
from sds.data_engine.contract import SchemaContract
from sds.vae.hyperparameters_config import BATCH_SIZE, MAX_GENERATION_ROUNDS, REGENERATION_OVERSAMPLE
from sds.vae.layout import activate
from sds.vae.losses import LOG_MAX_SIGMA, LOG_MIN_SIGMA
from sds.vae.training import TrainedVae


def seeded_generators(seed: int, count: int) -> list[torch.Generator]:
    """``count`` independent CPU generators from one seed (latent codes and noise do not share a stream)."""
    states = np.random.SeedSequence(seed).spawn(count)
    return [torch.Generator().manual_seed(int(s.generate_state(1)[0])) for s in states]


def sample(trained: TrainedVae, n_rows: int, *, seed: int = RANDOM_STATE, numeric_noise: bool | None = None,
           batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Synthetic encoded rows, activated and ready for ``decode``.

    Args:
        trained: A trained model (``train_vae``).
        n_rows: Number of rows to generate.
        seed: Seed of the latent codes and of the numeric noise.
        numeric_noise: Add ``sigma * eps`` to the numeric columns. ``None`` (the app default): only when
            the model learned a sigma (``mixed_sigma``): more faithful numbers and associations, privacy
            gate passed with a wider margin. ``False`` keeps the decoder mean, for comparisons only.
        batch_size: Rows generated at a time.

    Returns:
        Array of shape ``(n_rows, width)``, float32. The model is left in eval mode.

    Raises:
        ValueError: ``n_rows`` not positive, or noise asked for a model without a learned sigma.
    """
    if n_rows <= 0:
        raise ValueError(f"The number of rows must be positive, got {n_rows}.")
    model = trained.model
    if numeric_noise is None:
        numeric_noise = model.log_sigma is not None
    if numeric_noise and model.log_sigma is None:
        raise ValueError(f"A model trained with the {trained.loss!r} loss has no learned sigma to add noise with.")

    device = next(model.parameters()).device
    layout = trained.layout.to(device)
    latent_generator, noise_generator = seeded_generators(seed, 2)
    model.eval()
    rows = np.empty((n_rows, layout.width), dtype=np.float32)
    with torch.no_grad():
        sigma = model.log_sigma.clamp(LOG_MIN_SIGMA, LOG_MAX_SIGMA).exp() if numeric_noise else None
        for start in range(0, n_rows, batch_size):
            size = min(batch_size, n_rows - start)
            z = torch.randn(size, model.architecture.latent_dim, generator=latent_generator).to(device)
            values = activate(model.decode(z), layout, loss=trained.loss)
            if sigma is not None:
                noise = torch.randn(size, len(layout.numeric_index), generator=noise_generator).to(device)
                values[:, layout.numeric_index] += sigma * noise
            rows[start:start + size] = values.cpu().numpy()
    return rows


def round_seed(seed: int, round_index: int) -> int:
    """Seed of a regeneration round, derived from the seed of the table (round 0 is ``seed`` itself)."""
    return seed if round_index == 0 else int(np.random.SeedSequence([seed, round_index]).generate_state(1)[0])


def generate(trained: TrainedVae, contract: SchemaContract, n_rows: int, *, seed: int = RANDOM_STATE,
             numeric_noise: bool | None = None) -> pd.DataFrame:
    """A synthetic table with the columns and types of ``contract``.

    Rows that ``decode`` could not make coherent are replaced by new ones, for at most
    ``MAX_GENERATION_ROUNDS`` rounds; if some are still left, they are kept so the table has
    ``n_rows`` rows.

    Args:
        trained: A trained model.
        contract: The schema contract the model was trained on.
        n_rows: Number of rows.
        seed: Seed of the sampling and of the decoding (same seed, same table).
        numeric_noise: See ``sample``.

    Returns:
        The synthetic table, ``n_rows`` rows.
    """
    def draw(size: int, round_index: int) -> tuple[pd.DataFrame, np.ndarray]:
        this_seed = round_seed(seed, round_index)
        vectors = sample(trained, size, seed=this_seed, numeric_noise=numeric_noise)
        return decode_checked(vectors, contract, seed=this_seed)

    table, incoherent = draw(n_rows, 0)
    if not incoherent.any():
        return table
    kept, leftover = [table[~incoherent]], table[incoherent]
    missing = len(leftover)
    for round_index in range(1, MAX_GENERATION_ROUNDS):
        if not missing:
            break
        batch, bad = draw(int(np.ceil(missing * REGENERATION_OVERSAMPLE)), round_index)
        kept.append(batch[~bad].head(missing))
        missing -= len(kept[-1])
    if missing:
        kept.append(leftover.head(missing))           # given up: incoherent rows rather than fewer rows
    return pd.concat(kept, ignore_index=True)
