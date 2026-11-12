"""Training loop of the tabular VAE.

- The holdout of the Data Engine never comes here: a validation part is cut from the train rows.
- Data stay on the CPU and reach the device one fixed-size batch at a time (DataLoader), for training
  and evaluation alike: GPU memory does not grow with the number of rows. (RAM is bounded upstream,
  by the number of rows the app accepts.)
- The training length is set in epochs (``step_budget``) and capped in steps: with a fixed number of steps
  a small dataset is seen thousands of times and is over-fitted before early stopping may act.
- The validation loss is computed with z = mu and at the full KL weight: during the warm-up the
  weighted total grows even when the model improves, so early stopping starts after the warm-up.
  The best weights are restored at the end.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sds.config import RANDOM_STATE
from sds.vae.hyperparameters_config import (
    BATCH_SIZE,
    BETA,
    EVALUATIONS,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_STEPS,
    MIN_STEPS,
    PATIENCE,
    VALIDATION_FRACTION,
    WEIGHT_DECAY,
)
from sds.vae.layout import LOSS_NAMES, OutputLayout, VaeError
from sds.vae.losses import LossParts, active_latent_dims, kl_per_dim, kl_schedule, vae_loss, warmup_steps
from sds.vae.model import Architecture, TabularVae, build_vae
from sds.vae.runtime import get_device, set_seed


@dataclass(frozen=True)
class LossValues:
    """Loss components as plain floats (for history, progress reports and plots)."""

    total: float
    numeric: float
    categorical: float
    kl: float


@dataclass(frozen=True)
class Evaluation:
    """One evaluation during training.

    Attributes:
        step: Gradient steps done.
        kl_weight: KL weight used for training at this step (warm-up schedule).
        train: Mean training loss over the steps since the previous evaluation.
        validation: Loss on the validation rows, z = mu, full KL weight.
        active_dims: Latent dimensions that carry information on the validation rows.
    """

    step: int
    kl_weight: float
    train: LossValues
    validation: LossValues
    active_dims: int


@dataclass
class TrainedVae:
    """Result of a training run.

    Attributes:
        model: The network with the best weights, in eval mode, on the training device.
        layout: Layout of the contract it was trained on.
        loss: Name of the loss it was trained with (it fixes the activation for sampling).
        history: Every evaluation, in order.
        best_step: Step of the restored weights (``None``: the run ended before early stopping started).
        stopped_early: Stopped because the validation loss stopped improving.
        cancelled: Stopped by ``stop_event``.
        train_index: Rows of the input used for training.
        validation_index: Rows of the input used for validation.
    """

    model: TabularVae
    layout: OutputLayout
    loss: str
    history: list[Evaluation]
    best_step: int | None
    stopped_early: bool
    cancelled: bool
    train_index: np.ndarray
    validation_index: np.ndarray


class EarlyStopping:
    """Stop when the monitored value has not improved for ``patience`` evaluations.

    Values reported before ``start_step`` (the KL warm-up) are ignored.
    """

    def __init__(self, patience: int, start_step: int) -> None:
        self.patience = patience
        self.start_step = start_step
        self.best_step: int | None = None
        self.best_value = math.inf
        self.evaluations_without_improvement = 0

    def update(self, step: int, value: float) -> bool:
        """Record an evaluation; return ``True`` when it is a new best."""
        if step < self.start_step:
            return False
        if value < self.best_value:
            self.best_step, self.best_value = step, value
            self.evaluations_without_improvement = 0
            return True
        self.evaluations_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        """``patience`` evaluations in a row without improvement."""
        return self.evaluations_without_improvement >= self.patience


@dataclass(frozen=True)
class StepBudget:
    """Length of one training.

    Attributes:
        max_steps: Gradient steps at most (the KL warm-up is a share of them).
        eval_every: Steps between two evaluations on the validation rows.
    """

    max_steps: int
    eval_every: int


def evaluation_interval(max_steps: int) -> int:
    """Steps between two evaluations, so that a training has about ``EVALUATIONS`` of them."""
    return max(1, max_steps // EVALUATIONS)


def step_budget(n_rows: int, *, batch_size: int = BATCH_SIZE, max_epochs: int = MAX_EPOCHS) -> StepBudget:
    """Training length for ``n_rows`` train rows: ``max_epochs`` passes over the rows left after the
    validation cut, within ``[MIN_STEPS, MAX_STEPS]``.

    Args:
        n_rows: Train rows given to ``train_vae`` (before the validation cut).
        batch_size: Rows per batch (a batch larger than the rows is one step per epoch, as in ``train_vae``).
        max_epochs: Passes over the training rows.

    Returns:
        The step budget.

    Raises:
        ValueError: when there are fewer than 2 rows.
    """
    n_training = len(split_validation(n_rows)[0])
    steps_per_epoch = math.ceil(n_training / batch_size)
    max_steps = min(MAX_STEPS, max(MIN_STEPS, max_epochs * steps_per_epoch))
    return StepBudget(max_steps=max_steps, eval_every=evaluation_interval(max_steps))


def split_validation(n_rows: int, *, fraction: float = VALIDATION_FRACTION,
                     seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Random split of the train rows into training and validation indices (at least 1 row each).

    Raises:
        ValueError: when there are fewer than 2 rows.
    """
    if n_rows < 2:
        raise ValueError(f"At least 2 rows are needed to cut a validation part, got {n_rows}.")
    n_validation = min(max(1, round(fraction * n_rows)), n_rows - 1)
    order = np.random.default_rng(seed).permutation(n_rows)
    return np.sort(order[n_validation:]), np.sort(order[:n_validation])


def as_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Float32 tensor on the CPU, sharing memory with ``x`` when possible."""
    return torch.as_tensor(np.asarray(x, dtype=np.float32)) if not torch.is_tensor(x) else x.float().cpu()


def stacked(parts: LossParts) -> torch.Tensor:
    """The four components as one detached tensor (no device sync per step)."""
    return torch.stack([parts.total, parts.numeric, parts.categorical, parts.kl]).detach()


def evaluate(model: TabularVae, x: np.ndarray | torch.Tensor, layout: OutputLayout, *, loss: str,
             beta: float = BETA, batch_size: int = BATCH_SIZE) -> tuple[LossValues, int]:
    """Loss of ``model`` on rows ``x``, in batches, with z = mu and KL weight ``beta``.

    The model's train/eval mode is restored afterwards.

    Returns:
        The mean loss components over the rows and the number of active latent dimensions.
    """
    device = next(model.parameters()).device
    layout = layout.to(device)
    was_training = model.training
    model.eval()
    totals = torch.zeros(4, device=device)
    kl_dims = torch.zeros(model.architecture.latent_dim, device=device)
    n_rows = 0
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(as_tensor(x)), batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            logits, mu, logvar = model(batch)
            parts = vae_loss(logits, batch, mu, logvar, layout, loss=loss, kl_weight=beta,
                             log_sigma=model.log_sigma)
            totals += stacked(parts) * len(batch)
            kl_dims += kl_per_dim(mu, logvar) * len(batch)
            n_rows += len(batch)
    model.train(was_training)
    values = LossValues(*(totals / n_rows).tolist())
    return values, active_latent_dims(kl_dims / n_rows)


def train_vae(x: np.ndarray, layout: OutputLayout, *, loss: str, architecture: Architecture | None = None,
              beta: float = BETA, max_steps: int | None = None, batch_size: int = BATCH_SIZE,
              eval_every: int | None = None, patience: int = PATIENCE, seed: int = RANDOM_STATE,
              device: torch.device | str | None = None,
              on_progress: Callable[[Evaluation], None] | None = None,
              stop_event: threading.Event | None = None) -> TrainedVae:
    """Train a VAE on encoded train rows.

    Args:
        x: Encoded train rows, shape ``(n_rows, layout.width)`` (the holdout must not be here).
        layout: Layout of the contract.
        loss: One of ``LOSS_NAMES``.
        architecture: Network sizes; default ``choose_architecture(layout.width)``.
        beta: Final KL weight (0 for the sanity check).
        max_steps: Gradient steps at most; default ``step_budget(len(x), batch_size=batch_size)``.
        batch_size: Rows per batch (capped at the number of training rows).
        eval_every: Steps between two evaluations (one more at the end); default ``max_steps // EVALUATIONS``.
        patience: Evaluations without improvement before early stopping.
        seed: Seed of the split, the initial weights, the shuffling and the latent noise.
        device: Where to train; default ``get_device()``.
        on_progress: Called with every ``Evaluation``.
        stop_event: When set, training stops cleanly before the next step.

    Returns:
        The trained model with its history.

    Raises:
        VaeError: ``wrong_width`` when the rows do not have ``layout.width`` values.
        ValueError: unknown ``loss``, or fewer than 2 rows.
    """
    if loss not in LOSS_NAMES:
        raise ValueError(f"Unknown loss {loss!r}; expected one of {LOSS_NAMES}.")
    data = as_tensor(x)
    if data.ndim != 2 or data.shape[1] != layout.width:
        raise VaeError("wrong_width",
                       f"The rows have shape {tuple(data.shape)}; expected rows of {layout.width} values.")
    device = get_device() if device is None else torch.device(device)
    set_seed(seed)
    if max_steps is None:
        max_steps = step_budget(len(data), batch_size=batch_size).max_steps
    if eval_every is None:
        eval_every = evaluation_interval(max_steps)

    train_index, validation_index = split_validation(len(data), seed=seed)
    train_rows, validation_rows = data[train_index], data[validation_index]
    model = build_vae(layout, loss=loss, architecture=architecture).to(device)
    device_layout = layout.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(train_rows), batch_size=min(batch_size, len(train_rows)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda")
    stopping = EarlyStopping(patience, start_step=warmup_steps(max_steps))

    history: list[Evaluation] = []
    best_state: dict[str, torch.Tensor] | None = None
    running, n_steps_since_eval = torch.zeros(4, device=device), 0
    step, cancelled = 0, False
    model.train()
    while step < max_steps and not cancelled and not stopping.should_stop:
        for (batch,) in loader:
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                break
            batch = batch.to(device, non_blocking=True)
            logits, mu, logvar = model(batch)
            parts = vae_loss(logits, batch, mu, logvar, device_layout, loss=loss,
                             kl_weight=kl_schedule(step, max_steps, beta=beta), log_sigma=model.log_sigma)
            optimizer.zero_grad()
            parts.total.backward()
            optimizer.step()
            step += 1
            running += stacked(parts)
            n_steps_since_eval += 1

            if step % eval_every == 0 or step == max_steps:
                validation, active_dims = evaluate(model, validation_rows, device_layout, loss=loss, beta=beta,
                                                   batch_size=batch_size)
                evaluation = Evaluation(step, kl_schedule(step, max_steps, beta=beta),
                                        LossValues(*(running / n_steps_since_eval).tolist()), validation, active_dims)
                history.append(evaluation)
                running, n_steps_since_eval = torch.zeros(4, device=device), 0
                if stopping.update(step, validation.total):
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if on_progress is not None:
                    on_progress(evaluation)
                if stopping.should_stop:
                    break
            if step >= max_steps:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainedVae(model=model, layout=layout, loss=loss, history=history, best_step=stopping.best_step,
                      stopped_early=stopping.should_stop, cancelled=cancelled,
                      train_index=train_index, validation_index=validation_index)
