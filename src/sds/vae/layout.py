"""Output layout of the VAE and activation of its outputs.

The VAE reads the index map only from the schema contract, never from the scikit-learn encoder.
The model returns raw logits; the activation is tied to the loss the model was trained with:
- ``baseline`` (the assignment's MSE loss): sigmoid on every position, one-hot groups included.
  Groups do not sum to 1; ``decode`` cleans and renormalises them.
- ``mixed`` (MSE on numeric columns + cross-entropy per one-hot group): sigmoid on numeric and date
  positions, softmax on every categorical group, i.e. one probability distribution per group.
- ``mixed_sigma`` (Gaussian NLL with a learned sigma per numeric column + cross-entropy per group):
  the same activation as ``mixed``; the sigmoid gives the mean of each numeric column.
Keeping logits raw lets the cross-entropy work on logits (``F.cross_entropy``), which is stable
where a softmax followed by a log is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sds.data_engine.contract import SchemaContract

BASELINE_LOSS = "baseline"
MIXED_LOSS = "mixed"
MIXED_SIGMA_LOSS = "mixed_sigma"
LOSS_NAMES = (BASELINE_LOSS, MIXED_LOSS, MIXED_SIGMA_LOSS)

SCALAR_ROLES = frozenset({"numeric", "date"})   # one position each, scaled to [0, 1]
CATEGORICAL_ROLE = "categorical"
MIN_GROUP_SLOTS = 2                              # a one-slot group would be a constant column


class VaeError(Exception):
    """Blocking error of the generative core, with a stable code and a plain-language message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def invalid_contract(message: str) -> VaeError:
    """Error for a contract the VAE cannot use."""
    return VaeError("invalid_contract", f"The description of the data cannot be used: {message}.")


@dataclass(frozen=True)
class OutputLayout:
    """Where each kind of output lives in the encoded vector.

    Attributes:
        width: Length of the encoded vector.
        numeric_index: Positions of numeric and date columns (long tensor).
        categorical_slices: One ``slice(start, stop)`` per one-hot group, in contract order.
    """

    width: int
    numeric_index: torch.Tensor
    categorical_slices: tuple[slice, ...]

    @classmethod
    def from_contract(cls, contract: SchemaContract) -> OutputLayout:
        """Build and validate the layout of a schema contract.

        Args:
            contract: The schema contract of the fitted encoder.

        Returns:
            The layout, on the CPU.

        Raises:
            VaeError: ``invalid_contract`` when the columns do not cover ``[0, width)`` exactly, in
                order, or when a column has an unknown role or the wrong number of positions.
        """
        numeric, groups = [], []
        position = 0
        for column in contract.columns:
            if column.start != position:
                gap_or_overlap = "a gap" if column.start > position else "an overlap"
                raise invalid_contract(f"there is {gap_or_overlap} before column '{column.name}'")
            size = column.stop - column.start
            if column.role in SCALAR_ROLES:
                if size != 1:
                    raise invalid_contract(f"column '{column.name}' should take 1 position, not {size}")
                numeric.append(column.start)
            elif column.role == CATEGORICAL_ROLE:
                if size < MIN_GROUP_SLOTS:
                    raise invalid_contract(f"column '{column.name}' has fewer than {MIN_GROUP_SLOTS} values")
                groups.append(slice(column.start, column.stop))
            else:
                raise invalid_contract(f"column '{column.name}' has an unknown role '{column.role}'")
            position = column.stop
        if position != contract.width:
            raise invalid_contract(f"the columns take {position} positions, but the width is {contract.width}")
        return cls(contract.width, torch.tensor(numeric, dtype=torch.long), tuple(groups))

    def to(self, device: torch.device | str) -> OutputLayout:
        """The same layout with its index tensor on ``device``."""
        return OutputLayout(self.width, self.numeric_index.to(device), self.categorical_slices)


def activate(logits: torch.Tensor, layout: OutputLayout, *, loss: str) -> torch.Tensor:
    """Turn raw logits into decoder-ready outputs, with the activation of ``loss``.

    Args:
        logits: Tensor of shape ``(n_rows, layout.width)``. Not modified.
        layout: Layout of the contract the model was trained on.
        loss: One of ``LOSS_NAMES``: the loss the model was trained with.

    Returns:
        A new tensor of the same shape: values in (0, 1); with ``MIXED_LOSS`` and
        ``MIXED_SIGMA_LOSS`` every categorical group sums to 1. Gradients flow through it.

    Raises:
        VaeError: ``wrong_width`` when the logits do not have ``layout.width`` columns.
        ValueError: when ``loss`` is not one of ``LOSS_NAMES``.
    """
    if loss not in LOSS_NAMES:
        raise ValueError(f"Unknown loss {loss!r}; expected one of {LOSS_NAMES}.")
    if logits.ndim != 2 or logits.shape[1] != layout.width:
        raise VaeError(
            "wrong_width",
            f"The model output has shape {tuple(logits.shape)}; expected rows of {layout.width} values.",
        )
    if loss == BASELINE_LOSS:
        return torch.sigmoid(logits)
    # Writes go into a fresh tensor: overwriting the output of torch.sigmoid in place would break
    # its backward pass, which reuses that output.
    out = torch.empty_like(logits)
    out[:, layout.numeric_index] = torch.sigmoid(logits[:, layout.numeric_index])
    for group in layout.categorical_slices:
        out[:, group] = torch.softmax(logits[:, group], dim=1)
    return out
