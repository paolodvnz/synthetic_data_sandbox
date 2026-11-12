"""Decoding: from vectors (the VAE output) back to a table, reading only the schema contract.

- numeric: inverse MinMax, clipped to the train range, rounded to whole numbers or to the decimals of the
  train values;
- date: the same in whole days, then ``origin + days``;
- categorical: the group holds probabilities (the VAE applies the softmax). One level is *sampled*
  per row, with a seed: argmax would always pick the most likely level and squash the minority
  ones. Scores are cleaned first (negatives -> 0, renormalised, all-zero -> uniform), because the
  baseline MSE loss does not produce probabilities. A one-hot vector is already a distribution, so
  ``decode(encode(x))`` gives ``x`` back without a separate argmax mode.
- "Other": text columns write ``"Other"``; numeric codes (whole or not) pick one of the merged values with
  the train proportions, so the column stays made of numbers seen in the real data.
- coherence repair: sampling each group on its own can pair values no train row pairs (on Adult 25%
  of the rows, e.g. "education" HS-grad with "education-num" 10). In every row with a slot pair missing
  from ``contract.seen_pairs``, the conflicting column whose sampled level is the least likely (among
  those that have a compatible level) is resampled among the levels compatible with the rest of the
  row, for ``MAX_REPAIR_PASSES`` passes. Only that column changes: the other draws, and so the
  marginals, stay as sampled (rejecting the whole row instead favoured the confident levels). Rows
  left incoherent are flagged by ``decode_checked``; the sampler regenerates them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sds.config import RANDOM_STATE
from sds.data_engine.contract import OTHER_LABEL, ContractColumn, SchemaContract
from sds.data_engine.thresholds_config import MAX_REPAIR_PASSES
from sds.data_engine.types import DataEngineError


def unscale(scores: np.ndarray, column: ContractColumn) -> np.ndarray:
    """Inverse MinMax, clipped to the train range."""
    return np.clip(scores * (column.max - column.min) + column.min, column.min, column.max)


def clean_probabilities(scores: np.ndarray) -> np.ndarray:
    """Rows of non-negative weights summing to 1 (all-zero rows become uniform)."""
    weights = np.clip(scores, 0, None)
    totals = weights.sum(axis=1, keepdims=True)
    weights = np.where(totals > 0, weights, 1.0)
    return weights / weights.sum(axis=1, keepdims=True)


def sample_slots(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One slot index per row, drawn from that row's probabilities (inverse CDF)."""
    cumulative = probabilities.cumsum(axis=1)
    draws = rng.random(len(probabilities))[:, None]
    # '<=' keeps one-hot rows exact even for a draw of 0; the cap guards against rounding at 1.
    return np.minimum((cumulative <= draws).sum(axis=1), probabilities.shape[1] - 1)


def group_probabilities(vectors: np.ndarray, column: ContractColumn,
                        rows: np.ndarray | slice = slice(None)) -> np.ndarray:
    """Cleaned probabilities of one categorical group, for some rows (float64 for those rows only)."""
    return clean_probabilities(vectors[rows, column.start:column.stop].astype(np.float64))


def category_values(slots: np.ndarray, column: ContractColumn, rng: np.random.Generator) -> pd.Series:
    """Levels of one categorical group from its slots."""
    is_other = slots == len(column.levels)              # the "Other" slot is always the last one
    if column.numeric_codes:
        dtype = np.int64 if column.integer else np.float64
        values = np.zeros(len(slots), dtype=dtype)
        if column.levels:
            kept = np.array(column.levels, dtype=dtype)
            values[~is_other] = kept[slots[~is_other]]
        if is_other.any():
            merged = np.array([level for level, _ in column.merged], dtype=dtype)
            rows = np.array([n for _, n in column.merged], dtype=float)
            values[is_other] = rng.choice(merged, size=int(is_other.sum()), p=rows / rows.sum())
        return pd.Series(values, dtype=dtype)
    labels = np.array([*column.levels, OTHER_LABEL], dtype=object)
    return pd.Series(labels[slots], dtype="str")


def pair_tables(contract: SchemaContract, groups: list[ContractColumn]) -> dict[tuple[int, int], np.ndarray]:
    """``(i, j) -> seen[slot of i, slot of j]`` for every pair of groups with an entry, both orders."""
    index = {g.name: i for i, g in enumerate(groups)}
    tables = {}
    for pairs in contract.seen_pairs:
        if pairs.first not in index or pairs.second not in index:
            continue
        i, j = index[pairs.first], index[pairs.second]
        seen = np.zeros((groups[i].stop - groups[i].start, groups[j].stop - groups[j].start), dtype=bool)
        if pairs.slots:
            found = np.array(pairs.slots)
            seen[found[:, 0], found[:, 1]] = True
        tables[i, j], tables[j, i] = seen, seen.T
    return tables


def unseen_counts(slots: list[np.ndarray], tables: dict) -> np.ndarray:
    """``(groups, rows)``: how many unseen pairs each group of each row is part of."""
    counts = np.zeros((len(slots), len(slots[0])), dtype=np.int16)
    for (i, j), seen in tables.items():
        counts[i] += ~seen[slots[i], slots[j]]
    return counts


def repair_slots(vectors: np.ndarray, groups: list[ContractColumn], slots: list[np.ndarray], tables: dict,
                 rng: np.random.Generator) -> np.ndarray:
    """Resample, in every incoherent row, the least likely conflicting column among the compatible levels.

    ``slots`` is changed in place. Returns True for the rows still incoherent after the last pass.
    """
    for _ in range(MAX_REPAIR_PASSES):
        conflicts = unseen_counts(slots, tables)
        if not conflicts.any():
            break
        n_rows = len(slots[0])
        best = np.full(n_rows, np.inf)
        target = np.full(n_rows, -1)
        candidates = {}
        for i, group in enumerate(groups):
            rows = np.flatnonzero(conflicts[i])
            if not len(rows):
                continue
            probabilities = group_probabilities(vectors, group, rows)
            compatible = np.ones(probabilities.shape, dtype=bool)
            for j in range(len(groups)):
                if (i, j) in tables:
                    compatible &= tables[i, j][:, slots[j][rows]].T
            weights = probabilities * compatible
            fixable = weights.sum(axis=1) > 0
            likelihood = probabilities[np.arange(len(rows)), slots[i][rows]]
            better = fixable & (likelihood < best[rows])
            best[rows[better]], target[rows[better]] = likelihood[better], i
            candidates[i] = (rows, weights)
        if not (target >= 0).any():
            break
        for i, (rows, weights) in candidates.items():
            picked = target[rows] == i
            if picked.any():
                chosen = weights[picked]
                slots[i][rows[picked]] = sample_slots(chosen / chosen.sum(axis=1, keepdims=True), rng)
    return unseen_counts(slots, tables).any(axis=0)


def decode_checked(vectors: np.ndarray, contract: SchemaContract, *,
                   seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, np.ndarray]:
    """``decode``, plus which rows are still incoherent after the repair.

    Args:
        vectors: Array of shape ``(n_rows, contract.width)``: the VAE output, or encoded rows.
        contract: The schema contract of the fitted encoder.
        seed: Seed of the sampling of categorical levels and of the repair.

    Returns:
        The table (as ``decode``) and a boolean array, True for the rows that still pair categorical
        values never seen together in train (e.g. a one-hot real row with a rare pair: nothing else to
        choose, so it is kept as it is).

    Raises:
        DataEngineError: ``wrong_width`` when the rows do not have ``contract.width`` values.
    """
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[1] != contract.width:
        raise DataEngineError(
            "wrong_width",
            f"Expected rows of {contract.width} values, got shape {vectors.shape}.",
        )
    rng = np.random.default_rng(seed)
    groups = [c for c in contract.columns if c.role == "categorical"]
    # Slots only (int16: at most MAX_CATEGORIES per group); probabilities are recomputed for the rows
    # being repaired, so memory stays one float64 group at a time, as before the repair.
    slots = [sample_slots(group_probabilities(vectors, g), rng).astype(np.int16) for g in groups]
    tables = pair_tables(contract, groups)
    incoherent = (repair_slots(vectors, groups, slots, tables, rng) if tables and len(vectors)
                  else np.zeros(len(vectors), dtype=bool))
    group_slots = dict(zip((g.name for g in groups), slots, strict=True))
    table = {}
    for column in contract.columns:
        if column.role == "categorical":
            table[column.name] = category_values(group_slots[column.name], column, rng)
            continue
        # float64 one column at a time: a copy of the whole vector would double its memory.
        block = vectors[:, column.start:column.stop].astype(np.float64)
        if column.role == "date":
            days = np.rint(unscale(block[:, 0], column)).astype(np.int64)
            table[column.name] = pd.Timestamp(column.origin) + pd.to_timedelta(days, unit="D")
        else:
            values = unscale(block[:, 0], column)
            if column.integer:
                values = np.rint(values).astype(np.int64)
            elif column.decimals is not None:
                values = np.round(values, column.decimals)      # as many decimals as the real values
            table[column.name] = values
    return pd.DataFrame(table, copy=False), incoherent      # the decoded columns are new: no need to copy them


def decode(vectors: np.ndarray, contract: SchemaContract, *, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Turn encoded rows back into a table with the original column types.

    Args:
        vectors: Array of shape ``(n_rows, contract.width)``: the VAE output, or encoded rows.
        contract: The schema contract of the fitted encoder.
        seed: Seed of the sampling of categorical levels and of the repair (same seed, same table).

    Returns:
        One column per contract column, in contract order.

    Raises:
        DataEngineError: ``wrong_width`` when the rows do not have ``contract.width`` values.
    """
    return decode_checked(vectors, contract, seed=seed)[0]
