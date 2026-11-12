"""Privacy check of the synthetic data: exact copies and distance to the closest record (DCR).

Every row is an encoded vector. The synthetic rows are compared after decoding and encoding again with
the train encoder, so the check measures the file the user takes away, not the raw VAE output.

The distance is Gower's on the encoded vector: one term per column of the contract, averaged.
A numeric or date column contributes ``|a - b|`` on the MinMax scale; a one-hot group contributes
``1 - a . b`` (0 for the same level, 1 otherwise). Euclidean distance on the one-hot vector would give a
categorical mismatch a weight of sqrt(2) against at most 1 for a numeric column.

The reference is the distance of the holdout rows, which the VAE never saw, to the train rows: a
synthetic row should be no closer to the train than a new real person is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

from sds.config import RANDOM_STATE
from sds.data_engine.contract import SchemaContract
from sds.data_engine.types import Issue
from sds.validation.privacy_config import (
    DCR_BATCH_SIZE,
    DCR_MAX_ROWS,
    DCR_PERCENTILE,
    DCR_REFERENCE_BLOCK,
    EXACT_DISTANCE,
    EXACT_SHARE_TOLERANCE,
    MIN_DCR_RATIO,
    NOISE_Z,
)

CATEGORICAL_ROLE = "categorical"
TIE_WEIGHT = 0.5               # a synthetic row as close to train as to holdout counts half


@dataclass(frozen=True)
class PrivacyReport:
    """Outcome of the privacy check.

    Attributes:
        passed: True when no failure was found (download and chat can be unlocked).
        n_compared: Synthetic rows compared.
        synthetic_percentile: ``DCR_PERCENTILE``-th percentile of the synthetic -> train distances.
        holdout_percentile: The same percentile of the holdout -> train distances (reference).
        synthetic_exact_share: Share of synthetic rows identical to a train row.
        holdout_exact_share: Share of holdout rows identical to a train row (genuine duplicates).
        exact_share_limit: Highest synthetic exact share accepted.
        train_closer_share: Share of synthetic rows closer to the train than to the holdout (diagnostic).
        expected_train_closer_share: The same share for rows that copy nothing, ``n_train / (n_train + n_holdout)``.
        failures: Reasons the check failed, codes ``copied_rows`` and ``too_close`` (blocking).
        warnings: Limits of the check that do not block, code ``near_copies_unchecked``.
    """

    passed: bool
    n_compared: int
    synthetic_percentile: float
    holdout_percentile: float
    synthetic_exact_share: float
    holdout_exact_share: float
    exact_share_limit: float
    train_closer_share: float
    expected_train_closer_share: float
    failures: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)


def check_rows(rows: np.ndarray, contract: SchemaContract, name: str) -> None:
    """Raise ``ValueError`` unless ``rows`` is a non-empty matrix as wide as the contract."""
    if rows.ndim != 2 or rows.shape[1] != contract.width:
        raise ValueError(f"{name} must have shape (n, {contract.width}), got {rows.shape}")
    if len(rows) == 0:
        raise ValueError(f"{name} has no rows")


def gower_block(queries: np.ndarray, reference: np.ndarray, numeric: list[int], slots: list[int],
                n_groups: int, n_columns: int) -> np.ndarray:
    """Gower distances between every query and every reference row of one block."""
    distances = np.full((len(queries), len(reference)), float(n_groups))
    if numeric:
        distances += cdist(queries[:, numeric], reference[:, numeric], metric="cityblock")
    if slots:
        distances -= queries[:, slots].astype(np.float64) @ reference[:, slots].T.astype(np.float64)
    return distances / n_columns


def closest_record_distances(queries: np.ndarray, reference: np.ndarray, contract: SchemaContract, *,
                             batch_size: int = DCR_BATCH_SIZE) -> np.ndarray:
    """Gower distance of every query row to its closest reference row.

    Args:
        queries: Encoded rows to measure, shape ``(n, width)``.
        reference: Encoded rows to search, shape ``(m, width)``.
        contract: Schema contract of the encoder (positions and roles of the columns).
        batch_size: Query rows per block; reference rows go by ``DCR_REFERENCE_BLOCK``.

    Returns:
        Array of shape ``(n,)`` in query order.

    Raises:
        ValueError: Empty inputs or a width different from the contract.
    """
    check_rows(queries, contract, "queries")
    check_rows(reference, contract, "reference")
    numeric = [c.start for c in contract.columns if c.role != CATEGORICAL_ROLE]
    groups = [c for c in contract.columns if c.role == CATEGORICAL_ROLE]
    slots = [i for c in groups for i in range(c.start, c.stop)]
    closest = np.full(len(queries), np.inf)
    for q_start in range(0, len(queries), batch_size):
        q_block = queries[q_start:q_start + batch_size]
        for r_start in range(0, len(reference), DCR_REFERENCE_BLOCK):
            r_block = reference[r_start:r_start + DCR_REFERENCE_BLOCK]
            block_min = gower_block(q_block, r_block, numeric, slots, len(groups), len(contract.columns)).min(axis=1)
            np.minimum(closest[q_start:q_start + len(q_block)], block_min, out=closest[q_start:q_start + len(q_block)])
    return np.maximum(closest, 0.0)   # the dot product can leave -1e-16 on identical rows


def exact_share_limit(holdout_share: float, n_synthetic: int, n_holdout: int, exact_tolerance: float) -> float:
    """Highest share of exact copies the genuine duplicates and the sampling noise explain."""
    noise = NOISE_Z * np.sqrt(holdout_share * (1 - holdout_share) * (1 / n_synthetic + 1 / n_holdout))
    return float(holdout_share + exact_tolerance + noise)


def memorisation_causes(n_train: int) -> str:
    """The two usual reasons a model memorises, in plain language (the user can act on both, not on training)."""
    return (f"This usually happens when the dataset is small (the model learned from {n_train:,} rows) or when "
            "some columns identify a single person almost by themselves (codes, exact dates, rare values).")


def privacy_check(synthetic: np.ndarray, train: np.ndarray, holdout: np.ndarray, contract: SchemaContract, *,
                  max_rows: int = DCR_MAX_ROWS, seed: int = RANDOM_STATE, min_ratio: float = MIN_DCR_RATIO,
                  exact_tolerance: float = EXACT_SHARE_TOLERANCE) -> PrivacyReport:
    """Check that the synthetic rows do not copy the train rows.

    Args:
        synthetic: Encoded synthetic rows (decoded, then encoded with the train encoder).
        train: Encoded rows the VAE was trained on.
        holdout: Encoded real rows the VAE never saw.
        contract: Schema contract of the encoder.
        max_rows: Synthetic rows compared at most; a seeded random subset is taken beyond.
        seed: Seed of that subset.
        min_ratio: The synthetic percentile must reach ``min_ratio`` times the holdout one.
        exact_tolerance: Share of copies allowed beyond the genuine duplicates and the noise.

    Returns:
        The report; ``passed`` is False when any failure was found.

    Raises:
        ValueError: Empty inputs or a width different from the contract.
    """
    for rows, name in ((synthetic, "synthetic"), (train, "train"), (holdout, "holdout")):
        check_rows(rows, contract, name)
    if len(synthetic) > max_rows:
        chosen = np.random.default_rng(seed).choice(len(synthetic), size=max_rows, replace=False)
        synthetic = synthetic[np.sort(chosen)]

    to_train = closest_record_distances(synthetic, train, contract)
    to_holdout = closest_record_distances(synthetic, holdout, contract)
    reference = closest_record_distances(holdout, train, contract)

    synthetic_percentile = float(np.percentile(to_train, DCR_PERCENTILE))
    holdout_percentile = float(np.percentile(reference, DCR_PERCENTILE))
    synthetic_exact = float(np.mean(to_train <= EXACT_DISTANCE))
    holdout_exact = float(np.mean(reference <= EXACT_DISTANCE))
    limit = exact_share_limit(holdout_exact, len(synthetic), len(holdout), exact_tolerance)
    closer = np.where(to_train < to_holdout, 1.0, np.where(to_train == to_holdout, TIE_WEIGHT, 0.0))

    failures = []
    causes = memorisation_causes(len(train))
    if synthetic_exact > limit:
        failures.append(Issue("copied_rows", (
            f"{synthetic_exact:.1%} of the synthetic rows are identical to a real patient record, while the real "
            f"data alone would explain at most {limit:.1%}. The model memorised part of the data, so the synthetic "
            f"data cannot be shared. {causes}")))
    if synthetic_percentile < min_ratio * holdout_percentile:
        failures.append(Issue("too_close", (
            "The synthetic rows are much closer to the real records than new, unseen patients are. The model "
            f"memorised the data, so the synthetic data cannot be shared. {causes}")))
    warnings = []
    if holdout_percentile <= EXACT_DISTANCE:
        # Many new people are identical to a train row, so too_close compares against 0 and cannot fire.
        # No distance-based variant tells a near-copy from a new person on such data (measured on CDC).
        warnings.append(Issue("near_copies_unchecked", (
            "Many people in your data have identical or almost identical answers, so the check for near-copies "
            "of real records cannot work here. Exact copies of real records are still checked. Share the "
            "synthetic data with the same care you would use for aggregated results.")))
    return PrivacyReport(
        passed=not failures,
        n_compared=len(synthetic),
        synthetic_percentile=synthetic_percentile,
        holdout_percentile=holdout_percentile,
        synthetic_exact_share=synthetic_exact,
        holdout_exact_share=holdout_exact,
        exact_share_limit=limit,
        train_closer_share=float(closer.mean()),
        expected_train_closer_share=len(train) / (len(train) + len(holdout)),
        failures=failures,
        warnings=warnings,
    )
