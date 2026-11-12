"""Fidelity of the synthetic data: how close they are to real people, column by column and pair by pair.

The reference is the **holdout**, real rows the VAE never saw. Against the train rows a model that copies
would look perfect. Every distance comes with a **baseline** of real people: the real rows are cut into
``REAL_GROUPS`` disjoint groups of the same size (the holdout and ``TRAIN_SLICES`` slices of the train) and the
same distance is measured on every pair of groups. The mean over the pairs is the difference chance alone
produces, their standard deviation (the **spread**) how much that difference varies. The synthetic distance is
the mean over ``SYNTHETIC_SLICES`` synthetic slices of the same size, each compared with the holdout group.

Distances, all in [0, 1], lower = closer:
- numeric and date columns: Wasserstein distance (W1) on the [0, 1] scale of the train range;
- categorical columns: total variation distance between the level shares;
- associations: |synthetic - real| of the pairwise association matrix (|Pearson| between numbers,
  Cramér's V between categories, correlation ratio eta between a category and a number).

The report also carries what the UI draws (heatmaps of the two association matrices, histograms on common
bins) and the share of rows with a pair of categorical values never seen together in train. Its warnings
never block: fidelity decides how useful the data are, and that choice stays with the user. A warning needs a
difference both **real** (``z``, the distance beyond the baseline in spreads, above ``MIN_FIDELITY_Z``) and
**large** (the "excess", synthetic distance minus baseline, above a limit): z alone grows with the number of rows,
the excess alone with the noise of small samples (calibration in ``fidelity_config``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wasserstein_distance

from sds.config import RANDOM_STATE
from sds.data_engine.contract import OTHER_LABEL, ContractColumn, SchemaContract
from sds.data_engine.types import Issue
from sds.validation.fidelity_config import (
    FIDELITY_MAX_ROWS,
    HISTOGRAM_BINS,
    MAX_ASSOCIATION_EXCESS,
    MAX_CATEGORICAL_EXCESS,
    MAX_LISTED_COLUMNS,
    MAX_NUMERIC_EXCESS,
    MIN_BASELINE,
    MIN_FIDELITY_Z,
    REAL_GROUPS,
    SYNTHETIC_SLICES,
    TRAIN_SLICES,
)

CATEGORICAL_ROLE = "categorical"
DATE_ROLE = "date"


@dataclass(frozen=True)
class Measure:
    """One distance against the real-vs-real baseline.

    Attributes:
        distance: Synthetic vs holdout: mean over the synthetic slices.
        baseline: Mean of the same distance over the pairs of real groups.
        spread: Its standard deviation over the pairs.
        z: ``(distance - baseline) / max(spread, MIN_BASELINE)``: how many spreads beyond chance.
        excess: ``distance - baseline``: the difference beyond chance, on the scale of the distance.
    """

    distance: float
    baseline: float
    spread: float
    z: float
    excess: float


NO_MEASURE = Measure(*[float("nan")] * 5)       # a kind of column the data do not have


@dataclass(frozen=True)
class ColumnFidelity:
    """The measure of one column (W1 or total variation), with the fields of ``Measure``.

    Attributes:
        name: Column name.
        role: ``numeric``, ``date`` or ``categorical``.
        distance, baseline, spread, z, excess: As in ``Measure``.
    """

    name: str
    role: str
    distance: float
    baseline: float
    spread: float
    z: float
    excess: float


@dataclass(frozen=True)
class Histogram:
    """Shares of real and synthetic rows on common bins, ready to draw.

    Attributes:
        name: Column name.
        role: ``numeric``, ``date`` or ``categorical``.
        labels: Numeric: the ``HISTOGRAM_BINS + 1`` bin edges on the train range (dates as ISO strings);
            categorical: the levels.
        real: Share of holdout rows per bin or level.
        synthetic: Share of synthetic rows per bin or level.
    """

    name: str
    role: str
    labels: list
    real: list[float]
    synthetic: list[float]


@dataclass(frozen=True)
class FidelityLimits:
    """When the report warns: z above ``min_z`` **and** excess above the limit of the measure."""

    min_z: float = MIN_FIDELITY_Z
    numeric_excess: float = MAX_NUMERIC_EXCESS
    categorical_excess: float = MAX_CATEGORICAL_EXCESS
    association_excess: float = MAX_ASSOCIATION_EXCESS


@dataclass
class FidelityReport:
    """How close the synthetic data are to the real people of the holdout.

    Attributes:
        n_compared: Rows of every group and slice compared.
        columns: One entry per contract column, in contract order.
        numeric: Mean W1 of the numeric and date columns (``NO_MEASURE`` without such columns).
        categorical: The same with the total variation of the categorical columns.
        association: Mean |difference| off the diagonal of the association matrices.
        association_diff_max: Largest |difference| between the two heatmap matrices.
        association_real: Association matrix of the holdout group (heatmap).
        association_synthetic: Mean association matrix of the synthetic slices (heatmap).
        histograms: One per contract column (the holdout group and every synthetic slice).
        unseen_pairs_synthetic: Share of synthetic rows pairing categorical values never seen in train.
        unseen_pairs_real: The same for the holdout group (rare but true crossings).
        warnings: Not blocking: ``distributions_differ``, ``relationships_differ``.
    """

    n_compared: int
    columns: list[ColumnFidelity]
    numeric: Measure
    categorical: Measure
    association: Measure
    association_diff_max: float
    association_real: pd.DataFrame
    association_synthetic: pd.DataFrame
    histograms: list[Histogram]
    unseen_pairs_synthetic: float
    unseen_pairs_real: float
    warnings: list[Issue] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------------------------

def comparable(frame: pd.DataFrame, contract: SchemaContract) -> pd.DataFrame:
    """Real rows written like synthetic ones, so equal values compare equal.

    Merged rare text levels become "Other" (what the synthetic data write); integer codes become int64
    (a file column with missing values is float in the real rows, int in the synthetic ones) and decimal
    codes float64. Returns a copy.
    """
    frame = frame.copy()
    for column in contract.columns:
        if column.role != CATEGORICAL_ROLE:
            continue
        if column.numeric_codes:
            frame[column.name] = frame[column.name].astype("int64" if column.integer else "float64")
        elif column.has_other:
            merged = {level for level, _ in column.merged}
            frame[column.name] = frame[column.name].where(~frame[column.name].isin(merged), OTHER_LABEL)
    return frame


def as_number(series: pd.Series, column: ContractColumn) -> np.ndarray:
    """Values on the [0, 1] scale of the train range (dates in days from the first train date)."""
    values = (series - pd.Timestamp(column.origin)).dt.days if column.role == DATE_ROLE else series
    return ((values - column.min) / (column.max - column.min)).to_numpy(dtype=float)


def numeric_distance(real_values: pd.Series, synthetic_values: pd.Series, column: ContractColumn) -> float:
    """Wasserstein distance on the [0, 1] scale of the train range."""
    return float(wasserstein_distance(as_number(real_values, column), as_number(synthetic_values, column)))


def total_variation(real_values: pd.Series, synthetic_values: pd.Series) -> float:
    """Total variation distance between two categorical marginals (half the sum of share differences)."""
    p = real_values.value_counts(normalize=True)
    q = synthetic_values.value_counts(normalize=True)
    return float(0.5 * p.sub(q, fill_value=0).abs().sum())


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V between two categorical columns (0: independent, 1: one determines the other)."""
    table = pd.crosstab(a, b)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def correlation_ratio(categories: pd.Series, values: np.ndarray) -> float:
    """Correlation ratio eta: square root of the share of the numeric spread explained by the categories."""
    frame = pd.DataFrame({"c": categories.to_numpy(), "v": values})
    total = ((frame["v"] - frame["v"].mean()) ** 2).sum()
    if total == 0:
        return 0.0
    groups = frame.groupby("c", observed=True)["v"].agg(["mean", "size"])
    between = (groups["size"] * (groups["mean"] - frame["v"].mean()) ** 2).sum()
    return float(np.sqrt(between / total))


def association_matrix(frame: pd.DataFrame, contract: SchemaContract) -> pd.DataFrame:
    """Pairwise associations of every contract column, all in [0, 1] (undefined -> 0)."""
    names = [c.name for c in contract.columns]
    by_name = {c.name: c for c in contract.columns}
    numbers = {c.name: as_number(frame[c.name], c) for c in contract.columns if c.role != CATEGORICAL_ROLE}
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in combinations(names, 2):
        kinds = (by_name[a].role == CATEGORICAL_ROLE, by_name[b].role == CATEGORICAL_ROLE)
        if kinds == (False, False):
            with np.errstate(invalid="ignore", divide="ignore"):         # a constant sample: no correlation
                value = abs(np.corrcoef(numbers[a], numbers[b])[0, 1])
        elif kinds == (True, True):
            value = cramers_v(frame[a], frame[b])
        else:
            category, number = (a, b) if kinds[0] else (b, a)
            value = correlation_ratio(frame[category], numbers[number])
        matrix.loc[a, b] = matrix.loc[b, a] = 0.0 if np.isnan(value) else value
    return matrix


def off_diagonal(matrix: pd.DataFrame) -> np.ndarray:
    """Values of a square matrix outside its diagonal."""
    return matrix.to_numpy()[~np.eye(len(matrix), dtype=bool)]


def column_distance(real: pd.Series, other: pd.Series, column: ContractColumn) -> float:
    """W1 for a numeric or date column, total variation for a categorical one."""
    if column.role == CATEGORICAL_ROLE:
        return total_variation(real, other)
    return numeric_distance(real, other, column)


# --------------------------------------------------------------------------------------------
# Histograms and unseen pairs
# --------------------------------------------------------------------------------------------

def category_labels(column: ContractColumn, *frames: pd.Series) -> list:
    """Levels of a categorical column as the synthetic data write them, then any other value found."""
    if column.numeric_codes:
        labels = sorted([*column.levels, *(level for level, _ in column.merged)])
    else:
        labels = [*column.levels, *([OTHER_LABEL] if column.has_other else [])]
    extra = {v for series in frames for v in series.unique()} - set(labels)
    return labels + sorted(extra, key=str)


def histogram(real: pd.Series, synthetic: pd.Series, column: ContractColumn) -> Histogram:
    """Shares of the two samples on bins of the train range (values outside it go to the edge bins)."""
    if column.role == CATEGORICAL_ROLE:
        labels = category_labels(column, real, synthetic)
        shares = [series.value_counts().reindex(labels, fill_value=0).to_numpy() / len(series)
                  for series in (real, synthetic)]
        return Histogram(column.name, column.role, labels, *(s.tolist() for s in shares))
    edges = np.linspace(0, 1, HISTOGRAM_BINS + 1)
    shares = [np.histogram(np.clip(as_number(series, column), 0, 1), bins=edges)[0] / len(series)
              for series in (real, synthetic)]
    values = column.min + edges * (column.max - column.min)
    if column.role == DATE_ROLE:
        days = pd.to_timedelta(np.rint(values), unit="D")
        labels = [day.date().isoformat() for day in pd.Timestamp(column.origin) + days]
    else:
        labels = values.tolist()
    return Histogram(column.name, column.role, labels, *(s.tolist() for s in shares))


def unseen_pair_share(frame: pd.DataFrame, train: pd.DataFrame, names: list[str]) -> float:
    """Share of rows with at least one pair of categorical values never seen together in ``train``."""
    unseen = np.zeros(len(frame), dtype=bool)
    for a, b in combinations(names, 2):
        seen = pd.MultiIndex.from_arrays([train[a], train[b]]).unique()
        unseen |= ~pd.MultiIndex.from_arrays([frame[a], frame[b]]).isin(seen)
    return float(unseen.mean()) if len(frame) else 0.0


# --------------------------------------------------------------------------------------------
# Groups of real people and synthetic slices
# --------------------------------------------------------------------------------------------

def check_frame(frame: pd.DataFrame, contract: SchemaContract, name: str) -> None:
    """Raise ``ValueError`` when ``frame`` is empty or misses a contract column."""
    if len(frame) == 0:
        raise ValueError(f"{name} has no rows")
    missing = [c.name for c in contract.columns if c.name not in frame.columns]
    if missing:
        raise ValueError(f"{name} misses the columns {missing}")


def at_most(frame: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """A seeded random subset of at most ``n_rows`` rows, with a fresh index."""
    subset = frame.sample(n=n_rows, random_state=seed) if len(frame) > n_rows else frame
    return subset.reset_index(drop=True)


def group_size(n_holdout: int, n_train: int, n_synthetic: int, max_rows: int) -> int:
    """Rows of every group and slice: the holdout, a train slice and a synthetic slice all fit."""
    return min(n_holdout, n_train // TRAIN_SLICES, n_synthetic // SYNTHETIC_SLICES, max_rows)


def disjoint_slices(frame: pd.DataFrame, size: int, count: int, seed: int) -> list[pd.DataFrame]:
    """``count`` seeded random slices of ``size`` rows, no row in two slices."""
    order = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    return [order.iloc[i * size:(i + 1) * size].reset_index(drop=True) for i in range(count)]


def real_groups(holdout: pd.DataFrame, train: pd.DataFrame, size: int, seed: int) -> list[pd.DataFrame]:
    """The groups of real people: a subset of the holdout first, then ``TRAIN_SLICES`` train slices."""
    return [at_most(holdout, size, seed), *disjoint_slices(train, size, TRAIN_SLICES, seed)]


def synthetic_slices(synthetic: pd.DataFrame, size: int, seed: int) -> list[pd.DataFrame]:
    """``SYNTHETIC_SLICES`` disjoint slices of the synthetic rows, each compared with the holdout group."""
    return disjoint_slices(synthetic, size, SYNTHETIC_SLICES, seed)


# --------------------------------------------------------------------------------------------
# Measures against the baseline
# --------------------------------------------------------------------------------------------

def against_baseline(distances: Sequence[float], pairs: Sequence[float]) -> tuple[float, float, float, float, float]:
    """Distance (mean over the slices), baseline and spread (over the pairs), z and excess."""
    distance, baseline = float(np.mean(distances)), float(np.mean(pairs))
    spread = float(np.std(pairs, ddof=1))
    return distance, baseline, spread, (distance - baseline) / max(spread, MIN_BASELINE), distance - baseline


def group_distances(a: pd.DataFrame, b: pd.DataFrame, contract: SchemaContract) -> dict[str, float]:
    """Distance of every contract column between two groups of rows."""
    return {c.name: column_distance(a[c.name], b[c.name], c) for c in contract.columns}


def kind_measure(names: list[str], synthetic: list[dict[str, float]], pairs: list[dict[str, float]]) -> Measure:
    """Mean distance of some columns, slice by slice and pair by pair (``NO_MEASURE`` when none)."""
    if not names:
        return NO_MEASURE
    mean = lambda distances: float(np.mean([distances[name] for name in names]))   # noqa: E731
    return Measure(*against_baseline([mean(d) for d in synthetic], [mean(d) for d in pairs]))


def association_difference(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Mean |difference| off the diagonal of two association matrices."""
    return float(off_diagonal((a - b).abs()).mean())


def beyond(measure: Measure, excess_limit: float, limits: FidelityLimits) -> bool:
    """A difference both real and large (NaN compares False: no warning for a kind the data do not have)."""
    return measure.z > limits.min_z and measure.excess > excess_limit


def fidelity_warnings(columns: list[ColumnFidelity], measures: dict[str, Measure],
                      worst_pairs: list[tuple[str, str]], limits: FidelityLimits) -> list[Issue]:
    """Plain-language warnings for the measures that differ beyond chance by more than their limit.

    The columns named are the ones with the largest excess: z also grows when two real groups hardly differ
    (a column almost constant), which would name columns whose difference is tiny.
    """
    warnings = []
    excess_limits = {"numeric": limits.numeric_excess, CATEGORICAL_ROLE: limits.categorical_excess}
    kinds = [kind for kind, limit in excess_limits.items() if beyond(measures[kind], limit, limits)]
    if kinds:
        candidates = [c for c in columns
                      if (c.role == CATEGORICAL_ROLE and CATEGORICAL_ROLE in kinds)
                      or (c.role != CATEGORICAL_ROLE and "numeric" in kinds)]
        worst = sorted(candidates, key=lambda c: c.excess, reverse=True)[:MAX_LISTED_COLUMNS]
        warnings.append(Issue(
            "distributions_differ",
            "Some columns of the synthetic data are spread differently from the real ones "
            f"(most: {', '.join(c.name for c in worst)}). Charts of these columns may be less reliable.",
            worst[0].name,
        ))
    if beyond(measures["association"], limits.association_excess, limits):
        pairs = ", ".join(f"{a} and {b}" for a, b in worst_pairs)
        warnings.append(Issue(
            "relationships_differ",
            "Some relationships between columns are different in the synthetic data than in the real ones "
            f"(most: {pairs}). Charts that cross these columns may be less reliable.",
        ))
    return warnings


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------

def fidelity_report(synthetic: pd.DataFrame, holdout: pd.DataFrame, train: pd.DataFrame, contract: SchemaContract,
                    *, max_rows: int = FIDELITY_MAX_ROWS, seed: int = RANDOM_STATE,
                    limits: FidelityLimits = FidelityLimits()) -> FidelityReport:
    """Compare the synthetic rows with the holdout, against the differences between groups of real people.

    Args:
        synthetic: Decoded synthetic rows (the app makes as many as the train rows).
        holdout: Real rows the VAE never saw (``prepare(...).holdout``).
        train: Real rows the VAE learned from (the train slices of the baseline and the pairs seen).
        contract: Schema contract of the encoder.
        max_rows: Rows per group at most; the size is ``group_size(...)``.
        seed: Seed of the groups and slices.
        limits: When a summary measure gives a warning (z and excess, see ``FidelityLimits``).

    Returns:
        The report (warnings never block).

    Raises:
        ValueError: An empty frame, a missing contract column, or too few rows for the groups.
    """
    for frame, name in ((synthetic, "synthetic"), (holdout, "holdout"), (train, "train")):
        check_frame(frame, contract, name)
    names = [c.name for c in contract.columns]
    size = group_size(len(holdout), len(train), len(synthetic), max_rows)
    if size == 0:
        raise ValueError(f"too few rows for {REAL_GROUPS} groups of real people and {SYNTHETIC_SLICES} "
                         "synthetic slices")
    groups = real_groups(comparable(holdout[names], contract), comparable(train[names], contract), size, seed)
    slices = synthetic_slices(synthetic[names], size, seed)
    reference = groups[0]

    pairs = [group_distances(a, b, contract) for a, b in combinations(groups, 2)]
    compared = [group_distances(reference, s, contract) for s in slices]
    columns = [ColumnFidelity(c.name, c.role, *against_baseline([d[c.name] for d in compared],
                                                                [d[c.name] for d in pairs]))
               for c in contract.columns]
    scalar = [c.name for c in contract.columns if c.role != CATEGORICAL_ROLE]
    categorical = [c.name for c in contract.columns if c.role == CATEGORICAL_ROLE]

    real_matrices = [association_matrix(g, contract) for g in groups]
    slice_matrices = [association_matrix(s, contract) for s in slices]
    association = NO_MEASURE if len(names) < 2 else Measure(*against_baseline(
        [association_difference(m, real_matrices[0]) for m in slice_matrices],
        [association_difference(a, b) for a, b in combinations(real_matrices, 2)]))
    association_synthetic = sum(slice_matrices) / len(slice_matrices)
    difference = (association_synthetic - real_matrices[0]).abs()
    upper = list(combinations(names, 2))
    worst_pairs = sorted(upper, key=lambda pair: difference.loc[pair], reverse=True)[:MAX_LISTED_COLUMNS]

    measures = {"numeric": kind_measure(scalar, compared, pairs),
                CATEGORICAL_ROLE: kind_measure(categorical, compared, pairs), "association": association}
    every_slice = pd.concat(slices, ignore_index=True)
    seen_rows = comparable(train[categorical], contract) if categorical else train[[]]

    return FidelityReport(
        n_compared=size,
        columns=columns,
        numeric=measures["numeric"],
        categorical=measures[CATEGORICAL_ROLE],
        association=association,
        association_diff_max=float(off_diagonal(difference).max()) if upper else 0.0,
        association_real=real_matrices[0],
        association_synthetic=association_synthetic,
        histograms=[histogram(reference[c.name], every_slice[c.name], c) for c in contract.columns],
        unseen_pairs_synthetic=unseen_pair_share(every_slice, seen_rows, categorical),
        unseen_pairs_real=unseen_pair_share(reference, seen_rows, categorical),
        warnings=fidelity_warnings(columns, measures, worst_pairs, limits),
    )
