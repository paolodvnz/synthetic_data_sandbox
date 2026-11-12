"""Specification tests for the fidelity report: how close the synthetic data are to real people.

Everything is measured against the **holdout** (real rows the VAE never saw): against the train rows a model
that copies would look perfect. Each distance is read against a **baseline** of real people: the real rows are cut
into REAL_GROUPS disjoint groups of the same size (the holdout + TRAIN_SLICES slices of the train) and the same
distance is measured on every pair of groups (10 pairs). Their mean is the baseline, their standard deviation the
spread: the difference chance alone produces, and how much it varies.

Distances, all in [0, 1], lower = closer:
    numeric / date: Wasserstein distance (W1) on the [0, 1] scale of the train range
    categorical:    total variation distance between the level shares
    associations:   |synthetic - real| of the pairwise association matrix (|Pearson|, Cramer's V, eta)

Public API under test:
    comparable(frame, contract) -> pd.DataFrame
        real rows written like synthetic ones: merged rare text levels -> "Other", integer codes as int64
    numeric_distance, total_variation, cramers_v, correlation_ratio, association_matrix
    group_size(n_holdout, n_train, n_synthetic, max_rows) -> int
        rows per group: min(n_holdout, n_train // TRAIN_SLICES, n_synthetic // SYNTHETIC_SLICES, max_rows)
    real_groups(holdout, train, size, seed) -> list[pd.DataFrame]
        a seeded subset of the holdout first, then TRAIN_SLICES disjoint train slices, all of ``size`` rows
    synthetic_slices(synthetic, size, seed) -> list[pd.DataFrame]: SYNTHETIC_SLICES disjoint slices
    fidelity_report(synthetic, holdout, train, contract, *, max_rows=FIDELITY_MAX_ROWS, seed=RANDOM_STATE,
                    limits=FidelityLimits()) -> FidelityReport
        FidelityReport(n_compared, columns, numeric, categorical, association, association_diff_max,
                       association_real, association_synthetic, histograms, unseen_pairs_synthetic,
                       unseen_pairs_real, warnings)
        columns: ColumnFidelity(name, role, distance, baseline, spread, z, excess) per contract column
        numeric / categorical / association: Measure(distance, baseline, spread, z, excess); NaN without such
            columns. distance: mean over the synthetic slices of the distance from the holdout group;
            baseline / spread: mean / standard deviation over the pairs of real groups (a summary measure
            averages the columns pair by pair); z = (distance - baseline) / max(spread, MIN_BASELINE);
            excess = distance - baseline
        association_real: matrix of the holdout group; association_synthetic: mean of the slice matrices
        histograms: Histogram(name, role, labels, real, synthetic): shares on common bins (numeric: the train
                    range in HISTOGRAM_BINS bins, labels are the edges; categorical: the levels)
        unseen_pairs_*: share of rows with a pair of categorical values never seen together in train
        warnings (not blocking), when a measure is both real (z > limits.min_z) and large (excess > its limit):
        "distributions_differ" (numeric or categorical; columns named by excess, the first is the issue's
        column), "relationships_differ"
        n_compared: rows per group; too few rows for the groups is a ValueError
"""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from sds.data_engine import SchemaContract
from sds.data_engine.contract import OTHER_LABEL, ContractColumn
from sds.config import RANDOM_STATE
from sds.validation.fidelity import (
    FidelityLimits,
    association_matrix,
    column_distance,
    comparable,
    correlation_ratio,
    cramers_v,
    fidelity_report,
    group_size,
    numeric_distance,
    off_diagonal,
    real_groups,
    synthetic_slices,
    total_variation,
)
from sds.validation.fidelity_config import (
    FIDELITY_MAX_ROWS,
    HISTOGRAM_BINS,
    MIN_BASELINE,
    REAL_GROUPS,
    SYNTHETIC_SLICES,
    TRAIN_SLICES,
)

N_ROWS = 2000                      # holdout rows; train and synthetic rows are TRAIN_SLICES times as many
SAME_DISTRIBUTION_Z = 3.0          # a fresh sample of the true distribution stays well within this

CONTRACT = SchemaContract(version=2, width=12, columns=[
    ContractColumn("age", "numeric", 0, 1, min=17.0, max=90.0, integer=True),
    ContractColumn("bmi", "numeric", 1, 2, min=15.0, max=60.0, integer=False),
    ContractColumn("visit", "date", 2, 3, min=0.0, max=365.0, origin="2020-01-01"),
    ContractColumn("sex", "categorical", 3, 5, levels=("F", "M")),
    ContractColumn("grade", "categorical", 5, 9, levels=("a", "b", "c"), has_other=True, merged=(("z", 5),)),
    ContractColumn("code", "categorical", 9, 12, levels=(1, 2), integer=True, has_other=True, merged=((7, 5),)),
], dropped=[], rows={}, issues=[])
NUMERIC_ONLY = SchemaContract(version=2, width=2, columns=CONTRACT.columns[:2], dropped=[], rows={}, issues=[])


def toy_frame(n_rows: int, seed: int) -> pd.DataFrame:
    """Real-like rows with known relations: bmi grows with age, grade depends on sex."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(50, 12, n_rows), 17, 90).round().astype(np.int64)
    bmi = np.clip(15 + 0.3 * age + rng.normal(0, 3, n_rows), 15, 60)
    sex = np.where(rng.random(n_rows) < 0.6, "F", "M")
    grade = np.where(sex == "F", rng.choice(["a", "b"], n_rows, p=[0.8, 0.2]),
                     rng.choice(["b", "c"], n_rows, p=[0.3, 0.7]))
    code = rng.choice([1, 2, 7], n_rows, p=[0.5, 0.45, 0.05])
    visit = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 366, n_rows), unit="D")
    return pd.DataFrame({"age": age, "bmi": bmi, "visit": visit, "sex": sex, "grade": grade, "code": code})


@pytest.fixture(scope="module")
def real():
    return toy_frame(N_ROWS, seed=1), toy_frame(TRAIN_SLICES * N_ROWS, seed=2)    # holdout, train (as in the app)


def fresh(seed: int = 3) -> pd.DataFrame:
    """Synthetic rows as the app makes them, as many as the train, here drawn from the true distribution."""
    return toy_frame(TRAIN_SLICES * N_ROWS, seed=seed)


@pytest.fixture(scope="module")
def faithful(real):
    holdout, train = real
    return fidelity_report(fresh(), holdout, train, CONTRACT)


def by_name(report) -> dict:
    return {c.name: c for c in report.columns}


# --------------------------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------------------------

def test_numeric_distance_is_w1_on_the_train_scale():
    age = CONTRACT.columns[0]
    real, synthetic = pd.Series([20] * 10), pd.Series([27.3] * 10)
    assert numeric_distance(real, synthetic, age) == pytest.approx(7.3 / 73)


def test_date_distance_is_in_days_on_the_train_scale():
    visit = CONTRACT.columns[2]
    real = pd.Series(pd.to_datetime(["2020-01-11"] * 4))
    synthetic = pd.Series(pd.to_datetime(["2020-02-16"] * 4))
    assert numeric_distance(real, synthetic, visit) == pytest.approx(36 / 365)


def test_total_variation_by_hand():
    assert total_variation(pd.Series(list("aabb")), pd.Series(list("abbb"))) == pytest.approx(0.25)
    assert total_variation(pd.Series(list("aa")), pd.Series(list("bb"))) == pytest.approx(1.0)


def test_cramers_v_of_dependent_and_independent_columns():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.choice(list("xyz"), 5000))
    assert cramers_v(a, a.map({"x": 1, "y": 2, "z": 3})) == pytest.approx(1.0)
    assert cramers_v(a, pd.Series(rng.choice(list("pq"), 5000))) < 0.05
    assert cramers_v(a, pd.Series(["p"] * 5000)) == 0.0


def test_correlation_ratio_of_determined_and_constant_values():
    categories = pd.Series(list("aabb") * 50)
    assert correlation_ratio(categories, np.where(categories == "a", 0.2, 0.9)) == pytest.approx(1.0)
    assert correlation_ratio(categories, np.full(200, 0.5)) == 0.0
    # By hand: a -> 0, 1 and b -> 2, 3: total spread 5, between groups 4, eta = sqrt(4 / 5).
    assert correlation_ratio(pd.Series(list("aabb")), np.array([0.0, 1, 2, 3])) == pytest.approx(np.sqrt(0.8))


def test_association_matrix_is_symmetric_with_unit_diagonal(real):
    matrix = association_matrix(comparable(real[0], CONTRACT), CONTRACT)
    names = [c.name for c in CONTRACT.columns]
    assert list(matrix.index) == names and list(matrix.columns) == names
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
    assert np.allclose(np.diag(matrix), 1)
    assert ((matrix >= 0) & (matrix <= 1)).all().all()
    holdout = real[0]
    assert matrix.loc["age", "bmi"] == pytest.approx(abs(np.corrcoef(holdout["age"], holdout["bmi"])[0, 1]))
    assert matrix.loc["sex", "grade"] > 0.5                       # grade depends on sex
    assert matrix.loc["age", "sex"] < 0.1                         # independent


# --------------------------------------------------------------------------------------------
# Real rows written like synthetic ones
# --------------------------------------------------------------------------------------------

def test_merged_text_levels_become_other():
    frame = pd.DataFrame({"grade": ["a", "z", "c"], "code": [1, 7, 2]})
    assert comparable(frame, CONTRACT)["grade"].tolist() == ["a", OTHER_LABEL, "c"]
    assert comparable(frame, CONTRACT)["code"].tolist() == [1, 7, 2]        # integers keep their merged values
    as_float = comparable(frame.assign(code=frame["code"].astype(float)), CONTRACT)["code"]
    assert as_float.dtype == np.int64                                        # 7.0 -> 7, like the synthetic rows


def test_integer_codes_compare_equal_whatever_the_float_type(real):
    # Regression: a column with missing values in the file is float in the real rows.
    holdout, train = (frame.assign(code=frame["code"].astype(float)) for frame in real)
    report = fidelity_report(fresh(), holdout, train, CONTRACT)
    assert by_name(report)["code"].z < SAME_DISTRIBUTION_Z
    assert report.unseen_pairs_synthetic == 0 and report.unseen_pairs_real == 0


def test_the_input_frame_is_not_changed(real):
    holdout, _ = real
    before = holdout.copy()
    comparable(holdout, CONTRACT)
    pd.testing.assert_frame_equal(holdout, before)


# --------------------------------------------------------------------------------------------
# Report: real groups, baseline, spread, z and excess
# --------------------------------------------------------------------------------------------

def groups_of(real, size):
    """The real groups of the report, rebuilt the way the report builds them."""
    holdout, train = real
    return real_groups(comparable(holdout, CONTRACT), comparable(train, CONTRACT), size, RANDOM_STATE)


def test_group_size_fits_every_group():
    assert group_size(2000, TRAIN_SLICES * 2000, SYNTHETIC_SLICES * 2000, FIDELITY_MAX_ROWS) == 2000
    assert group_size(2000, TRAIN_SLICES * 1500, SYNTHETIC_SLICES * 2000, FIDELITY_MAX_ROWS) == 1500
    assert group_size(2000, TRAIN_SLICES * 2000, SYNTHETIC_SLICES * 1000, FIDELITY_MAX_ROWS) == 1000
    assert group_size(2000, TRAIN_SLICES * 2000, SYNTHETIC_SLICES * 2000, 300) == 300


def test_real_groups_are_the_holdout_and_disjoint_train_slices():
    holdout = pd.DataFrame({"id": range(1000)})
    train = pd.DataFrame({"id": range(1000, 5000)})
    groups = real_groups(holdout, train, 300, seed=0)
    assert len(groups) == REAL_GROUPS == 1 + TRAIN_SLICES
    assert all(len(group) == 300 for group in groups)
    assert set(groups[0]["id"]) <= set(holdout["id"])
    slices = [set(group["id"]) for group in groups[1:]]
    assert all(ids <= set(train["id"]) for ids in slices)
    assert len(set().union(*slices)) == TRAIN_SLICES * 300                 # no row in two slices


def test_baseline_and_spread_come_from_every_pair_of_real_groups(real, faithful):
    groups = groups_of(real, faithful.n_compared)
    for column, result in zip(CONTRACT.columns, faithful.columns):
        pairs = [column_distance(a[column.name], b[column.name], column) for a, b in combinations(groups, 2)]
        assert len(pairs) == REAL_GROUPS * (REAL_GROUPS - 1) // 2
        assert result.baseline == pytest.approx(np.mean(pairs))
        assert result.spread == pytest.approx(np.std(pairs, ddof=1))


def test_the_synthetic_distance_is_the_mean_over_its_slices(real, faithful):
    size = faithful.n_compared
    holdout_group = groups_of(real, size)[0]
    slices = synthetic_slices(fresh(), size, RANDOM_STATE)
    assert len(slices) == SYNTHETIC_SLICES and all(len(s) == size for s in slices)
    for column, result in zip(CONTRACT.columns, faithful.columns):
        by_hand = np.mean([column_distance(holdout_group[column.name], s[column.name], column) for s in slices])
        assert result.distance == pytest.approx(by_hand)


def test_z_and_excess_of_every_column(faithful):
    for column in faithful.columns:
        assert column.excess == pytest.approx(column.distance - column.baseline)
        assert column.z == pytest.approx((column.distance - column.baseline) / max(column.spread, MIN_BASELINE))


def test_summary_measures_average_the_columns_pair_by_pair(real, faithful):
    groups = groups_of(real, faithful.n_compared)
    scalar = [c for c in CONTRACT.columns if c.role != "categorical"]
    per_pair = [np.mean([column_distance(a[c.name], b[c.name], c) for c in scalar]) for a, b in combinations(groups, 2)]
    numeric = faithful.numeric
    assert numeric.baseline == pytest.approx(np.mean(per_pair))
    assert numeric.spread == pytest.approx(np.std(per_pair, ddof=1))
    assert numeric.distance == pytest.approx(np.mean([c.distance for c in faithful.columns if c.role != "categorical"]))
    assert numeric.excess == pytest.approx(numeric.distance - numeric.baseline)
    assert numeric.z == pytest.approx((numeric.distance - numeric.baseline) / max(numeric.spread, MIN_BASELINE))


def test_the_association_baseline_compares_the_matrices_pair_by_pair(real, faithful):
    matrices = [association_matrix(g, CONTRACT) for g in groups_of(real, faithful.n_compared)]
    per_pair = [off_diagonal((a - b).abs()).mean() for a, b in combinations(matrices, 2)]
    assert faithful.association.baseline == pytest.approx(np.mean(per_pair))
    assert faithful.association.spread == pytest.approx(np.std(per_pair, ddof=1))


def test_a_fresh_sample_of_the_same_people_is_as_close_as_real_data(faithful):
    for measure in (faithful.numeric, faithful.categorical, faithful.association):
        assert measure.z < SAME_DISTRIBUTION_Z
    assert faithful.warnings == []


def test_shifted_numbers_are_flagged_with_the_worst_column(real):
    holdout, train = real
    synthetic = fresh()
    synthetic["bmi"] = np.clip(synthetic["bmi"] + 10, 15, 60)          # not the first column: the worst is named
    report = fidelity_report(synthetic, holdout, train, CONTRACT)
    assert report.numeric.z > FidelityLimits().min_z
    assert [(w.code, w.column) for w in report.warnings] == [("distributions_differ", "bmi")]
    assert report.warnings[0].message


def test_a_wrong_level_share_is_flagged(real):
    holdout, train = real
    report = fidelity_report(fresh().assign(sex="F"), holdout, train, CONTRACT)
    assert report.categorical.z > FidelityLimits().min_z
    assert "distributions_differ" in [w.code for w in report.warnings]


def test_columns_are_named_by_excess_not_by_z(real):
    # sex almost always "F": two real groups hardly differ, so a small synthetic error has the larger z;
    # grade moves 10% of its rows: a larger error with a smaller z. The warning names grade first.
    holdout, train = (frame.copy() for frame in real)
    rng = np.random.default_rng(0)
    for frame in (holdout, train):
        frame["sex"] = np.where(rng.random(len(frame)) < 0.002, "M", "F")
    synthetic = fresh()
    synthetic["sex"] = np.where(rng.random(len(synthetic)) < 0.05, "M", "F")
    moved = rng.random(len(synthetic)) < 0.10
    synthetic.loc[moved, "grade"] = "c"
    report = fidelity_report(synthetic, holdout, train, CONTRACT)
    sex, grade = by_name(report)["sex"], by_name(report)["grade"]
    assert sex.z > grade.z and grade.excess > sex.excess                  # the case the test is about
    (warning,) = [w for w in report.warnings if w.code == "distributions_differ"]
    assert warning.column == "grade"
    assert warning.message.index("grade") < warning.message.index("sex")


def test_a_real_but_small_difference_is_not_flagged(real):
    # bmi shifted by 2 (4% of its range): clearly beyond chance on 2,000 rows per group, but small in the mean.
    holdout, train = real
    synthetic = fresh()
    synthetic["bmi"] = np.clip(synthetic["bmi"] + 2, 15, 60)
    report = fidelity_report(synthetic, holdout, train, CONTRACT)
    assert report.numeric.z > FidelityLimits().min_z
    assert report.numeric.excess < FidelityLimits().numeric_excess
    assert report.warnings == []


@pytest.mark.parametrize("limits", [FidelityLimits(min_z=1e6), FidelityLimits(numeric_excess=1.0)])
def test_a_warning_needs_both_a_real_and_a_large_difference(real, limits):
    holdout, train = real
    synthetic = fresh()
    synthetic["bmi"] = np.clip(synthetic["bmi"] + 10, 15, 60)
    report = fidelity_report(synthetic, holdout, train, CONTRACT, limits=limits)
    assert "distributions_differ" not in [w.code for w in report.warnings]


def test_lost_relationships_are_flagged_but_not_the_distributions(real):
    # Every column shuffled on its own: same distributions, no relation left between columns.
    holdout, train = real
    synthetic = fresh()
    rng = np.random.default_rng(0)
    for name in synthetic.columns:
        synthetic[name] = rng.permutation(synthetic[name].to_numpy())
    report = fidelity_report(synthetic, holdout, train, CONTRACT)
    assert report.association.z > FidelityLimits().min_z
    assert [w.code for w in report.warnings] == ["relationships_differ"]


def test_identical_real_groups_give_zero_spread_and_a_finite_z():
    # Every real row the same: no chance difference, baseline and spread are 0 and the floor keeps z finite.
    same = pd.DataFrame({"age": 50, "bmi": 30.0, "visit": pd.Timestamp("2020-06-01"), "sex": "F", "grade": "a",
                         "code": 1}, index=range(N_ROWS))
    report = fidelity_report(fresh(), same, pd.concat([same] * TRAIN_SLICES, ignore_index=True), CONTRACT)
    assert all(c.baseline == 0 and c.spread == 0 and np.isfinite(c.z) for c in report.columns)
    assert report.association.baseline == 0 and report.association.spread == 0
    assert np.isfinite(report.association.z)


def test_a_kind_of_column_the_data_do_not_have_gives_nan_and_no_warning(real):
    holdout, train = real
    columns = ["age", "bmi"]
    report = fidelity_report(holdout[columns] + 0, holdout[columns], train[columns], NUMERIC_ONLY)
    categorical = report.categorical
    assert all(np.isnan(v) for v in (categorical.distance, categorical.baseline, categorical.spread,
                                     categorical.z, categorical.excess))
    assert report.unseen_pairs_synthetic == 0 and report.unseen_pairs_real == 0
    assert report.warnings == []


def test_too_few_rows_for_the_groups_is_an_error(real):
    holdout, train = real
    with pytest.raises(ValueError, match="rows"):
        fidelity_report(fresh(), holdout, train.head(TRAIN_SLICES - 1), CONTRACT)


# --------------------------------------------------------------------------------------------
# Sizes and seed
# --------------------------------------------------------------------------------------------

def test_frames_are_compared_on_the_holdout_size(real):
    holdout, train = real
    report = fidelity_report(toy_frame(3 * N_ROWS, seed=3), holdout.head(500), train, CONTRACT)
    assert report.n_compared == 500


def test_max_rows_caps_the_comparison(real):
    holdout, train = real
    assert fidelity_report(toy_frame(N_ROWS, seed=3), holdout, train, CONTRACT, max_rows=300).n_compared == 300
    assert FIDELITY_MAX_ROWS >= 10_000                   # the whole holdout (at most 10,000 rows) by default


def test_the_report_is_seeded(real):
    holdout, train = real
    synthetic = toy_frame(3 * N_ROWS, seed=3)
    first = fidelity_report(synthetic, holdout, train, CONTRACT, seed=5)
    again = fidelity_report(synthetic, holdout, train, CONTRACT, seed=5)
    other = fidelity_report(synthetic, holdout, train, CONTRACT, seed=6)
    assert [c.distance for c in first.columns] == [c.distance for c in again.columns]
    assert [c.distance for c in first.columns] != [c.distance for c in other.columns]


@pytest.mark.parametrize("which", ["synthetic", "holdout", "train"])
def test_a_missing_column_is_an_error(real, which):
    holdout, train = real
    frames = {"synthetic": toy_frame(N_ROWS, seed=3), "holdout": holdout, "train": train}
    frames[which] = frames[which].drop(columns="bmi")
    with pytest.raises(ValueError, match="bmi"):
        fidelity_report(frames["synthetic"], frames["holdout"], frames["train"], CONTRACT)


@pytest.mark.parametrize("which", ["synthetic", "holdout", "train"])
def test_empty_frames_are_an_error(real, which):
    holdout, train = real
    frames = {"synthetic": toy_frame(N_ROWS, seed=3), "holdout": holdout, "train": train}
    frames[which] = frames[which].head(0)
    with pytest.raises(ValueError, match=which):
        fidelity_report(frames["synthetic"], frames["holdout"], frames["train"], CONTRACT)


# --------------------------------------------------------------------------------------------
# Heatmap, histograms, unseen pairs
# --------------------------------------------------------------------------------------------

def test_association_matrices_for_the_heatmap(faithful):
    names = [c.name for c in CONTRACT.columns]
    for matrix in (faithful.association_real, faithful.association_synthetic):
        assert list(matrix.index) == names and list(matrix.columns) == names
    difference = (faithful.association_synthetic - faithful.association_real).abs()
    assert faithful.association_diff_max == pytest.approx(off_diagonal(difference).max())
    # The mean of the slice matrices differs less than the slices one by one (triangle inequality).
    assert off_diagonal(difference).mean() <= faithful.association.distance + 1e-12


def test_numeric_histograms_share_the_train_range_bins(faithful):
    histograms = {h.name: h for h in faithful.histograms}
    assert list(histograms) == [c.name for c in CONTRACT.columns]
    age = histograms["age"]
    assert len(age.labels) == HISTOGRAM_BINS + 1
    assert age.labels[0] == 17.0 and age.labels[-1] == 90.0
    assert len(age.real) == len(age.synthetic) == HISTOGRAM_BINS
    assert sum(age.real) == pytest.approx(1) and sum(age.synthetic) == pytest.approx(1)
    visit = histograms["visit"]
    assert visit.labels[0] == "2020-01-01" and visit.labels[-1] == "2020-12-31"


def test_categorical_histograms_list_the_levels(faithful):
    histograms = {h.name: h for h in faithful.histograms}
    assert histograms["grade"].labels == ["a", "b", "c", OTHER_LABEL]
    assert histograms["code"].labels == [1, 2, 7]                   # merged integer values are real values
    for name in ("sex", "grade", "code"):
        assert sum(histograms[name].real) == pytest.approx(1)
        assert sum(histograms[name].synthetic) == pytest.approx(1)


def test_values_outside_the_train_range_fall_in_the_edge_bins(real):
    holdout, train = real
    far = holdout.assign(age=200)                  # a holdout outlier, not a synthetic value
    histogram = {h.name: h for h in fidelity_report(holdout, far, train, CONTRACT).histograms}["age"]
    assert histogram.real[-1] == pytest.approx(1)


def test_unseen_pairs_are_counted_against_train(real):
    holdout, train = real
    # In train "F" never meets "c": synthetic rows (F, c) are pairs no real row has.
    synthetic = fresh()
    contradictory = synthetic.index[:800]
    synthetic.loc[contradictory, ["sex", "grade"]] = ["F", "c"]
    report = fidelity_report(synthetic, holdout, train, CONTRACT)
    assert report.unseen_pairs_synthetic == pytest.approx(800 / len(synthetic), abs=0.01)   # every slice counted
    assert report.unseen_pairs_real < 0.01
    # The real reference is counted against train too: holdout rows (F, c) are rare crossings of real people.
    rare = holdout.copy()
    rare.loc[rare.index[:100], ["sex", "grade"]] = ["F", "c"]
    assert fidelity_report(synthetic, rare, train, CONTRACT).unseen_pairs_real == pytest.approx(100 / N_ROWS, abs=0.005)


# --------------------------------------------------------------------------------------------
# What the UI draws uses every synthetic slice
# --------------------------------------------------------------------------------------------

def one_odd_row(column: str, value) -> pd.DataFrame:
    """Synthetic rows as many as the train, one of them with ``value`` in ``column``."""
    synthetic = fresh()
    synthetic.loc[0, column] = value
    return synthetic


def test_the_synthetic_heatmap_is_the_mean_of_the_slice_matrices(real, faithful):
    slices = synthetic_slices(fresh(), faithful.n_compared, RANDOM_STATE)
    by_hand = sum(association_matrix(s, CONTRACT) for s in slices) / SYNTHETIC_SLICES
    pd.testing.assert_frame_equal(faithful.association_synthetic, by_hand)


def test_histograms_count_every_synthetic_slice(real):
    holdout, train = real
    synthetic = one_odd_row("age", 90).assign(age=lambda f: np.where(f.index == 0, 90, 50))   # one row in the last bin
    report = fidelity_report(synthetic, holdout, train, CONTRACT)
    age = {h.name: h for h in report.histograms}["age"]
    assert age.synthetic[-1] == pytest.approx(1 / (SYNTHETIC_SLICES * report.n_compared))


def test_unseen_pairs_count_every_synthetic_slice(real):
    holdout, train = real
    synthetic = one_odd_row("sex", "F")
    synthetic.loc[0, "grade"] = "c"                     # "F" never meets "c" in train: one unseen row
    report = fidelity_report(synthetic, holdout, train, CONTRACT)
    assert report.unseen_pairs_synthetic == pytest.approx(1 / (SYNTHETIC_SLICES * report.n_compared))
