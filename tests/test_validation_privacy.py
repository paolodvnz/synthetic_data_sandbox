"""Specification tests for the privacy check: exact copies + DCR (distance to closest record).

All rows are encoded vectors: train and holdout through ``TabularEncoder.transform``, the synthetic rows
through the same encoder **after decoding**, i.e. what the user actually takes away.

Distance: Gower on the encoded vector, one term per contract column, averaged over the columns:
    numeric / date: |a - b| on the MinMax scale (not clipped: a holdout row outside the train range
                    is genuinely far, and a larger reference makes the gate stricter)
    categorical:    1 - a_g . b_g  (0 same level, 1 different; an all-zero holdout group, i.e. an unseen
                    level without an "Other" slot, differs from every level)

Gate, against the holdout -> train reference (rows the VAE never saw):
    copied_rows: share of synthetic rows at distance 0 > exact_share_limit, where
                 exact_share_limit = p + exact_tolerance + NOISE_Z * sqrt(p (1 - p) (1/n_synthetic + 1/n_holdout))
                 and p is the holdout share at distance 0 (relative, so data with genuine duplicates, like
                 CDC, do not fail by themselves; the noise term absorbs the sampling noise of both shares)
    too_close:   DCR_PERCENTILE-th percentile of the synthetic DCR < min_ratio * the holdout one

Public API under test:
    closest_record_distances(queries, reference, contract, *, batch_size=DCR_BATCH_SIZE) -> np.ndarray
        shape (len(queries),); ValueError on empty inputs or a width different from the contract
    privacy_check(synthetic, train, holdout, contract, *, max_rows=DCR_MAX_ROWS, seed=RANDOM_STATE,
                  min_ratio=MIN_DCR_RATIO, exact_tolerance=EXACT_SHARE_TOLERANCE) -> PrivacyReport
        PrivacyReport(passed, n_compared, synthetic_percentile, holdout_percentile, synthetic_exact_share,
                      holdout_exact_share, exact_share_limit, train_closer_share,
                      expected_train_closer_share, failures, warnings)
        failures: list of Issue with codes "copied_rows" / "too_close" and a plain-language message;
        warnings: not blocking; "near_copies_unchecked" when the holdout percentile is 0 (too_close blind);
        more than max_rows synthetic rows -> a seeded random subset of max_rows is compared
"""

import numpy as np
import pytest

from sds.data_engine import SchemaContract
from sds.data_engine.contract import ContractColumn
from sds.data_engine.thresholds_config import MAX_HOLDOUT_ROWS
from sds.validation import privacy
from sds.validation.privacy import PrivacyReport, closest_record_distances, privacy_check
from sds.validation.privacy_config import (
    DCR_MAX_ROWS,
    DCR_PERCENTILE,
    EXACT_DISTANCE,
    EXACT_SHARE_TOLERANCE,
    MIN_DCR_RATIO,
    NOISE_Z,
)

ATOL = 1e-6
N_TRAIN = 2000
N_HOLDOUT = 1000
JITTER = 1e-3                  # a "memorised + tiny noise" model: no exact copy, but far too close
COPIED_SHARE = 0.10            # a model that memorised one row in ten

CONTRACT = SchemaContract(version=1, width=7, columns=[
    ContractColumn("age", "numeric", 0, 1, min=17.0, max=90.0, integer=True),
    ContractColumn("sex", "categorical", 1, 3, levels=("F", "M")),
    ContractColumn("bmi", "numeric", 3, 4, min=15.0, max=60.0, integer=False),
    ContractColumn("grade", "categorical", 4, 7, levels=("a", "b", "c")),
], dropped=[], rows={}, issues=[])
N_COLUMNS = len(CONTRACT.columns)

# Only binary columns: few distinct rows, so different people often coincide (like CDC, ~70% here).
N_BINARY = 12
BINARY_CONTRACT = SchemaContract(version=1, width=2 * N_BINARY, columns=[
    ContractColumn(f"flag_{i}", "categorical", 2 * i, 2 * i + 2, levels=(0, 1), integer=True)
    for i in range(N_BINARY)
], dropped=[], rows={}, issues=[])


def row(age: float, sex: str, bmi: float, grade: str) -> np.ndarray:
    """One encoded row of CONTRACT."""
    x = np.zeros(CONTRACT.width, dtype=np.float32)
    x[0], x[3] = age, bmi
    x[1 + "FM".index(sex)] = 1
    x[4 + "abc".index(grade)] = 1
    return x


def toy_rows(n_rows: int, seed: int) -> np.ndarray:
    """Encoded rows of CONTRACT from one fixed distribution, with some structure (bmi follows age)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_rows, CONTRACT.width), dtype=np.float32)
    x[:, 0] = np.clip(rng.normal(0.5, 0.2, n_rows), 0, 1)
    x[:, 3] = np.clip(0.6 * x[:, 0] + rng.normal(0.2, 0.1, n_rows), 0, 1)
    is_f = rng.random(n_rows) < 0.6
    x[is_f, 1], x[~is_f, 2] = 1, 1
    x[np.arange(n_rows), 4 + rng.choice(3, n_rows, p=[0.5, 0.3, 0.2])] = 1
    return x


def binary_rows(n_rows: int, seed: int) -> np.ndarray:
    """Encoded rows of BINARY_CONTRACT: skewed bits, so many genuine duplicates."""
    rng = np.random.default_rng(seed)
    bits = rng.random((n_rows, N_BINARY)) < 0.3
    x = np.zeros((n_rows, BINARY_CONTRACT.width), dtype=np.float32)
    for i in range(N_BINARY):
        x[np.arange(n_rows), 2 * i + bits[:, i]] = 1
    return x


@pytest.fixture(scope="module")
def train() -> np.ndarray:
    return toy_rows(N_TRAIN, seed=1)


@pytest.fixture(scope="module")
def holdout() -> np.ndarray:
    return toy_rows(N_HOLDOUT, seed=2)


@pytest.fixture(scope="module")
def fresh() -> np.ndarray:
    """A model that learned the distribution and nothing else: new rows from the same distribution."""
    return toy_rows(N_HOLDOUT, seed=3)


# --------------------------------------------------------------------------------------------
# Gower distance to the closest record
# --------------------------------------------------------------------------------------------

def test_identical_row_has_distance_zero():
    x = row(0.3, "F", 0.7, "b")[None, :]
    assert closest_record_distances(x, x, CONTRACT) == pytest.approx([0.0], abs=ATOL)


def test_distance_by_hand():
    query = row(0.2, "F", 0.5, "b")[None, :]
    reference = row(0.6, "M", 0.5, "b")[None, :]
    expected = (0.4 + 1 + 0 + 0) / N_COLUMNS
    assert closest_record_distances(query, reference, CONTRACT) == pytest.approx([expected], abs=ATOL)


def test_every_categorical_mismatch_weighs_one_column():
    query = row(0.5, "F", 0.5, "a")[None, :]
    reference = row(0.5, "M", 0.5, "c")[None, :]
    assert closest_record_distances(query, reference, CONTRACT) == pytest.approx([2 / N_COLUMNS], abs=ATOL)


def test_returns_the_closest_reference():
    query = row(0.5, "F", 0.5, "a")[None, :]
    reference = np.stack([row(0.9, "M", 0.1, "c"), row(0.55, "F", 0.5, "a"), row(0.5, "F", 0.5, "b")])
    assert closest_record_distances(query, reference, CONTRACT) == pytest.approx([0.05 / N_COLUMNS], abs=ATOL)


def test_one_distance_per_query_in_query_order():
    queries = np.stack([row(0.5, "F", 0.5, "a"), row(0.5, "M", 0.5, "a"), row(0.1, "F", 0.5, "a")])
    reference = row(0.5, "F", 0.5, "a")[None, :]
    distances = closest_record_distances(queries, reference, CONTRACT)
    assert distances.shape == (3,)
    assert distances == pytest.approx([0.0, 1 / N_COLUMNS, 0.4 / N_COLUMNS], abs=ATOL)


def test_unseen_level_without_other_slot_differs_from_every_level():
    query = row(0.5, "F", 0.5, "a")
    query[4:7] = 0                                   # holdout level never seen in train
    reference = np.stack([row(0.5, "F", 0.5, level) for level in "abc"])
    assert closest_record_distances(query[None, :], reference, CONTRACT) == pytest.approx([1 / N_COLUMNS], abs=ATOL)


def test_numeric_outside_train_range_is_not_clipped():
    query = row(1.5, "F", 0.5, "a")[None, :]         # holdout value above the train maximum
    reference = row(1.0, "F", 0.5, "a")[None, :]
    assert closest_record_distances(query, reference, CONTRACT) == pytest.approx([0.5 / N_COLUMNS], abs=ATOL)


def test_distances_of_in_range_rows_are_between_zero_and_one(train, holdout):
    distances = closest_record_distances(holdout, train, CONTRACT)
    assert np.all(distances >= 0) and np.all(distances <= 1)


@pytest.mark.parametrize("batch_size", [1, 7, 100_000])
def test_batch_size_does_not_change_the_result(train, holdout, batch_size):
    expected = closest_record_distances(holdout[:50], train, CONTRACT)
    got = closest_record_distances(holdout[:50], train, CONTRACT, batch_size=batch_size)
    np.testing.assert_allclose(got, expected, atol=ATOL)


def test_reference_split_in_blocks_keeps_the_overall_closest(train, holdout, monkeypatch):
    expected = closest_record_distances(holdout[:50], train, CONTRACT)
    monkeypatch.setattr(privacy, "DCR_REFERENCE_BLOCK", 7)
    np.testing.assert_allclose(closest_record_distances(holdout[:50], train, CONTRACT), expected, atol=ATOL)


def test_a_tie_between_train_and_holdout_counts_half():
    x = row(0.5, "F", 0.5, "a")[None, :]
    report = privacy_check(x, x, x, CONTRACT)
    assert report.train_closer_share == pytest.approx(0.5)


def test_matches_a_brute_force_gower(train, holdout):
    queries, reference = holdout[:40], train[:300]
    numeric = np.abs(queries[:, None, [0, 3]] - reference[None, :, [0, 3]]).sum(axis=2)
    categorical = sum(1 - queries[:, g] @ reference[:, g].T for g in (slice(1, 3), slice(4, 7)))
    expected = ((numeric + categorical) / N_COLUMNS).min(axis=1)
    np.testing.assert_allclose(closest_record_distances(queries, reference, CONTRACT), expected, atol=ATOL)


@pytest.mark.parametrize("queries, reference", [
    (np.zeros((0, 7), dtype=np.float32), np.zeros((3, 7), dtype=np.float32)),
    (np.zeros((3, 7), dtype=np.float32), np.zeros((0, 7), dtype=np.float32)),
    (np.zeros((3, 6), dtype=np.float32), np.zeros((3, 6), dtype=np.float32)),
    (np.zeros((3, 7), dtype=np.float32), np.zeros((3, 8), dtype=np.float32)),
])
def test_invalid_inputs_are_rejected(queries, reference):
    with pytest.raises(ValueError):
        closest_record_distances(queries, reference, CONTRACT)


# --------------------------------------------------------------------------------------------
# Privacy gate
# --------------------------------------------------------------------------------------------

def failure_codes(report: PrivacyReport) -> set[str]:
    return {issue.code for issue in report.failures}


def test_a_model_that_learned_only_the_distribution_passes(train, holdout, fresh):
    report = privacy_check(fresh, train, holdout, CONTRACT)
    assert report.passed, report.failures
    assert report.failures == []


def test_copying_the_train_rows_fails(train, holdout):
    report = privacy_check(train[:N_HOLDOUT], train, holdout, CONTRACT)
    assert not report.passed
    assert "copied_rows" in failure_codes(report)
    assert report.synthetic_exact_share == pytest.approx(1.0)


def test_copying_one_row_in_ten_fails(train, holdout, fresh):
    n_copied = int(COPIED_SHARE * N_HOLDOUT)
    synthetic = np.concatenate([fresh[n_copied:], train[:n_copied]])
    report = privacy_check(synthetic, train, holdout, CONTRACT)
    assert not report.passed
    assert "copied_rows" in failure_codes(report)
    assert report.synthetic_exact_share == pytest.approx(COPIED_SHARE, abs=0.01)


def test_memorised_rows_with_tiny_noise_fail_as_too_close(train, holdout):
    rng = np.random.default_rng(4)
    synthetic = train[:N_HOLDOUT].copy()
    synthetic[:, [0, 3]] += rng.uniform(-JITTER, JITTER, (N_HOLDOUT, 2)).astype(np.float32)
    report = privacy_check(synthetic, train, holdout, CONTRACT)
    assert not report.passed
    assert failure_codes(report) == {"too_close"}       # no exact copy: only the DCR sees it
    assert report.synthetic_exact_share == 0


def test_failures_explain_themselves_in_plain_language(train, holdout):
    report = privacy_check(train[:N_HOLDOUT], train, holdout, CONTRACT)
    for issue in report.failures:
        assert issue.message and "Traceback" not in issue.message


@pytest.mark.parametrize("kind", ["copied_rows", "too_close"])
def test_failures_name_the_two_causes_and_the_train_rows(train, holdout, kind):
    """The user has no training levers and training is deterministic: the message names what can change."""
    synthetic = train[:N_HOLDOUT].copy()
    if kind == "too_close":
        rng = np.random.default_rng(4)
        synthetic[:, [0, 3]] += rng.uniform(-JITTER, JITTER, (N_HOLDOUT, 2)).astype(np.float32)
    report = privacy_check(synthetic, train, holdout, CONTRACT)
    (issue,) = [i for i in report.failures if i.code == kind]
    assert f"{N_TRAIN:,} rows" in issue.message                  # small dataset: the count lets the user judge
    assert "identify" in issue.message                           # identifying columns
    assert "fewer steps" not in issue.message and "training again" not in issue.message.lower()


def test_genuine_duplicates_in_the_real_data_do_not_fail_by_themselves():
    train, holdout, fresh = binary_rows(N_TRAIN, 1), binary_rows(N_HOLDOUT, 2), binary_rows(N_HOLDOUT, 3)
    report = privacy_check(fresh, train, holdout, BINARY_CONTRACT)
    assert report.holdout_exact_share > 0.5              # different people coincide in the real data
    assert report.passed, report.failures


def test_report_numbers_match_the_distances(train, holdout, fresh):
    report = privacy_check(fresh, train, holdout, CONTRACT)
    synthetic_dcr = closest_record_distances(fresh, train, CONTRACT)
    holdout_dcr = closest_record_distances(holdout, train, CONTRACT)
    assert report.n_compared == N_HOLDOUT
    assert report.synthetic_percentile == pytest.approx(np.percentile(synthetic_dcr, DCR_PERCENTILE), abs=ATOL)
    assert report.holdout_percentile == pytest.approx(np.percentile(holdout_dcr, DCR_PERCENTILE), abs=ATOL)
    assert report.synthetic_exact_share == pytest.approx(np.mean(synthetic_dcr <= EXACT_DISTANCE))
    assert report.holdout_exact_share == pytest.approx(np.mean(holdout_dcr <= EXACT_DISTANCE))


def test_exact_share_limit_by_hand(train, holdout):
    synthetic = toy_rows(500, seed=6)
    duplicated_holdout = np.concatenate([holdout[:800], train[:200]])     # 20% of the holdout has a twin
    report = privacy_check(synthetic, train, duplicated_holdout, CONTRACT)
    p = report.holdout_exact_share
    assert p == pytest.approx(0.2)
    noise = NOISE_Z * np.sqrt(p * (1 - p) * (1 / 500 + 1 / N_HOLDOUT))
    assert report.exact_share_limit == pytest.approx(p + EXACT_SHARE_TOLERANCE + noise)


def test_without_duplicates_in_the_holdout_the_limit_is_the_tolerance(train, holdout, fresh):
    report = privacy_check(fresh, train, holdout, CONTRACT)
    assert report.holdout_exact_share == 0
    assert report.exact_share_limit == pytest.approx(EXACT_SHARE_TOLERANCE)


def test_train_closer_share_tells_copies_from_new_rows(train, holdout, fresh):
    expected = N_TRAIN / (N_TRAIN + N_HOLDOUT)
    new_rows = privacy_check(fresh, train, holdout, CONTRACT)
    copies = privacy_check(train[:N_HOLDOUT], train, holdout, CONTRACT)
    assert new_rows.expected_train_closer_share == pytest.approx(expected)
    assert new_rows.train_closer_share == pytest.approx(expected, abs=0.05)
    assert copies.train_closer_share > 0.99


def test_thresholds_can_be_tightened_or_loosened(train, holdout):
    rng = np.random.default_rng(4)
    synthetic = train[:N_HOLDOUT].copy()
    synthetic[:, [0, 3]] += rng.uniform(-JITTER, JITTER, (N_HOLDOUT, 2)).astype(np.float32)
    assert privacy_check(synthetic, train, holdout, CONTRACT, min_ratio=0.0).passed
    copies = privacy_check(train[:N_HOLDOUT], train, holdout, CONTRACT, min_ratio=0.0, exact_tolerance=1.0)
    assert copies.passed


def test_more_synthetic_rows_than_max_rows_are_subsampled(train, holdout):
    synthetic = toy_rows(3 * N_HOLDOUT, seed=5)
    report = privacy_check(synthetic, train, holdout, CONTRACT, max_rows=N_HOLDOUT)
    assert report.n_compared == N_HOLDOUT
    assert report.passed, report.failures


def test_same_seed_same_report(train, holdout):
    synthetic = toy_rows(3 * N_HOLDOUT, seed=5)
    first = privacy_check(synthetic, train, holdout, CONTRACT, max_rows=N_HOLDOUT, seed=7)
    second = privacy_check(synthetic, train, holdout, CONTRACT, max_rows=N_HOLDOUT, seed=7)
    assert first == second


def test_default_max_rows_covers_the_largest_holdout():
    assert DCR_MAX_ROWS >= MAX_HOLDOUT_ROWS


def test_default_thresholds_are_sane():
    assert 0 < MIN_DCR_RATIO <= 1
    assert 0 <= EXACT_SHARE_TOLERANCE < COPIED_SHARE
    assert 0 < DCR_PERCENTILE < 50
    assert NOISE_Z > 0


@pytest.mark.parametrize("which", ["synthetic", "train", "holdout"])
def test_empty_inputs_are_rejected(train, holdout, fresh, which):
    inputs = {"synthetic": fresh, "train": train, "holdout": holdout}
    inputs[which] = inputs[which][:0]
    with pytest.raises(ValueError):
        privacy_check(inputs["synthetic"], inputs["train"], inputs["holdout"], CONTRACT)


# --------------------------------------------------------------------------------------------
# Warning: the near-copy check cannot work when the reference percentile is 0
# --------------------------------------------------------------------------------------------
# With many genuine twins (CDC: 13.9% of new people identical to a train row) the holdout percentile is 0,
# so too_close (synthetic percentile < min_ratio x 0) can never fire. Measured in privacy_too_close: no
# distance-based variant tells a near-copy from a new person there. The check stays generic: it does not
# look at the data, only at whether too_close can work, and then warns (not blocking).

def warning_codes(report: PrivacyReport) -> set[str]:
    return {issue.code for issue in report.warnings}


def test_dense_data_warn_that_near_copies_are_not_checked():
    train, holdout, fresh = binary_rows(N_TRAIN, 1), binary_rows(N_HOLDOUT, 2), binary_rows(N_HOLDOUT, 3)
    report = privacy_check(fresh, train, holdout, BINARY_CONTRACT)
    assert report.holdout_percentile == 0
    assert warning_codes(report) == {"near_copies_unchecked"}
    assert report.passed                                 # a warning, not a failure
    (warning,) = report.warnings
    assert "near-copies" in warning.message and "Exact copies" in warning.message


def test_no_warning_when_the_reference_percentile_is_positive(train, holdout, fresh):
    report = privacy_check(fresh, train, holdout, CONTRACT)
    assert report.holdout_percentile > 0
    assert report.warnings == []


def test_the_warning_comes_with_failures_too():
    train, holdout = binary_rows(N_TRAIN, 1), binary_rows(N_HOLDOUT, 2)
    report = privacy_check(train[:N_HOLDOUT], train, holdout, BINARY_CONTRACT)
    assert failure_codes(report) == {"copied_rows"}
    assert warning_codes(report) == {"near_copies_unchecked"}
