"""Specification tests for the decoding step.

The decoder turns vectors (the VAE output) back into a table, reading only the schema contract.
Categorical groups carry probabilities (the VAE applies the softmax): one level is sampled per group,
with a seed. Public API under test:
    decode(vectors, contract, *, seed=RANDOM_STATE) -> pd.DataFrame
        columns in contract order, original types (int, float, datetime, text)
        numeric / date: inverse MinMax, clipped to the train range, rounded when whole numbers / days
        categorical: scores cleaned (negatives -> 0, renormalised, all-zero -> uniform), then sampled;
                     "Other" -> "Other" (text) or a merged value in train proportions (numeric codes)
    Wrong vector width -> DataEngineError("wrong_width").
    decode_checked(vectors, contract, *, seed=RANDOM_STATE) -> (pd.DataFrame, np.ndarray)
        the same table, plus a boolean per row: True when the row still pairs categorical values never
        seen together in train (contract.seen_pairs) after the repair
    Coherence repair: all groups are sampled first; in every row with an unseen pair, the conflicting
    column whose sampled level has the lowest probability (among those with a compatible level) is
    resampled among the levels compatible with the rest of the row, for a few passes. Coherent rows and
    pairs of columns without an entry in seen_pairs are left alone.
"""

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest

from sds.data_engine import (
    DataEngineError,
    SchemaContract,
    UserChoices,
    apply_choices,
    build_contract,
    decode,
    fit_encoder,
    inspect_csv,
    prepare,
)
from sds.data_engine.contract import OTHER_LABEL, ContractColumn, SeenPairs
from sds.data_engine.decoding import decode_checked

EXACT_INTEGER_RANGE = 2 ** 24      # float32 keeps whole numbers exact below this range
FLOAT_TOLERANCE = 1e-6             # relative to the column range
N_SAMPLES = 4000                   # rows used to check sampled proportions
PROPORTION_TOLERANCE = 0.03


# --------------------------------------------------------------------------------------------
# A hand-made contract: the decoder reads only the contract, so it can be tested on its own.
# --------------------------------------------------------------------------------------------

def handmade_contract() -> SchemaContract:
    columns = [
        ContractColumn("x", "numeric", 0, 1, min=10.0, max=20.0, integer=False),
        ContractColumn("n", "numeric", 1, 2, min=0.0, max=100.0, integer=True),
        ContractColumn("visit", "date", 2, 3, min=0.0, max=365.0, origin="2020-01-01"),
        ContractColumn("c", "categorical", 3, 6, levels=("a", "b"), has_other=True,
                       merged=(("r1", 9), ("r2", 1))),
        ContractColumn("k", "categorical", 6, 9, integer=True, levels=(0, 16), has_other=True,
                       merged=((17, 30), (18, 10))),
    ]
    return SchemaContract(version=1, width=9, columns=columns, dropped=[], rows={}, issues=[])


CONTRACT = handmade_contract()


def vectors(n_rows: int = 1, **slots) -> np.ndarray:
    """Rows of a valid vector (first level of every group) with some positions overridden.

    ``slots`` maps a column name to the values of its positions, e.g. ``c=[0, 0, 1]``.
    """
    base = np.zeros((n_rows, CONTRACT.width), dtype=np.float32)
    base[:, [3, 6]] = 1                                  # 'a' and 0
    for column in CONTRACT.columns:
        if column.name in slots:
            base[:, column.start:column.stop] = slots[column.name]
    return base


def test_numbers_are_rounded_to_the_decimals_of_the_train():
    rounded = SchemaContract(version=CONTRACT.version, width=1, dropped=[], rows={}, issues=[], columns=[
        ContractColumn("bmi", "numeric", 0, 1, min=18.2, max=67.1, integer=False, decimals=1)])
    scores = np.array([[0.0], [0.123456], [0.5], [1.0]], dtype=np.float32)
    values = decode(scores, rounded)["bmi"].to_numpy()
    assert np.array_equal(values, np.round(values, 1))
    assert values[0] == 18.2 and values[-1] == 67.1                  # still within the train range
    assert values[2] == round(18.2 + 0.5 * (67.1 - 18.2), 1)


def test_numbers_without_decimals_are_not_rounded():
    scores = np.array([[0.123456]], dtype=np.float32)
    x = decode(vectors(1, x=scores[0]), CONTRACT)["x"].iloc[0]       # CONTRACT: decimals=None
    assert x != round(x, 6)


def test_output_columns_and_types():
    table = decode(vectors(3), CONTRACT)
    assert list(table.columns) == ["x", "n", "visit", "c", "k"]
    assert pd.api.types.is_float_dtype(table["x"])
    assert pd.api.types.is_integer_dtype(table["n"])
    assert pd.api.types.is_datetime64_any_dtype(table["visit"])
    assert pd.api.types.is_string_dtype(table["c"])
    assert pd.api.types.is_integer_dtype(table["k"])
    assert len(table) == 3


def test_wrong_width_is_an_error():
    with pytest.raises(DataEngineError) as error:
        decode(np.zeros((2, CONTRACT.width + 1), dtype=np.float32), CONTRACT)
    assert error.value.code == "wrong_width"


# --------------------------------------------------------------------------------------------
# Numbers and dates
# --------------------------------------------------------------------------------------------

def test_inverse_scaling():
    table = decode(vectors(x=[0.25], n=[0.5]), CONTRACT)
    assert table["x"].iloc[0] == pytest.approx(12.5)
    assert table["n"].iloc[0] == 50


def test_numbers_are_clipped_to_the_train_range():
    table = decode(np.vstack([vectors(x=[-0.5], n=[-0.2]), vectors(x=[1.5], n=[1.3])]), CONTRACT)
    assert table["x"].tolist() == [10.0, 20.0]
    assert table["n"].tolist() == [0, 100]


def test_whole_numbers_are_rounded():
    assert decode(vectors(n=[0.124]), CONTRACT)["n"].iloc[0] == 12
    assert decode(vectors(n=[0.126]), CONTRACT)["n"].iloc[0] == 13


def test_dates_are_rounded_to_whole_days_and_clipped():
    rows = np.vstack([vectors(visit=[10.4 / 365]), vectors(visit=[2.0])])
    assert decode(rows, CONTRACT)["visit"].tolist() == [pd.Timestamp("2020-01-11"), pd.Timestamp("2020-12-31")]


# --------------------------------------------------------------------------------------------
# Categories: sampled from the probabilities
# --------------------------------------------------------------------------------------------

def test_a_one_hot_group_gives_its_level():
    assert decode(vectors(c=[0, 1, 0], k=[0, 1, 0]), CONTRACT)[["c", "k"]].iloc[0].tolist() == ["b", 16]


def test_levels_follow_the_probabilities():
    table = decode(vectors(N_SAMPLES, c=[0.7, 0.2, 0.1]), CONTRACT)
    shares = table["c"].value_counts(normalize=True)
    assert shares["a"] == pytest.approx(0.7, abs=PROPORTION_TOLERANCE)
    assert shares["b"] == pytest.approx(0.2, abs=PROPORTION_TOLERANCE)
    assert shares[OTHER_LABEL] == pytest.approx(0.1, abs=PROPORTION_TOLERANCE)


def test_sampling_is_reproducible():
    rows = vectors(200, c=[0.4, 0.3, 0.3], k=[0.3, 0.3, 0.4])
    assert decode(rows, CONTRACT, seed=1).equals(decode(rows, CONTRACT, seed=1))
    assert not decode(rows, CONTRACT, seed=1).equals(decode(rows, CONTRACT, seed=2))


def test_negative_scores_are_ignored():
    # The baseline loss (MSE) does not produce probabilities: negatives count as zero.
    table = decode(vectors(200, c=[-1.0, 2.0, -0.5]), CONTRACT)
    assert set(table["c"]) == {"b"}


def test_an_all_zero_group_is_sampled_uniformly():
    table = decode(vectors(N_SAMPLES, c=[0, 0, 0]), CONTRACT)
    shares = table["c"].value_counts(normalize=True)
    assert set(shares.index) == {"a", "b", OTHER_LABEL}
    assert shares.min() == pytest.approx(1 / 3, abs=PROPORTION_TOLERANCE)


def test_other_of_a_text_column_is_written_as_other():
    assert decode(vectors(c=[0, 0, 1]), CONTRACT)["c"].iloc[0] == OTHER_LABEL


def test_other_of_an_integer_column_picks_merged_values_in_train_proportions():
    table = decode(vectors(N_SAMPLES, k=[0, 0, 1]), CONTRACT)
    shares = table["k"].value_counts(normalize=True)
    assert set(shares.index) == {17, 18}
    assert shares[17] == pytest.approx(30 / 40, abs=PROPORTION_TOLERANCE)


# --------------------------------------------------------------------------------------------
# Coherence repair: no pair of categorical values that no train row has
# --------------------------------------------------------------------------------------------
# In CONTRACT "c" has slots (a, b, Other) and "k" slots (0, 16, Other). PAIRED allows only a<->0, b<->16,
# Other<->Other, as if one column were a recoding of the other (Adult: education <-> education-num).

ALLOWED = {("a", 0), ("b", 16)}
PAIRED = replace(CONTRACT, seen_pairs=[SeenPairs("c", "k", ((0, 0), (1, 1), (2, 2)))])
EVERY_PAIR = replace(CONTRACT, seen_pairs=[SeenPairs("c", "k", tuple(product(range(3), range(3))))])


def value_pairs(table: pd.DataFrame) -> set:
    return set(zip(table["c"], table["k"], strict=True))


def test_coherent_rows_are_kept():
    table, incoherent = decode_checked(vectors(c=[0, 1, 0], k=[0, 1, 0]), PAIRED)
    assert table[["c", "k"]].iloc[0].tolist() == ["b", 16]
    assert not incoherent.any()


def test_incoherent_rows_are_repaired():
    rows = vectors(N_SAMPLES, c=[0.5, 0.5, 0], k=[0.5, 0.5, 0])
    assert value_pairs(decode(rows, CONTRACT)) > ALLOWED          # independent sampling: ~half incoherent
    table, incoherent = decode_checked(rows, PAIRED)
    assert value_pairs(table) == ALLOWED
    assert not incoherent.any()


def test_the_less_confident_column_is_the_one_resampled():
    # "c" is almost surely "a"; "k" leans to 16, which "a" never meets: "k" is changed, "c" kept.
    table = decode(vectors(N_SAMPLES, c=[0.98, 0.02, 0], k=[0.25, 0.75, 0]), PAIRED)
    assert (table["c"] == "a").mean() == pytest.approx(0.98, abs=PROPORTION_TOLERANCE)
    assert value_pairs(table) <= ALLOWED


def test_the_less_confident_column_is_chosen_whatever_its_position():
    # Mirror of the test above: now the first column ("c") is the unsure one, "k" is kept.
    table = decode(vectors(N_SAMPLES, c=[0.25, 0.75, 0], k=[0.98, 0.02, 0]), PAIRED)
    assert (table["k"] == 0).mean() == pytest.approx(0.98, abs=PROPORTION_TOLERANCE)
    assert value_pairs(table) <= ALLOWED


def test_a_column_with_no_compatible_level_is_not_the_one_resampled():
    # "c" is never "b", so with k=16 it cannot be fixed even when it is the less likely: "k" is changed.
    rows = vectors(N_SAMPLES, c=[0.3, 0, 0.7], k=[0.05, 0.9, 0.05])
    table, incoherent = decode_checked(rows, PAIRED)
    assert not incoherent.any()
    assert value_pairs(table) <= ALLOWED | {(OTHER_LABEL, 17), (OTHER_LABEL, 18)}


def test_a_row_with_no_compatible_level_is_left_and_flagged():
    # One-hot rows (a real holdout row, say) give no other level to choose: decoded as they are.
    rows = np.vstack([vectors(c=[1, 0, 0], k=[0, 1, 0]), vectors(c=[1, 0, 0], k=[1, 0, 0])])
    table, incoherent = decode_checked(rows, PAIRED)
    assert table[["c", "k"]].values.tolist() == [["a", 16], ["a", 0]]
    assert incoherent.tolist() == [True, False]


def test_columns_without_seen_pairs_are_not_checked():
    rows = vectors(500, c=[0.5, 0.5, 0], k=[0.5, 0.5, 0])
    table, incoherent = decode_checked(rows, CONTRACT)
    assert value_pairs(table) > ALLOWED
    assert not incoherent.any()


def test_repair_changes_nothing_when_every_pair_was_seen():
    # Same draws as without seen pairs: the repair consumes no randomness on coherent rows.
    rows = vectors(500, c=[0.4, 0.3, 0.3], k=[0.3, 0.3, 0.4])
    pd.testing.assert_frame_equal(decode(rows, EVERY_PAIR, seed=3), decode(rows, CONTRACT, seed=3))


def test_repair_is_reproducible():
    rows = vectors(500, c=[0.4, 0.3, 0.3], k=[0.3, 0.3, 0.4])
    pd.testing.assert_frame_equal(decode(rows, PAIRED, seed=3), decode(rows, PAIRED, seed=3))


def test_repair_end_to_end_with_integer_codes_and_missing_values():
    # Regression: an integer category with missing values is float in the train table (9.0) and int in the
    # synthetic one (9); compared as text the pairs never matched. The repair works on slots, not on values.
    n_rows = 120
    labels = ["e1", "e2", "e3", "e4"]
    edu = [labels[i % 4] for i in range(n_rows)]
    code = [str(i % 4 + 1) if i % 20 else "?" for i in range(n_rows)]
    lines = ["edu,code"] + [f"{e},{c}" for e, c in zip(edu, code, strict=True)]
    inspection = inspect_csv(("\n".join(lines) + "\n").encode())
    schema = apply_choices(inspection, UserChoices())
    prepared = prepare(inspection, schema)
    encoder = fit_encoder(prepared.train, schema)
    contract = build_contract(inspection, schema, prepared, encoder)
    assert [c.role for c in contract.columns] == ["categorical", "categorical"]
    uniform = np.full((N_SAMPLES, contract.width), 0.25, dtype=np.float32)
    table, incoherent = decode_checked(uniform, contract)
    train_pairs = set(zip(prepared.train["edu"], prepared.train["code"].astype("int64"), strict=True))
    assert set(zip(table["edu"], table["code"], strict=True)) == train_pairs
    assert not incoherent.any()


# --------------------------------------------------------------------------------------------
# Round trip through the whole Data Engine: decode(encode(train)) == train
# --------------------------------------------------------------------------------------------

def assert_round_trip(train: pd.DataFrame, decoded: pd.DataFrame, contract: SchemaContract, label: str = "") -> None:
    """Decoded train rows equal the originals, up to float32 precision and the rare levels."""
    original = train.reset_index(drop=True)
    assert list(decoded.columns) == [c.name for c in contract.columns], label
    assert len(decoded) == len(original), label
    for column in contract.columns:
        before, after = original[column.name], decoded[column.name]
        where = f"{label} {column.name}"
        if column.role == "date":
            assert (after == before).all(), where
        elif column.role == "numeric":
            span = column.max - column.min
            if column.integer and span < EXACT_INTEGER_RANGE:
                assert (after.to_numpy() == before.to_numpy()).all(), where
            else:
                assert np.allclose(after, before, rtol=0, atol=span * FLOAT_TOLERANCE), where
        else:
            merged = {level for level, _ in column.merged}
            values = before.astype("int64") if column.integer else before
            rare = values.isin(merged)
            assert (after[~rare].to_numpy() == values[~rare].to_numpy()).all(), where
            if column.numeric_codes:
                assert after[rare].isin(merged).all(), where
            else:
                assert (after[rare] == OTHER_LABEL).all(), where


def base_columns() -> dict[str, list]:
    days = pd.date_range("2020-01-01", periods=60, freq="D")
    n_rows = 120
    return {
        "x": [f"{i}.5" for i in range(n_rows)],
        "n": [i % 40 for i in range(n_rows)],
        "visit": [days[i % 60].strftime("%Y-%m-%d") for i in range(n_rows)],
        "c": ["a"] * 60 + ["b"] * 50 + ["r1"] * 9 + ["r2"],
        "k": [0] * 100 + [16] * 11 + [17] * 9,
    }


def test_round_trip_on_a_small_file():
    names = list(base_columns())
    lines = [",".join(names)] + [",".join(map(str, row)) for row in zip(*base_columns().values(), strict=True)]
    inspection = inspect_csv(("\n".join(lines) + "\n").encode())
    schema = apply_choices(inspection, UserChoices())
    prepared = prepare(inspection, schema)
    encoder = fit_encoder(prepared.train, schema)
    contract = build_contract(inspection, schema, prepared, encoder)
    assert_round_trip(prepared.train, decode(encoder.transform(prepared.train), contract), contract)


# --------------------------------------------------------------------------------------------
# Memory: no float64 copy of the whole vector, no copy of the decoded columns
# --------------------------------------------------------------------------------------------
# On 3.8 M rows decode added 3.3 GB: a float64 copy of the whole input (2 x the float32 vector) and
# pd.DataFrame copying the decoded columns. Columns are cast one at a time and the frame wraps them.

MEMORY_ROWS = 100_000
MEMORY_COLUMNS = 30
MAX_DECODE_PEAK = 1.5           # peak of decode / bytes of the table it returns (either copy back: ~2)


def wide_contract() -> SchemaContract:
    """Many one-slot columns, as in real data: a copy of the whole vector would dominate the memory."""
    columns = [ContractColumn(f"x{i}", "numeric", i, i + 1, min=0.0, max=1.0, integer=False)
               for i in range(MEMORY_COLUMNS)]
    return SchemaContract(version=1, width=MEMORY_COLUMNS, columns=columns, dropped=[], rows={}, issues=[])


def test_decode_peak_stays_close_to_its_output():
    import tracemalloc

    contract = wide_contract()
    scores = np.random.default_rng(0).random((MEMORY_ROWS, contract.width), dtype=np.float32)
    tracemalloc.start()
    try:
        table = decode(scores, contract)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < MAX_DECODE_PEAK * table.memory_usage(deep=True).sum()


def test_float32_and_float64_vectors_decode_alike():
    scores = np.random.default_rng(1).normal(0.4, 0.5, (500, CONTRACT.width))
    pd.testing.assert_frame_equal(decode(scores.astype(np.float32), CONTRACT),
                                  decode(scores.astype(np.float32).astype(np.float64), CONTRACT))
