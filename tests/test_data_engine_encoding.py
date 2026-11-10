"""Specification tests for the encoding step.

The encoder is fit on the train rows only (no leakage) and turns a table into the vector the VAE
learns from: MinMax for numeric and date columns (dates as days), One-Hot for categorical ones.
Levels with fewer than MIN_LEVEL_ROWS train rows are merged into one internal "Other" slot, and a
categorical column has at most MAX_CATEGORIES slots. Built on scikit-learn (ColumnTransformer,
MinMaxScaler, OneHotEncoder). Public API under test:
    fit_encoder(train, schema) -> TabularEncoder
        .transform(frame) -> np.ndarray (float32), used for train and holdout rows
        .width            length of the vector
        .groups           list[FeatureGroup], the index map (schema order)
        .issues           list[Issue]: levels_merged, constant_in_train, all_levels_rare
    No column can be learned from the train rows -> DataEngineError("nothing_to_learn").
    FeatureGroup(name, role, start, stop, levels, has_other, merged)
        levels     kept levels, original type (int stays int), the "Other" slot excluded
        has_other  True when the group ends with the "Other" slot
        merged     {level: train rows} of the levels merged into "Other"
"""

import tracemalloc
import warnings
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sds.data_engine import DataEngineError, UserChoices, apply_choices, fit_encoder, inspect_csv
from sds.data_engine.thresholds_config import MAX_CATEGORIES, MIN_LEVEL_ROWS

N_ROWS = 120


def make_csv(columns: dict[str, list]) -> bytes:
    """CSV bytes from equal-length columns."""
    names = list(columns)
    rows = zip(*(columns[name] for name in names), strict=True)
    return ("\n".join([",".join(names)] + [",".join(map(str, row)) for row in rows]) + "\n").encode()


def base_columns() -> dict[str, list]:
    """'x' numeric, 'c' text with two rare levels, 'k' integer codes with one rare level."""
    return {
        "x": [f"{i}.5" for i in range(N_ROWS)],
        "c": ["a"] * 60 + ["b"] * 50 + ["r1"] * 9 + ["r2"],
        "k": [0] * 100 + [16] * 11 + [17] * 9,
    }


def fitted(columns: dict[str, list], choices: UserChoices | None = None, rows: slice = slice(None)):
    """Encoder fit on the complete rows of a small CSV (optionally only ``rows`` of them)."""
    inspection = inspect_csv(make_csv(columns))
    schema = apply_choices(inspection, choices or UserChoices())
    train = inspection.frame[list(schema.columns)].dropna().iloc[rows]
    return fit_encoder(train, schema), train


@pytest.fixture(scope="module")
def base():
    return fitted(base_columns())


def group(encoder, name):
    (found,) = [g for g in encoder.groups if g.name == name]
    return found


def codes(encoder, column=None) -> set[str]:
    return {i.code for i in encoder.issues if column is None or i.column == column}


# --------------------------------------------------------------------------------------------
# Shape and ranges
# --------------------------------------------------------------------------------------------

def test_vector_shape_and_type(base):
    encoder, train = base
    vectors = encoder.transform(train)
    assert vectors.shape == (N_ROWS, encoder.width)
    assert encoder.width == 1 + 3 + 3          # x | a, b, Other | 0, 16, Other
    assert vectors.dtype == np.float32


def test_numeric_columns_span_the_unit_interval_on_train(base):
    encoder, train = base
    x = group(encoder, "x")
    values = encoder.transform(train)[:, x.start:x.stop]
    assert (x.role, x.stop - x.start) == ("numeric", 1)
    assert values.min() == 0 and values.max() == 1


def test_each_one_hot_group_has_a_single_one_per_row(base):
    encoder, train = base
    vectors = encoder.transform(train)
    for g in encoder.groups:
        if g.role == "categorical":
            block = vectors[:, g.start:g.stop]
            assert set(np.unique(block)) <= {0.0, 1.0}, g.name
            assert (block.sum(axis=1) == 1).all(), g.name


def test_groups_cover_the_vector_in_schema_order(base):
    encoder, _ = base
    assert [g.name for g in encoder.groups] == ["x", "c", "k"]
    assert encoder.groups[0].start == 0
    for previous, current in pairwise(encoder.groups):
        assert current.start == previous.stop
    assert encoder.groups[-1].stop == encoder.width


# --------------------------------------------------------------------------------------------
# Rare levels and the One-Hot cap
# --------------------------------------------------------------------------------------------

def test_rare_text_levels_are_merged_into_other(base):
    encoder, _ = base
    c = group(encoder, "c")
    assert c.levels == ("a", "b")
    assert c.has_other and c.stop - c.start == 3
    assert c.merged == {"r1": 9, "r2": 1}
    assert "levels_merged" in codes(encoder, "c")


def test_rare_rows_land_in_the_other_slot(base):
    encoder, train = base
    c = group(encoder, "c")
    block = encoder.transform(train)[:, c.start:c.stop]
    assert block[:, -1].sum() == 10            # r1 (9) + r2 (1)


def test_integer_levels_keep_their_type(base):
    encoder, _ = base
    k = group(encoder, "k")
    assert k.levels == (0, 16) and all(type(level) is int for level in k.levels)
    assert k.merged == {17: 9} and all(type(level) is int for level in k.merged)


def test_a_level_with_exactly_the_minimum_rows_is_kept():
    columns = {"x": [f"{i}.5" for i in range(N_ROWS)],
               "c": ["a"] * 60 + ["b"] * (60 - MIN_LEVEL_ROWS) + ["r"] * MIN_LEVEL_ROWS}
    encoder, _ = fitted(columns)
    c = group(encoder, "c")
    assert c.levels == ("a", "b", "r") and not c.has_other and c.merged == {}
    assert "levels_merged" not in codes(encoder)


def test_a_real_other_level_stays_apart_from_the_rare_ones():
    # Adult and Diabetes 130 have a real "Other" race: rare levels go to the internal slot.
    columns = {"x": [f"{i}.5" for i in range(N_ROWS)],
               "race": ["White"] * 85 + ["Other"] * 30 + ["Rare"] * 5}
    encoder, _ = fitted(columns)
    race = group(encoder, "race")
    assert race.levels == ("Other", "White") and race.has_other
    assert race.merged == {"Rare": 5}


def test_one_hot_groups_are_capped():
    # 60 levels x 12 rows: none is rare, but only MAX_CATEGORIES slots are allowed.
    n_levels, per_level = 60, 12
    columns = {"x": [f"{i}.5" for i in range(n_levels * per_level)],
               "c": [f"v{i:02d}" for i in range(n_levels) for _ in range(per_level)]}
    # 60 text levels > MAX_CATEGORIES: the engine drops the column, the user brings it back.
    encoder, _ = fitted(columns, UserChoices(included={"c"}))
    c = group(encoder, "c")
    assert c.stop - c.start == MAX_CATEGORIES
    assert c.has_other and len(c.levels) == MAX_CATEGORIES - 1
    assert len(c.merged) == n_levels - (MAX_CATEGORIES - 1)


# --------------------------------------------------------------------------------------------
# Columns that cannot be learned from the train rows
# --------------------------------------------------------------------------------------------

def test_a_column_constant_in_train_gets_no_group():
    # Like metformin-rosiglitazone: the only different rows ended up in the holdout.
    encoder, train = fitted(base_columns(), rows=slice(0, 100))   # 'k' is 0 in the first 100 rows
    assert [g.name for g in encoder.groups] == ["x", "c"]
    assert "constant_in_train" in codes(encoder, "k")
    assert encoder.transform(train).shape == (100, encoder.width)


def test_a_column_of_only_rare_levels_gets_no_group():
    # A re-included text identifier: every level appears once, so all of them are rare.
    columns = {"x": [f"{i}.5" for i in range(N_ROWS)], "code": [f"P{i:04d}" for i in range(N_ROWS)]}
    encoder, _ = fitted(columns, UserChoices(included={"code"}))
    assert [g.name for g in encoder.groups] == ["x"]
    assert "all_levels_rare" in codes(encoder, "code")


def test_no_learnable_column_is_an_error():
    # The user keeps only a re-included identifier: nothing is left to learn, so the encoder refuses
    # instead of producing an empty vector.
    columns = {"x": [f"{i}.5" for i in range(N_ROWS)], "code": [f"P{i:04d}" for i in range(N_ROWS)]}
    with pytest.raises(DataEngineError) as error:
        fitted(columns, UserChoices(included={"code"}, excluded={"x"}))
    assert error.value.code == "nothing_to_learn"
    assert "'code'" in error.value.message


def test_every_unlearnable_column_is_named_in_the_error():
    # 'x' is left out by the user; in the first 100 rows 'code' has only rare levels and 'k' is
    # constant: two different reasons, both columns named.
    columns = {"x": [f"{i}.5" for i in range(N_ROWS)], "code": [f"P{i:04d}" for i in range(N_ROWS)],
               "k": [0] * 100 + [16] * 11 + [17] * 9}
    with pytest.raises(DataEngineError) as error:
        fitted(columns, UserChoices(included={"code"}, excluded={"x"}), rows=slice(0, 100))
    assert error.value.code == "nothing_to_learn"
    assert "'code'" in error.value.message and "'k'" in error.value.message


# --------------------------------------------------------------------------------------------
# Holdout rows: encoded with the train parameters, never refit
# --------------------------------------------------------------------------------------------

def test_scaling_is_fit_on_train_rows_only():
    encoder, train = fitted(base_columns(), rows=slice(0, 60))
    full = encoder.transform(fitted(base_columns())[1])
    x = group(encoder, "x")
    assert encoder.transform(train)[:, x.start].max() == 1
    assert full[:, x.start].max() > 1


def test_out_of_range_holdout_values_are_not_clipped(base):
    encoder, train = base
    rows = train.iloc[[0, 1]].copy()
    rows["x"] = [-10.0, 1000.0]
    x = group(encoder, "x")
    values = encoder.transform(rows)[:, x.start]
    assert values[0] < 0 and values[1] > 1


def test_unknown_holdout_level_goes_to_other(base):
    encoder, train = base
    row = train.iloc[[0]].copy()
    row["c"] = "never_seen"
    c = group(encoder, "c")
    assert encoder.transform(row)[0, c.start:c.stop].tolist() == [0, 0, 1]


def test_unknown_holdout_level_without_other_is_all_zeros():
    columns = {"x": [f"{i}.5" for i in range(N_ROWS)], "arm": ["a", "b"] * (N_ROWS // 2)}
    encoder, train = fitted(columns)
    row = train.iloc[[0]].copy()
    row["arm"] = "never_seen"
    arm = group(encoder, "arm")
    assert encoder.transform(row)[0, arm.start:arm.stop].tolist() == [0, 0]


# --------------------------------------------------------------------------------------------
# Dates: days since the first train date, then MinMax
# --------------------------------------------------------------------------------------------

def test_dates_become_scaled_days():
    days = pd.date_range("2020-01-01", periods=N_ROWS, freq="D")
    columns = {"x": [f"{i}.5" for i in range(N_ROWS)], "visit": [d.strftime("%Y-%m-%d") for d in days]}
    encoder, train = fitted(columns)
    visit = group(encoder, "visit")
    values = encoder.transform(train)[:, visit.start]
    assert (visit.role, visit.stop - visit.start) == ("date", 1)
    assert values.min() == 0 and values.max() == 1
    assert values[1] == pytest.approx(1 / (N_ROWS - 1))      # one day = one step


# --------------------------------------------------------------------------------------------
# Memory: fit without building the encoded table, transform in row blocks
# --------------------------------------------------------------------------------------------
# scikit-learn's ColumnTransformer.fit runs fit_transform and throws the float64 output away, and its
# transform stacks float64 copies before the cast to float32: on a 200 MB CSV the Data Engine peaked
# at 8.6 GB for a 0.96 GB float32 vector. The encoder must give the same vector with far less memory.

MIXED_ROWS = 200
MEMORY_REPEATS = 1_000          # 120 x 1,000 = 120,000 rows
MEMORY_BLOCK_ROWS = 10_000      # smaller than the rows, so the test sees several blocks
MAX_TRANSFORM_PEAK = 2.0        # peak of transform / bytes of the float32 vector it returns
MAX_FIT_PEAK = 1.0              # peak of fit_encoder / bytes of the float32 vector of its train rows
                                # (ColumnTransformer: 4.3; per column: 0.6, copies of one column)


def mixed_columns() -> dict[str, list]:
    """Every role and every One-Hot case: rare levels, integer codes, no "Other", dates."""
    days = pd.date_range("2020-01-01", periods=MIXED_ROWS, freq="D")
    return {
        "x": [f"{i}.5" for i in range(MIXED_ROWS)],
        "c": ["a"] * 100 + ["b"] * 80 + ["r1"] * 9 + ["r2"] * 11,
        "k": [0] * 150 + [16] * 41 + [17] * 9,
        "arm": ["p", "q"] * (MIXED_ROWS // 2),
        "visit": [d.strftime("%Y-%m-%d") for d in days],
    }


def reference_vector(encoder, train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """The vector as scikit-learn's ColumnTransformer builds it, with the same per-column transformers."""
    def as_input(rows: pd.DataFrame) -> pd.DataFrame:
        values = rows[[g.name for g in encoder.groups]].copy()
        for name, origin in encoder.date_origins.items():
            values[name] = (pd.to_datetime(values[name]) - origin) / pd.Timedelta(days=1)
        for name in encoder.integer_codes:
            values[name] = values[name].astype("int64")
        return values

    def transformer(role):
        if role == "categorical":
            return OneHotEncoder(min_frequency=MIN_LEVEL_ROWS, max_categories=MAX_CATEGORIES,
                                 handle_unknown="infrequent_if_exist", sparse_output=False)
        return MinMaxScaler()

    steps = [(f"col{i}", transformer(g.role), [g.name]) for i, g in enumerate(encoder.groups)]
    reference = ColumnTransformer(steps, remainder="drop").fit(as_input(train))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # unknown holdout levels: expected
        return reference.transform(as_input(frame)).astype(np.float32)


def test_vector_matches_scikit_learn_column_transformer(monkeypatch):
    monkeypatch.setattr("sds.data_engine.encoding.ENCODE_BLOCK_ROWS", 7)   # blocks of uneven size
    encoder, train = fitted(mixed_columns())
    holdout = train.iloc[:30].copy()
    holdout.loc[holdout.index[0], "c"] = "never_seen"           # -> Other
    holdout.loc[holdout.index[1], "arm"] = "never_seen"         # no Other -> all zeros
    holdout.loc[holdout.index[2], "x"] = 10_000.0               # out of the train range, not clipped
    for frame in (train, holdout):
        assert np.array_equal(encoder.transform(frame), reference_vector(encoder, train, frame))


def test_empty_frame_gives_an_empty_vector(base):
    encoder, train = base
    vector = encoder.transform(train.iloc[:0])
    assert vector.shape == (0, encoder.width) and vector.dtype == np.float32


def large_train():
    """120,000 rows: 'x' numeric, 'c' text with 20 frequent levels, 'k' integer codes."""
    columns = {"x": [f"{i}.5" for i in range(N_ROWS)], "c": [f"level{i % 20}" for i in range(N_ROWS)],
               "k": [0, 1, 2] * (N_ROWS // 3)}
    inspection = inspect_csv(make_csv(columns))
    schema = apply_choices(inspection, UserChoices())
    small = inspection.frame[list(schema.columns)].dropna()
    return pd.concat([small] * MEMORY_REPEATS, ignore_index=True), schema


def peak_bytes(function) -> tuple[object, int]:
    """Result of ``function()`` and the peak of the memory it allocated (tracemalloc)."""
    tracemalloc.start()
    try:
        result = function()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, peak


def test_transform_peak_stays_close_to_its_output(monkeypatch):
    monkeypatch.setattr("sds.data_engine.encoding.ENCODE_BLOCK_ROWS", MEMORY_BLOCK_ROWS)
    train, schema = large_train()
    encoder = fit_encoder(train, schema)
    vector, peak = peak_bytes(lambda: encoder.transform(train))
    assert peak < MAX_TRANSFORM_PEAK * vector.nbytes


def test_fit_does_not_build_the_encoded_table():
    train, schema = large_train()
    encoder, peak = peak_bytes(lambda: fit_encoder(train, schema))
    assert encoder.width == 1 + 20 + 3
    assert peak < MAX_FIT_PEAK * len(train) * encoder.width * np.dtype(np.float32).itemsize
