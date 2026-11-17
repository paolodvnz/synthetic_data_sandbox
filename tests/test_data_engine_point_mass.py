"""Specification tests for the point masses of the numeric columns (indicator of the dominant value).

A continuous decoder cannot put a share of rows on one exact value (Adult ``capital-gain`` = 0 in 92% of the rows).
The encoder finds, on the train rows only, the **point mass** of every numeric column: its most frequent value,
when it covers at least ``MIN_MASS_SHARE`` of the rows and is at least ``MIN_MASS_RATIO`` times as frequent as the
second value. Such a column takes three positions, ``[continuous, yes, no]``: the continuous value as today
(MinMax on every train row) and a two-slot one-hot indicator "the row has the mass value". Public API under test:
    fit_encoder(...).groups[i].mass          the mass value (original type) or None
    ContractColumn.mass                      the same in the contract (numeric columns only), JSON round trip
    SchemaContract.onehot_groups()           every one-hot group in vector order: categorical columns and the
                                             indicators of the mass columns (name, start, stop)
    CONTRACT_VERSION >= 5 (6 since the date formats)
    transform: indicator [1, 0] on the mass value, [0, 1] elsewhere (holdout rows with the train mass)
    decode: the indicator is sampled first; "yes" -> exactly the mass value, "no" -> the continuous value as today
    seen_pairs and the coherence repair cover the indicators (Adult: gain and loss never both non-zero)
The synthetic table has only the contract columns: no indicator ever reaches the user or the LLM.
"""

import hashlib
import json

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
from sds.data_engine.contract import CONTRACT_VERSION, ContractColumn
from sds.data_engine.decoding import decode_checked
from sds.data_engine.thresholds_config import MIN_MASS_RATIO, MIN_MASS_SHARE
from test_data_engine_decoding import assert_round_trip
from test_data_engine_encoding import make_csv

N_ROWS = 200
YES, NO = 0, 1                       # indicator slots
N_SAMPLES = 4000
PROPORTION_TOLERANCE = 0.03


def mass_columns() -> dict[str, list]:
    """Toy table with three masses and four columns without one.

    gain:  0 in 60% of the rows (mass on the minimum), distinct positive values elsewhere;
    loss:  0 in 80% of the rows, and never non-zero together with gain (as in Adult);
    hours: 40 in 50% of the rows (mass inside the range), shuffled: independent of gain and loss;
    x:     distinct floats (no mass); few: top value in 3% of the rows (below MIN_MASS_SHARE);
    flat:  top value twice the second (below MIN_MASS_RATIO); visit: a date with a dominant day (dates: no mass);
    c:     a category.
    """
    gain = [0] * 120 + list(range(1, 81))
    loss = [0] * 80 + list(range(100, 140)) + [0] * 80
    hours = [40] * 100 + [20 + i % 40 + (1 if 20 + i % 40 >= 40 else 0) for i in range(100)]
    hours = np.random.default_rng(1).permutation(hours).tolist()     # independent of gain and loss
    few = [7.25] * 6 + [i + 0.5 for i in range(194)]
    flat = [5.5] * 60 + [6.5] * 30 + [i + 0.25 for i in range(110)]
    visit = ["2020-01-01"] * 120 + [f"2021-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(80)]
    return {
        "gain": gain, "loss": loss, "hours": hours,
        "x": [i + 0.5 for i in range(N_ROWS)], "few": few, "flat": flat, "visit": visit,
        "c": ["a", "b"] * (N_ROWS // 2),
    }


def built(columns: dict[str, list], choices: UserChoices | None = None):
    """Inspection -> schema -> prepare -> encoder -> contract on a small CSV (every row complete)."""
    inspection = inspect_csv(make_csv(columns))
    schema = apply_choices(inspection, choices or UserChoices())
    prepared = prepare(inspection, schema)
    encoder = fit_encoder(prepared.train, schema)
    return prepared, encoder, build_contract(inspection, schema, prepared, encoder)


@pytest.fixture(scope="module")
def toy():
    return built(mass_columns())


def group(encoder, name):
    (found,) = [g for g in encoder.groups if g.name == name]
    return found


def column(contract, name) -> ContractColumn:
    (found,) = [c for c in contract.columns if c.name == name]
    return found


def mass_share(values: pd.Series, value) -> float:
    return float((values.astype(float) == value).mean())


# --------------------------------------------------------------------------------------------
# Which columns have a point mass
# --------------------------------------------------------------------------------------------

def test_thresholds_are_the_ones_of_the_experiment():
    assert MIN_MASS_SHARE == 0.05
    assert MIN_MASS_RATIO == 3


def test_the_mass_is_the_most_frequent_value(toy):
    _, encoder, _ = toy
    assert group(encoder, "gain").mass == 0
    assert group(encoder, "loss").mass == 0
    assert group(encoder, "hours").mass == 40          # inside the range: no difference


def test_columns_without_a_mass(toy):
    _, encoder, _ = toy
    for name in ("x", "few", "flat", "visit", "c"):
        assert group(encoder, name).mass is None, name


def test_thresholds_on_the_train_rows_exactly():
    """The rule on the train rows themselves (fit_encoder sees only them): boundaries included."""
    from sds.data_engine.encoding import point_mass
    n = 200
    assert point_mass(pd.Series([3.25] * 10 + [4.25] + [10.5 + i for i in range(n - 11)])) == 3.25
    assert point_mass(pd.Series([3.25] * 9 + [4.25] + [10.5 + i for i in range(n - 10)])) is None
    assert point_mass(pd.Series([3.25] * 30 + [4.25] * 10 + [10.5 + i for i in range(n - 40)])) == 3.25
    assert point_mass(pd.Series([3.25] * 29 + [4.25] * 10 + [10.5 + i for i in range(n - 39)])) is None
    assert point_mass(pd.Series([2.5] * 50 + [3.5] * 50)) is None          # two equal values: ratio 1
    # the share is over the rows, not over the distinct values: 4% of the rows (8 of 200) is not a mass,
    # even with few distinct values (8 / 97 levels would be 8%)
    repeated = [3.25] * 8 + [4.25] * 2 + [10.5 + i for i in range(95) for _ in range(2)]
    assert len(repeated) == n and point_mass(pd.Series(repeated)) is None
    assert point_mass(pd.Series([2.5] * 100)) is None                       # one value: no second value


def test_integer_mass_keeps_its_type():
    from sds.data_engine.encoding import point_mass
    found = point_mass(pd.Series([0] * 50 + list(range(1, 51)), dtype="int64"))
    assert found == 0 and isinstance(found, int)


def test_the_mass_is_found_on_the_train_rows_only():
    """Zeros only in the rows that end up in the holdout cannot create a mass."""
    columns = mass_columns()
    inspection = inspect_csv(make_csv(columns))
    schema = apply_choices(inspection, UserChoices())
    prepared = prepare(inspection, schema)
    train = prepared.train.copy()
    train["gain"] = np.arange(1, len(train) + 1)          # no zero left in the train rows
    encoder = fit_encoder(train, schema)
    assert group(encoder, "gain").mass is None
    # the holdout keeps its zeros, and they are encoded as plain values
    assert encoder.transform(prepared.holdout).shape[1] == encoder.width


def test_a_column_turned_numeric_by_the_user_follows_the_same_rule():
    columns = {"visits": [0] * 140 + [1 + i % 10 for i in range(60)], "c": ["a", "b"] * (N_ROWS // 2)}
    _, default, _ = built(columns)
    assert group(default, "visits").role == "categorical" and group(default, "visits").mass is None
    _, encoder, _ = built(columns, UserChoices(as_numeric={"visits"}))
    assert group(encoder, "visits").role == "numeric" and group(encoder, "visits").mass == 0


# --------------------------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------------------------

def test_the_point_masses_came_with_version_5(toy):
    # Version 6 added the date formats (test_data_engine_dates): the masses stay.
    assert CONTRACT_VERSION >= 5
    assert toy[2].version == CONTRACT_VERSION


def test_a_mass_column_takes_three_positions(toy):
    _, encoder, contract = toy
    for name, value in (("gain", 0), ("loss", 0), ("hours", 40)):
        entry = column(contract, name)
        assert (entry.stop - entry.start, entry.mass) == (3, value), name
        assert (group(encoder, name).stop - group(encoder, name).start) == 3, name
    for name in ("x", "few", "flat", "visit"):
        entry = column(contract, name)
        assert (entry.stop - entry.start, entry.mass) == (1, None), name
    assert contract.width == encoder.width == sum(c.stop - c.start for c in contract.columns)


def test_mass_keeps_the_type_of_the_column(toy):
    _, _, contract = toy
    assert isinstance(column(contract, "gain").mass, int)
    _, _, floats = built({"v": [2.5] * 120 + [i + 0.75 for i in range(80)], "c": ["a", "b"] * (N_ROWS // 2)})
    assert column(floats, "v").mass == 2.5 and isinstance(column(floats, "v").mass, float)


def test_mass_is_in_the_json_of_numeric_columns_only(toy):
    _, _, contract = toy
    data = contract.to_dict()
    by_name = {c["name"]: c for c in data["columns"]}
    assert by_name["gain"]["mass"] == 0
    assert by_name["x"]["mass"] is None
    assert "mass" not in by_name["visit"] and "mass" not in by_name["c"]


def test_json_round_trip_keeps_the_mass(toy):
    _, _, contract = toy
    again = SchemaContract.from_dict(json.loads(json.dumps(contract.to_dict())))
    assert again.columns == contract.columns
    assert again.seen_pairs == contract.seen_pairs


def test_version_4_contracts_are_rejected(toy):
    data = toy[2].to_dict() | {"version": 4}
    with pytest.raises(DataEngineError) as error:
        SchemaContract.from_dict(data)
    assert error.value.code == "unsupported_contract"


def test_the_llm_schema_does_not_know_about_the_mass(toy):
    _, _, contract = toy
    schema = contract.llm_schema()
    assert [entry["name"] for entry in schema] == [c.name for c in contract.columns]
    by_name = {entry["name"]: entry for entry in schema}
    assert by_name["gain"] == {"name": "gain", "type": "integer"}
    assert by_name["hours"] == {"name": "hours", "type": "integer"}
    assert "mass" not in json.dumps(schema)


def test_onehot_groups_list_categories_and_indicators_in_vector_order(toy):
    _, _, contract = toy
    groups = contract.onehot_groups()
    expected = []
    for entry in contract.columns:
        if entry.role == "categorical":
            expected.append((entry.name, entry.start, entry.stop))
        elif entry.mass is not None:
            expected.append((entry.name, entry.start + 1, entry.stop))
    assert [(g.name, g.start, g.stop) for g in groups] == expected
    assert [g.name for g in groups] == ["gain", "loss", "hours", "c"]


# --------------------------------------------------------------------------------------------
# Transform and decode
# --------------------------------------------------------------------------------------------

def test_transform_writes_the_indicator(toy):
    prepared, encoder, contract = toy
    vectors = encoder.transform(prepared.train)
    for name in ("gain", "loss", "hours"):
        entry = column(contract, name)
        on_mass = prepared.train[name].to_numpy() == entry.mass
        indicator = vectors[:, entry.start + 1:entry.stop]
        assert (indicator[on_mass] == [1, 0]).all(), name
        assert (indicator[~on_mass] == [0, 1]).all(), name


def test_the_continuous_position_is_scaled_as_today(toy):
    """MinMax fit on every train row, the mass rows included."""
    prepared, encoder, contract = toy
    vectors = encoder.transform(prepared.train)
    for name in ("gain", "hours"):
        entry = column(contract, name)
        values = prepared.train[name].to_numpy(dtype=float)
        assert (entry.min, entry.max) == (values.min(), values.max())
        expected = (values - entry.min) / (entry.max - entry.min)
        assert np.allclose(vectors[:, entry.start], expected, atol=1e-6), name


def test_holdout_rows_use_the_train_mass(toy):
    _, encoder, contract = toy
    rows = pd.DataFrame({name: [values[0]] * 2 for name, values in mass_columns().items()})
    rows["visit"] = pd.to_datetime(rows["visit"])
    rows.loc[1, "hours"] = 41
    entry = column(contract, "hours")
    indicator = encoder.transform(rows)[:, entry.start + 1:entry.stop]
    assert indicator.tolist() == [[1, 0], [0, 1]]


def test_decoding_the_encoded_train_gives_it_back(toy):
    prepared, encoder, contract = toy
    decoded = decode(encoder.transform(prepared.train), contract)
    assert_round_trip(prepared.train, decoded, contract)


def mass_vectors(contract, n_rows: int, **overrides) -> np.ndarray:
    """Coherent vectors: every value at the bottom of its range, every indicator on "yes", first level of "c"."""
    base = np.zeros((n_rows, contract.width), dtype=np.float32)
    for entry in contract.columns:
        if entry.mass is not None:
            base[:, entry.start + 1 + YES] = 1
        elif entry.role == "categorical":
            base[:, entry.start] = 1
    for name, values in overrides.items():
        entry = column(contract, name)
        base[:, entry.start:entry.stop] = values
    return base


def test_yes_gives_exactly_the_mass_whatever_the_continuous_value(toy):
    _, _, contract = toy
    table = decode(mass_vectors(contract, 3, hours=[[0.0, 1, 0], [0.7, 1, 0], [1.3, 1, 0]]), contract)
    assert table["hours"].tolist() == [40, 40, 40]
    assert table["hours"].dtype == np.int64


def test_no_gives_the_continuous_value_clipped_as_today(toy):
    _, _, contract = toy
    entry = column(contract, "hours")
    table = decode(mass_vectors(contract, 3, hours=[[-0.5, 0, 1], [1.5, 0, 1], [0.25, 0, 1]]), contract)
    quarter = round(entry.min + 0.25 * (entry.max - entry.min))
    assert table["hours"].tolist() == [entry.min, entry.max, quarter] == [20, 60, 30]


def test_the_indicator_is_sampled_not_argmax(toy):
    _, _, contract = toy
    vectors = mass_vectors(contract, N_SAMPLES, hours=[0.25, 0.3, 0.7])     # "no" -> 30, not 40
    table = decode(vectors, contract)
    assert mass_share(table["hours"], 40) == pytest.approx(0.3, abs=PROPORTION_TOLERANCE)


def test_same_seed_same_table(toy):
    _, _, contract = toy
    vectors = mass_vectors(contract, 500, hours=[0.5, 0.5, 0.5], gain=[0.2, 0.5, 0.5])
    first, second = decode(vectors, contract, seed=3), decode(vectors, contract, seed=3)
    pd.testing.assert_frame_equal(first, second)
    assert not decode(vectors, contract, seed=4).equals(first)


def test_the_synthetic_table_has_only_the_contract_columns(toy):
    _, _, contract = toy
    table = decode(mass_vectors(contract, 10), contract)
    assert list(table.columns) == [c.name for c in contract.columns]


# --------------------------------------------------------------------------------------------
# Seen pairs and coherence repair with the indicators
# --------------------------------------------------------------------------------------------

def test_seen_pairs_cover_the_indicators(toy):
    _, _, contract = toy
    pairs = {(p.first, p.second): set(p.slots) for p in contract.seen_pairs}
    assert set(pairs) == {("gain", "loss"), ("gain", "hours"), ("gain", "c"), ("loss", "hours"), ("loss", "c"),
                          ("hours", "c")}
    assert pairs["gain", "loss"] == {(YES, YES), (YES, NO), (NO, YES)}      # gain and loss never both non-zero


def test_the_repair_never_leaves_gain_and_loss_both_non_zero(toy):
    _, _, contract = toy
    gain = column(contract, "gain")
    vectors = mass_vectors(contract, N_SAMPLES, gain=[0.5, 0.0, 1.0], loss=[0.5, 0.2, 0.8])
    table, incoherent = decode_checked(vectors, contract)
    assert not incoherent.any()
    assert ((table["gain"] == gain.mass) | (table["loss"] == 0)).all()
    assert (table["loss"] == 0).all()                  # the only fixable column was the less likely "no" of loss


# Digest of ``decode`` on the table below, computed with the code before the point masses (contract version 4).
NO_MASS_DIGEST = "904c366ad700b01c163fce09132119b9aa1a4cee511f4d88a68a4231974787c7"


def test_a_contract_without_masses_decodes_exactly_as_before():
    """No mass: one position per numeric column and the same random draws as before (Breast Cancer stays identical)."""
    columns = {"x": [i + 0.5 for i in range(N_ROWS)], "n": [i % 37 * 3 for i in range(N_ROWS)],
               "c": ["a", "b", "c", "d"] * 50, "k": ["p", "q"] * 100}
    _, _, contract = built(columns)
    assert [c.mass for c in contract.columns] == [None] * 4
    assert [g.name for g in contract.onehot_groups()] == ["c", "k"]
    vectors = np.random.default_rng(0).random((300, contract.width)).astype(np.float32)
    table = decode(vectors, contract)
    assert hashlib.sha256(table.to_csv(index=False).encode()).hexdigest() == NO_MASS_DIGEST
