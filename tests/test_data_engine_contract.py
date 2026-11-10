"""Specification tests for the schema contract.

The contract is the readable description of what was learned: the decoder reads only the contract,
the UI shows its report, the LLM gets a reduced projection. Public API under test:
    build_contract(inspection, schema, prepared, encoder) -> SchemaContract
        .version, .width
        .columns  list[ContractColumn]  (name, role, start, stop; numeric: min, max, integer;
                  date: origin (ISO), min, max in days; categorical: levels, integer, has_other,
                  merged as ((level, train rows), ...))
        .dropped  list[DroppedColumn]   (name, reason): by the engine, by the user, or not learnable
        .rows     dict                  read, removed, train, holdout
        .issues   list[Issue]           warnings of every stage
        .seen_pairs list[SeenPairs]     (first, second, slots): for every pair of categorical columns, the
                  One-Hot slot pairs found together in at least one *train* row (the holdout never counts)
        .to_dict() / SchemaContract.from_dict(data)   JSON-ready; other version than CONTRACT_VERSION ->
                  DataEngineError("unsupported_contract")
        .llm_schema() -> list[dict]     names, types and values only: no statistics, no seen pairs
"""

import json
from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from sds.data_engine import (
    DataEngineError,
    SchemaContract,
    UserChoices,
    apply_choices,
    build_contract,
    fit_encoder,
    inspect_csv,
    prepare,
)
from sds.data_engine.contract import CONTRACT_VERSION

N_ROWS = 120
STATISTIC_KEYS = {"min", "max", "merged", "origin", "rows", "count", "frequency"}


def make_csv(columns: dict[str, list]) -> bytes:
    """CSV bytes from equal-length columns."""
    names = list(columns)
    rows = zip(*(columns[name] for name in names), strict=True)
    return ("\n".join([",".join(names)] + [",".join(map(str, row)) for row in rows]) + "\n").encode()


def base_columns() -> dict[str, list]:
    """One column per kind of contract entry."""
    days = pd.date_range("2020-01-01", periods=60, freq="D")
    return {
        "x": [f"{i}.5" for i in range(N_ROWS)],                     # numeric, decimals
        "n": [i % 40 for i in range(N_ROWS)],                       # numeric, whole numbers (40 values)
        "visit": [days[i % 60].strftime("%Y-%m-%d") for i in range(N_ROWS)],
        "c": ["a"] * 60 + ["b"] * 50 + ["r1"] * 9 + ["r2"],         # text with rare levels
        "k": [0] * 100 + [16] * 11 + [17] * 9,                      # integer codes with rare levels
        "pid": [f"P{i:04d}" for i in range(N_ROWS)],                # suspected ID: left out
        "arm": ["A", "B"] * (N_ROWS // 2),                          # removed by the user
    }


def built(columns: dict[str, list], choices: UserChoices):
    inspection = inspect_csv(make_csv(columns))
    schema = apply_choices(inspection, choices)
    prepared = prepare(inspection, schema)
    encoder = fit_encoder(prepared.train, schema)
    return build_contract(inspection, schema, prepared, encoder), prepared, encoder


@pytest.fixture(scope="module")
def base():
    return built(base_columns(), UserChoices(excluded={"arm"}))


def column(contract, name):
    (found,) = [c for c in contract.columns if c.name == name]
    return found


def through_json(contract) -> SchemaContract:
    return SchemaContract.from_dict(json.loads(json.dumps(contract.to_dict())))


# --------------------------------------------------------------------------------------------
# JSON round trip
# --------------------------------------------------------------------------------------------

def test_contract_survives_a_json_round_trip(base):
    contract, _, _ = base
    assert through_json(contract) == contract


def test_integer_levels_stay_integers_through_json(base):
    k = column(through_json(base[0]), "k")
    assert all(type(level) is int for level in k.levels)
    assert all(type(level) is int and type(rows) is int for level, rows in k.merged)


def test_unknown_version_is_refused(base):
    data = base[0].to_dict()
    data["version"] = 999
    with pytest.raises(DataEngineError) as error:
        SchemaContract.from_dict(data)
    assert error.value.code == "unsupported_contract"


def test_a_contract_of_the_previous_version_is_refused(base):
    # Version 1 had no seen pairs: decoding it would skip the coherence repair without saying so.
    data = base[0].to_dict()
    data["version"] = 1
    with pytest.raises(DataEngineError) as error:
        SchemaContract.from_dict(data)
    assert error.value.code == "unsupported_contract"


# --------------------------------------------------------------------------------------------
# Consistency with the encoder and the train rows
# --------------------------------------------------------------------------------------------

def test_positions_match_the_encoder(base):
    contract, _, encoder = base
    assert contract.version == CONTRACT_VERSION
    assert contract.width == encoder.width
    assert [(c.name, c.role, c.start, c.stop) for c in contract.columns] == \
        [(g.name, g.role, g.start, g.stop) for g in encoder.groups]


def test_numeric_ranges_come_from_the_train_rows(base):
    contract, prepared, _ = base
    for name, integer in [("x", False), ("n", True)]:
        entry = column(contract, name)
        assert entry.min == prepared.train[name].min() and entry.max == prepared.train[name].max()
        assert entry.integer is integer


def test_dates_are_described_in_days_from_the_first_train_date(base):
    contract, prepared, _ = base
    visit = column(contract, "visit")
    first, last = prepared.train["visit"].min(), prepared.train["visit"].max()
    assert visit.origin == first.date().isoformat()
    assert (visit.min, visit.max) == (0, (last - first).days)


def test_categories_match_the_encoder(base):
    contract, _, encoder = base
    for g in encoder.groups:
        if g.role == "categorical":
            entry = column(contract, g.name)
            assert entry.levels == g.levels and entry.has_other == g.has_other
            assert dict(entry.merged) == g.merged
    assert column(contract, "k").integer is True
    assert column(contract, "c").integer is False


# --------------------------------------------------------------------------------------------
# Report: dropped columns, rows, warnings
# --------------------------------------------------------------------------------------------

def test_columns_left_out_are_reported_with_a_reason(base):
    dropped = {d.name: d.reason for d in base[0].dropped}
    assert set(dropped) == {"pid", "arm"}
    assert all(dropped.values())


def test_columns_that_cannot_be_learned_are_reported():
    # A re-included identifier: every value is rare, so the encoder gives it no positions.
    contract, _, _ = built(base_columns(), UserChoices(excluded={"arm"}, included={"pid"}))
    assert "pid" in {d.name for d in contract.dropped}
    assert "pid" not in {c.name for c in contract.columns}


def test_row_counts(base):
    contract, prepared, _ = base
    assert contract.rows == {"read": N_ROWS, "removed": prepared.rows_removed,
                             "train": len(prepared.train), "holdout": len(prepared.holdout)}


def test_warnings_of_every_stage_are_collected(base):
    found = {i.code for i in base[0].issues}
    assert "suspected_id" in found      # reading / inference
    assert "few_rows" in found          # preparation
    assert "levels_merged" in found     # encoding


# --------------------------------------------------------------------------------------------
# Projection for the LLM: metadata of the synthetic data, no statistics
# --------------------------------------------------------------------------------------------

def test_llm_schema_types(base):
    types = {entry["name"]: entry["type"] for entry in base[0].llm_schema()}
    assert types == {"x": "float", "n": "integer", "visit": "date", "c": "category", "k": "integer category"}


def test_llm_schema_values(base):
    contract, _, _ = base
    entries = {entry["name"]: entry for entry in contract.llm_schema()}
    c, k = column(contract, "c"), column(contract, "k")
    assert entries["c"]["values"] == ([*c.levels, "Other"] if c.has_other else list(c.levels))
    # Merged integer values can appear in the synthetic CSV: listed, without their row counts.
    assert entries["k"]["values"] == sorted([*k.levels, *(level for level, _ in k.merged)])
    assert "values" not in entries["x"] and "values" not in entries["visit"]


def test_llm_schema_has_no_statistics_and_no_report(base):
    schema = base[0].llm_schema()
    json.dumps(schema)                                   # JSON-ready
    for entry in schema:
        assert set(entry) <= {"name", "type", "values"}
        assert not set(entry) & STATISTIC_KEYS
    assert {entry["name"] for entry in schema} == {"x", "n", "visit", "c", "k"}   # no dropped column


# --------------------------------------------------------------------------------------------
# Seen pairs: which categorical values appear together in the train rows
# --------------------------------------------------------------------------------------------
# Decoding samples every categorical group on its own, so it can write pairs no real row has (on Adult
# "education" HS-grad with "education-num" 10). The contract keeps, per pair of categorical columns,
# the One-Hot slot pairs seen in train; decoding repairs the rows that break them.

def train_slots(contract, encoder, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Slot index of every categorical column, from the encoded rows."""
    vectors = encoder.transform(rows)
    return {c.name: vectors[:, c.start:c.stop].argmax(axis=1) for c in contract.columns if c.role == "categorical"}


def as_dict(contract) -> dict[tuple[str, str], set]:
    return {(p.first, p.second): set(p.slots) for p in contract.seen_pairs}


def test_one_entry_per_pair_of_categorical_columns_in_contract_order(base):
    contract = base[0]
    categorical = [c.name for c in contract.columns if c.role == "categorical"]
    assert [(p.first, p.second) for p in contract.seen_pairs] == list(combinations(categorical, 2))


def test_seen_pairs_are_the_slot_pairs_of_the_train_rows(base):
    contract, prepared, encoder = base
    slots = train_slots(contract, encoder, prepared.train)
    for (first, second), seen in as_dict(contract).items():
        assert seen == set(zip(slots[first].tolist(), slots[second].tolist(), strict=True))


def test_slots_are_plain_ints_sorted(base):
    for pairs in base[0].seen_pairs:
        assert all(type(i) is int and type(j) is int for i, j in pairs.slots)
        assert list(pairs.slots) == sorted(pairs.slots)


def test_holdout_rows_do_not_count():
    inspection = inspect_csv(make_csv(base_columns()))
    schema = apply_choices(inspection, UserChoices(excluded={"arm"}))
    prepared = prepare(inspection, schema)
    encoder = fit_encoder(prepared.train, schema)
    contract = build_contract(inspection, schema, prepared, encoder)
    # Holdout rows with every combination of levels: the seen pairs must not change.
    other = prepared.holdout.copy()
    other["c"] = (["a", "b"] * len(other))[:len(other)]
    other["k"] = ([0, 0, 16, 16] * len(other))[:len(other)]
    changed = build_contract(inspection, schema, replace(prepared, holdout=other), encoder)
    assert changed.seen_pairs == contract.seen_pairs


def test_a_value_pair_never_in_train_is_not_seen():
    # "c" decides "k" in every row: a<->0, b<->16. (a, 16) must not be a seen pair.
    columns = {"c": ["a"] * 60 + ["b"] * 60, "k": [0] * 60 + [16] * 60}
    contract, _, _ = built(columns, UserChoices())
    c, k = column(contract, "c"), column(contract, "k")
    (pairs,) = contract.seen_pairs
    assert set(pairs.slots) == {(c.levels.index("a"), k.levels.index(0)), (c.levels.index("b"), k.levels.index(16))}


def test_seen_pairs_survive_json(base):
    assert through_json(base[0]).seen_pairs == base[0].seen_pairs


def test_llm_schema_does_not_carry_the_seen_pairs(base):
    text = json.dumps(base[0].llm_schema())
    assert "slots" not in text and "seen" not in text
