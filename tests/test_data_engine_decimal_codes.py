"""Specification tests for numbers with few values: they are categories, decimals or not.

A numeric column with at most MAX_DISCRETE_LEVELS distinct values is read as categorical by default, whether its
values are whole (codes 1, 2, 3) or not (damage multipliers 0.25, 0.5, 1, 2, 4; fixed ratios 11.2, 50, 88.1). Read
as numeric, the VAE spreads such a column between the real values (on a Pokemon dataset: 90% of the synthetic
values never seen in the real data). The reading can be changed like for whole numbers (3 to
MAX_CATEGORIES values). Decimal codes stay numbers end to end:
    fit_encoder   -> levels keep their decimals (never cast to int: 0.25 is not 0), ``decimal_codes``
    contract      -> ContractColumn.decimal_codes (version 4), ``numeric_codes`` for integer or decimal codes
    decode        -> float values among the levels; "Other" picks a merged value in train proportions
    fidelity      -> real and synthetic decimal codes compare equal
"""

import json

import numpy as np
import pandas as pd
import pytest

from sds.data_engine import (
    SchemaContract,
    UserChoices,
    apply_choices,
    build_contract,
    decode,
    fit_encoder,
    fixed_reason,
    inspect_csv,
    prepare,
)
from sds.data_engine.contract import CONTRACT_VERSION, OTHER_LABEL
from sds.data_engine.inference import base_role
from sds.data_engine.thresholds_config import MAX_CATEGORIES, MAX_DISCRETE_LEVELS, MIN_LEVEL_ROWS
from sds.validation.fidelity import comparable, fidelity_report
from sds.validation.fidelity_config import MIN_FIDELITY_Z

N_ROWS = 200
MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]            # 0.25 is rare (below MIN_LEVEL_ROWS): merged into "Other"
N_RARE = MIN_LEVEL_ROWS - 5
N_SAMPLES = 4000
PROPORTION_TOLERANCE = 0.03


def make_csv(columns: dict[str, list]) -> bytes:
    """CSV bytes from equal-length columns."""
    names = list(columns)
    rows = zip(*(columns[name] for name in names), strict=True)
    return ("\n".join([",".join(names)] + [",".join(map(str, row)) for row in rows]) + "\n").encode()


def columns() -> dict[str, list]:
    """'mult' decimal codes with a rare level, 'ratio' 7 fixed decimals, 'pair' two decimals, 'x' a measure."""
    common = [MULTIPLIERS[i % len(MULTIPLIERS)] for i in range(N_ROWS - N_RARE)]
    ratios = [0.0, 11.2, 24.6, 50.0, 75.4, 88.1, 100.0]
    return {
        "mult": common + [0.25] * N_RARE,
        "ratio": [ratios[i % len(ratios)] for i in range(N_ROWS)],
        "pair": [0.5 if i % 2 else 1.5 for i in range(N_ROWS)],
        "x": [f"{i}.5" for i in range(N_ROWS)],
    }


def decimals_with(n_values: int) -> list[str]:
    """``N_ROWS`` decimal values, ``n_values`` distinct (n.25 for n = 0, 1, ...)."""
    return [f"{i % n_values}.25" for i in range(N_ROWS)]


def with_measure(values: list[str]) -> bytes:
    """CSV of the column 'v' next to a measure (one column alone gives no separator to detect)."""
    return make_csv({"v": values, "x": [f"{i}.5" for i in range(len(values))]})


@pytest.fixture(scope="module")
def inspection():
    return inspect_csv(make_csv(columns()))


@pytest.fixture(scope="module")
def built(inspection):
    schema = apply_choices(inspection, UserChoices())
    prepared = prepare(inspection, schema)
    encoder = fit_encoder(prepared.train, schema)
    return build_contract(inspection, schema, prepared, encoder), prepared, encoder


def entry(contract, name):
    (found,) = [c for c in contract.columns if c.name == name]
    return found


def group(encoder, name):
    (found,) = [g for g in encoder.groups if g.name == name]
    return found


# --------------------------------------------------------------------------------------------
# Inference and user choices
# --------------------------------------------------------------------------------------------

def test_decimals_with_few_values_are_categories(inspection):
    for name in ("mult", "ratio", "pair"):
        assert inspection.columns[name].role == "categorical", name
    assert "5 distinct values" in inspection.columns["mult"].reason


def test_a_measure_with_many_decimals_stays_numeric(inspection):
    assert inspection.columns["x"].role == "numeric"


def test_the_limit_is_the_same_as_for_whole_numbers():
    at_limit = inspect_csv(with_measure(decimals_with(MAX_DISCRETE_LEVELS)))
    beyond = inspect_csv(with_measure(decimals_with(MAX_DISCRETE_LEVELS + 1)))
    assert at_limit.columns["v"].role == "categorical"
    assert beyond.columns["v"].role == "numeric"


def test_a_brought_back_decimal_column_follows_the_same_rule():
    assert base_role("numeric", MAX_DISCRETE_LEVELS) == "categorical"
    assert base_role("numeric", MAX_DISCRETE_LEVELS + 1) == "numeric"


def test_decimal_categories_can_be_read_as_quantities(inspection):
    assert fixed_reason(inspection.columns["mult"]) is None
    schema = apply_choices(inspection, UserChoices(as_numeric={"mult"}))
    assert schema.columns["mult"].role == "numeric"


def test_decimals_with_a_few_more_values_can_be_read_as_categories():
    inspection = inspect_csv(with_measure(decimals_with(MAX_DISCRETE_LEVELS + 1)))
    assert fixed_reason(inspection.columns["v"]) is None
    assert apply_choices(inspection, UserChoices(as_categorical={"v"})).columns["v"].role == "categorical"


def test_decimals_with_too_many_values_cannot_become_categories():
    inspection = inspect_csv(with_measure(decimals_with(MAX_CATEGORIES + 1)))
    assert "too many" in fixed_reason(inspection.columns["v"])


def test_two_decimal_values_keep_their_reading(inspection):
    assert "two values" in fixed_reason(inspection.columns["pair"])


# --------------------------------------------------------------------------------------------
# Encoding and contract
# --------------------------------------------------------------------------------------------

def test_decimal_levels_keep_their_decimals(built):
    _, prepared, encoder = built
    mult = group(encoder, "mult")
    assert mult.levels == tuple(MULTIPLIERS) and all(type(level) is float for level in mult.levels)
    assert mult.merged == {0.25: int((prepared.train["mult"] == 0.25).sum())}           # train rows only
    assert "mult" in encoder.decimal_codes and "mult" not in encoder.integer_codes


def test_the_rare_decimal_value_is_not_announced_as_other(built):
    # Like integer codes: the synthetic data write a merged value, never "Other".
    _, _, encoder = built
    (issue,) = [i for i in encoder.issues if i.code == "levels_merged" and i.column == "mult"]
    assert OTHER_LABEL not in issue.message and "'0.25'" in issue.message


def test_every_decimal_level_has_its_own_slot(built):
    _, prepared, encoder = built
    mult = group(encoder, "mult")
    slots = encoder.transform(prepared.train)[:, mult.start:mult.stop].argmax(axis=1)
    by_value = pd.Series(slots).groupby(prepared.train["mult"].to_numpy()).nunique()
    assert (by_value == 1).all()                                 # one slot per value
    assert pd.Series(slots).groupby(prepared.train["mult"].to_numpy()).first().nunique() == len(MULTIPLIERS) + 1


def test_the_contract_marks_decimal_codes(built):
    contract, _, _ = built
    mult, x = entry(contract, "mult"), entry(contract, "x")
    assert contract.version == CONTRACT_VERSION == 4
    assert mult.decimal_codes and not mult.integer and mult.numeric_codes
    assert not x.decimal_codes and not x.numeric_codes


def test_decimal_codes_survive_json(built):
    contract, _, _ = built
    again = SchemaContract.from_dict(json.loads(json.dumps(contract.to_dict())))
    assert again == contract
    assert all(type(level) is float for level in entry(again, "mult").levels)


def test_a_contract_of_version_3_is_refused(built):
    from sds.data_engine import DataEngineError

    data = built[0].to_dict()
    data["version"] = 3
    with pytest.raises(DataEngineError):
        SchemaContract.from_dict(data)


def test_llm_schema_lists_the_decimal_values(built):
    contract, _, _ = built
    (mult,) = [e for e in contract.llm_schema() if e["name"] == "mult"]
    assert mult == {"name": "mult", "type": "number category", "values": sorted([0.25, *MULTIPLIERS])}


# --------------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------------

def test_round_trip_gives_the_decimal_values_back(built):
    contract, prepared, encoder = built
    decoded = decode(encoder.transform(prepared.train), contract)
    kept = ~prepared.train["mult"].isin([0.25]).to_numpy()
    assert decoded["mult"].dtype == np.float64
    assert (decoded["mult"].to_numpy()[kept] == prepared.train["mult"].to_numpy()[kept]).all()
    assert (decoded["ratio"].to_numpy() == prepared.train["ratio"].to_numpy()).all()


def test_other_of_decimal_codes_picks_merged_values_not_a_label(built):
    contract, _, _ = built
    mult = entry(contract, "mult")
    vectors = np.zeros((N_SAMPLES, contract.width), dtype=np.float32)
    vectors[:, mult.stop - 1] = 1.0                             # the "Other" slot
    for other in contract.columns:
        if other.role == "categorical" and other.name != "mult":
            vectors[:, other.start] = 1.0
    decoded = decode(vectors, contract)
    assert OTHER_LABEL not in set(decoded["mult"].astype(str))
    assert set(decoded["mult"]) == {0.25}


def test_random_vectors_decode_only_to_real_values(built):
    contract, prepared, _ = built
    noise = np.random.default_rng(0).random((N_SAMPLES, contract.width)).astype(np.float32)
    decoded = decode(noise, contract)
    assert set(decoded["mult"]) <= {0.25, *MULTIPLIERS}
    assert set(decoded["ratio"]) <= set(prepared.train["ratio"])


def test_decoded_rows_encode_again_to_one_slot(built):
    # The privacy check re-encodes the synthetic rows: decimal codes must not become unknown values (all zeros).
    contract, _, encoder = built
    noise = np.random.default_rng(1).random((N_SAMPLES, contract.width)).astype(np.float32)
    again = encoder.transform(decode(noise, contract))
    mult = group(encoder, "mult")
    assert (again[:, mult.start:mult.stop].sum(axis=1) == 1).all()


# --------------------------------------------------------------------------------------------
# Fidelity
# --------------------------------------------------------------------------------------------

def test_real_and_synthetic_decimal_codes_compare_equal(built):
    contract, prepared, encoder = built
    decoded = decode(encoder.transform(prepared.train), contract)          # as many rows as the train, as in the app
    report = fidelity_report(decoded, prepared.holdout, prepared.train, contract)
    (mult,) = [c for c in report.columns if c.name == "mult"]
    (histogram,) = [h for h in report.histograms if h.name == "mult"]
    assert histogram.labels == sorted([0.25, *MULTIPLIERS])     # no value left outside the levels
    assert mult.z < MIN_FIDELITY_Z                     # as close as real people (values written apart: TV 1)
    assert comparable(prepared.holdout, contract)["mult"].dtype == np.float64


def test_proportions_of_the_merged_values():
    # Two rare decimal values: "Other" picks them in train proportions.
    values = [0.5] * 100 + [1.0] * 80 + [0.125] * 6 + [0.375] * 3
    inspection = inspect_csv(make_csv({"m": values, "x": [f"{i}.5" for i in range(len(values))]}))
    schema = apply_choices(inspection, UserChoices())
    prepared = prepare(inspection, schema)
    encoder = fit_encoder(inspection.frame[list(schema.columns)], schema)
    contract = build_contract(inspection, schema, prepared, encoder)
    m = entry(contract, "m")
    vectors = np.zeros((N_SAMPLES, contract.width), dtype=np.float32)
    vectors[:, m.stop - 1] = 1.0
    shares = decode(vectors, contract)["m"].value_counts(normalize=True)
    assert set(shares.index) == {0.125, 0.375}
    assert shares[0.125] == pytest.approx(6 / 9, abs=PROPORTION_TOLERANCE)
