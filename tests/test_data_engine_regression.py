"""Regression test of the Data Engine on the real benchmark datasets.

The torture fixture only checks the cases we imagined; real files keep surprising us (ICD-9 codes
taken for numbers). This module runs the whole Data Engine on the six benchmark CSVs: it compares
every proposed role with the hand-made labels in ``fixtures/ground_truth_columns.csv``, then checks
the user choices, the preparation, the encoding, the contract and the round trip on real data.

The CSVs are not in the repository (download them with ``sds-download-samples``). The shared fixtures
in ``conftest.py`` find them with ``sds.samples.data_dir()`` and skip these tests when they are missing.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sds.data_engine import (
    SchemaContract,
    UserChoices,
    apply_choices,
    build_contract,
    decode,
    fixed_reason,
    prepare,
)
from sds.data_engine.preparation import holdout_size
from sds.data_engine.types import USABLE_ROLES
from test_data_engine_decoding import assert_round_trip

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GROUND_TRUTH = FIXTURES / "ground_truth_columns.csv"

# Baseline measured on 2026-11-07: 140/146 columns right (or rightly dropped); 144/146 since ambiguous
# columns accept both readings (2026-11-09).
MIN_RIGHT = 144
# Known type errors at the baseline. Any error outside this list is a regression.
KNOWN_ERRORS = {
    ("diabetes_130", "patient_nbr"),               # repeated ID: not detectable (declared limit)
    ("diabetes_130", "discharge_disposition_id"),  # 26 codes > MAX_DISCRETE_LEVELS: fixed in Advanced
}
# Columns labelled ambiguous (small whole counts: days 0-30, pregnancies 0-17, an ordinal 1-16) are right
# as numeric or as categorical: categorical loses the order but keeps exact values (no invented counts,
# zeros preserved), numeric keeps the order.
AMBIGUOUS_ROLES = {"numeric", "categorical"}


@pytest.fixture(scope="module")
def evaluated(inspections) -> pd.DataFrame:
    """One row per column of every dataset: proposed role next to the true type."""
    rows = []
    for ds, inspection in inspections.items():
        rows += [{"dataset": ds, "column": name, "role": info.role} for name, info in inspection.columns.items()]
    proposed = pd.DataFrame(rows)
    truth = pd.read_csv(GROUND_TRUTH, keep_default_na=False)
    return proposed.merge(truth, on=["dataset", "column"], how="outer", validate="one_to_one", indicator=True)


def type_errors(evaluated: pd.DataFrame) -> set[tuple[str, str]]:
    """Columns whose role differs from the true type (dropping a column is not a type error; an ambiguous
    column may be read either way)."""
    either_way = (evaluated["ambiguous"] == "yes") & evaluated["role"].isin(AMBIGUOUS_ROLES)
    wrong = evaluated[(evaluated["role"] != evaluated["true_type"]) & (evaluated["role"] != "dropped")
                      & ~either_way]
    return set(zip(wrong["dataset"], wrong["column"], strict=True))


def test_every_column_has_a_label(evaluated):
    unmatched = evaluated[evaluated["_merge"] != "both"]
    assert unmatched.empty, f"columns and labels do not match:\n{unmatched[['dataset', 'column', '_merge']]}"


def test_no_new_type_errors(evaluated):
    new_errors = type_errors(evaluated) - KNOWN_ERRORS
    assert not new_errors, f"new type errors on real data: {sorted(new_errors)}"


def test_accuracy_does_not_drop(evaluated):
    n_right = len(evaluated) - len(type_errors(evaluated))
    assert n_right >= MIN_RIGHT, f"{n_right}/{len(evaluated)} right, baseline {MIN_RIGHT}"


# --------------------------------------------------------------------------------------------
# User choices on real data (the cases that motivated them)
# --------------------------------------------------------------------------------------------

def test_default_choices_keep_every_usable_column(inspections):
    for ds, inspection in inspections.items():
        usable = [n for n, info in inspection.columns.items() if info.role in USABLE_ROLES]
        assert list(apply_choices(inspection, UserChoices()).columns) == usable, ds


def test_discharge_codes_can_be_fixed_as_categories(inspections):
    # Known error: 26 codes inferred as numeric. The user fixes it in Advanced.
    schema = apply_choices(inspections["diabetes_130"], UserChoices(as_categorical={"discharge_disposition_id"}))
    assert schema.columns["discharge_disposition_id"].role == "categorical"


def test_pregnancies_can_be_fixed_as_a_quantity(inspections):
    schema = apply_choices(inspections["pima"], UserChoices(as_numeric={"Pregnancies"}))
    assert schema.columns["Pregnancies"].role == "numeric"


def test_changeable_readings_on_real_data(inspections):
    # Advanced panel, "You can change these": whole numbers with 3 to MAX_CATEGORIES values.
    def changeable(ds):
        inspection = inspections[ds]
        return {n for n, i in inspection.columns.items() if i.role in USABLE_ROLES and fixed_reason(i) is None}

    assert changeable("adult") == {"education-num"}
    assert changeable("breast_cancer") == set()
    assert changeable("cdc_diabetes") == {"GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"}
    assert "discharge_disposition_id" in changeable("diabetes_130")
    assert changeable("pima") == {"Pregnancies", "BloodPressure"}


def test_reincluding_payer_code_states_the_rows_lost(inspections):
    schema = apply_choices(inspections["diabetes_130"], UserChoices(included={"payer_code"}))
    (issue,) = [i for i in schema.issues if i.column == "payer_code"]
    assert "40%" in issue.message


# --------------------------------------------------------------------------------------------
# Preparation on real data: rows removed, split, small datasets
# --------------------------------------------------------------------------------------------

# Measured on 2026-11-07 with the default choices.
ROWS_REMOVED = {"adult": 2399, "breast_cancer": 0, "bank_marketing": 0, "cdc_diabetes": 0,
                "diabetes_130": 2273, "pima": 0}
SMALL_DATASETS = {"breast_cancer", "pima"}


def test_rows_removed_on_real_data(preparations):
    assert {ds: p.rows_removed for ds, p in preparations.items()} == ROWS_REMOVED


def test_split_on_real_data(preparations):
    for ds, p in preparations.items():
        assert not p.train.index.intersection(p.holdout.index).size, ds
        assert len(p.train) + len(p.holdout) == p.rows_read - p.rows_removed, ds
        assert len(p.holdout) == holdout_size(p.rows_read - p.rows_removed), ds


def test_small_datasets_get_the_warning(preparations):
    warned = {ds for ds, p in preparations.items() if any(i.code == "few_rows" for i in p.issues)}
    assert warned == SMALL_DATASETS


def test_bringing_back_incomplete_columns_names_them(inspections):
    # payer_code is 40% empty, medical_specialty ~49%: together most rows are lost.
    inspection = inspections["diabetes_130"]
    schema = apply_choices(inspection, UserChoices(included={"payer_code", "medical_specialty"}))
    (issue,) = [i for i in prepare(inspection, schema).issues if i.code == "many_rows_removed"]
    assert "'medical_specialty'" in issue.message and "'payer_code'" in issue.message


# --------------------------------------------------------------------------------------------
# Near-constant columns and encoding on real data
# --------------------------------------------------------------------------------------------

# Measured on 2026-11-08 on the whole file: rows differing from the most frequent value < 10.
NEAR_CONSTANT = {
    "diabetes_130": {"acetohexamide", "troglitazone", "glimepiride-pioglitazone",
                     "metformin-rosiglitazone", "metformin-pioglitazone"},
}


def test_near_constant_columns_on_real_data(inspections):
    for ds, inspection in inspections.items():
        found = {i.column for i in inspection.issues if i.code == "near_constant"}
        assert found == NEAR_CONSTANT.get(ds, set()), ds
    kept = inspections["diabetes_130"].columns
    assert kept["tolbutamide"].role == kept["glipizide-metformin"].role == "categorical"


def test_encoding_on_real_data(encoders, preparations):
    for ds, encoder in encoders.items():
        train = encoder.transform(preparations[ds].train)
        holdout = encoder.transform(preparations[ds].holdout)
        assert train.shape == (len(preparations[ds].train), encoder.width), ds
        assert holdout.shape[1] == encoder.width and not np.isnan(holdout).any(), ds
        for g in encoder.groups:
            block = train[:, g.start:g.stop]
            if g.role == "categorical":
                assert (block.sum(axis=1) == 1).all(), (ds, g.name)
            else:
                assert block.min() == 0 and block.max() == 1, (ds, g.name)


def test_rare_levels_on_real_data(encoders):
    assert "native-country" in {i.column for i in encoders["adult"].issues if i.code == "levels_merged"}
    pregnancies = next(g for g in encoders["pima"].groups if g.name == "Pregnancies")
    assert pregnancies.has_other and all(type(v) is int for v in pregnancies.merged)


# --------------------------------------------------------------------------------------------
# Schema contract on real data
# --------------------------------------------------------------------------------------------

def test_contract_on_real_data(inspections, preparations, encoders):
    for ds, inspection in inspections.items():
        schema = apply_choices(inspection, UserChoices())
        contract = build_contract(inspection, schema, preparations[ds], encoders[ds])
        assert SchemaContract.from_dict(json.loads(json.dumps(contract.to_dict()))) == contract, ds
        assert contract.width == encoders[ds].width, ds
        names = {c.name for c in contract.columns} | {d.name for d in contract.dropped}
        assert names == set(inspection.columns), ds          # every column is accounted for
        for entry in contract.llm_schema():
            assert set(entry) <= {"name", "type", "values"}, (ds, entry)


# --------------------------------------------------------------------------------------------
# Round trip on real data: decode(encode(train)) == train
# --------------------------------------------------------------------------------------------

def test_round_trip_on_real_data(inspections, preparations, encoders):
    for ds, inspection in inspections.items():
        schema = apply_choices(inspection, UserChoices())
        contract = build_contract(inspection, schema, preparations[ds], encoders[ds])
        train = preparations[ds].train
        assert_round_trip(train, decode(encoders[ds].transform(train), contract), contract, ds)
