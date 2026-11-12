"""Specification tests for api.serialization.to_jsonable: backend objects -> plain JSON values."""

import datetime as dt
import enum
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.serialization import to_jsonable
from sds.data_engine import Issue


class Colour(enum.Enum):
    RED = "red"


@dataclass(frozen=True)
class Inner:
    value: float
    issues: list[Issue] = field(default_factory=list)


@dataclass
class Outer:
    name: str
    inner: Inner
    pairs: tuple[int, ...]


def roundtrip(value):
    """``to_jsonable`` must produce something ``json.dumps`` accepts in strict mode (no NaN)."""
    result = to_jsonable(value)
    json.dumps(result, allow_nan=False)
    return result


@pytest.mark.parametrize("value", [None, True, 3, 2.5, "text"])
def test_plain_values_are_unchanged(value):
    assert roundtrip(value) == value


def test_nested_dataclasses_become_dicts():
    value = Outer("a", Inner(1.0, [Issue("rows_removed", "Some rows removed.", "age")]), (1, 2))
    assert roundtrip(value) == {
        "name": "a",
        "inner": {"value": 1.0, "issues": [{"code": "rows_removed", "message": "Some rows removed.", "column": "age"}]},
        "pairs": [1, 2],
    }


def test_issue_without_column():
    assert roundtrip(Issue("few_rows", "Few rows.")) == {"code": "few_rows", "message": "Few rows.", "column": None}


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, np.float32("nan"), np.float64("inf")])
def test_non_finite_numbers_become_null(value):
    assert roundtrip(value) is None


def test_numpy_scalars_become_python_types():
    result = roundtrip([np.int64(3), np.float32(0.5), np.bool_(True)])
    assert result == [3, 0.5, True]
    assert [type(v) for v in result] == [int, float, bool]


def test_numpy_arrays_become_nested_lists():
    assert roundtrip(np.array([[1.0, np.nan], [2.0, 3.0]])) == [[1.0, None], [2.0, 3.0]]


def test_dataframe_keeps_columns_index_and_values():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [0.5, 1.0]}, index=["a", "b"])
    assert roundtrip(frame) == {"columns": ["a", "b"], "index": ["a", "b"], "values": [[1.0, 0.5], [None, 1.0]]}


def test_series_becomes_a_list():
    assert roundtrip(pd.Series([1, 2])) == [1, 2]


def test_dates_become_iso_strings():
    assert roundtrip([dt.date(2026, 11, 10), pd.Timestamp("2026-11-10 08:30")]) == ["2026-11-10", "2026-11-10T08:30:00"]


def test_missing_timestamp_becomes_null():
    assert roundtrip(pd.NaT) is None


def test_enum_path_and_sets():
    # {10, 3, 1} iterates as 1, 10, 3 in CPython: only sorting gives [1, 3, 10].
    assert roundtrip({"c": Colour.RED, "p": Path("/tmp/x"), "s": {"b", "a"}, "f": frozenset({10, 3, 1})}) == {
        "c": "red", "p": "/tmp/x", "s": ["a", "b"], "f": [1, 3, 10]}


def test_dict_keys_become_strings():
    assert roundtrip({1: "a", ("x", "y"): "b"}) == {"1": "a", "('x', 'y')": "b"}


def test_unknown_objects_are_rejected():
    with pytest.raises(TypeError, match="object"):
        to_jsonable(object())
