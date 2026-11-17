"""Specification tests for the date formats.

Besides ISO, six formats are recognised: day/month/year and month/day/year with ``/``, ``-`` or ``.``, day and month
with or without a leading zero, a four-digit year, the same separator twice. A column is a date column when every
non-missing value is a real calendar date in **one** of them.

A column whose values all have the *shape* of a date (``n{1,2} SEP n{1,2} SEP nnnn``, or the ISO shape) but that is not
a date column is dropped with a clear warning:
    date_ambiguous    every value reads both as day/month and as month/day
    date_unreadable   mixed formats, impossible dates (31/02), ...
Anything else (a two-digit year, a time such as ``18.00.00``) follows the usual text rules, as before.

A few words among the dates follow the rule of the numbers: at least ``NUMERIC_MAJORITY`` of the values have the shape
of a date and the others are at most ``MAX_NULL_MARKERS`` different words -> the words are missing-value markers
(``mostly_dates`` warning) and the format rules apply to the rest; more different words, or fewer date-shaped values
-> text, silently (the warning is only for values turned into missing).

Public API under test:
    inspect_csv(...).columns[name].date_format   strftime format of a date column (ISO: "%Y-%m-%d"), else None
    ContractColumn.date_format                    the same in the contract (version 6)
    in_file_format(frame, contract) -> DataFrame  date columns written as text in their original format

Inside the app the dates stay dates (ISO when written as text): encoder, VAE, fidelity, privacy and ``llm_schema``
do not see the format.

Tests assert on stable codes, never on the user-facing messages.
"""

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
from sds.data_engine.contract import CONTRACT_VERSION

N_ROWS = 120
ISO = "%Y-%m-%d"
FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y"]
DAYS = pd.date_range("2024-01-01", periods=60, freq="D")          # days above 12: never ambiguous
AMBIGUOUS_DAYS = pd.DatetimeIndex([pd.Timestamp(2024, month, day) for month in range(1, 13) for day in (2, 5, 9, 11, 12)])


def make_csv(columns: dict[str, list]) -> bytes:
    """CSV bytes from equal-length columns."""
    names = list(columns)
    rows = zip(*(columns[name] for name in names), strict=True)
    return ("\n".join([",".join(names)] + [",".join(map(str, row)) for row in rows]) + "\n").encode()


def other_columns() -> dict[str, list]:
    """Ordinary columns next to the date, so that the file is usable without it."""
    return {"x": [f"{i}.5" for i in range(N_ROWS)], "c": ["a", "b", "c"] * (N_ROWS // 3)}


def date_values(fmt: str, days: pd.DatetimeIndex = DAYS) -> list[str]:
    return [days[i % len(days)].strftime(fmt) for i in range(N_ROWS)]


def inspected(values: list[str]):
    return inspect_csv(make_csv({"d": values, **other_columns()}))


def issue_codes(inspection, column: str = "d") -> set[str]:
    return {i.code for i in inspection.issues if i.column == column}


# --------------------------------------------------------------------------------------------
# Recognition: six formats besides ISO
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", FORMATS)
def test_each_format_is_a_date_column(fmt):
    inspection = inspected(date_values(fmt))
    info = inspection.columns["d"]
    assert info.role == "date" and info.kind == "date"
    assert info.date_format == fmt
    expected = pd.Series([DAYS[i % len(DAYS)] for i in range(N_ROWS)], name="d")
    pd.testing.assert_series_equal(inspection.frame["d"], expected, check_dtype=False)
    assert "date_privacy" in issue_codes(inspection)                 # same warning as an ISO date


def test_iso_dates_are_unchanged():
    inspection = inspected(date_values(ISO))
    assert inspection.columns["d"].role == "date"
    assert inspection.columns["d"].date_format == ISO


@pytest.mark.parametrize(("value", "fmt"), [("13/03/2024", "%d/%m/%Y"), ("03/13/2024", "%m/%d/%Y")])
def test_the_data_decide_day_first_or_month_first(value, fmt):
    # Every other value fits both readings: one day above 12 settles the column.
    values = date_values(fmt, AMBIGUOUS_DAYS)
    values[0] = value
    assert inspected(values).columns["d"].date_format == fmt


def test_leading_zeros_are_optional_in_the_same_column():
    values = [f"{d.day}/{d.month}/{d.year}" if i % 2 else d.strftime("%d/%m/%Y")
              for i, d in enumerate(DAYS[i % len(DAYS)] for i in range(N_ROWS))]
    info = inspected(values).columns["d"]
    assert info.role == "date" and info.date_format == "%d/%m/%Y"


def test_missing_values_do_not_prevent_a_date_column():
    values = date_values("%d.%m.%Y")
    values[3] = values[40] = ""
    inspection = inspected(values)
    assert inspection.columns["d"].role == "date"
    assert inspection.frame["d"].isna().sum() == 2


def test_columns_that_are_not_dates_have_no_format():
    inspection = inspected(date_values("%d/%m/%Y"))
    assert inspection.columns["x"].date_format is None
    assert inspection.columns["c"].date_format is None


# --------------------------------------------------------------------------------------------
# Dropped with a clear warning: the shape of a date, but not a date column
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("separator", ["/", "-", "."])
def test_an_ambiguous_column_is_dropped(separator):
    inspection = inspected(date_values(f"%d{separator}%m{separator}%Y", AMBIGUOUS_DAYS))
    assert inspection.columns["d"].role == "dropped"
    assert inspection.columns["d"].date_format is None
    assert "date_ambiguous" in issue_codes(inspection)
    assert "date_privacy" not in issue_codes(inspection)


@pytest.mark.parametrize(
    "change",
    [
        lambda v: [x if i % 2 else x.replace("/", "-") for i, x in enumerate(v)],     # two separators
        lambda v: [*v[:-1], "31/02/2024"],                                           # impossible date
        lambda v: [*v[:-2], "13/03/2024", "03/13/2024"],                             # day first and month first
        lambda v: [*v[:-1], "2024-03-15"],                                           # ISO among day/month
    ],
    ids=["separators", "impossible", "both_orders", "iso_mixed"],
)
def test_a_date_shaped_column_that_is_not_one_format_is_dropped(change):
    inspection = inspected(change(date_values("%d/%m/%Y")))
    assert inspection.columns["d"].role == "dropped"
    assert "date_unreadable" in issue_codes(inspection)


@pytest.mark.parametrize(
    ("values", "role"),
    [
        ([f"{h % 24:02d}.00.00" for h in range(N_ROWS)], "categorical"),           # Air Quality "Time"
        (date_values("%d/%m/%y"), "dropped"),                                         # two-digit year: > 50 values
        ([f"v{i % 30}" for i in range(N_ROWS)], "categorical"),
    ],
    ids=["time", "two_digit_year", "text"],
)
def test_other_text_follows_the_usual_rules_without_a_date_warning(values, role):
    inspection = inspected(values)
    assert inspection.columns["d"].role == role
    assert not issue_codes(inspection) & {"date_ambiguous", "date_unreadable", "date_privacy"}


def test_a_dropped_date_column_brought_back_is_text():
    inspection = inspected(date_values("%d/%m/%Y", AMBIGUOUS_DAYS))
    schema = apply_choices(inspection, UserChoices(included={"d"}))
    assert schema.columns["d"].role == "categorical"


def test_an_unreadable_date_column_is_not_a_suspected_id():
    # Daily dates are almost unique: the date warning comes before the identifier rule.
    days = pd.date_range("2020-01-01", periods=N_ROWS, freq="D")
    values = [d.strftime("%d/%m/%Y") for d in days]
    values[-1] = "31/02/2024"
    inspection = inspected(values)
    assert inspection.columns["d"].role == "dropped"
    assert "date_unreadable" in issue_codes(inspection) and "suspected_id" not in issue_codes(inspection)


# --------------------------------------------------------------------------------------------
# A few words among the dates: the rule of the numbers (NUMERIC_MAJORITY, MAX_NULL_MARKERS)
# --------------------------------------------------------------------------------------------

def with_words(values: list[str], words: list[str]) -> list[str]:
    """``values`` with the last ``len(words)`` cells replaced by ``words``."""
    return [*values[:len(values) - len(words)], *words]


@pytest.mark.parametrize("fmt", ["%d/%m/%Y", ISO])
def test_a_few_marker_words_become_missing_with_a_warning(fmt):
    words = ["n.d.", "n.d.", "unknown"]                              # 2 different words, 2.5% of the rows
    inspection = inspected(with_words(date_values(fmt), words))
    info = inspection.columns["d"]
    assert info.role == "date" and info.date_format == fmt
    assert inspection.frame["d"].isna().sum() == len(words)
    assert "mostly_dates" in issue_codes(inspection)


def test_many_different_words_keep_the_column_text_silently():
    words = ["w1", "w2", "w3", "w4"]                                 # more than MAX_NULL_MARKERS: labels, not markers
    inspection = inspected(with_words(date_values("%d/%m/%Y"), words))
    assert inspection.columns["d"].kind == "text"
    assert not issue_codes(inspection) & {"mostly_dates", "date_ambiguous", "date_unreadable", "date_privacy"}


def test_a_value_with_two_separators_is_a_marker_like_a_word():
    # "15-03/2024" has no date shape (two separators): like a malformed number, it becomes missing, with the warning.
    inspection = inspected(with_words(date_values("%d/%m/%Y"), ["15-03/2024"]))
    assert inspection.columns["d"].role == "date"
    assert "mostly_dates" in issue_codes(inspection)
    assert inspection.frame["d"].isna().sum() == 1


def test_too_many_words_follow_the_text_rules_silently():
    words = ["n.d."] * 10                                            # 8% of the rows: below NUMERIC_MAJORITY
    inspection = inspected(with_words(date_values("%d/%m/%Y"), words))
    assert inspection.columns["d"].kind == "text"
    assert not issue_codes(inspection) & {"mostly_dates", "date_ambiguous", "date_unreadable", "date_privacy"}


@pytest.mark.parametrize(
    ("values", "code"),
    [
        (date_values("%d/%m/%Y", AMBIGUOUS_DAYS), "date_ambiguous"),
        ([x if i % 2 else x.replace("/", "-") for i, x in enumerate(date_values("%d/%m/%Y"))], "date_unreadable"),
    ],
    ids=["ambiguous", "mixed"],
)
def test_without_the_markers_the_format_rules_apply(values, code):
    inspection = inspected(with_words(values, ["n.d.", "n.d."]))
    assert inspection.columns["d"].role == "dropped"
    assert {"mostly_dates", code} <= issue_codes(inspection)


# --------------------------------------------------------------------------------------------
# Times of day: dates are whole days, a time that is dropped is said (time_dropped)
# --------------------------------------------------------------------------------------------

HOURS = pd.date_range("2024-01-01", periods=N_ROWS, freq="7h")    # times of day that vary


@pytest.mark.parametrize("fmt", ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.000Z"])
def test_a_date_with_times_of_day_warns_that_the_time_is_dropped(fmt):
    inspection = inspected([t.strftime(fmt) for t in HOURS])
    assert inspection.columns["d"].role == "date"
    assert "time_dropped" in issue_codes(inspection)


@pytest.mark.parametrize("values", [date_values(ISO), date_values("%d/%m/%Y"), date_values("%Y-%m-%d 00:00:00")])
def test_whole_days_drop_nothing_and_do_not_warn(values):
    # Midnight everywhere ("2024-01-01 00:00:00") is a whole day too: no time is lost.
    inspection = inspected(values)
    assert inspection.columns["d"].role == "date"
    assert "time_dropped" not in issue_codes(inspection)


def test_one_time_of_day_is_enough_for_the_warning():
    values = date_values(ISO)
    values[0] = "2024-01-01 08:30:00"
    assert "time_dropped" in issue_codes(inspected(values))


# --------------------------------------------------------------------------------------------
# Contract (version 6) and decoding
# --------------------------------------------------------------------------------------------

def built(values: list[str]):
    inspection = inspected(values)
    schema = apply_choices(inspection, UserChoices())
    prepared = prepare(inspection, schema)
    encoder = fit_encoder(prepared.train, schema)
    return build_contract(inspection, schema, prepared, encoder), prepared, encoder


def entry(contract, name: str = "d"):
    (found,) = [c for c in contract.columns if c.name == name]
    return found


@pytest.fixture(scope="module")
def day_first():
    return built(date_values("%d/%m/%Y"))


@pytest.fixture(scope="module")
def iso():
    return built(date_values(ISO))


def test_contract_version_is_6(day_first):
    assert CONTRACT_VERSION == 6
    assert day_first[0].version == 6


def test_the_contract_keeps_the_format(day_first, iso):
    assert entry(day_first[0]).date_format == "%d/%m/%Y"
    assert entry(iso[0]).date_format == ISO


def test_the_format_survives_json(day_first):
    data = day_first[0].to_dict()
    assert next(c for c in data["columns"] if c["name"] == "d")["date_format"] == "%d/%m/%Y"
    assert "date_format" not in next(c for c in data["columns"] if c["name"] == "x")
    assert SchemaContract.from_dict(data) == day_first[0]


def test_version_5_contracts_are_rejected(day_first):
    with pytest.raises(DataEngineError) as error:
        SchemaContract.from_dict(day_first[0].to_dict() | {"version": 5})
    assert error.value.code == "unsupported_contract"


def test_the_format_changes_nothing_else(day_first, iso):
    """Same dates, other format: same contract (but the format), same vectors. VAE, fidelity and privacy work on
    these, so they cannot see the format."""
    contract_a, prepared_a, encoder_a = day_first
    contract_b, prepared_b, encoder_b = iso
    assert prepared_a.train.index.equals(prepared_b.train.index)
    a, b = entry(contract_a), entry(contract_b)
    assert {k: v for k, v in a.to_dict().items() if k != "date_format"} == \
           {k: v for k, v in b.to_dict().items() if k != "date_format"}
    np.testing.assert_array_equal(encoder_a.transform(prepared_a.train), encoder_b.transform(prepared_b.train))
    assert contract_a.llm_schema() == contract_b.llm_schema()


def test_decoded_dates_are_dates(day_first):
    contract, prepared, encoder = day_first
    table = decode(encoder.transform(prepared.train), contract)
    assert pd.api.types.is_datetime64_any_dtype(table["d"])


@pytest.mark.parametrize("fmt", [*FORMATS, ISO])
def test_in_file_format_writes_the_dates_as_in_the_file(fmt):
    from sds.data_engine.decoding import in_file_format     # inside: the other tests run before it exists

    contract, prepared, encoder = built(date_values(fmt))
    table = decode(encoder.transform(prepared.train), contract)
    table.loc[0, "d"] = pd.NaT
    written = in_file_format(table, contract)
    assert written["d"].iloc[1:].tolist() == table["d"].iloc[1:].dt.strftime(fmt).tolist()
    assert pd.isna(written["d"].iloc[0])                             # missing stays missing (an empty cell)
    pd.testing.assert_frame_equal(written.drop(columns="d"), table.drop(columns="d"))
    assert pd.api.types.is_datetime64_any_dtype(table["d"])          # the decoded table is not changed
