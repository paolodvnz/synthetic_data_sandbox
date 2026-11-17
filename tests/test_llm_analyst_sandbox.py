"""Tests for the safe execution of generated code: static check, child process, output.

Real answers of the model, recorded during development, run on random rows built from the schemas of Pima and Adult
(``fixtures/llm_schemas.json``): no sample download, no real data.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sds.llm_analyst import sandbox
from sds.llm_analyst.errors import NO_CHART, RUNTIME_ERROR, TIMEOUT, UNSAFE_CODE, AnalystError
from sds.llm_analyst.prompt import parse_answer
from sds.llm_analyst.sandbox import ChartResult, check_code, child_environment, columns_used, execute, run_code
from sds.llm_analyst.sandbox_config import CHART_DPI, MAX_SUMMARY_COLUMNS, MAX_SUMMARY_ROWS, PASSED_ENV

FIXTURES = Path(__file__).parent / "fixtures"
RESPONSES = json.loads((FIXTURES / "llm_responses.json").read_text())
SCHEMAS = json.loads((FIXTURES / "llm_schemas.json").read_text())
CHART_ANSWERS = [r for r in RESPONSES if r["kind"] == "chart"]
TEST_ROWS = 200
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

CHART = "fig, ax = plt.subplots()\ndf['age'].plot.hist(ax=ax)"
SMALL_CSV = pd.DataFrame({"age": [30, 40, 50, 60], "city": ["Città", "Mare", "Città", "Mare"]}).to_csv(index=False)


def frame_from_schema(schema: list[dict], rows: int = TEST_ROWS, seed: int = 0) -> pd.DataFrame:
    """Random rows with the columns, types and possible values of a schema."""
    rng = np.random.default_rng(seed)
    columns = {}
    for column in schema:
        if "values" in column:
            columns[column["name"]] = rng.choice(column["values"], size=rows)
        elif column["type"] == "integer":
            columns[column["name"]] = rng.integers(0, 100, size=rows)
        else:
            columns[column["name"]] = rng.uniform(0, 60, size=rows).round(2)
    return pd.DataFrame(columns)


CSV_OF = {name: frame_from_schema(schema).to_csv(index=False) for name, schema in SCHEMAS.items()}


def png_width(png: bytes) -> int:
    """Width in pixels, from the IHDR chunk of a PNG."""
    return struct.unpack(">I", png[16:20])[0]


# --- static check: what is refused -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("import os", "'os'"),
        ("import subprocess", "'subprocess'"),
        ("import numpy.ctypeslib", "'numpy.ctypeslib'"),
        ("from os import system", "'os'"),
        ("from . import data", "relative"),
        ("from pandas import read_csv", "'read_csv'"),
        ("open('/etc/passwd')", "'open'"),
        ("exec('1')", "'exec'"),
        ("eval('1')", "'eval'"),
        ("compile('1', 'x', 'exec')", "'compile'"),
        ("getattr(df, 'to_csv')", "'getattr'"),
        ("setattr(df, 'x', 1)", "'setattr'"),
        ("globals()", "'globals'"),
        ("vars()", "'vars'"),
        ("breakpoint()", "'breakpoint'"),
        ("input()", "'input'"),
        ("df.__class__", "'__class__'"),
        ("__builtins__", "'__builtins__'"),
        ("_hidden = 1", "'_hidden'"),
        ("df.to_csv('copy.csv')", "'to_csv'"),
        ("df.to_pickle('copy.pkl')", "'to_pickle'"),
        ("fig.savefig('chart.png')", "'savefig'"),
        ("pd.read_csv('data.csv')", "'read_csv'"),
        ("pd.read_pickle('data.pkl')", "'read_pickle'"),
        ("np.load('data.npy')", "'load'"),
        ("plt.imread('image.png')", "'imread'"),
        ("df.eval('a + b')", "'eval'"),
        ("df.query('age > 1')", "'query'"),
        ("pd.io.common.os.system('ls')", "'system'"),
        ("(x for x in []).gi_frame", "'gi_frame'"),
        ("frame.f_back.f_globals", "'f_globals'"),
    ],
)
def test_unsafe_code_is_refused_with_its_rule(code, reason):
    with pytest.raises(AnalystError) as error:
        check_code(code)
    assert error.value.code == UNSAFE_CODE
    assert reason in error.value.detail
    assert error.value.message and reason not in error.value.message


def test_syntax_errors_are_runtime_errors_with_a_detail():
    """The model can fix a syntax error like any failure, so it goes back to it with the reason."""
    with pytest.raises(AnalystError) as error:
        check_code("fig, ax = plt.subplots(\n")
    assert error.value.code == RUNTIME_ERROR
    assert error.value.detail.startswith("SyntaxError")


# --- static check: what is allowed -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib",
        "import matplotlib.pyplot as plt",
        "from matplotlib.ticker import PercentFormatter",
        "df['to_csv'].mean()",
        "pd.to_datetime(df['date'])",
        "df.to_numpy()",
        "bands = pd.cut(df['age'], bins=[0, 30, 60, 100])",
        "means = {name: group.mean() for name, group in df.groupby('city')}",
        "df.sort_values('age').plot(x='age', y='bmi', kind='scatter', ax=ax)",
        "print(f\"Correlation: {df['age'].corr(df['bmi']):.2f}\")",
    ],
)
def test_ordinary_analysis_code_is_allowed(code):
    check_code(code)


@pytest.mark.parametrize("response", CHART_ANSWERS, ids=[r["source"] for r in CHART_ANSWERS])
def test_real_answers_pass_the_check(response):
    check_code(parse_answer(response["content"], response["finish_reason"]).code)


# --- running: the real answers ---------------------------------------------------------------------------------


@pytest.mark.parametrize("response", CHART_ANSWERS, ids=[r["source"] for r in CHART_ANSWERS])
def test_real_answers_draw_a_chart(response):
    code = parse_answer(response["content"], response["finish_reason"]).code
    result = run_code(code, CSV_OF[response["dataset"]])
    assert result.png.startswith(PNG_SIGNATURE)
    assert result.figures >= 1


# --- running: hand-written cases -------------------------------------------------------------------------------


def test_a_chart_comes_back_as_png():
    result = run_code(CHART, SMALL_CSV)
    assert isinstance(result, ChartResult)
    assert result.png.startswith(PNG_SIGNATURE)
    assert (result.figures, result.summary) == (1, None)                 # no summary table: still a chart


def test_the_data_arrive_whole_through_standard_input():
    code = CHART + "\nsummary = pd.DataFrame({'rows': [len(df)], 'columns': [' '.join(df.columns)], 'first': [df['city'].iloc[0]]})"
    assert run_code(code, SMALL_CSV).summary == {
        "columns": ["rows", "columns", "first"], "rows": [[4, "age city", "Città"]], "truncated": False}


def test_numpy_pandas_and_pyplot_are_ready():
    code = "fig, ax = plt.subplots()\nax.plot(np.arange(3), pd.Series([1, 2, 3]))"
    assert run_code(code, SMALL_CSV).figures == 1


def test_printed_text_is_dropped():
    """The numbers go in ``summary``; what the code prints is not shown (nor can it break the child's answer)."""
    result = run_code(CHART + "\nprint('Mean age: 45.00')\nprint('{\"png\": null}')", SMALL_CSV)
    assert result.figures == 1 and result.summary is None


# --- the summary table -------------------------------------------------------------------------------------------


def summary_of(code: str, csv_text: str = SMALL_CSV) -> dict | None:
    return run_code(CHART + "\n" + code, csv_text).summary


def test_a_table_with_a_plain_index_is_kept_as_it_is():
    assert summary_of("summary = pd.DataFrame({'mean age': [45.0], 'people': [4]})") == {
        "columns": ["mean age", "people"], "rows": [[45.0, 4]], "truncated": False}


def test_the_labels_of_a_grouped_series_become_the_first_column():
    """``groupby(...).mean()``: the categories are the index, and they are part of the answer."""
    assert summary_of("summary = df.groupby('city')['age'].mean().round(2)") == {
        "columns": ["city", "age"], "rows": [["Città", 40.0], ["Mare", 50.0]], "truncated": False}


def test_row_positions_left_by_a_sort_are_dropped():
    """Seen with a real answer: sorting a grouped table leaves an unnamed integer index (positions, not labels)."""
    code = "summary = pd.DataFrame({'level': ['b', 'c', 'a'], 'order': [2, 3, 1]}).sort_values('order')"
    assert summary_of(code) == {"columns": ["level", "order"], "rows": [["a", 1], ["b", 2], ["c", 3]],
                                "truncated": False}


def test_a_named_integer_index_is_a_label():
    code = "summary = df.groupby('age').size().rename('people')"
    assert summary_of(code)["columns"] == ["age", "people"]


def test_unnamed_series_and_index_get_readable_headers():
    table = summary_of("summary = pd.Series([1, 2], index=['a', 'b'])")
    assert table == {"columns": ["", "value"], "rows": [["a", 1], ["b", 2]], "truncated": False}


def test_age_bands_are_written_as_text():
    """``pd.cut`` gives intervals: shown as the familiar "(20, 30]"."""
    code = "summary = df.groupby(pd.cut(df['age'], [20, 40, 60]), observed=False).size().rename('people')"
    assert summary_of(code) == {
        "columns": ["age", "people"], "rows": [["(20, 40]", 2], ["(40, 60]", 2]], "truncated": False}


def test_columns_on_two_levels_are_joined():
    code = "summary = df.pivot_table(index='city', columns=df['age'] > 40, values=['age'], aggfunc='count')"
    assert summary_of(code) == {"columns": ["city", "age / False", "age / True"],
                                "rows": [["Città", 1, 1], ["Mare", 1, 1]], "truncated": False}


def test_repeated_headers_are_made_unique():
    """A table can repeat a header (the index named like a column): the UI needs distinct names."""
    code = "summary = pd.DataFrame({'age': [1, 2]}, index=pd.Index(['x', 'y'], name='age'))"
    assert summary_of(code)["columns"] == ["age", "age (2)"]


def test_missing_values_become_empty_cells():
    assert summary_of("summary = pd.DataFrame({'x': [1.5, float('nan')]})")["rows"] == [[1.5], [None]]


def test_a_long_table_is_cut_and_marked():
    code = f"summary = pd.DataFrame({{'n': range({MAX_SUMMARY_ROWS + 5})}})"
    table = summary_of(code)
    assert len(table["rows"]) == MAX_SUMMARY_ROWS and table["truncated"] is True
    assert table["rows"][0] == [0]


def test_a_wide_table_is_cut_and_marked():
    code = f"summary = pd.DataFrame([list(range({MAX_SUMMARY_COLUMNS + 3}))])"
    table = summary_of(code)
    assert len(table["columns"]) == MAX_SUMMARY_COLUMNS and table["truncated"] is True


@pytest.mark.parametrize("value", ["42", "[1, 2]", "'text'", "pd.DataFrame()", "pd.Series(dtype=float)"])
def test_anything_but_a_non_empty_table_is_no_summary(value):
    """The table is a help: a missing, empty or wrong one never costs the chart."""
    assert summary_of(f"summary = {value}") is None


def test_the_last_figure_is_kept():
    """Two recorded answers opened an empty figure first: the last one is the chart."""
    code = "plt.figure(figsize=(1, 1))\nfig, ax = plt.subplots(figsize=(8, 4))\nax.plot([1, 2])"
    result = run_code(code, SMALL_CSV)
    assert result.figures == 2
    assert png_width(result.png) > 4 * CHART_DPI  # the 8-inch figure, not the 1-inch one


def test_no_figure_is_an_error():
    with pytest.raises(AnalystError) as error:
        run_code("print(df['age'].mean())", SMALL_CSV)
    assert error.value.code == NO_CHART and error.value.detail


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        ("df['glucose']", "KeyError: 'glucose'"),
        ("1 / 0", "ZeroDivisionError: division by zero"),
        ("raise SystemExit(0)", "SystemExit"),
        ("df['city'].astype(float)", "ValueError"),
    ],
)
def test_failures_are_runtime_errors_with_type_and_message(code, detail):
    with pytest.raises(AnalystError) as error:
        run_code(code, SMALL_CSV)
    assert error.value.code == RUNTIME_ERROR
    assert error.value.detail.startswith(detail)
    assert "Traceback" not in error.value.detail and "Traceback" not in error.value.message


def test_endless_code_is_stopped():
    with pytest.raises(AnalystError) as error:
        run_code("while True:\n    pass", SMALL_CSV, timeout=3)
    assert error.value.code == TIMEOUT


def test_unsafe_code_is_never_run(monkeypatch):
    def no_child(*args, **kwargs):
        raise AssertionError("a child process was started")

    monkeypatch.setattr(sandbox.subprocess, "run", no_child)
    with pytest.raises(AnalystError) as error:
        run_code("import os\nos.remove('x')", SMALL_CSV)
    assert error.value.code == UNSAFE_CODE


@pytest.mark.parametrize(
    "runner_code",
    [
        "import sys\nsys.exit(3)\n",
        'import sys\nprint(\'{"png": null, "summary": null, "figures": 1}\')\nsys.exit(1)\n',
    ],
    ids=["no-answer", "answer-then-failure"],
)
def test_a_failed_child_is_a_runtime_error(monkeypatch, tmp_path, runner_code):
    """A child that dies (crash, memory) is a failure, even if it wrote something that looks like an answer."""
    runner = tmp_path / "crash.py"
    runner.write_text(runner_code)
    monkeypatch.setattr(sandbox, "RUNNER_PATH", runner)
    with pytest.raises(AnalystError) as error:
        run_code(CHART, SMALL_CSV)
    assert error.value.code == RUNTIME_ERROR


# --- the second layer: the child itself ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        ("open('secret.txt')", "NameError"),
        ("eval('1')", "NameError"),
        ("import os", "ImportError"),
        ("import subprocess", "ImportError"),
    ],
)
def test_without_the_static_check_the_child_still_refuses(code, detail):
    """execute skips check_code: the child has no open/eval and imports only the allowed modules."""
    with pytest.raises(AnalystError) as error:
        execute(code, SMALL_CSV)
    assert error.value.code == RUNTIME_ERROR
    assert error.value.detail.startswith(detail)


def test_the_child_works_in_an_empty_folder():
    """A path the static check would refuse (``os``) shows what the child sees: an empty folder, no data file.

    It also shows the declared limit: once past the static check, pandas leads to ``os``.
    """
    code = CHART + "\nsummary = pd.DataFrame({'files': [len(pd.io.common.os.listdir('.'))]})"
    assert execute(code, SMALL_CSV).summary["rows"] == [[0]]


def test_the_child_environment_has_no_secret(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    environment = child_environment()
    assert "sk-test-not-a-real-key" not in environment.values()
    assert set(environment) <= {*PASSED_ENV, "MPLBACKEND", "MPLCONFIGDIR"}
    assert environment["MPLBACKEND"] == "Agg"


def test_the_child_really_gets_that_environment(monkeypatch):
    """End to end: the secret set in the API process does not reach the code (LC_CTYPE: set by Python itself)."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test-not-a-real-key")
    code = CHART + "\nsummary = pd.DataFrame({'name': sorted(pd.io.common.os.environ)})"
    names = {name for (name,) in execute(code, SMALL_CSV).summary["rows"]}
    assert "LLM_API_KEY" not in names
    assert names <= {*PASSED_ENV, "MPLBACKEND", "MPLCONFIGDIR", "LC_CTYPE"}


# --- columns used by the chart (the Ask step's reliability notes) --------------------------------------------------

NAMES = ["age", "sex", "marital-status", "income"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("df['age'].plot.hist(ax=ax)", ["age"]),
        ("df.groupby(['sex', 'income'])['age'].mean()", ["age", "sex", "income"]),
        ("df.age.plot.hist(ax=ax)", ["age"]),                                         # attribute access
        ("df['marital-status'].value_counts()", ["marital-status"]),
        ("ax.set_title('Age by sex')\nsummary = df['bmi'].describe()", []),           # not a column name
        ("for name in df.columns:\n    print(name)", []),                            # not named: no note (limit)
    ],
    ids=["subscript", "list", "attribute", "dash", "labels", "loop"],
)
def test_columns_used_are_the_names_in_the_code(code, expected):
    assert columns_used(code, NAMES) == expected          # in the order of the names (the contract's)
