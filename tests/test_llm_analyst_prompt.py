"""Tests for the prompt sent to the model and for reading its answer.

The 40 answers in ``fixtures/llm_responses.json`` are real answers of the development model (Groq,
``openai/gpt-oss-120b``) to test questions asked during development: code written for synthetic data, no real data.
"""

import ast
import json
from pathlib import Path

import pytest

from sds.llm_analyst.errors import INCOMPLETE_ANSWER, AnalystError
from sds.llm_analyst.llm_config import MAX_ERROR_CHARS
from sds.llm_analyst.prompt import Answer, fix_request, parse_answer, system_prompt

RESPONSES = json.loads((Path(__file__).parent / "fixtures" / "llm_responses.json").read_text())

SCHEMA = [
    {"name": "age", "type": "integer"},
    {"name": "bmi", "type": "float"},
    {"name": "region", "type": "category", "values": ["Città", "Mare", "Other"]},
    {"name": "code", "type": "integer category", "values": [1, 2, 3]},
]


# --- system prompt ---------------------------------------------------------------------------------------------


def test_each_column_is_one_json_line():
    prompt = system_prompt(SCHEMA)
    for column in SCHEMA:
        assert json.dumps(column, ensure_ascii=False) in prompt.splitlines()


def test_accented_values_stay_readable():
    assert "Città" in system_prompt(SCHEMA) and "\\u00e0" not in system_prompt(SCHEMA)


def test_no_placeholder_is_left():
    assert "{schema}" not in system_prompt(SCHEMA)


def test_braces_in_values_are_harmless():
    schema = [{"name": "note", "type": "category", "values": ["{x}", "a}b"]}]
    assert '"{x}"' in system_prompt(schema)


def test_prompt_keeps_the_measured_rules():
    """The prompt measured on the development model: its key rules are all there."""
    prompt = system_prompt(SCHEMA)
    for rule in (
        "SYNTHETIC",
        "ONE Python code block",
        "fig, ax = plt.subplots(",
        "natural order",
        "called `summary`",
        "do not print",
        "Do not read or write files",
        "never state numbers or results",
        "the language of the question",
        "single records",
        "a person who cannot program",
    ):
        assert rule in prompt
    for word in ("doctor", "patient"):            # the app does not know the job of its user
        assert word not in prompt.lower()


def test_prompt_sends_no_row_or_statistic():
    """Only what llm_schema gives: the prompt is the template plus the schema lines."""
    with_columns = system_prompt(SCHEMA)
    without_columns = system_prompt([])
    added = set(with_columns.splitlines()) - set(without_columns.splitlines())
    assert added == {json.dumps(column, ensure_ascii=False) for column in SCHEMA}


# --- fix request -----------------------------------------------------------------------------------------------


def test_fix_request_carries_the_error():
    message = fix_request("KeyError: 'glucose_level'")
    assert "KeyError: 'glucose_level'" in message
    assert message.startswith("Your code could not be run.")
    assert "same format" in message


def test_fix_request_cuts_long_errors():
    error = "x" * (MAX_ERROR_CHARS * 3)
    message = fix_request(error)
    assert "x" * MAX_ERROR_CHARS in message and "x" * (MAX_ERROR_CHARS + 1) not in message


# --- reading the answer: the real answers ----------------------------------------------------------------------


@pytest.mark.parametrize("response", RESPONSES, ids=[r["source"] for r in RESPONSES])
def test_real_answers_are_read(response):
    answer = parse_answer(response["content"], response["finish_reason"])
    if response["kind"] == "chart":
        ast.parse(answer.code)
        assert "plt" in answer.code or "ax" in answer.code
        assert "```" not in answer.code and "```" not in answer.text
    else:
        assert answer.code is None
        assert answer.text == response["content"].strip()


def test_real_answers_have_an_explanation_except_one():
    """One correction came back with code only: the explanation is optional."""
    answers = [parse_answer(r["content"], r["finish_reason"]) for r in RESPONSES if r["kind"] == "chart"]
    assert sum(not answer.text for answer in answers) == 1


# --- reading the answer: hand-written cases --------------------------------------------------------------------

CODE = "fig, ax = plt.subplots()\ndf['age'].plot.hist(ax=ax)"


@pytest.mark.parametrize("tag", ["python", "py", "Python", ""])
def test_code_block_tags(tag):
    answer = parse_answer(f"```{tag}\n{CODE}\n```\nRead the bars.", "stop")
    assert answer == Answer(code=CODE, text="Read the bars.")


def test_text_before_and_after_the_block_is_the_explanation():
    answer = parse_answer(f"Here is the chart.\n```python\n{CODE}\n```\nTaller bars mean more people.", "stop")
    assert answer.code == CODE
    assert answer.text == "Here is the chart.\n\nTaller bars mean more people."


def test_only_the_first_block_is_code_and_no_block_reaches_the_text():
    content = f"```python\n{CODE}\n```\nRead the bars.\n```python\nprint('extra')\n```"
    answer = parse_answer(content, "stop")
    assert answer == Answer(code=CODE, text="Read the bars.")


def test_code_without_explanation():
    assert parse_answer(f"```python\n{CODE}\n```", "stop") == Answer(code=CODE, text="")


def test_text_only_is_a_normal_answer():
    text = "The data have no cholesterol column. Try asking about glucose."
    assert parse_answer(f"  {text}\n", "stop") == Answer(code=None, text=text)


def test_windows_line_endings():
    answer = parse_answer(f"```python\r\n{CODE}\r\n```\r\nRead the bars.", "stop")
    assert answer.code.replace("\r\n", "\n") == CODE and answer.text == "Read the bars."


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [
        (f"```python\n{CODE}\n```\nRead the", "length"),
        ("", "length"),
        ("", "stop"),
        ("   \n", "stop"),
        (None, "stop"),
        (f"```python\n{CODE}\n```", "content_filter"),
        (f"```python\n{CODE}\n", "stop"),
        (f"```python\n{CODE}\n```\nRead the bars.\n```", "stop"),
    ],
    ids=["cut", "empty-cut", "empty", "blank", "none", "filtered", "unclosed", "odd-fences"],
)
def test_incomplete_answers_are_an_error(content, finish_reason):
    with pytest.raises(AnalystError) as error:
        parse_answer(content, finish_reason)
    assert error.value.code == INCOMPLETE_ANSWER
    assert error.value.message and "Traceback" not in error.value.message
