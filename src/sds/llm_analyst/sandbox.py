"""Safe execution of the code written by the model: a static check, then a separate process.

Three layers, none of which is a real sandbox on its own:

1. ``check_code`` reads the code without running it: imports only from ``ALLOWED_MODULES``, no forbidden
   built-in, attribute, or name starting with ``_``;
2. the child (``sandbox_runner.py``) gives the code built-ins without those functions and an import limited to the
   same modules;
3. the child runs in isolated mode, in an empty temporary folder, with an environment that holds no key, under a
   timeout, and receives only the synthetic data, through standard input.

The static check can be bypassed by creative code: the guarantee that matters is the third layer, because the
real data live only in the memory of the API and the key is not in the child's environment.
"""

from __future__ import annotations

import ast
import base64
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from sds.llm_analyst.errors import NO_CHART, RUNTIME_ERROR, TIMEOUT, UNSAFE_CODE, AnalystError
from sds.llm_analyst.llm_config import MAX_ERROR_CHARS
from sds.llm_analyst.sandbox_config import (
    ALLOWED_MODULES,
    CHART_DPI,
    FORBIDDEN_ATTRIBUTE_PREFIXES,
    FORBIDDEN_ATTRIBUTES,
    FORBIDDEN_BUILTINS,
    MAX_SUMMARY_COLUMNS,
    MAX_SUMMARY_ROWS,
    PASSED_ENV,
    RUN_TIMEOUT_SECONDS,
)

RUNNER_PATH = Path(__file__).with_name("sandbox_runner.py")

UNSAFE_MESSAGE = "The generated code tried to do something that is not allowed, so it was not run."
RUNTIME_MESSAGE = "The generated code failed while drawing the chart."
TIMEOUT_MESSAGE = "Drawing the chart took too long and was stopped."
NO_CHART_MESSAGE = "The generated code did not draw a chart."
NO_CHART_DETAIL = "no chart was drawn: create one with fig, ax = plt.subplots() and draw on ax"


@dataclass(frozen=True)
class ChartResult:
    """What a successful run produced.

    Attributes:
        png: The last figure, as PNG bytes.
        summary: The numbers behind the chart (``{"columns", "rows", "truncated"}``, at most ``MAX_SUMMARY_ROWS``
            rows and ``MAX_SUMMARY_COLUMNS`` columns), or ``None`` when the code left no ``summary`` table. It is
            shown to the user and never sent to the provider.
        figures: How many figures the code opened (the last one is the chart).
    """

    png: bytes
    summary: dict | None
    figures: int


def unsafe(detail: str) -> AnalystError:
    """The error of a rule broken by the code."""
    return AnalystError(UNSAFE_CODE, UNSAFE_MESSAGE, detail)


def forbidden_attribute(name: str) -> bool:
    """Whether an attribute or imported name is refused."""
    return name.startswith("_") or name in FORBIDDEN_ATTRIBUTES or name.startswith(FORBIDDEN_ATTRIBUTE_PREFIXES)


def check_code(code: str) -> None:
    """Refuse code that breaks a rule, without running it.

    Args:
        code: The code written by the model.

    Raises:
        AnalystError: ``unsafe_code`` with the broken rule in ``detail``; ``runtime_error`` for a syntax error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise AnalystError(RUNTIME_ERROR, RUNTIME_MESSAGE, f"SyntaxError: {error}"[:MAX_ERROR_CHARS]) from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    raise unsafe(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise unsafe("relative imports are not allowed")
            if node.module not in ALLOWED_MODULES:
                raise unsafe(f"import of {node.module!r} is not allowed")
            for alias in node.names:
                if forbidden_attribute(alias.name):
                    raise unsafe(f"name {alias.name!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in FORBIDDEN_BUILTINS:
                raise unsafe(f"name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if forbidden_attribute(node.attr):
                raise unsafe(f"attribute {node.attr!r} is not allowed")


def columns_used(code: str, names: list[str]) -> list[str]:
    """The columns the code names: text constants (``df['age']``) and attributes (``df.age``) equal to a column name.

    A reading of the code, not of what it does: a loop over ``df.columns`` names no column. Used for the reliability
    notes under the charts, never to decide what may run.

    Args:
        code: The code written by the model.
        names: Column names, in the order of the result.

    Returns:
        The names found, in the order of ``names``; none when the code does not parse.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return [name for name in names if name in found]


@cache
def font_cache_folder() -> str:
    """Matplotlib's own cache folder of this user: the child finds the font list ready (0.75 s instead of 2.7 s).

    It holds only the list of installed fonts, no data. Used as the child's whole configuration folder, so no
    personal matplotlibrc changes the charts.
    """
    import matplotlib

    return matplotlib.get_cachedir()


def child_environment() -> dict[str, str]:
    """The child's environment: what Python needs to start, the backend and the font cache; never a key."""
    passed = {name: os.environ[name] for name in PASSED_ENV if name in os.environ}
    return {**passed, "MPLBACKEND": "Agg", "MPLCONFIGDIR": font_cache_folder()}


def execute(code: str, csv_text: str, *, timeout: float = RUN_TIMEOUT_SECONDS) -> ChartResult:
    """Run code in the child process, **without** the static check (use ``run_code``).

    Args:
        code: The code to run.
        csv_text: The synthetic data, comma-separated, with a header.
        timeout: Seconds before the child is stopped.

    Returns:
        The chart and its summary table.

    Raises:
        AnalystError: ``runtime_error``, ``timeout`` or ``no_chart``.
    """
    request = {
        "code": code,
        "csv": csv_text,
        "allowed_modules": sorted(ALLOWED_MODULES),
        "forbidden_builtins": sorted(FORBIDDEN_BUILTINS),
        "dpi": CHART_DPI,
        "max_summary_rows": MAX_SUMMARY_ROWS,
        "max_summary_columns": MAX_SUMMARY_COLUMNS,
    }
    with tempfile.TemporaryDirectory(prefix="sds-run-") as empty_folder:
        try:
            done = subprocess.run(
                [sys.executable, "-I", str(RUNNER_PATH)],
                input=json.dumps(request).encode("utf-8"),
                capture_output=True,
                cwd=empty_folder,
                env=child_environment(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise AnalystError(TIMEOUT, TIMEOUT_MESSAGE, f"the code ran for more than {timeout:g} seconds") from None
    try:
        outcome = json.loads(done.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        outcome = None
    if done.returncode or not isinstance(outcome, dict):
        raise AnalystError(RUNTIME_ERROR, RUNTIME_MESSAGE, f"the code stopped the process (exit code {done.returncode})")
    if "error_type" in outcome:
        detail = f"{outcome['error_type']}: {outcome['error_message']}"[:MAX_ERROR_CHARS]
        raise AnalystError(RUNTIME_ERROR, RUNTIME_MESSAGE, detail)
    if not outcome["figures"]:
        raise AnalystError(NO_CHART, NO_CHART_MESSAGE, NO_CHART_DETAIL)
    return ChartResult(png=base64.b64decode(outcome["png"]), summary=outcome["summary"], figures=outcome["figures"])


def run_code(code: str, csv_text: str, *, timeout: float = RUN_TIMEOUT_SECONDS) -> ChartResult:
    """Check the code, then run it on the synthetic data in a separate process.

    Args:
        code: The code written by the model.
        csv_text: The synthetic data, comma-separated, with a header.
        timeout: Seconds before the child is stopped.

    Returns:
        The chart and its summary table.

    Raises:
        AnalystError: ``unsafe_code``, ``runtime_error``, ``timeout`` or ``no_chart``.
    """
    check_code(code)
    return execute(code, csv_text, timeout=timeout)
