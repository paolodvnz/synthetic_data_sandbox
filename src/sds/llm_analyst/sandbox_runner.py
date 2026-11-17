"""Child process that runs generated code on the synthetic data and returns the chart.

Started by ``sds.llm_analyst.sandbox`` as ``python -I sandbox_runner.py``, in an empty temporary folder and with an
environment that holds no secret. It imports nothing from ``sds``.

Standard input: one JSON object, UTF-8, with ``code``, ``csv`` (the synthetic data, comma-separated),
``allowed_modules``, ``forbidden_builtins``, ``dpi``, ``max_summary_rows`` and ``max_summary_columns``.
Standard output: one JSON object, either ``{"png", "summary", "figures"}`` (``png`` in base64, ``None`` without a
figure; ``summary`` as described in ``summary_table``) or ``{"error_type", "error_message"}``. What the code prints
is captured and dropped: standard output carries only the answer.
"""

import base64
import builtins
import contextlib
import io
import json
import sys


def restricted_builtins(allowed_modules: set[str], forbidden: set[str]) -> dict:
    """The built-ins of the generated code: no forbidden function, imports only from the allowed modules."""

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in allowed_modules:
            raise ImportError(f"import of {name!r} is not allowed")
        return builtins.__import__(name, globals, locals, fromlist, level)

    names = {name: getattr(builtins, name) for name in dir(builtins)
             if not name.startswith("_") and name not in forbidden}
    return {**names, "__import__": guarded_import}


def header(name) -> str:
    """A column or index name as text: levels joined with " / ", no name as ""."""
    if isinstance(name, tuple):
        return " / ".join(str(part) for part in name if part is not None and str(part) != "")
    return "" if name is None else str(name)


def unique_names(names: list[str]) -> list[str]:
    """The same names, a repeated one followed by " (2)", " (3)"...: the UI needs distinct headers."""
    seen: dict[str, int] = {}
    result = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return result


def is_row_positions(index) -> bool:
    """An unnamed integer index numbers the rows (left by a sort or a filter): it is not a label to show."""
    import pandas as pd

    return index.nlevels == 1 and index.name is None and pd.api.types.is_integer_dtype(index.dtype)


def summary_table(value, max_rows: int, max_columns: int) -> dict | None:
    """The ``summary`` variable of the code as a small table of plain values, or ``None``.

    A Series or a DataFrame becomes ``{"columns", "rows", "truncated"}``: an index with labels (categories, age
    bands) becomes the first columns, an index of row positions is dropped, cells are numbers, text or ``None``, and a table over the limits is cut and
    marked. Anything else, or an empty table, is no summary: the chart is the answer anyway.
    """
    import pandas as pd

    if isinstance(value, pd.Series):
        value = value.to_frame(name="value" if value.name is None else value.name)
    if not isinstance(value, pd.DataFrame) or value.empty:
        return None
    frame = value.copy()
    frame.columns = [header(name) for name in frame.columns]
    if is_row_positions(frame.index):
        frame = frame.reset_index(drop=True)
    else:
        frame.index.names = [header(name) for name in frame.index.names]
        frame = frame.reset_index(allow_duplicates=True)
    truncated = frame.shape[0] > max_rows or frame.shape[1] > max_columns
    frame = frame.iloc[:max_rows, :max_columns]
    frame.columns = unique_names([str(name) for name in frame.columns])
    rows = json.loads(frame.to_json(orient="values", date_format="iso", default_handler=str))
    return {"columns": list(frame.columns), "rows": rows, "truncated": truncated}


def run(request: dict) -> dict:
    """Run the code of a request and describe the outcome."""
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace = {
        "__builtins__": restricted_builtins(set(request["allowed_modules"]), set(request["forbidden_builtins"])),
        "__name__": "__answer__",
        "df": pd.read_csv(io.StringIO(request["csv"])),
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(request["code"], "<answer>", "exec"), namespace)
    except BaseException as error:  # SystemExit and KeyboardInterrupt too: the code must not end the child
        return {"error_type": type(error).__name__, "error_message": str(error)}
    numbers = plt.get_fignums()
    png = None
    if numbers:
        buffer = io.BytesIO()
        plt.figure(numbers[-1]).savefig(buffer, format="png", dpi=request["dpi"], bbox_inches="tight")
        png = base64.b64encode(buffer.getvalue()).decode("ascii")
    try:
        summary = summary_table(namespace.get("summary"), request["max_summary_rows"], request["max_summary_columns"])
    except Exception:  # an odd object in the table: the chart is still the answer
        summary = None
    return {"png": png, "summary": summary, "figures": len(numbers)}


def main() -> None:
    """Read the request from standard input and write the outcome to standard output (bytes, UTF-8)."""
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    outcome = run(request)
    sys.stdout.buffer.write(json.dumps(outcome).encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
