"""Turn backend objects (dataclasses, numpy, pandas) into plain JSON values.

The reports and the contract are dataclasses full of numpy numbers, NaN and DataFrames: FastAPI's encoder
would reject NaN and does not know how to lay out a DataFrame. One explicit function keeps the format
predictable for the UI and for the tests.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """A copy of ``value`` made only of ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and ``None``.

    Rules: dataclasses -> dict of their fields; ``Enum`` -> its value; numpy scalars and arrays -> Python
    numbers and lists; NaN and infinities -> ``None`` (JSON has no NaN); ``DataFrame`` ->
    ``{"columns", "index", "values"}``; ``Series`` -> list; dates and ``Timestamp`` -> ISO text (``NaT`` -> ``None``);
    ``Path`` -> text; sets -> sorted list; tuples -> list; dict keys -> text.

    Args:
        value: Any combination of the types above.

    Returns:
        The converted value, accepted by ``json.dumps(..., allow_nan=False)``.

    Raises:
        TypeError: For a type with no rule (better than sending something the UI cannot read).
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if value is pd.NaT:
        return None
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, pd.DataFrame):
        return {"columns": to_jsonable(list(value.columns)), "index": to_jsonable(list(value.index)),
                "values": to_jsonable(value.to_numpy())}
    if isinstance(value, pd.Series):
        return to_jsonable(value.tolist())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"cannot convert {type(value).__name__} to JSON: {value!r:.80}")
