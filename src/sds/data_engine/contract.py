"""Schema contract: the readable description of what was learned.

Built after fitting the encoder. Three readers:
- the decoder, which reads only the contract (never the scikit-learn internals);
- the UI, which shows the report (columns used and left out, rows, warnings);
- the LLM, which gets ``llm_schema()``: names, types and values of the synthetic data, no statistics.
The contract also keeps, for every pair of categorical columns, the One-Hot slot pairs seen together in
the train rows: decoding samples each group on its own and repairs the rows that pair values no real
row pairs (e.g. "education" HS-grad with "education-num" 10). They describe the real data, so they stay
local like the rest of the contract and never reach ``llm_schema()``.
Stored as JSON, never pickle: readable, and loading it cannot run code. The fitted encoder itself
stays in memory for the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from sds.data_engine.choices import FinalSchema
from sds.data_engine.encoding import FeatureGroup, TabularEncoder
from sds.data_engine.preparation import PreparedData
from sds.data_engine.thresholds_config import DECIMALS_TOLERANCE, MAX_DECIMALS
from sds.data_engine.types import USABLE_ROLES, CsvInspection, DataEngineError, Issue

CONTRACT_VERSION = 4      # 2: seen pairs of categorical slots; 3: decimals of the numeric columns; 4: decimal codes
OTHER_LABEL = "Other"   # how merged rare text levels appear in the synthetic data


@dataclass(frozen=True)
class ContractColumn:
    """One encoded column: its positions in the vector and what is needed to decode it.

    Attributes:
        name: Column name.
        role: ``numeric``, ``date`` or ``categorical``.
        start: First position in the vector.
        stop: Position after the last one.
        min: Train minimum (numeric; date: in days from ``origin``, so 0).
        max: Train maximum (numeric; date: in days from ``origin``).
        integer: Whole numbers (numeric: round when decoding; categorical: integer codes).
        decimal_codes: Categorical codes that are numbers with decimals (0.25, 0.5, ...): decoded as floats.
        decimals: Decimals of the train values (numeric, not whole numbers): the synthetic numbers are rounded to
            them; ``None``: no rounding (computed values with more than ``MAX_DECIMALS``).
        origin: First train date, ISO format (date columns only).
        levels: Kept levels in slot order, original type (categorical only).
        has_other: The group ends with the "Other" slot (categorical only).
        merged: ``((level, train rows), ...)`` of the levels merged into "Other".
    """

    name: str
    role: str
    start: int
    stop: int
    min: float | None = None
    max: float | None = None
    integer: bool = False
    decimal_codes: bool = False
    decimals: int | None = None
    origin: str | None = None
    levels: tuple = ()
    has_other: bool = False
    merged: tuple[tuple, ...] = ()

    def to_dict(self) -> dict:
        """JSON-ready dict with only the fields of this role."""
        data = {"name": self.name, "role": self.role, "start": self.start, "stop": self.stop}
        if self.role == "categorical":
            data |= {"levels": list(self.levels), "integer": self.integer, "decimal_codes": self.decimal_codes,
                     "has_other": self.has_other, "merged": [list(pair) for pair in self.merged]}
        else:
            data |= {"min": self.min, "max": self.max}
            data |= {"origin": self.origin} if self.role == "date" else {"integer": self.integer,
                                                                          "decimals": self.decimals}
        return data

    @property
    def numeric_codes(self) -> bool:
        """Categorical codes that are numbers (whole or not): written as numbers, "Other" picks a merged value."""
        return self.role == "categorical" and (self.integer or self.decimal_codes)

    @classmethod
    def from_dict(cls, data: dict) -> ContractColumn:
        """Inverse of ``to_dict`` (JSON lists back to tuples)."""
        values = dict(data)
        values["levels"] = tuple(values.get("levels", ()))
        values["merged"] = tuple(tuple(pair) for pair in values.get("merged", ()))
        return cls(**values)


@dataclass(frozen=True)
class DroppedColumn:
    """A column of the file that is not in the synthetic data, and why (plain language)."""

    name: str
    reason: str


@dataclass(frozen=True)
class SeenPairs:
    """Slot pairs of two categorical columns found together in at least one train row.

    Attributes:
        first: Name of the column that comes first in the contract.
        second: Name of the other column.
        slots: ``((slot of first, slot of second), ...)``, sorted; slots index the One-Hot group
            (the "Other" slot included).
    """

    first: str
    second: str
    slots: tuple[tuple[int, int], ...]


@dataclass
class SchemaContract:
    """What was learned from the file: positions, decoding parameters and the report.

    Attributes:
        version: Format version (``CONTRACT_VERSION``).
        width: Length of the encoded vector.
        columns: Encoded columns, in vector order.
        dropped: Columns of the file left out, in file order.
        rows: Row counts: ``read``, ``removed``, ``train``, ``holdout``.
        issues: Warnings of every stage (reading, choices, preparation, encoding).
        seen_pairs: For every pair of categorical columns, the slot pairs seen in train. A pair of
            columns without an entry is not checked when decoding.
    """

    version: int
    width: int
    columns: list[ContractColumn]
    dropped: list[DroppedColumn]
    rows: dict[str, int]
    issues: list[Issue] = field(default_factory=list)
    seen_pairs: list[SeenPairs] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-ready dict."""
        return {
            "version": self.version,
            "width": self.width,
            "columns": [c.to_dict() for c in self.columns],
            "dropped": [vars(d) for d in self.dropped],
            "rows": dict(self.rows),
            "issues": [vars(i) for i in self.issues],
            "seen_pairs": [{"first": p.first, "second": p.second, "slots": [list(s) for s in p.slots]}
                           for p in self.seen_pairs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SchemaContract:
        """Rebuild a contract from ``to_dict`` output.

        Raises:
            DataEngineError: ``unsupported_contract`` when the version is not ``CONTRACT_VERSION``.
        """
        if data.get("version") != CONTRACT_VERSION:
            raise DataEngineError(
                "unsupported_contract",
                f"This description of the data has version {data.get('version')!r}; "
                f"only version {CONTRACT_VERSION} can be read.",
            )
        return cls(
            version=data["version"],
            width=data["width"],
            columns=[ContractColumn.from_dict(c) for c in data["columns"]],
            dropped=[DroppedColumn(**d) for d in data["dropped"]],
            rows=dict(data["rows"]),
            issues=[Issue(**i) for i in data["issues"]],
            seen_pairs=[SeenPairs(p["first"], p["second"], tuple(tuple(s) for s in p["slots"]))
                        for p in data["seen_pairs"]],
        )

    def llm_schema(self) -> list[dict]:
        """What the LLM may know: names, types and possible values of the synthetic data.

        No statistics (min, max, row counts), no dropped columns, no warnings: they describe the
        real data. Merged numeric codes are listed because they can appear in the synthetic CSV.
        """
        schema = []
        for c in self.columns:
            if c.role == "numeric":
                schema.append({"name": c.name, "type": "integer" if c.integer else "float"})
            elif c.role == "date":
                schema.append({"name": c.name, "type": "date"})
            elif c.numeric_codes:
                values = sorted([*c.levels, *(level for level, _ in c.merged)])
                kind = "integer category" if c.integer else "number category"
                schema.append({"name": c.name, "type": kind, "values": values})
            else:
                values = list(c.levels)
                if c.has_other and OTHER_LABEL not in values:
                    values.append(OTHER_LABEL)
                schema.append({"name": c.name, "type": "category", "values": values})
        return schema


def decimal_places(values: pd.Series, max_decimals: int = MAX_DECIMALS) -> int | None:
    """Fewest decimals that write every non-missing value exactly, or ``None`` beyond ``max_decimals``.

    Values parsed from text such as ``"33.6"`` are the closest floats, so rounding them gives them back exactly.
    """
    present = values.dropna().to_numpy(dtype=np.float64)
    if not len(present):
        return None
    for decimals in range(max_decimals + 1):
        if np.allclose(np.round(present, decimals), present, rtol=DECIMALS_TOLERANCE, atol=0):
            return decimals
    return None


def contract_column(group: FeatureGroup, encoder: TabularEncoder, inspection: CsvInspection,
                    train: pd.DataFrame) -> ContractColumn:
    """Contract entry of one encoder group (``train``: the rows the encoder was fitted on)."""
    base = {"name": group.name, "role": group.role, "start": group.start, "stop": group.stop}
    if group.role == "categorical":
        return ContractColumn(
            **base,
            integer=group.name in encoder.integer_codes,
            decimal_codes=group.name in encoder.decimal_codes,
            levels=group.levels,
            has_other=group.has_other,
            merged=tuple(group.merged.items()),
        )
    low, high = encoder.value_range(group.name)
    if group.role == "date":
        origin = encoder.date_origins[group.name].date().isoformat()
        return ContractColumn(**base, min=low, max=high, origin=origin)
    integer = inspection.columns[group.name].is_integer
    decimals = None if integer else decimal_places(train[group.name])
    return ContractColumn(**base, min=low, max=high, integer=integer, decimals=decimals)


def dropped_columns(inspection: CsvInspection, schema: FinalSchema, encoder: TabularEncoder) -> list[DroppedColumn]:
    """Columns of the file that are not encoded, in file order, with the reason."""
    encoded = {g.name for g in encoder.groups}
    not_learned = {i.column: i.message for i in encoder.issues
                   if i.code in {"constant_in_train", "all_levels_rare"}}
    dropped = []
    for name, info in inspection.columns.items():
        if name in encoded:
            continue
        if name in not_learned:
            reason = not_learned[name]
        elif name in schema.columns:
            reason = "it could not be learned from the training rows"
        elif info.role in USABLE_ROLES:
            reason = "removed by you"
        else:
            reason = info.reason
        dropped.append(DroppedColumn(name, reason))
    return dropped


def seen_pairs(encoder: TabularEncoder, train: pd.DataFrame) -> list[SeenPairs]:
    """Slot pairs found together in the train rows, for every pair of categorical columns."""
    slots = encoder.category_slots(train)
    found = []
    for first, second in combinations(slots, 2):
        width = int(slots[second].max()) + 1
        codes = np.unique(slots[first] * width + slots[second])
        found.append(SeenPairs(first, second, tuple((int(c // width), int(c % width)) for c in codes)))
    return found


def build_contract(inspection: CsvInspection, schema: FinalSchema, prepared: PreparedData,
                   encoder: TabularEncoder) -> SchemaContract:
    """Describe what was learned, from the results of every Data Engine stage.

    Args:
        inspection: Result of ``inspect_csv``.
        schema: Result of ``apply_choices`` on the same inspection.
        prepared: Result of ``prepare``.
        encoder: Result of ``fit_encoder`` on ``prepared.train``.

    Returns:
        The contract, ready for ``to_dict`` / JSON.
    """
    return SchemaContract(
        version=CONTRACT_VERSION,
        width=encoder.width,
        columns=[contract_column(g, encoder, inspection, prepared.train) for g in encoder.groups],
        dropped=dropped_columns(inspection, schema, encoder),
        rows={"read": prepared.rows_read, "removed": prepared.rows_removed,
              "train": len(prepared.train), "holdout": len(prepared.holdout)},
        issues=[*inspection.issues, *schema.issues, *prepared.issues, *encoder.issues],
        seen_pairs=seen_pairs(encoder, prepared.train),       # train only: the holdout must not leak
    )
