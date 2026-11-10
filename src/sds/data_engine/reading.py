"""File level of the reading pipeline: raw bytes -> table of strings.

Nothing is interpreted here: every cell stays text. Only file-wide problems are handled
(encoding, separator, malformed lines, empty or binary files, header sanity).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sds.data_engine.thresholds_config import (
    BINARY_PROBE_BYTES,
    CANDIDATE_SEPARATORS,
    MAX_BAD_LINES_FRACTION,
    SNIFF_BYTES,
)
from sds.data_engine.types import DataEngineError, Issue


@dataclass
class RawTable:
    """A CSV split into cells, all kept as strings."""

    frame: pd.DataFrame
    separator: str
    encoding: str
    issues: list[Issue] = field(default_factory=list)


def load_bytes(source: str | Path | bytes) -> bytes:
    """Return the content of ``source``, which is either a path or the bytes themselves."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return Path(source).read_bytes()


def decode_text(data: bytes) -> tuple[str, str, list[Issue]]:
    """Decode as UTF-8 (BOM removed); fall back to Latin-1, which never fails, with a warning."""
    try:
        return data.decode("utf-8-sig"), "utf-8", []
    except UnicodeDecodeError:
        issue = Issue(
            "encoding_fallback",
            "The file is not UTF-8; it was read as Latin-1. Check that accented letters look right.",
        )
        return data.decode("latin-1"), "latin-1", [issue]


def sniff_separator(text: str) -> str:
    """Detect the field separator on a sample made of whole lines."""
    sample = text[:SNIFF_BYTES]
    if len(text) > SNIFF_BYTES and "\n" in sample:
        sample = sample[: sample.rfind("\n")]
    try:
        return csv.Sniffer().sniff(sample, delimiters=CANDIDATE_SEPARATORS).delimiter
    except csv.Error as error:
        raise DataEngineError(
            "unknown_separator",
            "We could not detect columns in this file. Is it a table saved as CSV?",
        ) from error


def unique_header(names: list[str]) -> tuple[list[str], list[Issue]]:
    """Strip names, name empty ones and make duplicates unique (``x``, ``x_2``...)."""
    issues: list[Issue] = []
    seen: dict[str, int] = {}
    result = []
    for position, raw in enumerate(names, start=1):
        name = raw.strip() or f"column_{position}"
        if name in seen:
            seen[name] += 1
            new_name = f"{name}_{seen[name]}"
            issues.append(Issue("column_renamed", f"Duplicate column name '{name}' renamed to '{new_name}'.", new_name))
            name = new_name
        else:
            seen[name] = 1
        result.append(name)
    return result, issues


def looks_headerless(names: list[str]) -> bool:
    """True when every column name is a number: the first data row was probably taken as header."""

    def is_number(value: str) -> bool:
        try:
            float(value.replace(",", "."))
        except ValueError:
            return False
        return True

    return all(is_number(name) for name in names)


def read_table(source: str | Path | bytes) -> RawTable:
    """Read a CSV into a DataFrame of strings, handling file-level problems.

    Args:
        source: Path of the CSV file, or its raw bytes.

    Returns:
        The table with every cell as text, plus the detected separator, encoding and warnings.

    Raises:
        DataEngineError: ``empty_file``, ``not_csv``, ``unknown_separator`` or ``too_many_bad_lines``.
    """
    data = load_bytes(source)
    if not data.strip():
        raise DataEngineError("empty_file", "The file is empty.")
    if b"\x00" in data[:BINARY_PROBE_BYTES]:
        raise DataEngineError(
            "not_csv", "This does not look like a CSV file. If it is an Excel file, save it as CSV first."
        )

    text, encoding, issues = decode_text(data)
    separator = sniff_separator(text)
    # csv.reader instead of pd.read_csv: pandas silently pads rows that have too few fields.
    rows = [row for row in csv.reader(io.StringIO(text), delimiter=separator) if row]
    header, body = rows[0], rows[1:]
    if not body:
        raise DataEngineError("empty_file", "The file has column names but no data rows.")

    good = [row for row in body if len(row) == len(header)]
    n_bad = len(body) - len(good)
    if n_bad / len(body) > MAX_BAD_LINES_FRACTION:
        raise DataEngineError(
            "too_many_bad_lines",
            f"{n_bad} of {len(body)} rows do not have {len(header)} values. The file seems damaged or misread.",
        )
    if n_bad:
        issues.append(Issue("bad_lines_skipped", f"{n_bad} malformed rows were skipped."))

    names, rename_issues = unique_header(header)
    issues += rename_issues
    if looks_headerless(names):
        issues.append(Issue(
            "missing_header",
            "All column names are numbers: the file may have no header row, so the first data row is used as names.",
        ))

    frame = pd.DataFrame(good, columns=names, dtype=str)
    return RawTable(frame=frame, separator=separator, encoding=encoding, issues=issues)
