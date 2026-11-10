"""Download the sample datasets used to test and demo the Synthetic Data Sandbox.

The files are kept as raw as possible: the only normalisation is adding a header row to files
that ship without one, because a real user's CSV would have column names. Separators, null
tokens ("?", "unknown"), leading spaces and fake zeros are left untouched on purpose: they are
exactly what the Data Engine must cope with.

Usage (the command is installed by ``pip install -e ".[dev]"``):
    sds-download-samples                        # all datasets into the sample folder (see location.py)
    sds-download-samples --dest ~/datasets      # custom destination
    sds-download-samples --only adult pima --force
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sds.samples.location import data_dir

logger = logging.getLogger("sds-download-samples")

USER_AGENT = "sds-download-samples/0.1"
TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1 << 16
# Upper bounds that protect against truncated/hostile downloads and zip bombs.
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_MEMBER_BYTES = 200 * 1024 * 1024

ADULT_COLUMNS = (
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
)
_WDBC_FEATURES = (
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension",
)
BREAST_CANCER_COLUMNS = ("id", "diagnosis") + tuple(
    f"{feature}_{stat}" for stat in ("mean", "se", "worst") for feature in _WDBC_FEATURES
)
PIMA_COLUMNS = (
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome",
)


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded, extracted or verified."""


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset comes from and how to turn it into a CSV file.

    Attributes:
        name: Output file stem (the file is saved as ``<name>.csv``).
        url: Source URL of a zip archive or of the raw file itself.
        member_path: Path through (possibly nested) zip archives to the wanted file;
            empty when ``url`` points directly to the file.
        header: Column names to prepend when the raw file has no header row, else ``None``.
        sha256: Expected SHA-256 of the output file, or ``None`` while bootstrapping.
        license: License of the source, for the record.
    """

    name: str
    url: str
    member_path: tuple[str, ...] = ()
    header: tuple[str, ...] | None = None
    sha256: str | None = None
    license: str = "unknown"


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="adult",
        url="https://archive.ics.uci.edu/static/public/2/adult.zip",
        member_path=("adult.data",),
        header=ADULT_COLUMNS,
        sha256="d57ce8b6a8e774c5e3a0f4b45c797c2b61a32400fa5961032db6ddfe8845dfe6",
        license="CC BY 4.0",
    ),
    DatasetSpec(
        name="breast_cancer",
        url="https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        member_path=("wdbc.data",),
        header=BREAST_CANCER_COLUMNS,
        sha256="1beebcb6fe65e01dc93a1aafd43bee201723dc5bbd2479e5f7ed8c634e788202",
        license="CC BY 4.0",
    ),
    DatasetSpec(
        name="bank_marketing",
        url="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        member_path=("bank.zip", "bank-full.csv"),
        sha256="d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d",
        license="CC BY 4.0",
    ),
    DatasetSpec(
        name="cdc_diabetes",
        url="https://archive.ics.uci.edu/static/public/891/data.csv",
        sha256="9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964",
        license="not stated by UCI (Kaggle source: CC0)",
    ),
    DatasetSpec(
        name="diabetes_130",
        url="https://archive.ics.uci.edu/static/public/296/"
        "diabetes+130-us+hospitals+for+years+1999-2008.zip",
        member_path=("diabetic_data.csv",),
        sha256="0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
        license="CC BY 4.0",
    ),
    DatasetSpec(
        name="pima",
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv",
        header=PIMA_COLUMNS,
        sha256="3c7023e0935097ac63e03c6f398928b6dfce30974d5591d3af09a9661876d9fe",
        license="unclear (Kaggle mirror: CC0)",
    ),
)

Fetcher = Callable[[str], bytes]


def fetch_bytes(url: str, max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    """Download ``url`` into memory, refusing payloads larger than ``max_bytes``.

    Args:
        url: HTTP(S) URL to download.
        max_bytes: Maximum accepted size of the response body.

    Returns:
        The response body.

    Raises:
        DownloadError: On network errors or when the body exceeds ``max_bytes``.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    buffer = io.BytesIO()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            while chunk := response.read(CHUNK_BYTES):
                buffer.write(chunk)
                if buffer.tell() > max_bytes:
                    raise DownloadError(f"{url}: response larger than {max_bytes} bytes")
    except OSError as error:  # URLError and timeouts are OSError subclasses
        raise DownloadError(f"{url}: {error}") from error
    return buffer.getvalue()


def extract_member(
    archive: bytes, member_path: tuple[str, ...], max_bytes: int = MAX_MEMBER_BYTES
) -> bytes:
    """Read one file out of (possibly nested) zip archives, entirely in memory.

    Only the named member is read and nothing is written to disk, so a hostile archive
    cannot place files outside the destination (path traversal).

    Args:
        archive: Bytes of the outer zip archive.
        member_path: Names of the members to follow, outermost first.
        max_bytes: Maximum declared uncompressed size of each member (zip-bomb guard).

    Returns:
        The bytes of the innermost member.

    Raises:
        DownloadError: If a member is missing, too large, or the data is not a zip archive.
    """
    data = archive
    for member in member_path:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                if member not in names:
                    raise DownloadError(f"member {member!r} not found; archive contains {names}")
                info = zf.getinfo(member)
                if info.file_size > max_bytes:
                    raise DownloadError(f"member {member!r} is {info.file_size} bytes (> {max_bytes})")
                data = zf.read(member)
        except zipfile.BadZipFile as error:
            raise DownloadError(f"cannot open archive to read {member!r}: {error}") from error
    return data


def add_header(data: bytes, header: tuple[str, ...] | None) -> bytes:
    """Prepend a comma-separated header row when ``header`` is given.

    Args:
        data: Raw file content.
        header: Column names, or ``None`` to leave the content untouched.

    Returns:
        The content with the header row on top.
    """
    if header is None:
        return data
    return (",".join(header) + "\n").encode("utf-8") + data


def sha256_of(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def write_atomically(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temporary file, so no half-written file is left behind.

    A plain ``.part`` sibling is used instead of ``tempfile.NamedTemporaryFile``, whose files are
    always created with mode 0600: the final CSV gets the usual permissions from the umask.
    """
    tmp_path = path.with_name(path.name + ".part")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


def download_dataset(
    spec: DatasetSpec, dest: Path, force: bool = False, fetch: Fetcher = fetch_bytes
) -> Path:
    """Download, extract, normalise and verify one dataset into ``dest/<name>.csv``.

    Args:
        spec: Description of the dataset.
        dest: Destination directory (created if missing).
        force: Re-download even when a verified file is already present.
        fetch: Function that downloads a URL; injectable for offline tests.

    Returns:
        Path of the CSV file.

    Raises:
        DownloadError: On download/extraction errors or checksum mismatch. In that case
            no file is written.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{spec.name}.csv"

    if out_path.exists() and not force:
        if spec.sha256 is None or sha256_of(out_path.read_bytes()) == spec.sha256:
            logger.info("%s: already present, skipped", spec.name)
            return out_path
        logger.warning("%s: existing file has a wrong checksum, downloading again", spec.name)

    raw = fetch(spec.url)
    content = add_header(extract_member(raw, spec.member_path), spec.header)
    digest = sha256_of(content)

    if spec.sha256 is None:
        logger.warning("%s: no expected checksum in the manifest; got sha256=%s", spec.name, digest)
    elif digest != spec.sha256:
        raise DownloadError(
            f"{spec.name}: checksum mismatch (expected {spec.sha256}, got {digest}); "
            "the source may have changed"
        )

    write_atomically(out_path, content)
    logger.info("%s: saved %s (%d bytes)", spec.name, out_path, len(content))
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dest", type=Path, default=None, help="destination directory (default: the sample folder)"
    )
    parser.add_argument("--force", action="store_true", help="download again even if present")
    parser.add_argument(
        "--only", nargs="+", choices=[spec.name for spec in DATASETS], help="subset of datasets"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Download the selected datasets; return a process exit code (0 = all succeeded)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    dest = args.dest if args.dest is not None else data_dir()
    selected = [spec for spec in DATASETS if args.only is None or spec.name in args.only]

    failures = 0
    for spec in selected:
        try:
            download_dataset(spec, dest, force=args.force)
        except DownloadError as error:
            logger.error("%s", error)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
