"""Where the sample datasets live: one rule shared by the download command, the tests and the API.

The folder is ``$SDS_DATA_DIR`` when that variable is set, else ``data/`` at the project root (the
folder holding ``pyproject.toml``). The project root is found from this file's position, which is
reliable because the app is installed in editable mode (``pip install -e .``), the only supported way.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "SDS_DATA_DIR"
DATA_DIR_NAME = "data"
# src/sds/samples/location.py -> parents: samples, sds, src, <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Folder of the sample datasets (it may not exist yet: nothing has been downloaded).

    Returns:
        ``$SDS_DATA_DIR`` if set and not empty, else ``<project root>/data``.
    """
    if env := os.environ.get(DATA_DIR_ENV):
        return Path(env).expanduser()
    return PROJECT_ROOT / DATA_DIR_NAME


def sample_path(name: str, folder: Path | None = None) -> Path:
    """Path of the CSV of one sample dataset.

    Args:
        name: Dataset name, as in the download manifest (e.g. ``"adult"``).
        folder: Folder to look in; ``data_dir()`` when ``None``.

    Returns:
        ``<folder>/<name>.csv`` (whether or not the file exists).
    """
    return (folder if folder is not None else data_dir()) / f"{name}.csv"


def missing_samples(names: list[str] | tuple[str, ...], folder: Path | None = None) -> list[str]:
    """Names of the sample datasets whose CSV is not on disk.

    Args:
        names: Dataset names to check.
        folder: Folder to look in; ``data_dir()`` when ``None``.

    Returns:
        The missing names, in the given order (empty when all are present).
    """
    return [name for name in names if not sample_path(name, folder).is_file()]
