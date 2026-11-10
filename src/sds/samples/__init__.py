"""Sample datasets for developers and examiners: download them and find them on disk.

They are not shipped with the app: run ``sds-download-samples`` after ``pip install -e ".[dev]"``.
Only the standard library is used, so importing this package is light.
"""

from __future__ import annotations

from sds.samples.download import DATASETS, DatasetSpec, DownloadError, download_dataset
from sds.samples.location import DATA_DIR_ENV, data_dir, missing_samples, sample_path

SAMPLE_NAMES: tuple[str, ...] = tuple(spec.name for spec in DATASETS)

__all__ = [
    "DATASETS",
    "DATA_DIR_ENV",
    "SAMPLE_NAMES",
    "DatasetSpec",
    "DownloadError",
    "data_dir",
    "download_dataset",
    "missing_samples",
    "sample_path",
]
