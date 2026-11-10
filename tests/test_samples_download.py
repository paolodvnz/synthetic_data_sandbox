"""Offline tests for sds.samples.download (no network: the fetcher is injected)."""

import io
import zipfile

import pytest

from sds.samples import download as dd


def make_zip(members: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive with the given members."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def fetcher_returning(payload: bytes):
    """Return a fake fetcher that serves ``payload`` and counts its calls."""

    def fetch(url: str) -> bytes:
        fetch.calls += 1
        return payload

    fetch.calls = 0
    return fetch


ROWS = b"1,2\n3,4\n"


def test_add_header_prepends_row():
    assert dd.add_header(ROWS, ("a", "b")) == b"a,b\n1,2\n3,4\n"


def test_add_header_none_leaves_content_untouched():
    assert dd.add_header(ROWS, None) == ROWS


def test_extract_member_from_nested_zip():
    inner = make_zip({"data.csv": ROWS})
    outer = make_zip({"inner.zip": inner, "README": b"x"})
    assert dd.extract_member(outer, ("inner.zip", "data.csv")) == ROWS


def test_extract_member_empty_path_returns_input():
    assert dd.extract_member(ROWS, ()) == ROWS


def test_extract_member_missing_lists_available_members():
    archive = make_zip({"other.csv": ROWS})
    with pytest.raises(dd.DownloadError, match=r"other\.csv"):
        dd.extract_member(archive, ("data.csv",))


def test_extract_member_rejects_oversized_member():
    archive = make_zip({"data.csv": ROWS})
    with pytest.raises(dd.DownloadError, match="bytes"):
        dd.extract_member(archive, ("data.csv",), max_bytes=len(ROWS) - 1)


def test_extract_member_rejects_non_zip():
    with pytest.raises(dd.DownloadError, match="cannot open archive"):
        dd.extract_member(b"not a zip", ("data.csv",))


def test_download_dataset_writes_csv_with_header(tmp_path):
    spec = dd.DatasetSpec(name="toy", url="u", member_path=("d.csv",), header=("a", "b"))
    path = dd.download_dataset(spec, tmp_path, fetch=fetcher_returning(make_zip({"d.csv": ROWS})))
    assert path == tmp_path / "toy.csv"
    assert path.read_bytes() == b"a,b\n" + ROWS


def test_download_dataset_checksum_mismatch_writes_nothing(tmp_path):
    spec = dd.DatasetSpec(name="toy", url="u", sha256="0" * 64)
    with pytest.raises(dd.DownloadError, match="checksum mismatch"):
        dd.download_dataset(spec, tmp_path, fetch=fetcher_returning(ROWS))
    assert list(tmp_path.iterdir()) == []


def test_download_dataset_skips_verified_file(tmp_path):
    spec = dd.DatasetSpec(name="toy", url="u", sha256=dd.sha256_of(ROWS))
    (tmp_path / "toy.csv").write_bytes(ROWS)
    fetch = fetcher_returning(ROWS)
    dd.download_dataset(spec, tmp_path, fetch=fetch)
    assert fetch.calls == 0


def test_download_dataset_replaces_corrupted_file(tmp_path):
    spec = dd.DatasetSpec(name="toy", url="u", sha256=dd.sha256_of(ROWS))
    (tmp_path / "toy.csv").write_bytes(b"corrupted")
    fetch = fetcher_returning(ROWS)
    dd.download_dataset(spec, tmp_path, fetch=fetch)
    assert fetch.calls == 1
    assert (tmp_path / "toy.csv").read_bytes() == ROWS


def test_force_downloads_again(tmp_path):
    spec = dd.DatasetSpec(name="toy", url="u", sha256=dd.sha256_of(ROWS))
    (tmp_path / "toy.csv").write_bytes(ROWS)
    fetch = fetcher_returning(ROWS)
    dd.download_dataset(spec, tmp_path, force=True, fetch=fetch)
    assert fetch.calls == 1


def test_manifest_names_are_unique_and_headers_have_expected_width():
    names = [spec.name for spec in dd.DATASETS]
    assert len(names) == len(set(names))
    assert len(dd.ADULT_COLUMNS) == 15
    assert len(dd.BREAST_CANCER_COLUMNS) == 32
    assert len(dd.PIMA_COLUMNS) == 9
