"""Specification tests for the upload endpoints: raw-body upload, size limit, choices, sample datasets."""

import json
import tempfile

import pytest
from fastapi.testclient import TestClient

import api.main
from api.main import create_app
from api_csv import make_csv
from sds.config import MAX_UPLOAD_MB
from sds.samples import DATA_DIR_ENV, sample_path

CSV_HEADERS = {"Content-Type": "text/csv"}
SMALL_LIMIT = 10_000                      # bytes: above make_csv() (~2.6 KB), far below 200 MB
MULTIPART_SPOOL_BYTES = 1024 * 1024       # Starlette spools multipart files above this size to disk
BIG_ROWS = 60_000


def client(*, dev_mode: bool = False) -> TestClient:
    return TestClient(create_app(dev_mode=dev_mode), raise_server_exceptions=False)


def upload(c: TestClient, data: bytes = None, name: str = "patients.csv"):
    return c.post("/upload", params={"name": name}, content=make_csv() if data is None else data,
                  headers=CSV_HEADERS)


def strict_json(response) -> dict:
    """Parse rejecting NaN and infinities, which are not JSON."""
    def refuse(constant):
        raise ValueError(f"not JSON: {constant}")
    return json.loads(response.text, parse_constant=refuse)


# --- /upload --------------------------------------------------------------------------------

def test_upload_returns_the_view():
    response = upload(client())
    assert response.status_code == 200
    body = strict_json(response)
    assert body["ready"] is True and body["file"]["name"] == "patients.csv"


def test_upload_without_a_name_gets_the_default():
    c = client()
    response = c.post("/upload", content=make_csv(), headers=CSV_HEADERS)
    assert response.status_code == 200 and response.json()["file"]["name"] == "dataset.csv"


def test_empty_file_is_422():
    response = upload(client(), b"")
    assert response.status_code == 422 and response.json()["code"] == "empty_file"


def test_too_large_body_is_413(monkeypatch):
    monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", SMALL_LIMIT)
    c = client()
    assert upload(c).status_code == 200                                  # an earlier dataset...
    response = upload(c, b"a,b\n" + b"1,2\n" * SMALL_LIMIT)
    assert response.status_code == 413 and response.json()["code"] == "file_too_large"
    assert c.get("/dataset").json()["file"]["name"] == "patients.csv"    # ...is kept


def test_chunked_body_over_the_limit_is_413(monkeypatch):
    # No Content-Length (chunked transfer, or a header that lies): the limit is enforced while reading.
    monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", SMALL_LIMIT)
    chunks = iter([b"a,b\n"] + [b"1,2\n" * 100] * (SMALL_LIMIT // 400 + 2))
    response = client().post("/upload", content=chunks, headers=CSV_HEADERS)
    assert response.status_code == 413 and response.json()["code"] == "file_too_large"


def test_declared_length_over_the_limit_is_refused_before_reading(monkeypatch):
    monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", SMALL_LIMIT)
    response = client().post("/upload", content=b"a,b\n1,2\n",
                             headers={**CSV_HEADERS, "Content-Length": str(SMALL_LIMIT + 1)})
    assert response.status_code == 413


def test_limit_in_bytes_follows_max_upload_mb():
    assert api.main.MAX_UPLOAD_BYTES == MAX_UPLOAD_MB * 1024 * 1024


def test_too_large_message_names_the_limit_in_mb():
    response = client().post("/upload", content=b"a,b\n1,2\n",
                             headers={**CSV_HEADERS, "Content-Length": str(api.main.MAX_UPLOAD_BYTES + 1)})
    assert f"{MAX_UPLOAD_MB} MB" in response.json()["message"]


def test_upload_writes_nothing_to_disk(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a temporary file was created")
    for name in ("SpooledTemporaryFile", "NamedTemporaryFile", "TemporaryFile", "mkstemp", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, forbidden)
    big = make_csv(BIG_ROWS)
    assert len(big) > MULTIPART_SPOOL_BYTES                              # multipart would write it to disk
    assert upload(client(), big).status_code == 200


def test_dataset_view_endpoint():
    c = client()
    assert c.get("/dataset").status_code == 409
    upload(c)
    assert c.get("/dataset").json()["file"]["name"] == "patients.csv"


# --- /choices -------------------------------------------------------------------------------

def test_choices_return_the_new_view():
    c = client()
    upload(c)
    response = c.post("/choices", json={"excluded": ["city"]})
    assert response.status_code == 200
    assert response.json()["choices"]["excluded"] == ["city"]


def test_invalid_choices_are_422():
    c = client()
    upload(c)
    response = c.post("/choices", json={"excluded": ["nope"]})
    assert response.status_code == 422 and response.json()["code"] == "invalid_choices"


def test_choices_must_be_an_object():
    c = client()
    upload(c)
    response = c.post("/choices", json=["city"])
    assert response.status_code == 422 and set(response.json()) == {"code", "message"}


def test_choices_without_a_dataset_are_409():
    response = client().post("/choices", json={})
    assert response.status_code == 409 and response.json()["code"] == "no_dataset"


# --- sample datasets (developer mode) -------------------------------------------------------

@pytest.fixture
def samples(monkeypatch, tmp_path):
    """A sample folder holding only pima.csv."""
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    sample_path("pima", tmp_path).write_bytes(make_csv())
    return tmp_path


def test_samples_do_not_exist_without_dev_mode(samples):
    c = client(dev_mode=False)
    assert c.get("/dev/samples").status_code == 404
    assert c.post("/dev/samples/pima").status_code == 404


def test_samples_list_availability(samples):
    listed = client(dev_mode=True).get("/dev/samples").json()
    assert {s["name"]: s["available"] for s in listed}["pima"] is True
    assert {s["name"]: s["available"] for s in listed}["adult"] is False
    assert [s["name"] for s in listed] == ["adult", "breast_cancer", "bank_marketing", "cdc_diabetes",
                                           "diabetes_130", "pima"]


def test_loading_a_sample(samples):
    response = client(dev_mode=True).post("/dev/samples/pima")
    assert response.status_code == 200 and response.json()["file"]["name"] == "pima.csv"


def test_second_upload_and_sample_are_refused_until_reset(samples):
    c = client(dev_mode=True)
    assert upload(c).status_code == 200
    for refused in (upload(c, name="other.csv"), c.post("/dev/samples/pima")):
        assert refused.status_code == 409 and refused.json()["code"] == "dataset_loaded"
    assert c.get("/dataset").json()["file"]["name"] == "patients.csv"
    assert c.post("/reset").status_code == 200
    assert upload(c, name="other.csv").status_code == 200


def test_missing_sample_names_the_command(samples):
    response = client(dev_mode=True).post("/dev/samples/adult")
    assert response.status_code == 404
    assert response.json()["code"] == "samples_missing" and "sds-download-samples" in response.json()["message"]


def test_unknown_sample(samples):
    response = client(dev_mode=True).post("/dev/samples/secret_file")
    assert response.status_code == 404 and response.json()["code"] == "unknown_sample"


# --- real data ------------------------------------------------------------------------------

def test_upload_of_a_real_dataset(samples_dir):
    c = client()
    response = upload(c, (samples_dir / "adult.csv").read_bytes(), "adult.csv")
    body = strict_json(response)
    assert response.status_code == 200 and body["ready"] is True
    assert len(body["columns"]) == 15 and body["rows"]["removed"] == 2399
