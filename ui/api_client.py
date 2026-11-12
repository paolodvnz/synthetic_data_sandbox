"""The only way the UI talks to the backend: HTTP calls to the API (never an import of ``sds``)."""

from __future__ import annotations

import os

import requests

from ui.ui_config import API_URL_ENV, DEFAULT_API_URL, REQUEST_SECONDS, STOP_REQUEST_SECONDS, WORK_REQUEST_SECONDS


class ApiUnavailable(Exception):
    """The API cannot be reached (not started, or stopped), or it answered with something that is not its own."""


class ApiRefused(Exception):
    """The API answered with one of its errors (``{"code", "message"}``): the message is meant for the user."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def api_url() -> str:
    """Base URL of the API: ``$SDS_API_URL`` (set by ``sds-app``) or the default local address."""
    return os.environ.get(API_URL_ENV) or DEFAULT_API_URL


def call(method: str, path: str, *, timeout: float = REQUEST_SECONDS, **kwargs) -> dict:
    """One request to the API, answered in JSON (``kwargs`` go to ``requests.request``).

    Raises:
        ApiUnavailable: When the API does not answer, or answers without its error format.
        ApiRefused: When the API refuses the request with ``{"code", "message"}``.
    """
    return request(method, path, timeout=timeout, **kwargs).json()


def request(method: str, path: str, *, timeout: float = REQUEST_SECONDS, **kwargs) -> requests.Response:
    """One request to the API; the successful response as it is (``call`` reads it as JSON).

    Raises:
        ApiUnavailable, ApiRefused: As ``call``.
    """
    try:
        response = requests.request(method, f"{api_url()}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as error:
        raise ApiUnavailable(str(error)) from error
    if response.ok:
        return response
    try:
        body = response.json()
        raise ApiRefused(response.status_code, body["code"], body["message"])
    except (ValueError, KeyError, TypeError):
        raise ApiUnavailable(f"HTTP {response.status_code} from {path}") from None


def health() -> dict:
    """``GET /health``: whether the API runs, the developer mode and the upload limit."""
    try:
        return call("GET", "/health")
    except ApiRefused as error:                     # /health never refuses: something else answered
        raise ApiUnavailable(error.message) from error


def get_session() -> dict:
    """``GET /session``: the stage reached and the latest training job (or ``None``)."""
    return call("GET", "/session")


def shutdown() -> dict:
    """``POST /shutdown``: close the app (the API ends itself, ``sds-app`` stops the UI)."""
    return call("POST", "/shutdown")


def reset() -> dict:
    """``POST /reset``: forget the dataset, the model and the results (stops a running training, waiting for it)."""
    return call("POST", "/reset", timeout=STOP_REQUEST_SECONDS)


def upload(content: bytes, name: str) -> dict:
    """``POST /upload``: send the CSV as the raw body; returns the dataset view."""
    return call("POST", "/upload", params={"name": name}, data=content, headers={"Content-Type": "text/csv"},
                timeout=WORK_REQUEST_SECONDS)


def get_dataset() -> dict:
    """``GET /dataset``: the dataset view (columns, choices, warnings, rows)."""
    return call("GET", "/dataset")


def set_choices(choices: dict) -> dict:
    """``POST /choices``: apply the user's choices and prepare the data again; returns the dataset view."""
    return call("POST", "/choices", json=choices, timeout=WORK_REQUEST_SECONDS)


def list_samples() -> list[dict]:
    """``GET /dev/samples`` (developer mode): the sample datasets and whether each one is downloaded."""
    return call("GET", "/dev/samples")


def load_sample(name: str) -> dict:
    """``POST /dev/samples/{name}`` (developer mode): load a sample as if uploaded; returns the dataset view."""
    return call("POST", f"/dev/samples/{name}", timeout=WORK_REQUEST_SECONDS)


def dev_contract() -> dict:
    """``GET /dev/contract`` (developer mode): the schema contract and what the LLM will receive."""
    return call("GET", "/dev/contract")


def dev_decoding_check() -> dict:
    """``GET /dev/decoding-check`` (developer mode): values back unchanged and rows decoded from random vectors."""
    return call("GET", "/dev/decoding-check")


def start_training() -> dict:
    """``POST /train``: start train → generate → validate in the background; returns the job view."""
    return call("POST", "/train")


def get_status(job_id: str) -> dict:
    """``GET /status/{job_id}``: progress of the job."""
    return call("GET", f"/status/{job_id}")


def stop_training() -> dict:
    """``POST /train/stop``: stop the job and wait for it (up to the API's stop timeout); returns the job view."""
    return call("POST", "/train/stop", timeout=STOP_REQUEST_SECONDS)


def dev_hyperparameters() -> dict:
    """``GET /dev/hyperparameters`` (developer mode): the training plan, with the sizes it comes from."""
    return call("GET", "/dev/hyperparameters")


def dev_training() -> dict:
    """``GET /dev/training`` (developer mode): the history of the evaluations, also while the job runs."""
    return call("GET", "/dev/training")


def get_validation() -> dict:
    """``GET /validation``: gate, warnings, data of the charts, synthetic preview, download name."""
    return call("GET", "/validation")


def download_csv() -> bytes:
    """``GET /download``: the synthetic CSV (refused with 403 until the privacy gate passes)."""
    return request("GET", "/download", timeout=WORK_REQUEST_SECONDS).content


def dev_validation() -> dict:
    """``GET /dev/validation`` (developer mode): the full fidelity and privacy reports."""
    return call("GET", "/dev/validation")
