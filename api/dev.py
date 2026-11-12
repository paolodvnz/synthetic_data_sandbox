"""Developer-mode endpoints (``/dev/*``): mounted only when ``SDS_DEV_MODE`` is on.

They show more of how the engine works (logic in ``api.dev_views``) and load the sample datasets; they never
weaken a protection (the privacy gate stays blocking) and never return a row of the real data.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from api.api_config import STATUS_NOT_FOUND
from api.dataset import dataset_view, load_dataset
from api.dev_views import (
    contract_view,
    decoding_check_view,
    hyperparameters_view,
    training_view,
    validation_details_view,
)
from api.errors import ApiError
from sds.samples import SAMPLE_NAMES, data_dir, missing_samples, sample_path

router = APIRouter(prefix="/dev", tags=["developer"])


@router.get("/samples")
def list_samples() -> list[dict]:
    """The sample datasets and whether each one has been downloaded."""
    missing = set(missing_samples(SAMPLE_NAMES))
    return [{"name": name, "available": name not in missing} for name in SAMPLE_NAMES]


@router.post("/samples/{name}")
async def load_sample(name: str, request: Request) -> dict:
    """Load a sample dataset as if the user had uploaded it.

    Raises:
        ApiError: 404 ``unknown_sample`` or ``samples_missing`` (with the command to download it).
    """
    if name not in SAMPLE_NAMES:
        raise ApiError(STATUS_NOT_FOUND, "unknown_sample", f"There is no sample dataset called '{name}'.")
    path = sample_path(name)
    if not path.is_file():
        raise ApiError(STATUS_NOT_FOUND, "samples_missing",
                       f"The sample '{name}' is not in {data_dir()}: run sds-download-samples to get it.")
    session = request.app.state.session
    await run_in_threadpool(load_dataset, session, path.read_bytes(), path.name)
    return await run_in_threadpool(dataset_view, session)


@router.get("/contract")
def contract(request: Request) -> dict:
    """The full contract and the schema the LLM would receive."""
    return contract_view(request.app.state.session)


@router.get("/decoding-check")
def decoding_check(request: Request) -> dict:
    """Share of train values that survive encode → decode, and random vectors decoded."""
    return decoding_check_view(request.app.state.session)


@router.get("/hyperparameters")
def hyperparameters(request: Request) -> dict:
    """The training plan the next job will use, with the rules it comes from."""
    return hyperparameters_view(request.app.state.session, request.app.state.train_options)


@router.get("/training")
def training(request: Request) -> dict:
    """Training history (also while it runs) and how it ended."""
    return training_view(request.app.state.session)


@router.get("/validation")
def validation(request: Request) -> dict:
    """The full fidelity and privacy reports."""
    return validation_details_view(request.app.state.session)
