"""FastAPI app of the Synthetic Data Sandbox.

Start it with ``uvicorn api.main:app --host 127.0.0.1`` from the project root (or ``sds-app``). Every error
reaches the UI as ``{"code", "message"}``: known errors with their own code, bugs with a fixed message
(the details stay in this process's log, never in a response).
"""

from __future__ import annotations

import logging
import os

from fastapi import BackgroundTasks, Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import dev
from api.api_config import (
    DEV_MODE_ENV,
    INTERNAL_ERROR_MESSAGE,
    MAX_UPLOAD_BYTES,
    NOT_CLOSABLE_MESSAGE,
    STATUS_CONFLICT,
    STATUS_INTERNAL,
    STATUS_TOO_LARGE,
    STATUS_UNPROCESSABLE,
    TRUE_WORDS,
)
from api.dataset import change_choices, dataset_view, load_dataset
from api.errors import ApiError
from api.pipeline import job_view, reset_session, session_view, start_training, stop_training
from api.results import content_disposition, download_name, synthetic_csv, validation_view
from api.session import Session
from sds.config import CLOSABLE_ENV, CLOSED_BY_USER_EXIT_CODE, MAX_UPLOAD_MB
from sds.data_engine import DataEngineError
from sds.vae.layout import VaeError

logger = logging.getLogger("sds.api")


def flag_from_environment(name: str) -> bool:
    """True when the variable ``name`` holds one of ``TRUE_WORDS`` (spaces and case ignored)."""
    return os.environ.get(name, "").strip().lower() in TRUE_WORDS


def dev_mode_from_environment() -> bool:
    """True when ``SDS_DEV_MODE`` holds one of ``TRUE_WORDS`` (spaces and case ignored)."""
    return flag_from_environment(DEV_MODE_ENV)


def close_process(code: int) -> None:
    """End the API process at once with ``code``: the dataset, the model and the results go with it.

    ``os._exit`` and not ``sys.exit``: it runs in a worker thread (where ``sys.exit`` would end only the thread)
    and it must not wait for a running training.
    """
    os._exit(code)


def error_response(status: int, code: str, message: str) -> JSONResponse:
    """The single error format of the API."""
    return JSONResponse(status_code=status, content={"code": code, "message": message})


def add_error_handlers(app: FastAPI) -> None:
    """Translate every exception into ``{"code", "message"}``."""

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        return error_response(error.status, error.code, error.message)

    @app.exception_handler(DataEngineError)
    async def data_engine_error(request: Request, error: DataEngineError) -> JSONResponse:
        return error_response(STATUS_UNPROCESSABLE, error.code, error.message)

    @app.exception_handler(VaeError)
    async def vae_error(request: Request, error: VaeError) -> JSONResponse:
        return error_response(STATUS_UNPROCESSABLE, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        return error_response(STATUS_UNPROCESSABLE, "invalid_request", "The request is not in the expected form.")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        return error_response(error.status_code, f"http_{error.status_code}", str(error.detail))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("unexpected error on %s %s", request.method, request.url.path)
        return error_response(STATUS_INTERNAL, "internal_error", INTERNAL_ERROR_MESSAGE)


def too_large() -> ApiError:
    """413 for an upload over the limit, with the limit in MB."""
    return ApiError(STATUS_TOO_LARGE, "file_too_large",
                    f"The file is larger than {MAX_UPLOAD_MB} MB, the largest the app accepts.")


async def read_body(request: Request, limit: int) -> bytes:
    """The request body, read in chunks into memory and refused as soon as it passes ``limit`` bytes.

    Raises:
        ApiError: 413 ``file_too_large`` (also before reading, when ``Content-Length`` declares too much).
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise too_large()
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise too_large()
    return bytes(body)


def create_app(dev_mode: bool | None = None, closable: bool | None = None) -> FastAPI:
    """Build the app with a fresh, empty session.

    Args:
        dev_mode: Turn the developer mode on or off; ``None`` reads ``SDS_DEV_MODE``.
        closable: Whether ``/shutdown`` may end the process (``sds-app`` then stops the UI); ``None`` reads
            ``SDS_CLOSABLE``, set by ``sds-app``.

    Returns:
        The FastAPI app; its session is ``app.state.session``.
    """
    app = FastAPI(title="Synthetic Data Sandbox API")
    app.state.session = Session()
    app.state.dev_mode = dev_mode_from_environment() if dev_mode is None else dev_mode
    app.state.closable = flag_from_environment(CLOSABLE_ENV) if closable is None else closable
    app.state.train_options = {}          # tests only (shorter trainings): the user has no training levers
    add_error_handlers(app)

    @app.get("/health")
    def health() -> dict:
        """Liveness check; tells the UI the developer mode, the upload limit and whether "Close" is possible."""
        return {"status": "ok", "dev_mode": app.state.dev_mode, "max_upload_mb": MAX_UPLOAD_MB,
                "closable": app.state.closable}

    @app.post("/upload")
    async def upload(request: Request, name: str | None = None) -> dict:
        """Load a CSV sent as the raw request body (``Content-Type: text/csv``); ``name`` is the file name."""
        data = await read_body(request, MAX_UPLOAD_BYTES)
        session = app.state.session
        await run_in_threadpool(load_dataset, session, data, name)
        return await run_in_threadpool(dataset_view, session)

    @app.get("/dataset")
    def dataset() -> dict:
        """The current dataset view (as returned by ``/upload`` and ``/choices``)."""
        return dataset_view(app.state.session)

    @app.post("/choices")
    def choices(data: dict = Body(...)) -> dict:
        """Apply the user's choices (``UserChoices.to_dict`` form) and prepare the dataset again."""
        change_choices(app.state.session, data)
        return dataset_view(app.state.session)

    @app.post("/train")
    def train() -> dict:
        """Start the training job (train → generate → validate) and return its view, ``job_id`` included."""
        session = app.state.session
        job = start_training(session, **app.state.train_options)
        return job_view(session, job.job_id)

    @app.get("/session")
    def session_state() -> dict:
        """The stage and the latest training job: the UI resumes from it after a page refresh."""
        return session_view(app.state.session)

    @app.get("/status/{job_id}")
    def status(job_id: str) -> dict:
        """Progress of the training job."""
        return job_view(app.state.session, job_id)

    @app.post("/train/stop")
    def stop() -> dict:
        """Stop the training job and wait for it: afterwards the choices can change again."""
        session = app.state.session
        job = stop_training(session)
        return job_view(session, job.job_id)

    @app.get("/validation")
    def validation() -> dict:
        """Results of the validation for the Validate step (gate, warnings, charts, synthetic preview)."""
        return validation_view(app.state.session)

    @app.get("/download")
    def download() -> Response:
        """The synthetic CSV, built in memory; 403 ``privacy_gate`` until the gate passes."""
        session = app.state.session
        data = synthetic_csv(session)
        return Response(content=data, media_type="text/csv",       # Starlette adds "; charset=utf-8"
                        headers={"Content-Disposition": content_disposition(download_name(session.source_name))})

    @app.post("/reset")
    def reset() -> dict:
        """Start over: stop the training if one runs and forget the dataset, the model and the results."""
        reset_session(app.state.session)
        return {"stage": app.state.session.stage.name.lower()}

    @app.post("/shutdown")
    def shutdown(background: BackgroundTasks) -> dict:
        """Close the app (the "Close" button): answer, then end the process with ``CLOSED_BY_USER_EXIT_CODE``.

        Only when started by ``sds-app``, which then stops the UI: otherwise 409 ``not_closable``.
        """
        if not app.state.closable:
            raise ApiError(STATUS_CONFLICT, "not_closable", NOT_CLOSABLE_MESSAGE)
        background.add_task(lambda: close_process(CLOSED_BY_USER_EXIT_CODE))     # after the answer is sent
        return {"closing": True}

    if app.state.dev_mode:
        app.include_router(dev.router)

    return app


app = create_app()
