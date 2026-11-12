"""Validation results for the user and the download of the synthetic table, behind the privacy gate.

The view carries only aggregates of the real data (association matrices, histogram shares) and, once the gate
has passed, a few synthetic rows. The CSV is built in memory and refused until the gate passes.
"""

from __future__ import annotations

from urllib.parse import quote

from api.api_config import (
    DEFAULT_DOWNLOAD_STEM,
    GATE_ADVICE,
    PREVIEW_ROWS,
    STATUS_FORBIDDEN,
    SYNTHETIC_SUFFIX,
)
from api.errors import ApiError
from api.serialization import to_jsonable
from api.session import Session, Stage


def gate_view(session: Session) -> dict:
    """Outcome of the privacy gate, with what to do in the app when it fails."""
    privacy = session.privacy
    return {"passed": privacy.passed, "failures": to_jsonable(privacy.failures),
            "advice": None if privacy.passed else GATE_ADVICE}


def validation_view(session: Session) -> dict:
    """What the Validate step shows: gate, non-blocking warnings, data for the charts, a synthetic preview.

    Raises:
        ApiError: 409 with the code of the first missing stage (``not_validated`` after a training).
    """
    with session.lock:
        session.require(Stage.VALIDATED)
        fidelity, privacy, synthetic = session.fidelity, session.privacy, session.synthetic
        warnings = ([{"source": "privacy", **to_jsonable(issue)} for issue in privacy.warnings]
                    + [{"source": "fidelity", **to_jsonable(issue)} for issue in fidelity.warnings])
        preview = None
        if privacy.passed:
            head = synthetic.head(PREVIEW_ROWS)
            preview = {"columns": to_jsonable(list(head.columns)), "rows": to_jsonable(head.to_numpy(dtype=object))}
        return {
            "gate": gate_view(session),
            "warnings": warnings,
            "associations": {"columns": to_jsonable(list(fidelity.association_real.columns)),
                             "real": to_jsonable(fidelity.association_real.to_numpy()),
                             "synthetic": to_jsonable(fidelity.association_synthetic.to_numpy())},
            "histograms": to_jsonable(fidelity.histograms),
            "preview": preview,
            "n_rows": len(synthetic),
            "download_name": download_name(session.source_name),
        }


def require_gate(session: Session) -> None:
    """Check that the synthetic table may leave the app.

    Raises:
        ApiError: 409 with the code of the first missing stage; 403 ``privacy_gate`` when the gate failed.
    """
    session.require(Stage.VALIDATED)
    if not session.privacy.passed:
        reasons = " ".join(issue.message for issue in session.privacy.failures)
        raise ApiError(STATUS_FORBIDDEN, "privacy_gate", f"{reasons} {GATE_ADVICE}".strip())


def synthetic_csv(session: Session) -> bytes:
    """The synthetic table as CSV bytes: separator of the uploaded file, UTF-8, no index, empty missing cells.

    Raises:
        ApiError: See ``require_gate``.
    """
    with session.lock:
        require_gate(session)
        return session.synthetic.to_csv(index=False, sep=session.inspection.separator).encode("utf-8")


def download_name(source_name: str) -> str:
    """``<uploaded name without extension>_synthetic.csv``."""
    stem = source_name.rsplit(".", 1)[0] if "." in source_name else source_name
    return f"{stem or DEFAULT_DOWNLOAD_STEM}{SYNTHETIC_SUFFIX}"


def content_disposition(file_name: str) -> str:
    """Attachment header with an ASCII fallback and the exact UTF-8 name (RFC 6266)."""
    fallback = "".join(ch for ch in file_name if " " <= ch <= "~" and ch not in '"\\')
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(file_name)}"
