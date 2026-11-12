"""The single in-memory session: every object of the current dataset, from the upload to the reports.

Nothing is written to disk: dropping the references (``reset``) or ending the process frees everything, even
after a crash. The stage is computed from the objects present, so the session cannot claim a step whose
results are missing.
"""

from __future__ import annotations

import threading
from enum import IntEnum

from api.api_config import STATUS_CONFLICT
from api.errors import ApiError


class Stage(IntEnum):
    """Steps of the guided path, in order."""

    EMPTY = 0
    INSPECTED = 1
    PREPARED = 2
    TRAINED = 3
    GENERATED = 4
    VALIDATED = 5


# Objects each stage adds. The user choices belong to INSPECTED: they stay even when the preparation fails.
STAGE_FIELDS: dict[Stage, tuple[str, ...]] = {
    Stage.INSPECTED: ("source_name", "source_bytes", "inspection", "case_merges", "choices", "schema"),
    Stage.PREPARED: ("prepared", "encoder", "contract"),
    Stage.TRAINED: ("trained",),
    Stage.GENERATED: ("synthetic",),
    Stage.VALIDATED: ("fidelity", "privacy"),
}

# Error returned by ``require`` when this stage is the first one missing.
MISSING_STAGE: dict[Stage, tuple[str, str]] = {
    Stage.INSPECTED: ("no_dataset", "Upload a dataset first."),
    Stage.PREPARED: ("not_prepared", "The dataset is not ready for training: check the columns and the warnings."),
    Stage.TRAINED: ("not_trained", "Train the model first."),
    Stage.GENERATED: ("not_generated", "Generate the synthetic data first."),
    Stage.VALIDATED: ("not_validated", "Check the synthetic data first (Validate)."),
}


class Session:
    """Everything the app knows about the current dataset.

    Attributes:
        lock: Held by every endpoint that reads or changes the session (re-entrant, so helpers can take it too).
        source_name, source_bytes, inspection, case_merges, choices, schema: Upload and user choices
            (``Stage.INSPECTED``). The bytes stay in memory to inspect again when the case option changes;
            ``case_merges`` are the merges of the default inspection, shown whatever the option.
        prepared, encoder, contract: Split, fitted encoder and schema contract (``Stage.PREPARED``).
        trained: The trained VAE (``Stage.TRAINED``).
        synthetic: The synthetic table (``Stage.GENERATED``).
        fidelity, privacy: The validation reports (``Stage.VALIDATED``).
        problem: ``{"code", "message"}`` of the blocking problem that stopped the preparation, else ``None``
            (not a stage: the user fixes it by changing the choices).
        job: The latest training job (``api.pipeline.Job``), running or ended, else ``None``. Not a stage: its
            results become stage objects only when it ends well.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.problem: dict | None = None
        self.job = None
        for names in STAGE_FIELDS.values():
            for name in names:
                setattr(self, name, None)

    @property
    def stage(self) -> Stage:
        """The last stage whose objects are all present, with every earlier stage complete too."""
        reached = Stage.EMPTY
        for stage in sorted(STAGE_FIELDS):
            if any(getattr(self, name) is None for name in STAGE_FIELDS[stage]):
                break
            reached = stage
        return reached

    def clear_from(self, stage: Stage) -> None:
        """Drop the objects of ``stage`` and of every later stage (a change invalidates what follows).

        Raises:
            ValueError: For ``Stage.EMPTY``, which has no objects.
        """
        if stage is Stage.EMPTY:
            raise ValueError("Stage.EMPTY has no objects to clear")
        for each, names in STAGE_FIELDS.items():
            if each >= stage:
                for name in names:
                    setattr(self, name, None)

    def reset(self) -> None:
        """Forget the whole dataset."""
        self.clear_from(Stage.INSPECTED)
        self.problem = None

    def require(self, stage: Stage) -> None:
        """Check that ``stage`` has been reached.

        Raises:
            ApiError: 409 with the code of the first missing stage.
        """
        current = self.stage
        if current >= stage:
            return
        code, message = MISSING_STAGE[Stage(current + 1)]
        raise ApiError(STATUS_CONFLICT, code, message)

    def require_empty(self) -> None:
        """Check that no dataset is loaded: another file comes only after Start over (``/reset``).

        Raises:
            ApiError: 409 ``dataset_loaded``.
        """
        if self.stage is not Stage.EMPTY:
            raise ApiError(STATUS_CONFLICT, "dataset_loaded",
                           "A dataset is already loaded: press Start over to use another file.")

    def require_idle(self) -> None:
        """Check that no training job is running (the dataset and the choices cannot change under it).

        Raises:
            ApiError: 409 ``training_running``.
        """
        if self.job is not None and self.job.active:
            raise ApiError(STATUS_CONFLICT, "training_running",
                           "A training is running: wait for it to finish or press Stop first.")
