"""A fake API for the UI tests: it stands in for ``ui.api_client``, so ``AppTest`` runs the page with no server."""

from __future__ import annotations

import copy

from ui.api_client import ApiRefused

PAGE_FUNCTIONS = ("health", "get_session", "reset", "upload", "get_dataset", "set_choices", "list_samples",
                  "load_sample", "dev_contract", "dev_decoding_check", "start_training", "get_status",
                  "stop_training", "dev_hyperparameters", "dev_training", "get_validation", "download_csv",
                  "dev_validation", "shutdown")


def job(state: str = "running", *, step: int = 10, phase: str = "training", stop_requested: bool = False,
        error: dict | None = None) -> dict:
    """A job view as ``/status`` returns it."""
    return {"job_id": "abc", "state": state, "phase": phase, "step": step, "max_steps": 100,
            "train_loss": 1.0, "validation_loss": 1.1, "elapsed_seconds": 102.0, "stop_requested": stop_requested,
            "error": error}


def evaluation(step: int) -> dict:
    """One evaluation of ``/dev/training``'s history."""
    losses = {"total": 3.0 / step, "numeric": 1.0 / step, "categorical": 2.0 / step, "kl": 0.1}
    return {"step": step, "kl_weight": min(1.0, step / 20), "train": losses,
            "validation": losses | {"total": 4.0 / step, "numeric": 1.5 / step},
            "active_dims": 2}


def column(name: str, role: str, *, kept_role: str | None = None, fixed: str | None = "it contains text",
           final: str | None = "same", examples=("a", "b", "c")) -> dict:
    """A column of the dataset view (``final="same"``: kept with its role)."""
    return {"name": name, "role": role, "reason": f"reason of {name}", "kept_role": kept_role or role,
            "fixed_reason": fixed, "null_fraction": 0.0, "examples": list(examples),
            "final_role": role if final == "same" else final, "changed_by_user": False}


def dataset(*, ready: bool = True, case_merges: bool = False) -> dict:
    """A dataset view like ``/upload`` returns: an identifier left out, a code-like integer column, notes."""
    return {
        "file": {"name": "patients.csv", "separator": ";", "encoding": "latin-1", "n_rows": 1000},
        "columns": [
            column("id", "id", kept_role="numeric", final=None, fixed="it has 1000 different values"),
            column("age", "numeric", fixed="it has 70 different values", examples=("39", "50", "38")),
            column("visits", "numeric", fixed=None, examples=("0", "3", "1")),
            column("sex", "categorical"),
        ],
        "choices": {"excluded": [], "included": [], "as_categorical": [], "as_numeric": [], "unify_case": True},
        "case_merges": ([{"code": "case_unified", "message": "'male' and 'Male' are treated as the same value.",
                          "column": "sex"}] if case_merges else []),
        "issues": {
            "file": [{"code": "suspected_id", "message": "Column 'id' looks like an identifier and is excluded.",
                      "column": "id"}],
            "choices": [],
            "rows": ([{"code": "rows_removed", "message": "12 rows with missing values were removed.",
                       "column": None},
                      {"code": "few_rows", "message": "Only 790 rows are used for learning.", "column": None}]
                     if ready else []),
            "encoding": ([{"code": "levels_merged", "message": "'Other' is rare in 'sex'.", "column": "sex"}]
                         if ready else []),
        },
        "rows": {"read": 1000, "removed": 12, "train": 790, "holdout": 198} if ready else None,
        "ready": ready,
        "problem": None if ready else {"code": "too_few_rows", "message": "Too few rows to learn from."},
    }


FAILURE = {"code": "too_close", "message": "The synthetic rows are much closer to the real records than new "
           "patients are.", "column": None}
ADVICE = "To fix it, go back to step 1 (Upload) and untick the columns that identify a single person."
COLUMNS = ["age", "sex", "visits"]


def validation(*, passed: bool = True, warnings: bool = True) -> dict:
    """A ``/validation`` view: three columns, one numeric histogram and one categorical."""
    return {
        "gate": {"passed": passed, "failures": [] if passed else [FAILURE], "advice": None if passed else ADVICE},
        "warnings": ([{"source": "privacy", "code": "near_copies_unchecked", "message": "Many real people share "
                       "the same values: near copies cannot be checked.", "column": None},
                      {"source": "fidelity", "code": "distributions_differ", "message": "Some columns are spread "
                       "differently (most: age).", "column": "age"}] if warnings else []),
        "associations": {"columns": COLUMNS,
                         "real": [[1.0, 0.3, 0.5], [0.3, 1.0, 0.1], [0.5, 0.1, 1.0]],
                         "synthetic": [[1.0, 0.25, 0.45], [0.25, 1.0, 0.2], [0.45, 0.2, 1.0]]},
        "histograms": [{"name": "age", "role": "numeric", "labels": [18.0, 42.5, 67.0, 91.5],
                        "real": [0.4, 0.4, 0.2], "synthetic": [0.35, 0.45, 0.2]},
                       {"name": "sex", "role": "categorical", "labels": ["F", "M"],
                        "real": [0.52, 0.48], "synthetic": [0.5, 0.5]}],
        "preview": ({"columns": COLUMNS, "rows": [[40, "F", 2], [63, "M", 0]]} if passed else None),
        "n_rows": 790,
        "download_name": "patients_synthetic.csv",
    }


def frame(matrix: list[list[float]]) -> dict:
    """A DataFrame as ``to_jsonable`` writes it."""
    return {"columns": COLUMNS, "index": COLUMNS, "values": matrix}


def validation_details(*, passed: bool = True) -> dict:
    """A ``/dev/validation`` view: the full reports."""
    data = validation(passed=passed)
    fidelity = {
        "n_compared": 198,
        "columns": [{"name": "age", "role": "numeric", "distance": 0.03, "baseline": 0.02, "spread": 0.004,
                     "z": 2.5, "excess": 0.01},
                    {"name": "sex", "role": "categorical", "distance": 0.02, "baseline": 0.04, "spread": 0.01,
                     "z": -2.0, "excess": -0.02}],
        "numeric": {"distance": 0.03, "baseline": 0.02, "spread": 0.004, "z": 2.5, "excess": 0.01},
        "categorical": {"distance": 0.02, "baseline": 0.04, "spread": 0.01, "z": -2.0, "excess": -0.02},
        "association": {"distance": 0.05, "baseline": 0.046, "spread": 0.002, "z": 2.0, "excess": 0.004},
        "association_diff_max": 0.1,
        "association_real": frame(data["associations"]["real"]),
        "association_synthetic": frame(data["associations"]["synthetic"]),
        "histograms": data["histograms"], "unseen_pairs_synthetic": 0.01, "unseen_pairs_real": 0.005,
        "warnings": [{"code": "distributions_differ", "message": "Some columns are spread differently.",
                      "column": "age"}],
    }
    privacy = {"passed": passed, "n_compared": 198, "synthetic_percentile": 0.03, "holdout_percentile": 0.025,
               "synthetic_exact_share": 0.0, "holdout_exact_share": 0.0, "exact_share_limit": 0.01,
               "train_closer_share": 0.81, "expected_train_closer_share": 0.8,
               "failures": [] if passed else [FAILURE], "warnings": []}
    return {"fidelity": fidelity, "privacy": privacy}


class FakeApi:
    """The API's answers for the page, and a record of what the page sent."""

    def __init__(self, view: dict, *, dev_mode: bool = False, data: dict | None = None, closable: bool = False):
        self.view, self.dev_mode, self.data = view, dev_mode, data
        self.resets, self.uploads, self.sent_choices, self.samples_loaded = 0, [], [], []
        self.closable, self.shutdowns, self.refuse_shutdown = closable, 0, None
        self.refuse_upload: ApiRefused | None = None
        self.dataset_reads = 0
        self.passed, self.downloads = True, 0

    def health(self) -> dict:
        return {"status": "ok", "dev_mode": self.dev_mode, "max_upload_mb": 200, "closable": self.closable}

    def shutdown(self) -> dict:
        if self.refuse_shutdown:
            raise self.refuse_shutdown
        self.shutdowns += 1
        return {"closing": True}

    def get_session(self) -> dict:
        return self.view

    def reset(self) -> dict:
        self.resets += 1
        self.view, self.data = {"stage": "empty", "job": None}, None
        return {"stage": "empty"}

    def loaded(self, data: dict) -> dict:
        self.data = data
        self.view = {"stage": "prepared" if data["ready"] else "inspected", "job": None}
        return copy.deepcopy(data)

    def upload(self, content: bytes, name: str) -> dict:
        self.uploads.append((name, content))
        if self.view["stage"] != "empty":
            raise ApiRefused(409, "dataset_loaded", "A dataset is already loaded: press Start over first.")
        if self.refuse_upload is not None:
            self.view, self.data = {"stage": "empty", "job": None}, None
            raise self.refuse_upload
        return self.loaded(dataset())

    def get_dataset(self) -> dict:
        self.dataset_reads += 1
        if self.data is None:
            raise ApiRefused(409, "no_dataset", "Upload a dataset first.")
        return copy.deepcopy(self.data)

    def set_choices(self, choices: dict) -> dict:
        self.sent_choices.append(choices)
        data = copy.deepcopy(self.data)
        data["choices"] = choices
        for col in data["columns"]:
            if col["name"] in choices["as_categorical"]:
                col["final_role"] = "categorical"
            if col["name"] in choices["excluded"]:
                col["final_role"] = None
        return self.loaded(data)

    def list_samples(self) -> list[dict]:
        return [{"name": "adult", "available": True}, {"name": "pima", "available": False}]

    def load_sample(self, name: str) -> dict:
        self.samples_loaded.append(name)
        return self.loaded(dataset())

    def start_training(self) -> dict:
        self.trainings = getattr(self, "trainings", 0) + 1
        self.view = {"stage": "prepared", "job": job("running", step=0)}
        return self.view["job"]

    def get_status(self, job_id: str) -> dict:
        if self.view["job"] is None or self.view["job"]["job_id"] != job_id:
            raise ApiRefused(404, "unknown_job", "This training is not known: start a new one.")
        return copy.deepcopy(self.view["job"])

    def stop_training(self) -> dict:
        self.stops = getattr(self, "stops", 0) + 1
        self.view["job"] = self.view["job"] | {"state": "stopped", "stop_requested": True}
        return copy.deepcopy(self.view["job"])

    def dev_hyperparameters(self) -> dict:
        return {"inputs": {"train_rows": 790, "training_rows": 711, "validation_rows": 79, "width": 13},
                "architecture": {"hidden_dims": [64], "latent_dim": 8, "rules": ["hidden = 2 x W within [64, 512]"]},
                "budget": {"epochs": 1000, "steps_per_epoch": 3, "max_steps": 3000, "limits": [200, 20000],
                           "eval_every": 150, "evaluations": 20},
                "kl": {"warmup_steps": 300, "beta": 1.0},
                "optimizer": {"learning_rate": 0.001, "weight_decay": 1e-05, "batch_size": 256},
                "early_stopping": {"patience": 10, "from_step": 300}, "loss": "mixed_sigma", "device": "cpu"}

    def dev_training(self) -> dict:
        if self.view["job"] is None:
            raise ApiRefused(409, "no_training", "No training has been started yet.")
        current = self.view["job"]
        done = current["state"] == "done"
        return {"state": current["state"], "phase": current["phase"], "max_steps": current["max_steps"],
                "elapsed_seconds": current["elapsed_seconds"], "history": [evaluation(10), evaluation(20)],
                "result": {"best_step": 20, "stopped_early": current["step"] < current["max_steps"],
                           "cancelled": False} if done else None}

    def get_validation(self) -> dict:
        if self.view["stage"] != "validated":
            raise ApiRefused(409, "not_validated", "Check the synthetic data first (Validate).")
        return validation(passed=self.passed)

    def download_csv(self) -> bytes:
        self.downloads += 1
        if not self.passed:
            raise ApiRefused(403, "privacy_gate", FAILURE["message"])
        return b"age;sex;visits\n40;F;2\n"

    def dev_validation(self) -> dict:
        return validation_details(passed=self.passed)

    def dev_contract(self) -> dict:
        return {"contract": {"version": 3, "width": 7}, "llm_schema": [{"name": "age", "type": "numeric"}]}

    def dev_decoding_check(self) -> dict:
        return {"n_checked": 790, "identical_share": {"age": 1.0, "sex": 0.98},
                "random_rows": {"columns": ["age", "sex"], "rows": [[40, "F"], [61, "M"], [25, "F"]]}}


def install(monkeypatch, api: FakeApi) -> FakeApi:
    """Replace the functions of ``ui.api_client`` the page calls with those of ``api``."""
    import ui.api_client
    for name in PAGE_FUNCTIONS:
        monkeypatch.setattr(ui.api_client, name, getattr(api, name))
    return api
