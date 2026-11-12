"""Developer panels: how the engine runs, shown only in developer mode with "Show developer panels" on.

One function per step; they show more, they never remove a protection (no real rows ever reach the UI).
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ui import api_client
from ui.api_client import ApiRefused
from ui.train_view import history_frames
from ui.validate_view import fidelity_table, privacy_table


def frame_from_json(table: dict) -> pd.DataFrame:
    """A table of rows sent by the API (``{"columns", "rows"}``)."""
    return pd.DataFrame(table["rows"], columns=table["columns"])


def session_panel(view: dict) -> None:
    """The ``/session`` view the page was built from (stage and latest job)."""
    with st.expander("Developer: session (GET /session)"):
        st.json(view)


def upload_panels(data: dict) -> None:
    """Upload step: how the file was read, the choices sent, the schema contract and the decoding check."""
    with st.expander("Developer: file and choices (GET /dataset)"):
        file = data["file"]
        st.markdown(f"Separator `{file['separator']}` · encoding `{file['encoding']}` · {file['n_rows']:,} rows read")
        st.json(data["choices"])
    if not data["ready"]:
        return
    try:
        contract, check = api_client.dev_contract(), api_client.dev_decoding_check()
    except ApiRefused as error:
        st.error(error.message)
        return
    with st.expander("Developer: schema contract (stays on this computer)"):
        st.json(contract["contract"], expanded=1)
        st.markdown("**Sent to the AI service** (`llm_schema`)")
        st.json(contract["llm_schema"], expanded=False)
    with st.expander(f"Developer: Decoding check ({check['n_checked']:,} training rows)"):
        st.caption("Training rows encoded and decoded back, before any model exists: every value should come back "
                   "unchanged, except rare values (shared slot) and the last digits of very large numbers.")
        changed = {name: share for name, share in check["identical_share"].items() if share < 1}
        if changed:
            st.dataframe(pd.DataFrame([{"Column": name, "Values back unchanged": f"{share:.3%}"}
                                       for name, share in changed.items()]), hide_index=True, width="stretch")
        else:
            st.markdown("All values come back unchanged.")
        st.markdown("**What a model that learned nothing would produce** (random vectors, decoded)")
        st.dataframe(frame_from_json(check["random_rows"]), hide_index=True, width="stretch")


def hyperparameters_panel() -> None:
    """Train step: the training plan (``/dev/hyperparameters``), the same values the job passes to ``train_vae``."""
    try:
        plan = api_client.dev_hyperparameters()
    except ApiRefused as error:
        st.error(error.message)
        return
    with st.expander("Developer: hyperparameters (scaled with the dataset)"):
        st.json(plan, expanded=2)


def training_panel() -> None:
    """Train step: curves of the evaluations (``/dev/training``), live while the job runs, and how it ended."""
    try:
        training = api_client.dev_training()
    except ApiRefused as error:
        st.error(error.message)
        return
    with st.expander("Developer: training curves", expanded=True):
        if not training["history"]:
            st.caption("No evaluation yet.")
        else:
            frames = history_frames(training["history"])
            st.markdown("**Total loss** (train and validation)")
            st.line_chart(frames["total"])
            st.markdown("**Validation loss per part**")
            st.line_chart(frames["components"])
            st.markdown("**KL weight and active latent dimensions**")
            st.line_chart(frames["schedule"])
        if training["result"] is not None:
            st.json(training["result"])


def validation_panels() -> None:
    """Validate step: the full reports (``/dev/validation``), as tables and as a JSON file to download."""
    try:
        reports = api_client.dev_validation()
    except ApiRefused as error:
        st.error(error.message)
        return
    fidelity, privacy = reports["fidelity"], reports["privacy"]
    with st.expander("Developer: fidelity report (against the holdout)"):
        st.caption("Distance of every column from the holdout; baseline and spread: mean and standard deviation of "
                   "the same distance between 5 groups of real people (10 pairs); z: how many spreads beyond the "
                   "baseline; excess: distance minus baseline. A warning needs both z and excess above their limits.")
        st.dataframe(fidelity_table(fidelity), hide_index=True, width="stretch")
        st.json({key: value for key, value in fidelity.items()
                 if key not in {"columns", "association_real", "association_synthetic", "histograms"}}, expanded=1)
    with st.expander("Developer: privacy report (DCR and exact copies)"):
        st.dataframe(privacy_table(privacy), hide_index=True, width="stretch")
        st.json(privacy, expanded=1)
    st.download_button("Download the full reports (JSON)", data=json.dumps(reports, indent=2),
                       file_name="validation_reports.json", mime="application/json", key="download_reports",
                       on_click="ignore")
