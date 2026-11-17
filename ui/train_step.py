"""Step 2, Train: one button, a progress bar that follows the job, Stop, and the outcome.

The job (train → generate → compare → privacy check) runs in the API. While it trains, a fragment asks ``/status``
every ``POLL_SECONDS`` and draws only the bar; when the synthetic table is ready (the privacy check goes on, shown
by the Validate step) or the job ends, it redraws the whole page, whose stage has changed.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from ui import api_client, dev_panels
from ui.api_client import ApiRefused, ApiUnavailable
from ui.navigation import checking_privacy
from ui.train_view import elapsed_text, ended_early, is_running, progress
from ui.ui_config import API_DOWN_MESSAGE, POLL_SECONDS

WHAT_HAPPENS = ("The model learns the patterns of your data on this computer. Then it creates the synthetic table, "
                "with as many rows as were used for learning, and compares it with your data. "
                "It can take from a few seconds to a few minutes. In step 3 you can look at the comparison while "
                "the app checks that no synthetic row copies a real person.")
CHECKING_TEXT = "Its privacy is being checked now: you can follow it in step 3 (Validate)."


def render(view: dict, show_dev: bool, open_validate: Callable[[], None]) -> None:
    """Draw the Train step.

    Args:
        view: ``/session`` (stage and latest job).
        show_dev: Whether the developer panels are on.
        open_validate: Callback of the "Next" button.
    """
    job = view["job"]
    if is_running(job) and not checking_privacy(job):
        live_job(job["job_id"], show_dev)
        return
    if view["stage"] in {"generated", "validated"}:
        finished(job, open_validate)
    else:
        ready_to_start(job)
    if show_dev:
        dev_panels.hyperparameters_panel()
        if job is not None:
            dev_panels.training_panel()


@st.fragment(run_every=POLL_SECONDS)
def live_job(job_id: str, show_dev: bool) -> None:
    """The bar of the running job, drawn again every ``POLL_SECONDS`` without redrawing the page.

    Streamlit keeps the errors of a fragment inside it, so the page's handler never sees them: the fragment has its
    own, for every call it makes.
    """
    try:
        draw_live_job(job_id, show_dev)
    except ApiUnavailable:
        st.error(API_DOWN_MESSAGE)


def draw_live_job(job_id: str, show_dev: bool) -> None:
    """The content of ``live_job``."""
    try:
        job = api_client.get_status(job_id)
    except ApiRefused:                                    # the job is gone (Start over elsewhere)
        st.rerun()
    if not is_running(job) or checking_privacy(job):
        st.rerun()                                        # the stage has changed: redraw the whole page
    fraction, text = progress(job)
    st.progress(fraction, text=text)
    st.caption(f"Time so far: {elapsed_text(job['elapsed_seconds'])}")
    if job["stop_requested"]:
        st.button("Stopping…", key="stop_training", disabled=True)
    elif st.button("Stop", key="stop_training"):
        with st.spinner("Stopping…"):
            try:
                api_client.stop_training()
            except ApiRefused:
                pass                                      # nothing to stop any more: the page shows why
        st.rerun()
    if show_dev:
        dev_panels.hyperparameters_panel()
        dev_panels.training_panel()


def ready_to_start(job: dict | None) -> None:
    """No result yet: what happens, the outcome of the last job if any, and the button."""
    if job is not None and job["state"] == "stopped":
        st.info("Training stopped. You can change the data in step 1 or start again.")
    if job is not None and job["state"] == "failed":
        st.error(job["error"]["message"])
    st.write(WHAT_HAPPENS)
    if st.button("Start training", key="start_training", type="primary"):
        try:
            api_client.start_training()
        except ApiRefused as error:
            st.error(error.message)
            return
        st.rerun()


def finished(job: dict | None, open_validate: Callable[[], None]) -> None:
    """The synthetic data is ready: how long it took, why it ended early if it did, and Next.

    While its privacy is checked the job is still running: no time yet, and a pointer to Validate.
    """
    done = job is not None and job["state"] == "done"
    took = f" (it took {elapsed_text(job['elapsed_seconds'])})" if done else ""
    st.success(f"The synthetic data is ready{took}.")
    if checking_privacy(job):
        st.info(CHECKING_TEXT)
    if job is not None and ended_early(job):
        st.info("The model stopped improving, so training ended early.")
    st.caption("Training again without changing the data gives the same result: to change it, go back to step 1.")
    st.button("Next: Validate →", key="next_validate", type="primary", on_click=open_validate)
