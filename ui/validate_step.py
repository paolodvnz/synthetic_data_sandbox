"""Step 3, Validate: the outcome of the privacy gate, the warnings, the download and how close the data are.

Nothing is computed here: the job compares the synthetic table with the real data, then checks its privacy. While
the check runs, the comparison is already shown and a fragment asks ``/status`` every ``POLL_SECONDS``; when the job
ends it redraws the whole page with the gate. The CSV is asked from the API only when the button is pressed
(``st.download_button`` with a function), so the page does not hold it between reruns.
"""

from __future__ import annotations

import streamlit as st

from ui import api_client, charts, dev_panels
from ui.api_client import ApiRefused, ApiUnavailable
from ui.navigation import checking_privacy
from ui.train_view import elapsed_text
from ui.ui_config import API_DOWN_MESSAGE, POLL_SECONDS
from ui.validate_view import heatmap_frame, histogram_frame

PASSED_TEXT = ("**Privacy check passed**: no synthetic row copies a real person or is closer to one than a new "
               "person would be. {n_rows:,} synthetic rows are ready.")
CHECKING_TEXT = ("**Checking privacy…** The app is making sure that no synthetic row copies a real person. The "
                 "download and step 4 (Ask) open when the check passes. Meanwhile, see below how close the synthetic "
                 "data are to yours.")
DISCARD_TEXT = "Stop and discard throws the synthetic data away: to change it, go back to step 1, then train again."


def render(view: dict, show_dev: bool) -> None:
    """Draw the Validate step.

    Args:
        view: ``/session`` (stage and latest job).
        show_dev: Whether the developer panels are on.
    """
    try:
        results = api_client.get_validation()
    except ApiRefused as error:
        st.info(error.message)
        return
    if results["gate"] is None:
        if checking_privacy(view["job"]):
            live_check(view["job"]["job_id"])
    else:
        gate_outcome(results)
    warnings(results["warnings"])
    if results["gate"] is not None and results["gate"]["passed"]:
        download(results["download_name"])
        preview(results["preview"])
    closeness(results)
    if show_dev:
        dev_panels.validation_panels()


@st.fragment(run_every=POLL_SECONDS)
def live_check(job_id: str) -> None:
    """The running privacy check, drawn again every ``POLL_SECONDS``; its end redraws the whole page.

    Streamlit keeps the errors of a fragment inside it, so the page's handler never sees them: the fragment has its
    own, for every call it makes.
    """
    try:
        draw_live_check(job_id)
    except ApiUnavailable:
        st.error(API_DOWN_MESSAGE)


def draw_live_check(job_id: str) -> None:
    """The content of ``live_check``."""
    try:
        job = api_client.get_status(job_id)
    except ApiRefused:                                    # the job is gone (Start over elsewhere)
        st.rerun()
    if not checking_privacy(job):
        st.rerun()                                        # the gate is known, or the table was dropped
    st.info(CHECKING_TEXT)
    st.caption(f"Time so far: {elapsed_text(job['elapsed_seconds'])}")
    if job["stop_requested"]:
        st.button("Stopping…", key="stop_check", disabled=True)
        return
    if st.button("Stop and discard", key="stop_check"):
        with st.spinner("Stopping…"):
            try:
                api_client.stop_training()
            except ApiRefused:
                pass                                      # nothing to stop any more: the page shows why
        st.rerun()
    st.caption(DISCARD_TEXT)


def gate_outcome(results: dict) -> None:
    """Passed: how many rows are ready. Failed: why (the engine) and what to do (the app)."""
    gate = results["gate"]
    if gate["passed"]:
        st.success(PASSED_TEXT.format(n_rows=results["n_rows"]))
        return
    reasons = " ".join(failure["message"] for failure in gate["failures"])
    st.error(f"**The synthetic data cannot be shared.** {reasons}\n\n{gate['advice']}")


def warnings(items: list[dict]) -> None:
    """Non-blocking warnings: worth knowing, the data can still be used."""
    if not items:
        return
    st.markdown("**Worth knowing** (these do not block the download)")
    for item in items:
        st.warning(item["message"])


def download(file_name: str) -> None:
    """The CSV, fetched from the API only on click."""
    st.download_button("Download the synthetic data (CSV)", data=api_client.download_csv, file_name=file_name,
                       mime="text/csv", type="primary", key="download_csv", on_click="ignore")
    st.caption(f"Saved as {file_name}. The real data never leave this computer; this file holds only synthetic rows.")


def preview(table: dict) -> None:
    """The first synthetic rows."""
    st.subheader("A look at the synthetic data")
    st.caption("These rows are synthetic: none of them is a real person.")
    st.dataframe(dev_panels.frame_from_json(table), hide_index=True, width="stretch")


def closeness(results: dict) -> None:
    """Heatmaps of the associations and the histogram of one column, real next to synthetic."""
    st.subheader("How close is it to your data?")
    associations = results["associations"]
    columns = associations["columns"]
    st.caption("How strongly each pair of columns goes together (darker = stronger). Similar pictures mean the "
               "synthetic data keep the relationships of the real data.")
    st.altair_chart(charts.heatmaps(heatmap_frame(columns, associations["real"]),
                                    heatmap_frame(columns, associations["synthetic"]), columns))
    by_name = {h["name"]: h for h in results["histograms"]}
    name = st.selectbox("Compare one column", list(by_name), key="histogram_column")
    st.altair_chart(charts.histogram(histogram_frame(by_name[name]), name), width="stretch")
