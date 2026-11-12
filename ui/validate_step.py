"""Step 3, Validate: the outcome of the privacy gate, the warnings, the download and how close the data are.

Nothing is computed here: the job already validated the synthetic table. The CSV is asked from the API only when
the button is pressed (``st.download_button`` with a function), so the page does not hold it between reruns.
"""

from __future__ import annotations

import streamlit as st

from ui import api_client, charts, dev_panels
from ui.api_client import ApiRefused
from ui.validate_view import heatmap_frame, histogram_frame

PASSED_TEXT = ("**Privacy check passed**: no synthetic row copies a real person or is closer to one than a new "
               "person would be. {n_rows:,} synthetic rows are ready.")


def render(show_dev: bool) -> None:
    """Draw the Validate step."""
    try:
        results = api_client.get_validation()
    except ApiRefused as error:
        st.info(error.message)
        return
    gate_outcome(results)
    warnings(results["warnings"])
    if results["gate"]["passed"]:
        download(results["download_name"])
        preview(results["preview"])
    closeness(results)
    if show_dev:
        dev_panels.validation_panels()


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
