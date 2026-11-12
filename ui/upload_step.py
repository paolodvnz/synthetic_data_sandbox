"""Step 1, Upload: the file, the columns the engine proposes, the choices, the warnings and the rows.

The file goes to the API once, then the uploader is emptied (new key): the UI does not keep the real data.
The dataset view is kept in ``st.session_state`` and read again from ``/dataset`` after a page refresh.
Choices are sent only with "Apply changes": every ``/choices`` prepares the data again (seconds on large files).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from ui import api_client, dev_panels, state
from ui.api_client import ApiRefused
from ui.upload_view import (READINGS, ROLE_LABELS, attention_issues, columns_frame, current_readings,
                            desired_choices, detail_issues)
from ui.ui_config import UPLOAD_TYPES

READING_LABELS = {role: label for label, role in READINGS.items()}


def render(status: dict, view: dict, show_dev: bool, open_train: Callable[[], None]) -> None:
    """Draw the Upload step.

    Args:
        status: ``/health`` (developer mode, upload limit).
        view: ``/session`` (stage and latest job).
        show_dev: Whether the developer panels are on.
        open_train: Callback of the "Next" button.
    """
    running = view["job"] is not None and view["job"]["state"] == "running"
    if running:
        st.info("A training is running: the data and the choices cannot change until it ends or you stop it.")
    data = current_dataset(view)

    choose_file(status, view["stage"] != "empty")
    if data is None:
        return
    show_dataset(data, running, open_train)
    if show_dev:
        dev_panels.upload_panels(data)


def current_dataset(view: dict) -> dict | None:
    """The dataset view: kept from the last answer, or read from the API (first run after a page refresh)."""
    if view["stage"] == "empty":
        st.session_state.pop(state.DATASET, None)
        return None
    if state.DATASET not in st.session_state:
        try:
            st.session_state[state.DATASET] = api_client.get_dataset()
        except ApiRefused:
            return None
    return st.session_state[state.DATASET]


def received(data: dict) -> None:
    """Keep a new dataset view and reset the table and Advanced to its choices."""
    st.session_state[state.DATASET] = data
    st.session_state[state.EDITOR_TOKEN] = st.session_state.get(state.EDITOR_TOKEN, 0) + 1


def choose_file(status: dict, loaded: bool) -> None:
    """The uploader (and, in developer mode, the sample datasets), only while no dataset is loaded.

    One dataset per session (the API refuses a second one): another file replaces nothing by surprise, it comes
    after Start over, which also throws away the model and the synthetic data.
    """
    if loaded:
        st.caption("To use another file, press Start over: it clears this dataset and everything made from it.")
        return
    if status["dev_mode"]:
        samples_menu()
    generation = st.session_state.setdefault(state.UPLOADER_GENERATION, 0)
    uploaded = st.file_uploader("Upload a CSV file", type=UPLOAD_TYPES, key=f"uploader_{generation}",
                                help=f"Up to {status['max_upload_mb']} MB. The file stays on this computer.")
    if uploaded is not None:
        send(lambda: api_client.upload(uploaded.getvalue(), uploaded.name))
    if message := st.session_state.get(state.UPLOAD_ERROR):
        st.error(message)


def send(load: Callable[[], dict]) -> None:
    """Load a dataset through the API, then empty the uploader and draw the page again."""
    with st.spinner("Reading the file…"):
        try:
            received(load())
            st.session_state.pop(state.UPLOAD_ERROR, None)
        except ApiRefused as error:
            st.session_state.pop(state.DATASET, None)
            st.session_state[state.UPLOAD_ERROR] = error.message
    st.session_state[state.UPLOADER_GENERATION] += 1
    st.rerun()


def samples_menu() -> None:
    """Developer mode: load one of the downloaded sample datasets."""
    names = [s["name"] for s in api_client.list_samples() if s["available"]]
    if not names:
        st.caption("No sample dataset downloaded: run sds-download-samples.")
        return
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    name = left.selectbox("Try a sample dataset", names, key="sample_name")
    if right.button("Load", key="load_sample", width="stretch"):
        send(lambda: api_client.load_sample(name))


def show_dataset(data: dict, running: bool, open_train: Callable[[], None]) -> None:
    """The current file: summary, warnings, columns, Advanced, result and rows."""
    token = st.session_state.setdefault(state.EDITOR_TOKEN, 0)
    st.markdown(f"**Current file: {data['file']['name']}** · {data['file']['n_rows']:,} rows · "
                f"{len(data['columns'])} columns")
    for issue in attention_issues(data["issues"]):
        st.warning(issue["message"])

    keep = columns_table(data, token, running)
    readings, unify_case = advanced(data, keep, token, running)
    desired = desired_choices(data["columns"], keep, readings, unify_case)
    pending = desired != data["choices"]
    if pending:
        apply_bar(desired, token)

    st.subheader("Result")
    if not data["ready"]:
        st.error(data["problem"]["message"])
    else:
        rows = data["rows"]
        left, middle, right = st.columns(3)
        left.metric("Used for learning", f"{rows['train']:,}")
        middle.metric("Kept apart for the privacy check", f"{rows['holdout']:,}")
        right.metric("Removed (missing values)", f"{rows['removed']:,}")
    details = detail_issues(data["issues"])
    if details:
        with st.expander(f"Details ({len(details)} notes)"):
            for issue in details:
                st.markdown(f"- {issue['message']}")
    if data["ready"]:
        st.button("Next: Train →", key="next_train", type="primary", on_click=open_train,
                  disabled=pending or running, help="Apply or discard your changes first." if pending else None)


def columns_table(data: dict, token: int, running: bool) -> dict[str, bool]:
    """The columns with a "Keep" tick; returns column name -> ticked."""
    st.subheader("Columns")
    st.write("We suggest which columns to use. **Untick columns that identify people** (patient ID, name, tax "
             "code). You can also tick a column we left out.")
    table = columns_frame(data)
    edited = st.data_editor(table, column_config={"Keep": st.column_config.CheckboxColumn()},
                            disabled=True if running else [c for c in table.columns if c != "Keep"],
                            hide_index=True, width="stretch", key=f"columns_{token}")
    return dict(zip(edited["Column"], edited["Keep"]))


def advanced(data: dict, keep: dict[str, bool], token: int, running: bool) -> tuple[dict[str, str], bool]:
    """Only the choices that have an effect on this file; returns the readings and the case option."""
    readings = current_readings(data["columns"])
    unify_case = data["choices"]["unify_case"]
    with st.expander("Advanced (optional)"):
        if data["case_merges"]:
            st.markdown("**Upper and lower case**")
            for issue in data["case_merges"]:
                st.markdown(f"- {issue['message']}")
            unify_case = st.checkbox("Treat values like 'Male' and 'male' as the same value", value=unify_case,
                                     key=f"case_{token}", disabled=running)

        kept = [c for c in data["columns"] if keep[c["name"]]]
        changeable = [c for c in kept if c["fixed_reason"] is None]
        st.markdown("**You can change these**")
        if changeable:
            st.caption("These columns contain numbers with few different values, which can be quantities (age, "
                       "number of visits) or codes (1 = emergency, 2 = urgent...). Change the reading only if we "
                       "got it wrong.")
        else:
            st.caption("No column you kept has an ambiguous reading.")
        for c in changeable:
            left, right = st.columns([3, 2])
            left.markdown(f"**{c['name']}** · e.g. {', '.join(c['examples'])}")
            picked = right.radio(f"Reading of {c['name']}", list(READINGS),
                                 index=list(READINGS).index(READING_LABELS[readings[c['name']]]), horizontal=True,
                                 label_visibility="collapsed", key=f"reading_{token}_{c['name']}", disabled=running)
            readings[c["name"]] = READINGS[picked]
            if READINGS[picked] != c["kept_role"]:
                right.caption(f"Changed by you (we read it as {ROLE_LABELS[c['kept_role']]})")

        fixed = [c for c in kept if c["fixed_reason"] is not None]
        if fixed:
            st.markdown("**Fixed by the data**")
            st.dataframe(pd.DataFrame([{"Column": c["name"], "Read as": ROLE_LABELS[c["final_role"] or c["kept_role"]],
                                        "Why it cannot change": c["fixed_reason"]} for c in fixed]),
                         hide_index=True, width="stretch")
    return readings, unify_case


def apply_bar(desired: dict, token: int) -> None:
    """The user changed something: send it, or go back to the applied choices."""
    st.info("You changed the choices: apply them to prepare the data again.")
    left, right = st.columns(2)
    if left.button("Apply changes", key="apply_choices", type="primary", width="stretch"):
        with st.spinner("Preparing the data…"):
            try:
                received(api_client.set_choices(desired))
            except ApiRefused as error:
                st.error(error.message)
                return
        st.rerun()
    if right.button("Discard changes", key="discard_choices", width="stretch"):
        st.session_state[state.EDITOR_TOKEN] = token + 1
        st.rerun()

