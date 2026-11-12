"""Streamlit UI of the Synthetic Data Sandbox (``sds-app``, or ``streamlit run ui/app.py`` from the project root).

A guided path in 4 steps: Upload → Train → Validate → Ask. The state lives in the API: at every run the page
reads ``/session`` and opens only the steps reached, so a page refresh (a new Streamlit session) resumes
where the user was, even during a training.
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script: the project root must be on the path to import the ui package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui import api_client, dev_panels, state, train_step, upload_step, validate_step  # noqa: E402
from ui.api_client import ApiRefused, ApiUnavailable  # noqa: E402
from ui.navigation import STEP_LABELS, Step, default_step, reachable_steps  # noqa: E402
from ui.ui_config import API_DOWN_MESSAGE, APP_TITLE, CLOSED_MESSAGE  # noqa: E402

STEP_KEY, GENERATION_KEY, DEV_KEY, CONFIRM_KEY = (state.STEP, state.UPLOADER_GENERATION, state.DEV_PANELS,
                                                   state.CONFIRMING_START_OVER)
CONFIRM_CLOSE_KEY, CLOSED_KEY, CLOSE_REQUESTED_KEY, CLOSE_ERROR_KEY = (state.CONFIRMING_CLOSE, state.CLOSED,
                                                                    state.CLOSE_REQUESTED, state.CLOSE_ERROR)


def open_step(step: Step) -> None:
    """Button callback: look at ``step``."""
    st.session_state[STEP_KEY] = step.value


def forget_ui_state() -> None:
    """Drop everything the UI kept (uploaded file included) and open Upload.

    The developer toggle is a display preference, not data: it stays (cleared, the switch would still look on
    in the browser while the panels disappear).
    """
    generation = st.session_state.get(GENERATION_KEY, 0)
    show_dev = st.session_state.get(DEV_KEY, False)
    st.session_state.clear()
    st.session_state[GENERATION_KEY] = generation + 1
    st.session_state[DEV_KEY] = show_dev
    st.session_state[STEP_KEY] = Step.UPLOAD.value


def ask_start_over() -> None:
    """Button callback: open the "Start over?" dialog."""
    st.session_state[CONFIRM_KEY] = True


def cancel_start_over() -> None:
    """Close the dialog without resetting (Cancel, the X or a click outside)."""
    st.session_state[CONFIRM_KEY] = False


@st.dialog("Start over?", on_dismiss=cancel_start_over)
def confirm_start_over() -> None:
    """Ask before throwing away the dataset, the model and the results.

    Opened by a flag rather than by the button's return value, so the dialog survives its own reruns.
    """
    st.write("The uploaded data, the trained model and the synthetic data will be forgotten. "
             "Download the synthetic data first if you need it.")
    left, right = st.columns(2)
    if left.button("Start over", key="start_over_confirm", type="primary"):
        with st.spinner("Stopping and forgetting everything…"):
            try:
                api_client.reset()
            except ApiRefused as error:
                st.error(error.message)
                return
            except ApiUnavailable:
                st.error(API_DOWN_MESSAGE)
                return
        forget_ui_state()
        st.rerun()
    if right.button("Cancel", key="start_over_cancel"):
        cancel_start_over()
        st.rerun()


def ask_close() -> None:
    """Button callback: open the "Close the app?" dialog."""
    st.session_state[CONFIRM_CLOSE_KEY] = True


def cancel_close() -> None:
    """Close the dialog and keep the app running (Cancel, the X or a click outside)."""
    st.session_state[CONFIRM_CLOSE_KEY] = False


@st.dialog("Close the app?", on_dismiss=cancel_close)
def confirm_close() -> None:
    """Ask before closing; the closing itself happens on the goodbye page (``goodbye``)."""
    st.write("The app will stop. The uploaded data, the trained model and the synthetic data will be discarded. "
             "Download the synthetic data first if you need it.")
    left, right = st.columns(2)
    if left.button("Close", key="close_confirm", type="primary"):
        st.session_state.clear()                       # nothing the UI kept outlives the app
        st.session_state[CLOSED_KEY] = True
        st.rerun()
    if right.button("Cancel", key="close_cancel"):
        cancel_close()
        st.rerun()


def goodbye() -> None:
    """The page of a closed app: the message first, then the API is asked to close (once).

    In this order because the API ends right after its answer and ``sds-app`` then stops the UI: a message drawn
    after the call might never reach the browser. The browser tab cannot close itself.
    """
    st.markdown(CLOSED_MESSAGE)
    if st.session_state.get(CLOSE_REQUESTED_KEY):
        st.stop()
    st.session_state[CLOSE_REQUESTED_KEY] = True
    try:
        api_client.shutdown()
    except ApiRefused as error:
        refused = error.message
    except ApiUnavailable:
        refused = API_DOWN_MESSAGE
    else:
        st.stop()
    st.session_state.clear()                           # still running: back to the app, with the reason
    st.session_state[CLOSE_ERROR_KEY] = refused
    st.rerun()


def step_bar(reachable: set[Step], current: Step) -> None:
    """The four steps in a row: the current one highlighted, the ones not reached yet disabled."""
    for column, step in zip(st.columns(len(Step)), Step):
        column.button(STEP_LABELS[step], key=f"step_{step.value}", on_click=open_step, args=(step,),
                      disabled=step not in reachable, type="primary" if step is current else "secondary",
                      width="stretch")


def show_step(step: Step, status: dict, view: dict, show_dev: bool) -> None:
    """The content of ``step`` (Ask: a placeholder until the language model is added)."""
    st.header(STEP_LABELS[step])
    if step is Step.UPLOAD:
        upload_step.render(status, view, show_dev, open_train=lambda: open_step(Step.TRAIN))
    elif step is Step.TRAIN:
        train_step.render(view, show_dev, open_validate=lambda: open_step(Step.VALIDATE))
    elif step is Step.VALIDATE:
        validate_step.render(show_dev)
    else:
        st.caption("This step is being built.")


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
if st.session_state.get(CLOSED_KEY):
    goodbye()
if CLOSE_ERROR_KEY in st.session_state:
    st.error(st.session_state.pop(CLOSE_ERROR_KEY))

try:
    status = api_client.health()
    view = api_client.get_session()
except (ApiUnavailable, ApiRefused):
    st.error(API_DOWN_MESSAGE)
    st.stop()

st.session_state.setdefault(GENERATION_KEY, 0)
reachable = reachable_steps(view)
if st.session_state.get(STEP_KEY) not in {step.value for step in reachable}:
    st.session_state[STEP_KEY] = default_step(view).value     # first run, or the step is no longer reachable
current = Step(st.session_state[STEP_KEY])

with st.sidebar:
    st.button("Start over", key="start_over", on_click=ask_start_over,
              disabled=view["stage"] == "empty" and view["job"] is None)
    if status["closable"]:                             # only when sds-app started the app: it stops the UI too
        st.button("Close", key="close", on_click=ask_close, help="Stop the app and discard the data in memory.")
    show_dev = status["dev_mode"] and st.toggle("Show developer panels", key=DEV_KEY)

if st.session_state.get(CONFIRM_KEY):
    confirm_start_over()
if st.session_state.get(CONFIRM_CLOSE_KEY):
    confirm_close()
step_bar(reachable, current)
try:
    show_step(current, status, view, show_dev)
except ApiUnavailable:
    st.error(API_DOWN_MESSAGE)
    st.stop()
if show_dev:
    dev_panels.session_panel(view)
