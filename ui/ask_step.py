"""Step 4, Ask: questions in plain language about the synthetic data, answered with a chart and a short explanation.

The conversation lives in the API: the page asks for it once (and again when a new training replaced it), then adds
the turns that ``/ask`` returns. A failed question is shown under it until the next one, never kept.
"""

from __future__ import annotations

import streamlit as st

from ui import api_client, dev_panels, state
from ui.api_client import ApiRefused
from ui.ask_view import conversation_key, escape_markdown, image_bytes, safe_text, summary_frame

PRIVACY_NOTE = ("Your questions and the column names (with their possible values) are sent to the AI service; "
                "no data row leaves this computer. The charts are drawn here, on the synthetic data.")
NOT_CONFIGURED_TEXT = ("Questions need an AI service. Copy `.env.example` to `.env` next to the app, write your key "
                       "and model in it, then restart the app.")
THINKING_TEXT = "Thinking… this usually takes a few seconds, at most a few minutes."
INPUT_PLACEHOLDER = "Ask a question about your data, e.g. how a value is spread, or how two columns are related"
SUMMARY_NOTE = ("Numbers computed here on the synthetic data: approximate, close to the real ones but not "
                "identical.")
TRUNCATED_NOTE = "Only the first rows and columns are shown."


def render(status: dict, view: dict, show_dev: bool) -> None:
    """Draw the Ask step."""
    st.caption(PRIVACY_NOTE)
    if not status["llm_configured"]:
        st.info(NOT_CONFIGURED_TEXT)
        return
    turns = conversation(view)
    if st.button("New conversation", key="new_conversation", disabled=not turns,
                 help="Forget the questions and start a new topic. The model and the synthetic data stay."):
        api_client.new_conversation()
        turns.clear()
        st.session_state.pop(state.ASK_ERROR, None)
        st.rerun()
    for each in turns:
        show_turn(each)
    failed = st.session_state.get(state.ASK_ERROR)
    if failed is not None:
        show_question(failed["question"])
        with st.chat_message("assistant"):
            st.error(failed["message"])
    question = st.chat_input(INPUT_PLACEHOLDER, key="ask_input", max_chars=status["max_question_chars"])
    if question:
        answer(question, turns)
    if show_dev:
        dev_panels.llm_panel()


def conversation(view: dict) -> list[dict]:
    """The turns of the conversation, asked from the API only when the page has none for the current job."""
    key = conversation_key(view)
    cached = st.session_state.get(state.CONVERSATION)
    if cached is None or cached["key"] != key:
        cached = {"key": key, "turns": api_client.get_conversation()["turns"]}
        st.session_state[state.CONVERSATION] = cached
        st.session_state.pop(state.ASK_ERROR, None)
    return cached["turns"]


def answer(question: str, turns: list[dict]) -> None:
    """Send the question, wait for the answer, keep it (or the error) and draw the page again."""
    st.session_state.pop(state.ASK_ERROR, None)
    show_question(question)
    with st.chat_message("assistant"), st.spinner(THINKING_TEXT):
        try:
            turns.append(api_client.ask(question))
        except ApiRefused as error:
            st.session_state[state.ASK_ERROR] = {"question": question, "message": error.message}
    st.rerun()


def show_question(question: str) -> None:
    """The user's question, as typed."""
    with st.chat_message("user"):
        st.markdown(escape_markdown(question))


def show_turn(turn: dict) -> None:
    """One answered question: the chart, its summary table, its reliability notes, the explanation and the code
    (closed)."""
    show_question(turn["question"])
    with st.chat_message("assistant"):
        png = image_bytes(turn)
        if png is not None:
            st.image(png)
        table = summary_frame(turn)
        if table is not None:
            st.dataframe(table, hide_index=True)
            st.caption(f"{SUMMARY_NOTE} {TRUNCATED_NOTE}" if turn["summary"]["truncated"] else SUMMARY_NOTE)
        for note in turn["notes"]:
            st.info(note)
        if turn["text"]:
            st.markdown(safe_text(turn["text"]))
        if turn["code"] is not None:
            with st.expander("Show the code"):
                st.code(turn["code"], language="python")
