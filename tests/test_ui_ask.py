"""Specification tests for the Ask step: the chat, "New conversation", errors, the developer panel.

The page runs with ``AppTest`` against a fake API (``ui_fake``): no model, no server.
"""

import pytest
from streamlit.testing.v1 import AppTest

from sds.config import PROJECT_ROOT
from sds.llm_analyst.llm_config import REQUEST_TIMEOUT_SECONDS
from sds.llm_analyst.sandbox_config import RUN_TIMEOUT_SECONDS
from ui.api_client import ApiRefused
from ui.ui_config import ASK_REQUEST_SECONDS
from ui.ask_view import conversation_key, escape_markdown, image_bytes, safe_text, summary_frame
from ui_fake import FakeApi, dataset, install, job, turn

APP = str(PROJECT_ROOT / "ui" / "app.py")
WAIT_SECONDS = 30
TABLE = {"columns": ["city", "age"], "rows": [["Città", 40.0], ["Mare", None]], "truncated": False}
BUSY = ApiRefused(503, "provider_busy", "Too many questions in a short time: please wait a minute and ask again.")


# --- pure functions -------------------------------------------------------------------------

def test_image_bytes_decodes_the_chart():
    assert image_bytes(turn()).startswith(b"\x89PNG")
    assert image_bytes(turn(kind="text")) is None


def test_summary_frame_rebuilds_the_table():
    frame = summary_frame(turn(summary=TABLE))
    assert list(frame.columns) == ["city", "age"]
    assert frame["city"].tolist() == ["Città", "Mare"] and frame["age"].iloc[0] == 40.0
    assert summary_frame(turn()) is None and summary_frame(turn(kind="text")) is None


def test_conversation_key_is_the_job_of_the_session():
    """A new training drops the conversation in the API: a new job id tells the page to ask for it again."""
    assert conversation_key({"stage": "validated", "job": job("done"), "gate_passed": True}) == "abc"
    assert conversation_key({"stage": "empty", "job": None, "gate_passed": None}) is None


def test_a_question_is_shown_as_typed():
    """What the user types is not markdown: no header, no list, no formula."""
    assert escape_markdown("# 1. *age* > $50") == r"\# 1\. \*age\* \> \$50"


def test_answer_text_keeps_markdown_but_not_formulas():
    """The model may write **bold**; a dollar sign must not start a formula (e.g. ">$50K")."""
    assert safe_text("**Income** above $50K and $10K") == r"**Income** above \$50K and \$10K"


# --- the page ---------------------------------------------------------------------------------

@pytest.fixture
def fake(monkeypatch):
    def make(*, configured: bool = True, dev_mode: bool = False, turns=(), answers=()) -> FakeApi:
        api = FakeApi({"stage": "validated", "job": job("done", step=100)}, dev_mode=dev_mode, data=dataset())
        api.llm_configured = configured
        api.turns, api.answers = list(turns), list(answers)
        return install(monkeypatch, api)
    return make


def open_ask(app: AppTest | None = None) -> AppTest:
    app = app or AppTest.from_file(APP, default_timeout=WAIT_SECONDS).run()
    app.button(key="step_ask").click().run()
    assert not app.exception and app.session_state["step"] == "ask"
    return app


def texts(elements) -> str:
    return " | ".join(str(e.value) for e in elements)


def messages(app: AppTest) -> list[tuple[str, str]]:
    """Every chat bubble: (who, all its text)."""
    return [(m.name, texts(m.markdown) + texts(m.text) + texts(m.error)) for m in app.chat_message]


def ask(app: AppTest, question: str) -> AppTest:
    app.chat_input(key="ask_input").set_value(question).run()
    assert not app.exception
    return app


def test_the_step_says_what_leaves_the_computer(fake):
    fake()
    caption = texts(open_ask().caption)
    assert "column names" in caption and "no data row leaves this computer" in caption


def test_not_configured_explains_how_to_add_the_key(fake):
    api = fake(configured=False)
    app = open_ask()
    assert ".env" in texts(app.info) and "restart" in texts(app.info)
    assert not app.chat_input and api.conversation_reads == 0


def test_an_empty_conversation(fake):
    fake()
    app = open_ask()
    assert not app.chat_message
    assert app.button(key="new_conversation").disabled
    assert app.chat_input(key="ask_input").max_chars == 500          # from /health, the API's rule


def test_the_conversation_comes_from_the_api(fake):
    fake(turns=[turn(summary=TABLE), turn("Who are they?", kind="text")])
    app = open_ask()
    bubbles = messages(app)
    assert [who for who, _ in bubbles] == ["user", "assistant", "user", "assistant"]
    assert "How are ages spread?" in bubbles[0][1]
    assert "The histogram shows how ages are spread." in bubbles[1][1]
    assert "I cannot name the patients." in bubbles[3][1]
    assert len(app.image) == 1                                  # the refusal has no chart
    code = app.chat_message[1].expander[0]
    assert code.label == "Show the code" and "plot.hist" in code.code[0].value
    assert not app.chat_message[3].expander                           # nor code


def test_the_summary_table_is_shown_under_the_chart(fake):
    fake(turns=[turn(summary=TABLE), turn("And without numbers?")])
    app = open_ask()
    first, second = app.chat_message[1], app.chat_message[3]
    assert len(first.dataframe) == 1 and first.dataframe[0].value["city"].tolist() == ["Città", "Mare"]
    assert "synthetic" in first.caption[0].value and "approximate" in first.caption[0].value
    assert not second.dataframe and not second.caption                # no table: nothing to explain


NOTE = "This chart uses 'age': in the synthetic data its values are spread differently from the real ones."


def test_the_reliability_notes_are_shown_under_the_chart(fake):
    fake(turns=[turn(summary=TABLE, notes=[NOTE]), turn("And without notes?")])
    app = open_ask()
    first, second = app.chat_message[1], app.chat_message[3]
    assert NOTE in texts(first.info)
    assert not second.info                                            # no note: nothing shown


def test_a_cut_table_says_so(fake):
    fake(turns=[turn(summary=TABLE | {"truncated": True})])
    app = open_ask()
    assert "first rows" in app.chat_message[1].caption[0].value


def test_a_new_question_gets_its_answer(fake):
    api = fake(answers=[turn("And for visits?")])
    app = ask(open_ask(), "And for visits?")
    assert api.questions == ["And for visits?"]
    assert [who for who, _ in messages(app)] == ["user", "assistant"]
    assert len(app.image) == 1
    assert not app.button(key="new_conversation").disabled


def test_the_conversation_is_asked_once_not_at_every_rerun(fake):
    api = fake(turns=[turn()], answers=[turn("And for visits?")])
    app = ask(open_ask(), "And for visits?")
    app.run()
    assert api.conversation_reads == 1
    assert len(app.chat_message) == 4                                 # the new turn was added from /ask


def test_an_error_is_shown_under_its_question_and_not_kept(fake):
    api = fake(turns=[turn()], answers=[BUSY, turn("Try again")])
    app = ask(open_ask(), "And for visits?")
    bubbles = messages(app)
    assert bubbles[-2] == ("user", bubbles[-2][1]) and "And for visits?" in bubbles[-2][1]
    assert "wait a minute" in texts(app.chat_message[-1].error)
    assert api.turns == [turn()]                                      # not in the conversation
    ask(app, "Try again")
    assert "wait a minute" not in texts(app.error)                    # gone with the next question
    assert len(app.chat_message) == 4


def test_a_new_conversation(fake):
    api = fake(turns=[turn()], answers=[BUSY])
    app = ask(open_ask(), "And for visits?")
    app.button(key="new_conversation").click().run()
    assert api.new_conversations == 1
    assert not app.chat_message and not app.error
    assert app.button(key="new_conversation").disabled


def test_a_new_training_brings_a_fresh_conversation(fake):
    api = fake(turns=[turn()])
    app = open_ask()
    api.turns = []
    api.view = {"stage": "validated", "job": job("done", step=100) | {"job_id": "def"}}
    app.run()
    assert api.conversation_reads == 2 and not app.chat_message


def test_start_over_forgets_the_chat(fake):
    api = fake(turns=[turn()])
    app = open_ask()
    app.button(key="start_over").click().run()
    app.button(key="start_over_confirm").click().run()
    assert api.resets == 1 and "conversation" not in app.session_state


def test_the_developer_panel_of_the_ask_step(fake):
    api = fake(dev_mode=True, turns=[turn()])
    app = open_ask()
    assert not [e for e in app.expander if e.label.startswith("Developer: language model")]
    app.toggle(key="dev_panels").set_value(True).run()
    panel = next(e for e in app.expander if e.label.startswith("Developer: language model"))
    shown = texts(panel.markdown) + texts(panel.code)
    assert "test-model" in shown and "https://provider.test/v1" in shown
    assert "700" in shown and "90" in shown and "corrected" in shown
    assert "You are a data analyst" in shown and "df['glucose']" in shown
    api.llm_error = True
    app.run()
    panel = next(e for e in app.expander if e.label.startswith("Developer: language model"))
    assert "KeyError: 'glucose'" in texts(panel.markdown) + texts(panel.code)


def test_the_chart_is_shown_as_a_png_inside_the_answer(fake):
    fake(turns=[turn()])
    app = open_ask()
    (image,) = app.chat_message[1].image
    assert image.proto.imgs[0].url.endswith(".png")          # Streamlit serves the bytes it was given


def test_the_page_waits_for_the_slowest_answer():
    """Two calls to the model (the answer and its fix), each followed by a run: the page must not give up first."""
    assert ASK_REQUEST_SECONDS > 2 * (REQUEST_TIMEOUT_SECONDS + RUN_TIMEOUT_SECONDS)
