"""Tests for the LLM provider settings: read on demand, never into os.environ, key never shown."""

import os

import pytest

from sds.llm_analyst.errors import LLM_NOT_CONFIGURED, AnalystError
from sds.llm_analyst.settings import (
    API_KEY_ENV,
    BASE_URL_ENV,
    MODEL_ENV,
    LlmNotConfiguredError,
    load_settings,
)

FAKE_KEY = "sk-test-not-a-real-key"
GROQ_URL = "https://api.groq.com/openai/v1"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    """No LLM variable in the environment and no .env reachable from the working folder."""
    for name in (API_KEY_ENV, BASE_URL_ENV, MODEL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def write_env_file(folder, **values):
    """Write a .env file in ``folder`` with the given variables and return its path."""
    path = folder / ".env"
    path.write_text("".join(f"{name}={value}\n" for name, value in values.items()))
    return path


def test_variable_names():
    assert (API_KEY_ENV, BASE_URL_ENV, MODEL_ENV) == ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    monkeypatch.setenv(BASE_URL_ENV, GROQ_URL)
    monkeypatch.setenv(MODEL_ENV, "some-model")
    settings = load_settings()
    assert (settings.api_key, settings.base_url, settings.model) == (FAKE_KEY, GROQ_URL, "some-model")


def test_settings_come_from_the_env_file(tmp_path):
    env_file = write_env_file(tmp_path, LLM_API_KEY=FAKE_KEY, LLM_BASE_URL=GROQ_URL, LLM_MODEL="m")
    settings = load_settings(env_file)
    assert (settings.api_key, settings.base_url, settings.model) == (FAKE_KEY, GROQ_URL, "m")


def test_env_file_is_found_from_a_subfolder_of_the_launch_folder(monkeypatch, tmp_path):
    """Launched from a subfolder, the app finds the file in a parent folder."""
    write_env_file(tmp_path, LLM_API_KEY=FAKE_KEY, LLM_MODEL="m")
    launch_folder = tmp_path / "subfolder"
    launch_folder.mkdir()
    monkeypatch.chdir(launch_folder)
    assert load_settings().api_key == FAKE_KEY


def test_the_environment_wins_over_the_file(monkeypatch, tmp_path):
    env_file = write_env_file(tmp_path, LLM_API_KEY="sk-from-file", LLM_MODEL="file-model")
    monkeypatch.setenv(MODEL_ENV, "env-model")
    settings = load_settings(env_file)
    assert (settings.api_key, settings.model) == ("sk-from-file", "env-model")


def test_the_file_never_enters_the_process_environment(tmp_path):
    """The key stays out of os.environ, so a child process cannot inherit it by mistake."""
    env_file = write_env_file(tmp_path, LLM_API_KEY=FAKE_KEY, LLM_MODEL="m")
    load_settings(env_file)
    assert API_KEY_ENV not in os.environ and MODEL_ENV not in os.environ


@pytest.mark.parametrize("value", [None, "", "   "], ids=["missing", "empty", "blank"])
def test_no_base_url_means_openai(monkeypatch, value):
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    monkeypatch.setenv(MODEL_ENV, "m")
    if value is not None:
        monkeypatch.setenv(BASE_URL_ENV, value)
    assert load_settings().base_url is None


def test_values_are_stripped(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, f"  {FAKE_KEY} ")
    monkeypatch.setenv(MODEL_ENV, " m ")
    settings = load_settings()
    assert (settings.api_key, settings.model) == (FAKE_KEY, "m")


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        ({}, (API_KEY_ENV, MODEL_ENV)),
        ({API_KEY_ENV: FAKE_KEY}, (MODEL_ENV,)),
        ({MODEL_ENV: "m"}, (API_KEY_ENV,)),
        ({API_KEY_ENV: "  ", MODEL_ENV: ""}, (API_KEY_ENV, MODEL_ENV)),
    ],
    ids=["nothing", "no-model", "no-key", "blank-values"],
)
def test_missing_settings_are_named(monkeypatch, present, missing):
    """The model is required like the key: there is no default model in the code."""
    for name, value in present.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(LlmNotConfiguredError) as error:
        load_settings()
    assert error.value.missing == missing
    assert all(name in error.value.message for name in missing)
    assert ".env" in error.value.message


def test_not_configured_is_an_analyst_error_with_its_code(monkeypatch):
    """The API turns every AnalystError into {"code", "message"} with one rule."""
    with pytest.raises(AnalystError) as error:
        load_settings()
    assert isinstance(error.value, LlmNotConfiguredError)
    assert error.value.code == LLM_NOT_CONFIGURED


def test_the_key_is_never_shown(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    monkeypatch.setenv(MODEL_ENV, "m")
    settings = load_settings()
    assert FAKE_KEY not in repr(settings) and FAKE_KEY not in str(settings)
    assert "m" in repr(settings)
