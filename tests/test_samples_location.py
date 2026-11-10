"""Tests for sds.samples.location: the single rule that finds the sample datasets."""

from pathlib import Path

from sds.samples import DATA_DIR_ENV, SAMPLE_NAMES, data_dir, missing_samples, sample_path
from sds.samples.location import PROJECT_ROOT


def test_project_root_holds_the_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_default_is_data_at_the_project_root(monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    assert data_dir() == PROJECT_ROOT / "data"


def test_environment_variable_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert data_dir() == tmp_path


def test_empty_environment_variable_is_ignored(monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, "")
    assert data_dir() == PROJECT_ROOT / "data"


def test_environment_variable_is_read_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "a"))
    first = data_dir()
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "b"))
    assert data_dir() != first


def test_sample_path_uses_the_given_folder(tmp_path):
    assert sample_path("adult", tmp_path) == tmp_path / "adult.csv"


def test_sample_path_defaults_to_the_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert sample_path("pima") == tmp_path / "pima.csv"


def test_missing_samples_lists_only_absent_files_in_order(tmp_path):
    (tmp_path / "pima.csv").write_text("a\n1\n")
    assert missing_samples(["adult", "pima", "bank_marketing"], tmp_path) == ["adult", "bank_marketing"]


def test_missing_samples_ignores_a_folder_with_the_csv_name(tmp_path):
    (tmp_path / "adult.csv").mkdir()
    assert missing_samples(["adult"], tmp_path) == ["adult"]


def test_missing_folder_means_everything_missing(tmp_path):
    assert missing_samples(SAMPLE_NAMES, tmp_path / "nowhere") == list(SAMPLE_NAMES)


def test_sample_names_follow_the_manifest():
    assert SAMPLE_NAMES == ("adult", "breast_cancer", "bank_marketing", "cdc_diabetes", "diabetes_130", "pima")


def test_command_writes_into_the_data_dir_by_default(monkeypatch, tmp_path):
    from sds.samples import download

    written = []
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(download, "download_dataset", lambda spec, dest, force: written.append(Path(dest)))
    assert download.main(["--only", "pima"]) == 0
    assert written == [tmp_path]


def test_command_dest_option_wins(monkeypatch, tmp_path):
    from sds.samples import download

    written = []
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "env"))
    monkeypatch.setattr(download, "download_dataset", lambda spec, dest, force: written.append(Path(dest)))
    assert download.main(["--only", "pima", "--dest", str(tmp_path / "cli")]) == 0
    assert written == [tmp_path / "cli"]
