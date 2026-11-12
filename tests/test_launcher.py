"""Specification tests for ``sds-app``: commands, ports, start order, supervision and shutdown.

Fake servers (``python -m http.server`` serving the two health files) stand in for the API and Streamlit, so
the whole flow runs without PyTorch or a browser. Every wait has a timeout.
"""

import socket
import subprocess
import sys
import threading
import time
import tomllib

import pytest

import sds.launcher as launcher
from sds.config import CLOSABLE_ENV, CLOSED_BY_USER_EXIT_CODE, MAX_UPLOAD_MB, PROJECT_ROOT
from sds.launcher import Command, build_commands, parse_args, port_in_use, stop, supervise, wait_until_ready
from sds.launcher_config import API_URL_ENV, DEFAULT_API_PORT, DEFAULT_UI_PORT

WAIT_SECONDS = 30
PYTHON = "/opt/python/bin/python3"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sleeper(code: str = "import time; time.sleep(60)") -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code])


CLOSING_SERVER = """
import functools, http.server, os, sys, threading, time
port, folder, after, code = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder)
server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(after)
os._exit(code)
"""


def closing_server(tmp_path, port: int, *, after: float = 1.5, code: int = CLOSED_BY_USER_EXIT_CODE) -> Command:
    """A fake server that answers the health paths, then ends with ``code`` (the user pressed "Close")."""
    fake_server(tmp_path, port)
    return Command("closing", [sys.executable, "-c", CLOSING_SERVER, str(port), str(tmp_path), str(after), str(code)],
                   {}, tmp_path)


def fake_server(tmp_path, port: int) -> Command:
    """A static server answering both health paths with 200."""
    (tmp_path / "health").write_text("ok")
    (tmp_path / "_stcore").mkdir(exist_ok=True)
    (tmp_path / "_stcore" / "health").write_text("ok")
    return Command("fake", [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
                            "--directory", str(tmp_path)], {}, tmp_path)


# --- commands -------------------------------------------------------------------------------

def test_api_command():
    api, _ = build_commands(dev=False, api_port=8123, ui_port=8765, python=PYTHON)
    assert api.args == [PYTHON, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8123"]
    assert api.cwd == PROJECT_ROOT


def test_ui_command_is_local_headless_and_without_telemetry():
    # toolbarMode viewer: no "Deploy" button nor developer menu in the toolbar (the app is local and single-user).
    _, ui = build_commands(dev=False, api_port=8123, ui_port=8765, python=PYTHON)
    assert ui.args[:5] == [PYTHON, "-m", "streamlit", "run", "ui/app.py"]
    options = dict(zip(ui.args[5::2], ui.args[6::2], strict=True))
    assert options == {"--server.address": "localhost", "--server.port": "8765", "--server.headless": "true",
                       "--server.maxUploadSize": str(MAX_UPLOAD_MB), "--browser.gatherUsageStats": "false",
                       "--client.toolbarMode": "viewer"}
    assert ui.env[API_URL_ENV] == "http://127.0.0.1:8123" and ui.cwd == PROJECT_ROOT


def test_only_the_api_may_close_the_app():
    api, ui = build_commands(dev=False, api_port=DEFAULT_API_PORT, ui_port=DEFAULT_UI_PORT)
    assert api.env[CLOSABLE_ENV] == "1" and CLOSABLE_ENV not in ui.env


@pytest.mark.parametrize("dev, value", [(True, "1"), (False, "0")])
def test_dev_mode_is_always_set_explicitly(dev, value):
    # "0" and not absent: an SDS_DEV_MODE=1 exported in the shell must not turn it on without --dev.
    api, _ = build_commands(dev=dev, api_port=1, ui_port=2, python=PYTHON)
    assert api.env == {"SDS_DEV_MODE": value, CLOSABLE_ENV: "1"}


def test_default_python_is_the_running_one():
    api, ui = build_commands(dev=False, api_port=1, ui_port=2)
    assert api.args[0] == ui.args[0] == sys.executable


def test_arguments():
    args = parse_args([])
    assert (args.dev, args.api_port, args.ui_port) == (False, DEFAULT_API_PORT, DEFAULT_UI_PORT)
    args = parse_args(["--dev", "--api-port", "9000", "--ui-port", "9001"])
    assert (args.dev, args.api_port, args.ui_port) == (True, 9000, 9001)


@pytest.mark.parametrize("argv", [["--api-port", "0"], ["--ui-port", "70000"], ["--api-port", "x"],
                                  ["--api-port", "9000", "--ui-port", "9000"]])
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_the_command_is_registered():
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert data["project"]["scripts"]["sds-app"] == "sds.launcher:main"


# --- ports, readiness, supervision ----------------------------------------------------------

def test_port_in_use():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert port_in_use("127.0.0.1", s.getsockname()[1])
    assert not port_in_use("127.0.0.1", free_port())


def test_a_port_left_in_time_wait_is_free():
    """Restarting right after a stop: the old connections linger in TIME_WAIT, but nothing listens any more."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    accepted, _ = listener.accept()
    accepted.close()                     # the server side closes first: it is the one left in TIME_WAIT
    client.close()
    listener.close()
    with socket.socket() as probe:       # the bind check of the first version refused this port
        with pytest.raises(OSError):
            probe.bind(("127.0.0.1", port))
    assert not port_in_use("127.0.0.1", port)


def test_wait_until_ready(tmp_path):
    port = free_port()
    server = launcher.start(fake_server(tmp_path, port))
    try:
        assert wait_until_ready(f"http://127.0.0.1:{port}/health", server, WAIT_SECONDS)
    finally:
        stop([server], WAIT_SECONDS)


def test_wait_gives_up_at_once_when_the_process_dies():
    dead = sleeper("raise SystemExit(3)")
    start = time.monotonic()
    assert not wait_until_ready(f"http://127.0.0.1:{free_port()}/health", dead, WAIT_SECONDS)
    assert time.monotonic() - start < WAIT_SECONDS / 2


def test_wait_gives_up_after_the_timeout():
    alive = sleeper()
    try:
        start = time.monotonic()
        assert not wait_until_ready(f"http://127.0.0.1:{free_port()}/health", alive, 0.5)
        assert time.monotonic() - start < WAIT_SECONDS / 2
    finally:
        stop([alive], WAIT_SECONDS)


def test_stop_terminates_and_then_kills():
    polite = sleeper()
    stubborn = sleeper("import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                       "print('ready', flush=True)\ntime.sleep(60)")
    time.sleep(0.5)                                           # let it install the handler
    start = time.monotonic()
    stop([polite, stubborn], timeout=1)
    assert polite.poll() is not None and stubborn.poll() is not None
    assert time.monotonic() - start < WAIT_SECONDS / 2


def test_stop_ignores_processes_already_ended():
    done = sleeper("pass")
    done.wait(WAIT_SECONDS)
    stop([done], timeout=1)


def test_supervise_returns_when_a_process_ends():
    alive, failing = sleeper(), sleeper("import time; time.sleep(0.3); raise SystemExit(4)")
    try:
        assert supervise([alive, failing]) == 4
    finally:
        stop([alive], WAIT_SECONDS)


def test_supervise_reports_a_clean_exit_as_a_failure():
    alive, quitting = sleeper(), sleeper("pass")
    try:
        assert supervise([alive, quitting]) == 1          # the app no longer works
    finally:
        stop([alive], WAIT_SECONDS)


# --- main -----------------------------------------------------------------------------------

@pytest.fixture
def started(monkeypatch):
    """Record every process the launcher starts; stop the leftovers at the end."""
    processes = []
    real = launcher.start

    def recording(command):
        process = real(command)
        processes.append(process)
        return process

    monkeypatch.setattr(launcher, "start", recording)
    yield processes
    stop(processes, WAIT_SECONDS)


def fake_commands(monkeypatch, tmp_path, api: Command | None = None, *, closing: str | None = None):
    """Fake API and UI; ``closing`` ("api" or "ui"): that one ends with the closing code after a while."""
    api_port, ui_port = free_port(), free_port()
    (tmp_path / "api").mkdir()
    (tmp_path / "ui").mkdir()
    server = {name: closing_server if closing == name else fake_server for name in ("api", "ui")}
    api = api or server["api"](tmp_path / "api", api_port)
    ui = server["ui"](tmp_path / "ui", ui_port)
    monkeypatch.setattr(launcher, "build_commands", lambda *, dev, api_port, ui_port: (api, ui))
    return ["--api-port", str(api_port), "--ui-port", str(ui_port)], api_port, ui_port


def run_main(argv) -> tuple[threading.Thread, list]:
    result = []
    thread = threading.Thread(target=lambda: result.append(launcher.main(argv)), daemon=True)
    thread.start()
    return thread, result


@pytest.mark.parametrize("which", ["--api-port", "--ui-port"])
def test_busy_port_starts_nothing(monkeypatch, started, capsys, which):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        other = "--ui-port" if which == "--api-port" else "--api-port"
        assert launcher.main([which, str(busy), other, str(free_port())]) == 1
    assert started == [] and f"Port {busy} is in use" in capsys.readouterr().err


def test_main_starts_both_opens_the_browser_and_stops_the_api_when_the_ui_dies(monkeypatch, tmp_path, started):
    argv, api_port, ui_port = fake_commands(monkeypatch, tmp_path)
    opened = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    thread, result = run_main(argv)
    deadline = time.monotonic() + WAIT_SECONDS
    while not opened:
        assert time.monotonic() < deadline, "the browser was never opened"
        time.sleep(0.05)
    assert opened == [f"http://localhost:{ui_port}"] and len(started) == 2
    api, ui = started
    ui.kill()                                                   # the UI dies by itself
    thread.join(WAIT_SECONDS)
    assert not thread.is_alive() and result[0] != 0
    assert api.poll() is not None                               # the API was stopped too


def test_api_that_never_answers_stops_everything(monkeypatch, tmp_path, started, capsys):
    broken = Command("api", [sys.executable, "-c", "raise SystemExit(2)"], {}, tmp_path)
    argv, _, _ = fake_commands(monkeypatch, tmp_path, api=broken)
    assert launcher.main(argv) == 1
    assert len(started) == 1                                    # the UI was never started
    assert "API" in capsys.readouterr().err


def test_closing_from_the_page_stops_both_with_success(monkeypatch, tmp_path, started, capsys):
    argv, _, _ = fake_commands(monkeypatch, tmp_path, closing="api")
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: None)
    assert launcher.main(argv) == 0
    api, ui = started
    assert api.returncode == CLOSED_BY_USER_EXIT_CODE and ui.poll() is not None     # the UI was stopped too
    out, err = capsys.readouterr()
    assert "Closed from the page" in out and err == ""


def test_the_closing_code_counts_only_from_the_api(monkeypatch, tmp_path, started, capsys):
    # The UI cannot close the app: ending with that code it is a failure like any other.
    argv, _, _ = fake_commands(monkeypatch, tmp_path, closing="ui")
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: None)
    assert launcher.main(argv) == 1
    assert "stopped unexpectedly" in capsys.readouterr().err


def test_ctrl_c_stops_both(monkeypatch, tmp_path, started):
    argv, _, _ = fake_commands(monkeypatch, tmp_path)
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: None)

    def interrupted(processes, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(launcher, "supervise", interrupted)
    assert launcher.main(argv) == 0
    assert len(started) == 2 and all(p.poll() is not None for p in started)


# --- signals: whatever stops sds-app also stops the API (no orphan with real data in memory) -----

CLOSING_LAUNCHER_SCRIPT = """
import sys
import sds.launcher as launcher
from pathlib import Path
from sds.launcher import Command

folder, api_port, ui_port, closing = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]

def fake(port):
    return Command("fake", [sys.executable, "-m", "http.server", port, "--bind", "127.0.0.1",
                            "--directory", str(folder)], {}, folder)

api = Command("closing", [sys.executable, "-c", closing, api_port, str(folder), "2", sys.argv[5]], {}, folder)
launcher.build_commands = lambda *, dev, api_port, ui_port: (api, fake(str(ui_port)))
launcher.webbrowser.open = lambda url: None
sys.exit(launcher.main(["--api-port", api_port, "--ui-port", ui_port]))
"""

LAUNCHER_SCRIPT = """
import sys
import sds.launcher as launcher
from pathlib import Path
from sds.launcher import Command

folder, api_port, ui_port = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

def fake(port):
    return Command("fake", [sys.executable, "-m", "http.server", port, "--bind", "127.0.0.1",
                            "--directory", str(folder)], {}, folder)

launcher.build_commands = lambda *, dev, api_port, ui_port: (fake(str(api_port)), fake(str(ui_port)))
launcher.webbrowser.open = lambda url: None
sys.exit(launcher.main(["--api-port", api_port, "--ui-port", ui_port]))
"""


def running_launcher(tmp_path, *, ignore_sigint: bool) -> tuple[subprocess.Popen, int, int]:
    """sds-app as a real process (fake servers); ``ignore_sigint`` as when started with ``&`` by a script."""
    import signal
    (tmp_path / "health").write_text("ok")
    (tmp_path / "_stcore").mkdir()
    (tmp_path / "_stcore" / "health").write_text("ok")
    api_port, ui_port = free_port(), free_port()
    preexec = (lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)) if ignore_sigint else None
    process = subprocess.Popen([sys.executable, "-c", LAUNCHER_SCRIPT, str(tmp_path), str(api_port), str(ui_port)],
                               stdout=subprocess.PIPE, text=True, preexec_fn=preexec)
    deadline = time.monotonic() + WAIT_SECONDS
    while not (port_in_use("127.0.0.1", api_port) and port_in_use("127.0.0.1", ui_port)):
        assert time.monotonic() < deadline and process.poll() is None, "sds-app did not start"
        time.sleep(0.1)
    assert "is running" in process.stdout.readline()
    return process, api_port, ui_port


@pytest.mark.parametrize("signal_name, ignore_sigint", [("SIGINT", True), ("SIGINT", False),
                                                        ("SIGTERM", False), ("SIGHUP", False)])
def test_a_signal_stops_both_processes(tmp_path, signal_name, ignore_sigint):
    import signal
    process, api_port, ui_port = running_launcher(tmp_path, ignore_sigint=ignore_sigint)
    try:
        process.send_signal(getattr(signal, signal_name))
        assert process.wait(WAIT_SECONDS) == 0
        deadline = time.monotonic() + WAIT_SECONDS
        while port_in_use("127.0.0.1", api_port) or port_in_use("127.0.0.1", ui_port):
            assert time.monotonic() < deadline, "a child outlived sds-app"
            time.sleep(0.1)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(WAIT_SECONDS)
        # Orphans of a failing run carry this test's folder in their command line: never leave them behind.
        subprocess.run(["pkill", "-f", f"--directory {tmp_path}"], check=False)


def test_closing_from_the_page_leaves_no_process_behind(tmp_path):
    # The same path on every system: an exit code read by Popen.poll, no signal (SIGTERM on Windows would end
    # sds-app without its finally and leave the UI behind).
    (tmp_path / "health").write_text("ok")
    (tmp_path / "_stcore").mkdir()
    (tmp_path / "_stcore" / "health").write_text("ok")
    api_port, ui_port = free_port(), free_port()
    process = subprocess.Popen([sys.executable, "-c", CLOSING_LAUNCHER_SCRIPT, str(tmp_path), str(api_port),
                                str(ui_port), CLOSING_SERVER, str(CLOSED_BY_USER_EXIT_CODE)],
                               stdout=subprocess.PIPE, text=True)
    try:
        assert process.wait(WAIT_SECONDS) == 0
        assert "Closed from the page" in process.stdout.read()
        deadline = time.monotonic() + WAIT_SECONDS
        while port_in_use("127.0.0.1", api_port) or port_in_use("127.0.0.1", ui_port):
            assert time.monotonic() < deadline, "a child outlived sds-app"
            time.sleep(0.1)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(WAIT_SECONDS)
        subprocess.run(["pkill", "-f", f"--directory {tmp_path}"], check=False)
