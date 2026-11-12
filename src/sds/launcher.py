"""``sds-app [--dev]``: start the API and the UI together, on this machine only, and stop both with Ctrl+C.

Order: free ports → API → wait for ``/health`` → UI → wait for Streamlit → open the browser. If either process
ends by itself, the other one is stopped too: stopping the API drops every dataset and model (nothing is on disk).
The "Close" button of the page ends the API with ``CLOSED_BY_USER_EXIT_CODE``: a wanted stop, not a failure.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from sds.config import CLOSABLE_ENV, CLOSED_BY_USER_EXIT_CODE, MAX_UPLOAD_MB, PROJECT_ROOT
from sds.launcher_config import (
    API_HEALTH_PATH,
    API_HOST,
    API_SCRIPT,
    API_START_SECONDS,
    API_URL_ENV,
    DEFAULT_API_PORT,
    DEFAULT_UI_PORT,
    HEALTH_REQUEST_SECONDS,
    MAX_PORT,
    POLL_SECONDS,
    STOP_SECONDS,
    UI_HEALTH_PATH,
    UI_HOST,
    UI_SCRIPT,
    UI_START_SECONDS,
)

DEV_MODE_ENV = "SDS_DEV_MODE"        # read by the API (api.api_config); repeated here to keep the launcher light


@dataclass(frozen=True)
class Command:
    """One process to start.

    Attributes:
        name: Shown in the messages.
        args: Program and arguments.
        env: Variables added to (or replacing) the current environment.
        cwd: Working directory.
    """

    name: str
    args: list[str]
    env: dict[str, str]
    cwd: Path


def build_commands(*, dev: bool, api_port: int, ui_port: int, python: str = sys.executable) -> tuple[Command, Command]:
    """The API and UI commands, run with the same Python (so the same virtual environment).

    ``SDS_DEV_MODE`` is always set ("1" or "0"), so a value exported in the shell cannot turn it on without ``--dev``.
    ``SDS_CLOSABLE`` tells the API that this launcher stops the UI when the API closes itself ("Close" button).
    """
    api = Command("API", [python, "-m", "uvicorn", API_SCRIPT, "--host", API_HOST, "--port", str(api_port)],
                  {DEV_MODE_ENV: "1" if dev else "0", CLOSABLE_ENV: "1"}, PROJECT_ROOT)
    ui = Command("UI", [python, "-m", "streamlit", "run", UI_SCRIPT,
                        "--server.address", UI_HOST,
                        "--server.port", str(ui_port),
                        "--server.headless", "true",                 # no e-mail prompt; the launcher opens the browser
                        "--server.maxUploadSize", str(MAX_UPLOAD_MB),
                        "--browser.gatherUsageStats", "false",       # no usage statistics leave the machine
                        "--client.toolbarMode", "viewer"],          # no "Deploy" button: the app is local
                 {API_URL_ENV: f"http://{API_HOST}:{api_port}"}, PROJECT_ROOT)
    return api, ui


def port_number(text: str) -> int:
    """argparse type: a TCP port."""
    value = int(text)
    if not 1 <= value <= MAX_PORT:
        raise argparse.ArgumentTypeError(f"{value} is not a port (1-{MAX_PORT})")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line options (see ``--help``)."""
    parser = argparse.ArgumentParser(prog="sds-app", description="Start the Synthetic Data Sandbox (API + UI).")
    parser.add_argument("--dev", action="store_true", help="developer mode: panels on how the engine works")
    parser.add_argument("--api-port", type=port_number, default=DEFAULT_API_PORT, help="port of the API")
    parser.add_argument("--ui-port", type=port_number, default=DEFAULT_UI_PORT, help="port of the UI")
    args = parser.parse_args(argv)
    if args.api_port == args.ui_port:
        parser.error("the API and the UI need two different ports")
    return args


def port_in_use(host: str, port: int) -> bool:
    """True when a program already listens on ``host:port``.

    A connection attempt, not a ``bind``: right after a stop the old connections linger in TIME_WAIT and a plain
    ``bind`` fails, although uvicorn and Streamlit (``SO_REUSEADDR``) would start fine.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(HEALTH_REQUEST_SECONDS)
        return probe.connect_ex((socket.gethostbyname(host), port)) == 0


def start(command: Command) -> subprocess.Popen:
    """Start one process."""
    return subprocess.Popen(command.args, cwd=command.cwd, env={**os.environ, **command.env})


def wait_until_ready(url: str, process: subprocess.Popen, timeout: float) -> bool:
    """Poll ``url`` until it answers 200; False if the process ends first or the timeout passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=HEALTH_REQUEST_SECONDS) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(POLL_SECONDS)
    return False


def stop(processes: list[subprocess.Popen], timeout: float = STOP_SECONDS) -> None:
    """Terminate the running processes, wait up to ``timeout``, then kill the ones still alive."""
    running = [p for p in processes if p.poll() is None]
    for process in running:
        process.terminate()
    deadline = time.monotonic() + timeout
    for process in running:
        try:
            process.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def supervise(processes: list[subprocess.Popen], *, poll: float = POLL_SECONDS) -> int:
    """Wait until one process ends; return its exit code, 1 if it ended "cleanly" (the app no longer works)."""
    while True:
        for process in processes:
            code = process.poll()
            if code is not None:
                return code or 1
        time.sleep(poll)


def interrupt(signum: int, frame) -> None:
    """Signal handler: stop the way Ctrl+C does, so ``main`` runs its ``finally`` and stops both processes."""
    raise KeyboardInterrupt


def install_signal_handlers() -> None:
    """Whatever stops ``sds-app`` stops the API too: otherwise it would live on, real data in memory.

    SIGINT is restored even when inherited as ignored (a command started with ``&`` by a script); SIGTERM and
    SIGHUP (``kill``, closed terminal) would end Python without running ``finally``. Only in the main thread,
    the one Python delivers signals to.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGINT, signal.default_int_handler)
    for name in ("SIGTERM", "SIGHUP"):                 # SIGHUP does not exist on Windows
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)


def fail(message: str) -> int:
    """Print ``message`` on stderr; the exit code of a failed start (1)."""
    print(f"sds-app: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``sds-app``; the exit code of the command."""
    args = parse_args(argv)
    install_signal_handlers()
    for host, port in ((API_HOST, args.api_port), (UI_HOST, args.ui_port)):
        if port_in_use(host, port):
            return fail(f"Port {port} is in use: is the app already running? "
                        "Close it, or choose other ports with --api-port / --ui-port.")
    api_command, ui_command = build_commands(dev=args.dev, api_port=args.api_port, ui_port=args.ui_port)
    processes: list[subprocess.Popen] = []
    try:
        api = start(api_command)
        processes.append(api)
        if not wait_until_ready(f"http://{API_HOST}:{args.api_port}{API_HEALTH_PATH}", api, API_START_SECONDS):
            return fail("the API did not start (see the messages above).")
        ui = start(ui_command)
        processes.append(ui)
        ui_url = f"http://{UI_HOST}:{args.ui_port}"
        if not wait_until_ready(f"{ui_url}{UI_HEALTH_PATH}", ui, UI_START_SECONDS):
            return fail("the UI did not start (see the messages above).")
        mode = " in developer mode" if args.dev else ""
        print(f"Synthetic Data Sandbox is running{mode} at {ui_url} (press Ctrl+C to stop).", flush=True)
        webbrowser.open(ui_url)
        code = supervise(processes)
        if api.poll() == CLOSED_BY_USER_EXIT_CODE:                # the "Close" button: only the API may close
            print("Closed from the page: the data in memory is discarded.", flush=True)
            return 0
        return fail(f"a process stopped unexpectedly (exit code {code}); the app was closed.")
    except KeyboardInterrupt:
        print("Stopping the Synthetic Data Sandbox: the data in memory is discarded.", flush=True)
        return 0
    finally:
        stop(processes)
