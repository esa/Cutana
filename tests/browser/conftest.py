#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Browser test infrastructure for Cutana UI via Voila + Playwright.

Key design decisions (see issue #14):
- Session-scoped Voila server: starts once, shared across all tests
- Dynamic port: avoids CI port conflicts
- subprocess.DEVNULL: prevents pipe-buffer deadlock on heavy kernel output
- Module-scoped pages: share Voila kernels across related tests in a module
- Video recording: retained only on failure for debugging
- wait_until="domcontentloaded": faster than "load" or "networkidle"

The notebook auto-selects the test CSV and triggers analysis so that
browser tests can verify both the start screen (with results) and
navigate to the main screen without manual file chooser interaction.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest
from playwright.sync_api import sync_playwright

from tests.browser.helpers import wait_for_analysis

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"


def _find_free_port():
    """Find a free TCP port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url, timeout=60):
    """Wait for Voila server to respond, using socket connect instead of requests."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _s(p):
    """Convert path to forward-slash string for embedding in notebook code."""
    return str(p).replace("\\", "/")


def _generate_notebook():
    """Generate a test notebook that auto-selects CSV and triggers analysis.

    The notebook starts the Cutana UI and programmatically triggers file
    selection + analysis with the test CSV, so the start screen loads
    with analysis results already populated. This avoids needing to
    interact with ipyfilechooser in Playwright.
    """
    test_csv = TEST_DATA_DIR / "euclid_cutana_catalogue_small.csv"

    # Start coverage inside the kernel so it tracks cutana/cutana_ui code.
    # The atexit handler saves .coverage.browser.* to PROJECT_ROOT on kernel exit.
    coverage_setup = f"""\
import coverage as _cov_mod, atexit as _atexit
_cov = _cov_mod.Coverage(source=["cutana", "cutana_ui"], data_file="{_s(PROJECT_ROOT)}/.coverage.browser", data_suffix=True)
_cov.start()
_atexit.register(lambda: (_cov.stop(), _cov.save()))
"""

    code = f"""\
{coverage_setup}
%load_ext autoreload
%autoreload 2

import sys, os
sys.path.insert(0, {_s(PROJECT_ROOT)!r})
os.chdir({_s(PROJECT_ROOT)!r})

import cutana_ui

app = cutana_ui.start(ui_scale=0.75)

# Auto-trigger file selection and analysis with test data
import asyncio

test_csv = {_s(test_csv)!r}
start_screen = app.container.children[0]
start_screen._on_file_selected(test_csv)
"""

    cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(keepends=True),
        }
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    fd, path = tempfile.mkstemp(suffix=".ipynb", prefix="cutana_browser_test_")
    with open(path, "w") as f:
        json.dump(notebook, f)
    return path


@pytest.fixture(scope="session")
def voila_port():
    """Allocate a free port for the Voila server."""
    return _find_free_port()


@pytest.fixture(scope="session")
def voila_server(voila_port):
    """Launch a session-scoped Voila server for browser testing.

    Uses subprocess.DEVNULL to prevent pipe-buffer deadlock when kernels
    produce heavy log output (>64KB).
    """
    notebook_path = _generate_notebook()

    cmd = [
        sys.executable,
        "-m",
        "voila",
        notebook_path,
        f"--port={voila_port}",
        "--Voila.ip=127.0.0.1",
        "--no-browser",
        "--enable_nbextensions=True",
        "--VoilaConfiguration.theme=dark",
        "--show_tracebacks=True",
    ]

    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT)
        + (";" if sys.platform == "win32" else ":")
        + os.environ.get("PYTHONPATH", ""),
    }

    popen_kwargs = {
        "cwd": str(PROJECT_ROOT),
        "env": env,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    # Use process group on Linux so SIGTERM reaches the kernel subprocess too
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(cmd, **popen_kwargs)

    url = f"http://127.0.0.1:{voila_port}"

    if not _wait_for_server(url, timeout=60):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        Path(notebook_path).unlink(missing_ok=True)
        pytest.fail(f"Voila server did not start within 60s on port {voila_port}")

    yield url

    # Cleanup: terminate Voila + kernel subprocess, then delete temp notebook
    if sys.platform != "win32":
        # Send SIGTERM to the whole process group so the kernel flushes coverage
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if sys.platform != "win32":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)
    # On Windows, Voila may still hold the file briefly after termination
    try:
        Path(notebook_path).unlink(missing_ok=True)
    except PermissionError:
        pass  # Best-effort cleanup; OS will reclaim temp file


@pytest.fixture(scope="session")
def _playwright_instance():
    """Session-scoped raw Playwright instance."""
    pw = sync_playwright().start()
    yield pw
    pw.stop()


@pytest.fixture(scope="session")
def _browser(_playwright_instance):
    """Session-scoped browser instance (chromium only for speed)."""
    browser = _playwright_instance.chromium.launch(headless=True)
    yield browser
    browser.close()


@pytest.fixture(scope="module")
def module_page(_browser, voila_server, tmp_path_factory):
    """Module-scoped page that shares a Voila kernel across related tests.

    Use this for read-only tests that inspect the UI without mutating state.
    Saves ~5s per test by avoiding new kernel starts.
    """
    video_dir = str(tmp_path_factory.mktemp("videos"))
    context = _browser.new_context(
        viewport={"width": 1280, "height": 720},
        record_video_dir=video_dir,
    )
    page = context.new_page()
    page.goto(voila_server, wait_until="domcontentloaded", timeout=60000)
    # Wait for widgets to render and analysis to complete
    wait_for_analysis(page)

    yield page

    page.close()
    context.close()


@pytest.fixture
def page(_browser, voila_server, request, tmp_path):
    """Function-scoped page for tests that mutate UI state.

    Video is recorded and retained only on failure.
    """
    video_dir = str(tmp_path / "videos")
    Path(video_dir).mkdir(exist_ok=True)

    context = _browser.new_context(
        viewport={"width": 1280, "height": 720},
        record_video_dir=video_dir,
    )
    page = context.new_page()

    yield page

    page.close()
    context.close()

    # Clean up videos for passing tests
    if not hasattr(request.node, "rep_call") or not request.node.rep_call.failed:
        for video_file in Path(video_dir).glob("*.webm"):
            video_file.unlink(missing_ok=True)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test result on the item for video retention logic."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
