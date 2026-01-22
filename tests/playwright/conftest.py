#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Playwright configuration for UI testing."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page


@pytest.fixture(autouse=True)
def configure_playwright_screenshots(page: Page):
    """Configure Playwright to take screenshots on test failures."""
    # Create screenshots directory if it doesn't exist
    screenshots_dir = Path(__file__).parent / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    # Configure page to take screenshots on failure
    page.set_viewport_size({"width": 1920, "height": 1080})

    yield page

    # This runs after each test
    # Note: We'll use pytest-playwright's built-in screenshot on failure feature


@pytest.fixture
def page_with_screenshot(page: Page, request):
    """Enhanced page fixture that captures screenshots on test failure."""
    screenshots_dir = Path(__file__).parent / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    yield page

    # Take screenshot if test failed
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        screenshot_path = screenshots_dir / f"{request.node.name}_failure.png"
        page.screenshot(path=str(screenshot_path))
        print(f"Screenshot saved: {screenshot_path}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for screenshot capture."""
    # Execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # Set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"
    setattr(item, "rep_" + rep.when, rep)

    # Take screenshot on failure
    if rep.when == "call" and rep.failed:
        # Get page fixture if available
        if hasattr(item, "funcargs") and "page" in item.funcargs:
            page = item.funcargs["page"]
            screenshots_dir = Path(__file__).parent / "screenshots"
            screenshots_dir.mkdir(exist_ok=True)

            screenshot_path = screenshots_dir / f"{item.name}_failure_{call.when}.png"
            try:
                page.screenshot(path=str(screenshot_path))
                print(f"Test failure screenshot saved: {screenshot_path}")
            except Exception as e:
                print(f"Failed to take screenshot: {e}")


@pytest.fixture(scope="class")
def voila_server():
    """Launch a Voila server to serve the Cutana UI demo as a standalone web app."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    examples_dir = project_root / "examples"

    if not examples_dir.exists():
        pytest.skip("Examples directory not found")

    notebook_path = examples_dir / "cutana_ui_demo.ipynb"
    if not notebook_path.exists():
        pytest.skip("cutana_ui_demo.ipynb not found")

    # Use conda/mamba python to ensure we get the environment with voila
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == "cutana":
        try:
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            if conda_prefix:
                if os.name == "nt":  # Windows
                    python_executable = os.path.join(conda_prefix, "python.exe")
                else:  # Unix-like
                    python_executable = os.path.join(conda_prefix, "bin", "python")

                if not os.path.exists(python_executable):
                    python_executable = sys.executable
            else:
                python_executable = sys.executable
        except Exception:
            python_executable = sys.executable
    else:
        python_executable = sys.executable

    # Command to start Voila server
    cmd = [
        python_executable,
        "-m",
        "voila",
        str(notebook_path),
        "--port=8866",
        "--Voila.ip=127.0.0.1",
        "--no-browser",
        "--enable_nbextensions=True",
        "--VoilaConfiguration.theme=dark",
    ]

    # Set environment to include cutana in PYTHONPATH
    env = os.environ.copy()
    if os.name == "nt":  # Windows
        env["PYTHONPATH"] = f"{project_root};{env.get('PYTHONPATH', '')}"
    else:  # Unix-like
        env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    process = None
    try:
        # Start the Voila server
        process = subprocess.Popen(
            cmd,
            cwd=examples_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to start
        max_wait_time = 45  # Give time for conda environment
        wait_time = 0
        server_started = False

        while wait_time < max_wait_time:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(
                    f"Voila server failed to start:\nCommand: {' '.join(cmd)}\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                )

            # Try to connect to the server
            try:
                import requests

                response = requests.get("http://127.0.0.1:8866", timeout=2)
                if response.status_code == 200:
                    server_started = True
                    break
            except Exception:
                pass

            time.sleep(1)
            wait_time += 1

        if not server_started:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
            else:
                process.terminate()
                stdout, stderr = process.communicate()
            pytest.fail(
                f"Voila server did not start within {max_wait_time} seconds.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            )

        yield "http://127.0.0.1:8866"

    finally:
        # Clean up
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
