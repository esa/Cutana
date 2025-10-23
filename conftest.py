#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Root conftest.py for Cutana project."""

import pytest


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for testing."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args, pytestconfig):
    """Configure browser launch arguments."""
    import os

    # Check if --headed was passed
    headed = pytestconfig.getoption("--headed", False)
    if headed:
        os.environ["HEADED"] = "true"

    return {
        **browser_type_launch_args,
        "headless": not headed,  # Invert headed flag
        "slow_mo": 500,  # Slow down operations for visibility
    }


@pytest.fixture(scope="session")
def browser_launch_args(browser_launch_args):
    """Additional browser launch configuration."""
    return {
        **browser_launch_args,
        "args": ["--start-maximized"],
    }


def pytest_configure(config):
    """Configure pytest for Playwright."""
    config.addinivalue_line("markers", "playwright: mark test as a Playwright end-to-end test")
