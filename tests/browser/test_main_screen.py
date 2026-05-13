#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Main screen browser tests for Cutana UI.

Uses a module-scoped page that navigates from start screen to main screen
once, then all tests inspect the main screen state (read-only).
"""

import pytest

from tests.browser.helpers import wait_for_analysis, wait_for_main_screen

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def main_screen_page(_browser, voila_server, tmp_path_factory):
    """Module-scoped page that navigates to the main screen.

    Starts a fresh Voila kernel, waits for analysis to complete,
    then clicks Start to navigate to the main screen.
    """
    video_dir = str(tmp_path_factory.mktemp("videos"))
    context = _browser.new_context(
        viewport={"width": 1280, "height": 720},
        record_video_dir=video_dir,
    )
    page = context.new_page()
    page.goto(voila_server, wait_until="domcontentloaded", timeout=60000)

    # Wait for analysis to complete, then click Start
    wait_for_analysis(page)
    page.locator("button").filter(has_text="Start").first.click()
    wait_for_main_screen(page)

    yield page

    page.close()
    context.close()


def test_main_screen_transition(main_screen_page):
    """Test that clicking Start navigates to the main screen."""
    config_text = main_screen_page.locator("text=/Configuration/i")
    preview_text = main_screen_page.locator("text=/Preview/i")
    status_text = main_screen_page.locator("text=/Status/i")

    total = config_text.count() + preview_text.count() + status_text.count()
    assert total >= 2, f"Main screen not loaded - only {total} panel headers found"


def test_configuration_panel(main_screen_page):
    """Test that configuration panel renders with dropdowns."""
    dropdowns = main_screen_page.locator("select")
    assert dropdowns.count() >= 1, "No configuration dropdowns found"

    resolution = main_screen_page.locator("text=/Resolution/i")
    assert resolution.count() > 0, "Resolution control not found"


def test_stretch_dropdown_value(main_screen_page):
    """Test that stretch dropdown shows 'linear' (the fix from the old tests)."""
    linear_text = main_screen_page.locator("text=/linear/i")
    assert linear_text.count() > 0, "Stretch value 'linear' not found in main screen"


def test_preview_panel(main_screen_page):
    """Test that preview panel renders."""
    preview = main_screen_page.locator("text=/Preview/i")
    assert preview.count() > 0, "Preview panel header not found"


def test_status_panel(main_screen_page):
    """Test that status panel renders."""
    status = main_screen_page.locator("text=/Status/i")
    assert status.count() > 0, "Status panel header not found"


def test_start_processing_button(main_screen_page):
    """Test that Start Processing button is present."""
    processing_btn = main_screen_page.locator("button").filter(has_text="Start Cutout Creation")
    if processing_btn.count() == 0:
        processing_btn = main_screen_page.locator("button").filter(has_text="Start")
    assert processing_btn.count() > 0, "Start Processing button not found"


def test_channel_buttons(main_screen_page):
    """Test that channel add/remove buttons are present on main screen."""
    add_btn = main_screen_page.locator("button").filter(has_text="Add Channel")
    remove_btn = main_screen_page.locator("button").filter(has_text="Remove Channel")

    assert add_btn.count() > 0, "Add Channel button not found"
    assert remove_btn.count() > 0, "Remove Channel button not found"


def test_filesize_prediction(main_screen_page):
    """Test that filesize prediction is displayed on main screen."""
    gb = main_screen_page.locator("text=/GB/i")
    mb = main_screen_page.locator("text=/MB/i")

    assert gb.count() > 0 or mb.count() > 0, "Filesize prediction not displayed"


def test_header_present_on_main_screen(main_screen_page):
    """Test that header/logo persists on main screen."""
    svg = main_screen_page.locator("svg")
    cutana_text = main_screen_page.locator("text=/CUTANA/i")

    assert svg.count() > 0 or cutana_text.count() > 0, "Header/logo not found on main screen"
