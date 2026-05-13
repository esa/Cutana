#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Main screen interaction browser tests for Cutana UI.

Tests that exercise interactive main screen features:
- Help panel toggle (show/hide)
- Configuration change triggering preview update

Uses a module-scoped page for read-only tests (help toggle) and
function-scoped page for tests that mutate state (config change).
"""

import pytest

from tests.browser.helpers import wait_for_analysis, wait_for_main_screen

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def main_screen_module_page(_browser, voila_server, tmp_path_factory):
    """Module-scoped page already navigated to the main screen.

    Suitable for read-only tests that inspect UI state without mutation.
    """
    video_dir = str(tmp_path_factory.mktemp("videos"))
    context = _browser.new_context(
        viewport={"width": 1280, "height": 720},
        record_video_dir=video_dir,
    )
    page = context.new_page()
    page.goto(voila_server, wait_until="domcontentloaded", timeout=60000)

    # Wait for start screen analysis, then navigate to main screen
    wait_for_analysis(page)
    page.locator("button").filter(has_text="Start").first.click()
    wait_for_main_screen(page)

    yield page

    page.close()
    context.close()


# ---------------------------------------------------------------------------
# Help panel toggle (read-only, uses module-scoped page)
# ---------------------------------------------------------------------------


def test_help_panel_toggle_show(main_screen_module_page):
    """Click Help button and verify help content appears."""
    pg = main_screen_module_page

    # The Help button lives in the header
    help_btn = pg.locator("button").filter(has_text="Help")
    assert help_btn.count() > 0, "Help button not found on main screen"

    # Click Help to open the panel
    help_btn.first.click()

    # The help panel should now be visible — look for "Cutana Help" title
    # or the "Back" close button that only appears inside the help panel
    cutana_help = pg.locator("text=/Cutana Help/i")
    cutana_help.first.wait_for(state="visible", timeout=10000)
    assert cutana_help.count() > 0, "Help panel title 'Cutana Help' not found after clicking Help"

    # The help button text should change to "Close Help"
    close_help_btn = pg.locator("button").filter(has_text="Close Help")
    assert close_help_btn.count() > 0, "Help button did not change to 'Close Help'"

    # The Back button inside the help panel should also be present
    back_btn = pg.locator("button").filter(has_text="Back")
    assert back_btn.count() > 0, "Back button not found in help panel"


def test_help_panel_toggle_hide(main_screen_module_page):
    """Close the help panel and verify preview panel returns."""
    pg = main_screen_module_page

    # Ensure help panel is currently open (from previous test)
    # If not open, open it first
    close_help_btn = pg.locator("button").filter(has_text="Close Help")
    if close_help_btn.count() == 0:
        help_btn = pg.locator("button").filter(has_text="Help")
        help_btn.first.click()
        pg.locator("text=/Cutana Help/i").first.wait_for(state="visible", timeout=10000)

    # Click "Close Help" to hide the help panel
    close_help_btn = pg.locator("button").filter(has_text="Close Help")
    close_help_btn.first.click()

    # Help panel content should disappear
    cutana_help = pg.locator("text=/Cutana Help/i")
    cutana_help.first.wait_for(state="hidden", timeout=10000)

    # The button should revert to "Help"
    help_btn = pg.locator("button").filter(has_text="Help")
    assert help_btn.count() > 0, "Help button did not revert to 'Help' after closing"

    # Preview panel should be back — look for "Preview" text
    preview = pg.locator("text=/Preview/i")
    assert preview.count() > 0, "Preview panel not restored after closing help"


# ---------------------------------------------------------------------------
# Config panel toggle (read-only, uses module-scoped page)
# ---------------------------------------------------------------------------


def test_config_panel_toggle_show(main_screen_module_page):
    """Click Config button and verify config content appears."""
    pg = main_screen_module_page

    config_btn = pg.locator("button").filter(has_text="Config")
    assert config_btn.count() > 0, "Config button not found"

    config_btn.first.click()

    # The config panel should become visible
    current_config_title = pg.locator("text=/Configuration Data/i")
    current_config_title.first.wait_for(state="visible", timeout=10000)
    assert current_config_title.count() > 0, "Config panel title not found after clicking show"

    # The button text should change to "Close Config"
    hide_config_btn = pg.locator("button").filter(has_text="Close Config")
    assert hide_config_btn.count() > 0, "Button did not change to 'Close Config'"


def test_config_panel_toggle_hide(main_screen_module_page):
    """Close the config panel and verify preview panel returns."""
    pg = main_screen_module_page

    # Ensure config panel is currently open
    hide_config_btn = pg.locator("button").filter(has_text="Close Config")
    if hide_config_btn.count() == 0:
        config_btn = pg.locator("button").filter(has_text="Config")
        config_btn.first.click()
        pg.locator("text=/Configuration Data/i").first.wait_for(state="visible", timeout=10000)

    # Click "Close Config"
    hide_config_btn = pg.locator("button").filter(has_text="Close Config")
    hide_config_btn.first.click()

    # Config panel content should disappear
    current_config_title = pg.locator("text=/Configuration Data/i")
    current_config_title.first.wait_for(state="hidden", timeout=10000)

    # Button revert
    config_btn = pg.locator("button").filter(has_text="Config")
    assert config_btn.count() > 0, "Button did not revert to 'Config'"

    # Preview restored
    preview = pg.locator("text=/Cutout Preview/i")
    assert preview.count() > 0, "Preview panel not restored after closing config"


def test_config_and_help_panel_interaction(main_screen_module_page):
    """Verify that opening Help while Config is open swaps them correctly."""
    pg = main_screen_module_page

    # Start by opening config
    config_btn = pg.locator("button").filter(has_text="Config").first
    config_btn.click()
    pg.locator("text=/Configuration Data/i").first.wait_for(state="visible", timeout=10000)

    # Now click Help
    help_btn = pg.locator("button").filter(has_text="Help")
    help_btn.first.click()

    # Help should appear, Config should disappear
    pg.locator("text=/Cutana Help/i").first.wait_for(state="visible", timeout=10000)
    pg.locator("text=/Current Configuration/i").first.wait_for(state="hidden", timeout=10000)

    # Hide help to cleanup
    close_help_btn = pg.locator("button").filter(has_text="Close Help")
    close_help_btn.first.click()
    pg.locator("text=/Cutana Help/i").first.wait_for(state="hidden", timeout=10000)


# ---------------------------------------------------------------------------
# Configuration change triggers preview update (mutating, function-scoped page)
# ---------------------------------------------------------------------------


def test_config_change_triggers_preview_update(page, voila_server):
    """Change a configuration value on the main screen and verify no crash."""
    page.goto(voila_server, wait_until="domcontentloaded", timeout=60000)
    wait_for_analysis(page)

    # Navigate to main screen
    page.locator("button").filter(has_text="Start").first.click()
    wait_for_main_screen(page)

    # Verify we are on the main screen
    assert page.locator("text=/Configuration/i").count() > 0, "Not on main screen"

    # Find the output format dropdown (<select>) — it has "zarr" and "fits" options
    output_select = page.locator("select").filter(has_text="zarr")
    if output_select.count() > 0:
        # Change the output format from "zarr" to "fits"
        output_select.first.select_option("fits")

        # Give the UI a moment to process the config change callback
        page.wait_for_timeout(1000)

        # Verify the dropdown value changed
        selected_value = output_select.first.input_value()
        assert selected_value == "fits", f"Output format not changed, got: {selected_value}"

        # Verify the page is still functional (no crash) — panels still visible
        assert page.locator("text=/Configuration/i").count() > 0, (
            "Configuration panel gone after config change"
        )
        assert (
            page.locator("text=/Preview/i").count() > 0
            or page.locator("text=/Cutout Preview/i").count() > 0
        ), "Preview panel gone after config change"
    else:
        # Fallback: try changing the resolution input
        resolution_inputs = page.locator('input[type="number"]')
        assert resolution_inputs.count() > 0, "No number inputs found to test config change"

        # Change resolution value
        first_input = resolution_inputs.first
        first_input.fill("64")
        first_input.press("Enter")

        page.wait_for_timeout(1000)

        # Verify UI is still functional
        assert page.locator("text=/Configuration/i").count() > 0, (
            "Configuration panel gone after config change"
        )
