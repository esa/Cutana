#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end processing test for Cutana UI.

This test runs the complete workflow: start screen -> main screen -> processing.
Marked as 'slow' so it is excluded from CI (run locally with -m browser).
"""

import pytest

from tests.browser.helpers import wait_for_analysis, wait_for_main_screen

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_complete_processing_workflow(page, voila_server):
    """Test the complete cutout processing workflow end-to-end.

    Flow: auto-analyzed start screen -> click Start -> main screen ->
    click Start Processing -> monitor progress.
    """
    page.goto(voila_server, wait_until="domcontentloaded", timeout=60000)

    # Wait for analysis to complete
    wait_for_analysis(page)

    # Click Start to navigate to main screen
    start_btn = page.locator("button").filter(has_text="Start")
    assert start_btn.count() > 0, "Start button not found after analysis"
    start_btn.first.click()
    wait_for_main_screen(page)

    # Should be on main screen now
    config_count = page.locator("text=/Configuration/i").count()
    preview_count = page.locator("text=/Preview/i").count()
    status_count = page.locator("text=/Status/i").count()
    assert config_count + preview_count + status_count >= 2, "Did not navigate to main screen"

    # Click Start Processing
    processing_btn = page.locator("button").filter(has_text="Start Cutout Creation")
    if processing_btn.count() == 0:
        processing_btn = page.locator("button").filter(has_text="Start")
    if processing_btn.count() > 0:
        processing_btn.first.click()
        # Wait for processing to begin — look for Stop button or progress bar
        page.locator("button:has-text('Stop'), progress").first.wait_for(
            state="visible", timeout=30000
        )

        # Look for processing indicators
        processing = page.locator("text=/Processing/i").count()
        running = page.locator("text=/Running/i").count()
        stop = page.locator("text=/Stop/i").count()
        progress = page.locator("progress").count()
        assert processing + running + stop + progress > 0, "No processing indicators after start"
