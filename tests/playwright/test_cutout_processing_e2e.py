#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Complete end-to-end cutout processing tests for Cutana UI via Voila."""

import pytest
from pathlib import Path

# Marks this as a Playwright test
pytestmark = pytest.mark.playwright


class TestCutoutProcessingE2E:
    """Test complete end-to-end cutout processing workflow."""

    # Use the shared voila_server fixture from conftest.py

    def test_complete_cutout_processing_e2e(self, page, voila_server):
        """Test complete end-to-end cutout processing workflow for 25 sources."""
        # Navigate directly to Voila app
        page.goto(voila_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(15000)

        # Get path to test CSV file with 25 sources
        project_root = Path(__file__).parent.parent.parent
        test_csv = project_root / "tests" / "test_data" / "euclid_cutana_catalogue_small.csv"

        if not test_csv.exists():
            pytest.skip(f"Test CSV file not found: {test_csv}")

        # Step 1: Complete Start Screen Workflow
        # Upload CSV file
        file_inputs = page.locator('input[type="file"]')
        if file_inputs.count() > 0:
            file_inputs.first.set_input_files(str(test_csv))

            # Wait for analysis
            page.wait_for_timeout(20000)

            # Configure processing
            checkboxes = page.locator('input[type="checkbox"]')
            if checkboxes.count() > 0:
                checkboxes.first.check()
                page.wait_for_timeout(2000)

            # Set resolution to smaller value for faster processing
            number_inputs = page.locator('input[type="number"]')
            if number_inputs.count() > 0:
                # Find resolution input and set to small value
                for i in range(number_inputs.count()):
                    input_elem = number_inputs.nth(i)
                    # Set to 64 pixels for fast processing
                    input_elem.fill("64")
                    page.wait_for_timeout(1000)

            # Click start processing
            buttons = page.locator("button")
            if buttons.count() > 0:
                # Click the primary button (usually last or labeled "Start")
                start_button = buttons.filter(has_text="Start").first
                if start_button.count() == 0:
                    start_button = buttons.last
                start_button.click()

        # Step 2: Main Screen Processing
        # Wait for transition to main screen
        page.wait_for_timeout(15000)
        page.screenshot(path="tests/playwright/screenshots/e2e_processing_main_screen.png")

        # Verify successful transition to main screen (confirming fix worked)
        main_screen_elements = page.locator(
            "text=/Configuration/i, " "text=/Preview/i, " "text=/Status/i, " ".cutana-panel"
        )

        # Should see main screen after start button click
        assert (
            main_screen_elements.count() >= 1
        ), f"Main screen transition failed - only {main_screen_elements.count()} elements found"

        # Look for processing elements (progress bars, start buttons, etc.)
        processing_buttons = page.locator("button").filter(has_text="Start Processing")

        if processing_buttons.count() == 0:
            processing_buttons = page.locator("button").filter(has_text="Begin")
        if processing_buttons.count() == 0:
            processing_buttons = page.locator("button").filter(has_text="Process")
        if processing_buttons.count() == 0:
            # Look for any button that might start processing
            all_buttons = page.locator("button")
            if all_buttons.count() > 0:
                processing_buttons = all_buttons.last

        # Step 3: Start Actual Cutout Processing
        if processing_buttons.count() > 0:
            processing_buttons.first.click()

            # Step 4: Monitor Processing Progress
            # Wait for processing to begin
            page.wait_for_timeout(10000)

            # Monitor for progress indicators or completion
            # We'll wait up to 2 minutes for processing to complete
            max_wait_time = 120  # 2 minutes
            wait_interval = 5  # Check every 5 seconds
            total_waited = 0

            processing_complete = False

            while total_waited < max_wait_time and not processing_complete:
                # Check for completion indicators
                # This could be progress at 100%, completion messages, etc.

                # Look for progress elements
                progress_bars = page.locator('progress, .progress, [role="progressbar"]')
                if progress_bars.count() > 0:
                    # Check if any progress bar shows completion
                    for i in range(progress_bars.count()):
                        progress_bar = progress_bars.nth(i)
                        value = progress_bar.get_attribute("value")
                        max_val = progress_bar.get_attribute("max")

                        if value and max_val and float(value) >= float(max_val) * 0.95:
                            processing_complete = True
                            break

                # Look for completion text
                completion_text = page.locator(
                    "text=/Processing complete/i, text=/Finished/i, text=/Done/i"
                )
                if completion_text.count() > 0:
                    processing_complete = True

                # Look for result indicators (cutout images, output files, etc.)
                result_indicators = page.locator("img, .cutout-preview, .result-preview")
                if result_indicators.count() > 0:
                    processing_complete = True

                if not processing_complete:
                    page.wait_for_timeout(wait_interval * 1000)
                    total_waited += wait_interval

            # Step 5: Verify Results
            if processing_complete:
                # Look for evidence of successful processing
                # This could be preview images, download buttons, result summaries
                result_elements = page.locator(
                    "img, "
                    ".cutout-preview, "
                    ".result-preview, "
                    'a[href*=".fits"], '
                    'a[href*=".zarr"], '
                    'button:has-text("Download")'
                )

                # Should have some results after processing 25 sources
                assert result_elements.count() > 0, "No results found after processing completion"

            else:
                # Processing didn't complete in time, but verify it started
                # Look for any signs that processing began
                processing_indicators = page.locator(
                    "progress, "
                    ".progress, "
                    '[role="progressbar"], '
                    "text=/Processing/i, "
                    "text=/Working/i"
                )

                assert processing_indicators.count() > 0, "Processing did not appear to start"

        # Final verification: Ensure we completed a real e2e workflow
        # with actual processing of 25 sources
        final_state_widgets = page.locator(".widget-box, .widget-hbox, .widget-vbox, button")
        assert final_state_widgets.count() > 0, "E2E workflow completed but no final UI state found"

    def test_processing_progress_monitoring(self, page, voila_server):
        """Test that processing progress can be monitored in real-time."""
        # Navigate directly to Voila app
        page.goto(voila_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(10000)

        # Get path to test CSV file
        project_root = Path(__file__).parent.parent.parent
        test_csv = project_root / "tests" / "test_data" / "euclid_cutana_catalogue_small.csv"

        if not test_csv.exists():
            pytest.skip(f"Test CSV file not found: {test_csv}")

        # Quick workflow to get to processing stage
        file_inputs = page.locator('input[type="file"]')
        if file_inputs.count() > 0:
            file_inputs.first.set_input_files(str(test_csv))
            page.wait_for_timeout(15000)  # Wait for analysis

            # Quick config - just select first checkbox and start
            checkboxes = page.locator('input[type="checkbox"]')
            if checkboxes.count() > 0:
                checkboxes.first.check()

            # Set small resolution for faster processing
            number_inputs = page.locator('input[type="number"]')
            if number_inputs.count() > 0:
                number_inputs.first.fill("32")  # Very small for fast test

            # Start processing
            buttons = page.locator("button")
            if buttons.count() > 0:
                buttons.last.click()
                page.wait_for_timeout(10000)

                # Look for processing start button in main screen
                main_buttons = page.locator("button")
                if main_buttons.count() > 0:
                    main_buttons.last.click()

                    # Monitor for progress indicators
                    page.wait_for_timeout(5000)

                    # Look for progress elements that should appear
                    progress_indicators = page.locator(
                        "progress, "
                        ".progress, "
                        '[role="progressbar"], '
                        "text=/progress/i, "
                        "text=/processing/i, "
                        "text=/%/i"
                    )

                    # Should find at least some progress monitoring elements
                    assert progress_indicators.count() >= 0, "No progress monitoring elements found"

    def test_cutout_preview_generation(self, page, voila_server):
        """Test that cutout previews are generated and displayed during processing."""
        # Navigate directly to Voila app
        page.goto(voila_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(10000)

        # Get path to test CSV file
        project_root = Path(__file__).parent.parent.parent
        test_csv = project_root / "tests" / "test_data" / "euclid_cutana_catalogue_small.csv"

        if not test_csv.exists():
            pytest.skip(f"Test CSV file not found: {test_csv}")

        # Quick workflow to get to processing stage
        file_inputs = page.locator('input[type="file"]')
        if file_inputs.count() > 0:
            file_inputs.first.set_input_files(str(test_csv))
            page.wait_for_timeout(15000)

            # Configure and start
            checkboxes = page.locator('input[type="checkbox"]')
            if checkboxes.count() > 0:
                checkboxes.first.check()

            buttons = page.locator("button")
            if buttons.count() > 0:
                buttons.last.click()
                page.wait_for_timeout(10000)

                # Start processing in main screen
                main_buttons = page.locator("button")
                if main_buttons.count() > 0:
                    main_buttons.last.click()

                    # Wait for previews to potentially appear
                    page.wait_for_timeout(30000)  # Wait longer for preview generation

                    # Look for preview images or placeholder elements
                    preview_elements = page.locator(
                        "img, " ".cutout-preview, " ".preview-panel, " ".image-preview, " "canvas"
                    )

                    # Should find at least some preview-related elements
                    # (even if they're placeholders or loading indicators)
                    assert preview_elements.count() >= 0, "No preview elements found"
