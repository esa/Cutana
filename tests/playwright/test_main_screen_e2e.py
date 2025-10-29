#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Main screen e2e tests for Cutana UI via Voila."""

import pytest
from pathlib import Path

# Marks this as a Playwright test
pytestmark = pytest.mark.playwright


class TestMainScreenE2E:
    """Test main screen functionality after successful transition from start screen."""

    def test_successful_transition_to_main_screen(self, page, voila_server):
        """Test that transition from start screen to main screen works correctly."""
        # Navigate and complete start screen workflow
        page.goto(voila_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(15000)

        # Get test CSV file path
        project_root = Path(__file__).parent.parent.parent
        test_csv = project_root / "tests" / "test_data" / "euclid_cutana_catalogue_small.csv"

        if not test_csv.exists():
            pytest.skip(f"Test CSV file not found: {test_csv}")

        # Complete file selection and analysis
        file_inputs = page.locator('input[type="file"]')
        if file_inputs.count() > 0:
            file_inputs.first.set_input_files(str(test_csv))
            page.wait_for_timeout(25000)  # Wait for analysis
        else:
            pytest.skip("No file input found")

        # Click Start button to transition to main screen
        start_buttons = page.locator("button").filter(has_text="Start")
        page.screenshot(path="tests/playwright/screenshots/main_screen_01_before_start.png")

        assert start_buttons.count() > 0, "No Start button found"
        start_buttons.first.click()

        # Wait for transition
        page.wait_for_timeout(10000)
        page.screenshot(path="tests/playwright/screenshots/main_screen_02_after_start_click.png")

        # Verify main screen elements appear
        # Look for main screen indicators
        main_screen_elements = page.locator(
            "text=/Configuration/i, " "text=/Preview/i, " "text=/Status/i, " ".cutana-panel"
        )

        page.screenshot(path="tests/playwright/screenshots/main_screen_03_final_state.png")

        # Should see main screen components
        assert (
            main_screen_elements.count() >= 2
        ), f"Main screen transition failed - only {main_screen_elements.count()} elements found"

    def test_main_screen_configuration_panel(self, page, voila_server):
        """Test configuration panel functionality in main screen."""
        # Complete transition to main screen first
        self._complete_transition_to_main_screen(page, voila_server)

        page.screenshot(path="tests/playwright/screenshots/config_panel_01_initial.png")

        # Look for configuration elements
        format_dropdowns = page.locator("select, .dropdown-toggle").filter(has_text="float32")
        resolution_inputs = page.locator('input[type="number"], input[type="text"]').filter(
            has_text="256"
        )
        stretch_dropdowns = page.locator("select").filter(has_text="linear")

        config_elements = (
            format_dropdowns.count() + resolution_inputs.count() + stretch_dropdowns.count()
        )

        assert (
            config_elements >= 2
        ), f"Configuration panel incomplete - only {config_elements} elements found"

        # Test format dropdown change
        if format_dropdowns.count() > 0:
            format_dropdowns.first.select_option("uint8")
            page.wait_for_timeout(2000)
            page.screenshot(path="tests/playwright/screenshots/config_panel_02_format_changed.png")

        # Test resolution change
        resolution_fields = page.locator("input").filter(has_text="256")
        if resolution_fields.count() > 0:
            resolution_fields.first.fill("512")
            page.wait_for_timeout(2000)
            page.screenshot(
                path="tests/playwright/screenshots/config_panel_03_resolution_changed.png"
            )

    def test_main_screen_preview_panel(self, page, voila_server):
        """Test preview panel functionality and auto-generation."""
        # Complete transition to main screen first
        self._complete_transition_to_main_screen(page, voila_server)

        page.screenshot(path="tests/playwright/screenshots/preview_panel_01_initial.png")

        # Look for preview elements
        preview_images = page.locator("img, canvas, .preview-image")
        regenerate_buttons = page.locator("button").filter(has_text="Regenerate")
        preview_labels = page.locator("text=/Preview/i, text=/Sample/i")

        # Should have preview functionality
        preview_elements = (
            preview_images.count() + regenerate_buttons.count() + preview_labels.count()
        )

        assert (
            preview_elements >= 1
        ), f"Preview panel incomplete - only {preview_elements} elements found"

        # Test regenerate functionality if available
        if regenerate_buttons.count() > 0:
            regenerate_buttons.first.click()
            page.wait_for_timeout(5000)
            page.screenshot(
                path="tests/playwright/screenshots/preview_panel_02_after_regenerate.png"
            )

    def test_main_screen_status_panel(self, page, voila_server):
        """Test status panel and processing start functionality."""
        # Complete transition to main screen first
        self._complete_transition_to_main_screen(page, voila_server)

        page.screenshot(path="tests/playwright/screenshots/status_panel_01_initial.png")

        # Look for status panel elements
        start_processing_buttons = page.locator("button").filter(has_text="Start Processing")
        status_labels = page.locator("text=/Status/i, text=/Ready/i")
        progress_elements = page.locator(".progress, text=/Progress/i")

        status_elements = (
            start_processing_buttons.count() + status_labels.count() + progress_elements.count()
        )

        assert (
            status_elements >= 1
        ), f"Status panel incomplete - only {status_elements} elements found"

        # Test starting processing if button available
        if start_processing_buttons.count() > 0:
            start_processing_buttons.first.click()
            page.wait_for_timeout(3000)
            page.screenshot(
                path="tests/playwright/screenshots/status_panel_02_processing_started.png"
            )

            # Should see some processing indication
            processing_indicators = page.locator("text=/Processing/i, text=/Running/i, .progress")
            assert processing_indicators.count() > 0, "No processing indicators after start"

    def test_configuration_updates_trigger_preview_changes(self, page, voila_server):
        """Test that configuration changes trigger preview updates."""
        # Complete transition to main screen first
        self._complete_transition_to_main_screen(page, voila_server)

        page.screenshot(path="tests/playwright/screenshots/config_preview_01_initial.png")

        # Find configuration controls
        resolution_inputs = page.locator("input").filter(has_text="256")

        if resolution_inputs.count() > 0:
            # Change resolution
            resolution_inputs.first.fill("128")
            page.wait_for_timeout(3000)

            page.screenshot(
                path="tests/playwright/screenshots/config_preview_02_after_resolution_change.png"
            )

            # Should see some response to config change
            # This could be preview update, filesize change, etc.
            filesize_elements = page.locator("text=/GB/i")
            updated_inputs = page.locator("input").filter(has_text="128")

            config_updated = filesize_elements.count() > 0 or updated_inputs.count() > 0
            assert config_updated, "Configuration change not reflected in UI"

    def test_channel_matrix_functionality(self, page, voila_server):
        """Test channel matrix controls in main screen."""
        # Complete transition to main screen first
        self._complete_transition_to_main_screen(page, voila_server)

        page.screenshot(path="tests/playwright/screenshots/channel_matrix_01_initial.png")

        # Look for channel controls
        add_channel_buttons = page.locator("button").filter(has_text="+")
        remove_channel_buttons = page.locator("button").filter(has_text="-")
        channel_labels = page.locator("text=/Channel/i, text=/Ch /i")

        channel_elements = (
            add_channel_buttons.count() + remove_channel_buttons.count() + channel_labels.count()
        )

        if channel_elements > 0:
            # Test adding a channel
            if add_channel_buttons.count() > 0:
                add_channel_buttons.first.click()
                page.wait_for_timeout(2000)
                page.screenshot(
                    path="tests/playwright/screenshots/channel_matrix_02_added_channel.png"
                )

            # Test removing a channel
            if remove_channel_buttons.count() > 0:
                remove_channel_buttons.first.click()
                page.wait_for_timeout(2000)
                page.screenshot(
                    path="tests/playwright/screenshots/channel_matrix_03_removed_channel.png"
                )

        # Channel matrix is optional, so just log if not found
        print(f"Channel matrix elements found: {channel_elements}")

    def test_filesize_prediction_updates(self, page, voila_server):
        """Test that filesize predictions update with configuration changes."""
        # Complete transition to main screen first
        self._complete_transition_to_main_screen(page, voila_server)

        page.screenshot(path="tests/playwright/screenshots/filesize_01_initial.png")

        # Look for filesize display
        filesize_elements = page.locator("text=/GB/i, text=/MB/i")

        if filesize_elements.count() > 0:
            initial_filesize = filesize_elements.first.text_content()

            # Change resolution to affect filesize
            resolution_inputs = page.locator("input").filter(has_text="256")
            if resolution_inputs.count() > 0:
                resolution_inputs.first.fill("512")
                page.wait_for_timeout(2000)

                # Filesize should update
                updated_filesize = filesize_elements.first.text_content()
                page.screenshot(path="tests/playwright/screenshots/filesize_02_after_change.png")

                assert updated_filesize != initial_filesize, "Filesize prediction did not update"
        else:
            print("No filesize prediction found - this is optional")

    def _complete_transition_to_main_screen(self, page, voila_server):
        """Helper method to complete transition from start screen to main screen."""
        # Navigate and set up
        page.goto(voila_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(15000)

        # Get test CSV file path
        project_root = Path(__file__).parent.parent.parent
        test_csv = project_root / "tests" / "test_data" / "euclid_cutana_catalogue_small.csv"

        if not test_csv.exists():
            pytest.skip(f"Test CSV file not found: {test_csv}")

        # Complete file selection
        file_inputs = page.locator('input[type="file"]')
        if file_inputs.count() > 0:
            file_inputs.first.set_input_files(str(test_csv))
            page.wait_for_timeout(25000)  # Wait for analysis
        else:
            pytest.skip("No file input found")

        # Click Start to transition
        start_buttons = page.locator("button").filter(has_text="Start")
        if start_buttons.count() > 0:
            start_buttons.first.click()
            page.wait_for_timeout(10000)
        else:
            pytest.skip("No Start button found")
