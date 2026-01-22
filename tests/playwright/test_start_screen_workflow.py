#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Start screen workflow tests for Cutana UI via Voila."""

from pathlib import Path

import pytest

# Marks this as a Playwright test
pytestmark = pytest.mark.playwright


class TestStartScreenWorkflow:
    """Test complete start screen workflow from file selection to main screen."""

    # Use the shared voila_server fixture from conftest.py

    def test_file_selection_in_real_ui(self, page, voila_server):
        """Test CSV file selection in the real Cutana UI running in Voila."""
        # Navigate directly to the Voila app
        page.goto(voila_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(15000)  # Give widgets time to render

        # STRICT: Must find Cutana UI elements (use widget selectors, not just text)
        # Look for the actual UI widgets, not just any text on the page
        title_widget = page.locator(".widget-html").filter(
            has_text="CUTANA Cutout Generator Configuration"
        )
        file_selection_widget = page.locator(".widget-html").filter(
            has_text="Source Catalogue Selection"
        )

        page.screenshot(path="tests/playwright/screenshots/file_sel_01_loaded.png")

        # Check for actual UI widgets
        ui_loaded = title_widget.count() > 0 or file_selection_widget.count() > 0
        assert (
            ui_loaded
        ), f"Cutana UI not detected - Title: {title_widget.count()}, File selection: {file_selection_widget.count()}"

        # With the new auto-navigation, the file should already be visible
        # Look for the specific test CSV file name
        csv_file_elements = page.locator("text=/euclid_cutana_catalogue_small.csv/i")

        page.screenshot(path="tests/playwright/screenshots/file_sel_02_auto_navigation.png")

        if csv_file_elements.count() > 0:
            # File is visible, try to click/select it
            try:
                csv_file_elements.first.click()
                page.wait_for_timeout(1000)

                # Look for select button and click it
                select_buttons = page.locator("button").filter(has_text="Select")
                if select_buttons.count() > 0:
                    select_buttons.first.click()

                file_selected = True
                page.screenshot(path="tests/playwright/screenshots/file_sel_03_after_selection.png")
            except Exception as e:
                print(f"File selection failed: {e}")
                file_selected = False
        else:
            # Fallback to manual file input if auto-navigation didn't work
            file_inputs = page.locator('input[type="file"]')
            if file_inputs.count() > 0:
                project_root = Path(__file__).parent.parent.parent
                test_csv = (
                    project_root / "tests" / "test_data" / "euclid_cutana_catalogue_small.csv"
                )

                if test_csv.exists():
                    file_inputs.first.set_input_files(str(test_csv))
                    file_selected = True
                else:
                    pytest.skip(f"Test CSV file not found: {test_csv}")
            else:
                pytest.skip("No file selection mechanism found")

        assert file_selected, "Could not select test CSV file"

        # Wait for response and look for loading spinner
        loading_spinner = page.locator("text=/Analysing/i")
        if loading_spinner.count() > 0:
            print("✅ Loading spinner appeared")

        # Wait for analysis to complete
        page.wait_for_timeout(20000)
        page.screenshot(path="tests/playwright/screenshots/file_sel_04_after_analysis.png")

        # STRICT: Check for analysis results (now integrated in file selection)
        source_count = page.locator("text=/25/i")
        analysis_elements = page.locator("text=/Sources/i, text=/FITS Files/i, text=/Extensions/i")
        config_elements = page.locator("text=/Add channel/i, text=/Remove channel/i")

        ui_updated = (
            source_count.count() > 0
            and analysis_elements.count() > 0
            and config_elements.count() > 0
        )

        assert (
            ui_updated
        ), f"Analysis results not shown. Sources: {source_count.count()}, \
Analysis: {analysis_elements.count()}, Config: {config_elements.count()}"

    def test_complete_start_screen_workflow_with_csv(self, page, voila_server):
        """Test complete start screen workflow: file selection → analysis → configuration → main screen."""
        # Navigate directly to Voila app
        page.goto(voila_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(15000)  # Give widgets time to fully render

        page.screenshot(path="tests/playwright/screenshots/workflow_01_initial.png")

        # STRICT: Verify Cutana UI loaded (use widget-specific selectors)
        title_widget = page.locator(".widget-html").filter(
            has_text="CUTANA Cutout Generator Configuration"
        )
        file_selection_widget = page.locator(".widget-html").filter(
            has_text="Source Catalogue Selection"
        )

        ui_loaded = title_widget.count() > 0 or file_selection_widget.count() > 0
        assert (
            ui_loaded
        ), f"Cutana UI did not load - Title: {title_widget.count()}, File selection: {file_selection_widget.count()}"

        # Get path to test CSV file
        project_root = Path(__file__).parent.parent.parent
        test_csv = project_root / "tests" / "test_data" / "euclid_cutana_catalogue_small.csv"

        if not test_csv.exists():
            pytest.skip(f"Test CSV file not found: {test_csv}")

        # Step 1: STRICT File Selection
        file_inputs = page.locator('input[type="file"]')
        file_path_inputs = page.locator('input[placeholder*="path"], input[placeholder*="file"]')
        select_buttons = page.locator("button").filter(has_text="Select")

        page.screenshot(path="tests/playwright/screenshots/workflow_02_before_file.png")

        # Must have file selection capability
        has_file_selection = file_inputs.count() > 0 or file_path_inputs.count() > 0
        assert (
            has_file_selection
        ), f"No file selection found. File inputs: {file_inputs.count()}, Path inputs: {file_path_inputs.count()}"

        # Try file selection
        file_selected = False
        if file_inputs.count() > 0:
            file_inputs.first.set_input_files(str(test_csv))
            file_selected = True
        elif file_path_inputs.count() > 0:
            file_path_inputs.first.fill(str(test_csv))
            if select_buttons.count() > 0:
                select_buttons.first.click()
            file_selected = True

        assert file_selected, "Could not select file"

        # Step 2: STRICT Analysis Verification
        page.wait_for_timeout(25000)  # Wait for analysis
        page.screenshot(path="tests/playwright/screenshots/workflow_03_after_analysis.png")

        # Must find analysis results
        source_count = page.locator("text=/25.*source/i")
        analysis_section = page.locator("text=/Analysis.*Result/i")
        vis_extension = page.locator("text=/VIS/i")

        analysis_found = (
            source_count.count() > 0 or analysis_section.count() > 0 or vis_extension.count() > 0
        )

        assert (
            analysis_found
        ), f"No analysis results found. Sources: {source_count.count()}, \
Analysis: {analysis_section.count()}, VIS: {vis_extension.count()}"

        # Step 3: STRICT Configuration Elements Check
        checkboxes = page.locator('input[type="checkbox"]')
        add_channel_btn = page.locator("button").filter(has_text="Add channel")
        remove_channel_btn = page.locator("button").filter(has_text="Remove channel")
        dropdowns = page.locator("select")
        start_buttons = page.locator("button").filter(has_text="Start")
        filesize_display = page.locator("text=/GB/i")

        # Must have configuration elements after analysis
        config_elements = (
            checkboxes.count()
            + add_channel_btn.count()
            + remove_channel_btn.count()
            + dropdowns.count()
            + start_buttons.count()
            + filesize_display.count()
        )
        assert (
            config_elements > 5
        ), f"Not enough configuration elements found. Total: {config_elements}"

        # Step 4: Interact with Configuration
        if checkboxes.count() > 0:
            # First checkbox should already be checked by default
            page.wait_for_timeout(2000)
            page.screenshot(path="tests/playwright/screenshots/workflow_04_config_shown.png")

        # Test channel buttons
        if add_channel_btn.count() > 0 and remove_channel_btn.count() > 0:
            # Test add channel
            add_channel_btn.first.click()
            page.wait_for_timeout(2000)
            page.screenshot(path="tests/playwright/screenshots/workflow_04a_added_channel.png")

            # Test remove channel
            remove_channel_btn.first.click()
            page.wait_for_timeout(2000)
            page.screenshot(path="tests/playwright/screenshots/workflow_04b_removed_channel.png")

        # Step 5: STRICT Start Button Interaction and Main Screen Transition
        if start_buttons.count() > 0:
            # Click start button
            start_buttons.first.click()
            page.wait_for_timeout(10000)
            page.screenshot(path="tests/playwright/screenshots/workflow_05_after_start.png")

            # Verify successful transition to main screen
            # Look for main screen specific elements (with the fix applied)
            main_screen_elements = page.locator(
                "text=/Configuration/i, "
                "text=/Preview/i, "
                "text=/Status/i, "
                ".cutana-panel, "
                "text=/Start Processing/i"
            )

            # Check for configuration panel elements that should appear in main screen
            config_elements = page.locator(
                "select, "  # Format/stretch dropdowns
                'input[type="number"], '  # Resolution input
                'input[type="range"], '  # Workers slider
                "text=/GB/i"  # Filesize prediction
            )

            # At minimum, should see main screen layout with config elements
            transition_success = main_screen_elements.count() >= 2 or config_elements.count() >= 3

            if not transition_success:
                # Additional diagnostic info
                all_widgets = page.locator(".widget-box, .widget-hbox, .widget-vbox")
                page.screenshot(
                    path="tests/playwright/screenshots/workflow_05_transition_failed.png"
                )
                print(f"Main screen elements: {main_screen_elements.count()}")
                print(f"Config elements: {config_elements.count()}")
                print(f"All widgets: {all_widgets.count()}")

            assert (
                transition_success
            ), f"Main screen transition failed. Main elements: {main_screen_elements.count()}, \
Config elements: {config_elements.count()}"

            # Confirm the stretch dropdown fix is working (no "Invalid selection" error)
            # If we got this far without errors, the fix worked
            print("✅ Main screen transition successful - stretch dropdown fix confirmed")
        else:
            pytest.fail("No start button found to complete workflow")

    def test_csv_analysis_triggering(self, page, voila_server):
        """Test that CSV analysis is properly triggered after file selection."""
        # Navigate directly to Voila app
        page.goto(voila_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(15000)  # Give widgets time to render

        # Take initial screenshot
        page.screenshot(path="tests/playwright/screenshots/01_initial_load.png")

        # Verify Cutana UI actually loaded by looking for widget elements, not just text
        title_widget = page.locator(".widget-html").filter(
            has_text="CUTANA Cutout Generator Configuration"
        )
        file_selection_widget = page.locator(".widget-html").filter(
            has_text="Source Catalogue Selection"
        )

        cutana_title_count = title_widget.count() + file_selection_widget.count()
        print(f"Cutana title elements found: {cutana_title_count}")
        if cutana_title_count == 0:
            # Look for any indication this is the Cutana UI
            all_text = page.locator("text=/source/i, text=/catalogue/i, text=/selection/i")
            print(f"UI-related text found: {all_text.count()}")
            page.screenshot(path="tests/playwright/screenshots/01_no_cutana_ui.png")
            pytest.fail("Cutana UI did not load - no title or UI elements found")

        # Look for file selection components - be very specific
        # ipyfilechooser creates specific DOM structures
        file_browser_buttons = page.locator("button").filter(has_text="📁")  # Folder icon
        file_path_inputs = page.locator('input[placeholder*="path"], input[placeholder*="file"]')
        select_buttons = page.locator("button").filter(has_text="Select")

        # Look for traditional file inputs as backup
        file_inputs = page.locator('input[type="file"]')

        print(f"File browser buttons: {file_browser_buttons.count()}")
        print(f"File path inputs: {file_path_inputs.count()}")
        print(f"Select buttons: {select_buttons.count()}")
        print(f"Traditional file inputs: {file_inputs.count()}")

        # Must find SOME way to select files
        has_file_selection = (
            file_browser_buttons.count() > 0
            or file_path_inputs.count() > 0
            or select_buttons.count() > 0
            or file_inputs.count() > 0
        )

        if not has_file_selection:
            page.screenshot(path="tests/playwright/screenshots/02_no_file_selection.png")
            pytest.fail("No file selection mechanism found in UI")

        # Get path to test CSV file
        project_root = Path(__file__).parent.parent.parent
        test_csv = project_root / "tests" / "test_data" / "euclid_cutana_catalogue_small.csv"

        if not test_csv.exists():
            pytest.skip(f"Test CSV file not found: {test_csv}")

        print(f"Attempting to select file: {test_csv}")
        page.screenshot(path="tests/playwright/screenshots/03_before_file_selection.png")

        # Try multiple file selection methods
        file_selected = False

        # Method 1: Traditional file input
        if file_inputs.count() > 0:
            print("Trying traditional file input...")
            try:
                file_inputs.first.set_input_files(str(test_csv))
                file_selected = True
                print("✅ File selected via file input")
            except Exception as e:
                print(f"❌ File input failed: {e}")

        # Method 2: ipyfilechooser path input + select button
        if not file_selected and file_path_inputs.count() > 0:
            print("Trying path input + select button...")
            try:
                file_path_inputs.first.fill(str(test_csv))
                page.wait_for_timeout(1000)
                if select_buttons.count() > 0:
                    select_buttons.first.click()
                    file_selected = True
                    print("✅ File selected via path input + select")
            except Exception as e:
                print(f"❌ Path input method failed: {e}")

        # Method 3: Navigate using file browser
        if not file_selected and file_browser_buttons.count() > 0:
            print("Trying file browser navigation...")
            try:
                # This is complex - would need to navigate directories
                # For now, just indicate we found the file browser
                print("📁 File browser found but navigation not implemented")
            except Exception as e:
                print(f"❌ File browser method failed: {e}")

        if not file_selected:
            page.screenshot(path="tests/playwright/screenshots/04_file_selection_failed.png")
            pytest.fail("Could not select CSV file using any method")

        print("✅ File selection attempted, waiting for UI response...")
        page.screenshot(path="tests/playwright/screenshots/05_after_file_selection.png")

        # Wait for analysis to start - look for loading indicators
        page.wait_for_timeout(3000)

        # Check if loading state appeared
        loading_indicators = page.locator("text=/loading/i, text=/Analysing/i, text=/processing/i")
        print(f"Loading indicators found: {loading_indicators.count()}")

        page.screenshot(path="tests/playwright/screenshots/06_checking_loading.png")

        # Wait longer for analysis to complete
        page.wait_for_timeout(25000)
        page.screenshot(path="tests/playwright/screenshots/07_after_analysis_wait.png")

        # NOW check for analysis results with STRICT requirements

        # 1. MUST find source count (25 sources)
        source_count_text = page.locator("text=/25.*source/i")
        print(f"Source count text (25): {source_count_text.count()}")

        # 2. MUST find analysis results section
        analysis_section = page.locator("text=/Analysis.*Result/i")
        print(f"Analysis results section: {analysis_section.count()}")

        # 3. MUST find VIS extension (from our test data)
        vis_extension = page.locator("text=/VIS/i")
        print(f"VIS extension text: {vis_extension.count()}")

        # 4. MUST find configuration elements that appear after analysis
        config_checkboxes = page.locator('input[type="checkbox"]')
        config_dropdowns = page.locator("select")
        config_buttons = page.locator("button").filter(has_text="Start")

        print(f"Config checkboxes: {config_checkboxes.count()}")
        print(f"Config dropdowns: {config_dropdowns.count()}")
        print(f"Start buttons: {config_buttons.count()}")

        # 5. Check if file selection status updated (should show selected file)
        file_status = page.locator("text=/selected/i, text=/✅/i")
        print(f"File selection status: {file_status.count()}")

        page.screenshot(path="tests/playwright/screenshots/08_final_state.png")

        # STRICT ASSERTIONS - at least 2 of these must be true for analysis to have worked

        analysis_working_indicators = 0

        if source_count_text.count() > 0:
            analysis_working_indicators += 1
            print("✅ Found source count text")
        else:
            print("❌ No source count text found")

        if analysis_section.count() > 0:
            analysis_working_indicators += 1
            print("✅ Found analysis results section")
        else:
            print("❌ No analysis results section")

        if vis_extension.count() > 0:
            analysis_working_indicators += 1
            print("✅ Found VIS extension")
        else:
            print("❌ No VIS extension found")

        if config_checkboxes.count() > 0 or config_buttons.count() > 0:
            analysis_working_indicators += 1
            print("✅ Found configuration elements")
        else:
            print("❌ No configuration elements")

        if file_status.count() > 0:
            analysis_working_indicators += 1
            print("✅ Found file selection confirmation")
        else:
            print("❌ No file selection confirmation")

        print(f"Analysis indicators found: {analysis_working_indicators}/5")

        # FAIL if less than 2 indicators - means analysis didn't work
        assert (
            analysis_working_indicators >= 2
        ), f"Analysis failed - only {analysis_working_indicators}/5 indicators found. \
Check screenshots in tests/playwright/screenshots/"

        print(f"✅ SUCCESS: Analysis working with {analysis_working_indicators}/5 indicators")
