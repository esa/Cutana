#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Basic Voila server integration tests for Cutana UI."""

import pytest

# Marks this as a Playwright test
pytestmark = pytest.mark.playwright


class TestVoilaServerBasics:
    """Basic tests for Voila server integration."""

    # Use the shared voila_server fixture from conftest.py
    pass  # No local fixture needed, using shared one

    def test_voila_server_starts(self, voila_server):
        """Test that Voila server starts successfully."""
        assert "127.0.0.1:8866" in voila_server

    def test_cutana_ui_loads_in_voila(self, page, voila_server):
        """Test that the real Cutana UI loads in the Voila web app."""
        # Navigate directly to the Voila app (no cell execution needed)
        page.goto(voila_server)

        # Wait for Voila app to load - be more patient
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(5000)  # Initial wait

        # Check if there's an error or loading state
        error_elements = page.locator("text=/error/i, text=/failed/i")
        loading_elements = page.locator("text=/loading/i, text=/executing/i")

        # Wait longer if still loading
        if loading_elements.count() > 0:
            page.wait_for_timeout(20000)

        # Take a screenshot for debugging
        page.screenshot(path="tests/playwright/screenshots/voila_load_debug.png")

        # Check that we have the Voila app container
        voila_app = page.locator("body")
        assert voila_app.count() > 0, "No Voila app found"

        # Look for any indication the app is working
        # First check for any widgets at all
        widget_elements = page.locator(".widget-box, .widget-hbox, .widget-vbox")

        # Then look for Cutana-specific elements (more flexible)
        cutana_text = page.locator("text=/cutana/i")
        source_text = page.locator("text=/source/i")
        catalogue_text = page.locator("text=/catalogue/i")

        # Also check for ipywidgets elements
        ipywidget_elements = page.locator(".jupyter-widgets, [data-widget-id]")

        # Debug output
        page_text = page.locator("body").text_content()
        print(f"Page contains widgets: {widget_elements.count()}")
        print(f"Page contains 'cutana': {cutana_text.count()}")
        print(f"Page contains 'source': {source_text.count()}")
        print(f"Page contains 'catalogue': {catalogue_text.count()}")
        print(f"Page contains ipywidgets: {ipywidget_elements.count()}")
        print(f"Page text preview: {page_text[:500] if page_text else 'No text'}")

        # Check if any errors occurred
        if error_elements.count() > 0:
            error_text = error_elements.first.text_content()
            pytest.fail(f"Voila app shows error: {error_text}")

        # Success if we find widgets OR cutana-related text
        ui_working = (
            widget_elements.count() > 0 or cutana_text.count() > 0 or ipywidget_elements.count() > 0
        )

        assert (
            ui_working
        ), f"No UI elements found. Widgets: {widget_elements.count()}, \
Cutana text: {cutana_text.count()}, IPywidgets: {ipywidget_elements.count()}"

    def test_real_ui_widgets_present(self, page, voila_server):
        """Test that real ipywidgets are present and functional in Voila."""
        # Navigate to Voila app
        page.goto(voila_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(15000)  # Give widgets time to render

        # Look for widget-specific elements
        # Voila creates cleaner DOM without Jupyter cell structures
        widget_elements = page.locator(
            '.widget-box, .widget-hbox, .widget-vbox, .widget-button, input[type="file"]'
        )

        # Wait for widgets to fully load
        if widget_elements.count() == 0:
            page.wait_for_timeout(10000)
            widget_elements = page.locator(
                '.widget-box, .widget-hbox, .widget-vbox, .widget-button, input[type="file"]'
            )

        # Test for real widgets - Voila should have cleaner widget presentation
        assert widget_elements.count() > 0, "No real widgets found in Voila app"

        # Test widget interaction capability
        interactive_elements = page.locator("button, input, select")
        assert interactive_elements.count() > 0, "No interactive elements found in Voila app"

        # Verify UI is responsive by testing basic interaction
        buttons = page.locator("button")
        if buttons.count() > 0:
            # Test that we can at least hover over buttons (indicates they're functional)
            buttons.first.hover()
            page.wait_for_timeout(1000)

        # Screenshot for debugging
        page.screenshot(path="tests/playwright/screenshots/voila_widgets_loaded.png")
