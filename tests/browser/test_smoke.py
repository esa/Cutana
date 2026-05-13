#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Smoke tests for Cutana UI in Voila.

These are fast browser tests that verify the infrastructure works:
- Voila server starts and serves the notebook
- IPyWidgets render in the browser
- The Cutana start screen loads with expected elements

Marked as 'browser' (not 'slow'), so they run in CI.
"""

import urllib.request

import pytest

pytestmark = pytest.mark.browser


def test_voila_serves_page(voila_server):
    """Test that the Voila server is reachable."""
    response = urllib.request.urlopen(voila_server, timeout=10)
    assert response.status == 200


def test_widgets_render(module_page):
    """Test that ipywidgets render in the Voila page."""
    widget_elements = module_page.locator(
        ".widget-box, .widget-hbox, .widget-vbox, .jupyter-widgets"
    )
    assert widget_elements.count() > 0, "No ipywidgets found on page"


def test_start_screen_loads(module_page):
    """Test that the Cutana start screen loads with expected UI elements."""
    cutana_text = module_page.locator("text=/cutana/i")
    source_text = module_page.locator("text=/source/i")

    assert cutana_text.count() > 0 or source_text.count() > 0, (
        "Cutana start screen elements not found"
    )


def test_interactive_elements_present(module_page):
    """Test that interactive elements (buttons, inputs) are present."""
    interactive = module_page.locator("button, input, select")
    assert interactive.count() > 0, "No interactive elements found"
