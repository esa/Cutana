#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Start screen browser tests for Cutana UI.

The notebook auto-triggers file selection and analysis with the test CSV,
so these tests verify the UI state after analysis completes.

Uses module_page (shared kernel) since tests are read-only.
"""

import pytest

pytestmark = pytest.mark.browser


def test_header_and_logo(module_page):
    """Test that the header with ESA logo is present."""
    svg_count = module_page.locator("svg").count()
    header_count = module_page.locator("text=/CUTANA/i").count()

    assert svg_count > 0 or header_count > 0, "No header/logo elements found"


def test_file_selection_component(module_page):
    """Test that file selection component renders with chooser widget."""
    chooser = module_page.locator(".cutana-file-chooser")
    selects = module_page.locator("select")
    catalogue_text = module_page.locator("text=/catalogue/i")

    total = chooser.count() + selects.count() + catalogue_text.count()
    assert total > 0, "File selection component not found"


def test_analysis_results_shown(module_page):
    """Test that analysis results are displayed after auto-analysis."""
    source_count = module_page.locator("text=/25/")
    sources_text = module_page.locator("text=/Sources/i")
    fits_text = module_page.locator("text=/FITS/i")

    total = source_count.count() + sources_text.count() + fits_text.count()
    assert total > 0, "Analysis results not shown (expected 25 sources from test CSV)"


def test_configuration_panel_visible(module_page):
    """Test that configuration panel appears after analysis."""
    add_channel = module_page.locator("text=/Add channel/i")
    resolution = module_page.locator("text=/Resolution/i")
    checkboxes = module_page.locator('input[type="checkbox"]')

    total = add_channel.count() + resolution.count() + checkboxes.count()
    assert total > 0, "Configuration panel not visible after analysis"


def test_extension_checkboxes(module_page):
    """Test that FITS extension checkboxes render from analysis."""
    checkboxes = module_page.locator('input[type="checkbox"]')
    vis_text = module_page.locator("text=/VIS/i")

    assert checkboxes.count() > 0, "No extension checkboxes found"
    assert vis_text.count() > 0, "VIS extension not found in checkboxes"


def test_output_folder_component(module_page):
    """Test that output folder chooser is present."""
    output_text = module_page.locator("text=/Output/i")
    choosers = module_page.locator(".cutana-file-chooser")

    assert output_text.count() > 0 or choosers.count() >= 2, "Output folder component not found"


def test_start_button_visible(module_page):
    """Test that Start button is visible after analysis."""
    start_btn = module_page.locator("button").filter(has_text="Start")
    assert start_btn.count() > 0, "Start button not found"
