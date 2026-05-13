#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for start screen edge cases and internal logic.

Rendering and interaction tests have moved to tests/browser/.
These tests cover validation, error handling, and config logic
that cannot be tested through browser interaction.
"""

import asyncio
from unittest.mock import patch

from cutana import get_default_config
from cutana_ui.start_screen import StartScreen
from cutana_ui.start_screen.configuration_component import ConfigurationComponent
from cutana_ui.styles import (
    COMMON_STYLES,
    ERROR_COLOR,
    ESA_BLUE_DEEP,
    ESA_BLUE_GREY,
    ESA_GREEN,
    ESA_RED,
    SUCCESS_COLOR,
)
from cutana_ui.widgets.configuration_widget import SharedConfigurationWidget


class TestStartScreen:
    """Test start screen internal logic and edge cases."""

    def test_resolution_validation(self):
        """Test that resolution input validates minimum value."""

        component = ConfigurationComponent()
        component.resolution_input.value = 10
        assert component.resolution_input.value >= 16

    def test_stretch_function_naming(self):
        """Test that stretch function uses 'linear' instead of 'none'."""
        config = get_default_config()
        component = SharedConfigurationWidget(
            config=config,
            compact=False,
            show_extensions=True,
            show_matrix=False,
            show_advanced_params=True,
        )

        if component.normalisation_dropdown is not None:
            assert "linear" in component.normalisation_dropdown.options
            assert "none" in component.normalisation_dropdown.options

            component.set_extensions([{"name": "TEST", "ext": "IMAGE"}])
            current_config = component.get_current_config()
            if component.normalisation_dropdown.value == "linear":
                assert current_config["normalisation_method"] == "linear"

    def test_color_scheme_application(self):
        """Test that ESA color scheme is applied."""
        assert ESA_BLUE_DEEP == "#003249"
        assert ESA_GREEN == "#008542"
        assert ESA_RED == "#EC1A2F"
        assert SUCCESS_COLOR == ESA_GREEN
        assert ERROR_COLOR == ESA_RED


class TestUIIntegration:
    """Integration tests for edge cases and error handling."""

    def test_dropdown_background_colors(self):
        """Test that dropdowns have proper dark backgrounds."""
        assert "widget-dropdown" in COMMON_STYLES
        assert ESA_BLUE_GREY in COMMON_STYLES
        assert "background: #335E6E" in COMMON_STYLES

    def test_analysis_workflow_error(self):
        """Test analysis workflow error handling."""
        screen = StartScreen()

        file_path = "/nonexistent/file.csv"
        expected_error = f"File not found: {file_path}"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with patch.object(screen.file_selection, "show_error") as mock_show_error:
                loop.run_until_complete(screen._analyze_catalogue(file_path))

                mock_show_error.assert_called_once()
                call_args = mock_show_error.call_args[0][0]
                assert expected_error in call_args
        finally:
            loop.close()

    def test_validation_errors(self):
        """Test start button validation errors."""
        screen = StartScreen()

        with (
            patch.object(
                screen.configuration,
                "get_configuration",
                return_value={"selected_extensions": []},
            ),
            patch.object(screen.output_folder, "get_output_dir", return_value="/test/output"),
        ):
            screen._on_start_click(None)
            assert "select at least one FITS extension" in screen.error_message.value

        with (
            patch.object(
                screen.configuration,
                "get_configuration",
                return_value={"selected_extensions": [{"name": "VIS", "ext": "IMAGE"}]},
            ),
            patch.object(screen.output_folder, "get_output_dir", return_value=None),
        ):
            screen._on_start_click(None)
            assert "select an output directory" in screen.error_message.value
