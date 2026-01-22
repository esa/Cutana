#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the main application with unified UI."""

from unittest.mock import patch

from dotmap import DotMap

from cutana_ui.app import CutanaApp, start


class TestCutanaApp:
    """Test the main Cutana application with unified start screen."""

    def test_init(self):
        """Test application initialization."""
        app = CutanaApp()
        assert app.config_data == {}
        assert len(app.container.children) == 1
        # Should start with StartScreen in the container
        assert hasattr(app, "container")

    def test_config_data_storage(self):
        """Test configuration data storage."""
        app = CutanaApp()
        test_data = {"source_catalogue": "/test/catalogue.csv", "num_sources": 100}

        app.config_data.update(test_data)

        assert app.config_data["source_catalogue"] == "/test/catalogue.csv"
        assert app.config_data["num_sources"] == 100

    def test_configuration_complete_handler(self):
        """Test configuration completion handler."""
        app = CutanaApp()
        full_config = DotMap(
            {
                "source_catalogue": "/test/catalogue.csv",
                "data_format": "uint8",
                "target_resolution": 512,
                "output_dir": "/test/output",
            }
        )
        config_path = "/test/config.json"

        # Test the handler exists and can be called
        assert hasattr(app, "_on_configuration_complete")

        # Mock the MainScreen, logging setup, and patch the container assignment to avoid widget errors
        with (
            patch("cutana_ui.app.MainScreen") as mock_main_screen,
            patch("cutana_ui.app.setup_ui_logging") as mock_setup_logging,
            patch.object(app, "container") as mock_container,
        ):

            # Create a mock MainScreen instance that won't cause widget errors
            mock_main_screen_instance = mock_main_screen.return_value

            app._on_configuration_complete(full_config)

            # Verify MainScreen was created with correct parameters
            mock_main_screen.assert_called_once_with(config=full_config)
            # Verify container.children was set
            assert mock_container.children == [mock_main_screen_instance]

    def test_show_start_screen(self):
        """Test start screen initialization."""
        app = CutanaApp()

        # Test that start screen method exists and creates StartScreen
        assert hasattr(app, "_show_start_screen")

        # Should have one child (the StartScreen)
        assert len(app.container.children) == 1

    def test_container_styling(self):
        """Test that container has proper styling."""
        app = CutanaApp()

        assert app.container.layout.width == "100%"
        assert "cutana-container" in app.container._dom_classes


class TestStartFunction:
    """Test the start function."""

    @patch("cutana_ui.app.display")
    def test_start(self, mock_display):
        """Test the start function."""
        app = start()

        assert isinstance(app, CutanaApp)
        mock_display.assert_called_once_with(app.container)
