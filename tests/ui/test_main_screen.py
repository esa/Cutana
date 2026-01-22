#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the main screen functionality."""


class TestMainScreen:
    """Test the main screen functionality."""

    def test_main_screen_initialization(self):
        """Test that main screen initializes with all components."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.main_screen import MainScreen

        config = get_default_config()
        config.num_sources = 25
        config.available_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        config.normalisation_method = "linear"  # This should be handled by the fix
        config.flux_conserved_resizing = False

        screen = MainScreen(config=config)

        # Check that all components are created
        assert hasattr(screen, "config_panel")
        assert hasattr(screen, "preview_panel")
        assert hasattr(screen, "status_panel")
        assert hasattr(screen, "main_container")

    def test_configuration_panel_initialization(self):
        """Test configuration panel with stretch dropdown fix."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.configuration_panel import ConfigurationPanel

        config = get_default_config()
        config.normalisation_method = "linear"  # This should be converted to "linear"
        config.target_resolution = 256
        config.available_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        config.num_sources = 25

        panel = ConfigurationPanel(config=config)

        # Check the stretch dropdown fix is applied
        assert panel.normalisation_dropdown.value == "linear"

        # Check other configuration elements
        assert panel.resolution_input.value == 256
        assert panel.format_dropdown.value == "float32"
        assert panel.output_format_dropdown.value == "zarr"

    def test_configuration_panel_stretch_fix(self):
        """Test that 'none' stretch value is properly converted to 'linear'."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.configuration_panel import ConfigurationPanel

        config = get_default_config()
        config.normalisation_method = "linear"
        panel = ConfigurationPanel(config=config)

        # Should convert "none" to "linear" for the dropdown
        assert panel.normalisation_dropdown.value == "linear"

        # But when getting config back, should use the normalisation method
        current_config = panel.get_current_config()
        assert current_config["normalisation_method"] == "linear"

    def test_configuration_panel_channel_matrix(self):
        """Test channel matrix functionality."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.configuration_panel import ConfigurationPanel

        config = get_default_config()
        extensions = [{"name": "VIS", "ext": "IMAGE"}, {"name": "NIR", "ext": "IMAGE"}]
        config.available_extensions = extensions
        config.num_sources = 25
        panel = ConfigurationPanel(config=config)

        # Initially should have 1 channel (access through shared config)
        assert panel.shared_config.current_channels == 1

        # Test adding channel
        panel.shared_config._add_channel()
        assert panel.shared_config.current_channels == 2

        # Test removing channel
        panel.shared_config._remove_channel()
        assert panel.shared_config.current_channels == 1

        # Can't go below 1 channel
        panel.shared_config._remove_channel()
        assert panel.shared_config.current_channels == 1

    def test_configuration_panel_filesize_prediction(self):
        """Test filesize prediction updates."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.configuration_panel import ConfigurationPanel

        config = get_default_config()
        config.num_sources = 100
        config.target_resolution = 256
        panel = ConfigurationPanel(config=config)

        # Initial filesize calculation (access through shared config)
        panel.shared_config._update_filesize_prediction()

        # Should have some filesize display
        assert "GB" in panel.shared_config.filesize_display.value

        # Change resolution and check update
        panel.resolution_input.value = 512
        panel.shared_config._update_filesize_prediction()

        # Should still have filesize display with updated value
        assert "GB" in panel.shared_config.filesize_display.value

    def test_main_screen_start_button(self):
        """Test that main screen has start button (moved from configuration panel)."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.main_screen import MainScreen

        config = get_default_config()
        config.num_sources = 25
        config.available_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        screen = MainScreen(config=config)

        # Check that start button exists in main screen
        assert hasattr(screen, "start_button")
        assert hasattr(screen, "start_button_container")
        assert screen.start_button.description == "Start Cutout Creation"
        assert screen.start_button.button_style == "success"

        # Check processing state change functionality
        screen.set_processing_state(True)
        assert screen.start_button.description == "Stop Processing"
        assert screen.start_button.button_style == "danger"

        screen.set_processing_state(False)
        assert screen.start_button.description == "Start Cutout Creation"
        assert screen.start_button.button_style == "success"

    def test_preview_panel_initialization(self):
        """Test preview panel initialization."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.preview_panel import PreviewPanel

        config = get_default_config()
        config.source_catalogue = "test.csv"
        config.num_sources = 25

        panel = PreviewPanel(config=config)

        # Check basic components exist
        assert hasattr(panel, "title")
        assert hasattr(panel, "refresh_button")
        assert hasattr(panel, "header")
        assert hasattr(panel, "preview_container")
        assert hasattr(panel, "loading_spinner")
        assert hasattr(panel, "info_text")

        # Check refresh button is properly configured
        assert panel.refresh_button.icon == "refresh"

    def test_preview_panel_load_sources_method(self):
        """Test preview panel load_preview_sources method."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.preview_panel import PreviewPanel

        config = get_default_config()
        config.source_catalogue = "tests/test_data/euclid_cutana_catalogue_small.csv"
        config.selected_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        config.num_sources = 25

        panel = PreviewPanel(config=config)

        # Check that the new method exists
        assert hasattr(panel, "load_preview_sources")
        assert callable(panel.load_preview_sources)

    def test_preview_panel_reload_sources_method(self):
        """Test preview panel reload_preview_sources method."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.preview_panel import PreviewPanel

        config = get_default_config()
        config.source_catalogue = "tests/test_data/euclid_cutana_catalogue_small.csv"
        config.selected_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        config.num_sources = 25

        panel = PreviewPanel(config=config)

        # Check that the new method exists
        assert hasattr(panel, "reload_preview_sources")
        assert callable(panel.reload_preview_sources)

    def test_preview_panel_refresh_functionality(self):
        """Test preview panel refresh button functionality."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.preview_panel import PreviewPanel

        config = get_default_config()
        config.source_catalogue = "test.csv"
        config.num_sources = 25
        panel = PreviewPanel(config=config)

        # Check refresh button callback is set up
        assert hasattr(panel, "_on_refresh_clicked")

        # Check that the refresh button has click handlers registered
        assert hasattr(panel.refresh_button, "_click_handlers")
        # Verify the handler is callable
        assert callable(panel._on_refresh_clicked)

    def test_preview_panel_color_display_logic(self):
        """Test that preview panel handles different image formats correctly."""
        import numpy as np

        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.preview_panel import PreviewPanel

        config = get_default_config()
        config.num_sources = 25
        panel = PreviewPanel(config=config)

        # Test single channel (grayscale) array
        grayscale_array = np.random.rand(64, 64).astype(np.uint8)
        widget = panel._create_preview_widget(12.345, 56.789, grayscale_array)
        assert widget is not None

        # Test three channel (RGB) array
        rgb_array = np.random.rand(64, 64, 3).astype(np.uint8)
        widget = panel._create_preview_widget(12.345, 56.789, rgb_array)
        assert widget is not None

        # Test other channel configurations
        two_channel_array = np.random.rand(64, 64, 2).astype(np.uint8)
        widget = panel._create_preview_widget(12.345, 56.789, two_channel_array)
        assert widget is not None

    def test_status_panel_initialization(self):
        """Test status panel initialization."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.status_panel import StatusPanel

        config = get_default_config()
        config.num_sources = 25
        panel = StatusPanel(config=config)

        # Check basic components exist (start button moved to configuration panel)
        assert hasattr(panel, "title")
        assert hasattr(panel, "ready_status")
        assert hasattr(panel, "progress_bar")
        assert hasattr(panel, "progress_container")
        assert hasattr(panel, "stats_html")
        assert hasattr(panel, "processing_indicator")

        # Initially should show ready status
        assert panel.ready_status.layout.display != "none"

    def test_main_screen_config_change_callbacks(self):
        """Test configuration change callbacks between components."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.main_screen import MainScreen

        config = get_default_config()
        config.num_sources = 25
        config.available_extensions = [{"name": "VIS", "ext": "IMAGE"}]

        screen = MainScreen(config=config)

        # Should have config change callback set up (on shared config)
        assert hasattr(screen.config_panel.shared_config, "_config_change_callback")

    def test_configuration_panel_config_update(self):
        """Test updating configuration from external source."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.configuration_panel import ConfigurationPanel

        initial_config = get_default_config()
        initial_config.num_sources = 25
        initial_config.available_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        panel = ConfigurationPanel(initial_config)

        # Update with new config
        new_config = get_default_config()
        new_config.num_sources = 50
        new_config.available_extensions = [
            {"name": "VIS", "ext": "IMAGE"},
            {"name": "NIR", "ext": "IMAGE"},
        ]

        panel.update_config(new_config)

        # Should update internal state (access through shared config and properties)
        assert panel.num_sources == 50
        assert len(panel.shared_config.extensions) == 2

    def test_preview_panel_config_change_triggers_reload(self):
        """Test that changing catalogue or extensions triggers source reload."""
        from cutana.get_default_config import get_default_config
        from cutana_ui.main_screen.preview_panel import PreviewPanel

        initial_config = get_default_config()
        initial_config.source_catalogue = "catalogue1.csv"
        initial_config.selected_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        initial_config.num_sources = 25

        panel = PreviewPanel(initial_config)

        # Mock the reload method to track if it's called
        reload_called = False

        def mock_reload():
            nonlocal reload_called
            reload_called = True

        panel.reload_preview_sources = mock_reload

        # Update config with different catalogue - should trigger reload
        new_config = initial_config.copy()
        new_config["source_catalogue"] = "catalogue2.csv"
        panel.update_config(new_config)

        # Should have triggered reload due to catalogue change
        assert reload_called

        # Reset flag
        reload_called = False

        # Update config with different extensions - should trigger reload
        new_config2 = new_config.copy()
        new_config2["selected_extensions"] = [{"name": "NIR", "ext": "IMAGE"}]
        panel.update_config(new_config2)

        # Should have triggered reload due to extensions change
        assert reload_called

        # Reset flag
        reload_called = False

        # Update config with same catalogue and extensions but different other params - should not trigger reload
        new_config3 = new_config2.copy()
        new_config3["target_resolution"] = 512  # Different resolution but same core params
        panel.update_config(new_config3)

        # Should not have triggered reload
        assert not reload_called


class TestMainScreenIntegration:
    """Test main screen integration and workflow."""

    def test_main_screen_with_real_config(self):
        """Test main screen with realistic configuration data."""
        from cutana.get_default_config import get_default_config

        config = get_default_config()
        config.source_catalogue = "tests/test_data/euclid_cutana_catalogue_small.csv"
        config.output_dir = "output"
        config.output_format = "zarr"
        config.data_type = "float32"
        config.target_resolution = 256
        config.normalisation_method = "linear"  # Test the fix
        config.selected_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        config.max_workers = 8
        config.normalisation_method = "linear"
        config.interpolation = "bilinear"
        config.num_sources = 25
        config.fits_files = ["test.fits"]
        config.available_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        config.data_format = "float32"
        config.channel_matrix = [[1.0]]

        from cutana_ui.main_screen.main_screen import MainScreen

        # Should not raise any errors with this realistic config
        screen = MainScreen(config=config)

        # Check components are properly initialized
        assert screen.config_panel.normalisation_dropdown.value == "linear"
        assert screen.config_panel.resolution_input.value == 256
        assert screen.config_panel.num_sources == 25
