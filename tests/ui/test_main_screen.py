#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for main screen internal logic.

Rendering and panel visibility tests have moved to tests/browser/.
These tests cover internal state management, config update logic,
and preview processing that cannot be tested through browser interaction.
"""

import numpy as np

from cutana.get_default_config import get_default_config
from cutana_ui.main_screen.configuration_panel import ConfigurationPanel
from cutana_ui.main_screen.main_screen import MainScreen
from cutana_ui.main_screen.preview_panel import PreviewPanel


class TestMainScreen:
    """Test main screen internal logic and edge cases."""

    def test_preview_panel_load_sources_method(self):
        """Test preview panel load_preview_sources method exists."""

        config = get_default_config()
        config.source_catalogue = "tests/test_data/euclid_cutana_catalogue_small.csv"
        config.selected_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        config.num_sources = 25

        panel = PreviewPanel(config=config)
        assert hasattr(panel, "load_preview_sources")
        assert callable(panel.load_preview_sources)

    def test_preview_panel_reload_sources_method(self):
        """Test preview panel reload_preview_sources method exists."""
        config = get_default_config()
        config.source_catalogue = "tests/test_data/euclid_cutana_catalogue_small.csv"
        config.selected_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        config.num_sources = 25

        panel = PreviewPanel(config=config)
        assert hasattr(panel, "reload_preview_sources")
        assert callable(panel.reload_preview_sources)

    def test_preview_panel_refresh_functionality(self):
        """Test preview panel refresh button callback is set up."""
        config = get_default_config()
        config.source_catalogue = "test.csv"
        config.num_sources = 25
        panel = PreviewPanel(config=config)

        assert hasattr(panel, "_on_refresh_clicked")
        assert hasattr(panel.refresh_button, "_click_handlers")
        assert callable(panel._on_refresh_clicked)

    def test_preview_panel_color_display_logic(self):
        """Test that preview panel handles different image formats correctly."""
        config = get_default_config()
        config.num_sources = 25
        panel = PreviewPanel(config=config)

        grayscale_array = np.random.rand(64, 64).astype(np.uint8)
        widget = panel._create_preview_widget(12.345, 56.789, grayscale_array)
        assert widget is not None

        rgb_array = np.random.rand(64, 64, 3).astype(np.uint8)
        widget = panel._create_preview_widget(12.345, 56.789, rgb_array)
        assert widget is not None

        two_channel_array = np.random.rand(64, 64, 2).astype(np.uint8)
        widget = panel._create_preview_widget(12.345, 56.789, two_channel_array)
        assert widget is not None

    def test_main_screen_config_change_callbacks(self):
        """Test configuration change callbacks between components."""
        config = get_default_config()
        config.num_sources = 25
        config.available_extensions = [{"name": "VIS", "ext": "IMAGE"}]

        screen = MainScreen(config=config)
        assert hasattr(screen.config_panel.shared_config, "_config_change_callback")

    def test_configuration_panel_config_update(self):
        """Test updating configuration from external source."""
        initial_config = get_default_config()
        initial_config.num_sources = 25
        initial_config.available_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        panel = ConfigurationPanel(initial_config)

        new_config = get_default_config()
        new_config.num_sources = 50
        new_config.available_extensions = [
            {"name": "VIS", "ext": "IMAGE"},
            {"name": "NIR", "ext": "IMAGE"},
        ]

        panel.update_config(new_config)
        assert panel.num_sources == 50
        assert len(panel.shared_config.extensions) == 2

    def test_preview_panel_config_change_triggers_reload(self):
        """Test that changing catalogue or extensions triggers source reload."""
        initial_config = get_default_config()
        initial_config.source_catalogue = "catalogue1.csv"
        initial_config.selected_extensions = [{"name": "VIS", "ext": "IMAGE"}]
        initial_config.num_sources = 25

        panel = PreviewPanel(initial_config)

        reload_called = False

        def mock_reload():
            nonlocal reload_called
            reload_called = True

        panel.reload_preview_sources = mock_reload

        new_config = initial_config.copy()
        new_config["source_catalogue"] = "catalogue2.csv"
        panel.update_config(new_config)
        assert reload_called

        reload_called = False
        new_config2 = new_config.copy()
        new_config2["selected_extensions"] = [{"name": "NIR", "ext": "IMAGE"}]
        panel.update_config(new_config2)
        assert reload_called

        reload_called = False
        new_config3 = new_config2.copy()
        new_config3["target_resolution"] = 512
        panel.update_config(new_config3)
        assert not reload_called

    def test_show_current_config_internal_logic(self):
        """Test that the Show Current Config callback toggles state."""
        config = get_default_config()
        config.num_sources = 25
        screen = MainScreen(config=config)

        # Assert initial state
        assert hasattr(screen, "config_button")
        assert not screen.showing_config

        # Manually invoke the click handler
        screen._toggle_config(None)
        assert screen.showing_config

        # Manually invoke again to hide
        screen._toggle_config(None)
        assert not screen.showing_config

    def test_copy_config_to_clipboard_builds_javascript(self, monkeypatch):
        """Test that the clipboard callback emits a Javascript payload."""
        config = get_default_config()
        config.num_sources = 25
        screen = MainScreen(config=config)
        screen._config_copy_text = '{"alpha": 1}'

        # Build the config panel first (normally done by _show_config)
        screen._build_config_panel("<span>test</span>")

        captured = {}

        def fake_display(obj):
            captured["obj"] = obj

        monkeypatch.setattr("cutana_ui.main_screen.main_screen.display", fake_display)

        screen._copy_config_to_clipboard()

        assert "obj" in captured
        assert captured["obj"].data is not None
        assert "navigator.clipboard.writeText" in captured["obj"].data
        # Check that the config text is embedded (may be escaped in the JS)
        assert "alpha" in captured["obj"].data
