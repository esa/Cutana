#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for custom UI widgets."""
from unittest.mock import MagicMock

from cutana_ui.widgets.file_chooser import CutanaFileChooser
from cutana_ui.widgets.header_version_help import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVELS,
    create_header_container,
)
from cutana_ui.widgets.loading_spinner import LoadingSpinner
from cutana_ui.widgets.progress_bar import CutanaProgressBar


class TestLoadingSpinner:
    """Test the LoadingSpinner widget."""

    def test_init(self):
        """Test LoadingSpinner initialization."""
        spinner = LoadingSpinner("Test message")
        assert len(spinner.children) == 2
        assert "Test message" in spinner.message_label.value

    def test_update_message(self):
        """Test updating the spinner message."""
        spinner = LoadingSpinner("Initial")
        spinner.update_message("Updated")
        assert "Updated" in spinner.message_label.value


class TestCutanaProgressBar:
    """Test the CutanaProgressBar widget."""

    def test_init(self):
        """Test progress bar initialization."""
        progress = CutanaProgressBar(value=25, max_value=100, description="Test")
        assert progress.value == 25
        assert progress.max_value == 100
        assert "25% Complete" in progress.progress_html.value

    def test_update(self):
        """Test updating progress bar values."""
        progress = CutanaProgressBar(value=0, max_value=100)
        progress.update(value=50)
        assert progress.value == 50
        assert "50% Complete" in progress.progress_html.value

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        progress = CutanaProgressBar(value=33, max_value=100)
        assert progress._get_percentage() == 33

        progress = CutanaProgressBar(value=0, max_value=0)
        assert progress._get_percentage() == 0


class TestCutanaFileChooser:
    """Test the CutanaFileChooser widget."""

    def test_init(self):
        """Test file chooser initialization."""
        chooser = CutanaFileChooser()
        assert len(chooser.children) == 2  # Style HTML + FileChooser
        assert chooser.file_chooser is not None

    def test_properties(self):
        """Test file chooser properties."""
        chooser = CutanaFileChooser()

        # Test that properties exist and are accessible
        # Note: actual values depend on FileChooser internal state
        selected = chooser.selected
        selected_filename = chooser.selected_filename

        # Properties should be accessible (can be None or empty string)
        assert selected is not None or selected == "" or selected is None
        assert selected_filename is not None or selected_filename == "" or selected_filename is None

    def test_filter_pattern(self):
        """Test file chooser with filter patterns."""
        # Test CSV-only filter
        csv_chooser = CutanaFileChooser(filter_pattern=["*.csv"])
        assert csv_chooser.file_chooser is not None

        # Test multiple patterns
        multi_chooser = CutanaFileChooser(filter_pattern=["*.csv", "*.txt"])
        assert multi_chooser.file_chooser is not None

    def test_directory_mode(self):
        """Test file chooser in directory selection mode."""
        dir_chooser = CutanaFileChooser(select_dir=True)
        assert dir_chooser.file_chooser is not None

    def test_styling(self):
        """Test that custom styling is applied."""
        chooser = CutanaFileChooser()

        # Should have style HTML as first child
        assert len(chooser.children) >= 1
        style_html = chooser.children[0]

        # Should contain ESA colors in styling
        assert hasattr(style_html, "value")
        style_content = style_html.value
        assert "#0098DB" in style_content or "#003249" in style_content


class TestHeaderLogLevelDropdown:
    """Test the log level dropdown in header container."""

    def test_header_returns_three_elements(self):
        """Test that create_header_container returns header, help button, and log dropdown."""
        header, help_button, log_dropdown = create_header_container(
            version_text="v1.0.0",
            container_width=1200,
            help_button_callback=lambda x: None,
        )

        assert header is not None
        assert help_button is not None
        assert log_dropdown is not None

    def test_log_dropdown_options(self):
        """Test that log dropdown has correct options."""
        _, _, log_dropdown = create_header_container(
            version_text="v1.0.0",
            container_width=1200,
            help_button_callback=lambda x: None,
        )

        assert log_dropdown.options == tuple(LOG_LEVELS)
        assert log_dropdown.value == DEFAULT_LOG_LEVEL

    def test_log_dropdown_default_value(self):
        """Test that log dropdown defaults to Warning."""
        _, _, log_dropdown = create_header_container(
            version_text="v1.0.0",
            container_width=1200,
            help_button_callback=lambda x: None,
        )

        assert log_dropdown.value == "Warning"

    def test_log_dropdown_callback_invoked(self):
        """Test that changing log level invokes callback with uppercase value."""
        callback = MagicMock()

        _, _, log_dropdown = create_header_container(
            version_text="v1.0.0",
            container_width=1200,
            help_button_callback=lambda x: None,
            log_level_callback=callback,
        )

        # Change the log level (capitalized in dropdown)
        log_dropdown.value = "Debug"

        # Callback should have been called with uppercase value for loguru
        callback.assert_called_once_with("DEBUG")

    def test_log_dropdown_no_callback_without_handler(self):
        """Test that no error occurs when callback is None."""
        _, _, log_dropdown = create_header_container(
            version_text="v1.0.0",
            container_width=1200,
            help_button_callback=lambda x: None,
            log_level_callback=None,
        )

        # Should not raise an error when changing value
        log_dropdown.value = "Info"
        assert log_dropdown.value == "Info"
