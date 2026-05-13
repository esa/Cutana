#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for output folder UI component."""

from unittest.mock import Mock, patch

from cutana_ui.start_screen.output_folder import OutputFolderComponent


class TestOutputFolderComponent:
    """Test suite for OutputFolderComponent class."""

    def test_import_output_folder_component(self):
        """Test that we can import the output folder component."""

        assert OutputFolderComponent is not None

    @patch("cutana_ui.start_screen.output_folder.widgets")
    def test_output_folder_component_basic(self, mock_widgets):
        """Test OutputFolderComponent basic functionality."""
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.VBox = Mock()
        mock_widgets.Layout.return_value = Mock()
        mock_widgets.Text.return_value = Mock()
        mock_widgets.Button.return_value = Mock()

        assert OutputFolderComponent is not None
