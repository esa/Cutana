#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for status panel UI component."""

from unittest.mock import Mock, patch

from cutana_ui.main_screen.status_panel import StatusPanel


class TestStatusPanelComponent:
    """Test suite for StatusPanel class."""

    def test_import_status_panel(self):
        """Test that we can import the status panel component."""

        assert StatusPanel is not None

    @patch("cutana_ui.main_screen.status_panel.widgets")
    def test_status_panel_basic(self, mock_widgets):
        """Test StatusPanel basic functionality."""
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.VBox = Mock()
        mock_widgets.FloatProgress.return_value = Mock()
        mock_widgets.Layout.return_value = Mock()

        assert StatusPanel is not None
