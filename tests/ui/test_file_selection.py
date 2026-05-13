#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for file selection UI component."""

import cutana_ui.start_screen.file_selection


def test_import_file_selection_module():
    """Test importing the file selection module."""
    assert cutana_ui.start_screen.file_selection is not None
