#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Saved matrices preserve named inputs and positional outputs."""

from cutana.get_default_config import get_default_config
from cutana_ui.widgets.configuration_widget import SharedConfigurationWidget


def test_asymmetric_matrix_round_trip():
    config = get_default_config()
    config.available_extensions = [
        {"name": "VIS", "ext": "PRIMARY"},
        {"name": "NIR-H", "ext": "PRIMARY"},
    ]
    config.selected_extensions = config.available_extensions.copy()
    config.channel_weights = {"NIR-H": [0.2, 0.4, 0.6], "VIS": [0.1, 0.3, 0.5]}
    widget = SharedConfigurationWidget(config, show_extensions=False)
    widget.update_config(config)
    assert [[cell.value for cell in row] for row in widget.channel_matrices] == [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    assert widget.get_current_config().channel_weights == config.channel_weights
