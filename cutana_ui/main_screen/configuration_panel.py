#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Configuration panel for the main screen."""

import ipywidgets as widgets

from ..styles import (
    BACKGROUND_DARK,
    BORDER_COLOR,
    ESA_BLUE_ACCENT,
    PANEL_WIDTH,
    scale_px,
)
from ..widgets.configuration_widget import SharedConfigurationWidget


class ConfigurationPanel(SharedConfigurationWidget):
    """Enhanced configuration panel with all processing parameters."""

    def __init__(self, config, on_start=None, on_stop=None):
        self.on_start = on_start
        self.on_stop = on_stop
        self.is_processing = False

        # Title - more compact
        self.title = widgets.HTML(
            value=f'<h2 style="color: {ESA_BLUE_ACCENT}; margin: 0 0 5px 0; font-size: {scale_px(18)}px;">Configuration</h2>'
        )

        # Initialize shared configuration widget in compact mode without extensions selector but with matrix
        super().__init__(
            config=config,
            compact=True,
            show_extensions=False,
            show_matrix=True,
        )

        # Maintain backward compatibility for tests and other components that expect .shared_config
        self.shared_config = self

        # Prepend title to children
        self.children = [self.title] + list(self.children)

        # Container - compact padding for space efficiency
        self.layout.padding = f"{scale_px(8)}px"
        self.layout.background = BACKGROUND_DARK
        self.layout.border_radius = f"{scale_px(10)}px"
        self.layout.width = "100%"
        self.layout.max_width = f"{PANEL_WIDTH + 60}px"
        self.layout.min_width = f"{scale_px(360)}px"
        self.layout.overflow = "visible"  # Ensure all content is visible without scrolling
        self.layout.border = f"1px solid {BORDER_COLOR}"
        self.add_class("cutana-panel")
