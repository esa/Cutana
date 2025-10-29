#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Configuration panel for the main screen."""

import ipywidgets as widgets
from ..widgets.configuration_widget import SharedConfigurationWidget
from ..styles import (
    ESA_BLUE_ACCENT,
    BACKGROUND_DARK,
    BORDER_COLOR,
    PANEL_WIDTH,
    scale_px,
)


class ConfigurationPanel(widgets.VBox):
    """Enhanced configuration panel with all processing parameters."""

    def __init__(self, config, on_start=None, on_stop=None):

        self.config = config
        self.on_start = on_start
        self.on_stop = on_stop
        self.is_processing = False

        # Title - more compact
        self.title = widgets.HTML(
            value=f'<h2 style="color: {ESA_BLUE_ACCENT}; margin: 0 0 5px 0; font-size: {scale_px(18)}px;">Configuration</h2>'
        )

        # Shared configuration widget in compact mode without extensions selector but with matrix
        self.shared_config = SharedConfigurationWidget(
            compact=True, config=self.config, show_extensions=False, show_matrix=True
        )

        # Start button removed from configuration panel - now handled by main screen

        # Container - compact padding for space efficiency
        super().__init__(
            children=[
                self.title,
                self.shared_config,
            ],
            layout=widgets.Layout(
                padding=f"{scale_px(8)}px",  # Reduced padding for more space
                background=BACKGROUND_DARK,
                border_radius=f"{scale_px(10)}px",
                width="100%",
                max_width=f"{PANEL_WIDTH + 60}px",  # Further increased width for wider grid
                min_width=f"{scale_px(360)}px",  # Increased minimum width for wider elements
                max_height=f"{scale_px(1000)}px",  # Restored proper height for full visibility
                overflow="visible",  # Enable scrolling when needed
                border=f"1px solid {BORDER_COLOR}",
            ),
        )
        self.add_class("cutana-panel")

    # Delegate methods to shared widget
    def set_extensions(self, extensions):
        """Set available extensions and create checkboxes."""
        return self.shared_config.set_extensions(extensions)

    def set_config_change_callback(self, callback):
        """Set callback for configuration changes."""
        return self.shared_config.set_config_change_callback(callback)

    def update_config(self, config):
        """Update configuration from external source."""
        self.config = config
        return self.shared_config.update_config(config)

    def get_current_config(self):
        """Get current configuration including any changes."""
        return self.shared_config.get_current_config()

    # Expose commonly used attributes
    @property
    def format_dropdown(self):
        return self.shared_config.format_dropdown

    @property
    def output_format_dropdown(self):
        return self.shared_config.output_format_dropdown

    @property
    def resolution_input(self):
        return self.shared_config.resolution_input

    @property
    def normalisation_dropdown(self):
        return self.shared_config.normalisation_dropdown

    @property
    def num_sources(self):
        return self.shared_config.num_sources

    @num_sources.setter
    def num_sources(self, value):
        self.shared_config.num_sources = value
