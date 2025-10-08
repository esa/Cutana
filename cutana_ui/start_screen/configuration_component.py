#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Configuration component with extension matrix and processing parameters."""

import ipywidgets as widgets
from ..widgets.configuration_widget import SharedConfigurationWidget
from ..styles import (
    ESA_BLUE_ACCENT,
    BACKGROUND_DARK,
    BORDER_COLOR,
    TEXT_COLOR_MUTED,
    scale_px,
)

from cutana.get_default_config import get_default_config


class ConfigurationComponent(widgets.VBox):
    """Component for configuring processing parameters."""

    def __init__(self):
        # Title and analysis stats in header
        self.title = widgets.HTML(
            value=f'<h3 style="color: {ESA_BLUE_ACCENT}; margin-bottom: 10px;">Processing Configuration</h3>'
        )

        # Analysis stats (initially hidden)
        self.analysis_stats = widgets.HTML(value="", layout=widgets.Layout(display="none"))

        self.header = widgets.HBox(
            [self.title, self.analysis_stats],
            layout=widgets.Layout(justify_content="space-between", width="100%"),
        )

        # Initialize base config for the shared widget
        self.base_config = get_default_config()

        # Shared configuration widget in full mode (not compact) without matrix, without advanced params, and without filesize
        self.shared_config = SharedConfigurationWidget(
            compact=False,
            config=self.base_config,
            show_matrix=False,
            show_advanced_params=False,
            show_filesize=False,
        )

        # Create content area with config and button side by side
        self.start_button = None  # Will be set by parent
        self.content_with_button = widgets.HBox(
            children=[self.shared_config],  # Start with just config, button added later
            layout=widgets.Layout(
                width="100%",
                justify_content="flex-start",
                align_items="flex-end",  # Align button to bottom of config content
                gap="20px",  # Gap between config and button
            ),
        )

        super().__init__(
            children=[
                self.header,
                self.content_with_button,  # Config and button side by side
            ],
            layout=widgets.Layout(
                width="auto",  # Let it size based on content
                flex="1",  # Take remaining space in HBox
                max_width=f"{scale_px(700)}px",  # Larger max width for side-by-side layout
                max_height=f"{scale_px(600)}px",
                padding=f"{scale_px(15)}px",
                background=BACKGROUND_DARK,
                border_radius=f"{scale_px(8)}px",
                border=f"1px solid {BORDER_COLOR}",
                margin=f"0 0 {scale_px(15)}px 0",
            ),
        )

    # Method to add start button from parent
    def set_start_button(self, button):
        """Set the start button to display aligned with last configuration element."""
        self.start_button = button
        # Add button to the side of the configuration
        self.content_with_button.children = [self.shared_config, button]

    # Delegate methods to shared widget
    def set_extensions(self, extensions):
        """Set available extensions and create checkboxes."""
        return self.shared_config.set_extensions(extensions)

    def set_num_sources(self, num_sources):
        """Set the number of sources for filesize calculation."""
        return self.shared_config.set_num_sources(num_sources)

    def get_configuration(self):
        """Get current configuration."""
        return self.shared_config.get_current_config()

    def set_analysis_results(self, result):
        """Set analysis results to display in header."""
        stats_html = f"""
        <div style="display: flex; gap: 15px; align-items: center;">
            <div style="text-align: center; font-size: 12px;">
                <div style="font-size: 14px; font-weight: bold; color: {ESA_BLUE_ACCENT};">{result.get("num_sources", 0)}</div>
                <div style="color: {TEXT_COLOR_MUTED};">Sources</div>
            </div>
            <div style="text-align: center; font-size: 12px;">
                <div style="font-size: 14px; font-weight: bold; \
color: {ESA_BLUE_ACCENT};">{len(result.get("extensions", []))}</div>
                <div style="color: {TEXT_COLOR_MUTED};">Ext.</div>
            </div>
        </div>
        """
        self.analysis_stats.value = stats_html
        self.analysis_stats.layout.display = "block"

        # Update base config with analysis results, including critical fields for main screen
        self.base_config.update(
            {
                "num_sources": result.get("num_sources", 0),
                "fits_files": result.get("fits_files", []),
                "available_extensions": result.get("extensions", []),
                "selected_extensions": result.get(
                    "extensions", []
                ),  # Initially default to all extensions, but this will be updated by user selection
                "source_catalogue": result.get(
                    "source_catalogue"
                ),  # Now properly included from start screen
            }
        )

        # Update the shared config's config
        self.shared_config.config = self.base_config
        self.shared_config.set_num_sources(result.get("num_sources", 0))

    # Expose commonly accessed properties
    @property
    def format_dropdown(self):
        return self.shared_config.format_dropdown

    @property
    def resolution_input(self):
        return self.shared_config.resolution_input

    @property
    def normalisation_dropdown(self):
        return self.shared_config.normalisation_dropdown
