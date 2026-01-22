#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Main application entry point for Cutana UI."""

import ipywidgets as widgets
from IPython.display import display
from loguru import logger

from .main_screen import MainScreen
from .start_screen import StartScreen
from .styles import BACKGROUND_DARK
from .utils.log_manager import setup_ui_logging

# Global UI scaling factor - can be modified by the CutanaApp
UI_SCALE = 0.75  # Default value


class CutanaApp:
    """Main application class for Cutana UI."""

    def __init__(self, ui_scale=0.75):
        global UI_SCALE
        UI_SCALE = ui_scale

        # Update styles module to use the new scale
        self._update_styles_scale(ui_scale)

        self.config_data = {}

        # Create main container with proper styling
        self.container = widgets.VBox(
            layout=widgets.Layout(
                width="100%",
                background=BACKGROUND_DARK,
                padding="0",
            )
        )
        self.container.add_class("cutana-container")

        # Initialize with start screen
        self._show_start_screen()

    def _show_start_screen(self):
        """Show the unified start screen."""
        start_screen = StartScreen(on_complete=self._on_configuration_complete)
        self.container.children = [start_screen]

    def _on_configuration_complete(self, full_config):
        """Handle configuration completion and show main screen."""
        logger.debug(f"Channel configuration complete")

        # Reconfigure logging to use the output directory from config
        session_timestamp = getattr(full_config, "session_timestamp", None)
        setup_ui_logging(full_config.output_dir, session_timestamp)
        logger.info(f"UI logging reconfigured to use output directory: {full_config.output_dir}")

        # Show main screen
        main_screen = MainScreen(config=full_config)
        self.container.children = [main_screen]

    def _update_styles_scale(self, ui_scale):
        """Update the styles module with the new UI scale."""
        from . import styles

        styles.set_ui_scale(ui_scale)


def start(ui_scale=0.75):
    """Start the Cutana UI application.

    Args:
        ui_scale (float): UI scaling factor for different screen sizes.
                         Default is 0.75. Common values:
                         - 0.75 for 1920x1080 screens

    """
    # limit ui scale to range 0.6-1.0
    ui_scale = max(0.6, min(1.0, ui_scale))

    logger.debug(f"Starting Cutana UI with UI scale: {ui_scale}")

    # Create and display the app
    app = CutanaApp(ui_scale=ui_scale)
    display(app.container)

    return app
