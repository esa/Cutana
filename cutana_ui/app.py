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
import sys
from pathlib import Path

from .start_screen import StartScreen
from .main_screen import MainScreen
from .styles import BACKGROUND_DARK

# Global UI scaling factor - can be modified by the CutanaApp
UI_SCALE = 0.75  # Default value


def setup_ui_logging(output_dir=None, session_timestamp=None):
    """Set up loguru logging for UI with optional output directory and separate UI files."""
    try:

        if output_dir is not None:
            # Use output directory for logs when available
            log_dir = Path(output_dir) / "logs"
        else:
            # Fallback to current directory for initial UI setup
            log_dir = Path.cwd() / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing handlers
        logger.remove()

        # Add console handler (warnings/errors only)
        logger.add(
            sys.stderr,
            level="WARNING",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )

        # Create timestamp for UI log files
        if session_timestamp is None:
            from datetime import datetime

            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Add UI-specific log file
        logger.add(
            str(log_dir / f"ui_{session_timestamp}.log"),
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,
        )

        # Add UI error log file
        logger.add(
            str(log_dir / f"ui_errors_{session_timestamp}.log"),
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="5 MB",
            retention="30 days",
            compression="zip",
            enqueue=True,
        )

        logger.debug(f"UI logging configured with directory: {log_dir}")

    except ImportError:
        # Fallback if cutana.logging_config not available
        logger.remove()
        logger.add(sys.stderr, level="WARNING", colorize=True)
        logger.warning("Using fallback UI logging - cutana.logging_config not available")


# Set up initial logging when module is imported (without output dir)
setup_ui_logging()


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
                min_height="100vh",
                overflow="auto",
            )
        )
        self.container.add_class("cutana-container")

        # Initialize with start screen
        self._show_start_screen()

    def _show_start_screen(self):
        """Show the unified start screen."""
        start_screen = StartScreen(on_complete=self._on_configuration_complete)
        self.container.children = [start_screen]

    def _on_configuration_complete(self, full_config, config_path):
        """Handle configuration completion and show main screen."""
        logger.debug(f"Configuration complete, config saved to: {config_path}")

        # Reconfigure logging to use the output directory from config
        session_timestamp = getattr(full_config, "session_timestamp", None)
        setup_ui_logging(full_config.output_dir, session_timestamp)
        logger.info(f"UI logging reconfigured to use output directory: {full_config.output_dir}")

        # Show main screen
        main_screen = MainScreen(config=full_config, config_path=config_path)
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
