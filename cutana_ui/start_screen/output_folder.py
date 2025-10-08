#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Output folder selection component."""

import ipywidgets as widgets
from pathlib import Path
from loguru import logger

from ..widgets.file_chooser import CutanaFileChooser
from ..styles import ESA_BLUE_ACCENT, BACKGROUND_DARK, BORDER_COLOR, scale_px
from cutana.get_default_config import get_default_config


class OutputFolderComponent(widgets.VBox):
    """Component for selecting output directory."""

    def __init__(self):
        self.selected_dir = None

        # Title
        self.title = widgets.HTML(
            value=f'<h4 style="color: {ESA_BLUE_ACCENT}; margin-bottom: 5px;">Output Directory</h4>'
        )

        # Get default output directory from centralized config
        cfg = get_default_config()
        self.default_path = cfg.output_dir

        # Directory chooser
        self.dir_chooser = CutanaFileChooser(show_only_dirs=True, select_default=False)

        # Try to set default directory
        try:
            if self.default_path and Path(self.default_path).exists():
                parent_dir = (
                    str(Path(self.default_path).parent)
                    if Path(self.default_path).name == "output"
                    else str(Path(self.default_path))
                )
                self.dir_chooser.file_chooser.default_path = parent_dir
                logger.debug(f"Set default output directory to {parent_dir}")
        except Exception:
            pass

        # Status message
        self.status = widgets.HTML(
            value=(
                f'<p style="color: #52c41a; word-wrap: break-word; word-break: break-all; '
                f'white-space: normal;">Default: {self.default_path}</p>'
            )
        )

        super().__init__(
            children=[self.title, self.dir_chooser, self.status],
            layout=widgets.Layout(
                width="48%",  # Match file selection proportions
                max_width=f"{scale_px(600)}px",
                min_height=f"{scale_px(150)}px",  # Increased by 30px to match file selection
                max_height=f"{scale_px(480)}px",  # Increased proportionally to match file selection
                padding=f"{scale_px(12)}px",  # Match file selection padding
                background=BACKGROUND_DARK,
                border_radius=f"{scale_px(8)}px",
                border=f"1px solid {BORDER_COLOR}",
                margin=f"0 0 {scale_px(10)}px 0",  # Match file selection margin
                overflow="auto",  # Allow scrolling only when needed
            ),
        )

        self._setup_events()

    def _setup_events(self):
        """Set up event handlers."""
        self._last_selected = None  # Track last selection to avoid duplicates

        def on_dir_change(chooser):
            """Handle directory selection changes via register_callback."""
            # Get current selection from dir chooser
            selected_dir = chooser.selected

            logger.debug(f"Directory selection callback triggered: selected={selected_dir}")

            # Check for valid selection
            dir_path = None
            if selected_dir and str(selected_dir).strip() and str(selected_dir) not in ["None", ""]:
                dir_path = str(selected_dir)

            logger.debug(f"Resolved directory path: {dir_path}")

            self.selected_dir = dir_path

        # Also add traditional observe as backup
        event_names = ["selected", "value", "selected_filename"]
        for event_name in event_names:
            try:
                if hasattr(self.dir_chooser.file_chooser, event_name):

                    def make_observer(attr_name):
                        def observe_change(change):
                            logger.debug(f"Observe change on {attr_name}: {change}")
                            # Call the main handler with the chooser object
                            on_dir_change(self.dir_chooser.file_chooser)

                        return observe_change

                    self.dir_chooser.file_chooser.observe(
                        make_observer(event_name), names=[event_name]
                    )
                    logger.debug(f"Added observer for {event_name}")
            except Exception as e:
                logger.debug(f"Could not observe {event_name}: {e}")

        # Button click handlers as additional backup
        try:
            if hasattr(self.dir_chooser.file_chooser, "_select"):

                def on_select_click(button):
                    logger.debug("Directory Select button clicked")
                    # Small delay then check for changes
                    import threading

                    def delayed_check():
                        import time

                        time.sleep(0.5)
                        on_dir_change(self.dir_chooser.file_chooser)

                    threading.Thread(target=delayed_check, daemon=True).start()

                self.dir_chooser.file_chooser._select.on_click(on_select_click)
                logger.debug("✅ Added click handler to directory select button")
        except Exception as e:
            logger.debug(f"Could not add directory button handlers: {e}")

        logger.debug("🔧 Directory selection event setup completed")

    def get_output_dir(self):
        """Get the selected output directory."""
        if self.selected_dir:
            return self.selected_dir

        # Fallback to centralized default
        try:
            cfg = get_default_config()
            return cfg.output_dir
        except Exception as e:
            logger.warning(f"Failed to get default from config: {e}")
            return str(Path.cwd() / "cutana" / "output")
