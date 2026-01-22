#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""File selection component for start screen."""

from pathlib import Path

import ipywidgets as widgets
from loguru import logger

from ..styles import BACKGROUND_DARK, BORDER_COLOR, ESA_BLUE_ACCENT, TEXT_COLOR_LIGHT, scale_px
from ..widgets.file_chooser import CutanaFileChooser


class FileSelectionComponent(widgets.VBox):
    """Component for selecting source catalogue file (CSV or Parquet)."""

    def __init__(self, on_file_selected=None):
        self.on_file_selected = on_file_selected
        self.analysis_result = None

        # Title
        self.title = widgets.HTML(
            value=f'<h3 style="color: {ESA_BLUE_ACCENT}; margin-bottom: 2px;">Source Catalogue Selection</h3>'
        )

        # Instructions
        self.instructions = widgets.HTML(
            value=f"""
            <p style="color: {TEXT_COLOR_LIGHT}; font-size: 12px; margin-bottom: 2px;">
                Select a CSV or Parquet file containing source catalogue data (see help for format).
            </p>
            """
        )

        # File chooser - CSV and Parquet formats
        self.file_chooser = CutanaFileChooser(filter_pattern=["*.csv", "*.parquet"])

        # Error display (initially hidden)
        self.error_display = widgets.HTML(
            value="",
            layout=widgets.Layout(
                display="none",
                width="100%",
                padding=f"{scale_px(10)}px",
                margin=f"{scale_px(5)}px 0",
                border_radius=f"{scale_px(5)}px",
                background_color="#ffebee",
                border="1px solid #f44336",
            ),
        )

        super().__init__(
            children=[
                self.title,
                self.instructions,
                self.file_chooser,
                self.error_display,
            ],
            layout=widgets.Layout(
                width="48%",  # Match output folder proportions
                max_width=f"{scale_px(600)}px",
                min_height=f"{scale_px(150)}px",  # Increased by 30px for more vertical space
                max_height=f"{scale_px(480)}px",  # Increased proportionally
                padding=f"{scale_px(12)}px",
                background=BACKGROUND_DARK,
                border_radius=f"{scale_px(8)}px",
                border=f"1px solid {BORDER_COLOR}",
                margin=f"0 0 {scale_px(10)}px 0",
            ),
        )

        self._setup_events()

    def _setup_events(self):
        """Set up event handlers."""
        self._last_selected = None  # Track last selection to avoid duplicates

        def on_file_change(chooser):
            """Handle file selection changes via register_callback."""
            try:
                # Get current selection from file chooser
                selected_file = chooser.selected
                selected_filename = chooser.selected_filename

                logger.info(
                    f"File selection callback triggered: selected={selected_file}, filename={selected_filename}"
                )

                # Check for valid selection
                file_path = None
                if (
                    selected_file
                    and str(selected_file).strip()
                    and str(selected_file) not in ["None", ""]
                ):
                    file_path = str(selected_file)
                elif (
                    selected_filename
                    and str(selected_filename).strip()
                    and str(selected_filename) not in ["None", ""]
                ):
                    # Try to construct full path using the file chooser's current path
                    try:
                        current_path = getattr(chooser, "path", Path.cwd())
                        file_path = str(Path(current_path) / selected_filename)
                    except Exception as e:
                        logger.debug(f"Could not construct full file path: {e}")
                        file_path = str(selected_filename)

                logger.info(f"Resolved file path: {file_path}")

                # Avoid duplicate triggers and validate path
                if file_path and file_path != self._last_selected:
                    self._last_selected = file_path

                    if file_path.lower().endswith((".csv", ".parquet")):
                        logger.info(f"✅ Catalogue file selected: {file_path}")

                        # Trigger callback for automatic analysis
                        if self.on_file_selected:
                            logger.info(f"🚀 Triggering file selection callback for: {file_path}")
                            try:
                                self.on_file_selected(file_path)
                                logger.info("✅ File selection callback completed successfully")
                            except Exception as e:
                                logger.error(f"❌ Error in file selection callback: {e}")
                                import traceback

                                logger.error(traceback.format_exc())
                    else:
                        logger.warning(f"Non-catalogue file selected: {file_path}")
                else:
                    if not file_path:
                        logger.debug("File path skipped - empty or invalid selection")
                    else:
                        logger.debug(f"File path skipped - already processed: {file_path}")

            except Exception as e:
                logger.error(f"❌ Error in file selection handler: {e}")
                import traceback

                logger.error(traceback.format_exc())

        # Use the correct ipyfilechooser callback registration method
        try:
            self.file_chooser.file_chooser.register_callback(on_file_change)
            logger.debug("✅ Registered file selection callback with ipyfilechooser")
        except Exception as e:
            logger.error(f"❌ Could not register callback: {e}")

        # Also add traditional observe as backup
        event_names = ["selected", "value", "selected_filename"]
        for event_name in event_names:
            try:
                if hasattr(self.file_chooser.file_chooser, event_name):

                    def make_observer(attr_name):
                        def observe_change(change):
                            logger.debug(f"Observe change on {attr_name}: {change}")
                            # Call the main handler with the chooser object
                            on_file_change(self.file_chooser.file_chooser)

                        return observe_change

                    self.file_chooser.file_chooser.observe(
                        make_observer(event_name), names=[event_name]
                    )
                    logger.debug(f"Added observer for {event_name}")
            except Exception as e:
                logger.debug(f"Could not observe {event_name}: {e}")

        # Button click handlers as additional backup
        try:
            if hasattr(self.file_chooser.file_chooser, "_select"):

                def on_select_click(button):
                    logger.debug("Select button clicked")
                    # Small delay then check for changes
                    import threading

                    def delayed_check():
                        import time

                        time.sleep(0.5)
                        on_file_change(self.file_chooser.file_chooser)

                    threading.Thread(target=delayed_check, daemon=True).start()

                self.file_chooser.file_chooser._select.on_click(on_select_click)
                logger.debug("✅ Added click handler to select button")
        except Exception as e:
            logger.debug(f"Could not add button handlers: {e}")

        logger.debug("🔧 File selection event setup completed")

    def show_analysis_results(self, result):
        """Hide loading spinner when analysis completes successfully."""
        self.analysis_result = result
        self.hide_error()  # Hide any previous errors

    def show_error(self, error_message: str):
        """
        Show validation error message to user.

        Args:
            error_message: Error message to display
        """
        logger.error(f"Displaying error to user: {error_message}")

        # Format error message with nice styling
        # Fix f-string issue by doing replacement outside the f-string
        error_html = error_message.replace("\n", "<br>")
        formatted_message = f"""
        <div style="color: #d32f2f; font-weight: bold; margin-bottom: 8px;">
            ❌ Catalogue Validation Failed
        </div>
        <div style="color: #666; font-size: 12px; line-height: 1.4;">
            {error_html}
        </div>
        <div style="color: #999; font-size: 11px; margin-top: 8px; font-style: italic;">
            Please fix the issues above and select a new catalogue file.
        </div>
        """

        self.error_display.value = formatted_message
        self.error_display.layout.display = "block"
        self.hide_loading()  # Hide loading spinner

    def hide_error(self):
        """Hide error display."""
        self.error_display.layout.display = "none"
        self.error_display.value = ""
