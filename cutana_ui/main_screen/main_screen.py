#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Main processing screen for Cutana UI."""

import ipywidgets as widgets
from loguru import logger

from cutana.get_default_config import save_config_with_timestamp

from ..styles import (
    BACKGROUND_DARK,
    BORDER_COLOR,
    COMMON_STYLES,
    MAIN_WIDTH,
    PANEL_WIDTH,
    scale_px,
)
from ..utils.log_manager import get_console_log_level, set_console_log_level
from ..widgets.header_version_help import (
    HelpPopup,
    create_header_container,
)
from .configuration_panel import ConfigurationPanel
from .preview_panel import PreviewPanel
from .status_panel import StatusPanel


class MainScreen(widgets.VBox):
    """Main processing interface with configuration, preview, and status panels."""

    def __init__(self, config):
        self.config = config

        # Log selected extensions coming from start screen
        logger.info(
            f"Main screen received {len(self.config.selected_extensions)} selected extensions"
        )
        logger.info(f"Available extensions: {len(self.config.available_extensions)}")

        # Ensure selected_extensions is properly set (should come from start screen)
        if not self.config.selected_extensions and self.config.available_extensions:
            logger.warning("No selected extensions provided, defaulting to all available")
            self.config.selected_extensions = self.config.available_extensions.copy()

        # Apply common styles
        self.style_html = widgets.HTML(value=COMMON_STYLES)

        # Get Cutana version
        try:
            from cutana.__init__ import __version__ as cutana_version

            version_text = f"v{cutana_version}"
        except ImportError:
            logger.warning("Could not import cutana version")
            version_text = "version unknown"

        # Create header container with version, log level dropdown, and help button
        self.header_container, self.help_button, self.log_level_dropdown = create_header_container(
            version_text=version_text,
            container_width=MAIN_WIDTH,
            help_button_callback=self._toggle_help,
            log_level_callback=set_console_log_level,
            logo_title="CUTANA Cutout Generator",
            initial_log_level=get_console_log_level(),
        )

        # Create panels with explicit height ratios
        self.config_panel = ConfigurationPanel(
            config=self.config, on_start=self._on_start_processing, on_stop=self._on_stop_processing
        )
        self.preview_panel = PreviewPanel(config=self.config)
        self.status_panel = StatusPanel(config=self.config)

        # Create standalone start button container
        self.start_button = widgets.Button(
            description="Start Cutout Creation",
            button_style="success",
            layout=widgets.Layout(width="100%", height=f"{scale_px(40)}px", margin=f"0 0 0 0"),
        )
        self.start_button.add_class("cutana-button-primary")
        self.start_button.on_click(self._on_start_click)

        self.start_button_container = widgets.VBox(
            children=[self.start_button],
            layout=widgets.Layout(
                width="100%",
                max_width=f"{PANEL_WIDTH}px",  # Match Configuration Panel width
                margin=f"{scale_px(10)}px 0 0 0",  # Add top margin to create space from config panel
                padding=f"{scale_px(12)}px",
                background=BACKGROUND_DARK,
                border_radius=f"{scale_px(10)}px",
                border=f"1px solid {BORDER_COLOR}",
                align_items="center",
                justify_content="center",
            ),
        )
        self.start_button_container.add_class("cutana-panel")

        # Processing state tracking
        self.is_processing = False

        # Ensure configuration panel is properly updated with full config
        self.config_panel.update_config(self.config)

        # Set fixed heights independent of configuration panel - increased for better preview
        self.preview_panel.layout.height = f"{scale_px(720)}px"  # Increased from 670px
        self.preview_panel.layout.min_height = f"{scale_px(680)}px"  # Increased from 630px
        self.preview_panel.layout.max_height = f"{scale_px(760)}px"  # Increased from 710px

        self.status_panel.layout.height = f"{scale_px(140)}px"  # Increased by 10px from 130px
        self.status_panel.layout.min_height = f"{scale_px(120)}px"  # Increased by 10px from 110px
        self.status_panel.layout.max_height = f"{scale_px(160)}px"  # Increased by 10px from 150px

        # Set up configuration change callback for preview updates
        self.config_panel.set_config_change_callback(self._on_config_change)

        # Configuration panel no longer shows extensions selector - they are set from start screen

        # Create help panel
        self.help_panel = None  # Will be created on demand
        self.showing_help = False

        # Status panel without help button
        # Create left column with config panel and button
        self.left_container = widgets.VBox(
            children=[self.config_panel, self.start_button_container],
            layout=widgets.Layout(
                width="100%",
                max_width=f"{PANEL_WIDTH}px",
                min_height=f"{scale_px(860)}px",  # Minimum height instead of fixed
                overflow="visible",
            ),
        )

        # Create help panel
        self.help_panel = None  # Will be created on demand
        self.showing_help = False

        # Status panel without help button
        self.status_container = widgets.HBox(
            children=[self.status_panel],
            layout=widgets.Layout(
                width="100%",
                justify_content="space-between",
                align_items="flex-start",
            ),
        )

        # Layout: Config panel on left, preview and status on right with flexible height
        self.right_container = widgets.VBox(
            children=[self.preview_panel, self.status_container],
            layout=widgets.Layout(
                width="100%",
                margin=f"0 0 0 {scale_px(10)}px",
                min_height=f"{scale_px(860)}px",  # Minimum height instead of fixed
                overflow="visible",  # Allow content to be visible instead of hidden
            ),
        )

        self.main_container = widgets.HBox(
            children=[self.left_container, self.right_container],
            layout=widgets.Layout(
                width="100%",
                max_width=f"{MAIN_WIDTH}px",
                margin="5px auto 0 auto",  # Reduced top margin
                align_items="flex-start",  # Align panels to top
                justify_content="flex-start",  # Align panels to left
            ),
        )

        super().__init__(
            children=[
                self.style_html,
                self.header_container,
                self.main_container,
            ],
            layout=widgets.Layout(
                width="100%",
                background=BACKGROUND_DARK,
                padding=f"{scale_px(3)}px",  # Reduced padding for tighter spacing
            ),
        )

    def _on_start_click(self, _b):
        """Handle start/stop button click."""
        if self.is_processing:
            # Currently processing - stop
            self._on_stop_processing()
        else:
            # Not processing - start
            self._on_start_processing()

    def _on_start_processing(self):
        """Handle start processing button click."""
        logger.info("Main screen: Start processing clicked")

        # Update button state
        self.set_processing_state(True)

        # Start status panel processing
        self.status_panel.start_processing()

        # Get updated configuration from config panel
        updated_config = self.config_panel.get_current_config()

        # write the config for the user
        config_path = save_config_with_timestamp(updated_config, self.config["output_dir"])
        logger.info(f"Configuration saved to: {config_path}")

        # Call backend to start processing
        import asyncio

        from ..utils.backend_interface import BackendInterface

        async def start_backend():
            try:
                logger.info("Starting backend processing with direct status panel updates...")
                result = await BackendInterface.start_processing(
                    updated_config, status_panel=self.status_panel
                )
                if result.get("status") == "error":
                    logger.error(f"Backend processing failed: {result.get('error')}")
                    self.status_panel.processing_indicator.value = f'<p style="color: #ff6b6b; margin: 3px 0; font-size: 11px;">ERROR: {result.get("error", "Unknown error")}</p>'
                    self.set_processing_state(False)
                    self.status_panel.stop_processing()
                else:
                    logger.info("Backend processing completed successfully")
                    self.status_panel.processing_indicator.value = '<div style="color: #52c41a; margin: 3px 0; font-size: 11px; font-weight: 600;">✓ Processing completed successfully!</div>'
                    self.set_processing_state(False)
            except Exception as e:
                logger.error(f"Failed to start processing: {str(e)}")
                self.status_panel.processing_indicator.value = f'<p style="color: #ff6b6b; margin: 3px 0; font-size: 11px;">ERROR: {str(e)}</p>'
                self.set_processing_state(False)
                self.status_panel.stop_processing()

        # Start the async task with better error handling
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running (like in Jupyter), create task
                asyncio.create_task(start_backend())
            else:
                # If no loop is running, start it
                loop.run_until_complete(start_backend())
        except RuntimeError as e:
            # No event loop available, try to create one
            try:
                logger.warning(f"No event loop available ({e}), creating new one")
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(start_backend())
                new_loop.close()
            except Exception as e2:
                logger.error(f"Failed to create event loop: {e2}")
                self.status_panel.processing_indicator.value = f'<p style="color: #ff6b6b; margin: 3px 0; font-size: 11px;">ERROR: Cannot start processing - {str(e2)}</p>'
                self.set_processing_state(False)
                self.status_panel.stop_processing()

    def _on_stop_processing(self):
        """Handle stop processing request from UI."""
        logger.info("MainScreen: Stop processing requested")

        # Update UI state immediately
        self.set_processing_state(False)

        # Update status panel UI
        self.status_panel.stop_processing()

        # Actually stop the backend processing
        import asyncio

        from ..utils.backend_interface import BackendInterface

        async def stop_backend():
            try:
                logger.info("Stopping backend processing...")
                result = await BackendInterface.stop_processing()
                logger.info(f"Backend stop result: {result}")
            except Exception as e:
                logger.error(f"Error stopping backend: {e}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(stop_backend())
            else:
                loop.run_until_complete(stop_backend())
        except Exception as e:
            logger.error(f"Failed to stop backend: {e}")

    def set_processing_state(self, is_processing):
        """Update button state based on processing status."""
        self.is_processing = is_processing
        if is_processing:
            self.start_button.description = "Stop Processing"
            self.start_button.button_style = "danger"
        else:
            self.start_button.description = "Start Cutout Creation"
            self.start_button.button_style = "success"

    def _on_config_change(self):
        """Handle configuration changes - update preview panel."""
        logger.debug("MainScreen: Configuration change callback triggered")

        # Get current config and update preview panel
        current_config = self.config_panel.get_current_config()
        logger.debug(
            f"MainScreen: Got current config, normalisation_method: {current_config.normalisation_method}"
        )
        logger.debug(
            f"MainScreen: Normalisation params - a: {current_config.normalisation.a}, percentile: {current_config.normalisation.percentile}"
        )

        self.preview_panel.update_config(current_config)

        # Trigger preview regeneration if config affects preview
        logger.debug("MainScreen: Triggering preview regeneration")
        self.preview_panel.regenerate_preview()

    def _toggle_help(self, _):
        """Toggle between help panel and preview panel."""
        if self.showing_help:
            self._hide_help()
        else:
            self._show_help()

    def _show_help(self):
        """Replace the preview panel with the help panel."""
        logger.info("Showing help panel")

        # Create help panel if it doesn't exist
        if not self.help_panel:
            self.help_panel = HelpPopup(on_close_callback=self._hide_help)
            self.help_panel.layout.height = self.preview_panel.layout.height
            self.help_panel.layout.min_height = self.preview_panel.layout.min_height
            self.help_panel.layout.max_height = self.preview_panel.layout.max_height

        # Save current children
        current_children = list(self.right_container.children)

        # Replace preview panel with help panel
        current_children[0] = self.help_panel
        self.right_container.children = current_children

        # Update help button text
        self.help_button.description = "Close Help"

        # Set state
        self.showing_help = True

    def _hide_help(self):
        """Replace the help panel with the preview panel."""
        logger.info("Hiding help panel")

        # Save current children
        current_children = list(self.right_container.children)

        # Replace help panel with preview panel
        current_children[0] = self.preview_panel
        self.right_container.children = current_children

        # Update help button text
        self.help_button.description = "Help"

        # Set state
        self.showing_help = False
