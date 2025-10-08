#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unified start screen combining file selection, analysis, and configuration."""

import ipywidgets as widgets
import asyncio
from loguru import logger

from .file_selection import FileSelectionComponent
from .configuration_component import ConfigurationComponent
from .output_folder import OutputFolderComponent
from ..widgets.loading_spinner import LoadingSpinner
from ..widgets.header_version_help import (
    HelpPopup,
    create_header_container,
)
from ..utils.backend_interface import BackendInterface
from cutana.get_default_config import get_default_config, save_config_with_timestamp
from ..styles import (
    COMMON_STYLES,
    BACKGROUND_DARK,
    CONTAINER_WIDTH,
    CONTAINER_HEIGHT,
    scale_px,
    scale_vh,
)


class StartScreen(widgets.VBox):
    """Unified start screen with all configuration in one view."""

    def __init__(self, on_complete=None):
        self.on_complete = on_complete
        self.config_data = get_default_config()
        self.analysis_result = None

        # Apply common styles
        self.style_html = widgets.HTML(value=COMMON_STYLES)

        # Get Cutana version
        try:
            from cutana.__init__ import __version__ as cutana_version

            version_text = f"v{cutana_version}"
        except ImportError:
            logger.warning("Could not import cutana version")
            version_text = "version unknown"

        # Create header container with version and help button
        self.header_container, self.help_button = create_header_container(
            version_text=version_text,
            container_width=CONTAINER_WIDTH,
            help_button_callback=self._toggle_help,
            logo_title="CUTANA Cutout Generator Configuration",
        )

        # Create help panel
        self.help_panel = None  # Will be created on demand
        self.showing_help = False
        self.original_layout = None  # Store original layout when showing help

        # Main container with vertical layout - reduced padding and margin
        self.main_container = widgets.VBox(
            layout=widgets.Layout(
                width="100%",
                max_width=f"{CONTAINER_WIDTH}px",
                height=f"{CONTAINER_HEIGHT}px",
                margin="5px auto 0 auto",  # Reduced top margin
                padding=f"{scale_px(3)}px",  # Reduced padding
                background=BACKGROUND_DARK,
                border_radius=f"{scale_px(5)}px",
            )
        )

        # Top section for file selection and output folder side by side
        self.top_section = widgets.HBox(
            layout=widgets.Layout(
                width="100%",
                padding=f"{scale_px(3)}px",  # Reduced padding
                justify_content="center",  # Center the components
                gap=f"{scale_px(20)}px",  # Gap between components
            )
        )

        # Bottom section for configuration and start button side by side
        self.bottom_section = widgets.HBox(
            layout=widgets.Layout(
                width="100%",
                padding=f"{scale_px(3)}px",  # Reduced padding
                justify_content="center",  # Center the overall layout
                gap=f"{scale_px(20)}px",  # Gap between components
                align_items="flex-start",
            )
        )

        # Components
        self.file_selection = FileSelectionComponent(on_file_selected=self._on_file_selected)
        # analysis_display is now integrated into file_selection
        self.configuration = ConfigurationComponent()
        self.output_folder = OutputFolderComponent()

        # Configuration loading spinner (shown during analysis)
        self.config_loading_spinner = LoadingSpinner("Analysing catalogue...")
        self.config_loading_spinner.layout.display = "none"

        # Initially hide configuration until analysis completes
        self.configuration.layout.display = "none"

        # Start button (initially hidden) - positioned in bottom right
        self.start_button = widgets.Button(
            description="Start Cutana",
            button_style="primary",
            layout=widgets.Layout(width="200px", height="50px", display="none"),
        )

        # No longer need separate button container since button goes in config component
        self.start_button.add_class("cutana-button-primary")
        self.start_button.on_click(self._on_start_click)

        # Error message
        self.error_message = widgets.HTML(value="")

        # Layout components
        self._update_layout()

        super().__init__(
            children=[
                self.style_html,
                self.header_container,
                self.main_container,
            ],
            layout=widgets.Layout(
                width="100%",
                min_height=f"{scale_vh(95)}vh",
                background=BACKGROUND_DARK,
                padding=f"{scale_px(3)}px",  # Reduced padding for tighter spacing
            ),
        )

    def _update_layout(self):
        """Update the layout of components with proper horizontal arrangements."""
        # Top section: file selection and output folder side by side
        self.top_section.children = [self.file_selection, self.output_folder]

        # Bottom section handling
        if self.analysis_result:
            # Analysis complete - show configuration with embedded button and pass the button to it
            self.configuration.set_start_button(self.start_button)
            self.bottom_section.children = [self.configuration]
        elif (
            hasattr(self, "config_loading_spinner")
            and self.config_loading_spinner.layout.display == "block"
        ):
            # Show loading spinner (centered)
            loading_container = widgets.HBox(
                children=[self.config_loading_spinner],
                layout=widgets.Layout(justify_content="center", width="100%"),
            )
            self.bottom_section.children = [loading_container]
        else:
            self.bottom_section.children = []

        # Add error message if present
        if self.error_message.value:
            # Create error container that spans full width
            current_children = list(self.bottom_section.children)
            error_section = widgets.VBox(
                children=current_children + [self.error_message],
                layout=widgets.Layout(width="100%"),
            )
            self.bottom_section.children = [error_section]

        # Main container: both sections vertically
        self.main_container.children = [
            self.top_section,
            self.bottom_section,
        ]

    def _on_file_selected(self, file_path):
        """Handle file selection and start analysis."""
        logger.info(f"🎯 StartScreen._on_file_selected called with: {file_path}")
        self.config_data["source_catalogue"] = file_path

        # Show immediate visual feedback in config section where config will appear
        self.config_loading_spinner.layout.display = "block"
        self._update_layout()
        logger.info("📄 Showing loading state in config section")

        # Start analysis
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create task
                asyncio.create_task(self._analyze_catalogue(file_path))
                logger.info("📋 Created analysis task in running loop")
            else:
                # No running loop, run directly
                loop.run_until_complete(self._analyze_catalogue(file_path))
                logger.info("📋 Ran analysis directly in new loop")
        except RuntimeError:
            # No event loop, create new one
            logger.info("📋 No event loop found, creating new one for analysis")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._analyze_catalogue(file_path))
                logger.info("✅ Analysis completed in new loop")
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"❌ Error starting analysis: {e}")
            self.error_message.value = (
                f'<p style="color: red;">Error starting analysis: {str(e)}</p>'
            )

    async def _analyze_catalogue(self, file_path):
        """Analyze the selected catalogue."""
        logger.info(f"🔍 Starting catalogue analysis for: {file_path}")
        try:
            # Perform analysis
            logger.info("📞 Calling BackendInterface.check_source_catalogue...")
            result = await BackendInterface.check_source_catalogue(file_path)
            self.analysis_result = result
            logger.info(
                f"✅ Analysis completed: {result.get('num_sources', 0)} sources, {len(result.get('extensions', []))} extensions"
            )

            # Hide config loading spinner and update file selection
            self.config_loading_spinner.layout.display = "none"
            self.file_selection.show_analysis_results(result)
            logger.info("📊 Analysis results displayed in file selection component")

            # Show configuration panel
            extensions = result.get("extensions", [])
            num_sources = result.get("num_sources", 0)
            self.configuration.set_extensions(extensions)
            self.configuration.set_num_sources(num_sources)

            # Add source_catalogue to result before passing to configuration
            result_with_catalogue = result.copy()
            result_with_catalogue["source_catalogue"] = self.config_data["source_catalogue"]
            self.configuration.set_analysis_results(
                result_with_catalogue
            )  # Add analysis stats to header
            self.configuration.layout.display = "block"
            self.start_button.layout.display = "block"
            logger.info(
                f"⚙️ Configuration panel shown with {len(extensions)} extensions and {num_sources} sources"
            )

            # Update config data
            self.config_data.update(
                {
                    "num_sources": result.get("num_sources", 0),
                    "fits_files": result.get("fits_files", []),
                    "available_extensions": result.get("extensions", []),
                }
            )

            # Update layout to show everything
            self._update_layout()
            logger.info("🎨 Layout updated - analysis complete!")

        except ValueError as e:
            # Handle validation errors specifically
            error_msg = str(e)
            if "Catalogue validation failed:" in error_msg:
                # Extract just the validation message part
                validation_error = error_msg.replace("Catalogue validation failed: ", "")
                logger.error(f"❌ Catalogue validation failed: {validation_error}")
                self.config_loading_spinner.layout.display = "none"
                self.file_selection.show_error(validation_error)
            else:
                logger.error(f"❌ Catalogue loading error: {e}")
                self.config_loading_spinner.layout.display = "none"
                self.file_selection.show_error(f"Failed to load catalogue: {str(e)}")
            self._update_layout()

        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Show generic error message
            self.config_loading_spinner.layout.display = "none"
            self.file_selection.show_error(f"Unexpected error during catalogue analysis: {str(e)}")
            self._update_layout()

    def _on_start_click(self, b):
        """Handle start button click."""
        try:
            # Gather all configuration
            config = self.configuration.get_configuration()
            config["output_dir"] = self.output_folder.get_output_dir()

            # Validate
            if not config.get("selected_extensions"):
                self.error_message.value = (
                    '<p style="color: red;">Please select at least one FITS extension.</p>'
                )
                return

            if not config.get("output_dir"):
                self.error_message.value = (
                    '<p style="color: red;">Please select an output directory.</p>'
                )
                return

            # Get default config and update with UI values
            full_config = get_default_config()

            # Update config with UI values using consistent parameter names
            for key, value in self.config_data.items():
                if hasattr(full_config, key):
                    setattr(full_config, key, value)

            for key, value in config.items():
                # Map UI parameter names to backend names
                if key == "output_resolution":
                    full_config.target_resolution = value
                elif key == "num_workers":
                    full_config.max_workers = value
                elif key == "normalize_method":
                    full_config.normalisation_method = value
                elif key == "channel_matrix":
                    # Ensure channel matrix is properly transferred
                    full_config.channel_matrix = value
                elif key == "num_channels":
                    # Ensure num_channels is properly transferred
                    full_config.num_channels = value
                elif hasattr(full_config, key):
                    setattr(full_config, key, value)

            # CRITICAL: Ensure only selected extensions persist to main screen
            selected_ext_names = [ext.get("name", "") for ext in config["selected_extensions"]]
            available_ext_names = [
                ext.get("name", "") for ext in self.config_data.available_extensions
            ]
            logger.info(
                f"Selected extensions from start screen: {selected_ext_names} (out of {available_ext_names})"
            )

            # Save configuration with timestamp
            config_path = save_config_with_timestamp(full_config, config["output_dir"])
            logger.info(f"Configuration saved to: {config_path}")

            # Proceed to main screen
            if self.on_complete:
                self.on_complete(full_config, config_path)

        except Exception as e:
            logger.error(f"Error starting Cutana: {e}")
            self.error_message.value = f'<p style="color: red;">Error: {str(e)}</p>'

    def _toggle_help(self, _):
        """Toggle between help panel and main content."""
        if self.showing_help:
            self._hide_help()
        else:
            self._show_help()

    def _show_help(self):
        """Show full-page help panel."""
        logger.info("Showing help panel")

        # Create help panel if it doesn't exist
        if not self.help_panel:
            self.help_panel = HelpPopup(on_close_callback=self._hide_help)
            self.help_panel.layout.width = "100%"
            self.help_panel.layout.height = f"{CONTAINER_HEIGHT}px"
            self.help_panel.layout.max_width = f"{CONTAINER_WIDTH}px"
            self.help_panel.layout.margin = "0 auto"

        # Save original content
        self.original_layout = self.main_container

        # Replace main container with help panel
        self.children = [
            self.style_html,
            self.header_container,
            self.help_panel,
        ]

        # Update help button text
        self.help_button.description = "Close Help"

        # Set state
        self.showing_help = True

    def _hide_help(self):
        """Hide help panel and show main content."""
        logger.info("Hiding help panel")

        # Restore original layout
        self.children = [
            self.style_html,
            self.header_container,
            self.main_container,
        ]

        # Update help button text
        self.help_button.description = "Help"

        # Set state
        self.showing_help = False
