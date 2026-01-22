#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Preview panel for the main screen."""

import asyncio
import base64
import io

import ipywidgets as widgets
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.figure import Figure

from ..styles import (
    BACKGROUND_DARK,
    BORDER_COLOR,
    ESA_BLUE_ACCENT,
    TEXT_COLOR_MUTED,
    scale_px,
)
from ..utils.backend_interface import BackendInterface
from ..widgets.loading_spinner import LoadingSpinner


class PreviewPanel(widgets.VBox):
    """Panel showing preview of cutouts."""

    def __init__(self, config):
        self.config = config
        self.preview_cutouts = []

        # Title
        self.title = widgets.HTML(
            value=f'<h2 style="color: {ESA_BLUE_ACCENT}; margin: 0; font-size: {scale_px(20)}px;">Cutout Preview</h2>'
        )

        # Refresh button in top-right
        self.refresh_button = widgets.Button(
            description="",
            icon="refresh",
            layout=widgets.Layout(
                width=f"{scale_px(32)}px",
                height=f"{scale_px(32)}px",
                margin="0px",
                min_width=f"{scale_px(32)}px",
            ),
        )
        self.refresh_button.style.button_color = ESA_BLUE_ACCENT
        self.refresh_button.on_click(self._on_refresh_clicked)

        # Header container with title and refresh button - minimal margin
        self.header = widgets.HBox(
            children=[self.title, self.refresh_button],
            layout=widgets.Layout(
                justify_content="space-between",
                align_items="center",
                margin=f"0px 0px {scale_px(2)}px 0px",
                overflow="hidden",
            ),
        )

        # Info text (initially hidden)
        self.info_text = widgets.HTML(value="", layout=widgets.Layout(display="none"))

        # Preview container - 3x5 grid with better spacing for RA/Dec labels
        self.preview_container = widgets.GridBox(
            layout=widgets.Layout(
                grid_template_columns="repeat(5, 1fr)",
                grid_template_rows="repeat(3, minmax(0, 1fr))",  # 3 rows instead of 2
                grid_gap=f"{scale_px(2)}px",  # Small but visible spacing
                margin=f"{scale_px(2)}px 0",  # Small margin for breathing room
                display="none",
                width="100%",
                height="100%",  # Use all available space from flex layout
                overflow="hidden",
            )
        )

        # Loading spinner (initially hidden)
        self.loading_spinner = LoadingSpinner("Preparing preview...")
        self.loading_spinner.layout.display = "none"

        # Container - fixed size with more breathing room for cutouts
        super().__init__(
            children=[
                self.header,
                self.info_text,
                self.loading_spinner,
                self.preview_container,
            ],
            layout=widgets.Layout(
                padding=f"{scale_px(8)}px {scale_px(10)}px",  # Increased padding for better spacing
                background=BACKGROUND_DARK,
                border_radius=f"{scale_px(10)}px",
                border=f"1px solid {BORDER_COLOR}",
                width="100%",
                height="100%",  # Fill the fixed height from main screen
                overflow="hidden",
            ),
        )
        self.add_class("cutana-panel")

        # Set up event handlers
        self._setup_events()

        # Auto-generate initial preview on first launch
        self._auto_generate_initial_preview()

    def _setup_events(self):
        """Set up event handlers."""
        pass

    def _on_refresh_clicked(self, _b):
        """Handle refresh button click - generate new samples."""
        # Generate new seed to get fresh sample selection
        BackendInterface.regenerate_preview_seed()
        self.regenerate_preview()

    def _update_progress(self, message: str):
        """Update the loading spinner with progress message."""
        self.loading_spinner.update_message(message)

    async def load_preview_sources(self):
        """Load and cache sources for fast preview generation."""
        try:
            if not self.config.source_catalogue:
                logger.info("No source catalogue available for preview source loading")
                return

            logger.info("Loading preview sources into cache...")
            cache_info = await BackendInterface.load_sources_for_previews(
                catalogue_path=self.config.source_catalogue,
                config=self.config,
                progress_callback=self._update_progress,
            )

            logger.info(
                f"Preview source cache loaded: {cache_info['num_cached_sources']} "
                f"sources, {cache_info['num_cached_fits']} FITS files"
            )
            return cache_info

        except Exception as e:
            logger.warning(f"Failed to load preview sources: {e}")
            return None

    def _generate_preview(self):
        """Generate preview cutouts."""

        async def generate():
            try:
                # Show loading spinner
                self.loading_spinner.layout.display = "block"
                self.preview_container.layout.display = "none"

                # Get fast preview cutouts from cache, fallback to full pipeline if needed
                cutouts = await BackendInterface.generate_previews(
                    num_samples=15,  # 3 rows × 5 columns
                    size=self.config.target_resolution,
                    config=self.config,
                    progress_callback=self._update_progress,
                )

                # Create preview widgets
                preview_widgets = []
                for ra, dec, cutout_array in cutouts:
                    preview_widget = self._create_preview_widget(ra, dec, cutout_array)
                    preview_widgets.append(preview_widget)

                # Update container
                self.preview_container.children = preview_widgets

                # Hide spinner, show previews, and clear any error messages
                self.loading_spinner.layout.display = "none"
                self.preview_container.layout.display = "grid"
                self.info_text.layout.display = "none"  # Clear any previous error messages

            except Exception as e:
                self.loading_spinner.layout.display = "none"
                self.preview_container.layout.display = "none"  # Hide preview on error
                self.info_text.layout.display = "block"
                self.info_text.value = f'<p style="color: #ff6b6b; margin: 5px 0;">Error generating preview: {str(e)}</p>'

        # Run async task - handle case where no event loop is running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(generate())
            else:
                loop.run_until_complete(generate())
        except RuntimeError as e:
            # No event loop available
            self.info_text.layout.display = "block"
            self.info_text.value = (
                f'<p style="color: #ff6b6b; margin: 5px 0;">Error generating preview: {str(e)}</p>'
            )
        except Exception as e:
            # Other errors during generation
            self.info_text.layout.display = "block"
            self.info_text.value = (
                f'<p style="color: #ff6b6b; margin: 5px 0;">Error generating preview: {str(e)}</p>'
            )

    def _create_preview_widget(self, ra, dec, cutout_array):
        """Create a preview widget for a single cutout."""
        # Create figure with smaller size for 3-row layout
        fig = Figure(figsize=(1.4, 1.4), dpi=100)
        ax = fig.add_subplot(111)

        # Display cutout based on number of dimensions and channels
        if len(cutout_array.shape) == 2:
            # 2D array - single channel grayscale
            ax.imshow(cutout_array, cmap="gray", origin="lower")
        elif len(cutout_array.shape) == 3:
            # 3D array - check number of channels
            num_channels = cutout_array.shape[2]

            if num_channels == 1:
                # Single channel in 3D format - show as grayscale
                ax.imshow(cutout_array[:, :, 0], cmap="gray", origin="lower")
            elif num_channels == 2:
                # Two channels - show as grayscale (average)
                cutout_display = cutout_array.mean(axis=2)
                ax.imshow(cutout_display, cmap="gray", origin="lower")
            elif num_channels == 3:
                # Three channels - show as RGB color
                ax.imshow(cutout_array, origin="lower")
            else:
                # More than 3 channels - use first 3 as RGB
                ax.imshow(cutout_array[:, :, :3], origin="lower")
        else:
            # Unexpected shape - try to display anyway
            logger.warning(f"Unexpected cutout shape: {cutout_array.shape}")
            ax.imshow(cutout_array, cmap="gray", origin="lower")

        ax.axis("off")
        fig.tight_layout(pad=0)

        # Convert to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black")
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        # Create HTML widget with better spacing for readable labels
        html = f"""
        <div style="background: {BACKGROUND_DARK}; border-radius: 3px; 
        padding: 3px; text-align: center; border: 1px solid {BORDER_COLOR};">
            <img src="data:image/png;base64,{img_data}" style="width: 100%; height: auto; border-radius: 2px; display: block;">
            <div style="color: {TEXT_COLOR_MUTED}; font-size: {scale_px(16)}px; margin-top: 2px; line-height: 1.0;">
                {ra:.4f}°, {dec:.4f}°
            </div>
        </div>
        """

        return widgets.HTML(value=html)

    def _auto_generate_initial_preview(self):
        """Auto-generate preview on first launch if config is available."""
        logger.debug(
            f"Preview auto-generation check: catalogue={self.config.source_catalogue}, "
            f"selected_extensions={self.config.selected_extensions}, "
            f"available_extensions={self.config.available_extensions}"
        )

        # Only auto-generate if we have a source catalogue AND selected extensions
        if (
            self.config.source_catalogue
            and self.config.selected_extensions
            and len(self.config.selected_extensions) > 0
        ):
            logger.info("Starting auto-generation of initial preview")
            try:
                # Show loading spinner before starting
                self.loading_spinner.layout.display = "block"
                self.preview_container.layout.display = "none"
                self.info_text.layout.display = "none"

                # Load sources first, then generate preview
                async def load_and_generate():
                    try:
                        await self.load_preview_sources()
                        self._generate_preview()
                    except Exception as e:
                        logger.error(f"Error during initial preview generation: {e}")
                        # Hide spinner and show error
                        self.loading_spinner.layout.display = "none"
                        self.info_text.layout.display = "block"
                        self.info_text.value = (
                            f'<p style="color: #ff6b6b; margin: 5px 0;">'
                            f"Error generating initial preview: {str(e)}</p>"
                        )

                # Run async task
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(load_and_generate())
                    else:
                        loop.run_until_complete(load_and_generate())
                except RuntimeError:
                    # No event loop available, skip auto-generation and hide spinner
                    logger.warning("No event loop available for preview auto-generation")
                    self.loading_spinner.layout.display = "none"

            except Exception as e:
                # If auto-generation fails, hide spinner
                logger.error(f"Failed to start preview auto-generation: {e}")
                self.loading_spinner.layout.display = "none"
        else:
            logger.info(
                f"Skipping preview auto-generation: catalogue={self.config.source_catalogue}, "
                f"selected_extensions={self.config.selected_extensions}"
            )
            # Show info message about missing requirements
            self.info_text.layout.display = "block"
            self.info_text.value = (
                '<p style="color: #faad14; margin: 5px 0;">'
                "Preview will be available after catalogue analysis and extension selection</p>"
            )

    def update_config(self, new_config):
        """Update configuration for preview generation."""
        logger.debug("PreviewPanel: update_config called")

        # Store complete old config for comparison
        old_config = self.config.copy()

        # Update to new config
        self.config = new_config

        logger.debug(f"PreviewPanel: Old normalisation: {old_config.normalisation}")
        logger.debug(f"PreviewPanel: New normalisation: {new_config.normalisation}")

        # Check if we need to reload sources (catalogue or extensions changed)
        if (
            old_config.source_catalogue != new_config.source_catalogue
            or old_config.selected_extensions != new_config.selected_extensions
        ):
            logger.debug("PreviewPanel: Catalogue or extensions changed, reloading sources")
            self.reload_preview_sources()
        else:
            # Check if other processing parameters changed that require regeneration
            if (
                old_config.target_resolution != new_config.target_resolution
                or old_config.normalisation_method != new_config.normalisation_method
                or old_config.channel_weights != new_config.channel_weights
                or old_config.normalisation != new_config.normalisation
            ):
                logger.debug("PreviewPanel: Processing parameters changed, regenerating preview")
                self.regenerate_preview()
            else:
                logger.debug("PreviewPanel: No significant config changes detected")

    def reload_preview_sources(self):
        """Reload preview sources and regenerate preview."""

        async def reload_and_generate():
            await self.load_preview_sources()
            self.regenerate_preview()

        # Run async task
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(reload_and_generate())
            else:
                loop.run_until_complete(reload_and_generate())
        except RuntimeError:
            # No event loop available, just regenerate with existing cache
            self.regenerate_preview()

    def regenerate_preview(self):
        """Regenerate preview with current configuration (using cached sources)."""
        logger.debug("PreviewPanel: regenerate_preview called")
        # Clear any previous error messages when starting new generation
        self.info_text.layout.display = "none"
        self.info_text.value = ""

        # Only regenerate if we have the necessary config
        if self.config.source_catalogue and self.config.selected_extensions:
            logger.debug("PreviewPanel: Config valid, calling _generate_preview")
            self._generate_preview()
        else:
            logger.debug(
                f"PreviewPanel: Skipping regeneration - catalogue: {self.config.source_catalogue}, extensions: {self.config.selected_extensions}"
            )
