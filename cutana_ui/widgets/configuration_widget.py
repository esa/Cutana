#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Shared configuration widget used by both start screen and main screen."""

import ipywidgets as widgets
from dotmap import DotMap
from loguru import logger

from ..styles import ESA_BLUE_ACCENT, ESA_BLUE_GREY, TEXT_COLOR_LIGHT
from .normalisation_config_widget import NormalisationConfigWidget


class SharedConfigurationWidget(widgets.VBox):
    """Shared configuration widget with all processing parameters."""

    def __init__(
        self,
        config,
        compact=False,
        show_extensions=True,
        show_matrix=True,
        show_advanced_params=True,
        show_filesize=True,
    ):
        self.config = config
        self.compact = compact  # True for main screen, False for start screen
        self.show_extensions = show_extensions  # False for main screen, True for start screen
        self.show_matrix = show_matrix  # False for start screen, True for main screen
        self.show_advanced_params = (
            show_advanced_params  # False for start screen to hide advanced params
        )
        self.show_filesize = show_filesize  # False to hide filesize display
        self.extensions = self.config.available_extensions
        self.channel_matrices = []
        # Initialize channels based on mode and selected extensions
        if show_matrix and config.selected_extensions:
            # For main screen: number of channels equals number of selected extensions
            self.current_channels = len(config.selected_extensions)
        else:
            # Default to 1 channel for start screen or when no extensions selected
            self.current_channels = 1
        self.num_sources = self.config.num_sources
        self._config_change_callback = None

        # Extension checkboxes container
        self.extensions_label = widgets.HTML(
            value=f'<label style="color: {TEXT_COLOR_LIGHT}; font-weight: 500;">\
{"FITS Extensions:" if not compact else "Extensions:"}</label>'
        )
        self.extensions_container = widgets.VBox(
            layout=widgets.Layout(
                padding="4px",  # Reduced padding
                background=ESA_BLUE_GREY,
                border_radius="5px",
                margin="0",  # Remove all margins to eliminate space between checklist and Max. Size
                width="100%",  # Make extensions container span full width
            )
        )

        # Matrix controls - more compact
        self.add_channel_btn = widgets.Button(
            description="Add Channel",
            layout=widgets.Layout(width="110px", height="26px"),  # Smaller dimensions
            tooltip="Add channel",
        )
        self.remove_channel_btn = widgets.Button(
            description="Remove Channel",
            layout=widgets.Layout(width="110px", height="26px"),  # Smaller dimensions
            tooltip="Remove channel",
        )
        self.matrix_controls = widgets.HBox(
            children=[self.remove_channel_btn, self.add_channel_btn],
            layout=widgets.Layout(
                justify_content="space-between", margin="0", width="100%"
            ),  # No margin
        )

        self.add_channel_btn.style.button_color = "#52c41a"
        self.remove_channel_btn.style.button_color = "red"

        self.matrix_container = widgets.VBox(
            layout=widgets.Layout(
                padding="5px",  # Reduced padding
                background=ESA_BLUE_GREY,
                border_radius="5px",
                margin="0",  # No margin
                width="100%",  # Make matrix container span full width
                max_height="150px" if compact else "180px",  # Reduced max height
                overflow="auto",  # Enable scrolling when content exceeds container
                align_content="flex-start",
            )
        )

        # Processing parameters - compact appearance
        self.format_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 11px; display: flex; align-items: center; height: 100%;">Format:</div>',
            layout=widgets.Layout(height="28px", width="100%"),
        )
        self.format_dropdown = widgets.Dropdown(
            options=["float32", "uint8"],
            value="float32",
            layout=widgets.Layout(width="140px", height="28px"),
        )
        self.format_dropdown.add_class("config-grid-item")

        self.output_format_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 11px; display: flex; align-items: center; height: 100%;">Output:</div>',
            layout=widgets.Layout(height="28px", width="100%"),
        )
        self.output_format_dropdown = widgets.Dropdown(
            options=["zarr", "fits"],
            value="zarr",
            layout=widgets.Layout(width="140px", height="28px"),
        )
        self.output_format_dropdown.add_class("config-grid-item")

        self.resolution_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 11px; display: flex; align-items: center; height: 100%;">Resolution:</div>',
            layout=widgets.Layout(height="28px", width="100%"),  # Compact alignment
        )
        self.resolution_input = widgets.BoundedIntText(
            value=self.config.target_resolution,
            min=16,
            max=2048,  # Set appropriate max value to prevent clamping
            layout=widgets.Layout(
                width="140px", height="28px"
            ),  # Further increased width to prevent scroll
        )
        self.resolution_input.add_class("config-grid-item")

        self.padding_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 11px; display: flex; align-items: center; height: 100%;">Zoom-out:</div>',
            layout=widgets.Layout(height="28px", width="100%"),
        )
        self.padding_slider = widgets.FloatSlider(
            value=self.config.padding_factor if hasattr(self.config, "padding_factor") else 1.0,
            min=0.25,
            max=10.0,
            step=0.25,
            layout=widgets.Layout(width="140px", height="28px"),
            readout_format=".2f",
            tooltip="Set a factor to change the cutout size.",
            style={"handle_color": ESA_BLUE_ACCENT, "description_width": "initial"},
        )
        self.padding_slider.add_class("config-grid-item")
        # Apply custom styling to the slider readout
        self.padding_slider.add_class("cutana-slider-compact")

        # Raw cutout only checkbox - disables all processing when checked
        self.do_only_cutout_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 11px; display: flex; align-items: center; height: 100%;">Raw cutout:</div>',
            layout=widgets.Layout(height="28px", width="100%"),
        )
        self.do_only_cutout_checkbox = widgets.Checkbox(
            value=getattr(self.config, "do_only_cutout_extraction", False),
            layout=widgets.Layout(width="140px", height="28px"),
            tooltip="Extract raw cutouts without processing. Forces FITS output, float32, disables resizing and normalisation.",
        )
        self.do_only_cutout_checkbox.add_class("config-grid-item")

        # Create the normalisation widget only if advanced params are shown
        if self.show_advanced_params:
            self.normalisation_widget = NormalisationConfigWidget(config, compact)
        else:
            self.normalisation_widget = None

        # Layout in aligned grid with proper spacing to prevent scroll bars
        grid_min_width = "320px" if compact else "400px"  # Further increased width
        label_width = "80px" if compact else "90px"  # Increased label width
        input_width = "160px"  # Further increased input width

        self.config_grid = widgets.GridBox(
            children=[
                self.format_label,
                self.format_dropdown,
                self.output_format_label,
                self.output_format_dropdown,
                self.resolution_label,
                self.resolution_input,
                self.padding_label,
                self.padding_slider,
                self.do_only_cutout_label,
                self.do_only_cutout_checkbox,
            ],
            layout=widgets.Layout(
                grid_template_columns=f"{label_width} {input_width}",  # Fixed widths for perfect alignment
                grid_gap="5px 10px",  # Better gaps for proper spacing
                margin="3px 0",  # Small margin for breathing room
                min_width=grid_min_width,
                width="100%",
                align_items="center",  # Center align all grid items vertically
                justify_items="flex-start",  # Align items to start horizontally
            ),
        )

        # Predicted filesize as small colored label
        self.filesize_display = widgets.HTML(
            value='<span style="color: #52c41a; font-weight: bold; font-size: 10px;">Max: 0.0 GB</span>'
        )

        # Build children list based on mode
        children = []

        # Only show extensions selector on start screen
        if self.show_extensions:
            children.extend(
                [
                    self.extensions_label,
                    self.extensions_container,
                ]
            )

        # Only show channel matrix on main screen
        if self.show_matrix:
            children.extend(
                [
                    # No spacing before matrix controls
                    self.matrix_controls,
                    self.matrix_container,
                    # No spacing after matrix
                ]
            )

        # Only show advanced parameters (config grid and normalisation) if enabled
        if self.show_advanced_params:
            children.extend(
                [
                    self.config_grid,
                    # No spacing before normalisation widget
                    self.normalisation_widget,
                ]
            )

            # Only add filesize section if enabled
            if self.show_filesize:
                children.append(self.filesize_display)

            # No bottom spacer needed
        else:
            # For start screen without advanced params, conditionally show filesize
            if self.show_filesize:
                children.append(self.filesize_display)
            # No bottom spacer needed

        super().__init__(
            children=children,
            layout=widgets.Layout(
                width="100%", overflow="visible"  # Ensure content is not clipped
            ),
        )

        # Initialize with config data
        if self.extensions:
            if self.show_extensions:
                self.set_extensions(self.extensions)
            elif self.show_matrix:
                # For main screen, just update the matrix without checkboxes
                self._update_matrix()

        # Set up event handlers
        self._setup_events()

        # Set callback for normalisation widget if it exists
        if self.show_advanced_params and self.normalisation_widget:
            self.normalisation_widget.set_config_change_callback(self._on_config_change)

    @property
    def normalisation_dropdown(self):
        """Forward access to normalisation dropdown for backward compatibility with tests."""
        if self.show_advanced_params:
            return self.normalisation_widget.normalisation_dropdown
        else:
            return None

    def _on_config_change(self):
        """Handle configuration changes from child widgets."""
        logger.debug(
            "SharedConfigurationWidget: Configuration change detected, triggering callback"
        )
        if self._config_change_callback:
            logger.debug("SharedConfigurationWidget: Calling config change callback")
            self._config_change_callback()
        else:
            logger.debug("SharedConfigurationWidget: No config change callback set")

    def _setup_events(self):
        """Set up event handlers."""

        def add_channel_handler(_b):
            self._add_channel()

        def remove_channel_handler(_b):
            self._remove_channel()

        self.add_channel_btn.on_click(add_channel_handler)
        self.remove_channel_btn.on_click(remove_channel_handler)

        def validate_resolution(change):
            if change["new"] < 16:
                self.resolution_input.value = 16
            self._update_filesize_prediction()

        def on_format_change(change):
            self._update_filesize_prediction()

        def on_config_change(change):
            if self._config_change_callback:
                self._config_change_callback()

        self.resolution_input.observe(validate_resolution, names="value")
        self.format_dropdown.observe(on_format_change, names="value")
        self.resolution_input.observe(on_config_change, names="value")
        self.format_dropdown.observe(on_config_change, names="value")
        self.padding_slider.observe(on_config_change, names="value")

        if self.compact and self.output_format_dropdown:
            self.output_format_dropdown.observe(on_config_change, names="value")

        # Connect format dropdown to normalisation widget for flux_conserved override
        if self.normalisation_widget:
            self.normalisation_widget.set_format_dropdown_ref(self.format_dropdown)

        # Set up do_only_cutout checkbox handler
        def on_do_only_cutout_change(change):
            logger.debug(f"Do only cutout changed: {change['old']} -> {change['new']}")
            if change["new"]:
                # Force FITS output format and disable dropdown
                if self.output_format_dropdown:
                    self.output_format_dropdown.value = "fits"
                    self.output_format_dropdown.disabled = True
                # Force float32 format and disable dropdown
                self.format_dropdown.value = "float32"
                self.format_dropdown.disabled = True
                # Disable resolution input (greyed out)
                self.resolution_input.disabled = True
                # Hide normalisation widget
                if self.normalisation_widget:
                    self.normalisation_widget.layout.display = "none"
            else:
                # Re-enable output format dropdown
                if self.output_format_dropdown:
                    self.output_format_dropdown.disabled = False
                # Re-enable format dropdown (unless flux_conserved is on in normalisation widget)
                flux_conserved = False
                if self.normalisation_widget:
                    flux_conserved = self.normalisation_widget.flux_conserved_checkbox.value
                self.format_dropdown.disabled = flux_conserved
                # Re-enable resolution input
                self.resolution_input.disabled = False
                # Show normalisation widget
                if self.normalisation_widget:
                    self.normalisation_widget.layout.display = ""
            if self._config_change_callback:
                self._config_change_callback()

        self.do_only_cutout_checkbox.observe(on_do_only_cutout_change, names="value")

        # Apply initial state if do_only_cutout is already checked
        if self.do_only_cutout_checkbox.value:
            on_do_only_cutout_change({"old": False, "new": True})

    def set_extensions(self, extensions):
        """Set available extensions and create checkboxes."""
        self.extensions = extensions
        self.extension_checkboxes = []

        # Determine which extensions should be checked based on config's selected_extensions
        selected_ext_names = set()
        if self.config.selected_extensions:
            for ext in self.config.selected_extensions:
                if isinstance(ext, dict):
                    selected_ext_names.add(ext.get("name", ""))
                else:
                    selected_ext_names.add(str(ext))

        for ext in extensions:
            ext_name = ext.get("name", "") if isinstance(ext, dict) else str(ext)
            # If no selected_extensions in config, default to all checked (backward compatibility)
            # Otherwise, only check if this extension is in selected_extensions
            should_check = not self.config.selected_extensions or ext_name in selected_ext_names

            checkbox = widgets.Checkbox(
                value=should_check,
                description=(f"{ext['name']}" if self.compact else f"{ext['name']}/{ext['ext']}"),
                style={"description_width": "initial", "text_color": TEXT_COLOR_LIGHT},
                layout=widgets.Layout(width="auto" if self.compact else "200px"),
            )
            checkbox.style.color = TEXT_COLOR_LIGHT
            checkbox.add_class("widget-checkbox")

            def on_checkbox_change(change):
                # Update matrix when extensions are selected/deselected
                self._update_matrix()
                if self._config_change_callback:
                    self._config_change_callback()

            checkbox.observe(on_checkbox_change, names="value")
            self.extension_checkboxes.append(checkbox)

        self.extensions_container.children = self.extension_checkboxes
        self._update_matrix()

    def set_config_change_callback(self, callback):
        """Set callback for configuration changes."""
        self._config_change_callback = callback

    def _get_selected_extensions(self):
        """Get currently selected extensions."""
        selected_extensions = []
        if hasattr(self, "extension_checkboxes") and self.extension_checkboxes:
            for i, checkbox in enumerate(self.extension_checkboxes):
                if checkbox.value and i < len(self.extensions):
                    selected_extensions.append(self.extensions[i])
        return selected_extensions

    def _update_matrix(self):
        """Update the channel combination matrix display."""
        if not self.extensions:
            return

        # Show selected extensions on both screens
        if self.show_extensions:
            # Start screen: use checkbox selections
            selected_extensions = self._get_selected_extensions()
            if not selected_extensions:
                # No extensions selected, hide matrix
                self.matrix_container.children = []
                return
            extensions_to_show = selected_extensions
        else:
            # Main screen: use config.selected_extensions (from start screen)
            extensions_to_show = (
                self.config.selected_extensions
                if self.config.selected_extensions
                else self.extensions
            )

        # Allow channels to exceed number of selected extensions

        num_extensions = len(extensions_to_show)

        header_widgets = [
            widgets.HTML(
                value=f'<div style="width: 35px; color: {TEXT_COLOR_LIGHT}; \
font-size: 8px; font-weight: bold; text-align: center;">Ch</div>',
                layout=widgets.Layout(width="35px", flex="0 0 auto"),
            )
        ]
        for ext in extensions_to_show:
            header_widgets.append(
                widgets.HTML(
                    value=f'<div style="text-align: center; color: {TEXT_COLOR_LIGHT}; \
font-size: 8px; font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{ext["name"]}</div>',
                    layout=widgets.Layout(flex="1 1 auto", min_width="44px", max_width="88px"),
                )
            )
        header = widgets.HBox(
            header_widgets,
            layout=widgets.Layout(width="100%", justify_content="flex-start", margin="0 0 4px 0"),
        )

        # Create matrix rows
        rows = [header]
        self.channel_matrices = []

        for i in range(self.current_channels):
            # Channel label with consistent width
            row_widgets = [
                widgets.HTML(
                    value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-size: 8px; text-align: center;">{i+1}</div>',
                    layout=widgets.Layout(width="25px", flex="0 0 auto", overflow="hidden"),
                )
            ]

            row_values = []

            for j in range(num_extensions):
                # Default to 1.0 only if channel index matches extension index and within bounds
                # Extra channels beyond extensions get 0.0 for all extensions
                default_value = 1.0 if (i == j and i < num_extensions) else 0.0

                input_widget = widgets.BoundedFloatText(
                    value=default_value,
                    min=0.0,
                    max=10.0,  # Set appropriate max value for channel weights
                    step=0.1,
                    layout=widgets.Layout(
                        flex="1 1 auto",
                        min_width="44px",
                        max_width="88px",
                        height="26px",  # Increased by ~10% from 24px
                        margin="0 1px",  # Tighter margin,
                        overflow="hidden",
                    ),
                )

                # Reduce font size for compact appearance
                input_widget.add_class("cutana-matrix-input")

                # Add observer for config changes when matrix values change
                def on_matrix_change(change):
                    if self._config_change_callback:
                        self._config_change_callback()

                input_widget.observe(on_matrix_change, names="value")
                row_widgets.append(input_widget)
                row_values.append(input_widget)

            self.channel_matrices.append(row_values)
            rows.append(
                widgets.HBox(
                    row_widgets,
                    layout=widgets.Layout(
                        margin="1px 0",  # Tighter vertical margin
                        width="100%",
                        justify_content="flex-start",
                        height="26px",  # Fixed row height for consistency
                    ),
                )
            )

        self.matrix_container.children = rows
        self._update_filesize_prediction()

    def _add_channel(self):
        """Add a new channel row to the matrix."""
        # Allow unlimited channels (no maximum restriction)
        self.current_channels = self.current_channels + 1
        self._update_matrix()
        # Trigger config change callback for preview updates
        if self._config_change_callback:
            self._config_change_callback()

    def _remove_channel(self):
        """Remove the last channel row from the matrix."""
        self.current_channels = max(1, self.current_channels - 1)
        self._update_matrix()
        # Trigger config change callback for preview updates
        if self._config_change_callback:
            self._config_change_callback()

    def _update_filesize_prediction(self):
        """Update the predicted filesize display."""
        try:
            if self.show_advanced_params:
                # Use current UI values when advanced params are shown
                resolution = self.resolution_input.value
                data_type = self.format_dropdown.value
            else:
                # Use config values when advanced params are hidden
                resolution = self.config.target_resolution
                data_type = self.config.data_type

            n_sources = self.num_sources
            n_channels = self.current_channels

            bytes_per_pixel = 4 if data_type == "float32" else 1
            total_bytes = n_sources * n_channels * resolution * resolution * bytes_per_pixel
            total_gb = total_bytes / (1024**3)

            color = "#52c41a" if total_gb < 10 else "#faad14" if total_gb < 50 else "#ff4d4f"

            self.filesize_display.value = f'<span style="color: {color}; font-weight: bold; font-size: 10px;">Max: {total_gb:.1f} GB</span>'

        except Exception as e:
            logger.debug(f"Error calculating filesize: {e}")
            self.filesize_display.value = '<span style="color: #52c41a; font-weight: bold; font-size: 10px;">Max: -- GB</span>'

    def _update_channel_weights(self, config):
        """Update channel matrix weights from configuration."""
        if not (config.channel_weights and self.channel_matrices):
            return

        channel_weights = config.channel_weights
        selected_extensions = config.selected_extensions

        if isinstance(channel_weights, dict):
            # Dictionary format: {"VIS": [1.0, 0.0, 0.5], "NIR": [0.0, 1.0, 0.5]}
            extension_names = [
                ext_info.get("name", f"EXT_{i}") if isinstance(ext_info, dict) else str(ext_info)
                for i, ext_info in enumerate(selected_extensions)
            ]

            for i, (row, ext_name) in enumerate(zip(self.channel_matrices, extension_names)):
                if ext_name in channel_weights:
                    weights = channel_weights[ext_name]
                    for widget, weight in zip(row, weights):
                        widget.value = weight

        elif isinstance(channel_weights, (list, tuple)):
            # Legacy list format: [[1.0, 0.0], [0.0, 1.0]] - backward compatibility
            for row, row_weights in zip(self.channel_matrices, channel_weights):
                if isinstance(row_weights, (list, tuple)):
                    for widget, weight in zip(row, row_weights):
                        widget.value = weight

    def update_config(self, config):
        """Update configuration from external source."""
        self.config = config
        self.num_sources = config.num_sources

        # Restore channel count BEFORE setting extensions
        self.current_channels = getattr(config, "num_channels", 1)

        if config.available_extensions:
            self.set_extensions(config.available_extensions)
            self._update_channel_weights(config)

            # Restore extension selections
            if config.selected_extensions and hasattr(self, "extension_checkboxes"):
                selected_extension_names = {
                    ext.get("name", "") for ext in config.selected_extensions
                }
                for checkbox, ext in zip(self.extension_checkboxes, self.extensions):
                    ext_name = ext.get("name", "")
                    checkbox.value = ext_name in selected_extension_names

        # Update normalisation widget if advanced params are shown
        if self.show_advanced_params:
            self.normalisation_widget.update_config(config)

        # Restore other UI parameter values only if advanced params are shown
        if self.show_advanced_params:
            self.resolution_input.value = config.target_resolution
            self.format_dropdown.value = config.data_type
            self.padding_slider.value = (
                config.padding_factor if hasattr(config, "padding_factor") else 1.0
            )
            if self.compact:
                self.output_format_dropdown.value = config.output_format

            # Restore do_only_cutout checkbox
            self.do_only_cutout_checkbox.value = getattr(config, "do_only_cutout_extraction", False)

        self._update_filesize_prediction()

    def set_num_sources(self, num_sources):
        """Set the number of sources for filesize calculation."""
        self.num_sources = num_sources
        self._update_filesize_prediction()

    def get_current_config(self):
        """Get current configuration including any changes."""
        # Get selected extensions
        selected_extensions = []
        if self.show_extensions:
            # Start screen: use checkboxes
            if hasattr(self, "extension_checkboxes") and self.extension_checkboxes:
                for i, checkbox in enumerate(self.extension_checkboxes):
                    if checkbox.value and i < len(self.extensions):
                        selected_extensions.append(self.extensions[i])
            elif self.extensions:
                # If no checkboxes yet, default to all extensions
                selected_extensions = self.extensions.copy()
        else:
            # Main screen: use config.selected_extensions from start screen
            selected_extensions = (
                self.config.selected_extensions if self.config.selected_extensions else []
            )

        # Get channel weights values - convert UI matrix to backend dictionary format
        channel_weights = {}
        if self.show_matrix and hasattr(self, "channel_matrices") and self.channel_matrices:
            # Get the extensions to use for naming (either from checkboxes or from config)
            extensions_for_naming = selected_extensions if selected_extensions else []

            # Convert UI matrix format to backend dictionary format using extension names
            # rows correspond to output channels, columns to input extensions, want an dictionary that is "input_ext_name": [weights for each output channel]
            for i, row in enumerate(self.channel_matrices):
                for j, weight in enumerate(row):
                    if j < len(extensions_for_naming):
                        input_ext_name = extensions_for_naming[j].get("name", f"EXT_{j}")
                        if input_ext_name not in channel_weights:
                            channel_weights[input_ext_name] = []
                        channel_weights[input_ext_name].append(weight.value)

        elif not self.show_matrix and selected_extensions:
            # Start screen: create default identity matrix for selected extensions
            for i, ext_info in enumerate(selected_extensions):
                if isinstance(ext_info, dict):
                    ext_name = ext_info.get("name", f"EXT_{i}")
                else:
                    ext_name = str(ext_info)
                # Create identity row: 1.0 for matching extension, 0.0 for others
                weights = [1.0 if j == i else 0.0 for j in range(len(selected_extensions))]
                channel_weights[ext_name] = weights

        # Start with existing config to preserve backend values
        # Ensure copy is not dynamic to prevent auto-creation of nested DotMaps
        current_config = DotMap(self.config, _dynamic=False)

        # Check for do_only_cutout_extraction mode first (takes precedence)
        do_only_cutout = self.do_only_cutout_checkbox.value

        if do_only_cutout:
            # Raw cutout extraction mode - force FITS output, float32, none normalisation
            current_config.do_only_cutout_extraction = True
            current_config.output_format = "fits"
            current_config.data_type = "float32"
            current_config.normalisation_method = "none"
            current_config.flux_conserved_resizing = False
            # Set default normalisation params for config completeness
            if self.show_advanced_params and self.normalisation_widget:
                normalisation_config = self.normalisation_widget.get_normalisation_config()
                current_config.normalisation = normalisation_config.normalisation
                current_config.interpolation = normalisation_config.interpolation
            current_config.target_resolution = self.resolution_input.value
            current_config.padding_factor = self.padding_slider.value
        elif self.show_advanced_params:
            # Get normalisation configuration from the dedicated widget
            normalisation_config = self.normalisation_widget.get_normalisation_config()
            current_config.do_only_cutout_extraction = False

            if normalisation_config.flux_conserved_resizing:
                # Flux conserved workflow - force float32 and none normalisation
                current_config.data_type = "float32"
                current_config.normalisation_method = "none"
                # Still need to set normalisation params and interpolation for preview workaround
                current_config.normalisation = normalisation_config.normalisation
                current_config.interpolation = normalisation_config.interpolation
            else:
                # Normal workflow
                current_config.data_type = self.format_dropdown.value
                current_config.normalisation_method = normalisation_config.normalisation_method
                current_config.normalisation = normalisation_config.normalisation
                current_config.interpolation = normalisation_config.interpolation
            # get normalisation params
            current_config.flux_conserved_resizing = normalisation_config.flux_conserved_resizing
            current_config.target_resolution = self.resolution_input.value
            current_config.padding_factor = self.padding_slider.value
        else:
            # Use default values when advanced params are hidden
            current_config.do_only_cutout_extraction = False
            normalisation_config = {}

        current_config.selected_extensions = selected_extensions
        current_config.channel_weights = channel_weights
        # Set num_channels based on context
        if self.show_matrix:
            # Main screen: use current matrix channel count
            current_config.num_channels = self.current_channels
        else:
            # Start screen: use selected extensions count
            current_config.num_channels = len(selected_extensions) if selected_extensions else 1

        # Update normalisation configuration from UI
        current_config.update(normalisation_config)

        # Add output format for main screen (compact mode)
        if self.compact and self.output_format_dropdown:
            current_config.output_format = self.output_format_dropdown.value

        return current_config
