#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Normalisation configuration widget for Cutana UI.

This module provides a dedicated normalisation configuration widget that handles
all normalisation-related parameters including method selection and method-specific
parameters (a, n_samples, contrast).
"""

import ipywidgets as widgets
from dotmap import DotMap
from loguru import logger

from cutana.normalisation_parameters import (
    NormalisationRanges,
    NormalisationSteps,
    get_method_specific_a_default,
    get_method_specific_a_range,
    get_method_specific_a_step,
    get_method_tooltip,
)

from ..styles import TEXT_COLOR_LIGHT


class NormalisationConfigWidget(widgets.VBox):
    """Dedicated normalisation configuration widget with title and method-specific parameters."""

    def __init__(self, config, compact=False):
        self.config = config
        self.compact = compact
        self._config_change_callback = None

        # Normalisation section title
        self.normalisation_title = widgets.HTML(
            value=f'<h3 style="color: {TEXT_COLOR_LIGHT}; font-weight: 600; font-size: 14px; \
margin: 10px 0 5px 0; border-bottom: 1px solid #335E6E; padding-bottom: 3px;">Normalisation Parameters</h3>'
        )

        # Handle normalisation value from config
        config_normalisation = self.config.normalisation_method
        logger.debug(f"Config normalisation method: {config_normalisation}")
        normalisation_value = "linear" if config_normalisation == "none" else config_normalisation

        # Reference to format dropdown (to be set externally)
        self.format_dropdown_ref = None

        # Percentile input (appears for all stretch methods) - aligned with main parameters
        self.percentile_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 12px; display: flex; align-items: center; height: 100%;">Percentile:</div>',
            layout=widgets.Layout(height="32px", width="100%"),
        )
        self.percentile_input = widgets.BoundedFloatText(
            value=self.config.normalisation.percentile,
            min=NormalisationRanges.PERCENTILE_MIN + 0.1,  # UI minimum slightly above 0
            max=NormalisationRanges.PERCENTILE_MAX,
            step=NormalisationSteps.PERCENTILE_STEP,
            layout=widgets.Layout(width="120px", height="32px"),
            tooltip=f"Percentile clipping for normalization ({NormalisationRanges.PERCENTILE_MIN + 0.1}-{NormalisationRanges.PERCENTILE_MAX})",
        )

        # Flux-conserved resizing checkbox - placed before normalisation dropdown
        self.flux_conserved_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 12px; display: flex; align-items: center; height: 100%; white-space: nowrap;">Flux-conserving resize:</div>',
            layout=widgets.Layout(height="32px", width="100%"),
        )
        self.flux_conserved_checkbox = widgets.Checkbox(
            value=getattr(self.config, "flux_conserved_resizing", False),
            layout=widgets.Layout(width="120px", height="32px"),
            tooltip="Use flux-conserved resizing (drizzle). Forces float32 output and normalisation=none",
        )

        # Normalisation method dropdown - aligned with main parameters
        self.normalisation_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 12px; display: flex; align-items: center; height: 100%;">Normalisation:</div>',
            layout=widgets.Layout(height="32px", width="100%"),
        )
        self.normalisation_dropdown = widgets.Dropdown(
            options=["none", "linear", "log", "asinh", "zscale", "midtones"],
            value=normalisation_value,
            layout=widgets.Layout(width="120px", height="32px"),
            tooltip="Normalisation method (none = no normalisation applied)",
        )

        # Unified 'a' parameter input (conditional - for ASINH and LOG) - aligned with main parameters
        self.a_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 12px; display: flex; align-items: center; height: 100%;">a:</div>',
            layout=widgets.Layout(height="32px", width="100%"),
        )
        default_a = self.config.normalisation.a
        self.a_input = widgets.BoundedFloatText(
            value=default_a,
            min=NormalisationRanges.ASINH_A_MIN,  # Will be updated dynamically
            max=NormalisationRanges.LOG_A_MAX,  # Will be updated dynamically
            step=NormalisationSteps.ASINH_A_STEP,  # Will be updated dynamically
            layout=widgets.Layout(width="120px", height="32px"),
            tooltip=get_method_tooltip("asinh"),  # Will be updated dynamically
        )

        # ZScale parameters (conditional - only for ZSCALE) - aligned with main parameters
        self.n_samples_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 12px; display: flex; align-items: center; height: 100%;">N Samples:</div>',
            layout=widgets.Layout(height="32px", width="100%"),
        )
        self.n_samples_input = widgets.BoundedIntText(
            value=self.config.normalisation.n_samples,
            min=NormalisationRanges.N_SAMPLES_MIN,
            max=NormalisationRanges.N_SAMPLES_MAX,
            step=NormalisationSteps.N_SAMPLES_STEP,
            layout=widgets.Layout(width="120px", height="32px"),
            tooltip=f"Number of samples for ZScale computation \
({NormalisationRanges.N_SAMPLES_MIN}-{NormalisationRanges.N_SAMPLES_MAX})",
        )

        self.contrast_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 12px; display: flex; align-items: center; height: 100%;">Contrast:</div>',
            layout=widgets.Layout(height="32px", width="100%"),
        )
        self.contrast_input = widgets.BoundedFloatText(
            value=self.config.normalisation.contrast,
            min=NormalisationRanges.CONTRAST_MIN,
            max=NormalisationRanges.CONTRAST_MAX,
            step=NormalisationSteps.CONTRAST_STEP,
            layout=widgets.Layout(width="120px", height="32px"),
            tooltip=f"ZScale contrast parameter ({NormalisationRanges.CONTRAST_MIN}-{NormalisationRanges.CONTRAST_MAX})",
        )

        # Crop parameters (for norm_crop_for_maximum_value) - aligned with main parameters
        self.crop_enable_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 12px; display: flex; align-items: center; height: 100%;">Use center crop for max:</div>',
            layout=widgets.Layout(height="32px", width="100%"),
        )
        self.crop_enable_checkbox = widgets.Checkbox(
            value=self.config.normalisation.crop_enable,
            layout=widgets.Layout(width="120px", height="32px"),
            tooltip="Enable cropping to center region when computing maximum value for normalisation (prevents bright sources neraby to dim out target)",
        )

        self.crop_size_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 12px; display: flex; align-items: center; height: 100%;">Crop Size:</div>',
            layout=widgets.Layout(height="32px", width="100%"),
        )
        # Use the smaller of height/width as the default, and use the more restrictive range
        default_size = min(
            self.config.normalisation.crop_height, self.config.normalisation.crop_width
        )
        min_size = max(NormalisationRanges.CROP_HEIGHT_MIN, NormalisationRanges.CROP_WIDTH_MIN)
        max_size = min(NormalisationRanges.CROP_HEIGHT_MAX, NormalisationRanges.CROP_WIDTH_MAX)
        step_size = max(NormalisationSteps.CROP_HEIGHT_STEP, NormalisationSteps.CROP_WIDTH_STEP)

        self.crop_size_input = widgets.BoundedIntText(
            value=default_size,
            min=min_size,
            max=max_size,
            step=step_size,
            layout=widgets.Layout(width="120px", height="32px"),
            tooltip=f"Size of square crop region for maximum value computation ({min_size}-{max_size})",
        )

        # Interpolation parameters - aligned with main parameters
        self.interpolation_label = widgets.HTML(
            value=f'<div style="color: {TEXT_COLOR_LIGHT}; font-weight: 500; font-size: 12px; display: flex; align-items: center; height: 100%;">Interpolation:</div>',
            layout=widgets.Layout(height="32px", width="100%"),
        )
        self.interpolation_dropdown = widgets.Dropdown(
            options=["nearest", "bilinear", "biquadratic", "bicubic"],
            value=getattr(self.config, "interpolation", "bilinear"),
            layout=widgets.Layout(width="120px", height="32px"),
            tooltip="Interpolation method for image resizing (nearest, bilinear, biquadratic, bicubic)",
        )

        # Set initial visibility
        self._update_parameter_visibility()

        # Create layout grid - aligned with main parameters
        label_width = "100px" if compact else "110px"  # Match main parameter grid
        input_width = "120px"  # Match main parameter grid
        self.normalisation_grid = widgets.GridBox(
            children=[
                self.flux_conserved_label,
                self.flux_conserved_checkbox,
                self.normalisation_label,
                self.normalisation_dropdown,
                self.percentile_label,
                self.percentile_input,
                # Conditional parameters
                self.a_label,
                self.a_input,
                self.n_samples_label,
                self.n_samples_input,
                self.contrast_label,
                self.contrast_input,
                # Crop parameters (TODO reimplement in future)
                # self.crop_enable_label,
                # self.crop_enable_checkbox,
                # self.crop_size_label,
                # self.crop_size_input,
                # Interpolation parameter
                self.interpolation_label,
                self.interpolation_dropdown,
            ],
            layout=widgets.Layout(
                grid_template_columns=f"{label_width} {input_width}",  # Fixed widths matching main grid
                grid_gap="6px 10px",  # Match main parameter grid gaps - reduced vertical spacing
                margin="8px 0",
                width="100%",
                align_items="center",  # Center align all grid items vertically
                justify_items="flex-start",  # Align items to start horizontally
                overflow="visible",  # Remove scrollbar
            ),
        )

        # Initialize children
        children = [self.normalisation_title, self.normalisation_grid]
        super().__init__(children=children)

        # Set up event handlers
        self._setup_events()

    def set_format_dropdown_ref(self, format_dropdown):
        """
        Set reference to the format dropdown widget.

        This allows the normalisation widget to control the format dropdown
        when flux_conserved resizing is enabled (forcing float32).

        Args:
            format_dropdown: The format dropdown widget from configuration_widget
        """
        self.format_dropdown_ref = format_dropdown
        logger.debug(f"Format dropdown reference set")

    def _setup_events(self):
        """Set up event handlers for normalisation parameters."""

        def on_flux_conserved_change(change):
            logger.debug(f"Flux conserved changed: {change['old']} -> {change['new']}")
            if change["new"]:
                # Force normalisation to none and disable normalisation dropdown
                self.normalisation_dropdown.value = "none"
                self.normalisation_dropdown.disabled = True
                # Force float32 in format dropdown if reference exists
                if self.format_dropdown_ref is not None:
                    self.format_dropdown_ref.value = "float32"
                    self.format_dropdown_ref.disabled = True
            else:
                # Re-enable normalisation dropdown
                self.normalisation_dropdown.disabled = False
                # Re-enable format dropdown if reference exists
                if self.format_dropdown_ref is not None:
                    self.format_dropdown_ref.disabled = False
            self._update_parameter_visibility()
            if self._config_change_callback:
                self._config_change_callback()

        def on_normalisation_change(change):
            logger.debug(f"Normalisation method changed: {change['old']} -> {change['new']}")
            self._update_parameter_visibility()
            if self._config_change_callback:
                self._config_change_callback()

        def on_config_change(change):
            logger.debug(
                f"Normalisation parameter changed: {change['owner'].description} -> {change['new']}"
            )
            if self._config_change_callback:
                self._config_change_callback()

        def on_crop_enable_change(change):
            logger.debug(f"Crop enable changed: {change['old']} -> {change['new']}")
            self._update_parameter_visibility()
            if self._config_change_callback:
                self._config_change_callback()

        # Connect configuration change callbacks
        self.flux_conserved_checkbox.observe(on_flux_conserved_change, names="value")
        self.normalisation_dropdown.observe(on_normalisation_change, names="value")
        self.percentile_input.observe(on_config_change, names="value")
        self.a_input.observe(on_config_change, names="value")
        self.n_samples_input.observe(on_config_change, names="value")
        self.contrast_input.observe(on_config_change, names="value")
        self.crop_enable_checkbox.observe(on_crop_enable_change, names="value")
        self.crop_size_input.observe(on_config_change, names="value")
        self.interpolation_dropdown.observe(on_config_change, names="value")

    def _update_parameter_visibility(self):
        """Show/hide method-specific parameters based on selected normalisation method."""
        normalisation_method = self.normalisation_dropdown.value
        flux_conserved = self.flux_conserved_checkbox.value

        # If normalisation is "none", hide all normalisation parameters
        is_none = normalisation_method == "none"

        # Percentile is shown for all methods except "none"
        self.percentile_label.layout.display = "none" if is_none else ""
        self.percentile_input.layout.display = "none" if is_none else ""

        # Show or hide unified 'a' parameter for ASINH and LOG (but not for "none")
        needs_a_param = normalisation_method in ["asinh", "log", "midtones"]
        self.a_label.layout.display = "" if needs_a_param else "none"
        self.a_input.layout.display = "" if needs_a_param else "none"

        # Show or hide ZScale parameter controls (but not for "none")
        is_zscale = normalisation_method == "zscale"
        self.n_samples_label.layout.display = "" if is_zscale else "none"
        self.n_samples_input.layout.display = "" if is_zscale else "none"
        self.contrast_label.layout.display = "" if is_zscale else "none"
        self.contrast_input.layout.display = "" if is_zscale else "none"

        # Crop parameters are visible for all normalization methods except "none"
        # The crop_size input is only visible when crop is enabled
        crop_enabled = self.crop_enable_checkbox.value and not is_none
        self.crop_size_label.layout.display = "" if crop_enabled else "none"
        self.crop_size_input.layout.display = "" if crop_enabled else "none"

        # Update the 'a' parameter value and range based on method using centralized parameters
        if needs_a_param:
            # Get method-specific ranges and defaults
            min_val, max_val = get_method_specific_a_range(normalisation_method)
            step_val = get_method_specific_a_step(normalisation_method)
            tooltip = get_method_tooltip(normalisation_method)

            # Use custom default values based on normalisation method
            if normalisation_method == "log":
                default_val = 1000.0
            elif normalisation_method == "asinh":
                default_val = 0.1
            elif normalisation_method == "midtones":
                default_val = 0.2
            else:
                default_val = get_method_specific_a_default(normalisation_method)

            # Update widget parameters
            self.a_input.min = min_val
            self.a_input.max = max_val
            self.a_input.step = step_val
            self.a_input.tooltip = tooltip

            # Always set the default value when switching normalization methods
            # This ensures users see the appropriate default for each method
            self.a_input.value = default_val

        # Update normalisation dropdown state based on flux_conserved
        self.normalisation_dropdown.disabled = flux_conserved

        # Disable interpolation dropdown when flux_conserved is enabled
        # (flux conserved uses drizzle which has its own resampling method)
        # Keep visible but disabled to maintain consistent UI layout
        self.interpolation_dropdown.disabled = flux_conserved

        # Update format dropdown state based on flux_conserved if reference exists
        if self.format_dropdown_ref is not None:
            self.format_dropdown_ref.disabled = flux_conserved

    def set_config_change_callback(self, callback):
        """Set callback for configuration changes."""
        self._config_change_callback = callback

    def update_config(self, config):
        """Update normalisation parameters from config."""
        self.config = config

        # Restore flux-conserved checkbox
        self.flux_conserved_checkbox.value = config.flux_conserved_resizing

        # Restore normalization parameter values
        self.percentile_input.value = config.normalisation.percentile
        self.a_input.value = config.normalisation.a
        self.n_samples_input.value = config.normalisation.n_samples
        self.contrast_input.value = config.normalisation.contrast

        # Restore crop parameter values
        self.crop_enable_checkbox.value = config.normalisation.crop_enable
        # Use the smaller of height/width as the crop size value
        crop_size = min(config.normalisation.crop_height, config.normalisation.crop_width)
        self.crop_size_input.value = crop_size

        # Restore interpolation value
        self.interpolation_dropdown.value = getattr(config, "interpolation", "bilinear")

        # Restore normalisation method
        self.normalisation_dropdown.value = config.normalisation_method

        # Update parameter visibility
        self._update_parameter_visibility()

    def get_normalisation_config(self):
        """Get current normalisation configuration."""
        # Use the same size for both height and width (square crop)
        crop_size = self.crop_size_input.value
        flux_conserved = self.flux_conserved_checkbox.value
        config = DotMap(_dynamic=False)
        config.flux_conserved_resizing = flux_conserved
        config.normalisation_method = self.normalisation_dropdown.value
        config.interpolation = self.interpolation_dropdown.value
        config.normalisation = DotMap(_dynamic=False)
        config.normalisation.percentile = self.percentile_input.value
        config.normalisation.a = self.a_input.value
        config.normalisation.n_samples = self.n_samples_input.value
        config.normalisation.contrast = self.contrast_input.value
        config.normalisation.crop_enable = self.crop_enable_checkbox.value
        config.normalisation.crop_height = crop_size
        config.normalisation.crop_width = crop_size

        return config
