#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""CSS styling constants and utilities for Cutana UI."""

# UI Scaling Configuration
# This will be set by the CutanaApp at runtime
# Default values for different screen sizes:
# For 1920x1080 screens: UI_SCALE = 0.75
UI_SCALE = 0.75  # Default value, will be overridden by CutanaApp

# Base dimensions (these will be scaled)
BASE_CONTAINER_WIDTH = 1600
BASE_CONTAINER_HEIGHT = 900
BASE_MAIN_WIDTH = 1400
BASE_PANEL_WIDTH = 380
BASE_PADDING = 20
BASE_VIEWPORT_HEIGHT = 100  # vh units

# Scaled dimensions
CONTAINER_WIDTH = int(BASE_CONTAINER_WIDTH * UI_SCALE)
CONTAINER_HEIGHT = int(BASE_CONTAINER_HEIGHT * UI_SCALE)
MAIN_WIDTH = int(BASE_MAIN_WIDTH * UI_SCALE)
PANEL_WIDTH = int(BASE_PANEL_WIDTH * UI_SCALE)
PADDING = int(BASE_PADDING * UI_SCALE)
VIEWPORT_HEIGHT = int(BASE_VIEWPORT_HEIGHT * UI_SCALE)


def scale_px(pixels):
    """Scale pixel values by the UI scaling factor."""
    # Use the current UI_SCALE value (may be updated by CutanaApp)
    return int(pixels * UI_SCALE)


def scale_vh(vh_value):
    """Scale viewport height values by the UI scaling factor."""
    # Use the current UI_SCALE value (may be updated by CutanaApp)
    return int(vh_value * UI_SCALE)


def scale_percent(base_pixels, container_pixels):
    """Convert scaled pixels to percentage of container for responsive design."""
    scaled_base = scale_px(base_pixels)
    scaled_container = scale_px(container_pixels)
    return min(100, max(0, (scaled_base / scaled_container) * 100))


def get_responsive_width(target_px, max_px=None):
    """Get a responsive width specification that scales properly."""
    scaled_target = scale_px(target_px)
    if max_px:
        scaled_max = scale_px(max_px)
        return f"min(100%, {scaled_target}px, {scaled_max}px)"
    return f"min(100%, {scaled_target}px)"


def get_container_constraints(width_px, height_px=None):
    """Get consistent container constraints that maintain aspect ratios."""
    constraints = {"width": "100%", "max_width": f"{scale_px(width_px)}px"}
    if height_px:
        constraints["max_height"] = f"{scale_px(height_px)}px"
        constraints["height"] = "auto"
    return constraints


def get_ui_scale():
    """Get the current UI scale factor."""
    return UI_SCALE


def set_ui_scale(scale):
    """Set the UI scale factor and recalculate dimensions."""
    global UI_SCALE, CONTAINER_WIDTH, CONTAINER_HEIGHT, MAIN_WIDTH, PANEL_WIDTH, PADDING, VIEWPORT_HEIGHT
    UI_SCALE = scale

    # Recalculate scaled dimensions
    CONTAINER_WIDTH = int(BASE_CONTAINER_WIDTH * UI_SCALE)
    CONTAINER_HEIGHT = int(BASE_CONTAINER_HEIGHT * UI_SCALE)
    MAIN_WIDTH = int(BASE_MAIN_WIDTH * UI_SCALE)
    PANEL_WIDTH = int(BASE_PANEL_WIDTH * UI_SCALE)
    PADDING = int(BASE_PADDING * UI_SCALE)
    VIEWPORT_HEIGHT = int(BASE_VIEWPORT_HEIGHT * UI_SCALE)


# ESA Official Colors from colours.txt
ESA_BLUE_DEEP = "#003249"  # Deep Space Blue (primary)
ESA_BLUE_GREY = "#335E6E"  # Blue-grey
ESA_BLUE_BRIGHT = "#009BDA"  # Bright blue
ESA_BLUE_LIGHT = "#6DCFF6"  # Light blue
ESA_BLUE_ACCENT = "#0098DB"  # Light blue accent
ESA_TEAL = "#00AE9C"  # Teal/turquoise
ESA_GREEN = "#008542"  # Green (for success)
ESA_RED = "#EC1A2F"  # Bright red (for errors)
ESA_ORANGE = "#FBAB18"  # Orange/amber (for warnings)

# Background colors
BACKGROUND_DARK = "#000000"  # Pure black background
BACKGROUND_PANEL = "#003249"  # ESA Deep Space Blue for panels
BORDER_COLOR = "#335E6E"  # ESA Blue-grey for borders

# Text colors
TEXT_COLOR = "#FFFFFF"  # Pure white for main text
TEXT_COLOR_LIGHT = "#E7E8E3"  # Light grey for secondary text
TEXT_COLOR_MUTED = "#9A9B9C"  # Muted grey for less important text

# Status colors
SUCCESS_COLOR = "#008542"  # ESA Green
ERROR_COLOR = "#EC1A2F"  # ESA Red
WARNING_COLOR = "#FBAB18"  # ESA Orange

COMMON_STYLES = """
<style>
.cutana-container {
    background: %(bg_dark)s;
    color: %(text)s;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6;
}

.cutana-panel {
    background: %(bg_dark)s;
    border-radius: 10px;
    padding: 25px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    border: 1px solid %(border)s;
}

.cutana-button-primary {
    background: %(accent)s;
    color: %(text)s;
    padding: 12px 24px;
    border: none;
    border-radius: 5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 16px;
    min-height: 45px;
    min-width: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1.2;
    box-sizing: border-box;
}

.cutana-button-primary:hover {
    background: %(accent_hover)s;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(0, 152, 219, 0.3);
}

.cutana-button-secondary {
    background: %(esa_blue_grey)s;
    color: %(text)s;
    padding: 12px 24px;
    border: 1px solid %(border)s;
    border-radius: 5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 16px;
    min-height: 45px;
    min-width: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1.2;
    box-sizing: border-box;
}

.cutana-button-secondary:hover {
    background: %(border)s;
    color: %(text)s;
}

/* Dropdown styling with darker background */
.widget-dropdown select,
.cutana-dropdown {
    width: 100%%;
    padding: 10px 12px;
    background: %(esa_blue_grey)s !important;
    border: 1px solid %(border)s;
    border-radius: 5px;
    color: %(text)s !important;
    font-size: 14px;
    transition: all 0.3s ease;
    min-height: 40px;
    box-sizing: border-box;
    display: block;
    line-height: 1.4;
}

.widget-dropdown select option {
    background: %(esa_blue_grey)s !important;
    color: %(text)s !important;
}

.widget-dropdown select:focus,
.cutana-dropdown:focus {
    outline: none;
    border-color: %(accent)s;
    box-shadow: 0 0 0 3px rgba(0, 152, 219, 0.2);
}

/* Input field styling */
.cutana-input,
input[type="text"],
input[type="number"] {
    width: 100%%;
    padding: 10px 12px;
    background: %(esa_blue_grey)s;
    border: 1px solid %(border)s;
    border-radius: 5px;
    color: %(text)s;
    font-size: 14px;
    transition: all 0.3s ease;
    min-height: 40px;
    box-sizing: border-box;
    display: block;
    line-height: 1.4;
}

.cutana-input:focus,
input[type="text"]:focus,
input[type="number"]:focus {
    outline: none;
    border-color: %(accent)s;
    box-shadow: 0 0 0 3px rgba(0, 152, 219, 0.2);
}

/* Compact matrix input styling */
.cutana-matrix-input input {
    padding: 2px 6px !important;
    font-size: 11px !important;
    min-height: 20px !important;
    height: 22px !important;
    border: 1px solid %(border)s !important;
    border-radius: 3px !important;
    background: %(esa_blue_grey)s !important;
    color: %(text)s !important;
    text-align: center !important;
    box-sizing: border-box !important;
}

.cutana-matrix-input input:focus {
    outline: none !important;
    border-color: %(accent)s !important;
    box-shadow: 0 0 0 1px rgba(0, 152, 219, 0.3) !important;
}

.cutana-label {
    display: block;
    margin-bottom: 8px;
    color: %(text_light)s;
    font-weight: 500;
}

.cutana-info-text {
    color: %(text_muted)s;
    font-size: 0.9em;
    margin-top: 5px;
}

.cutana-heading {
    color: %(accent)s;
    margin-bottom: 20px;
    font-size: 1.3em;
}

.cutana-progress-bar {
    width: 100%%;
    height: 30px;
    background: %(bg_dark)s;
    border-radius: 15px;
    overflow: hidden;
    position: relative;
    border: 1px solid %(border)s;
}

.cutana-progress-fill {
    height: 100%%;
    background: linear-gradient(90deg, %(accent)s 0%%, #7bc3e0 100%%);
    transition: width 0.3s ease;
    position: relative;
    overflow: hidden;
}

.cutana-spinner {
    display: inline-block;
    width: 40px;
    height: 40px;
    border: 4px solid %(border)s;
    border-radius: 50%%;
    border-top-color: %(accent)s;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.cutana-slider .widget-readout {
    color: %(accent)s !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

.cutana-slider .widget-slider .ui-slider {
    background: %(bg_dark)s !important;
}

.cutana-slider .widget-slider .ui-slider .ui-slider-handle {
    background: %(accent)s !important;
    border-color: %(accent)s !important;
}

.cutana-slider .widget-slider .ui-slider .ui-slider-range {
    background: %(accent)s !important;
}

/* Compact slider with blue background readout */
.cutana-slider-compact .widget-readout {
    color: %(text)s !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    background: %(esa_blue_grey)s !important;
    padding: 2px 6px !important;
    border-radius: 3px !important;
    border: 1px solid %(border)s !important;
    min-width: 40px !important;
}

.cutana-slider-compact .widget-slider {
    flex: 1 !important;
}

.cutana-slider-compact .widget-slider .ui-slider {
    background: %(esa_blue_grey)s !important;
}

.cutana-slider-compact .widget-slider .ui-slider .ui-slider-handle {
    background: %(accent)s !important;
    border-color: %(accent)s !important;
    width: 12px !important;
    height: 12px !important;
}

.cutana-slider-compact .widget-slider .ui-slider .ui-slider-range {
    background: %(accent)s !important;
}

/* Fix checkbox styling - force white text color */
.widget-checkbox label,
.widget-checkbox .widget-label {
    color: %(text)s !important;
}

.widget-checkbox input[type="checkbox"] {
    background: %(esa_blue_grey)s !important;
}

/* Fix number and text input backgrounds */
.widget-text input[type="text"],
.widget-text input[type="number"],
.widget-textarea textarea {
    background: %(esa_blue_grey)s !important;
    color: %(text)s !important;
    border: 1px solid %(border)s !important;
    padding: 4px 8px !important;
    margin: 2px 0 !important;
    border-radius: 4px !important;
}

/* Configuration grid styling for better spacing */
.config-grid-item {
    margin: 2px 4px !important;
    padding: 2px !important;
}

/* Dropdown spacing in configuration grid */
.widget-dropdown select {
    margin: 2px 0 !important;
    padding: 4px 8px !important;
    border-radius: 4px !important;
}

/* Float slider styling for configuration grid */
.widget-hslider {
    margin: 2px 0 !important;
}

.widget-hslider .widget-readout {
    margin-left: 8px !important;
}
/* Color status text */
.success-text { color: %(success)s !important; }
.error-text { color: %(error)s !important; }
.warning-text { color: %(warning)s !important; }

/* 16:9 aspect ratio container */
.aspect-ratio-16-9 {
    position: relative;
    width: 100%%;
    padding-bottom: 56.25%%; /* 9/16 = 0.5625 */
}

/* 5:2 aspect ratio container for preview grid */
.aspect-ratio-5-2 {
    position: relative;
    width: 100%%;
    padding-bottom: 40%%; /* 2/5 = 0.4 */
}

/* Preview grid with proper aspect ratio */
.preview-grid-container {
    width: 100%%;
    height: 0;
    padding-bottom: 40%%; /* 2/5 aspect ratio for 2 rows x 5 columns */
    position: relative;
}

.preview-grid-container .widget-gridbox {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100%% !important;
    height: 100%% !important;
}

.aspect-ratio-content {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%%;
    height: 100%%;
}
</style>
""" % {
    "esa_blue_grey": ESA_BLUE_GREY,
    "accent": ESA_BLUE_ACCENT,
    "accent_hover": ESA_BLUE_LIGHT,
    "bg_dark": BACKGROUND_DARK,
    "border": BORDER_COLOR,
    "text": TEXT_COLOR,
    "text_light": TEXT_COLOR_LIGHT,
    "text_muted": TEXT_COLOR_MUTED,
    "success": SUCCESS_COLOR,
    "error": ERROR_COLOR,
    "warning": WARNING_COLOR,
}


def apply_cutana_style(widget, class_name="cutana-container"):
    """Apply Cutana styling to a widget."""
    widget.add_class(class_name)
    return widget
