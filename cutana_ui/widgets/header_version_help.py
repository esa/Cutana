#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Help panel component for Cutana UI."""

import os
from importlib.metadata import metadata
from importlib.resources import files

import ipywidgets as widgets

from ..styles import BACKGROUND_DARK, BORDER_COLOR, ESA_BLUE_ACCENT, ESA_BLUE_BRIGHT, scale_px
from ..utils.markdown_loader import format_markdown_display, get_markdown_content
from ..utils.svg_loader import get_logo_html


def _get_readme_path(package_name: str, filename: str, relative_fallback: str) -> str:
    """
    Get the path to a README file, trying relative path first, then package resources.

    Args:
        package_name: Name of the package to look in (e.g., "cutana_ui")
        filename: Name of the file to find (e.g., "README.md")
        relative_fallback: Relative path from this file as fallback for dev installs

    Returns:
        Absolute path to the README file
    """
    # First try relative path (works for development/editable installs)
    relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_fallback))
    if os.path.exists(relative_path):
        return relative_path

    # Try to find via importlib.resources (works for pip installs)
    try:
        package_files = files(package_name)
        readme_file = package_files / filename
        # Convert to string path - works with Python 3.9+ Traversable
        if hasattr(readme_file, "__fspath__"):
            resource_path = os.fspath(readme_file)
        else:
            resource_path = str(readme_file)
        if os.path.exists(resource_path):
            return resource_path
    except (ModuleNotFoundError, TypeError, FileNotFoundError):
        pass

    # Return relative path as last resort
    return relative_path


def _get_main_readme_source() -> str:
    """
    Get the source location of the main README (file path or metadata indicator).

    Returns:
        Path string if file exists, or description of source for pip installs
    """
    relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../README.md"))
    if os.path.exists(relative_path):
        return relative_path
    return "package metadata (pip install)"


def _get_main_readme_content() -> str:
    """
    Get main README content, trying file first, then package metadata.

    Returns:
        README content as string
    """
    # First try relative path (works for development/editable installs)
    relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../README.md"))
    if os.path.exists(relative_path):
        return get_markdown_content(relative_path)

    # For pip installs, get README from package metadata (set via readme= in pyproject.toml)
    try:
        pkg_metadata = metadata("cutana")
        description = pkg_metadata.get_payload()
        if description:
            return description
    except Exception:
        pass

    return "Main README not available. See https://github.com/esa/Cutana for documentation."


HELP_BUTTON_WIDTH = 130
HELP_BUTTON_HEIGHT = 40

LOG_LEVELS = ["Debug", "Info", "Warning", "Error"]
DEFAULT_LOG_LEVEL = "Warning"


def create_header_container(
    version_text,
    container_width,
    help_button_callback,
    log_level_callback=None,
    logo_title=None,
    initial_log_level=None,
):
    """
    Create a header container with ESA logo, version display, log level selector, and help button.

    Args:
        version_text (str): The version text to display
        container_width (int): The width of the container in pixels
        help_button_callback (callable): Function to call when help button is clicked
        log_level_callback (callable, optional): Function to call when log level is changed.
            Receives the new log level string as argument.
        logo_title (str, optional): Title text for the logo. If provided, logo will be displayed.
        initial_log_level (str, optional): Initial log level to set in the dropdown.
            If None, uses DEFAULT_LOG_LEVEL.

    Returns:
        tuple: (header_container, help_button, log_level_dropdown)
    """
    # Version display (left side) - fixed width
    version_display = widgets.HTML(
        value=f'<span style="color: #aaaaaa; font-size: 12px; padding: {scale_px(5)}px;">{version_text}</span>',
        layout=widgets.Layout(
            width=f"{scale_px(120)}px", justify_content="flex-start"  # Fixed width for version
        ),
    )

    # ESA Logo (center) - takes remaining space
    logo_widget = None
    if logo_title:
        logo_content = widgets.HTML(value=get_logo_html(logo_title))
        logo_widget = widgets.HBox(
            children=[logo_content],
            layout=widgets.Layout(
                margin="0 auto",
                justify_content="center",
                align_items="center",
                flex="1",  # Takes all remaining space
                overflow="hidden",
            ),
        )

    # Log level label
    log_level_label = widgets.HTML(
        value=f'<span style="color: #aaaaaa; font-size: {scale_px(12)}px;">Log Level </span>',
    )

    # Log level dropdown (right side, before help button) - same width as help button
    log_level_value = initial_log_level if initial_log_level in LOG_LEVELS else DEFAULT_LOG_LEVEL
    log_level_dropdown = widgets.Dropdown(
        options=LOG_LEVELS,
        value=log_level_value,
        description="",
        layout=widgets.Layout(width=f"{scale_px(HELP_BUTTON_WIDTH)}px"),
    )
    log_level_dropdown.add_class("cutana-log-dropdown")
    if log_level_callback:
        log_level_dropdown.observe(
            lambda change: (
                log_level_callback(change["new"].upper()) if change["name"] == "value" else None
            ),
            names="value",
        )

    # Help button (right side)
    help_button = widgets.Button(
        description="Help",
        button_style="danger",
        layout=widgets.Layout(
            width=f"{scale_px(HELP_BUTTON_WIDTH)}px",
            height=f"{scale_px(HELP_BUTTON_HEIGHT)}px",
        ),
    )
    help_button.on_click(help_button_callback)

    # Right side container with log level label, dropdown, and help button
    right_container = widgets.HBox(
        children=[log_level_label, log_level_dropdown, help_button],
        layout=widgets.Layout(
            justify_content="flex-end",
            align_items="center",
            gap=f"{scale_px(5)}px",
        ),
    )

    # Create children list based on whether logo is provided
    if logo_widget:
        children = [version_display, logo_widget, right_container]
    else:
        children = [version_display, right_container]

    # Create a header container
    header_container = widgets.HBox(
        children=children,
        layout=widgets.Layout(
            width="100%",
            max_width=f"{container_width}px",
            margin="0 auto 0 auto",  # Remove bottom margin
            justify_content="space-between",
            padding=f"{scale_px(3)}px",  # Reduced padding
            align_items="center",
            height="auto",
            overflow="hidden",
        ),
    )

    return header_container, help_button, log_level_dropdown


class HelpPopup(widgets.VBox):
    """
    A panel widget to display help information including README content.
    """

    def __init__(self, on_close_callback=None):
        """
        Initialize the help panel.

        Args:
            on_close_callback (callable, optional): Function to call when close button is clicked
        """
        self.on_close_callback = on_close_callback

        # Header with title, switch button, and close button
        self.title = widgets.HTML(
            value=f'<h2 style="margin: 0; color: {ESA_BLUE_ACCENT}; font-size:  {scale_px(20)}px;">Cutana Help</h2>'
        )

        self.close_button = widgets.Button(
            description="Back",
            button_style="primary",
            layout=widgets.Layout(width="70px", height="30px"),
        )
        self.close_button.on_click(self._on_close)
        self.close_button.style.button_color = ESA_BLUE_ACCENT

        # Switch button for toggling between READMEs
        self.switch_button = widgets.Button(
            description="Switch to UI Help",
            button_style="info",
            layout=widgets.Layout(width="180px", height="30px"),
        )
        self.switch_button.on_click(self._toggle_readme)
        self.switch_button.style.button_color = ESA_BLUE_BRIGHT

        self.header = widgets.HBox(
            [self.title, self.switch_button, self.close_button],
            layout=widgets.Layout(
                justify_content="space-between",
                align_items="center",
                padding=f"{scale_px(10)}px",
                border_bottom=f"1px solid {BORDER_COLOR}",
            ),
        )

        # Get README paths
        self.ui_readme_path = _get_readme_path("cutana_ui", "README.md", "../README.md")
        self.main_readme_source = _get_main_readme_source()

        # Track which readme is being displayed: "main" or "ui"
        self.current_readme = "main"

        # Contact information with paths (useful for developers to find install location)
        self.contact_info = widgets.HTML(
            value=f"""
            <div style="padding: {scale_px(10)}px; border-bottom: 1px solid {BORDER_COLOR}; background: {BACKGROUND_DARK};">
                <p style="margin: 5px 0; font-weight: bold; color: #c8d0e0;">
                    <span style="color: #88c0d0;">Main README:</span> {self.main_readme_source}
                </p>
                <p style="margin: 5px 0; font-weight: bold; color: #c8d0e0;">
                    <span style="color: #88c0d0;">UI README:</span> {self.ui_readme_path}
                </p>
                <p style="margin: 5px 0; font-weight: bold; color: #c8d0e0;">
                    <span style="color: #88c0d0;">Documentation:</span>
                    <a href="https://github.com/esa/Cutana" style="color: #8fbcbb; text-decoration: none;">github.com/esa/Cutana</a>
                </p>
                <p style="margin: 5px 0; font-weight: bold; color: #c8d0e0;">
                    <span style="color: #88c0d0;">Report Issues:</span>
                    <a href="https://github.com/esa/Cutana/issues" style="color: #8fbcbb; text-decoration: none;">github.com/esa/Cutana/issues</a>
                </p>
                <p style="margin: 5px 0; font-weight: bold; color: #c8d0e0;">
                    <span style="color: #88c0d0;">Contact:</span>
                    <a href="mailto:david.oryan@esa.int" style="color: #8fbcbb; text-decoration: none;">david.oryan@esa.int</a>
                </p>
            </div>
            """
        )

        # Load README content (main README from file or package metadata)
        readme_content = _get_main_readme_content()
        formatted_content = format_markdown_display(readme_content)

        self.readme_display = widgets.HTML(
            value=formatted_content,
            layout=widgets.Layout(
                overflow="auto", padding=f"{scale_px(0)}px", flex="1", height="auto"
            ),
        )

        # Assemble panel
        super().__init__(
            children=[self.header, self.contact_info, self.readme_display],
            layout=widgets.Layout(
                width="100%",
                height="100%",
                border=f"1px solid {BORDER_COLOR}",
                border_radius=f"{scale_px(10)}px",
                background=BACKGROUND_DARK,
                overflow="hidden",
                display="flex",
                flex_flow="column",
            ),
        )

    def _toggle_readme(self, _):
        """Toggle between main README and UI README."""
        if self.current_readme == "main":
            self.current_readme = "ui"
            self.switch_button.description = "Switch to General Help"
            readme_content = get_markdown_content(self.ui_readme_path)
        else:
            self.current_readme = "main"
            self.switch_button.description = "Switch to UI Help"
            readme_content = _get_main_readme_content()

        # Update content
        formatted_content = format_markdown_display(readme_content)
        self.readme_display.value = formatted_content

    def _on_close(self, _):
        """Handle close button click."""
        if self.on_close_callback:
            self.on_close_callback()
