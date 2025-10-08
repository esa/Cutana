#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Help panel component for Cutana UI."""

import os
import ipywidgets as widgets

from ..utils.markdown_loader import get_markdown_content, format_markdown_display
from ..utils.svg_loader import get_logo_html
from ..styles import scale_px, BORDER_COLOR, BACKGROUND_DARK, ESA_BLUE_ACCENT, ESA_BLUE_BRIGHT

HELP_BUTTON_WIDTH = 130
HELP_BUTTON_HEIGHT = 40


def create_header_container(version_text, container_width, help_button_callback, logo_title=None):
    """
    Create a header container with ESA logo, version display, and help button.

    Args:
        version_text (str): The version text to display
        container_width (int): The width of the container in pixels
        help_button_callback (callable): Function to call when help button is clicked
        logo_title (str, optional): Title text for the logo. If provided, logo will be displayed.

    Returns:
        widgets.HBox: The header container
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
                overflow="visible",  # Allow full logo to be visible
            ),
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

    # Help button container (right side) - fixed width
    help_button_container = widgets.HBox(
        children=[help_button],
        layout=widgets.Layout(
            width=f"{scale_px(HELP_BUTTON_WIDTH + 20)}px",  # Fixed width (button width + padding)
            justify_content="flex-end",
            align_items="center",
        ),
    )

    # Create children list based on whether logo is provided
    if logo_widget:
        children = [version_display, logo_widget, help_button_container]
    else:
        children = [version_display, help_button_container]

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
            overflow="visible",  # Ensure logo is not clipped
        ),
    )

    return header_container, help_button


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
        self.main_readme_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../README.md")
        )
        self.ui_readme_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../README.md")
        )

        # Track current readme being displayed
        self.current_readme_path = self.main_readme_path

        # Contact information with paths
        # TODO link github Issues page
        self.contact_info = widgets.HTML(
            value=f"""
            <div style="padding: {scale_px(10)}px; border-bottom: 1px solid {BORDER_COLOR}; background: {BACKGROUND_DARK};">
                <p style="margin: 5px 0; font-weight: bold; color: #c8d0e0;">
                    <span style="color: #88c0d0;">Main README:</span> {self.main_readme_path}
                </p>
                <p style="margin: 5px 0; font-weight: bold; color: #c8d0e0;">
                    <span style="color: #88c0d0;">UI README:</span> {self.ui_readme_path}
                </p>
                <p style="margin: 5px 0; font-weight: bold; color: #c8d0e0;">
                    <span style="color: #88c0d0;">Contact:</span> 
                    <a href="mailto:david.oryan@esa.int" style="color: #8fbcbb; text-decoration: none;">david.oryan@esa.int</a>
                </p>
            </div>
            """
        )

        # Load README content
        readme_content = get_markdown_content(self.current_readme_path)
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
        if self.current_readme_path == self.main_readme_path:
            self.current_readme_path = self.ui_readme_path
            self.switch_button.description = "Switch to General Help"
        else:
            self.current_readme_path = self.main_readme_path
            self.switch_button.description = "Switch to UI Help"

        # Update content
        readme_content = get_markdown_content(self.current_readme_path)
        formatted_content = format_markdown_display(readme_content)
        self.readme_display.value = formatted_content

    def _on_close(self, _):
        """Handle close button click."""
        if self.on_close_callback:
            self.on_close_callback()
