#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Loading spinner widget for Cutana UI."""

import ipywidgets as widgets

from ..styles import BORDER_COLOR, ESA_BLUE_ACCENT


class LoadingSpinner(widgets.VBox):
    """A loading spinner widget with optional message."""

    def __init__(self, message="Loading..."):
        self.message_label = widgets.HTML(
            value=f'<div style="text-align: center; color: {ESA_BLUE_ACCENT}; margin-top: 10px;">{message}</div>'
        )

        spinner_html = widgets.HTML(
            value=f"""
        <style>
        .cutana-spinner {{
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid {BORDER_COLOR};
            border-radius: 50%;
            border-top-color: {ESA_BLUE_ACCENT};
            animation: spin 1s ease-in-out infinite;
            margin: 0 auto;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        </style>
        <div style="text-align: center;">
            <div class="cutana-spinner"></div>
        </div>
        """
        )

        super().__init__(
            children=[spinner_html, self.message_label],
            layout=widgets.Layout(align_items="center", justify_content="center", padding="20px"),
        )

    def update_message(self, message):
        """Update the loading message."""
        self.message_label.value = f'<div style="text-align: center; color: {ESA_BLUE_ACCENT}; \
margin-top: 10px;">{message}</div>'
