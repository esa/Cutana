#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Custom progress bar widget for Cutana UI."""

import ipywidgets as widgets

from ..styles import BACKGROUND_DARK, BORDER_COLOR, ESA_BLUE_ACCENT, scale_px


class CutanaProgressBar(widgets.VBox):
    """Styled progress bar for Cutana UI."""

    def __init__(self, value=0, max_value=100, description=""):
        self.value = value
        self.max_value = max_value

        # Progress bar container
        self.progress_html = widgets.HTML()

        # Description label
        self.description_label = widgets.Label(
            value=description,
            style={"description_width": "initial"},
            layout=widgets.Layout(margin=f"0 0 {scale_px(5)}px 0"),
            height="100%",
        )

        # Style
        self.style_html = widgets.HTML(
            value=f"""
        <style>
        .cutana-progress-bar {{
            width: 100%;
            height: {scale_px(32)}px;
            background: {BACKGROUND_DARK};
            border-radius: {scale_px(16)}px;
            overflow: hidden;
            position: relative;
            border: 1px solid {BORDER_COLOR};
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .cutana-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, {ESA_BLUE_ACCENT} 0%, #7bc3e0 100%);
            transition: width 0.3s ease;
            position: absolute;
            left: 0;
            top: 0;
            z-index: 1;
        }}

        .cutana-progress-text {{
            position: relative;
            z-index: 2;
            color: white;
            font-size: 12px;
            font-weight: 600;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        }}

        .cutana-progress-fill::after {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.2),
                transparent
            );
            animation: shimmer 2s infinite;
        }}

        @keyframes shimmer {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        </style>
        """
        )

        children = [self.style_html]
        if description:
            children.append(self.description_label)
        children.append(self.progress_html)

        super().__init__(children=children)

        self._update_display()

    def _get_percentage(self):
        """Calculate current percentage."""
        if self.max_value == 0:
            return 0
        return int((self.value / self.max_value) * 100)

    def _update_display(self):
        """Update the progress bar display."""
        percentage = self._get_percentage()
        self.progress_html.value = f"""
        <div class="cutana-progress-bar">
            <div class="cutana-progress-fill" style="width: {percentage}%"></div>
            <div class="cutana-progress-text">{percentage}% Complete</div>
        </div>
        """

    def update(self, value=None, max_value=None, description=None):
        """Update progress bar values."""
        if value is not None:
            self.value = value
        if max_value is not None:
            self.max_value = max_value
        if description is not None:
            self.description_label.value = description
        self._update_display()
