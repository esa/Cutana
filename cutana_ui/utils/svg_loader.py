#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""SVG loading utilities for Cutana UI."""

from pathlib import Path
from loguru import logger


def load_esa_logo() -> str:
    """Load ESA logo SVG from assets directory.

    Returns:
        str: SVG content as string, or empty string if loading fails
    """
    try:
        # Find the assets directory relative to this file
        current_file = Path(__file__)
        project_root = (
            current_file.parent.parent.parent
        )  # cutana_ui/utils -> cutana_ui -> project_root
        logo_path = project_root / "assets" / "ESA_logo.svg"

        if logo_path.exists():
            svg_content = logo_path.read_text(encoding="utf-8")
            logger.debug(f"Successfully loaded ESA logo from: {logo_path}")
            return svg_content
        else:
            logger.warning(f"ESA logo file not found at: {logo_path}")
            return ""

    except Exception as e:
        logger.error(f"Error loading ESA logo: {e}")
        return ""


def get_logo_html(title: str = "CUTANA") -> str:
    """Get HTML with ESA logo and title.

    Args:
        title: Title text to display next to logo

    Returns:
        str: HTML content with logo and title
    """
    svg_content = load_esa_logo()

    return f"""
    <div style="display: flex; align-items: center; margin-bottom: 5px; padding: 5px;">
        {svg_content}
        <h1 style="color: white; font-size: 24px; font-weight: 300; margin-left: 20px;">{title}</h1>
    </div>
    """
