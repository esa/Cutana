#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Custom file/folder chooser wrapper for Cutana UI."""

import os
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from loguru import logger

from ..styles import (
    BACKGROUND_DARK,
    BORDER_COLOR,
    TEXT_COLOR,
    TEXT_COLOR_LIGHT,
    ESA_BLUE_ACCENT,
    scale_px,
)


class CutanaFileChooser(widgets.VBox):
    """Styled file chooser for Cutana UI."""

    def __init__(self, path=os.getcwd(), select_default=False, show_only_dirs=False, **kwargs):
        self.file_chooser = FileChooser(
            path=path,
            select_default=select_default,
            filename="input_cat.csv",
            show_only_dirs=show_only_dirs,
            **kwargs,
        )

        # Apply specific file chooser styling - avoid global selectors
        style = f"""
        <style>
        /* Only target the file chooser container directly */
        .cutana-file-chooser .widget-file-chooser,
        .cutana-file-chooser .filechooser {{
            background-color: {BACKGROUND_DARK} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 5px !important;
            max-height: {scale_px(320)}px !important;
            overflow: auto !important;
        }}

        /* File chooser specific inputs only */
        .cutana-file-chooser select,
        .cutana-file-chooser input {{
            background-color: {BACKGROUND_DARK} !important;
            color: {TEXT_COLOR} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 3px !important;
            padding: {scale_px(6)}px !important;
            min-height: {scale_px(28)}px !important;
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }}

        /* Focus states for file chooser */
        .cutana-file-chooser select:focus,
        .cutana-file-chooser input:focus {{
            outline: none !important;
            border-color: {ESA_BLUE_ACCENT} !important;
            box-shadow: 0 0 0 2px rgba(94, 179, 214, 0.2) !important;
        }}

        /* File chooser buttons only */
        .cutana-file-chooser button {{
            background-color: {ESA_BLUE_ACCENT} !important;
            color: {BACKGROUND_DARK} !important;
            border: none !important;
            border-radius: 3px !important;
            padding: 8px 12px !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            min-height: 32px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            line-height: 1.2 !important;
            box-sizing: border-box !important;
        }}

        /* File chooser button hover */
        .cutana-file-chooser button:hover {{
            background-color: #7bc3e0 !important;
            transform: translateY(-1px) !important;
        }}

        /* File chooser labels and text */
        .cutana-file-chooser .filename,
        .cutana-file-chooser label {{
            color: {TEXT_COLOR_LIGHT} !important;
        }}

        /* File chooser dropdown options */
        .cutana-file-chooser select option {{
            background-color: {BACKGROUND_DARK} !important;
            color: {TEXT_COLOR} !important;
        }}

        /* File chooser 'No Selection' text styling - comprehensive selectors */
        .cutana-file-chooser .filechooser-label,
        .cutana-file-chooser .no-selection,
        .cutana-file-chooser .placeholder,
        .cutana-file-chooser .filename-label,
        .cutana-file-chooser input[placeholder],
        .cutana-file-chooser .widget-text,
        .cutana-file-chooser .widget-html,
        .cutana-file-chooser .widget-readout,
        .cutana-file-chooser .file-selector-label,
        .cutana-file-chooser .selected-path {{
            color: {TEXT_COLOR_LIGHT} !important;
        }}

        /* Override any black text specifically in labels and divs */
        .cutana-file-chooser div,
        .cutana-file-chooser span,
        .cutana-file-chooser p {{
            color: {TEXT_COLOR_LIGHT} !important;
        }}
        
        /* Force all file chooser internal containers to respect width */
        .cutana-file-chooser .widget-box,
        .cutana-file-chooser .widget-hbox,
        .cutana-file-chooser .widget-vbox {{
            width: 100% !important;
            max-width: 100% !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
        }}

        /* Force override inline styles for black text */
        .cutana-file-chooser [style*="color: black"],
        .cutana-file-chooser [style*="color: #000"],
        .cutana-file-chooser [style*="color: rgb(0, 0, 0)"] {{
            color: {TEXT_COLOR_LIGHT} !important;
        }}

        /* Ensure file chooser has responsive size */
        .cutana-file-chooser {{
            min-height: {scale_px(180)}px !important;
            width: 100% !important;
            max-width: 100% !important;
        }}
        
        /* Prevent horizontal overflow in file chooser - comprehensive */
        .cutana-file-chooser,
        .cutana-file-chooser .widget-file-chooser,
        .cutana-file-chooser .filechooser,
        .cutana-file-chooser > *,
        .cutana-file-chooser * {{
            max-width: 100% !important;
            overflow-x: hidden !important;
            box-sizing: border-box !important;
        }}
        </style>
        """

        super().__init__(
            children=[widgets.HTML(value=style), self.file_chooser],
            layout=widgets.Layout(
                min_height=f"{scale_px(300)}px",
                max_height=f"{scale_px(350)}px",
                width="100%",
                max_width="100%",
            ),
        )
        self.add_class("cutana-file-chooser")

    @property
    def selected(self):
        """Get the selected file/folder path."""
        return self.file_chooser.selected

    @selected.setter
    def selected(self, value):
        """Set the selected file/folder path."""
        self.file_chooser.selected = value

    @property
    def selected_filename(self):
        """Get the selected filename."""
        return self.file_chooser.selected_filename

    def set_selected_file(self, file_path):
        """Try to programmatically select a file in the file chooser."""
        try:
            from pathlib import Path

            if not file_path or not Path(file_path).exists():
                logger.debug(f"File path invalid or doesn't exist: {file_path}")
                return False

            file_path_obj = Path(file_path)
            parent_dir = str(file_path_obj.parent)
            filename = file_path_obj.name

            logger.debug(f"Attempting to select file: {file_path}")
            logger.debug(f"Parent directory: {parent_dir}")
            logger.debug(f"Filename: {filename}")

            # Set the directory path first
            self.file_chooser.path = parent_dir

            # Method 1: Use default_filename and default_path (most reliable for ipyfilechooser)
            success = False
            try:
                self.file_chooser.default_path = parent_dir
                self.file_chooser.default_filename = filename

                # Force refresh to update the UI
                if hasattr(self.file_chooser, "refresh"):
                    self.file_chooser.refresh()

                # Check if selection worked
                current_selection = self.selected
                if current_selection and Path(current_selection).name == filename:
                    success = True
                    logger.debug("Method 1 (default_path + default_filename) succeeded")

            except Exception as e:
                logger.debug(f"Method 1 failed: {e}")

            # Method 2: Try setting the 'default' property directly
            if not success:
                try:
                    self.file_chooser.default = str(file_path)
                    current_selection = self.selected
                    if current_selection and str(current_selection) == str(file_path):
                        success = True
                        logger.debug("Method 2 (default property) succeeded")
                except Exception as e:
                    logger.debug(f"Method 2 failed: {e}")

            # Method 3: Direct assignment to selected (might not work but worth trying)
            if not success:
                try:
                    # This is a last resort and might not work due to widget constraints
                    if hasattr(self.file_chooser, "_selected"):
                        self.file_chooser._selected = str(file_path)
                        current_selection = self.selected
                        if current_selection and str(current_selection) == str(file_path):
                            success = True
                            logger.debug("Method 3 (direct _selected) succeeded")
                except Exception as e:
                    logger.debug(f"Method 3 failed: {e}")

            final_selection = self.selected
            logger.debug(f"Final selection: {final_selection}")
            logger.debug(f"Selection success: {success}")

            return success

        except Exception as e:
            logger.error(f"set_selected_file failed: {e}")
            return False

    def reset(self, path=None):
        """Reset the file chooser."""
        if path:
            self.file_chooser.reset(path)
        else:
            self.file_chooser.reset()
