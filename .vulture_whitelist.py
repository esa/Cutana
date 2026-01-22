#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Vulture whitelist - false positives for legitimate code.

These entries represent code that IS used but vulture cannot detect the usage:
- WCS attributes: Written by our code, read by external libraries (astropy, drizzle)
- DotMap config: Accessed dynamically via attribute access
- ipywidgets attributes: Set by our code, read by the UI framework
- Public API: Exported for external use, documented in README

Usage: vulture cutana cutana_ui .vulture_whitelist.py --min-confidence 60
"""

# WCS attributes - written and consumed by external libraries (astropy, drizzle)
_.crpix  # noqa
_.cdelt  # noqa
_.crval  # noqa
_.ctype  # noqa
_.array_shape  # noqa

# Config attributes accessed dynamically via DotMap
_.num_unique_fits_files  # noqa
_.preview_samples  # noqa
_.preview_size  # noqa
_.auto_regenerate_preview  # noqa
_.created_at  # noqa

# UI widget attributes (ipywidgets style/layout attributes set but read by framework)
_.button_style  # noqa
_.button_color  # noqa
_.preview_cutouts  # noqa
_.original_layout  # noqa
_.channel_matrix  # noqa
_.max_width  # noqa
_.margin  # noqa
_.crop_enable_label  # noqa
_.disabled  # noqa
_.default_filename  # noqa

# UI methods used for polling/event handling
get_processing_status  # noqa - called via asyncio polling from main_screen

# UI style constants imported by test files that verify theming
ESA_BLUE_DEEP  # noqa
ESA_GREEN  # noqa
ESA_RED  # noqa

# PreviewCache class attribute - accessed dynamically within class methods
_.config_cache  # noqa

# StreamingOrchestrator public API - documented in README, used in examples/async_streaming.py
init_streaming  # noqa - public API for batch streaming workflow
next_batch  # noqa - public API for getting next batch of cutouts
get_batch_count  # noqa - public API for getting total batch count
get_batch  # noqa - public API for random access to batches

# SystemMonitor utility methods - public API for resource monitoring
check_memory_constraints  # noqa - utility for checking available memory
estimate_memory_usage  # noqa - utility for estimating memory requirements
record_resource_snapshot  # noqa - utility for recording resource history
get_resource_history  # noqa - utility for retrieving resource history
get_conservative_cpu_limit  # noqa - utility for conservative CPU allocation

# UILogManager public API - imported and used by app.py, main_screen.py, start_screen.py
setup_ui_logging  # noqa - public API for setting up UI logging with file handler
set_console_log_level  # noqa - public API for dynamically changing console log level

# Styles module public API - utility functions for UI scaling
scale_vh  # noqa - public API for scaling viewport height values (symmetric with scale_px)
