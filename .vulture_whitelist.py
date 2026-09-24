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
_.min_width  # noqa
_.padding  # noqa
_.margin  # noqa
_.background  # noqa
_.border  # noqa
_.border_radius  # noqa
_.overflow  # noqa
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

# StreamingOrchestrator instance attributes - stored for introspection, not read internally
_._min_workers  # noqa - set by init_streaming, exposed for external inspection

# StreamingOrchestrator public API - documented in README, used in examples/
init_streaming  # noqa - public API for batch streaming workflow
next_batch  # noqa - public API for getting next batch of cutouts
get_batch_count  # noqa - public API for getting total batch count
get_worker_info  # noqa - public API for per-worker batch/timing detail, used in benchmark scripts
get_delivery_report  # noqa - public API for inspecting per-worker cutout shortfalls

# Eager catalogue API documented in README and exercised by catalogue preprocessor tests.
# Discovery uses bounded sampling, but external callers can still request a full DataFrame.
load_and_validate_catalogue  # noqa

# WorkerInfo schema fields (cutana/profiling_types.py) - deliberately defined up front so
# consumers of get_worker_info() know the full per-worker schema (issue #354/#312), even
# the fields read only by external benchmark scripts rather than inside cutana/.
_worker_info_schema = None  # noqa
_worker_info_schema.pool_slot  # noqa - exposed via get_worker_info() for external consumers
_worker_info_schema.sources_per_fits_set  # noqa - exposed via get_worker_info()
_worker_info_schema.batch_index  # noqa - exposed via get_worker_info() for external consumers
_worker_info_schema.end_time  # noqa - set on completion, read by benchmark scripts (not in cutana/)
_worker_info_schema.performance  # noqa - per-stage stats, read by benchmark scripts (not in cutana/)

# Canonical stage list (cutana/profiling_types.py) - the benchmark scripts (outside
# cutana/, so invisible to vulture here) build their STAGE_ORDER from it.
COMPUTE_STAGES  # noqa - consumed by benchmarking/profile_cutana.py and profile_plots.py

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
