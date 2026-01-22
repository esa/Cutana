#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Cutana - High-performance Python pipeline for creating astronomical image cutouts.

This package provides tools for efficiently creating cutouts from large FITS tile
collections with parallel processing capabilities and flexible output formats.

Logging:
    Cutana uses loguru for logging and follows library best practices by disabling
    logging by default. To see cutana's logs, users can:

    1. Enable logging: logger.enable("cutana")
    2. Configure their own handlers: logger.add(...)

    Or use cutana's setup_logging() which automatically enables and configures logging.
"""

from loguru import logger

# Disable logging by default (library best practice)
# Users can re-enable with: logger.enable("cutana")
# Application entry points (Orchestrator, UI) will enable logging when needed
logger.disable("cutana")

__version__ = "0.2.1"
__author__ = "ESA Datalabs"

# Import main classes for easy access
# These imports are after logger.disable() to ensure logging is disabled before module initialization
# Import deployment validation
from .deployment_validator import deployment_validation  # noqa: E402

# Import configuration management functions
from .get_default_config import (  # noqa: E402
    create_config_from_dict,
    get_default_config,
    load_config_toml,
    save_config_toml,
)
from .job_tracker import JobTracker  # noqa: E402
from .orchestrator import Orchestrator  # noqa: E402
from .streaming_orchestrator import StreamingOrchestrator  # noqa: E402
from .validate_config import validate_config, validate_config_for_processing  # noqa: E402

__all__ = [
    "Orchestrator",
    "StreamingOrchestrator",
    "JobTracker",
    "get_default_config",
    "create_config_from_dict",
    "save_config_toml",
    "load_config_toml",
    "validate_config",
    "validate_config_for_processing",
    "deployment_validation",
]
