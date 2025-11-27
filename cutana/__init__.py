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
"""

__version__ = "0.1.2"
__author__ = "ESA Datalabs"
__email__ = "datalabs@esa.int"

# Import main classes for easy access
from .orchestrator import Orchestrator
from .job_tracker import JobTracker

# Import configuration management functions
from .get_default_config import (
    get_default_config,
    create_config_from_dict,
    save_config_toml,
    load_config_toml,
)
from .validate_config import validate_config, validate_config_for_processing

# Import deployment validation
from .deployment_validator import deployment_validation

__all__ = [
    "Orchestrator",
    "JobTracker",
    "get_default_config",
    "create_config_from_dict",
    "save_config_toml",
    "load_config_toml",
    "validate_config",
    "validate_config_for_processing",
    "deployment_validation",
]
