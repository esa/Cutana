#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Default configuration for Cutana astronomical cutout pipeline.

This module provides the default configuration using DotMap for dot-accessible dictionaries.
All configuration parameters are documented with their purpose and valid ranges.
"""

import importlib
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import toml
from dotmap import DotMap

from cutana.system_monitor import SystemMonitor

from .normalisation_parameters import get_default_normalisation_config


def get_default_config():
    """Returns the default configuration for Cutana.

    Returns:
        DotMap: The default configuration with dot-accessible parameters
    """
    cfg = DotMap(_dynamic=False)

    # === General Settings ===
    cfg.name = "cutana_run"  # Run identifier
    cfg.log_level = "INFO"  # Logging level for files: DEBUG, INFO, WARNING, ERROR, CRITICAL
    cfg.console_log_level = "WARNING"  # Console/notebook logging level: WARNING, ERROR

    # Generate session timestamp for consistent log file naming across all processes
    cfg.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., "20250925_143022"

    # === Input/Output Configuration ===
    cfg.source_catalogue = None  # Path to source catalogue CSV file (required)

    # Set default output directory with timestamp (same logic as UI)

    try:
        system_monitor = SystemMonitor()
        if system_monitor._is_datalabs_environment():
            # Use datalabs-specific workspace directory with timestamp
            cfg.output_dir = f"/media/home/my_workspace/example_notebook_outputs/cutana_output/{cfg.session_timestamp}"
        else:
            # Default to cutana_output in current working directory
            cfg.output_dir = str(Path.cwd() / "cutana_output")
    except Exception:
        # Fallback if system detection fails
        cfg.output_dir = "cutana_output"

    cfg.output_format = "zarr"  # Output format: "zarr" or "fits"
    cfg.data_type = "float32"  # Output data type: "float32", "float64", "int32", etc.
    cfg.write_to_disk = True  # Write outputs to disk (False for in-memory streaming mode)

    # === Processing Configuration ===
    # Default max_workers to available CPU count (will be capped to N-1 by LoadBalancer)
    try:
        _monitor = SystemMonitor()
        cfg.max_workers = _monitor.get_cpu_count()
    except Exception:
        cfg.max_workers = 16  # Fallback if CPU detection fails
    cfg.N_batch_cutout_process = 1000  # Batch size within each process
    cfg.max_workflow_time_seconds = 1354571  # Maximum total workflow time (default ~2 weeks)
    cfg.process_threads = (
        None  # Optional: Override thread limit per process (None = auto: available_cores // 4)
    )

    # === Cutout Processing Parameters ===
    # Only extract cutouts (dtype=input dtype, fits output, no processing aside flux conversion)
    cfg.do_only_cutout_extraction = False
    # = Further cutout parameters =
    cfg.target_resolution = 256  # Target cutout size in pixels (square cutouts)
    cfg.padding_factor = 1.0  # Padding factor for cutout extraction (0.5-10.0, 1.0 = no padding)
    cfg.normalisation_method = (
        "linear"  # Normalisation method: "linear", "log", "asinh", "zscale", "none"
    )
    cfg.interpolation = "bilinear"  # Interpolation method: "bilinear", "nearest", "cubic"
    cfg.flux_conserved_resizing = (
        False  # Whether to use flux-conserved resizing (drizzle, much slower)
    )

    # === FITS File Handling ===
    cfg.fits_extensions = ["PRIMARY"]  # Default FITS extensions to process
    cfg.selected_extensions = []  # Extensions selected by user (set by UI)
    cfg.available_extensions = []  # Available extensions (discovered during analysis)

    # === Flux Conversion Settings ===
    cfg.apply_flux_conversion = True  # Whether to apply flux conversion (for Euclid data)
    cfg.flux_conversion_keywords = DotMap(_dynamic=False)  # Keywords for flux conversion
    cfg.flux_conversion_keywords.AB_zeropoint = "MAGZERO"  # Header keyword for zeropoint
    cfg.user_flux_conversion_function = None  # Deprecated

    # === Image Normalization Parameters ===
    cfg.normalisation = get_default_normalisation_config()  # Use centralized defaults

    # === External Fitsbolt Configuration ===
    # Optional: External fitsbolt config DotMap from AnomalyMatch or other callers
    # When provided, this config will be used directly for normalization instead of
    # cutana's own normalization settings. This ensures consistent normalization
    # between training and inference when used with ML pipelines.
    cfg.external_fitsbolt_cfg = None

    # === Advanced Processing Settings ===
    cfg.channel_weights = {
        "PRIMARY": [1.0]
    }  # Channel weights dict for multi-channel processing with 3 output channels {"VIS": [1.0, 0.0, 0.75], "NIR-H": [0.0, 1.0, 0.75]}

    # === File Management ===
    cfg.tracking_file = "workflow_tracking.json"  # Job tracking file
    cfg.config_file = None  # Path to saved configuration file

    # === Analysis Results (populated during catalogue analysis) ===
    cfg.num_sources = 0  # Number of sources in catalogue
    cfg.fits_files = []  # List of unique FITS files
    cfg.num_unique_fits_files = 0  # Number of unique FITS files

    # === Load Balancer Configuration ===
    cfg.loadbalancer = DotMap(_dynamic=False)  # Load balancer settings
    cfg.loadbalancer.memory_safety_margin = 0.15  # 10% safety margin for memory allocation
    cfg.loadbalancer.memory_poll_interval = 3  # Poll memory every 3 seconds
    cfg.loadbalancer.memory_peak_window = 30  # Track peak memory over 30 second windows
    cfg.loadbalancer.main_process_memory_reserve_gb = 4.0  # Reserved memory for main process
    # Factor for estimating worker memory (size_of_one_fits_set + N_batch*HWC*n_bits*factor)
    cfg.loadbalancer.initial_workers = 1  # Start with only 1 worker until memory usage is known
    cfg.loadbalancer.max_sources_per_process = None  # Optional: Maximum sources per job/process (None = auto-determined: 12500 for <1M sources, 100000 otherwise)
    cfg.loadbalancer.log_interval = 30  # Log memory estimates every 30 seconds
    cfg.loadbalancer.event_log_file = (
        None  # Optional: File path for LoadBalancer event logging (None = disabled)
    )
    cfg.loadbalancer.skip_memory_calibration_wait = (
        False  # Skip waiting for first worker memory measurements (useful for benchmarking)
    )

    # === UI Configuration (internal use) ===
    cfg.ui = DotMap(_dynamic=False)  # UI-specific settings
    cfg.ui.preview_samples = 10  # Number of preview samples to generate
    cfg.ui.preview_size = 256  # Size of preview cutouts
    cfg.ui.auto_regenerate_preview = True  # Auto-regenerate preview on config change

    # Note: Parameter aliases are handled by validation and mapping layers
    # Core configuration uses canonical names only to avoid confusion

    return cfg


def create_config_from_dict(config_dict):
    """Create a Cutana config from a dictionary, merging with defaults.

    Args:
        config_dict (dict): Configuration dictionary to merge with defaults

    Returns:
        DotMap: Complete configuration with defaults applied
    """
    cfg = get_default_config()

    # Restore special types from their serialized string forms
    def _restore_special_types(d, convert_to_dotmap=False):
        """Recursively restore numpy dtypes and enums from TOML strings.

        Args:
            d: Dictionary to process
            convert_to_dotmap: If True, convert nested dicts to DotMaps (for external configs)
        """
        restored = {}
        for k, v in d.items():
            if isinstance(v, DotMap):
                # Already processed, keep as-is
                restored[k] = v
            elif isinstance(v, dict):
                # Recursively process nested dicts, converting to DotMap if requested
                nested = _restore_special_types(v, convert_to_dotmap)
                if convert_to_dotmap:
                    restored[k] = DotMap(nested, _dynamic=False)
                else:
                    restored[k] = nested
            elif isinstance(v, str):
                if v.startswith("__numpy_dtype__"):
                    # Restore numpy dtype class (e.g., "__numpy_dtype__uint8" -> numpy.uint8)
                    dtype_name = v[len("__numpy_dtype__") :]
                    restored[k] = getattr(np, dtype_name)
                elif v.startswith("__enum__"):
                    # Restore enum (e.g., "__enum__fitsbolt.normalisation.NormalisationMethod__0")
                    parts = v[len("__enum__") :].split("__")
                    enum_path = parts[0]  # e.g., "fitsbolt.normalisation.NormalisationMethod"
                    enum_value = int(parts[1])  # e.g., 0
                    # Split into module and class name
                    module_path = ".".join(enum_path.split(".")[:-1])
                    class_name = enum_path.split(".")[-1]
                    module = importlib.import_module(module_path)
                    enum_class = getattr(module, class_name)
                    restored[k] = enum_class(enum_value)
                else:
                    restored[k] = v
            else:
                restored[k] = v
        return restored

    # Process external_fitsbolt_cfg separately first - convert to DotMap
    if "external_fitsbolt_cfg" in config_dict and config_dict["external_fitsbolt_cfg"] is not None:
        external_cfg = _restore_special_types(
            {"external_fitsbolt_cfg": config_dict["external_fitsbolt_cfg"]}, convert_to_dotmap=True
        )
        config_dict["external_fitsbolt_cfg"] = external_cfg["external_fitsbolt_cfg"]

    # Process rest of config without DotMap conversion (defaults already have DotMaps)
    # DotMaps (like external_fitsbolt_cfg) will be preserved
    config_dict = _restore_special_types(config_dict)

    # Deep merge the provided config
    def _deep_merge(default, override):
        """Recursively merge override into default."""
        for key, value in override.items():
            if key in default and isinstance(default[key], DotMap) and isinstance(value, dict):
                _deep_merge(default[key], value)
            else:
                default[key] = value

    _deep_merge(cfg, config_dict)
    return cfg


def save_config_toml(config, filepath):
    """Save configuration to TOML file.

    Args:
        config (DotMap): Configuration to save
        filepath (str): Path to save TOML file

    Returns:
        str: Path to saved file
    """

    # Convert DotMap to regular dict for TOML serialization
    def _dotmap_to_dict(obj):
        """Convert DotMap to regular dict recursively."""
        if isinstance(obj, DotMap):
            return {k: _dotmap_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_dotmap_to_dict(item) for item in obj]
        else:
            return obj

    config_dict = _dotmap_to_dict(config)

    # Remove None values and functions, convert special types for TOML serialization
    def _clean_dict(d):
        """Remove None values and non-serializable objects, convert special types."""
        cleaned = {}
        for k, v in d.items():
            if v is None:
                continue
            elif isinstance(v, dict):
                cleaned_sub = _clean_dict(v)
                if cleaned_sub:  # Only include non-empty dicts
                    cleaned[k] = cleaned_sub
            elif isinstance(v, type) and issubclass(v, np.generic):
                # Handle numpy dtype classes (e.g., numpy.uint8)
                # Must check before callable() since type objects are callable
                cleaned[k] = f"__numpy_dtype__{v.__name__}"
            elif isinstance(v, Enum):
                # Handle enum values - store as integer with marker
                # Get the fully qualified class name from __module__ (which already includes class for some enums)
                enum_module = type(v).__module__
                enum_class = type(v).__qualname__  # Use qualname for nested classes
                cleaned[k] = f"__enum__{enum_module}.{enum_class}__{v.value}"
            elif callable(v):
                # Skip other callable objects (functions, etc.)
                continue
            else:
                cleaned[k] = v
        return cleaned

    clean_config = _clean_dict(config_dict)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        toml.dump(clean_config, f)

    return str(filepath)


def load_config_toml(filepath):
    """Load configuration from TOML file.

    Args:
        filepath (str): Path to TOML configuration file

    Returns:
        DotMap: Loaded configuration merged with defaults
    """
    with open(filepath, "r") as f:
        config_dict = toml.load(f)

    return create_config_from_dict(config_dict)


def save_config_with_timestamp(config, output_dir):
    """Save configuration with timestamp for UI usage.

    Args:
        config (DotMap): Configuration to save
        output_dir (str): Directory to save configuration

    Returns:
        str: Path to saved file
    """
    # Add UI-specific timestamp metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.timestamp = timestamp
    config.created_at = datetime.now().isoformat()

    # Use timestamped filename for UI
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_file = output_path / f"cutana_config_{timestamp}.toml"
    config.config_file = str(config_file)

    # Save using backend function
    return save_config_toml(config, config_file)
