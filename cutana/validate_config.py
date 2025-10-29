#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Configuration validation for Cutana astronomical cutout pipeline.

This module provides comprehensive validation for Cutana configuration parameters,
ensuring all required parameters are present and have valid values.
"""

from dotmap import DotMap
from loguru import logger
import os
import inspect
from .normalisation_parameters import NormalisationRanges


def _return_required_and_optional_keys():
    """
    Returns the configuration parameters specification for Cutana.

    Returns:
        dict: Dictionary with parameter_name as key and [dtype, min, max, optional, allowed_values] as value
              - dtype: expected data type (str, int, float, bool, list, tuple, 'directory', 'file', 'special')
              - min: minimum value (None if not applicable)
              - max: maximum value (None if not applicable)
              - optional: True if parameter is optional, False if required
              - allowed_values: list of allowed values (None if not applicable)
    """
    config_spec = {
        # === General Settings ===
        "name": [str, None, None, False, None],
        "log_level": [
            str,
            None,
            None,
            False,
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"],
        ],
        "console_log_level": [
            str,
            None,
            None,
            False,
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"],
        ],
        # === Input/Output Configuration ===
        "source_catalogue": ["file", None, None, False, None],  # Required for processing
        "output_dir": ["directory_or_create", None, None, False, None],  # Can be created
        "output_format": [str, None, None, False, ["zarr", "fits"]],
        "data_type": [str, None, None, False, ["float32", "uint8"]],
        # === Processing Configuration ===
        "max_workers": [int, 1, 1024, False, None],  # 1-1024 workers
        "N_batch_cutout_process": [int, 10, 10000, False, None],  # 10-10k batch size
        "max_workflow_time_seconds": [int, 600, 5e6, False, None],  # 10min - 60 days
        # === Cutout Processing Parameters ===
        "target_resolution": [int, 16, 2048, False, None],  # 16-2048 pixels
        "padding_factor": [float, 0.25, 10.0, False, None],  # 0.25-10.0 padding factor (mandatory)
        "normalisation_method": [
            str,
            None,
            None,
            False,
            ["linear", "log", "asinh", "zscale", "none"],
        ],
        "interpolation": [
            str,
            None,
            None,
            False,
            ["bilinear", "nearest", "bicubic", "biquadratic"],
        ],
        # === FITS File Handling ===
        "fits_extensions": ["special_fits_extensions", None, None, False, None],
        "selected_extensions": [
            "special_selected_extensions",
            None,
            None,
            True,
            None,
        ],  # Optional, set by UI
        "available_extensions": [
            list,
            None,
            None,
            True,
            None,
        ],  # Optional, discovered during analysis
        # === Flux Conversion Settings ===
        "apply_flux_conversion": [bool, None, None, False, None],
        "flux_conversion_keywords": ["special_flux_keywords", None, None, True, None],
        "flux_conversion_keywords.AB_zeropoint": [
            str,
            None,
            None,
            True,
            None,
        ],  # Header keyword for zeropoint
        "user_flux_conversion_function": ["special_function", None, None, True, None],
        # === Image Normalization Parameters ===
        "normalisation": [DotMap, None, None, True, None],  # Nested config
        # Nested normalisation parameters
        "normalisation.a": [
            float,
            NormalisationRanges.ASINH_A_MIN,
            NormalisationRanges.LOG_A_MAX,
            True,
            None,
        ],
        "normalisation.percentile": [
            float,
            NormalisationRanges.PERCENTILE_MIN,
            NormalisationRanges.PERCENTILE_MAX,
            True,
            None,
        ],
        "normalisation.n_samples": [
            int,
            NormalisationRanges.N_SAMPLES_MIN,
            NormalisationRanges.N_SAMPLES_MAX,
            True,
            None,
        ],  # Optional, ZScale samples
        "normalisation.contrast": [
            float,
            NormalisationRanges.CONTRAST_MIN,
            NormalisationRanges.CONTRAST_MAX,
            True,
            None,
        ],  # Optional, ZScale contrast
        "normalisation.crop_width": [
            int,
            0,
            5000,
            True,
            None,
        ],
        "normalisation.crop_height": [
            int,
            0,
            5000,
            True,
            None,
        ],
        "normalisation.crop_enable": [bool, None, None, True, None],
        # === Advanced Processing Settings ===
        "channel_weights": ["special_channel_weights_dict", None, None, True, None],
        # === File Management ===
        "tracking_file": [str, None, None, False, None],
        "config_file": [str, None, None, True, None],  # Optional
        # === Analysis Results (populated during catalogue analysis) ===
        "num_sources": [int, 0, None, True, None],  # Optional, can be 0
        "fits_files": [list, None, None, True, None],  # Optional
        "num_unique_fits_files": [int, 0, None, True, None],  # Optional, can be 0
        # === Memory and Resource Management ===
        "memory_limit_gb": [float, 0.1, 1024.0, True, None],  # Optional, 0.1GB - 1TB
        # === LoadBalancer Configuration ===
        "loadbalancer.memory_safety_margin": [
            float,
            0.01,
            0.5,
            True,
            None,
        ],  # 1% - 50% safety margin
        "loadbalancer": [DotMap, None, None, True, None],  # Nested config
        "loadbalancer.memory_poll_interval": [int, 1, 60, True, None],  # 1-60 seconds poll interval
        "loadbalancer.memory_peak_window": [int, 10, 300, True, None],  # 10-300 seconds peak window
        "loadbalancer.main_process_memory_reserve_gb": [
            float,
            0.5,
            10.0,
            True,
            None,
        ],  # 0.5-10GB main process reserve
        "loadbalancer.memory_limit_gb": [
            float,
            0.1,
            1024.0,
            True,
            None,
        ],  # Optional, memory limit set by LoadBalancer
        "loadbalancer.memory_limit_bytes": [
            int,
            None,
            None,
            True,
            None,
        ],  # Optional, memory limit in bytes
        "loadbalancer.max_sources_per_process": [
            int,
            1,
            None,
            True,
            None,
        ],  # Optional, max sources per process
        "loadbalancer.resource_source": [
            str,
            None,
            None,
            True,
            None,
        ],  # Optional, resource detection source
        "loadbalancer.cpu_count": [int, 1, None, True, None],  # Optional, detected CPU count
        "loadbalancer.memory_available_gb": [
            float,
            0.0,
            None,
            True,
            None,
        ],  # Optional, available memory
        "loadbalancer.memory_total_gb": [
            float,
            0.0,
            None,
            True,
            None,
        ],  # Optional, total system memory
        "loadbalancer.safety_margin": [float, 0.0, 1.0, True, None],  # Optional, safety margin
        "loadbalancer.initial_workers": [int, 1, 8, True, None],  # 1-8 initial workers
        "loadbalancer.log_interval": [int, 5, 300, True, None],  # 5-300 seconds log interval
        "loadbalancer.event_log_file": [
            str,
            None,
            None,
            True,
            None,
        ],
        # === UI Configuration ===
        "ui": [DotMap, None, None, True, None],  # Nested config
        "ui.preview_samples": [int, 1, 50, False, None],  # 1-50 preview samples
        "ui.preview_size": [int, 16, 512, False, None],  # 16-512 pixel preview size
        "ui.auto_regenerate_preview": [bool, None, None, False, None],
    }

    return config_spec


def _get_nested_value(cfg: DotMap, key: str):
    """Get a nested value from the config using dot notation.

    Args:
        cfg (DotMap): Configuration object
        key (str): Key in dot notation (e.g., 'ui.preview_samples')

    Returns:
        Any: Value from the config
    """
    current = cfg
    for part in key.split("."):
        try:
            current = current[part]
        except (KeyError, TypeError):
            raise ValueError(f"Missing key in config: {key}")
    return current


def _get_all_keys(cfg: DotMap, parent_key: str = ""):
    """Get all keys in the config using dot notation.

    Args:
        cfg (DotMap): Configuration object
        parent_key (str): Parent key for nested values

    Returns:
        Set[str]: Set of all keys in dot notation
    """
    keys = set()
    for key, value in cfg.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        keys.add(current_key)
        if isinstance(value, DotMap):
            keys.update(_get_all_keys(value, current_key))
    return keys


def validate_config(cfg: DotMap, check_paths: bool = True) -> None:
    """Validate configuration against Cutana requirements.

    Args:
        cfg (DotMap): Configuration to validate
        check_paths (bool): Whether to check if file and directory paths exist

    Raises:
        ValueError: If configuration is invalid
    """
    # Get configuration specification
    config_spec = _return_required_and_optional_keys()

    # Keep track of checked keys
    expected_keys = set()

    # Validate each parameter
    for param_name, (dtype, min_val, max_val, optional, allowed_values) in config_spec.items():
        expected_keys.add(param_name)

        # Try to get the value, handle missing optional parameters
        try:
            value = _get_nested_value(cfg, param_name)
        except ValueError:
            if optional:
                continue  # Skip missing optional parameters
            else:
                raise ValueError(
                    f"Missing required parameter: {param_name} "
                    f"(type: {dtype.__name__ if hasattr(dtype, '__name__') else dtype})"
                )

        # Skip validation for None values on optional parameters
        if value is None and optional:
            continue

        # Helper function to format constraint info
        def _format_constraints():
            constraints = []
            if min_val is not None:
                constraints.append(f"min: {min_val}")
            if max_val is not None:
                constraints.append(f"max: {max_val}")
            if allowed_values is not None:
                constraints.append(f"allowed: {allowed_values}")
            return f" ({', '.join(constraints)})" if constraints else ""

        # Validate based on data type
        if dtype == str:
            if not isinstance(value, str):
                raise ValueError(
                    f"{param_name} must be a string, got {type(value).__name__}{_format_constraints()}"
                )
            # Check allowed values for string types
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(f"{param_name} must be one of {allowed_values}, got '{value}'")

        elif dtype == "directory_or_create":
            if not isinstance(value, str):
                raise ValueError(
                    f"{param_name} must be a string/directory path, got {type(value).__name__}"
                )
            # For output directories, we allow creation, so don't require existence

        elif dtype == "file":
            if not isinstance(value, str):
                raise ValueError(
                    f"{param_name} must be a string/file path, got {type(value).__name__}"
                )
            if check_paths and not os.path.isfile(value):
                raise ValueError(f"{param_name} file does not exist: {value}")

        elif dtype == int:
            if not isinstance(value, int):
                raise ValueError(
                    f"{param_name} must be an integer, got {type(value).__name__}{_format_constraints()}"
                )
            if min_val is not None and value < min_val:
                raise ValueError(
                    f"{param_name} must be >= {min_val}, got {value}{_format_constraints()}"
                )
            if max_val is not None and value > max_val:
                raise ValueError(
                    f"{param_name} must be <= {max_val}, got {value}{_format_constraints()}"
                )
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(f"{param_name} must be one of {allowed_values}, got {value}")

        elif dtype == float:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{param_name} must be a number, got {type(value).__name__}{_format_constraints()}"
                )
            if min_val is not None and value < min_val:
                raise ValueError(
                    f"{param_name} must be >= {min_val}, got {value}{_format_constraints()}"
                )
            if max_val is not None and value > max_val:
                raise ValueError(
                    f"{param_name} must be <= {max_val}, got {value}{_format_constraints()}"
                )

        elif dtype == bool:
            if not isinstance(value, bool):
                raise ValueError(f"{param_name} must be a boolean, got {type(value).__name__}")

        elif dtype == list:
            if not isinstance(value, list):
                raise ValueError(f"{param_name} must be a list, got {type(value).__name__}")

        elif dtype == DotMap:
            if not isinstance(value, DotMap):
                raise ValueError(f"{param_name} must be a DotMap, got {type(value).__name__}")

        # Handle special validation cases
        elif dtype == "special_fits_extensions":
            if not isinstance(value, list):
                raise ValueError(f"{param_name} must be a list, got {type(value).__name__}")
            for ext in value:
                if not isinstance(ext, (str, int)):
                    raise ValueError(
                        f"{param_name} list elements must be str or int, got {type(ext).__name__}"
                    )

        elif dtype == "special_selected_extensions":
            if value is not None:
                if not isinstance(value, list):
                    raise ValueError(f"{param_name} must be a list, got {type(value).__name__}")
                for ext in value:
                    if isinstance(ext, dict):
                        # UI format: {"name": "VIS", "ext": "IMAGE"}
                        if "name" not in ext:
                            raise ValueError(f"{param_name} dict elements must have 'name' key")
                    elif not isinstance(ext, (str, int)):
                        raise ValueError(
                            f"{param_name} list elements must be str, int, or dict, got {type(ext).__name__}"
                        )

        elif dtype == "special_flux_keywords":
            if value is not None:
                if not isinstance(value, (dict, DotMap)):
                    raise ValueError(
                        f"{param_name} must be a dict or DotMap, got {type(value).__name__}"
                    )

        elif dtype == "special_function":
            if value is not None and not callable(value):
                raise ValueError(
                    f"{param_name} must be callable or None, got {type(value).__name__}"
                )

        elif dtype == "special_channel_weights_dict":
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"{param_name} must be a dict, got {type(value).__name__}")
                for channel_name, weights in value.items():
                    if not isinstance(channel_name, str):
                        raise ValueError(f"{param_name} channel names must be strings")
                    if not isinstance(weights, list):
                        raise ValueError(f"{param_name} weights must be lists")
                    if not all(isinstance(w, (int, float)) for w in weights):
                        raise ValueError(f"{param_name} weights must contain only numbers")

        else:
            raise ValueError(f"Unknown data type for {param_name}: {dtype}")

    # Custom cross-parameter validation
    _validate_flux_conversion_config(cfg)

    # Check for unexpected keys (warn only)
    actual_keys = _get_all_keys(cfg)
    unexpected_keys = actual_keys - expected_keys

    if unexpected_keys:
        logger.warning(f"Found unexpected keys in config: {sorted(unexpected_keys)}")
        logger.info("Config: validation partially successful")
    else:
        logger.info("Config: validation successful")


def _validate_flux_conversion_config(config):
    """
    Validate flux conversion configuration.

    Args:
        config: The configuration object (DotMap).

    Raises:
        ValueError: If the configuration is invalid.
    """
    if config.user_flux_conversion_function:
        # test that function takes an image & header, returns an image
        if not callable(config.user_flux_conversion_function):
            raise ValueError("user_flux_conversion_function must be callable")
        # test that it takes 2 inputs
        signature = inspect.signature(config.user_flux_conversion_function)
        if len(signature.parameters) != 2:
            raise ValueError("user_flux_conversion_function must take exactly 2 arguments")


def validate_config_for_processing(cfg: DotMap, check_paths: bool = True):
    """Additional validation specifically for processing workflows.

    Args:
        cfg (DotMap): Configuration to validate for processing
        check_paths (bool): Whether to check if file paths exist (default: True)

    Raises:
        ValueError: If configuration is not ready for processing
    """
    # Required for processing
    required_for_processing = ["source_catalogue", "output_dir", "selected_extensions"]

    for param in required_for_processing:
        try:
            value = _get_nested_value(cfg, param)
            if not value:  # Check for None, empty list, empty string
                raise ValueError(
                    f"Parameter {param} is required for processing but is empty or None"
                )
        except ValueError:
            raise ValueError(f"Parameter {param} is required for processing but is missing")

    # Validate selected extensions are not empty
    if not cfg.selected_extensions or len(cfg.selected_extensions) == 0:
        raise ValueError("At least one FITS extension must be selected for processing")

    # Validate source catalogue exists (only if check_paths is True)
    if check_paths and not os.path.isfile(cfg.source_catalogue):
        raise ValueError(f"Source catalogue file does not exist: {cfg.source_catalogue}")

    logger.info("Config: processing validation successful")


def validate_channel_order_consistency(tensor_channel_names, channel_weights, weak_check=True):
    """
    Validate that channel order in data tensor matches channel_weights order.

    This critical validation prevents silent data corruption where channel weights
    are applied to wrong channels due to non-deterministic set() ordering in
    resize_batch_tensor.

    Args:
        tensor_channel_names (list): Channel names in the order they appear in the tensor
        channel_weights (dict): Dictionary mapping channel names to weight arrays
        weak_check (bool): If True, use substring matching for channel names (default: True)

    Raises:
        AssertionError: If channel order doesn't match exactly
    """
    # Get channel names from weights (in the order they will be applied)
    config_channel_names = list(channel_weights.keys())

    if weak_check:
        # Weak check: use substring matching for cases where tensor names contain full paths
        # but config names are just extension identifiers
        def find_matching_config_channel(tensor_name):
            """Find config channel name that appears as substring in tensor name."""
            for config_name in config_channel_names:
                if config_name in tensor_name:
                    return config_name
            return None

        # Map tensor channels to their corresponding config channels
        mapped_channels = []
        for tensor_name in tensor_channel_names:
            matching_config = find_matching_config_channel(tensor_name)
            # This allows to skip unmatched channels
            if matching_config is None:
                logger.warning(
                    f"Tensor channel '{tensor_name}' does not match any config channel name. "
                    f"Config channels: {config_channel_names}"
                )
            else:
                mapped_channels.append(matching_config)
        assert len(mapped_channels) != 0, (
            f"Tensor channel '{tensor_channel_names}' do not contain any config channel name. "
            f"Config channels: {config_channel_names}"
        )
        # Verify all config channels are represented
        mapped_set = set(mapped_channels)
        config_set = set(config_channel_names)
        assert mapped_set == config_set, (
            f"Channel mapping incomplete. "
            f"Mapped channels: {mapped_set}, "
            f"Config channels: {config_set}. "
            f"Missing: {config_set - mapped_set}, "
            f"Extra: {mapped_set - config_set}"
        )

        # Check order consistency (mapped channels should match config order)
        assert mapped_channels == config_channel_names, (
            f"Channel order mismatch! Data tensor maps to channels in order: {mapped_channels}, "
            f"but channel_weights expects: {config_channel_names}. "
            f"Tensor channels: {tensor_channel_names}"
        )

    else:
        # Original strict check: exact string matching
        tensor_channels_set = set(tensor_channel_names)
        config_channels_set = set(config_channel_names)

        assert tensor_channels_set == config_channels_set, (
            f"Channel mismatch between data and configuration. "
            f"Data channels: {tensor_channels_set}, "
            f"Config channels: {config_channels_set}. "
            f"Missing from config: {tensor_channels_set - config_channels_set}, "
            f"Extra in config: {config_channels_set - tensor_channels_set}"
        )

        # Critical validation: channel order must match exactly
        # This prevents silent data corruption where weights are applied to wrong channels
        assert tensor_channel_names == config_channel_names, (
            f"Channel order mismatch! Data tensor has channels in order: {tensor_channel_names}, "
            f"but channel_weights expects: {config_channel_names}. "
            f"The extension order in the data tensor is determined by the non-deterministic set() "
            f"behavior in resize_batch_tensor. To fix this, ensure your channel_weights keys "
            f"are ordered to match the actual data processing order, or implement deterministic "
            f"channel ordering in the data processing pipeline."
        )
