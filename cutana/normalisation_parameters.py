#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Normalisation parameters for Cutana astronomical image processing.

This module defines all default values, ranges, and validation constants for
image normalization/stretching methods. These parameters can be easily updated
by users to customize the normalization behavior.

The parameters are organized by normalization method and include:
- Default values for each parameter
- Valid ranges (min/max values)
- Step sizes for UI controls
- Method-specific defaults for unified parameters
"""

from typing import Dict, Any, Tuple
from dotmap import DotMap
import fitsbolt
from loguru import logger


# =============================================================================
# GENERAL NORMALIZATION PARAMETERS
# =============================================================================


class NormalisationDefaults:
    """Default values for normalization parameters."""

    PERCENTILE = 99.8  # Percentile cutting applied to all methods. Default value from Euclid bulk cutouts documentation.
    N_SAMPLES = 1000  # Default for ZScale n_samples. Values are Astropy default valus.
    CONTRAST = 0.25

    # Method-specific defaults for unified 'a' parameter
    ASINH_A = 0.1  # Default ASINH transition parameter. Default value from Astropy documentation.
    LOG_A = 1000.0  # Default LOG scale factor. Default value from Astropy Documentation.

    # Crop parameters for norm_crop_for_maximum_value
    CROP_ENABLE = False  # Whether to enable crop for maximum value computation
    CROP_HEIGHT = 64  # Default crop height (must be smaller than image resolution)
    CROP_WIDTH = 64  # Default crop width (must be smaller than image resolution)

    # Interpolation method for image resizing
    INTERPOLATION = "bilinear"  # Default interpolation method


class NormalisationRanges:
    """Valid ranges for normalization parameters."""

    # Percentile clipping (applied to all methods)
    PERCENTILE_MIN = 0.0
    PERCENTILE_MAX = 100.0

    # ASINH transition parameter
    ASINH_A_MIN = 0.001
    ASINH_A_MAX = 3.0  # ASINH 'a' max value chosen as function becomes linear beyond this point.

    # LOG scale factor
    LOG_A_MIN = 0.01
    LOG_A_MAX = 10000.0  # LOG 'a' max value chosen as function becomes linear beyond this point.

    # ZScale parameters
    N_SAMPLES_MIN = 100
    N_SAMPLES_MAX = 10000

    CONTRAST_MIN = 0.01
    CONTRAST_MAX = 1.0

    # Crop parameters (must be smaller than image resolution)
    CROP_HEIGHT_MIN = 8
    CROP_HEIGHT_MAX = 1024  # Should be smaller than typical image resolutions
    CROP_WIDTH_MIN = 8
    CROP_WIDTH_MAX = 1024  # Should be smaller than typical image resolutions


class NormalisationSteps:
    """Step sizes for UI controls."""

    PERCENTILE_STEP = 0.1
    ASINH_A_STEP = 0.001
    LOG_A_STEP = 0.01
    N_SAMPLES_STEP = 100
    CONTRAST_STEP = 0.05

    # Crop parameters
    CROP_HEIGHT_STEP = 1
    CROP_WIDTH_STEP = 1


# =============================================================================
# METHOD-SPECIFIC CONFIGURATIONS
# =============================================================================


def get_method_specific_a_default(method: str) -> float:
    """Get the default 'a' parameter value for a specific normalization method.

    Args:
        method: Normalization method ('asinh', 'log', 'linear', 'zscale')

    Returns:
        Default 'a' parameter value for the method
    """
    method_lower = method.lower()
    if method_lower == "asinh":
        return NormalisationDefaults.ASINH_A
    elif method_lower == "log":
        return NormalisationDefaults.LOG_A
    else:
        # For methods that don't use 'a' parameter, return ASINH default
        return NormalisationDefaults.ASINH_A


def get_method_specific_a_range(method: str) -> Tuple[float, float]:
    """Get the valid range for 'a' parameter for a specific normalization method.

    Args:
        method: Normalization method ('asinh', 'log', 'linear', 'zscale')

    Returns:
        Tuple of (min_value, max_value) for the method
    """
    method_lower = method.lower()
    if method_lower == "asinh":
        return (NormalisationRanges.ASINH_A_MIN, NormalisationRanges.ASINH_A_MAX)
    elif method_lower == "log":
        return (NormalisationRanges.LOG_A_MIN, NormalisationRanges.LOG_A_MAX)
    else:
        # For methods that don't use 'a' parameter, return ASINH range
        return (NormalisationRanges.ASINH_A_MIN, NormalisationRanges.ASINH_A_MAX)


def get_method_specific_a_step(method: str) -> float:
    """Get the step size for 'a' parameter for a specific normalization method.

    Args:
        method: Normalization method ('asinh', 'log', 'linear', 'zscale')

    Returns:
        Step size for the method
    """
    method_lower = method.lower()
    if method_lower == "asinh":
        return NormalisationSteps.ASINH_A_STEP
    elif method_lower == "log":
        return NormalisationSteps.LOG_A_STEP
    else:
        # For methods that don't use 'a' parameter, return ASINH step
        return NormalisationSteps.ASINH_A_STEP


# =============================================================================
# CONFIGURATION BUILDERS
# =============================================================================


def get_default_normalisation_config() -> DotMap:
    """Get default normalization configuration as DotMap.

    Returns:
        DotMap with default normalization parameters
    """
    return DotMap(
        {
            "percentile": NormalisationDefaults.PERCENTILE,
            "a": NormalisationDefaults.ASINH_A,  # Default to ASINH
            "n_samples": NormalisationDefaults.N_SAMPLES,
            "contrast": NormalisationDefaults.CONTRAST,
            "crop_enable": NormalisationDefaults.CROP_ENABLE,
            "crop_height": NormalisationDefaults.CROP_HEIGHT,
            "crop_width": NormalisationDefaults.CROP_WIDTH,
        },
        _dynamic=False,
    )


def get_method_tooltip(method: str) -> str:
    """Get tooltip text for 'a' parameter based on normalization method.

    Args:
        method: Normalization method

    Returns:
        Tooltip text string
    """
    method_lower = method.lower()
    if method_lower == "asinh":
        min_val, max_val = get_method_specific_a_range(method)
        return f"ASINH transition parameter ({min_val}-{max_val})"
    elif method_lower == "log":
        min_val, max_val = get_method_specific_a_range(method)
        return f"Log scale factor ({min_val}-{max_val})"
    else:
        return "Transition parameter (ASINH: 0.001-3.0, LOG: 0.01-10000)"


# =============================================================================
# FITSBOLT CONVERSION
# =============================================================================


def convert_cfg_to_fitsbolt_cfg(config: DotMap, num_channels: int = 1) -> Dict[str, Any]:
    """Convert Cutana configuration to fitsbolt parameters dictionary.

    This function consolidates all the parameter extraction logic from the config
    and builds the fitsbolt_params dictionary based on the normalisation method.
    The images array should be added separately to avoid unnecessary data copying.

    Args:
        config: Configuration DotMap containing all normalisation parameters
        num_channels: Number of channels in the image array (for asinh parameters)

    Returns:
        Dictionary containing fitsbolt parameters ready for normalise_images()
        (Note: 'images' key should be added separately by the caller)
    """
    # Extract method from config
    method = config.normalisation_method.lower()

    # Convert method string to fitsbolt enum
    if method == "log":
        norm_method = fitsbolt.NormalisationMethod.LOG
    elif method == "linear":
        norm_method = fitsbolt.NormalisationMethod.CONVERSION_ONLY
    elif method == "asinh":
        norm_method = fitsbolt.NormalisationMethod.ASINH
    elif method == "zscale":
        norm_method = fitsbolt.NormalisationMethod.ZSCALE
    else:
        norm_method = fitsbolt.NormalisationMethod.CONVERSION_ONLY

    # Build base fitsbolt parameters (without images array)
    fitsbolt_params = {
        "normalisation_method": norm_method,
        "show_progress": False,
    }

    # Extract general parameters from config
    percentile = config.normalisation.percentile
    a = config.normalisation.a

    # Add crop parameters if enabled
    try:
        if config.normalisation.crop_enable:
            crop_height = config.normalisation.crop_height
            crop_width = config.normalisation.crop_width
            fitsbolt_params["norm_crop_for_maximum_value"] = (crop_height, crop_width)
            logger.debug(f"Crop for maximum value enabled: {crop_height}x{crop_width}")
    except Exception as e:
        logger.debug(f"Crop for maximum value not enabled or invalid parameters: {e}")
    # Add method-specific parameters and debug logging based on normalisation method
    if method == "log":
        fitsbolt_params["norm_log_scale_a"] = a
        logger.debug(f"Log normalisation parameter: a={a}")
        logger.debug(f"Log normalization: percentile={percentile}")
    elif method == "zscale":
        fitsbolt_params["norm_zscale_n_samples"] = config.normalisation.n_samples
        fitsbolt_params["norm_zscale_contrast"] = config.normalisation.contrast
        logger.debug(
            f"ZScale normalization: n_samples={config.normalisation.n_samples}, contrast={config.normalisation.contrast}"
        )
        logger.debug(f"ZScale normalization: percentile={percentile}")
    elif method == "asinh":
        # Use provided number of channels for asinh parameters
        norm_scale = [a] * num_channels  # Scale 'a' for each channel
        norm_clip = [percentile] * num_channels  # Use general percentile for all channels

        fitsbolt_params["norm_asinh_scale"] = norm_scale
        fitsbolt_params["norm_asinh_clip"] = norm_clip
        logger.debug(
            f"ASINH normalization: a={a}, percentile={percentile}, channels={num_channels}"
        )

    return fitsbolt_params
