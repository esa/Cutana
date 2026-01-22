#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Image processor module for Cutana - handles image processing and normalization.

This module provides static functions for:
- Image resizing to target resolution
- Data type conversion using skimage
- Normalization using fitsbolt (stretch and normalization are the same)
- Multi-channel image processing
"""

from typing import Dict, List, Tuple

import drizzle
import fitsbolt
import numpy as np
from astropy.wcs import WCS
from dotmap import DotMap
from loguru import logger
from skimage import transform, util

from .normalisation_parameters import (
    build_fitsbolt_params_from_external_cfg,
    convert_cfg_to_fitsbolt_cfg,
)


class PixmapCache:
    """Context-local cache for drizzle pixmap computation.

    Avoids recomputing pixmaps when WCS parameters are identical across
    consecutive resize operations, which is common in batch processing.
    """

    def __init__(self):
        self.last_source_shape = None
        self.last_source_pxscale = None
        self.last_target_resolution = None
        self.last_target_pxscale = None
        self.cached_pixmap = None

    def get(self, source_shape, source_pxscale, target_resolution, target_pxscale):
        """Get cached pixmap if parameters match, otherwise return None."""
        if (
            self.last_source_shape == source_shape
            and self.last_source_pxscale == source_pxscale
            and self.last_target_resolution == target_resolution
            and self.last_target_pxscale == target_pxscale
            and self.cached_pixmap is not None
        ):
            return self.cached_pixmap
        return None

    def set(self, source_shape, source_pxscale, target_resolution, target_pxscale, pixmap):
        """Store pixmap and its associated parameters in cache."""
        self.last_source_shape = source_shape
        self.last_source_pxscale = source_pxscale
        self.last_target_resolution = target_resolution
        self.last_target_pxscale = target_pxscale
        self.cached_pixmap = pixmap

    def clear(self):
        """Clear all cached data."""
        self.last_source_shape = None
        self.last_source_pxscale = None
        self.last_target_resolution = None
        self.last_target_pxscale = None
        self.cached_pixmap = None


def resize_batch_tensor(
    source_cutouts: Dict[str, Dict[str, np.ndarray]],
    target_resolution: Tuple[int, int],
    interpolation: str,
    flux_conserved_resizing: bool,
    pixel_scales_dict: Dict[str, float],
) -> np.ndarray:
    """
    Resize all source cutouts and return as (N_sources, H, W, N_extensions) tensor.

    Args:
        source_cutouts: Dict mapping source_id -> {channel_key: cutout}
        target_resolution: Target (height, width)
        interpolation: Interpolation method
        flux_conserved_resizing: Whether to use flux-conserved resizing (activates drizzle)
        pixel_scales_dict: Dict mapping channel_key to pixel scale in arcsec/pixel

    Returns:
        Tensor of shape (N_sources, H, W, N_extensions)
    """
    source_ids = list(source_cutouts.keys())
    N_sources = len(source_ids)

    # Get all unique extension names in order of first appearance (deterministic)
    extension_names = []
    for source_cutouts_dict in source_cutouts.values():
        for ext_name in source_cutouts_dict.keys():
            if ext_name not in extension_names:
                extension_names.append(ext_name)
    N_extensions = len(extension_names)

    H, W = target_resolution

    # Pre-allocate tensor
    batch_tensor = np.zeros((N_sources, H, W, N_extensions), dtype=np.float32)

    # Map interpolation methods
    if interpolation == "nearest":
        order = 0
    elif interpolation == "bilinear":
        order = 1
    elif interpolation == "biquadratic":
        order = 2
    elif interpolation == "bicubic":
        order = 3
    else:
        order = 1  # default to bilinear

    # Create pixmap cache for this batch if using flux-conserved resizing
    pixmap_cache = PixmapCache() if flux_conserved_resizing else None

    # Fill tensor
    for i, source_id in enumerate(source_ids):
        source_cutouts_dict = source_cutouts[source_id]
        for j, ext_name in enumerate(extension_names):
            if ext_name in source_cutouts_dict:
                cutout = source_cutouts_dict[ext_name]
                if cutout is not None and cutout.size > 0:
                    # Resize if needed
                    if cutout.shape != target_resolution:
                        try:
                            if flux_conserved_resizing:
                                resized = resize_flux_conserved(
                                    cutout,
                                    target_resolution,
                                    pixel_scales_dict[ext_name],
                                    pixmap_cache,
                                )
                            else:
                                resized = transform.resize(
                                    cutout,
                                    target_resolution,
                                    order=order,
                                    mode="symmetric",
                                    preserve_range=True,
                                    anti_aliasing=True,
                                ).astype(cutout.dtype)

                        except Exception as e:
                            logger.error(f"Image resizing failed: {e}")
                            # Fallback: return zeros of target size
                            resized = np.zeros(target_resolution, dtype=cutout.dtype)
                    else:
                        resized = cutout.copy()
                    batch_tensor[i, :, :, j] = resized

    # Cleanup: clear cache after batch processing is complete
    if pixmap_cache is not None:
        pixmap_cache.clear()
    del pixmap_cache
    return batch_tensor


def resize_flux_conserved(
    cutout, target_resolution, pixel_scale_arcsecppix, pixmap_cache: PixmapCache = None
) -> np.ndarray:
    """Resize image cutout to target resolution using flux-conserved drizzle algorithm.

    Uses optional caching to avoid recomputing pixmap when WCS parameters are identical
    to the previous call, which is common in batch processing.

    Args:
        cutout (np.ndarray): Input image cutout
        target_resolution (Tuple[int, int]): Target (height, width) resolution
        pixel_scale_arcsecppix (float): Pixel scale in arcseconds per pixel
        pixmap_cache (PixmapCache, optional): Cache instance for pixmap reuse

    Returns:
        np.ndarray: Resized image cutout
    """
    source_wcs_shape = cutout.shape
    source_wcs_pxscale = pixel_scale_arcsecppix / 3600  # convert to degrees/pixel

    # Calculate target pixel scale
    target_pxscale = source_wcs_pxscale * (source_wcs_shape[0] / target_resolution[0])

    # Try to get cached pixmap if cache is provided
    pixmap = None
    if pixmap_cache is not None:
        pixmap = pixmap_cache.get(
            source_wcs_shape, source_wcs_pxscale, target_resolution, target_pxscale
        )

    if pixmap is None:
        # Compute new pixmap
        source_wcs = WCS(naxis=2)
        source_wcs.array_shape = source_wcs_shape
        source_wcs.wcs.crpix = [source_wcs_shape[1] / 2, source_wcs_shape[0] / 2]
        source_wcs.wcs.cdelt = [source_wcs_pxscale, source_wcs_pxscale]
        source_wcs.wcs.crval = [0, 0]

        target_output_wcs = WCS(naxis=2)
        target_output_wcs.wcs.crpix = [target_resolution[1] / 2, target_resolution[0] / 2]
        target_output_wcs.wcs.cdelt = [target_pxscale, target_pxscale]

        pixmap = drizzle.utils.calc_pixmap(source_wcs, target_output_wcs)

        # Store in cache if provided
        if pixmap_cache is not None:
            pixmap_cache.set(
                source_wcs_shape, source_wcs_pxscale, target_resolution, target_pxscale, pixmap
            )

    # Apply drizzle with pixmap
    driz = drizzle.resample.Drizzle(
        out_shape=(
            target_resolution[0],
            target_resolution[1],
        )
    )
    driz.add_image(cutout, exptime=1, pixmap=pixmap, pixfrac=1.0, weight_map=None)
    resized_image = driz.out_img * driz.out_wht
    del driz
    return resized_image


def convert_data_type(images: np.ndarray, target_dtype: str) -> np.ndarray:
    """
    Convert images to target data type using skimage utilities.
    Handles both single images and batches of images.

    Args:
        images: Input images in shape (H, W), (N, H, W) or (N, H, W, C)
        target_dtype: Target data type ('float32', 'float64', 'uint8', 'uint16', 'int16')

    Returns:
        Images with target data type (same shape as input)
    """
    try:
        if target_dtype == "float32":
            return util.img_as_float32(images)
        elif target_dtype == "float64":
            return util.img_as_float64(images)
        elif target_dtype == "uint8":
            return util.img_as_ubyte(images)
        elif target_dtype == "uint16":
            return util.img_as_uint(images)
        elif target_dtype == "int16":
            # Convert to int16 manually since skimage doesn't have img_as_int16
            # First normalize to [0, 1] range, then scale to int16 range
            int16_info = np.iinfo(np.int16)
            normalized = util.img_as_float64(images)
            scale = int16_info.max - int16_info.min
            return ((normalized * scale) + int16_info.min).astype(np.int16)
        else:
            logger.warning(f"Unknown data type {target_dtype}, keeping original")
            return images

    except Exception as e:
        logger.error(f"Data type conversion failed: {e}")
        return images


def apply_normalisation(images: np.ndarray, config: DotMap) -> np.ndarray:
    """
    Apply normalization/stretch to a batch of images using fitsbolt batch processing.

    Args:
        images: Batch of images in format (N, H, W) or (N, H, W, C)
        config: Configuration DotMap containing all normalization parameters.
                If config.external_fitsbolt_cfg is set, uses that directly for
                normalization (for ML pipeline integration with AnomalyMatch).

    Returns:
        Batch of normalized/stretched image arrays
    """
    # Prepare images for fitsbolt batch processing
    if len(images.shape) == 3:
        # N,H,W -> N,H,W,1 for fitsbolt
        images_array = images[:, :, :, np.newaxis]
    else:
        # Already in N,H,W,C format
        images_array = images

    num_channels = images_array.shape[-1]

    # Check for external fitsbolt config (from AnomalyMatch or other ML pipelines)
    # A valid external config must have 'normalisation_method' key
    external_cfg = config.external_fitsbolt_cfg
    if external_cfg is not None and "normalisation_method" in external_cfg:
        # Sync cutana config's crop settings from external fitsbolt config
        crop_value = getattr(external_cfg.normalisation, "crop_for_maximum_value", None)
        if crop_value is not None:
            config.normalisation.crop_enable = True
            config.normalisation.crop_height = crop_value[0]
            config.normalisation.crop_width = crop_value[1]
            logger.debug(f"Synced crop settings from external config: {crop_value}")
        else:
            config.normalisation.crop_enable = False

        fitsbolt_params = build_fitsbolt_params_from_external_cfg(external_cfg, num_channels)
        logger.debug("Using external fitsbolt config for normalization")
    else:
        # Use cutana's own config converted to fitsbolt parameters
        fitsbolt_params = convert_cfg_to_fitsbolt_cfg(config, num_channels)

    # Add images array to parameters (done here to avoid unnecessary copying)
    fitsbolt_params["images"] = images_array

    try:
        # Apply fitsbolt batch normalization with parameters
        normalized_images = fitsbolt.normalise_images(**fitsbolt_params)

        # Return in original shape format
        if len(images.shape) == 3:
            return normalized_images[:, :, :, 0]  # Remove channel dimension
        else:
            return normalized_images

    except Exception as e:
        logger.error(f"Fitsbolt batch normalization failed: {e}, using fallback")
        # Fallback batch normalization
        normalized_batch = []
        for img in images:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                normalized = (img - img_min) / (img_max - img_min)
            else:
                normalized = np.zeros_like(img)
            normalized_batch.append(normalized)
        return np.array(normalized_batch)


def combine_channels(
    batch_cutouts: np.ndarray, channel_weights: Dict[str, List[float]]
) -> np.ndarray:
    """
    Combine multiple channels using fitsbolt batch channel combination.

    Args:
        batch_cutouts: Batch of cutouts with shape (N_sources, H, W, N_extensions)
        channel_weights: Dictionary mapping channel names to output weight arrays
                        e.g., {"VIS": [1.0, 0.0, 0.75], "NIR-H": [0.0, 1.0, 0.75]}
                        Number of output channels determined by weight array length

    Returns:
        Combined images with shape (N_sources, H, W, N_output_channels)
        N_output_channels determined by length of weight arrays in channel_weights
    """
    # Input validation assertions
    assert isinstance(batch_cutouts, np.ndarray), "batch_cutouts must be numpy array"
    assert isinstance(channel_weights, dict), "channel_weights must be a dictionary"
    assert len(batch_cutouts.shape) == 4, "batch_cutouts must have 4 dimensions (N,H,W,C)"
    assert len(channel_weights) > 0, "channel_weights dictionary cannot be empty"

    # Validate channel_weights format
    weight_lengths = set()
    for channel, weights in channel_weights.items():
        assert isinstance(channel, str), f"Channel key {channel} must be a string"
        assert isinstance(weights, list), f"Weights for {channel} must be a list"
        assert len(weights) > 0, f"Weights for {channel} cannot be empty"
        assert all(
            isinstance(w, (int, float)) for w in weights
        ), f"All weights for {channel} must be numeric"
        weight_lengths.add(len(weights))

    # All weight arrays must have the same length (same number of output channels)
    assert (
        len(weight_lengths) == 1
    ), f"All weight arrays must have the same length, got: {weight_lengths}"

    # Convert channel_weights dict to numpy array for fitsbolt
    channel_names = list(channel_weights.keys())
    N_sources, H, W, N_extensions = batch_cutouts.shape

    # Determine number of output channels from weight array length
    first_weights = next(iter(channel_weights.values()))
    n_output_channels = len(first_weights)

    # Build channel combination matrix (n_output_channels, n_extensions)
    channel_combination = np.zeros((n_output_channels, N_extensions), dtype=np.float32)

    for ext_idx, channel_name in enumerate(channel_names):
        if ext_idx < N_extensions:  # Ensure we don't exceed available extensions
            weights = channel_weights[channel_name]
            for output_idx in range(n_output_channels):
                channel_combination[output_idx, ext_idx] = weights[output_idx]

    # Apply fitsbolt batch channel combination
    combined_batch = fitsbolt.channel_mixing.batch_channel_combination(
        images=batch_cutouts,
        channel_combination=channel_combination,
    )

    return combined_batch
