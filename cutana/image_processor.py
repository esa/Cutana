#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Image processor module for Cutana - handles image processing and normalization.

This module provides static functions for:
- Image resizing to target resolution (OpenCV, or drizzle for flux-conserved)
- Normalization using fitsbolt (stretch and normalization are the same)
- Multi-channel image processing
"""

from typing import Dict, List, Optional, Tuple

import cv2
import drizzle
import fitsbolt
import numpy as np
from astropy.wcs import WCS
from dotmap import DotMap
from loguru import logger

from .normalisation_parameters import (
    build_fitsbolt_params_from_external_cfg,
    convert_cfg_to_fitsbolt_cfg,
)
from .validate_config import validate_channel_order_consistency

# Maps the `interpolation` config value to the cv2 upscaling kernel flag.
# Downscaling always uses cv2.INTER_AREA regardless of this setting.
_CV2_UPSCALE_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


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
    N_sources = len(source_cutouts)

    # Every source carries the same extensions in the same order: each source's
    # channel dict is filled in a single `for ext_name in fits_extensions` pass
    # (extract_cutouts_batch_vectorized), and out-of-bounds bands are zero-padded
    # rather than dropped, so the set is identical across sources. The first
    # source therefore defines the extension -> tensor-column mapping for the
    # whole batch. The inner loop below accesses each source by name (not by
    # position), so a source that ever broke this invariant fails hard with a
    # KeyError instead of silently shifting a band into the wrong column.
    extension_names = list(next(iter(source_cutouts.values()))) if source_cutouts else []
    N_extensions = len(extension_names)

    H, W = target_resolution

    # Pre-allocate tensor
    batch_tensor = np.zeros((N_sources, H, W, N_extensions), dtype=np.float32)

    # The `interpolation` config selects the upscaling kernel and acts as the
    # resize quality control, in increasing quality/cost order:
    # nearest < bilinear (default) < bicubic < lanczos. Euclid cutouts are
    # typically upsampled (small native size -> target resolution), so this is
    # the common path. For the rarer downscaling case we always use INTER_AREA
    # (area-averaging antialiasing, the correct choice for decimation).
    # Values are validated in validate_config.py, so a missing key is a real
    # bug, not user input -> fail hard rather than silently defaulting.
    upscale_interpolation = _CV2_UPSCALE_INTERPOLATION[interpolation]

    # Create pixmap cache for this batch if using flux-conserved resizing
    pixmap_cache = PixmapCache() if flux_conserved_resizing else None

    # Fill tensor. The outer loop is positional over sources (their iteration
    # order defines the tensor's N_sources axis); the inner loop is keyed by
    # extension name so each band always lands in its own fixed column.
    for i, source_cutouts_dict in enumerate(source_cutouts.values()):
        for j, ext_name in enumerate(extension_names):
            batch_tensor[i, :, :, j] = _resize_cutout(
                source_cutouts_dict[ext_name],
                target_resolution,
                upscale_interpolation,
                flux_conserved_resizing,
                pixel_scales_dict[ext_name],
                pixmap_cache,
            )

    # Cleanup: clear cache after batch processing is complete
    if pixmap_cache is not None:
        pixmap_cache.clear()
    del pixmap_cache
    return batch_tensor


def _resize_cutout(
    cutout: np.ndarray,
    target_resolution: Tuple[int, int],
    upscale_interpolation: int,
    flux_conserved_resizing: bool,
    pixel_scale: float,
    pixmap_cache: "PixmapCache",
) -> np.ndarray:
    """Resize a single cutout to ``target_resolution``.

    Returns the cutout unchanged (copied) when it already matches the target.
    On resize failure, logs the error and returns zeros of the target size.
    """
    if cutout.shape == target_resolution:
        return cutout.copy()

    try:
        if flux_conserved_resizing:
            return resize_flux_conserved(cutout, target_resolution, pixel_scale, pixmap_cache)

        # cv2.resize takes dsize as (width, height). Use INTER_AREA when
        # downscaling, the configured kernel when upscaling.
        downscaling = (
            target_resolution[0] < cutout.shape[0] or target_resolution[1] < cutout.shape[1]
        )
        interpolation_flag = cv2.INTER_AREA if downscaling else upscale_interpolation
        # cv2 reads raw bytes in native order, so a big-endian FITS cutout
        # (>f4, as astropy returns) would be mis-decoded into garbage. Force a
        # native-order float32 copy; lossless here because batch_tensor is
        # float32 anyway.
        return cv2.resize(
            np.ascontiguousarray(cutout, dtype=np.float32),
            (target_resolution[1], target_resolution[0]),
            interpolation=interpolation_flag,
        )
    except Exception as e:
        logger.error(f"Image resizing failed: {e}")
        # Fallback: return zeros of target size
        return np.zeros(target_resolution, dtype=cutout.dtype)


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
        # A copy: the crop settings below are derived from the external config and used only to
        # build this call's parameters. Writing them into the caller's DotMap changed it underneath
        # an interactive caller that reuses one config across many renders.
        config = DotMap(config.toDict(), _dynamic=False)
        # `crop_for_maximum_value` is an optional fitsbolt parameter on the
        # externally-provided config; its presence is checked explicitly
        # rather than via a getattr fallback.
        if "crop_for_maximum_value" in external_cfg.normalisation:
            crop_value = external_cfg.normalisation.crop_for_maximum_value
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
        normalized_images = np.asarray(fitsbolt.normalise_images(**fitsbolt_params))

        # Return in original shape format
        if len(images.shape) == 3:
            return normalized_images[:, :, :, 0]  # Remove channel dimension
        else:
            return normalized_images

    except Exception as e:
        raise RuntimeError(
            f"Fitsbolt normalisation failed: {e}. "
            f"This may indicate a mismatch between the external_fitsbolt_cfg "
            f"(e.g. per-channel ASINH params sized for {num_channels} channels) "
            f"and the actual image data shape {images_array.shape}. "
            f"Ensure that channel_weights expands the data to the expected "
            f"number of output channels before normalisation."
        ) from e


def combine_channels(
    batch_cutouts: np.ndarray,
    channel_weights: Dict[str, List[float]],
    channel_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Combine multiple channels using fitsbolt batch channel combination.

    Dictionary keys identify input channels independently of insertion order. Pass the tensor's
    channel_names (returned by create_cutouts_direct); ambiguous or missing mappings fail.

    Args:
        batch_cutouts: Batch of cutouts with shape (N_sources, H, W, N_extensions)
        channel_weights: Dictionary mapping channel names to output weight arrays
                        e.g., {"VIS": [1.0, 0.0, 0.75], "NIR-H": [0.0, 1.0, 0.75]}
                        Number of output channels determined by weight array length
        channel_names: Required tensor extension names, in tensor order.

    Returns:
        Combined images with shape (N_sources, H, W, N_output_channels)
        N_output_channels determined by length of weight arrays in channel_weights

    Raises:
        ValueError: If the number of weight entries does not match the number of extensions.
                    Dropping or zero-weighting the difference silently produces plausible pixels
                    that are wrong, which is worse than refusing.
        ValueError: If names are absent or do not resolve unambiguously to weight keys.
        AssertionError: If the tensor or weight arrays have an invalid format.
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
        assert all(isinstance(w, (int, float)) for w in weights), (
            f"All weights for {channel} must be numeric"
        )
        weight_lengths.add(len(weights))

    # All weight arrays must have the same length (same number of output channels)
    assert len(weight_lengths) == 1, (
        f"All weight arrays must have the same length, got: {weight_lengths}"
    )

    if channel_names is None:
        raise ValueError("channel_names is required to resolve channel_weights by name")
    weight_names = validate_channel_order_consistency(channel_names, channel_weights)

    # Convert channel_weights dict to numpy array for fitsbolt
    N_sources, H, W, N_extensions = batch_cutouts.shape

    # Caught here rather than skipped past: an extra weight was dropped and a missing one left an
    # extension at weight zero, both without a word.
    if len(weight_names) != N_extensions:
        raise ValueError(
            f"channel_weights has {len(weight_names)} entries ({weight_names}) but the tensor has "
            f"{N_extensions} extensions. Each tensor channel must have exactly one weight entry."
        )

    # Determine number of output channels from weight array length
    first_weights = next(iter(channel_weights.values()))
    n_output_channels = len(first_weights)

    # Build channel combination matrix (n_output_channels, n_extensions)
    channel_combination = np.zeros((n_output_channels, N_extensions), dtype=np.float32)

    for ext_idx, channel_name in enumerate(weight_names):
        weights = channel_weights[channel_name]
        for output_idx in range(n_output_channels):
            channel_combination[output_idx, ext_idx] = weights[output_idx]

    # Apply fitsbolt batch channel combination
    combined_batch = np.asarray(
        fitsbolt.channel_mixing.batch_channel_combination(
            images=batch_cutouts,
            channel_combination=channel_combination,
        ),
        dtype=batch_cutouts.dtype,
    )

    return combined_batch
