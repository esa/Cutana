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
import numpy as np
from skimage import transform, util
from loguru import logger
import fitsbolt
from dotmap import DotMap
from .normalisation_parameters import convert_cfg_to_fitsbolt_cfg


def resize_images(
    images, target_size: Tuple[int, int], interpolation: str = "bilinear"
) -> np.ndarray:
    """
    Resize images to target size using skimage transform.
    Handles both single images and batches of images.

    Args:
        images: Input image array (H, W) or list of images or batch (N, H, W)
        target_size: Tuple of (height, width) for target size
        interpolation: Interpolation method (nearest, bilinear, bicubic)

    Returns:
        Batch array of shape (N, H, W) with resized images
    """
    # Convert to list if single image or batch
    if isinstance(images, np.ndarray):
        if len(images.shape) == 2:  # Single image (H, W)
            images = [images]
        elif len(images.shape) == 3:  # Batch (N, H, W)
            images = [images[i] for i in range(images.shape[0])]

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

    resized_batch = []
    for image in images:
        if image.shape[:2] == target_size:
            resized_batch.append(image.copy())
            continue

        try:
            # Resize using skimage
            resized = transform.resize(
                image, target_size, order=order, preserve_range=True, anti_aliasing=True
            ).astype(image.dtype)
            resized_batch.append(resized)
        except Exception as e:
            logger.error(f"Image resizing failed: {e}")
            # Fallback: return zeros of target size
            resized_batch.append(np.zeros(target_size, dtype=image.dtype))

    return np.stack(resized_batch, axis=0)


def resize_batch_tensor(
    source_cutouts: Dict[str, Dict[str, np.ndarray]],
    target_resolution: Tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resize all source cutouts and return as (N_sources, H, W, N_extensions) tensor.

    Args:
        source_cutouts: Dict mapping source_id -> {channel_key: cutout}
        target_resolution: Target (height, width)
        interpolation: Interpolation method

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

    # Fill tensor
    for i, source_id in enumerate(source_ids):
        source_cutouts_dict = source_cutouts[source_id]
        for j, ext_name in enumerate(extension_names):
            if ext_name in source_cutouts_dict:
                cutout = source_cutouts_dict[ext_name]
                if cutout is not None and cutout.size > 0:
                    # Resize if needed
                    if cutout.shape != target_resolution:
                        resized = resize_images(cutout, target_resolution, interpolation)[0]
                    else:
                        resized = cutout.copy()
                    batch_tensor[i, :, :, j] = resized

    return batch_tensor


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
        config: Configuration DotMap containing all normalization parameters

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

    # Get fitsbolt parameters from config (with debugging logs included)
    num_channels = images_array.shape[-1]
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
