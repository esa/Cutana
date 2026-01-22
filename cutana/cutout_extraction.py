#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Vectorized cutout extraction utilities for Cutana.

This module provides optimized, vectorized functions for extracting multiple
cutouts simultaneously from FITS files using batch processing of coordinates
and bounds calculations.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger

from .flux_conversion import apply_flux_conversion

# Suppress specific astropy warnings for cleaner output
warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)


def get_pixel_scale_arcsec_per_pixel(wcs_obj: WCS) -> float:
    """
    Calculate pixel scale in arcseconds per pixel from WCS.

    Args:
        wcs_obj: WCS object for coordinate transformation

    Returns:
        Pixel scale in arcseconds per pixel
    """
    try:
        # Get pixel scale matrix
        pixel_scale_matrix = wcs_obj.pixel_scale_matrix
        # Use the mean of the diagonal elements for square pixels
        pixel_scale_deg = np.sqrt(abs(np.linalg.det(pixel_scale_matrix)))
        # Convert to arcseconds
        pixel_scale_arcsec = pixel_scale_deg * 3600.0
        return pixel_scale_arcsec
    except Exception as e:
        logger.warning(f"Could not determine pixel scale from WCS, using default: {e}")
        # Fallback to a reasonable default (typical for astronomical images)
        return 0.1  # arcsec/pixel


def arcsec_to_pixels(diameter_arcsec: float, wcs_obj: WCS) -> int:
    """
    Convert diameter in arcseconds to pixels using WCS information.

    Args:
        diameter_arcsec: Diameter in arcseconds
        wcs_obj: WCS object for pixel scale calculation

    Returns:
        Diameter in pixels (integer)
    """
    try:
        pixel_scale = get_pixel_scale_arcsec_per_pixel(wcs_obj)
        diameter_pixels = int(round(diameter_arcsec / pixel_scale))
        logger.debug(
            f'Converted {diameter_arcsec}" to {diameter_pixels} pixels (scale: {pixel_scale}"/pix)'
        )
        return max(1, diameter_pixels)  # Ensure at least 1 pixel
    except Exception as e:
        logger.error(f"Failed to convert arcsec to pixels: {e}")
        # Fallback calculation
        return max(1, int(round(diameter_arcsec / 0.1)))


def extract_cutouts_vectorized_from_extension(
    hdu: fits.ImageHDU,
    wcs_obj: WCS,
    ra_array: np.ndarray,
    dec_array: np.ndarray,
    size_pixels_array: np.ndarray,
    source_ids: List[str] = None,
    padding_factor: float = 1.0,
    config=None,
) -> Tuple[List[Optional[np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract multiple cutouts from a single FITS extension using vectorized operations.

    This function implements the vectorized approach:
    Step 1: Build source batch arrays (ra, dec, size_pixels)
    Step 2: Use vectorized SkyCoord for coordinate transformation
    Step 3: Vectorized bound computation
    Step 4: Broadcasted array operations for cutout extraction
    Step 5: Vectorized padding

    Args:
        hdu: FITS ImageHDU object
        wcs_obj: WCS object for this extension
        ra_array: Array of right ascensions in degrees
        dec_array: Array of declinations in degrees
        size_pixels_array: Array of cutout sizes in pixels
        source_ids: Optional list of source IDs for logging
        padding_factor: Factor to scale the extraction area (1.0 = no padding)

    Returns:
        Tuple of (cutout_list, success_mask, pixel_offset_x, pixel_offset_y) where:
        - cutout_list: List of cutout arrays (or None for failures)
        - success_mask: Boolean array indicating successful extractions
        - pixel_offset_x: Array of sub-pixel X offsets (positive = target toward right)
        - pixel_offset_y: Array of sub-pixel Y offsets (positive = target toward top)
    """
    n_sources = len(ra_array)
    logger.debug(f"Starting vectorized cutout extraction for {n_sources} sources")

    if source_ids is None:
        source_ids = [f"source_{i}" for i in range(n_sources)]

    # Get image data once
    image_data = hdu.data
    if image_data is None:
        logger.error("No image data in HDU")
        return (
            [None] * n_sources,
            np.zeros(n_sources, dtype=bool),
            np.zeros(n_sources, dtype=np.float64),
            np.zeros(n_sources, dtype=np.float64),
        )

    img_height, img_width = image_data.shape

    # Step 2: Vectorized coordinate transformation using SkyCoord
    try:
        logger.debug("Step 2: Vectorized coordinate transformation")
        # Create vectorized SkyCoord object for all sources at once
        coords = SkyCoord(ra=ra_array * u.degree, dec=dec_array * u.degree, frame="icrs")

        # Transform all coordinates to pixel space at once
        pixel_coords = wcs_obj.world_to_pixel(coords)
        pixel_x_array = np.array(pixel_coords[0])
        pixel_y_array = np.array(pixel_coords[1])

    except Exception as e:
        logger.error(f"Vectorized coordinate transformation failed: {e}")
        return (
            [None] * n_sources,
            np.zeros(n_sources, dtype=bool),
            np.zeros(n_sources, dtype=np.float64),
            np.zeros(n_sources, dtype=np.float64),
        )

    # Step 3: Vectorized bound computation
    logger.debug("Step 3: Vectorized bound computation")

    # Apply padding factor to extraction area
    # padding_factor < 1.0 means zoom-in (smaller extraction area)
    # padding_factor > 1.0 means zoom-out (larger extraction area)
    extraction_sizes = (size_pixels_array * padding_factor).astype(int)

    # Calculate bounds for all sources simultaneously
    # For proper extraction, we need to extract the full requested size
    # The key insight: for size N, we want exactly N pixels
    # Center the extraction around the pixel coordinate, handling both odd and even sizes:
    # - Size 5 (odd): extract from [center-2, center+3) = 5 pixels
    # - Size 6 (even): extract from [center-3, center+3) = 6 pixels
    half_sizes_left = extraction_sizes // 2
    half_sizes_right = extraction_sizes - half_sizes_left

    # Use proper bounds calculation without double-flooring
    # pixel_x_array and pixel_y_array are already floats from WCS transform
    x_mins = (pixel_x_array - half_sizes_left).astype(int)
    x_maxs = (pixel_x_array + half_sizes_right).astype(int)
    y_mins = (pixel_y_array - half_sizes_left).astype(int)
    y_maxs = (pixel_y_array + half_sizes_right).astype(int)

    # Compute pixel offsets: the sub-pixel difference between the target position
    # and the center of the extracted cutout. Following FITS convention:
    # - Positive offset means target is toward top-right (larger pixel indices)
    # - Negative offset means target is toward bottom-left (smaller pixel indices)
    # The cutout center in pixel coords is at (x_mins + extraction_sizes/2, y_mins + extraction_sizes/2)
    cutout_center_x = x_mins + extraction_sizes / 2.0
    cutout_center_y = y_mins + extraction_sizes / 2.0
    pixel_offset_x = pixel_x_array - cutout_center_x
    pixel_offset_y = pixel_y_array - cutout_center_y

    # Clip bounds to image dimensions (vectorized)
    x_mins_clipped = np.maximum(0, x_mins)
    x_maxs_clipped = np.minimum(img_width, x_maxs)
    y_mins_clipped = np.maximum(0, y_mins)
    y_maxs_clipped = np.minimum(img_height, y_maxs)

    # Check for valid regions (vectorized)
    valid_mask = (x_maxs_clipped > x_mins_clipped) & (y_maxs_clipped > y_mins_clipped)

    # Step 4: Extract cutouts using advanced indexing and broadcasting
    logger.debug("Step 4: Vectorized cutout extraction")

    # Pre-allocate results list in correct order
    cutouts = [None] * n_sources
    success_mask = np.zeros(n_sources, dtype=bool)

    # Group sources by cutout size for more efficient processing
    unique_sizes = np.unique(size_pixels_array)

    for target_size in unique_sizes:
        size_mask = (size_pixels_array == target_size) & valid_mask
        if not np.any(size_mask):
            continue

        indices = np.where(size_mask)[0]

        # Extract cutouts for this size group
        for i in indices:
            try:
                # Extract raw cutout
                y_min, y_max = y_mins_clipped[i], y_maxs_clipped[i]
                x_min, x_max = x_mins_clipped[i], x_maxs_clipped[i]

                raw_cutout = image_data[y_min:y_max, x_min:x_max].copy()

                # Step 5: Handle padding/resizing based on extraction size
                # If extraction area is larger than target due to padding, we need to resize down
                # If extraction area is smaller (at edges), we need to pad with zeros
                extraction_size = extraction_sizes[i]

                # First, ensure raw cutout is padded to full extraction size if needed
                if raw_cutout.shape[0] != extraction_size or raw_cutout.shape[1] != extraction_size:
                    # Always pad with zeros for out-of-bounds regions
                    # NEVER use reflection, mirroring, or edge values
                    padded_extraction = np.zeros(
                        (extraction_size, extraction_size), dtype=raw_cutout.dtype
                    )

                    # Calculate padding to place raw cutout in center of padded area
                    # This ensures consistent behavior regardless of edge position
                    pad_y_start = (extraction_size - raw_cutout.shape[0]) // 2
                    pad_x_start = (extraction_size - raw_cutout.shape[1]) // 2

                    # Handle negative padding (shouldn't happen but be defensive)
                    pad_y_start = max(0, pad_y_start)
                    pad_x_start = max(0, pad_x_start)

                    # Calculate end positions
                    pad_y_end = min(pad_y_start + raw_cutout.shape[0], extraction_size)
                    pad_x_end = min(pad_x_start + raw_cutout.shape[1], extraction_size)

                    # Calculate how much of raw_cutout to use
                    raw_y_end = min(raw_cutout.shape[0], pad_y_end - pad_y_start)
                    raw_x_end = min(raw_cutout.shape[1], pad_x_end - pad_x_start)

                    # Place the raw cutout data into the padded array
                    padded_extraction[pad_y_start:pad_y_end, pad_x_start:pad_x_end] = raw_cutout[
                        :raw_y_end, :raw_x_end
                    ]

                    # Adjust pixel offsets to account for symmetric padding
                    # The cutout center has shifted by the padding offset
                    # So, the new center is shifted by (pad_x_start, pad_y_start)
                    # We want the offset to be relative to the center of the padded extraction
                    # So, subtract the padding offset from the original offset
                    pixel_offset_x[i] -= pad_x_start
                    pixel_offset_y[i] -= pad_y_start

                    raw_cutout = padded_extraction

                # apply flux conversion here
                if config and config.apply_flux_conversion:
                    logger.debug("Step 4.5: Apply Flux conversion")

                    raw_cutout = apply_flux_conversion(config, raw_cutout, hdu.header)

                # Store the extracted cutout at its actual extraction size
                # The extraction size is determined by target_size * padding_factor
                # Resizing to final target_size will be handled later in the processing pipeline
                cutouts[i] = raw_cutout

                success_mask[i] = True

            except Exception as e:
                logger.debug(f"Failed to extract cutout for source {source_ids[i]}: {e}")
                cutouts[i] = None  # Already None, but explicit

    # Log invalid sources (cutouts[i] already None for these)
    for i in range(n_sources):
        if not valid_mask[i]:
            logger.warning(
                f"Invalid cutout region for source {source_ids[i]} at RA={ra_array[i]}, Dec={dec_array[i]}"
            )

    successful_count = np.sum(success_mask)
    logger.debug(f"Vectorized extraction completed: {successful_count}/{n_sources} successful")

    return cutouts, success_mask, pixel_offset_x, pixel_offset_y


def extract_cutouts_batch_vectorized(
    sources_batch: List[Dict[str, Any]],
    hdul: fits.HDUList,
    wcs_dict: Dict[str, WCS],
    fits_extensions: List[str] = None,
    padding_factor: float = 1.0,
    config=None,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, WCS]],
    List[str],
    float,
    Dict[str, Dict[str, float]],
]:
    """
    Extract cutouts for a batch of sources using vectorized operations.

    This function groups sources and processes them with vectorized coordinate
    transformations and batch cutout extraction.

    Args:
        sources_batch: List of source dictionaries containing RA, Dec, size info
        hdul: Pre-loaded FITS HDUList
        wcs_dict: WCS dictionary for the FITS file
        fits_extensions: List of extension names to process
        padding_factor: Factor to scale the extraction area (1.0 = no padding)

    Returns:
        Tuple of (combined_cutouts, combined_wcs, source_ids, pixel_scale, combined_offsets) where:
        - combined_cutouts: Dict mapping source_id -> {ext_name: cutout_array}
        - combined_wcs: Dict mapping source_id -> {ext_name: wcs_object}
        - source_ids: List of source IDs that were processed
        - pixel_scale: Pixel scale in arcseconds per pixel
        - combined_offsets: Dict mapping source_id -> {"x": offset_x, "y": offset_y}
    """
    if fits_extensions is None:
        fits_extensions = ["PRIMARY"]

    n_sources = len(sources_batch)
    logger.info(f"Starting vectorized batch extraction for {n_sources} sources")

    # Step 1: Build source batch arrays
    logger.debug("Step 1: Building source batch arrays")

    source_ids = []
    ra_array = np.zeros(n_sources)
    dec_array = np.zeros(n_sources)
    size_pixels_array = np.zeros(n_sources, dtype=int)

    for i, source_data in enumerate(sources_batch):
        source_ids.append(source_data["SourceID"])
        ra_array[i] = float(source_data["RA"])
        dec_array[i] = float(source_data["Dec"])

        # Handle size parameters
        if "diameter_pixel" in source_data and source_data.get("diameter_pixel") is not None:
            size_pixels_array[i] = int(source_data["diameter_pixel"])
        elif "diameter_arcsec" in source_data and source_data.get("diameter_arcsec") is not None:
            # For vectorized processing, convert arcsec to pixels using first extension's WCS
            first_ext = fits_extensions[0]
            if first_ext in wcs_dict:
                size_pixels_array[i] = arcsec_to_pixels(
                    float(source_data["diameter_arcsec"]), wcs_dict[first_ext]
                )
            else:
                logger.warning(f"No WCS for extension {first_ext}, defaulting to 128 pixels")
                size_pixels_array[i] = 128
        else:
            logger.warning(
                f"No valid size parameter for source {source_ids[i]}, defaulting to 128 pixels"
            )
            size_pixels_array[i] = 128

    # Process each extension
    pixel_scale = get_pixel_scale_arcsec_per_pixel(wcs_dict[fits_extensions[0]])
    combined_cutouts = {}
    combined_wcs = {}
    combined_offsets = {}  # source_id -> {"x": offset_x, "y": offset_y}

    for ext_name in fits_extensions:
        if ext_name not in hdul or ext_name not in wcs_dict:
            logger.warning(f"Extension {ext_name} not available, skipping")
            continue

        logger.debug(f"Processing extension {ext_name} for {n_sources} sources")

        # Extract cutouts for all sources in this extension using vectorized method
        cutout_list, success_mask, offset_x_array, offset_y_array = (
            extract_cutouts_vectorized_from_extension(
                hdul[ext_name],
                wcs_dict[ext_name],
                ra_array,
                dec_array,
                size_pixels_array,
                source_ids,
                padding_factor,
                config,
            )
        )

        # Organize results by source ID
        for i, (source_id, cutout) in enumerate(zip(source_ids, cutout_list)):
            if cutout is not None:
                if source_id not in combined_cutouts:
                    combined_cutouts[source_id] = {}
                    combined_wcs[source_id] = {}
                    # Store pixel offsets (same for all extensions since coords are the same)
                    combined_offsets[source_id] = {
                        "x": float(offset_x_array[i]),
                        "y": float(offset_y_array[i]),
                    }

                combined_cutouts[source_id][ext_name] = cutout
                combined_wcs[source_id][ext_name] = wcs_dict[ext_name]

    successful_sources = len(combined_cutouts)
    logger.info(
        f"Vectorized batch extraction completed: {successful_sources}/{n_sources} sources successful"
    )

    return combined_cutouts, combined_wcs, source_ids, pixel_scale, combined_offsets
