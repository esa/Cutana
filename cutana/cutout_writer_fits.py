#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
FITS cutout writer module for Cutana - handles individual FITS file output.

This module provides static functions for:
- Individual FITS file creation for each cutout
- Proper WCS header preservation
- Multi-extension FITS handling
- File naming conventions and organization
- Metadata embedding in FITS headers
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from astropy.io import fits
from astropy.wcs import WCS
from dotmap import DotMap
from loguru import logger

# Cache for WCS header conversions - key is id(wcs_object)
_wcs_header_cache: Dict[int, Tuple[fits.Header, Any]] = {}


def _get_cached_wcs_info(wcs: WCS) -> Tuple[fits.Header, Any]:
    """Get cached WCS header and pixel scale matrix, computing if not cached."""
    wcs_id = id(wcs)
    if wcs_id not in _wcs_header_cache:
        header = wcs.to_header()
        try:
            pixel_scale_matrix = wcs.pixel_scale_matrix
        except Exception:
            pixel_scale_matrix = None
        _wcs_header_cache[wcs_id] = (header, pixel_scale_matrix)
    return _wcs_header_cache[wcs_id]


def ensure_output_directory(path: Path) -> None:
    """
    Ensure output directory exists.

    Args:
        path: Path to output directory
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {path}")
    except Exception as e:
        logger.error(f"Failed to create output directory {path}: {e}")
        raise


def generate_fits_filename(
    source_id: str,
    file_naming_template: str,
    modifier: str,
    metadata: Dict[str, Any],
) -> str:
    """
    Generate FITS filename based on template and parameters.

    Args:
        source_id: Source identifier
        file_naming_template: Template for filename generation
        modifier: includes the tilestring for euclid
        metadata: Metadata dictionary containing additional information

    Returns:
        Generated filename
    """
    try:
        # Available template variables
        template_vars = {
            "modifier": modifier,
            "source_id": source_id,
            "ra": metadata.get("ra", 0.0),
            "dec": metadata.get("dec", 0.0),
            "timestamp": int(time.time()),
        }

        # Generate filename
        filename = file_naming_template.format(**template_vars)

        # Ensure .fits extension
        if not filename.lower().endswith(".fits"):
            filename += ".fits"

        # Sanitize filename (remove invalid characters)
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        return filename

    except Exception as e:
        logger.error(f"Failed to generate filename for {source_id}: {e}")
        # Fallback to simple naming
        return f"{source_id}_cutout.fits"


def create_wcs_header(
    cutout_shape: tuple,
    original_wcs: Optional[WCS] = None,
    ra_center: Optional[float] = None,
    dec_center: Optional[float] = None,
    pixel_scale: Optional[float] = None,
    resize_factor: Optional[float] = None,
    rescaled_offset_x: Optional[float] = None,
    rescaled_offset_y: Optional[float] = None,
) -> fits.Header:
    """
    Create WCS header for cutout.

    Args:
        cutout_shape: Shape of the cutout (height, width)
        original_wcs: Original WCS from parent image
        ra_center: RA of cutout center in degrees
        dec_center: Dec of cutout center in degrees
        pixel_scale: Pixel scale in arcsec/pixel
        resize_factor: Factor by which the cutout was resized (new_size/original_size)
            Used ONLY for adjusting pixel scale in WCS, NOT for offset scaling.
        rescaled_offset_x: Sub-pixel X offset in FINAL image coordinates (positive = target toward right).
            This offset is ALREADY scaled by resize_factor and should be used as-is.
        rescaled_offset_y: Sub-pixel Y offset in FINAL image coordinates (positive = target toward top).
            This offset is ALREADY scaled by resize_factor and should be used as-is.

    Returns:
        FITS header with WCS information
    """
    # Default offsets to 0 if not provided
    if rescaled_offset_x is None:
        rescaled_offset_x = 0.0
    if rescaled_offset_y is None:
        rescaled_offset_y = 0.0

    try:
        if original_wcs is not None:
            # Use cached WCS header conversion (expensive operation)
            cached_header, cached_pixel_scale_matrix = _get_cached_wcs_info(original_wcs)
            header = cached_header.copy()

            # Update reference pixel to center of cutout, adjusted by rescaled offset
            # CRPIX follows FITS convention: 1-based indexing where pixel (1,1) is bottom-left
            # For an N-pixel image, the geometric center is at (N/2 + 0.5) in FITS 1-based coords
            # The rescaled_offset is in 0-based pixel coordinates, so we add it after converting center to 1-based
            height, width = cutout_shape
            fits_center_x = width / 2.0 + 0.5  # Convert 0-based center to FITS 1-based
            fits_center_y = height / 2.0 + 0.5
            header["CRPIX1"] = fits_center_x + rescaled_offset_x
            header["CRPIX2"] = fits_center_y + rescaled_offset_y
            logger.debug(
                f"WCS CRPIX: FITS_center=({fits_center_x:.2f}, {fits_center_y:.2f}) + "
                f"offset=({rescaled_offset_x:.4f}, {rescaled_offset_y:.4f}) = "
                f"({header['CRPIX1']:.4f}, {header['CRPIX2']:.4f})"
            )

            # Update reference coordinates if provided
            if ra_center is not None and dec_center is not None:
                header["CRVAL1"] = ra_center
                header["CRVAL2"] = dec_center

        elif ra_center is not None and dec_center is not None:
            # Create minimal WCS header
            header = fits.Header()
            height, width = cutout_shape

            header["WCSAXES"] = 2
            header["CTYPE1"] = "RA---TAN"
            header["CTYPE2"] = "DEC--TAN"
            # FITS uses 1-based indexing: center of N-pixel image is at (N/2 + 0.5)
            fits_center_x = width / 2.0 + 0.5
            fits_center_y = height / 2.0 + 0.5
            header["CRPIX1"] = fits_center_x + rescaled_offset_x
            header["CRPIX2"] = fits_center_y + rescaled_offset_y
            header["CRVAL1"] = ra_center
            header["CRVAL2"] = dec_center
            logger.debug(
                f"Minimal WCS CRPIX: FITS_center=({fits_center_x:.2f}, {fits_center_y:.2f}) + "
                f"offset=({rescaled_offset_x:.4f}, {rescaled_offset_y:.4f}) = "
                f"({header['CRPIX1']:.4f}, {header['CRPIX2']:.4f})"
            )

            # Use provided pixel scale or a clearly invalid placeholder
            if pixel_scale is not None:
                scale = pixel_scale / 3600.0  # Convert arcsec to degrees
            else:
                # Use NaN to clearly indicate missing/invalid pixel scale in output headers
                scale = float("nan")
                logger.warning("No pixel scale provided for minimal WCS, using NaN as placeholder")

            # Apply resize factor to pixel scale if provided
            if resize_factor is not None and resize_factor != 1.0:
                scale = scale / resize_factor
                logger.debug(f"Applied resize factor {resize_factor} to fallback WCS pixel scale")

            header["CDELT1"] = -scale  # RA decreases with increasing X
            header["CDELT2"] = scale  # Dec increases with increasing Y
            header["CUNIT1"] = "deg"
            header["CUNIT2"] = "deg"

            # Return early since we've already handled the resize factor for minimal WCS
            return header

        else:
            # No WCS info available
            return fits.Header()

        # Apply resize factor to pixel scale when we have original_wcs
        if original_wcs is not None and resize_factor is not None and resize_factor != 1.0:
            # Scale pixel scale by the resize factor
            # If image was made smaller (resize_factor < 1), pixels represent larger sky area
            # If image was made larger (resize_factor > 1), pixels represent smaller sky area

            # Use cached pixel scale matrix (computed once per WCS)
            if cached_pixel_scale_matrix is not None:
                original_pixel_scale_x = cached_pixel_scale_matrix[0, 0]
                original_pixel_scale_y = cached_pixel_scale_matrix[1, 1]

                # Apply resize factor to get new pixel scale
                new_pixel_scale_x = original_pixel_scale_x / resize_factor
                new_pixel_scale_y = original_pixel_scale_y / resize_factor

                # Handle CD matrix (preferred modern format)
                if "CD1_1" in header and "CD2_2" in header:
                    header["CD1_1"] = header["CD1_1"] / resize_factor
                    header["CD2_2"] = header["CD2_2"] / resize_factor
                    if "CD1_2" in header:
                        header["CD1_2"] = header["CD1_2"] / resize_factor
                    if "CD2_1" in header:
                        header["CD2_1"] = header["CD2_1"] / resize_factor

                # Handle CDELT format or PC+CDELT format
                elif "CDELT1" in header and "CDELT2" in header:
                    # For PC+CDELT format, set CDELT to achieve desired pixel scale
                    if "PC1_1" in header and "PC2_2" in header:
                        pc1_1 = header.get("PC1_1", 1.0)
                        pc2_2 = header.get("PC2_2", 1.0)
                        header["CDELT1"] = new_pixel_scale_x / pc1_1
                        header["CDELT2"] = new_pixel_scale_y / pc2_2
                    else:
                        header["CDELT1"] = new_pixel_scale_x
                        header["CDELT2"] = new_pixel_scale_y
            else:
                # Fallback: simple scaling of existing header values
                if "CD1_1" in header and "CD2_2" in header:
                    header["CD1_1"] = header["CD1_1"] / resize_factor
                    header["CD2_2"] = header["CD2_2"] / resize_factor
                elif "CDELT1" in header and "CDELT2" in header:
                    header["CDELT1"] = header["CDELT1"] / resize_factor
                    header["CDELT2"] = header["CDELT2"] / resize_factor

        return header

    except Exception as e:
        logger.error(f"Failed to create WCS header: {e}")
        # Return minimal header
        return fits.Header()


def write_single_fits_cutout(
    cutout_data: Dict[str, Any],
    output_path: str,
    preserve_wcs: bool = True,
    compression: Optional[str] = None,
    overwrite: bool = False,
) -> bool:
    """
    Write a single cutout as individual FITS file.

    Args:
        cutout_data: Dictionary containing cutout data and metadata
        output_path: Full path for output FITS file
        preserve_wcs: Whether to preserve WCS information
        compression: Optional compression method ('gzip', 'rice', etc.)
        overwrite: Whether to overwrite existing files

    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract data
        source_id = cutout_data["source_id"]
        processed_cutouts = cutout_data.get("processed_cutouts", {})
        metadata = cutout_data.get("metadata", {})
        wcs_info = cutout_data.get("wcs_info", {})

        if not processed_cutouts:
            logger.error(f"No cutout data for source {source_id}")
            return False

        # Check if file exists
        if Path(output_path).exists() and not overwrite:
            logger.warning(f"File already exists: {output_path}")
            return False

        # Create primary HDU
        primary_hdu = fits.PrimaryHDU()

        # Add metadata to primary header using batch update (more efficient)
        primary_hdu.header.update(
            {
                "SOURCE": source_id,
                "RA": metadata.get("ra", 0.0),
                "DEC": metadata.get("dec", 0.0),
                "SIZEARC": metadata.get("diameter_arcsec", 0.0),
                "SIZEPIX": metadata.get("diameter_pixel", 0),
                "PROCTIME": metadata.get("processing_timestamp", time.time()),
                "STRETCH": metadata.get("stretch", "linear"),
                "DTYPE": metadata.get("data_type", "float32"),
            }
        )

        # Create HDU list
        hdu_list = [primary_hdu]

        # Process each channel/filter
        for i, (channel, cutout) in enumerate(processed_cutouts.items()):
            if cutout is None:
                continue

            # Create image HDU
            if compression:
                image_hdu = fits.CompImageHDU(data=cutout, name=channel)
                image_hdu.header["COMPRESS"] = compression
            else:
                image_hdu = fits.ImageHDU(data=cutout, name=channel)

            # Add WCS information if available and requested
            if preserve_wcs:
                try:
                    # Calculate resize factor from metadata
                    resize_factor = None
                    original_size = metadata.get("original_cutout_size")  # Original extraction size
                    final_size = cutout.shape[0]  # Assuming square cutouts, use height

                    if original_size is not None and original_size != final_size:
                        resize_factor = final_size / original_size
                        logger.debug(
                            f"Calculated resize factor: {resize_factor} (from {original_size} to {final_size})"
                        )

                    # Get rescaled offsets from metadata (already scaled by resize factor in cutout_process_utils)
                    rescaled_offset_x = metadata.get("rescaled_offset_x", 0.0)
                    rescaled_offset_y = metadata.get("rescaled_offset_y", 0.0)
                    logger.debug(
                        f"Retrieved rescaled offsets from metadata: ({rescaled_offset_x:.4f}, {rescaled_offset_y:.4f})"
                    )

                    if channel in wcs_info:
                        logger.debug(
                            f"Creating WCS header for channel {channel} using original WCS"
                        )
                        wcs_header = create_wcs_header(
                            cutout.shape,
                            original_wcs=wcs_info[channel],
                            ra_center=metadata.get("ra"),
                            dec_center=metadata.get("dec"),
                            resize_factor=resize_factor,
                            rescaled_offset_x=rescaled_offset_x,
                            rescaled_offset_y=rescaled_offset_y,
                        )
                    else:
                        # Fallback: create minimal WCS using source coordinates
                        logger.debug(
                            f"Creating minimal WCS header for channel {channel} using RA/Dec"
                        )
                        wcs_header = create_wcs_header(
                            cutout.shape,
                            original_wcs=None,
                            ra_center=metadata.get("ra"),
                            dec_center=metadata.get("dec"),
                            resize_factor=resize_factor,
                            rescaled_offset_x=rescaled_offset_x,
                            rescaled_offset_y=rescaled_offset_y,
                        )

                    if wcs_header:
                        image_hdu.header.update(wcs_header)
                        logger.debug(
                            f"Added WCS header with {len(wcs_header)} keywords for channel {channel}"
                        )
                    else:
                        logger.warning(
                            f"WCS header creation returned empty header for channel {channel}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to add WCS for channel {channel}: {e}")

            # Add channel-specific metadata (batch update)
            image_hdu.header.update({"CHANNEL": channel, "FILTER": channel})

            hdu_list.append(image_hdu)

        # Write FITS file (skip verification for performance)
        fits_hdu_list = fits.HDUList(hdu_list)
        fits_hdu_list.writeto(output_path, overwrite=overwrite, output_verify="ignore")

        logger.debug(f"Wrote FITS cutout: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to write FITS cutout to {output_path}: {e}")
        return False


def write_fits_batch(
    batch_data: List[Dict[str, Any]],
    output_directory: str,
    config: DotMap,
    file_naming_template: str = None,
    preserve_wcs: bool = True,
    compression: Optional[str] = None,
    create_subdirs: bool = False,
    overwrite: bool = False,
    modifier: str = "",
) -> List[str]:
    """
    Write a batch of cutouts as FITS files.

    Args:
        batch_data: List of cutout data dictionaries
        output_directory: Base output directory
        config: Configuration DotMap
        file_naming_template: Template for filename generation
        preserve_wcs: Whether to preserve WCS information
        compression: Optional compression method
        create_subdirs: Whether to create subdirectories for organization
        overwrite: Whether to overwrite existing files
        multi_extension: Whether to write as single multi-extension file
        modifier: None

    Returns:
        List of written file paths
    """
    logger.debug(f"Starting FITS batch write to {output_directory} of {len(batch_data)} items")

    if file_naming_template is None:
        file_naming_template = "{modifier}{source_id}_{ra:.6f}_{dec:.6f}_cutout.fits"
    try:
        output_path = Path(output_directory)
        ensure_output_directory(output_path)

        written_files = []

        # Handle the correct data structure: batch_data is a list of batch results
        # Each batch result contains "cutouts" tensor, "metadata" list, "wcs_info" list, and "channel_names"
        for batch_result in batch_data:
            cutouts_tensor = batch_result.get("cutouts")  # Shape: (N, H, W, C)
            metadata_list = batch_result.get("metadata")  # list of metadata dicts
            # list of WCS dicts for each source
            wcs_list = batch_result.get("wcs", batch_result.get("wcs_info", []))
            # ordered channel names matching tensor
            channel_names = batch_result.get("channel_names", [])
            # len(array) returns the size of the first dimension
            N_image = len(cutouts_tensor) if cutouts_tensor is not None else 0

            if cutouts_tensor is None or len(metadata_list) == 0:
                logger.warning("No cutout data or metadata in batch result")
                continue

            # Pre-compute channel weight keys to avoid repeated list() calls
            channel_weight_keys = (
                list(config.channel_weights.keys()) if config.do_only_cutout_extraction else None
            )

            # Process each source in the batch
            for source_idx, metadata in enumerate(metadata_list):
                source_id = metadata["source_id"]

                # Extract cutout for this source from the tensor

                if source_idx >= N_image:
                    logger.warning(
                        f"Metadata index {source_idx} exceeds cutout tensor size {N_image}"
                    )
                    continue
                if config.do_only_cutout_extraction:
                    source_cutout = cutouts_tensor[source_idx]  # Shape: (H, W, C)
                else:
                    source_cutout = cutouts_tensor[source_idx, :, :, :]  # Shape: (H, W, C)

                # Convert tensor to dict format expected by write_single_fits_cutout
                processed_cutouts = {}
                source_wcs_info = {}
                source_wcs_dict = wcs_list[source_idx] if source_idx < len(wcs_list) else {}
                for ij in range(source_cutout.shape[2]):
                    if channel_weight_keys:
                        channel_name = channel_weight_keys[ij]
                    else:
                        channel_name = f"channel_{ij+1}"  # Generic output channel names
                    processed_cutouts[channel_name] = source_cutout[:, :, ij]

                    # Look up WCS using the original channel name from channel_names if available,
                    # otherwise try the output channel_name directly
                    wcs_lookup_key = channel_names[ij] if ij < len(channel_names) else channel_name
                    if wcs_lookup_key in source_wcs_dict:
                        source_wcs_info[channel_name] = source_wcs_dict[wcs_lookup_key]
                    elif channel_name in source_wcs_dict:
                        source_wcs_info[channel_name] = source_wcs_dict[channel_name]
                cutout_data = {
                    "source_id": source_id,
                    "metadata": metadata,
                    "processed_cutouts": processed_cutouts,
                    "wcs_info": source_wcs_info,  # Use properly mapped WCS info
                }

                # Determine output directory for this source
                if create_subdirs:
                    # Create subdirectory based on first few characters of source ID
                    subdir_name = source_id[:3] if len(source_id) >= 3 else "misc"
                    source_output_dir = output_path / subdir_name
                    ensure_output_directory(source_output_dir)
                else:
                    source_output_dir = output_path

                # Generate filename
                filename = generate_fits_filename(
                    source_id, file_naming_template, modifier, metadata
                )
                full_path = source_output_dir / filename

                # Write FITS file
                success = write_single_fits_cutout(
                    cutout_data,
                    str(full_path),
                    preserve_wcs=preserve_wcs,
                    compression=compression,
                    overwrite=overwrite,
                )

                if success:
                    written_files.append(str(full_path))

        logger.info(f"Successfully wrote {len(written_files)} FITS files")
        return written_files

    except Exception as e:
        logger.error(f"Failed to write FITS batch: {e}")
        return []
