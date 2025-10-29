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
from typing import Dict, List, Any, Optional
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger


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
) -> fits.Header:
    """
    Create WCS header for cutout.

    Args:
        cutout_shape: Shape of the cutout (height, width)
        original_wcs: Original WCS from parent image
        ra_center: RA of cutout center in degrees
        dec_center: Dec of cutout center in degrees
        pixel_scale: Pixel scale in arcsec/pixel

    Returns:
        FITS header with WCS information
    """
    try:
        header = fits.Header()

        if original_wcs is not None:
            # Use original WCS as base and update for cutout
            wcs_header = original_wcs.to_header()
            header.update(wcs_header)

            # Update reference pixel to center of cutout
            height, width = cutout_shape
            header["CRPIX1"] = width / 2.0
            header["CRPIX2"] = height / 2.0

            # Update reference coordinates if provided
            if ra_center is not None and dec_center is not None:
                header["CRVAL1"] = ra_center
                header["CRVAL2"] = dec_center

        elif ra_center is not None and dec_center is not None:
            # Create minimal WCS header
            height, width = cutout_shape

            header["WCSAXES"] = 2
            header["CTYPE1"] = "RA---TAN"
            header["CTYPE2"] = "DEC--TAN"
            header["CRPIX1"] = width / 2.0
            header["CRPIX2"] = height / 2.0
            header["CRVAL1"] = ra_center
            header["CRVAL2"] = dec_center

            # Use provided pixel scale or default
            scale = pixel_scale / 3600.0 if pixel_scale else -0.000167  # Default ~0.6 arcsec/pixel
            header["CDELT1"] = -scale  # RA decreases with increasing X
            header["CDELT2"] = scale  # Dec increases with increasing Y
            header["CUNIT1"] = "deg"
            header["CUNIT2"] = "deg"

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

        # Add metadata to primary header
        primary_hdu.header["SOURCE"] = source_id
        primary_hdu.header["RA"] = metadata.get("ra", 0.0)
        primary_hdu.header["DEC"] = metadata.get("dec", 0.0)
        primary_hdu.header["SIZEARC"] = metadata.get("diameter_arcsec", 0.0)
        primary_hdu.header["SIZEPIX"] = metadata.get("diameter_pixel", 0)
        primary_hdu.header["PROCTIME"] = metadata.get("processing_timestamp", time.time())
        primary_hdu.header["STRETCH"] = metadata.get("stretch", "linear")
        primary_hdu.header["DTYPE"] = metadata.get("data_type", "float32")

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
            if preserve_wcs and channel in wcs_info:
                try:
                    wcs_header = create_wcs_header(
                        cutout.shape,
                        original_wcs=wcs_info[channel],
                        ra_center=metadata.get("ra"),
                        dec_center=metadata.get("dec"),
                    )
                    image_hdu.header.update(wcs_header)
                except Exception as e:
                    logger.warning(f"Failed to add WCS for channel {channel}: {e}")

            # Add channel-specific metadata
            image_hdu.header["CHANNEL"] = channel
            image_hdu.header["FILTER"] = channel  # Alias for compatibility

            hdu_list.append(image_hdu)

        # Write FITS file
        fits_hdu_list = fits.HDUList(hdu_list)
        fits_hdu_list.writeto(output_path, overwrite=overwrite)

        logger.debug(f"Wrote FITS cutout: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to write FITS cutout to {output_path}: {e}")
        return False


def write_fits_batch(
    batch_data: List[Dict[str, Any]],
    output_directory: str,
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
        # Each batch result contains "cutouts" tensor and "metadata" list
        for batch_result in batch_data:
            cutouts_tensor = batch_result.get("cutouts")  # Shape: (N, H, W, C)
            metadata_list = batch_result.get("metadata")  # list of metadata dicts

            if cutouts_tensor is None or len(metadata_list) == 0:
                logger.warning("No cutout data or metadata in batch result")
                continue

            # Process each source in the batch
            for i, metadata in enumerate(metadata_list):
                source_id = metadata["source_id"]

                # Extract cutout for this source from the tensor
                if i >= cutouts_tensor.shape[0]:
                    logger.warning(
                        f"Metadata index {i} exceeds cutout tensor size {cutouts_tensor.shape[0]}"
                    )
                    continue

                source_cutout = cutouts_tensor[i, :, :, :]  # Shape: (H, W, C)

                # Convert tensor to dict format expected by write_single_fits_cutout
                processed_cutouts = {}
                for i in range(source_cutout.shape[2]):
                    channel_name = f"channel_{i+1}"  # Generic output channel names
                    processed_cutouts[channel_name] = source_cutout[:, :, i]

                cutout_data = {
                    "source_id": source_id,
                    "metadata": metadata,
                    "processed_cutouts": processed_cutouts,
                    "wcs_info": {},  # WCS info not preserved in current tensor format
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


def validate_fits_file(fits_path: str) -> Dict[str, Any]:
    """
    Validate a FITS file and return basic information.

    Args:
        fits_path: Path to FITS file

    Returns:
        Dictionary containing validation results and file info
    """
    try:
        with fits.open(fits_path) as hdul:
            info = {
                "valid": True,
                "num_extensions": len(hdul),
                "extensions": [],
                "file_size": Path(fits_path).stat().st_size,
            }

            # Collect extension information
            for i, hdu in enumerate(hdul):
                ext_info = {
                    "index": i,
                    "name": hdu.name,
                    "type": type(hdu).__name__,
                    "shape": getattr(hdu.data, "shape", None) if hdu.data is not None else None,
                    "dtype": str(hdu.data.dtype) if hdu.data is not None else None,
                }
                info["extensions"].append(ext_info)

            return info

    except Exception as e:
        logger.error(f"FITS validation failed for {fits_path}: {e}")
        return {
            "valid": False,
            "error": str(e),
            "file_size": Path(fits_path).stat().st_size if Path(fits_path).exists() else 0,
        }
