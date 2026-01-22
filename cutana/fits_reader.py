#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
FITS file reader module for Cutana - handles FITS file loading and WCS extraction.

This module provides optimized FITS loading strategies:
- For preview (< 200 sources): Memory-mapped lazy loading
- For small batches (< 500 sources): Memory-mapped lazy loading
- For large batches (>= 500 sources): fsspec-based partial loading
- All using astropy for consistency
"""

import os
from typing import Dict, List, Optional, Tuple

from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger


def load_fits_file(
    fits_path: str,
    fits_extensions: Optional[List[str]] = None,
    n_sources: int = 100,
    is_preview: bool = False,
) -> Tuple[fits.HDUList, Dict[str, WCS]]:
    """
    Load FITS file with optimized strategy based on source count and use case.

    For NFS-based storage with ~1.4GB FITS files per extension:
    - Preview mode: Use memory mapping with lazy loading for fast response
    - Small batches (< 500 sources): Use memory mapping for efficiency
    - Large batches (>= 500 sources): Use fsspec for partial loading

    Args:
        fits_path: Path to the FITS file
        fits_extensions: List of extension names to process
        n_sources: Number of sources being processed (determines strategy)
        is_preview: Whether this is for preview generation (affects strategy)

    Returns:
        Tuple of (HDUList, WCS dictionary keyed by extension name)

    Raises:
        FileNotFoundError: If FITS file doesn't exist
        ValueError: If FITS file is corrupted or invalid
    """
    if fits_extensions is None:
        fits_extensions = ["PRIMARY"]

    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    try:
        # Determine loading strategy
        if is_preview or n_sources < 500:
            # Use memory mapping for small source counts or preview
            logger.debug(
                f"Loading FITS with memory mapping (n_sources={n_sources}, preview={is_preview}): {fits_path}"
            )
            use_fsspec = False
            lazy_load_hdus = True
            memmap = True
        else:
            # Use fsspec for large source counts
            logger.debug(f"Loading FITS with fsspec (n_sources={n_sources}): {fits_path}")
            use_fsspec = True
            lazy_load_hdus = True
            memmap = False

        # Open FITS file with appropriate strategy
        if use_fsspec:
            # Use fsspec for partial loading over network
            # This is optimal for large batches on NFS
            file_url = f"file://{os.path.abspath(fits_path)}"
            hdul = fits.open(
                file_url,
                use_fsspec=True,
                fsspec_kwargs={"block_size": 1_000_000, "cache_type": "bytes"},
                lazy_load_hdus=lazy_load_hdus,
            )
        else:
            # Use memory mapping for local/fast access
            hdul = fits.open(
                fits_path,
                memmap=memmap,
                lazy_load_hdus=lazy_load_hdus,
            )

        wcs_dict = {}

        # Log available extensions for debugging
        available_extensions = [
            hdu.name if hasattr(hdu, "name") else f"HDU{i}" for i, hdu in enumerate(hdul)
        ]
        logger.debug(f"Available extensions in {fits_path}: {available_extensions}")

        # Extract WCS for each requested extension
        for ext_name in fits_extensions:
            try:
                # Handle PRIMARY extension (index 0)
                if ext_name == "PRIMARY" and len(hdul) > 0:
                    header = hdul[0].header
                elif ext_name in hdul:
                    header = hdul[ext_name].header
                else:
                    logger.warning(f"Extension {ext_name} not found in {fits_path}")
                    continue

                # Check if this extension has image data
                if ext_name == "PRIMARY":
                    hdu = hdul[0]
                else:
                    hdu = hdul[ext_name]

                if hdu.data is None:
                    logger.warning(f"Extension {ext_name} has no image data in {fits_path}")
                    continue

                wcs_obj = WCS(header)
                wcs_dict[ext_name] = wcs_obj
                logger.debug(f"Loaded WCS for extension {ext_name}")

            except Exception as e:
                logger.error(f"Failed to load WCS for extension {ext_name}: {e}")

        if not wcs_dict:
            logger.error(f"No valid extensions found in {fits_path}")

        return hdul, wcs_dict

    except Exception as e:
        logger.error(f"Failed to load FITS file {fits_path}: {e}")
        raise ValueError(f"Invalid FITS file: {fits_path}")
