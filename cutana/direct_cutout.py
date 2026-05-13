#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Fast in-process cutout generation for small batches.

Provides create_cutouts_direct() for rapid cutout creation without subprocess
overhead. Reuses the same vectorized processing pipeline as the full orchestrator
but runs entirely in-process with mmap FITS loading.

Typical use cases:
- Preview generation (< 200 cutouts)
- Quick-look analysis (< 500 cutouts)
- Interactive exploration in notebooks

For large catalogues (> 1000 sources), use StreamingOrchestrator instead.
"""

from typing import Any, Dict, List

import pandas as pd
from dotmap import DotMap
from loguru import logger

from .cutout_process_utils import _process_sources_batch_vectorized_with_fits_set
from .fits_dataset import load_fits_sets, prepare_fits_sets_and_sources


def create_cutouts_direct(
    catalogue_df: pd.DataFrame,
    config: DotMap,
) -> List[Dict[str, Any]]:
    """Create cutouts directly in-process without subprocess overhead.

    This is the fast path for small batches (< ~1000 sources). It runs the same
    vectorized processing pipeline as the full orchestrator but avoids subprocess
    spawning, JSON/TOML serialization, shared memory IPC, and job tracking.

    FITS files are loaded with memory mapping for fast access.

    Args:
        catalogue_df: DataFrame with columns: SourceID, RA, Dec,
            diameter_pixel (or diameter_arcsec), fits_file_paths.
        config: Cutana configuration DotMap (from get_default_config()).

    Returns:
        List of batch result dicts, each containing:
            - "cutouts": ndarray of shape (N_sources, H, W, N_channels)
            - "metadata": list of per-source metadata dicts
            - "wcs": list of WCS dicts per source
            - "channel_names": list of channel name strings

    Raises:
        ValueError: If catalogue_df is empty.
        KeyError: If required columns are missing.
    """
    if len(catalogue_df) == 0:
        raise ValueError("Empty catalogue provided")

    required_columns = {"SourceID", "RA", "Dec", "fits_file_paths"}
    missing = required_columns - set(catalogue_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    has_size = "diameter_pixel" in catalogue_df.columns or "diameter_arcsec" in catalogue_df.columns
    if not has_size:
        raise KeyError("Need either 'diameter_pixel' or 'diameter_arcsec' column")

    n_sources = len(catalogue_df)
    logger.info(f"Creating {n_sources} cutouts directly (in-process)")

    # Convert DataFrame to list of dicts
    source_batch = catalogue_df.to_dict("records")

    # Group sources by their FITS file sets
    fits_set_to_sources = prepare_fits_sets_and_sources(source_batch)
    logger.info(f"Grouped into {len(fits_set_to_sources)} unique FITS file sets")

    # Determine FITS extensions to load
    fits_extensions = config.fits_extensions

    # Load all needed FITS files with mmap (is_preview=True in load_fits_sets)
    loaded_fits_data = load_fits_sets(list(fits_set_to_sources.keys()), fits_extensions)
    logger.info(f"Loaded {len(loaded_fits_data)} FITS files (mmap)")

    # Process each FITS set through the vectorized pipeline
    all_results = []
    try:
        for fits_set, sources_for_set in fits_set_to_sources.items():
            set_fits_data = {p: loaded_fits_data[p] for p in fits_set if p in loaded_fits_data}

            if not set_fits_data:
                logger.error(f"No FITS data available for set: {fits_set}")
                continue

            batch_results = _process_sources_batch_vectorized_with_fits_set(
                sources_for_set, set_fits_data, config, profiler=None
            )
            all_results.extend(batch_results)
    finally:
        # Always close FITS files. A close failure is best-effort cleanup (we
        # may be in an error path), but surface the root cause rather than
        # silently discarding it.
        for fits_path, (hdul, _) in loaded_fits_data.items():
            try:
                hdul.close()
            except Exception as close_error:
                logger.warning(f"Failed to close FITS file {fits_path}: {close_error}")

    if not all_results:
        raise RuntimeError("No valid cutouts were generated")

    total_cutouts = sum(len(r["metadata"]) for r in all_results if "metadata" in r)
    logger.info(f"Generated {total_cutouts} cutouts directly")

    return all_results
