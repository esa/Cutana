#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Preview cutout generation for Cutana UI.

This module provides efficient preview generation functionality with intelligent
caching and memory management for handling large source catalogues.
"""

import ast
import asyncio
import os
import time
from typing import Dict, List, Tuple, Optional
from dotmap import DotMap
from pathlib import Path
import numpy as np
from loguru import logger

from .catalogue_preprocessor import (
    load_catalogue,
    CatalogueValidationError,
)
from .cutout_process import (
    _process_sources_batch_vectorized_with_fits_set,
)
from .fits_dataset import prepare_fits_sets_and_sources, load_fits_sets


class PreviewCache:
    """Class to hold preview cache state."""

    sources_cache: Optional[List[Dict]] = None
    fits_sets_cache: Optional[Dict] = None
    fits_data_cache: Optional[Dict] = None
    config_cache: Optional[Dict] = None
    preview_seed: Optional[int] = None  # Seed for consistent preview sampling


def _parse_fits_file_paths(fits_paths_str) -> List[str]:
    """Parse FITS file paths from string format."""
    if isinstance(fits_paths_str, str):
        if fits_paths_str.startswith("[") and fits_paths_str.endswith("]"):
            try:
                return ast.literal_eval(fits_paths_str)
            except (ValueError, SyntaxError):
                return [path.strip().strip("'\"") for path in fits_paths_str.strip("[]").split(",")]
        else:
            return [fits_paths_str]
    else:
        return fits_paths_str


def _get_selected_extensions(config: DotMap) -> List[str]:
    """Extract selected extensions from config."""
    selected_extensions = []
    if config and config.selected_extensions:
        for ext in config.selected_extensions:
            if isinstance(ext, dict):
                actual_ext = ext.get("ext", "PRIMARY")
                if actual_ext == "PrimaryHDU":
                    actual_ext = "PRIMARY"
                selected_extensions.append(actual_ext)
            else:
                selected_extensions.append(str(ext))

    return selected_extensions if selected_extensions else ["PRIMARY"]


def _extract_cutout_from_batch_result(result: Dict) -> List[Tuple[float, float, np.ndarray]]:
    """Extract cutout arrays from batch result."""
    cutouts = []

    if "cutouts" not in result or "metadata" not in result:
        logger.debug("Missing required keys in batch result")
        return cutouts

    batch_tensor = result["cutouts"]  # Shape: (N_sources, H, W, N_channels)
    metadata_list = result["metadata"]

    logger.debug(
        f"Processing batch with {len(metadata_list)} sources, tensor shape: {getattr(batch_tensor, 'shape', 'unknown')}"
    )

    # Process each source in the batch
    for i, metadata in enumerate(metadata_list):
        if i < batch_tensor.shape[0]:
            source_cutout = batch_tensor[i]  # Shape: (H, W, N_channels)
            ra = metadata.get("ra", 0.0)
            dec = metadata.get("dec", 0.0)

            # Return the full multi-channel cutout for proper display
            if len(source_cutout.shape) == 3:
                num_channels = source_cutout.shape[2]

                # Handle different channel configurations
                if num_channels == 1:
                    # Single channel - return as 2D grayscale
                    h, w = source_cutout.shape[:2]
                    cutout_data = np.zeros((h, w, 3), dtype=source_cutout.dtype)
                    for ii in range(3):
                        cutout_data[:, :, ii] = source_cutout[:, :, 0]
                elif num_channels == 2:
                    # Two channels - create RGB with RG channels, B channel empty (zero)
                    h, w = source_cutout.shape[:2]
                    cutout_data = np.zeros((h, w, 3), dtype=source_cutout.dtype)
                    cutout_data[:, :, 0] = source_cutout[:, :, 0]  # R channel
                    cutout_data[:, :, 1] = source_cutout[:, :, 1]  # G channel
                    # B channel remains zero
                elif num_channels == 3:
                    # Three channels - return as RGB
                    cutout_data = source_cutout
                else:
                    # More than 3 channels - use first 3 as RGB
                    cutout_data = source_cutout[:, :, :3]

                # Normalize to 0-255 range
                cutout_data = np.nan_to_num(cutout_data, nan=0)
                if cutout_data.max() > cutout_data.min():
                    cutout_data = (cutout_data - cutout_data.min()) / (
                        cutout_data.max() - cutout_data.min()
                    )
                    cutout_array = (cutout_data * 255).astype(np.uint8)
                else:
                    cutout_array = np.zeros_like(cutout_data, dtype=np.uint8)

                cutouts.append((ra, dec, cutout_array))
            else:
                logger.debug(f"Skipped cutout {i}: invalid shape {source_cutout.shape}")
        else:
            logger.debug(f"Skipped cutout {i}: batch index out of range")

    logger.debug(f"Extracted {len(cutouts)} valid cutouts from batch")
    return cutouts


async def load_sources_for_previews(
    catalogue_path: str, config: DotMap = None, progress_callback=None
) -> Dict:
    """
    Load and cache 200 randomly selected sources for fast preview generation.

    Uses a memory-efficient approach:
    1. Get unique FITS file sets from catalogue (much fewer than sources)
    2. Select top 1 FITS file set by source count
    3. Extract sources using those FITS sets and apply size quartile filtering
    4. Randomly select 200 sources from upper quartiles

    Args:
        catalogue_path: Path to source catalogue
        config: Configuration DotMap with processing parameters

    Returns:
        Dictionary with cache statistics and info
    """
    logger.info("Loading and caching sources for fast preview generation")
    if progress_callback:
        progress_callback("Loading catalogue and analyzing FITS sets [1/3]...")

    if not catalogue_path or not Path(catalogue_path).exists():
        raise FileNotFoundError(f"No valid catalogue path provided: {catalogue_path}")

    # Load and validate source catalogue, only load do not check (was already done)
    try:
        catalogue_df = load_catalogue(catalogue_path)
        if len(catalogue_df) == 0:
            raise ValueError("Empty catalogue provided")
    except CatalogueValidationError as e:
        raise ValueError(f"Catalogue validation failed: {e}") from e

    logger.info(f"Loaded catalogue with {len(catalogue_df)} sources")

    # for preview only, considerably shorten the catalogue for speed up:
    if len(catalogue_df) > 20000:
        # take first 1k and sample 19k from the rest
        sample_indices = np.concatenate(
            [
                np.arange(1000),
                np.random.choice(np.arange(1000, len(catalogue_df)), 19000, replace=False),
            ]
        )
        catalogue_df = catalogue_df.iloc[sample_indices]

    # Get unique FITS file sets directly from catalogue
    logger.info("Analyzing unique FITS file sets in catalogue...")

    fits_set_counts = {}
    parse_errors = 0
    for _, row in catalogue_df.iterrows():
        try:
            fits_paths = _parse_fits_file_paths(row["fits_file_paths"])
            fits_paths = [os.path.normpath(path) for path in fits_paths]
            fits_set = tuple((fits_paths))
            fits_set_counts[fits_set] = fits_set_counts.get(fits_set, 0) + 1
        except Exception:
            parse_errors += 1
            continue

    if parse_errors > 0:
        logger.warning(f"Failed to parse FITS paths for {parse_errors} sources")

    logger.info(f"Found {len(fits_set_counts)} unique FITS file sets")

    # Select top 1 FITS file set by source count for efficient preview generation
    top_fits_sets = sorted(fits_set_counts.items(), key=lambda x: x[1], reverse=True)[:1]
    selected_fits_sets = {fits_set for fits_set, _ in top_fits_sets}

    # Extract sources that use these selected FITS file sets
    logger.info("Extracting sources for selected FITS file sets...")
    matching_sources = []
    process_errors = 0

    for _, row in catalogue_df.iterrows():
        try:
            fits_paths = _parse_fits_file_paths(row["fits_file_paths"])
            fits_paths = [os.path.normpath(path) for path in fits_paths]
            fits_set = tuple(fits_paths)

            if fits_set in selected_fits_sets:
                source_data = {
                    "SourceID": row["SourceID"],
                    "RA": row["RA"],
                    "Dec": row["Dec"],
                    "diameter_arcsec": row.get("diameter_arcsec"),
                    "diameter_pixel": row.get("diameter_pixel"),
                    "fits_file_paths": row["fits_file_paths"],
                }
                matching_sources.append(source_data)

        except Exception:
            process_errors += 1
            continue

    if process_errors > 0:
        logger.warning(f"Failed to process {process_errors} sources")

    logger.info(f"Found {len(matching_sources)} sources using selected FITS file sets")

    # Apply size quartile filtering to get 200 sources from upper quartiles
    if len(matching_sources) > 0:
        # Determine size column
        size_col = None
        if any(s.get("diameter_pixel") is not None for s in matching_sources[:100]):
            size_col = "diameter_pixel"
        elif any(s.get("diameter_arcsec") is not None for s in matching_sources[:100]):
            size_col = "diameter_arcsec"

        if size_col:
            # Extract size values and apply quartile filtering
            size_values = [s.get(size_col) for s in matching_sources if s.get(size_col) is not None]
            if len(size_values) > 0:
                size_q50 = np.percentile(size_values, 50)
                upper_quartile_sources = [
                    s for s in matching_sources if s.get(size_col, 0) >= size_q50
                ]
                logger.info(
                    f"Applied size quartile filtering: {len(upper_quartile_sources)}/"
                    f"{len(matching_sources)} sources in upper two quartiles of {size_col} "
                    f"(threshold: {size_q50:.2f})"
                )
            else:
                upper_quartile_sources = matching_sources
                logger.warning("No valid size values found, using all sources")
        else:
            upper_quartile_sources = matching_sources
            logger.warning("No size column found, using all sources")

        # Randomly select up to 200 sources from upper quartiles
        preview_seed = int(time.time() * 1000) % (2**31)
        PreviewCache.preview_seed = preview_seed

        num_to_select = min(200, len(upper_quartile_sources))
        if len(upper_quartile_sources) > 200:
            rng = np.random.RandomState(preview_seed)
            selected_indices = rng.choice(
                len(upper_quartile_sources), size=num_to_select, replace=False
            )
            filtered_sources = [upper_quartile_sources[i] for i in selected_indices]
        else:
            filtered_sources = upper_quartile_sources

        logger.info(f"Final selection: {len(filtered_sources)} sources for caching")
    else:
        filtered_sources = []
        logger.warning("No sources found for selected FITS file sets")

    # Group final filtered sources by FITS file sets for caching
    fits_set_to_sources = prepare_fits_sets_and_sources(filtered_sources)
    selected_extensions = _get_selected_extensions(config)

    # Load FITS data using the refactored function
    logger.info(f"Pre-loading FITS data for {len(fits_set_to_sources)} unique FITS file sets...")
    if progress_callback:
        progress_callback("Pre-loading FITS data for preview [2/3]...")

    loop = asyncio.get_event_loop()
    fits_data_cache = await loop.run_in_executor(
        None, load_fits_sets, list(fits_set_to_sources.keys()), selected_extensions
    )

    logger.info(f"Pre-loaded {len(fits_data_cache)} FITS files into cache")
    if progress_callback:
        progress_callback("Finalizing preview source cache [3/3]...")

    # Store in cache
    PreviewCache.sources_cache = filtered_sources
    PreviewCache.fits_sets_cache = fits_set_to_sources
    PreviewCache.fits_data_cache = fits_data_cache
    PreviewCache.config_cache = {
        "catalogue_path": catalogue_path,
        "selected_extensions": selected_extensions,
        "num_cached_sources": len(filtered_sources),
        "num_cached_fits": len(fits_data_cache),
        "cache_timestamp": asyncio.get_event_loop().time(),
    }

    cache_info = {
        "status": "success",
        "num_cached_sources": len(filtered_sources),
        "num_cached_fits": len(fits_data_cache),
        "fits_file_sets": len(fits_set_to_sources),
        "source_size_range": (
            (
                min(
                    s.get("diameter_pixel") or s.get("diameter_arcsec") or 0
                    for s in filtered_sources
                ),
                max(
                    s.get("diameter_pixel") or s.get("diameter_arcsec") or 0
                    for s in filtered_sources
                ),
            )
            if filtered_sources
            else (0, 0)
        ),
    }

    logger.info(f"Preview source caching completed: {cache_info}")
    if progress_callback:
        progress_callback("Preview sources ready!")
    return cache_info


async def generate_previews(
    num_samples: int = 10, size: int = 256, config: DotMap = None, progress_callback=None
) -> List[Tuple[float, float, "np.ndarray"]]:
    """
    Generate preview cutouts using cached sources and FITS data for fast response.

    Args:
        num_samples: Number of preview cutouts to generate
        size: Size of cutouts in pixels
        config: Configuration DotMap with processing parameters

    Returns:
        List of (ra, dec, cutout_array) tuples
    """
    logger.debug(f"generate_previews called with config: {config}")
    logger.info(f"Generating {num_samples} preview cutouts from cache")
    if progress_callback:
        progress_callback(f"Selecting {num_samples} sources for preview [1/2]...")

    # Check if cache is available and not empty
    if (
        not PreviewCache.sources_cache
        or not PreviewCache.fits_data_cache
        or not PreviewCache.fits_sets_cache
    ):
        logger.warning("No cached sources available, loading into cache first")
        clear_preview_cache()
        cache_info = await load_sources_for_previews(
            config.source_catalogue, config, progress_callback
        )

    cached_sources = PreviewCache.sources_cache
    cached_fits_data = PreviewCache.fits_data_cache

    if not cached_sources or not cached_fits_data:
        raise RuntimeError("Cache is empty - call load_sources_for_previews() first")

    # Select sources from cache using consistent seed for reproducible previews
    num_available = len(cached_sources)
    num_to_select = min(num_samples, num_available)

    if num_available > num_samples:
        preview_seed = PreviewCache.preview_seed or 42
        rng = np.random.RandomState(preview_seed)
        selected_indices = rng.choice(num_available, size=num_to_select, replace=False)
        preview_sources = [cached_sources[i] for i in selected_indices]
    else:
        preview_sources = cached_sources[:num_to_select]

    logger.info(f"Selected {len(preview_sources)} sources from cache of {num_available}")
    if progress_callback:
        progress_callback(f"Processing {len(preview_sources)} preview cutouts [2/2]...")

    # Group selected sources by FITS sets using the refactored function
    fits_set_to_sources = prepare_fits_sets_and_sources(preview_sources)
    selected_extensions = _get_selected_extensions(config)

    # Use provided config directly - override only size for preview
    preview_config = config.copy()
    preview_config.process_id = "cutana_preview"
    preview_config.target_resolution = size
    preview_config.fits_extensions = selected_extensions
    preview_config.log_level = "WARNING"  # Enable debug logging for preview
    preview_config.max_workers = 1  # Single worker for preview

    # Process cutouts using cached FITS data and the refactored processing function
    all_results = []

    for fits_set, sources_for_set in fits_set_to_sources.items():
        try:
            # Get cached FITS data for this set
            loaded_fits_data = {}
            for fits_path in fits_set:
                if fits_path in cached_fits_data:
                    loaded_fits_data[fits_path] = cached_fits_data[fits_path]
                else:
                    logger.warning(f"FITS file {fits_path} not found in cache")

            if not loaded_fits_data:
                logger.error(f"No cached FITS data available for set: {fits_set}")
                continue

            # Use the refactored processing function with cached FITS data
            batch_results = _process_sources_batch_vectorized_with_fits_set(
                sources_for_set, loaded_fits_data, preview_config, profiler=None
            )
            all_results.extend(batch_results)

        except Exception as e:
            logger.error(f"Error processing FITS set {fits_set}: {e}")
            continue

    if not all_results:
        raise RuntimeError("No valid cutouts were generated from cache")

    logger.debug(f"Processing {len(all_results)} batch results for cutout extraction")

    # Extract cutout arrays from results and format for display
    cutouts = []
    for result in all_results:
        if result:
            extracted = _extract_cutout_from_batch_result(result)
            cutouts.extend(extracted)

    logger.debug(f"Total cutouts extracted: {len(cutouts)}")

    if not cutouts:
        raise RuntimeError("No valid cutouts were generated from cache")

    logger.info(f"Generated {len(cutouts)} preview cutouts from cache successfully")
    if progress_callback:
        progress_callback("Preview cutouts ready!")
    return cutouts


def clear_preview_cache():
    """Clear the preview cache."""
    PreviewCache.sources_cache = None
    PreviewCache.fits_sets_cache = None
    PreviewCache.fits_data_cache = None
    PreviewCache.config_cache = None
    PreviewCache.preview_seed = None
    logger.info("Preview cache cleared")


def regenerate_preview_seed():
    """Generate a new preview seed for fresh sample selection."""
    new_seed = int(time.time() * 1000) % (2**31)
    PreviewCache.preview_seed = new_seed
    logger.info(f"Generated new preview seed: {new_seed}")
    return new_seed


def get_cache_status() -> Dict:
    """Get current cache status information."""
    if PreviewCache.config_cache:
        return {
            "cached": True,
            "num_sources": PreviewCache.config_cache.get("num_cached_sources", 0),
            "num_fits": PreviewCache.config_cache.get("num_cached_fits", 0),
            "cache_timestamp": PreviewCache.config_cache.get("cache_timestamp", 0),
        }
    else:
        return {"cached": False}
