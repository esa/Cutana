#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Shared utilities for cutout processing in Cutana.

This module provides common functions used by both regular and streaming cutout processing:
- Thread limit management
- Progress stage reporting
- Sub-batch processing logic
- Vectorized cutout processing with FITS sets
"""

import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from dotmap import DotMap
from loguru import logger

from .catalogue_preprocessor import parse_fits_file_paths
from .cutout_extraction import extract_cutouts_batch_vectorized
from .fits_dataset import prepare_fits_sets_and_sources
from .image_processor import (
    apply_normalisation,
    combine_channels,
    resize_batch_tensor,
)
from .job_tracker import JobTracker
from .performance_profiler import ContextProfiler, PerformanceProfiler
from .profiling_types import Stage
from .system_monitor import SystemMonitor


def _extract_tile_basename(fits_file_paths: Any) -> str:
    """
    Extract the tile basename(s) from a source's fits_file_paths field.

    For single-channel sources this returns the FITS filename (basename). Multi-channel
    sources return a comma-separated string of all channel filenames so the exact
    tile set is preserved in the per-source metadata.

    Args:
        fits_file_paths: Raw fits_file_paths value from the source record. Accepts
            the string-encoded list (e.g. "['a.fits', 'b.fits']"), a list, or a bare
            path string.

    Returns:
        Comma-separated basenames of the source's FITS files.

    Raises:
        ValueError: If ``fits_file_paths`` is None, malformed, or parses to an empty
            list. Every source must have at least one valid FITS file path — a
            missing or empty value is a broken invariant, not a recoverable state.
    """
    if fits_file_paths is None:
        raise ValueError("fits_file_paths is required for tile metadata extraction")
    paths = parse_fits_file_paths(fits_file_paths, normalize=False)
    if not paths:
        raise ValueError(f"fits_file_paths parsed to empty list: {fits_file_paths!r}")
    return ",".join(os.path.basename(p) for p in paths)


def _set_thread_limits_for_process(system_monitor=None, thread_override=None):
    """
    Set thread limits for the current process to use only 1/4 of available cores.

    This limits various threading libraries to prevent each cutout process from
    using all available cores, which could overwhelm the system when running
    multiple parallel processes.

    Args:
        system_monitor: SystemMonitor instance to reuse, creates new one if None
        thread_override: Optional manual override for thread count (from config.process_threads)
    """
    try:
        if system_monitor is None:
            system_monitor = SystemMonitor()
        available_cores = system_monitor.get_effective_cpu_count()

        # Use override if provided, otherwise use 1/4 of available cores
        if thread_override is not None:
            process_threads = max(1, thread_override)
            logger.info(f"Using manual thread override: {process_threads} threads")
        else:
            process_threads = max(1, available_cores // 4)

        # Set environment variables for various threading libraries
        thread_env_vars = {
            "OMP_NUM_THREADS": str(process_threads),
            "MKL_NUM_THREADS": str(process_threads),
            "OPENBLAS_NUM_THREADS": str(process_threads),
            "NUMBA_NUM_THREADS": str(process_threads),
            "VECLIB_MAXIMUM_THREADS": str(process_threads),
            "NUMEXPR_NUM_THREADS": str(process_threads),
        }

        for var, value in thread_env_vars.items():
            os.environ[var] = value

        logger.info(
            f"Set thread limits for cutout process: {process_threads} threads "
            f"(from {available_cores} available cores)"
        )

    except Exception as e:
        logger.warning(f"Failed to set thread limits: {e}")


def _report_stage(process_name: str, stage: str, job_tracker: JobTracker) -> None:
    """
    Report current processing stage to job tracker.

    Args:
        process_name: Process identifier
        stage: Current processing stage
        job_tracker: JobTracker instance to use for reporting
    """
    if not job_tracker.update_process_stage(process_name, stage):
        logger.error(f"{process_name}: Failed to update stage to '{stage}'")
    else:
        logger.debug(f"{process_name}: Stage updated to '{stage}'")


def _process_source_sub_batch(
    source_sub_batch: List[Dict[str, Any]],
    loaded_fits_data: Dict[str, tuple],
    config: DotMap,
    profiler: PerformanceProfiler,
    process_name: str,
    job_tracker: JobTracker,
    sources_completed_so_far: int = 0,
    system_monitor: SystemMonitor = None,
) -> List[Dict[str, Any]]:
    """
    Process a sub-batch of sources using pre-loaded FITS data from process cache.

    Uses pre-loaded FITS data to avoid redundant file loading across sub-batches.

    Args:
        source_sub_batch: List of source dictionaries for this sub-batch
        loaded_fits_data: Pre-loaded FITS data from process cache
        config: Configuration DotMap
        profiler: Performance profiler instance
        process_name: Name of the process for logging
        job_tracker: JobTracker instance for reporting stages
        sources_completed_so_far: Number of sources completed before this sub-batch
        system_monitor: SystemMonitor instance for memory tracking


    Returns:
        List of results for sources in this sub-batch
    """
    # Report stage: organizing sources by FITS sets
    _report_stage(process_name, "Processing FITS set sources", job_tracker)

    # Group sources by their FITS file sets (should be mostly 1 set per sub-batch now)
    fits_set_to_sources = prepare_fits_sets_and_sources(source_sub_batch)

    logger.debug(
        f"Sub-batch processing {len(fits_set_to_sources)} unique FITS file sets for {len(source_sub_batch)} sources using pre-loaded FITS data"
    )

    # Note: FITS data is now pre-loaded and passed in via loaded_fits_data parameter

    # Report stage: starting source processing
    _report_stage(process_name, f"Processing {len(source_sub_batch)} sources", job_tracker)

    # Report peak memory usage after FITS files are loaded (peak processing time)
    try:
        if system_monitor is None:
            system_monitor = SystemMonitor()
            logger.debug(f"{process_name}: Created new SystemMonitor for memory reporting")
        else:
            logger.debug(f"{process_name}: Reusing existing SystemMonitor for memory reporting")

        logger.debug(
            f"{process_name}: About to report peak memory usage, completed_sources={sources_completed_so_far}"
        )
        # Use centralized memory reporting function
        success = system_monitor.report_process_memory_to_tracker(
            job_tracker, process_name, sources_completed_so_far, update_type="peak"
        )
        logger.debug(f"{process_name}: Memory reporting success: {success}")
        if not success:
            logger.warning(f"{process_name}: Memory reporting returned False - check JobTracker")
    except Exception as e:
        logger.error(f"Failed to report peak memory usage: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

    # Process each FITS file set with all sources that use it
    sub_batch_results = []
    fits_sets_processed = 0
    remaining_fits_sets = list(fits_set_to_sources.items())

    for i, (fits_set, sources_for_set) in enumerate(remaining_fits_sets):
        try:
            fits_sets_processed += 1

            set_description = f"{len(fits_set)} FITS files"
            if len(fits_set) <= 3:
                set_description = ", ".join(os.path.basename(f) for f in fits_set)

            # Report stage: processing specific FITS set
            _report_stage(
                process_name,
                f"Processing FITS set {fits_sets_processed}/{len(fits_set_to_sources)} with {len(sources_for_set)} sources",
                job_tracker,
            )

            logger.debug(
                f"Processing FITS set {fits_sets_processed}/{len(fits_set_to_sources)}: [{set_description}] "
                f"with {len(sources_for_set)} sources"
            )

            # Get loaded FITS data for this set
            set_loaded_fits_data = {}
            for fits_path in fits_set:
                if fits_path in loaded_fits_data:
                    set_loaded_fits_data[fits_path] = loaded_fits_data[fits_path]

            if not set_loaded_fits_data:
                logger.error(f"No FITS files could be loaded from set: {fits_set}")
                continue

            # Report stage: extracting and processing cutouts
            _report_stage(process_name, "Extracting and processing cutouts", job_tracker)

            # Use true vectorized batch processing for all sources sharing this FITS set
            batch_results = _process_sources_batch_vectorized_with_fits_set(
                sources_for_set, set_loaded_fits_data, config, profiler, process_name, job_tracker
            )
            sub_batch_results.extend(batch_results)

            # Sample memory during processing (for even more accurate peak detection)
            try:
                if system_monitor is None:
                    system_monitor = SystemMonitor()
                    logger.debug(f"{process_name}: Created new SystemMonitor for sampling")

                logger.debug(
                    f"{process_name}: About to sample memory, completed_sources={sources_completed_so_far}"
                )
                # Use centralized memory reporting function with the main job_tracker
                # At this point, we're still processing this sub-batch, so use sources_completed_so_far
                success = system_monitor.report_process_memory_to_tracker(
                    job_tracker, process_name, sources_completed_so_far, update_type="sample"
                )
                logger.debug(f"{process_name}: Memory sampling success: {success}")
            except Exception as e:
                logger.error(f"Failed to sample memory during processing: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")

            # Note: FITS file memory management is now handled at process level

        except Exception as e:
            logger.error(f"Failed to process FITS set {fits_set}: {e}")
            continue

    return sub_batch_results


class _OriginalSizes(NamedTuple):
    """Per-source original cutout sizes plus aggregated diagnostics for one batch."""

    sizes: np.ndarray  # int px per source; 0 means "no usable size" (→ None downstream)
    n_no_scale: int  # arcsec-sized sources with no pixel scale available
    n_subpixel: int  # arcsec-sized sources whose diameter rounds below one pixel


def _compute_original_sizes(
    diameter_pixels: np.ndarray,
    diameter_arcsecs: np.ndarray,
    pixel_scale: Optional[float],
) -> _OriginalSizes:
    """Resolve the per-source original cutout size (px) for metadata and resize math.

    Precedence: an explicit ``diameter_pixel`` always wins (independent of the pixel
    scale); otherwise fall back to ``round(diameter_arcsec / pixel_scale)``. Sources
    with no usable size — no diameter at all, no pixel scale for an arcsec value, or
    an arcsec value that rounds below one pixel — get 0, which the caller maps to
    ``original_cutout_size = None``. Those cutouts are still extracted upstream
    (clamped to >= 1 px); only the recorded original size is undefined.

    Sizes are assigned through boolean masks rather than ``np.where``: ``np.where``
    evaluates both branches and would cast an all-NaN array to int for every source
    lacking a column, emitting a spurious "invalid value encountered in cast"
    warning. Masking only ever casts the finite, in-mask subset, so it never fires.

    Args:
        diameter_pixels: Per-source diameter in pixels, NaN where absent.
        diameter_arcsecs: Per-source diameter in arcsec, NaN where absent.
        pixel_scale: Tile pixel scale (arcsec/px), or None when no WCS is available.

    Returns:
        An ``_OriginalSizes`` with the int size array and the counts of sources that
        ended up without a usable size, broken down by cause for one-per-batch warns.

    Raises:
        ValueError: If a degenerate pixel scale (<= 0 or non-finite) would be used
            for an arcsec→pixel conversion. A broken WCS is a hard fault, not a
            recoverable per-source condition, so it must stop the run loudly.
    """
    n_sources = diameter_pixels.shape[0]
    sizes = np.zeros(n_sources, dtype=int)

    have_pixel = ~np.isnan(diameter_pixels)
    sizes[have_pixel] = diameter_pixels[have_pixel].astype(int)

    from_arcsec = ~have_pixel & ~np.isnan(diameter_arcsecs)
    n_arcsec = int(np.count_nonzero(from_arcsec))

    if pixel_scale is None:
        # No WCS → arcsec values cannot be converted; those sources have no size.
        return _OriginalSizes(sizes, n_no_scale=n_arcsec, n_subpixel=0)

    if n_arcsec:
        if not np.isfinite(pixel_scale) or pixel_scale <= 0:
            raise ValueError(
                f"Degenerate pixel scale {pixel_scale!r} arcsec/px for {n_arcsec} "
                "arcsec-sized source(s): the tile WCS is invalid, so diameter_arcsec "
                "cannot be converted to pixels."
            )
        sizes[from_arcsec] = np.round(diameter_arcsecs[from_arcsec] / pixel_scale).astype(int)

    n_subpixel = int(np.count_nonzero(from_arcsec & (sizes == 0)))
    return _OriginalSizes(sizes, n_no_scale=0, n_subpixel=n_subpixel)


def _process_sources_batch_vectorized_with_fits_set(
    sources_batch: List[Dict[str, Any]],
    loaded_fits_data: Dict[str, tuple],
    config: DotMap,
    profiler: Optional[PerformanceProfiler] = None,
    process_name: Optional[str] = None,
    job_tracker: Optional[JobTracker] = None,
) -> List[Dict[str, Any]]:
    """
    Process a batch of sources that share the same FITS file set using vectorized operations.

    This function processes all sources in the batch simultaneously for maximum performance,
    handling both single-channel and multi-channel scenarios efficiently.

    Args:
        sources_batch: List of source dictionaries that share the same FITS file set
        loaded_fits_data: Pre-loaded FITS data dict mapping fits_path -> (hdul, wcs_dict)
        config: Configuration DotMap
        profiler: Optional performance profiler instance
        process_name: Optional process name for stage reporting
        job_tracker: Optional JobTracker for stage reporting

    Returns:
        List of processed results for the sources in the batch
        Dictionary with cutouts N_images, H, W, N_out
                    and metadata list of metadata dictionaries
    """
    fits_extensions = config.fits_extensions
    batch_results = []

    # Collect all cutouts for all sources from all FITS files using vectorized processing
    all_source_cutouts = {}  # source_id -> {channel_key: cutout}
    pixel_scales_dict = {}  # channel_key -> pixel_scale (for flux-conserved resizing)
    all_source_wcs = {}  # source_id -> {channel_key: wcs_object}
    # if output is fits then compute_full_wcs
    compute_full_wcs = config.output_format == "fits"

    # Report stage if tracker available
    if process_name and job_tracker:
        _report_stage(process_name, "Extracting cutouts from FITS data", job_tracker)

    # Track pixel offsets for each source (for accurate WCS in output)
    all_source_offsets = {}  # source_id -> {"x": offset_x, "y": offset_y}

    # Process each FITS file in the set using vectorized batch processing
    with ContextProfiler(profiler, Stage.CUTOUT_EXTRACTION):
        for fits_path, (hdul, wcs_dict) in loaded_fits_data.items():
            logger.debug(
                f"Vectorized processing {len(sources_batch)} sources from {Path(fits_path).name}"
            )

            # Extract cutouts for ALL sources at once using vectorized processing
            combined_cutouts, combined_wcs, _, pixel_scale, combined_offsets = (
                extract_cutouts_batch_vectorized(
                    sources_batch, hdul, wcs_dict, fits_extensions, config.padding_factor, config
                )
            )

            # Organize cutouts by source with channel keys for multi-channel support
            fits_basename = Path(fits_path).stem
            for source_id, source_cutouts in combined_cutouts.items():
                if source_id not in all_source_cutouts:
                    all_source_cutouts[source_id] = {}
                if source_id not in all_source_wcs:
                    all_source_wcs[source_id] = {}
                # Store pixel offsets for this source (from first FITS file that has it)
                if source_id not in all_source_offsets and source_id in combined_offsets:
                    all_source_offsets[source_id] = combined_offsets[source_id]

                # Add cutouts from this FITS file with proper channel keys
                for ext_name, cutout in source_cutouts.items():
                    channel_key = (
                        f"{fits_basename}_{ext_name}" if ext_name != "PRIMARY" else fits_basename
                    )
                    all_source_cutouts[source_id][channel_key] = cutout
                    # Track pixel scale for each channel (for flux-conserved resizing)
                    if channel_key not in pixel_scales_dict:
                        pixel_scales_dict[channel_key] = pixel_scale
                    # Preserve WCS information with the same channel key
                    if compute_full_wcs:
                        all_source_wcs[source_id][channel_key] = combined_wcs[source_id][ext_name]

    # Get processing parameters from config - all should be present from default config
    target_resolution = config.target_resolution
    if isinstance(target_resolution, int):
        target_resolution = (target_resolution, target_resolution)
    target_dtype = config.data_type
    interpolation = config.interpolation

    # Check for channel combination configuration
    channel_weights = config.channel_weights
    assert channel_weights is not None, "channel_weights must be specified in config"
    assert isinstance(channel_weights, dict), "channel_weights must be a dictionary"

    # Report stage: resizing cutouts
    if process_name and job_tracker:
        _report_stage(process_name, "Resizing cutouts", job_tracker)

    # Get flux conservation setting from config
    flux_conserved_resizing = config.flux_conserved_resizing

    # Resize all cutouts to tensor format
    if not config.do_only_cutout_extraction:
        with ContextProfiler(profiler, Stage.IMAGE_RESIZING):
            batch_cutouts = resize_batch_tensor(
                all_source_cutouts,
                target_resolution,
                interpolation,
                flux_conserved_resizing,
                pixel_scales_dict,
            )
    # Extension names in deterministic order (same mapping resize_batch_tensor
    # uses). Every source carries the same extensions in the same order, so the
    # first source defines the tensor's column order for the whole batch.
    tensor_channel_names = (
        list(next(iter(all_source_cutouts.values()))) if all_source_cutouts else []
    )

    # Report stage: combining channels
    if process_name and job_tracker:
        _report_stage(process_name, "Combining channels", job_tracker)

    # Apply batch channel combination
    source_ids = list(all_source_cutouts.keys())
    if not config.do_only_cutout_extraction:
        with ContextProfiler(profiler, Stage.CHANNEL_MIXING):
            # Resolve weights from labels already in memory; no catalogue or FITS I/O is needed.
            cutouts_batch = combine_channels(batch_cutouts, channel_weights, tensor_channel_names)

        # Report stage: applying normalization and data type conversion
        if process_name and job_tracker:
            _report_stage(
                process_name, "Applying normalization and data type conversion", job_tracker
            )

        # Normalization and data type conversion
        with ContextProfiler(profiler, Stage.NORMALISATION):
            final_cutouts_batch = apply_normalisation(cutouts_batch, config)
    else:
        final_cutouts_batch = combine_unresized_cutouts_to_list(all_source_cutouts)

    # Report stage: finalizing metadata
    if process_name and job_tracker:
        _report_stage(process_name, "Finalizing metadata", job_tracker)

    # Metadata postprocessing - create list of metadata dicts and WCS dicts
    with ContextProfiler(profiler, Stage.METADATA_POSTPROCESSING):
        # Build lookup dict once for O(1) access instead of O(n) per source
        source_lookup = {s["SourceID"]: s for s in sources_batch}
        batch_timestamp = time.time()
        n_sources = len(source_ids)

        # Pre-compute sample pixel scale for diameter_arcsec conversion
        first_sample_key = next(iter(pixel_scales_dict), None)
        first_pixel_scale = pixel_scales_dict.get(first_sample_key) if first_sample_key else None
        if first_pixel_scale is None:
            # No WCS for this batch → pixel_scale_arcsec_per_pixel is undefined for
            # every source. One notice per batch, not per source, since batches can
            # carry >1M sources. (Any arcsec→pixel sizing affected by the missing
            # scale is reported separately, with an exact count, below.)
            logger.warning(
                f"No pixel scale available for this batch of {n_sources} sources — "
                "PIXSCALE metadata (pixel_scale_arcsec_per_pixel) will be undefined."
            )

        # Vectorized extraction of source data
        source_data_list = [source_lookup.get(sid, {}) for sid in source_ids]

        # Vectorized computation of original_cutout_size is only needed when a
        # downstream consumer will actually use it:
        # - compute_full_wcs (FITS output reconstructs per-source WCS from it)
        # - resizing (pixel_scale + offset scaling use orig_size / target_resolution)
        # The two N-length list comprehensions below dominate metadata-build time
        # at >1M sources, so skip them for the do_only_cutout_extraction + zarr path.
        need_original_sizes = compute_full_wcs or not config.do_only_cutout_extraction
        if need_original_sizes:
            # Extract diameter_pixel and diameter_arcsec arrays
            diameter_pixels = np.array(
                [
                    s.get("diameter_pixel") if s.get("diameter_pixel") is not None else np.nan
                    for s in source_data_list
                ]
            )
            diameter_arcsecs = np.array(
                [
                    s.get("diameter_arcsec") if s.get("diameter_arcsec") is not None else np.nan
                    for s in source_data_list
                ]
            )

            # Prefer diameter_pixel, fall back to diameter_arcsec; fails hard on a
            # degenerate pixel scale. See _compute_original_sizes for the masking
            # rationale (avoids the spurious "invalid value encountered in cast").
            size_result = _compute_original_sizes(
                diameter_pixels, diameter_arcsecs, first_pixel_scale
            )
            original_sizes = size_result.sizes

            # Report sources that ended up without a usable original_cutout_size once
            # per cause per batch (never per source — batches can carry >1M sources).
            # The cutouts themselves are still extracted, clamped to >= 1 px upstream.
            if size_result.n_no_scale:
                logger.warning(
                    f"{size_result.n_no_scale}/{n_sources} sources are sized by "
                    "diameter_arcsec but no pixel scale is available for this batch; "
                    "their original_cutout_size is undefined (cutouts still extracted)."
                )
            if size_result.n_subpixel:
                logger.warning(
                    f"{size_result.n_subpixel}/{n_sources} sources have diameter_arcsec below "
                    f"one pixel ({first_pixel_scale:.4g} arcsec/px) and were extracted "
                    "at the 1 px minimum; their original_cutout_size is undefined."
                )
        else:
            original_sizes = np.zeros(n_sources, dtype=int)

        # Resolve the scalar target resolution used for resize_factor math. Config
        # can carry either an int or a (H, W) tuple; assume square output for the
        # purposes of offset/pixel-scale scaling.
        if isinstance(config.target_resolution, (tuple, list)):
            target_resolution_scalar = int(config.target_resolution[0])
        else:
            target_resolution_scalar = int(config.target_resolution)

        # Build metadata list and WCS list
        metadata_list = []
        wcs_list = []
        for i, source_id in enumerate(source_ids):
            source_data = source_data_list[i]
            orig_size = int(original_sizes[i]) if original_sizes[i] > 0 else None

            # Get pixel offsets for this source (in original extraction pixel coordinates)
            source_offsets = all_source_offsets.get(source_id, {"x": 0.0, "y": 0.0})
            extraction_offset_x = source_offsets.get("x", 0.0)
            extraction_offset_y = source_offsets.get("y", 0.0)
            # Integer extraction origin/size (parent-tile pixels, pre-resize), computed
            # vectorised at extraction time. Passed through so the FITS writer can build
            # the cutout WCS without recomputing world_to_pixel and the window bounds.
            extraction_origin_x = source_offsets.get("origin_x")
            extraction_origin_y = source_offsets.get("origin_y")
            extraction_size = source_offsets.get("extraction_size")

            # When resizing is applied, scale offsets and pixel scale by the same
            # resize_factor so both stay consistent with the final output coords.
            # `first_pixel_scale` is arcsec/pixel in the original FITS tile; the
            # output pixel scale shrinks by 1/resize_factor (more output pixels →
            # smaller arcsec per pixel).
            resizing_applied = (
                not config.do_only_cutout_extraction and orig_size is not None and orig_size > 0
            )
            if resizing_applied:
                resize_factor = target_resolution_scalar / orig_size
                rescaled_offset_x = extraction_offset_x * resize_factor
                rescaled_offset_y = extraction_offset_y * resize_factor
                pixel_scale_arcsec_per_pixel = (
                    float(first_pixel_scale / resize_factor)
                    if first_pixel_scale is not None
                    else None
                )
                logger.debug(
                    f"{source_id}: Offset scaling - extraction:({extraction_offset_x:.4f}, {extraction_offset_y:.4f}) "
                    f"-> rescaled:({rescaled_offset_x:.4f}, {rescaled_offset_y:.4f}) [factor={resize_factor:.2f}]"
                )
            else:
                # No resizing: offsets and pixel scale stay in original coordinates
                rescaled_offset_x = extraction_offset_x
                rescaled_offset_y = extraction_offset_y
                pixel_scale_arcsec_per_pixel = (
                    float(first_pixel_scale) if first_pixel_scale is not None else None
                )

            # Tile identifier(s): basename(s) of the FITS file(s) this source came from.
            # Multi-channel sources concatenate all tile basenames separated by a comma
            # so downstream users can recover the exact tile set.
            tile = _extract_tile_basename(source_data.get("fits_file_paths"))

            metadata_list.append(
                {
                    "source_id": source_id,
                    "ra": source_data.get("RA"),
                    "dec": source_data.get("Dec"),
                    "diameter_arcsec": source_data.get("diameter_arcsec"),
                    "diameter_pixel": source_data.get("diameter_pixel"),
                    "original_cutout_size": orig_size,
                    "pixel_scale_arcsec_per_pixel": pixel_scale_arcsec_per_pixel,
                    "tile": tile,
                    "processing_timestamp": batch_timestamp,
                    "rescaled_offset_x": rescaled_offset_x,
                    "rescaled_offset_y": rescaled_offset_y,
                    "extraction_origin_x": extraction_origin_x,
                    "extraction_origin_y": extraction_origin_y,
                    "extraction_size": extraction_size,
                }
            )
            wcs_list.append(all_source_wcs.get(source_id, {}))

            if profiler:
                profiler.record_source_processed()

        # Return single result with batch tensor, metadata list, WCS info, and channel mapping
        batch_result = {
            "cutouts": final_cutouts_batch,  # Shape: (N_sources, H, W, N_channels)
            "metadata": metadata_list,
            "wcs": wcs_list,
            "channel_names": tensor_channel_names,
        }
        batch_results = [batch_result]

    logger.info(
        f"Vectorized batch processing completed: {len(batch_results)}/{len(sources_batch)} sources successful"
    )
    return batch_results


def combine_unresized_cutouts_to_list(
    source_cutouts: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Combine unresized cutouts into a list of source dictionaries.

    Args:
        source_cutouts: Dict mapping source_id -> {channel_key: cutout}

    Returns:
        List of source dictionaries with unresized cutouts
    """
    extension_names = list(dict.fromkeys(ext for d in source_cutouts.values() for ext in d))
    combined_results = [
        np.dstack([d[ext] for ext in extension_names if ext in d.keys()])
        for d in source_cutouts.values()
    ]

    return combined_results
