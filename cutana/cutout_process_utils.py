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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotmap import DotMap
from loguru import logger

from .cutout_extraction import extract_cutouts_batch_vectorized
from .fits_dataset import prepare_fits_sets_and_sources
from .image_processor import (
    apply_normalisation,
    combine_channels,
    convert_data_type,
    resize_batch_tensor,
)
from .job_tracker import JobTracker
from .performance_profiler import ContextProfiler, PerformanceProfiler
from .system_monitor import SystemMonitor
from .validate_config import validate_channel_order_consistency


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
        import traceback

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
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")

            # Note: FITS file memory management is now handled at process level

        except Exception as e:
            logger.error(f"Failed to process FITS set {fits_set}: {e}")
            continue

    return sub_batch_results


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
    with ContextProfiler(profiler, "CutoutExtraction"):
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
        with ContextProfiler(profiler, "ImageResizing"):
            batch_cutouts = resize_batch_tensor(
                all_source_cutouts,
                target_resolution,
                interpolation,
                flux_conserved_resizing,
                pixel_scales_dict,
            )
    # Get the actual extension names in deterministic order (same as resize_batch_tensor)
    tensor_channel_names = []
    for source_cutouts_dict in all_source_cutouts.values():
        for ext_name in source_cutouts_dict.keys():
            if ext_name not in tensor_channel_names:
                tensor_channel_names.append(ext_name)

    # Validate that channel order in data matches channel_weights order (only for multi-channel)
    if len(channel_weights) > 1:
        # Use dedicated validation function

        validate_channel_order_consistency(tensor_channel_names, channel_weights)

    # Report stage: combining channels
    if process_name and job_tracker:
        _report_stage(process_name, "Combining channels", job_tracker)

    # Apply batch channel combination
    source_ids = list(all_source_cutouts.keys())
    if not config.do_only_cutout_extraction:
        with ContextProfiler(profiler, "ChannelMixing"):
            cutouts_batch = combine_channels(batch_cutouts, channel_weights)

        # Report stage: applying normalization
        if process_name and job_tracker:
            _report_stage(process_name, "Applying normalization", job_tracker)

        # Normalization
        with ContextProfiler(profiler, "Normalisation"):
            processed_cutouts_batch = apply_normalisation(cutouts_batch, config)

        # Report stage: converting data types
        if process_name and job_tracker:
            _report_stage(process_name, "Converting data types", job_tracker)

        # Data type conversion
        with ContextProfiler(profiler, "DataTypeConversion"):
            final_cutouts_batch = convert_data_type(processed_cutouts_batch, target_dtype)
    else:
        final_cutouts_batch = combine_unresized_cutouts_to_list(all_source_cutouts)

    # Report stage: finalizing metadata
    if process_name and job_tracker:
        _report_stage(process_name, "Finalizing metadata", job_tracker)

    # Metadata postprocessing - create list of metadata dicts and WCS dicts
    with ContextProfiler(profiler, "MetaDataPostprocessing"):
        # Build lookup dict once for O(1) access instead of O(n) per source
        source_lookup = {s["SourceID"]: s for s in sources_batch}
        batch_timestamp = time.time()
        n_sources = len(source_ids)

        # Pre-compute sample pixel scale for diameter_arcsec conversion
        first_sample_key = next(iter(pixel_scales_dict), None)
        first_pixel_scale = pixel_scales_dict.get(first_sample_key) if first_sample_key else None

        # Vectorized extraction of source data
        source_data_list = [source_lookup.get(sid, {}) for sid in source_ids]

        # Vectorized computation of original_cutout_size if needed
        if compute_full_wcs:
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

            # Compute sizes: prefer diameter_pixel, fallback to diameter_arcsec
            original_sizes = np.where(
                ~np.isnan(diameter_pixels),
                diameter_pixels.astype(int),
                np.where(
                    (~np.isnan(diameter_arcsecs)) & (first_pixel_scale is not None),
                    np.round(diameter_arcsecs / first_pixel_scale).astype(int),
                    0,  # Will be converted to None below
                ),
            )
        else:
            original_sizes = np.zeros(n_sources, dtype=int)

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

            # Scale pixel offsets by resize factor if resizing was applied
            # The rescaled offset is in the final output image pixel coordinates
            # (larger final image = proportionally larger offset in final pixel coords)
            if not config.do_only_cutout_extraction and orig_size is not None and orig_size > 0:
                resize_factor = config.target_resolution / orig_size
                rescaled_offset_x = extraction_offset_x * resize_factor
                rescaled_offset_y = extraction_offset_y * resize_factor
                logger.debug(
                    f"{source_id}: Offset scaling - extraction:({extraction_offset_x:.4f}, {extraction_offset_y:.4f}) "
                    f"-> rescaled:({rescaled_offset_x:.4f}, {rescaled_offset_y:.4f}) [factor={resize_factor:.2f}]"
                )
            else:
                # No resizing: offsets remain in extraction coordinates
                rescaled_offset_x = extraction_offset_x
                rescaled_offset_y = extraction_offset_y

            metadata_list.append(
                {
                    "source_id": source_id,
                    "ra": source_data.get("RA"),
                    "dec": source_data.get("Dec"),
                    "diameter_arcsec": source_data.get("diameter_arcsec"),
                    "diameter_pixel": source_data.get("diameter_pixel"),
                    "original_cutout_size": orig_size,
                    "processing_timestamp": batch_timestamp,
                    "rescaled_offset_x": rescaled_offset_x,
                    "rescaled_offset_y": rescaled_offset_y,
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
