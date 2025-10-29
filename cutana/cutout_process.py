#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Cutout process script for Cutana - handles individual cutout creation as subprocess.

This script is designed to be run as a subprocess and provides functions for:
- FITS file loading and processing
- Cutout extraction from FITS tiles
- WCS coordinate transformation
- Error handling for missing/corrupted files
- Integration with image processor

Usage:
    Can be run as a script or imported for function use.
    Main entry point: create_cutouts_main()
"""

import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger
from dotmap import DotMap

from .image_processor import (
    resize_batch_tensor,
    apply_normalisation,
    convert_data_type,
    combine_channels,
)
from .logging_config import setup_logging
from .cutout_writer_zarr import (
    create_process_zarr_archive_initial,
    append_to_zarr_archive,
    generate_process_subfolder,
)
from .cutout_writer_fits import write_fits_batch
from .performance_profiler import PerformanceProfiler, ContextProfiler
from .cutout_extraction import (
    extract_cutouts_batch_vectorized,
)
from .get_default_config import load_config_toml
from .validate_config import validate_channel_order_consistency
from .job_tracker import JobTracker
from .system_monitor import SystemMonitor
from .fits_dataset import FITSDataset, prepare_fits_sets_and_sources


def _set_thread_limits_for_process(system_monitor=None):
    """
    Set thread limits for the current process to use only 1/4 of available cores.

    This limits various threading libraries to prevent each cutout process from
    using all available cores, which could overwhelm the system when running
    multiple parallel processes.

    Args:
        system_monitor: SystemMonitor instance to reuse, creates new one if None
    """
    try:
        if system_monitor is None:
            system_monitor = SystemMonitor()
        available_cores = system_monitor.get_effective_cpu_count()
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


def create_cutouts_batch(
    source_batch: List[Dict[str, Any]], config: DotMap, job_tracker: JobTracker
) -> List[Dict[str, Any]]:
    """
    Create cutouts for a batch of sources with optimized FITS loading and sub-batch processing.

    Groups sources by their FITS files at process level and loads each FITS file only once,
    processing sources in sub-batches to provide progress updates while maintaining FITS cache.

    Args:
        source_batch: List of source dictionaries
        config: Configuration DotMap
        job_tracker: JobTracker instance for progress reporting

    Returns:
        List of results for each source
    """
    # Create single SystemMonitor instance for this process
    system_monitor = SystemMonitor()

    # Set thread limits for this process if not already set
    _set_thread_limits_for_process(system_monitor)

    # Use process_id from config if available, fallback to PID-based name
    process_id = config.process_id
    process_name = process_id
    logger.info(f"Starting {process_name} for {len(source_batch)} sources")

    # Report initial progress and stage
    if not job_tracker.report_process_progress(process_name, 0, len(source_batch)):
        logger.error(f"{process_name}: Failed to report initial progress")

    _report_stage(process_name, "Initializing", job_tracker)

    # Get batch processing parameters
    batch_size = config.N_batch_cutout_process

    # Initialize performance profiler
    profiler = PerformanceProfiler(process_name)

    # Initialize FITS dataset for process-level caching
    _report_stage(process_name, "Organizing sources by FITS sets", job_tracker)
    fits_dataset = FITSDataset(config, profiler, job_tracker, process_name)
    fits_dataset.initialize_from_sources(source_batch)

    # Group sources by FITS sets first to process one set at a time
    fits_set_to_sources = prepare_fits_sets_and_sources(source_batch)

    # Create sub-batches organized by FITS sets, respecting batch_size limit
    sub_batches = []
    for fits_set, sources_for_set in fits_set_to_sources.items():
        # If FITS set has more sources than batch_size, split it
        if len(sources_for_set) > batch_size:
            for i in range(0, len(sources_for_set), batch_size):
                sub_batch_sources = sources_for_set[i : i + batch_size]
                sub_batches.append(sub_batch_sources)
        else:
            # Keep entire FITS set together as one sub-batch
            sub_batches.append(sources_for_set)

    logger.info(
        f"Organized {len(source_batch)} sources into {len(sub_batches)} FITS-set-based sub-batches "
        f"from {len(fits_set_to_sources)} unique FITS sets"
    )

    # Prepare zarr output path if using zarr format
    zarr_output_path = None
    if config.output_format == "zarr":
        output_dir = Path(config.output_dir)
        subfolder = generate_process_subfolder(process_id)
        zarr_output_path = output_dir / subfolder / "images.zarr"

    try:
        all_metadata = []  # Keep track of all metadata for FITS output
        all_batch_results = []  # Keep track of all batch results for FITS output
        total_processed = 0
        actual_processed_count = 0

        for batch_idx, sub_batch in enumerate(sub_batches):
            logger.info(
                f"{process_name}: Processing FITS-set-based sub-batch {batch_idx + 1}/{len(sub_batches)} "
                f"({len(sub_batch)} sources)"
            )

            # Report stage for each sub-batch
            _report_stage(
                process_name,
                f"Loading FITS files for sub-batch {batch_idx + 1}/{len(sub_batches)}",
                job_tracker,
            )

            # Prepare FITS data for this sub-batch (loads only files needed for this FITS set)
            sub_batch_fits_data = fits_dataset.prepare_sub_batch(sub_batch)

            # Process this sub-batch using cached FITS data
            sub_batch_results = _process_source_sub_batch(
                sub_batch,
                sub_batch_fits_data,
                config,
                profiler,
                process_name,
                job_tracker,
                actual_processed_count,
                system_monitor,
            )
            total_processed += len(sub_batch)

            # Write sub-batch results immediately to reduce memory footprint
            if sub_batch_results:
                for batch_result in sub_batch_results:
                    if "metadata" in batch_result:
                        actual_processed_count += len(batch_result["metadata"])

                        # For FITS output, accumulate batch results and metadata
                        if config.output_format == "fits":
                            all_metadata.extend(batch_result["metadata"])
                            all_batch_results.append(batch_result)

                        # For Zarr output, write immediately
                        if (
                            config.output_format == "zarr"
                            and batch_result.get("cutouts") is not None
                        ):
                            _report_stage(
                                process_name,
                                f"Saving sub-batch {batch_idx + 1} to zarr",
                                job_tracker,
                            )
                            with ContextProfiler(profiler, "ZarrSaving"):
                                if batch_idx == 0:
                                    # Create initial zarr archive
                                    create_process_zarr_archive_initial(
                                        batch_result, str(zarr_output_path), config
                                    )
                                    logger.info(
                                        f"{process_name}: Created initial Zarr archive at {zarr_output_path}"
                                    )
                                else:
                                    # Append to existing zarr archive
                                    append_to_zarr_archive(
                                        batch_result, str(zarr_output_path), config
                                    )
                                    logger.info(
                                        f"{process_name}: Appended sub-batch {batch_idx + 1} to Zarr archive"
                                    )

            # Clear sub_batch_results to free memory
            del sub_batch_results

            # Free FITS files that won't be used in remaining sub-batches
            fits_dataset.free_unused_after_sub_batch(sub_batch, sub_batches[batch_idx + 1 :])

            # Report progress after each sub-batch - use total_processed not actual_processed_count
            # to avoid jumping issue
            progress_percent = (total_processed / len(source_batch)) * 100
            logger.info(
                f"{process_name}: Completed FITS-set sub-batch {batch_idx + 1}/{len(sub_batches)}, "
                f"processed {total_processed}/{len(source_batch)} sources ({progress_percent:.1f}%)"
            )

            # Report progress to job tracker with total processed count
            if not job_tracker.report_process_progress(
                process_name, total_processed, len(source_batch)
            ):
                logger.error(
                    f"{process_name}: Failed to report progress ({total_processed}/{len(source_batch)} sources)"
                )

        # Log performance summary
        profiler.log_performance_summary()
        bottlenecks = profiler.get_bottlenecks()
        if bottlenecks:
            logger.warning(f"Performance bottlenecks detected: {bottlenecks}")

        # Clean up any remaining FITS files in cache
        fits_dataset.cleanup()

        # Final memory report before completion
        try:
            logger.debug(f"{process_name}: Final memory report before completion")
            success = system_monitor.report_process_memory_to_tracker(
                job_tracker, process_name, actual_processed_count, update_type="sample"
            )
            logger.info(f"{process_name}: Final memory report success: {success}")
        except Exception as e:
            logger.error(f"{process_name}: Failed final memory report: {e}")

        # Final report to ensure we're at 100% if all succeeded
        if actual_processed_count == len(source_batch):
            # Ensure final progress shows 100%
            job_tracker.report_process_progress(process_name, len(source_batch), len(source_batch))

        logger.info(
            f"{process_name} completed: {actual_processed_count} successful results from {len(source_batch)} sources"
        )

        # For FITS output, return batch results for writing individual files
        if config.output_format == "fits":
            return all_batch_results if all_batch_results else [{"metadata": []}]

        # For Zarr output, results are already written incrementally
        # Only return success indicator if we actually processed sources
        if actual_processed_count > 0:
            return [{"metadata": [{"source_id": "written_incrementally"}]}]
        else:
            return [{"metadata": []}]

    except Exception as e:
        logger.error(f"Fatal error in {process_name}: {e}")
        # Still log performance summary on error
        try:
            profiler.log_performance_summary()
        except Exception:
            pass
        return [{"metadata": []}]


def _process_source_sub_batch(
    source_sub_batch: List[Dict[str, Any]],
    loaded_fits_data: Dict[str, Tuple[fits.HDUList, Dict[str, WCS]]],
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


def create_cutouts_main():
    """
    Main entry point for subprocess execution.

    Expects command line arguments:
    1. JSON string containing source batch data
    2. JSON string containing configuration
    """
    try:
        # Create single SystemMonitor for main process
        main_system_monitor = SystemMonitor()

        # Set thread limits early to prevent library initialization with wrong settings
        _set_thread_limits_for_process(main_system_monitor)

        # Chcking system arguments
        if len(sys.argv) != 3:
            logger.error("Usage: cutout_process.py <source_batch_file> <config_file>")
            logger.error(f"Got {len(sys.argv)} arguments: {sys.argv}")
            sys.exit(1)

        # Parse command line arguments (now file paths)
        source_batch_file = sys.argv[1]
        config_file = sys.argv[2]

        # Load config as TOML and convert to DotMap
        config = load_config_toml(config_file)

        # Set up logging in the output directory
        log_level = config.log_level
        console_level = config.console_log_level
        log_dir = Path(config.output_dir) / "logs"
        setup_logging(
            log_level=log_level,
            log_dir=str(log_dir),
            colorize=False,
            console_level=console_level,
            session_timestamp=config.session_timestamp,
        )

        logger.debug(f"Source batch file: {source_batch_file}")
        logger.debug(f"Config file: {config_file}")

        # Load data from files - source batch as JSON, config as TOML
        logger.debug("Loading data from files...")
        with open(source_batch_file, "r") as f:
            source_batch = json.load(f)

        # Clean up temp files after reading
        try:
            os.unlink(source_batch_file)
            os.unlink(config_file)
        except OSError:
            pass

        logger.debug(f"Decoded {len(source_batch)} sources from JSON")

        logger.info(f"Cutout process started with {len(source_batch)} sources")
        logger.debug("Starting cutout processing...")

        # Initialize profiler for the main process
        process_id = config.process_id
        main_profiler = PerformanceProfiler(process_id)

        job_tracker = JobTracker(
            progress_dir=tempfile.gettempdir(), session_id=config.job_tracker_session_id
        )

        # Process cutouts
        results = create_cutouts_batch(source_batch, config, job_tracker)

        # Calculate actual number of sources processed from batch results
        actual_processed_count = 0
        if results and len(results) > 0:
            for batch_result in results:
                if "metadata" in batch_result:
                    # Check if it's the incremental writing marker
                    if (
                        len(batch_result["metadata"]) == 1
                        and batch_result["metadata"][0].get("source_id") == "written_incrementally"
                    ):
                        # For zarr incremental writing, all sources were processed
                        actual_processed_count = len(source_batch)
                    else:
                        actual_processed_count += len(batch_result["metadata"])

        # For zarr format, if we got here without errors, all sources were processed
        if config.output_format == "zarr" and actual_processed_count == 0:
            actual_processed_count = len(source_batch)

        # Report final completion to job tracker with actual processed count
        # This is the FINAL update that should show 100% completion
        if not job_tracker.report_process_progress(
            process_id, actual_processed_count, len(source_batch)
        ):
            logger.error(f"Failed to report final progress for process {process_id}")
        else:
            logger.info(
                f"{process_id}: Reported final progress - {actual_processed_count}/{len(source_batch)} sources"
            )

        # Write output files only for FITS format (Zarr already written incrementally)
        if results and config.output_format == "fits":
            _report_stage(process_id, "Saving FITS files to disk", job_tracker)
            with ContextProfiler(main_profiler, "FitsSaving"):
                try:
                    output_dir = Path(config.output_dir)

                    # Write individual FITS files
                    written_fits_paths = write_fits_batch(
                        results, str(output_dir), modifier=process_id
                    )
                    logger.info(
                        f"{process_id}: Created {len(written_fits_paths)} FITS files in {output_dir}"
                    )
                except Exception as e:
                    logger.error(f"Failed to write FITS files: {e}")

        # Report final stage as completed BEFORE printing output
        _report_stage(process_id, "Completed", job_tracker)

        output = {
            "processed_count": actual_processed_count,
            "total_count": len(source_batch),
        }
        print(json.dumps(output))

    except Exception as e:
        logger.error(f"Cutout process failed: {e}")
        error_output = {
            "processed_count": 0,
            "total_count": len(source_batch) if "source_batch" in locals() else 0,
            "error": str(e),
        }
        print(json.dumps(error_output))
        sys.exit(1)


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

    # Report stage if tracker available
    if process_name and job_tracker:
        _report_stage(process_name, "Extracting cutouts from FITS data", job_tracker)

    # Process each FITS file in the set using vectorized batch processing
    with ContextProfiler(profiler, "CutoutExtraction"):
        for fits_path, (hdul, wcs_dict) in loaded_fits_data.items():
            logger.debug(
                f"Vectorized processing {len(sources_batch)} sources from {Path(fits_path).name}"
            )

            # Extract cutouts for ALL sources at once using vectorized processing
            combined_cutouts, combined_wcs, processed_source_ids = extract_cutouts_batch_vectorized(
                sources_batch, hdul, wcs_dict, fits_extensions, config.padding_factor, config
            )

            # Organize cutouts by source with channel keys for multi-channel support
            fits_basename = Path(fits_path).stem
            for source_id, source_cutouts in combined_cutouts.items():
                if source_id not in all_source_cutouts:
                    all_source_cutouts[source_id] = {}

                # Add cutouts from this FITS file with proper channel keys
                for ext_name, cutout in source_cutouts.items():
                    channel_key = (
                        f"{fits_basename}_{ext_name}" if ext_name != "PRIMARY" else fits_basename
                    )
                    all_source_cutouts[source_id][channel_key] = cutout

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

    # Resize all cutouts to tensor format
    with ContextProfiler(profiler, "ImageResizing"):
        batch_cutouts = resize_batch_tensor(all_source_cutouts, target_resolution, interpolation)

    # Validate that channel order in data matches channel_weights order (only for multi-channel)
    if len(channel_weights) > 1:
        # Get the actual extension names in deterministic order (same as resize_batch_tensor)
        tensor_channel_names = []
        for source_cutouts_dict in all_source_cutouts.values():
            for ext_name in source_cutouts_dict.keys():
                if ext_name not in tensor_channel_names:
                    tensor_channel_names.append(ext_name)

        # Use dedicated validation function

        validate_channel_order_consistency(tensor_channel_names, channel_weights)

    # Report stage: combining channels
    if process_name and job_tracker:
        _report_stage(process_name, "Combining channels", job_tracker)

    # Apply batch channel combination
    source_ids = list(all_source_cutouts.keys())
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

    # Report stage: finalizing metadata
    if process_name and job_tracker:
        _report_stage(process_name, "Finalizing metadata", job_tracker)

    # Metadata postprocessing - create list of metadata dicts
    with ContextProfiler(profiler, "MetaDataPostprocessing"):
        # Build metadata list in source order (matching tensor order)
        metadata_list = []

        for source_id in source_ids:
            # Find the corresponding source data
            source_data = next((s for s in sources_batch if s["SourceID"] == source_id), {})
            metadata_dict = {
                "source_id": source_id,
                "ra": source_data.get("RA"),
                "dec": source_data.get("Dec"),
                "diameter_arcsec": source_data.get("diameter_arcsec"),
                "diameter_pixel": source_data.get("diameter_pixel"),
                "processing_timestamp": time.time(),
            }
            metadata_list.append(metadata_dict)

            if profiler:
                profiler.record_source_processed()

        # Return single result with batch tensor and metadata list
        batch_result = {
            "cutouts": final_cutouts_batch,  # Shape: (N_sources, H, W, N_channels)
            "metadata": metadata_list,
        }
        batch_results = [batch_result]

    logger.info(
        f"Vectorized batch processing completed: {len(batch_results)}/{len(sources_batch)} sources successful"
    )
    return batch_results


# Legacy function for backward compatibility with orchestrator
def create_cutouts(source_batch: List[Dict[str, Any]], config: DotMap) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility.

    Args:
        source_batch: List of source dictionaries
        config: Configuration DotMap

    Returns:
        List of results for each source
    """
    job_tracker = JobTracker(
        progress_dir=tempfile.gettempdir(), session_id=config.job_tracker_session_id
    )
    return create_cutouts_batch(source_batch, config, job_tracker)


if __name__ == "__main__":
    create_cutouts_main()
