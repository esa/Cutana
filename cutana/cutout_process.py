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
import tempfile
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotmap import DotMap
from loguru import logger

from .cutout_process_utils import (
    _process_source_sub_batch,
    _report_stage,
    _set_thread_limits_for_process,
)
from .cutout_writer_fits import write_fits_batch
from .cutout_writer_zarr import (
    append_to_zarr_archive,
    create_process_zarr_archive_initial,
    generate_process_subfolder,
)
from .fits_dataset import FITSDataset, prepare_fits_sets_and_sources
from .get_default_config import load_config_toml
from .job_tracker import JobTracker
from .logging_config import setup_logging
from .performance_profiler import ContextProfiler, PerformanceProfiler
from .system_monitor import SystemMonitor


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
    _set_thread_limits_for_process(system_monitor, config.process_threads)

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

    # Check if write_to_disk is disabled (for in-memory streaming mode)
    write_to_disk = config.write_to_disk

    # Prepare zarr output path if using zarr format and writing to disk
    zarr_output_path = None
    if config.output_format == "zarr" and write_to_disk:
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

                        # For FITS output or in-memory mode, accumulate batch results and metadata
                        if (
                            config.output_format == "fits"
                            or not write_to_disk
                            or config.do_only_cutout_extraction
                        ):
                            all_metadata.extend(batch_result["metadata"])
                            all_batch_results.append(batch_result)

                        # For Zarr output with write_to_disk, write immediately
                        if (
                            config.output_format == "zarr"
                            and write_to_disk
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

        # For in-memory mode (write_to_disk=False), return all batch results with cutouts
        if not write_to_disk:
            logger.info(
                f"{process_name}: Returning {len(all_batch_results)} batch results in memory"
            )
            return all_batch_results if all_batch_results else [{"metadata": []}]

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

        # Note: Thread limits will be set after config is loaded (see below)

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
        _set_thread_limits_for_process(main_system_monitor, config.process_threads)

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

        # Handle in-memory mode (write_to_disk=False) - stream via shared memory
        write_to_disk = config.write_to_disk
        if not write_to_disk:
            _report_stage(process_id, "Streaming cutouts via shared memory", job_tracker)
            try:
                # Use N_batch_cutout_process as chunk size (already calculated by LoadBalancer)
                # This is passed via config and respects available memory
                chunk_size = config.N_batch_cutout_process
                logger.info(
                    f"{process_id}: Using chunk size {chunk_size} for shared memory streaming"
                )
                stream_cutouts_via_shm(results, process_id, chunk_size=chunk_size)
            except Exception as e:
                logger.error(f"Failed to stream cutouts via shared memory: {e}")

        # Write output files only for FITS format (Zarr already written incrementally)
        elif results and config.output_format == "fits":
            _report_stage(process_id, "Saving FITS files to disk", job_tracker)
            with ContextProfiler(main_profiler, "FitsSaving"):
                try:
                    output_dir = Path(config.output_dir)

                    # Write individual FITS files
                    written_fits_paths = write_fits_batch(
                        results, str(output_dir), config=config, modifier=process_id
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


def stream_cutouts_via_shm(
    batch_results: List[Dict[str, Any]],
    process_id: str,
    chunk_size: int,
) -> None:
    """
    Stream cutouts to parent orchestrator via shared memory to keep them in memory (no disk I/O).

    This function enables in-memory streaming mode where cutouts are kept in memory
    from the worker process to the orchestrator without writing to disk. This is critical
    for streaming workflows where cutouts should remain in memory for immediate processing.

    Uses multiprocessing.shared_memory for OS-independent shared memory access.
    Writes cutouts to shared memory and sends metadata via stdout.
    Waits for ACK from parent before proceeding to next chunk and cleaning up.

    Args:
        batch_results: List of batch result dictionaries with 'cutouts' and 'metadata'
        process_id: Unique process ID for naming shared memory blocks
        chunk_size: Number of cutouts per chunk (from config.N_batch_cutout_process or calculated)
    """
    # Extract all cutouts and metadata
    all_cutouts = []
    all_metadata = []

    for batch_result in batch_results:
        if "cutouts" in batch_result:
            all_cutouts.extend(batch_result["cutouts"])
        if "metadata" in batch_result:
            all_metadata.extend(batch_result["metadata"])

    if not all_cutouts:
        logger.warning(f"{process_id}: No cutouts to stream")
        return

    logger.info(f"{process_id}: Streaming {len(all_cutouts)} cutouts in chunks of {chunk_size}")

    # Use process_id (includes UUID) for unique shared memory naming
    # This avoids PID reuse collisions

    # Process in chunks
    for chunk_idx in range(0, len(all_cutouts), chunk_size):
        chunk_cutouts = all_cutouts[chunk_idx : chunk_idx + chunk_size]
        chunk_metadata = all_metadata[chunk_idx : chunk_idx + chunk_size]

        # Stack cutouts into array
        chunk_array = np.stack(chunk_cutouts)

        # Create unique name for this chunk using process_id (includes UUID)
        # Format: cutana_processid_chunkidx to avoid PID reuse collisions
        shm_name = f"cutana_{process_id}_{chunk_idx}".replace("-", "_")

        shm = None
        try:
            # Create shared memory block
            nbytes = chunk_array.nbytes
            shm = shared_memory.SharedMemory(create=True, size=nbytes, name=shm_name)

            # Create numpy array backed by shared memory
            shm_array = np.ndarray(chunk_array.shape, dtype=chunk_array.dtype, buffer=shm.buf)

            # Copy data to shared memory
            shm_array[:] = chunk_array[:]

            # Send chunk metadata to parent via stdout
            metadata_msg = {
                "type": "chunk",
                "shm_name": shm_name,
                "shape": list(chunk_array.shape),
                "dtype": str(chunk_array.dtype),
                "chunk_idx": chunk_idx,
                "chunk_size": len(chunk_cutouts),
                "nbytes": nbytes,
                "metadata": chunk_metadata,
            }

            sys.stdout.write(json.dumps(metadata_msg) + "\n")
            sys.stdout.flush()

            logger.debug(
                f"{process_id}: Sent chunk {chunk_idx // chunk_size + 1} "
                f"({len(chunk_cutouts)} cutouts, {nbytes / 1024 / 1024:.1f}MB) via shm:{shm_name}"
            )

            # Wait for ACK from parent with timeout to prevent hanging
            ack_timeout = 60  # seconds
            ack = None
            try:
                import signal

                def timeout_handler(_signum, _frame):
                    raise TimeoutError("Timeout waiting for ACK from parent")

                # Use signal alarm for timeout (Unix-like systems)
                if hasattr(signal, "SIGALRM"):
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(ack_timeout)
                    try:
                        ack = sys.stdin.readline().strip()
                    finally:
                        signal.alarm(0)  # Cancel alarm
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    # Windows - no SIGALRM, just do blocking read with warning
                    # The overall subprocess timeout will handle hung processes
                    ack = sys.stdin.readline().strip()

            except TimeoutError:
                logger.error(
                    f"{process_id}: Timeout waiting for ACK from parent after {ack_timeout}s"
                )
                return  # Exit streaming, parent likely crashed

            if ack != "ACK":
                logger.error(f"{process_id}: Expected ACK, got: {ack}")

        finally:
            # Clean up shared memory chunk after parent acknowledges
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()  # Delete the shared memory block
                    logger.debug(f"{process_id}: Cleaned up shm:{shm_name}")
                except Exception as e:
                    logger.error(f"{process_id}: Failed to cleanup shm:{shm_name}: {e}")

    # Send completion message
    completion_msg = {"type": "complete", "total_cutouts": len(all_cutouts)}
    sys.stdout.write(json.dumps(completion_msg) + "\n")
    sys.stdout.flush()

    logger.info(f"{process_id}: Finished streaming {len(all_cutouts)} cutouts via shared memory")


if __name__ == "__main__":
    create_cutouts_main()
