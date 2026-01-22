#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Orchestrator module for Cutana - manages process spawning and delegation.

This module handles:
- Delegation of sourceIDs/fitstile to cutout processes
- Spawning processes to create cutouts in the background
- Respecting system memory limitations and CPU cores
- Progress tracking and status reporting
- Workflow resumption capability
"""

import json
import os
import select
import subprocess
import sys
import tempfile
import time
import traceback
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotmap import DotMap
from loguru import logger

from .catalogue_preprocessor import preprocess_catalogue, validate_catalogue_sample
from .catalogue_streamer import CatalogueBatchReader, CatalogueIndex, estimate_catalogue_size
from .get_default_config import save_config_toml
from .job_tracker import JobTracker
from .loadbalancer import LoadBalancer
from .progress_report import ProgressReport
from .validate_config import validate_config, validate_config_for_processing


class Orchestrator:
    """
    Main orchestrator for managing cutout creation workflows.

    Handles process spawning, resource management, and progress tracking
    while respecting system limitations.
    """

    def __init__(self, config: DotMap, status_panel=None):
        """
        Initialize the orchestrator with configuration.

        Args:
            config: Configuration DotMap containing workflow parameters
            status_panel: Optional reference to UI status panel for direct updates
        """
        # Only accept DotMaps - no conversion needed
        if not isinstance(config, DotMap):
            raise TypeError(f"Config must be DotMap, got {type(config)}")

        self.config = config
        self.status_panel = status_panel  # Reference to UI status panel for direct updates

        # Validate configuration (includes flux conversion validation)
        validate_config(self.config, check_paths=False)

        # Extract key parameters with proper defaults
        self.max_workers = self.config.max_workers

        # Logging is disabled by default (library best practice).
        # Users who want to see logs should call logger.enable("cutana")
        # and add their own handlers, or use cutana.setup_logging() explicitly.
        logger.info("Configuration validation completed successfully")

        # Process management
        self.active_processes: Dict[str, subprocess.Popen] = {}

        # Job tracking - use unified JobTracker that coordinates everything
        progress_dir = tempfile.gettempdir()
        self.job_tracker = JobTracker(progress_dir=progress_dir)

        # Load balancer for dynamic resource management
        self.load_balancer = LoadBalancer(
            progress_dir=progress_dir, session_id=self.job_tracker.session_id
        )

        # UI update tracking
        self.last_ui_update_time = 0.0
        self.ui_update_interval = 0.5  # Update UI every 0.5 seconds max

        # Stop flag to prevent new processes from being spawned after stop is requested
        self._stop_requested = False

        logger.info(f"Orchestrator initialized with max_workers={self.max_workers}")
        if self.status_panel:
            logger.info("Status panel reference provided for direct UI updates")
        logger.debug(f"Configuration: {dict(self.config)}")

    def _init_catalogue_index_and_reader(
        self, catalogue_path: str
    ) -> Tuple[CatalogueIndex, CatalogueBatchReader]:
        """
        Initialize catalogue index and batch reader for on-demand row access.

        This method builds the infrastructure for memory-efficient batch processing:
        1. Estimates catalogue size for logging/planning
        2. Validates a sample of the catalogue (first 10k rows)
        3. Builds a CatalogueIndex: lightweight mapping of FITS sets to row indices
        4. Creates a CatalogueBatchReader: reads specific rows on-demand

        Used by both Orchestrator.start_processing() and StreamingOrchestrator.init_streaming()
        to avoid code duplication.

        Memory usage: O(index_size) where index stores only row indices per FITS set,
        NOT the actual catalogue data. Actual data is loaded on-demand per batch.

        Args:
            catalogue_path: Path to catalogue file (CSV or Parquet)

        Returns:
            Tuple of (CatalogueIndex, CatalogueBatchReader)

        Raises:
            ValueError: If catalogue validation fails or file cannot be read
        """
        # Estimate catalogue size for logging
        try:
            estimated_rows = estimate_catalogue_size(catalogue_path)
            logger.info(f"Estimated catalogue size: {estimated_rows:,} rows")
        except Exception as e:
            logger.warning(f"Could not estimate catalogue size: {e}")

        # Validate a sample from the catalogue
        logger.info("Validating catalogue sample...")
        validation_errors = validate_catalogue_sample(
            catalogue_path,
            sample_size=10000,
            skip_fits_check=False,
        )
        if validation_errors:
            error_msg = "; ".join(validation_errors[:5])
            if len(validation_errors) > 5:
                error_msg += f" (and {len(validation_errors) - 5} more errors)"
            raise ValueError(f"Catalogue validation failed: {error_msg}")
        logger.info("Catalogue sample validation passed")

        # Build lightweight catalogue index
        logger.info("Building catalogue index (streaming mode)...")
        catalogue_index = CatalogueIndex.build_from_path(
            catalogue_path,
            batch_size=100000,
        )
        total_sources = catalogue_index.row_count
        logger.info(
            f"Built index: {total_sources:,} sources, "
            f"{len(catalogue_index.fits_set_to_row_indices):,} unique FITS sets"
        )

        # Log FITS set statistics
        stats = catalogue_index.get_fits_set_statistics()
        logger.info(
            f"FITS set statistics: avg {stats['avg_sources_per_set']:.1f} sources/set, "
            f"max {stats['max_sources_per_set']}, min {stats['min_sources_per_set']}"
        )

        # Initialize batch reader
        batch_reader = CatalogueBatchReader(catalogue_path)

        return catalogue_index, batch_reader

    def _send_ui_update(self, force: bool = False, completed_sources: int = None):
        """
        Send progress update directly to UI status panel if available.

        Args:
            force: Force update regardless of time since last update
            completed_sources: Use specific completed_sources value instead of recalculating
        """
        if not self.status_panel:
            return

        current_time = time.time()

        # Rate limit UI updates unless forced
        if not force and (current_time - self.last_ui_update_time) < self.ui_update_interval:
            return

        try:
            # Get progress report, passing completed_sources to avoid recalculation inconsistencies
            progress_report = self.get_progress_for_ui(completed_sources=completed_sources)

            # Send to status panel
            if hasattr(self.status_panel, "receive_status_UI_update"):
                logger.debug(
                    f"Sending UI update: {progress_report.completed_sources}/{progress_report.total_sources} sources ({progress_report.progress_percent:.1f}%)"
                )
                self.status_panel.receive_status_UI_update(progress_report)
                self.last_ui_update_time = current_time
            else:
                logger.warning("Status panel does not have receive_status_UI_update method")

        except Exception as e:
            logger.error(f"Error sending UI update: {e}")

    def _log_periodic_progress_update(
        self,
        current_time: float,
        start_time: float,
        completed_batches: int,
        total_batches: int,
        completed_sources: int = 0,
        total_sources: int = 0,
    ) -> None:
        """
        Log periodic progress update with per-process details and memory utilization.

        Args:
            current_time: Current timestamp
            start_time: Workflow start timestamp
            completed_batches: Number of completed batches
            total_batches: Total number of batches
            completed_sources: Number of completed sources (for throughput calculation)
            total_sources: Total number of sources
        """
        runtime = current_time - start_time
        progress_percent = (completed_batches / total_batches * 100) if total_batches > 0 else 0

        # Get system resources directly from LoadBalancer (which uses SystemMonitor)
        load_balancer_status = self.load_balancer.get_resource_status()
        system_resources = load_balancer_status.get("system", {})
        resource_source = system_resources.get("resource_source", "system")

        # Calculate throughput
        sources_per_second = (
            completed_sources / runtime if runtime > 0 and completed_sources > 0 else 0.0
        )
        batches_per_second = (
            completed_batches / runtime if runtime > 0 and completed_batches > 0 else 0.0
        )

        logger.info("=== PROGRESS UPDATE ===")
        logger.info(
            f"Runtime: {runtime:.1f}s | Progress: {completed_batches}/{total_batches} ({progress_percent:.1f}%)"
        )
        if completed_sources > 0:
            logger.info(
                f"Throughput: {sources_per_second:.1f} sources/sec, {batches_per_second:.1f} batches/sec"
            )

        # Get memory percentage from LoadBalancer (already calculated correctly)
        memory_percent = system_resources.get("memory_percent", 0.0)

        # Show memory utilization with limits (LoadBalancer provides values already in GB)
        memory_total_gb = system_resources.get("memory_total_gb", 0.0)
        memory_available_gb = system_resources.get("memory_available_gb", 0.0)
        memory_used_gb = memory_total_gb - memory_available_gb

        logger.info(
            f"Memory ({resource_source}): {memory_used_gb:.1f}GB used / {memory_total_gb:.1f}GB total ({memory_percent:.1f}%) | CPU: {system_resources.get('cpu_percent', 0.0):.1f}%"
        )

        # Show worker memory allocation (same metric LoadBalancer uses for decisions)
        performance_info = load_balancer_status.get("performance", {})
        worker_allocation_mb = performance_info.get("worker_allocation_mb")
        worker_remaining_mb = performance_info.get("worker_remaining_mb")

        if worker_allocation_mb is not None and worker_remaining_mb is not None:
            logger.info(
                f"Worker Memory: {worker_remaining_mb:.0f}MB remaining / {worker_allocation_mb:.0f}MB allocated"
            )
        else:
            # Fallback - calculate remaining directly from LoadBalancer
            remaining_mb = self.load_balancer._get_remaining_worker_memory()
            logger.info(f"Worker Memory: {remaining_mb:.0f}MB remaining (calculated directly)")

        logger.info(f"System Available Memory: {memory_available_gb:.1f}GB")

        # Get per-process details from job tracker
        process_details = self.job_tracker.get_process_details()

        if process_details:
            logger.info(f"Active Processes: {len(process_details)}")
            for process_id, details in process_details.items():
                runtime = details["runtime"]
                progress = details["progress_percent"]
                assigned = details["sources_assigned"]
                completed = details["completed_sources"]
                memory_mb = details.get("memory_footprint_mb")
                errors = details.get("errors")
                current_stage = details["current_stage"]

                logger.info(
                    f"  {process_id}: {completed}/{assigned} sources ({progress:.1f}%) | "
                    f"Stage: {current_stage} | Runtime: {runtime:.1f}s | Memory: {memory_mb:.0f}MB | Errors: {errors}"
                )
        else:
            logger.info("No active processes")

        logger.info("======================")

    def _spawn_cutout_process(
        self, process_id: str, source_batch: pd.DataFrame, write_to_disk: bool
    ) -> None:
        """
        Spawn a cutout process for a batch of sources using subprocess.

        Uses temporary files to avoid Windows command line length limitations.

        Args:
            process_id: Unique identifier for the process
            source_batch: DataFrame containing sources for this process
            write_to_disk: Whether to write outputs to disk (True) or keep in memory (False)
        """
        temp_files = []
        try:
            # Convert DataFrame to list of dicts for process communication
            source_list = source_batch.to_dict("records")

            # Create temporary files for large data instead of command line args
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as source_file:
                json.dump(source_list, source_file)
                source_temp_path = source_file.name
                temp_files.append(source_temp_path)

            # Prepare config for subprocess - pass as-is since validation ensures consistency
            subprocess_config = DotMap(self.config.copy(), _dynamic=False)

            # Set write_to_disk for this subprocess
            subprocess_config.write_to_disk = write_to_disk

            # Extract batch index from process_id (e.g., "cutout_process_001_unique_id" -> "001")
            batch_index = process_id.split("_")[2] if "_" in process_id else process_id
            subprocess_config.batch_index = batch_index
            subprocess_config.process_id = process_id
            subprocess_config.job_tracker_session_id = self.job_tracker.session_id

            # Save config as TOML for subprocess communication
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as config_file:
                config_temp_path = save_config_toml(subprocess_config, config_file.name)
                temp_files.append(config_temp_path)

            # Create subprocess command with temp file paths
            cmd = [
                sys.executable,
                "-m",
                "cutana.cutout_process",
                source_temp_path,
                config_temp_path,
            ]

            # Register with job tracker and verify progress file creation
            self.job_tracker.register_process(process_id, len(source_batch))

            # Verify progress file was created to avoid race condition
            max_retries = 5
            retry_delay = 0.1  # seconds
            progress_file_exists = False

            for attempt in range(max_retries):
                if self.job_tracker.has_process_progress_file(process_id):
                    progress_file_exists = True
                    logger.debug(f"Confirmed progress file exists for {process_id}")
                    break
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    time.sleep(retry_delay)
                    logger.debug(
                        f"Waiting for progress file creation, attempt {attempt + 1}/{max_retries}"
                    )

            if not progress_file_exists:
                logger.warning(f"Progress file not confirmed for {process_id}, proceeding anyway")

            # Check if in-memory mode (write_to_disk=False) for pipe communication
            write_to_disk = subprocess_config.write_to_disk

            logger.debug(f"Starting subprocess with command: {' '.join(cmd)}")
            logger.debug(f"write_to_disk={write_to_disk}")

            if not write_to_disk:
                # In-memory mode: use PIPE for stdin/stdout communication (shared memory streaming)
                # stderr still goes to file for logging
                log_dir = Path(self.config.output_dir) / "logs" / "subprocesses"
                log_dir.mkdir(parents=True, exist_ok=True)
                stderr_file = log_dir / f"{process_id}_stderr.log"

                logger.debug(
                    f"In-memory mode: using PIPE for stdin/stdout, stderr to {stderr_file}"
                )

                with open(stderr_file, "w") as stderr_f:
                    process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=stderr_f,
                        text=True,
                        bufsize=1,  # Line buffered
                    )
            else:
                # Disk mode: redirect output to files to avoid pipe deadlock
                log_dir = Path(self.config.output_dir) / "logs" / "subprocesses"
                log_dir.mkdir(parents=True, exist_ok=True)

                stdout_file = log_dir / f"{process_id}_stdout.log"
                stderr_file = log_dir / f"{process_id}_stderr.log"

                logger.debug(f"Disk mode: stdout={stdout_file}, stderr={stderr_file}")
                logger.debug(f"Temp files: {temp_files}")

                with open(stdout_file, "w") as stdout_f, open(stderr_file, "w") as stderr_f:
                    process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, text=True)

            # Store temp files with process for cleanup later
            process._temp_files = temp_files
            self.active_processes[process_id] = process

            # Update LoadBalancer with new worker count
            self.load_balancer.update_active_worker_count(len(self.active_processes))

            logger.info(
                f"Spawned subprocess {process_id} for {len(source_batch)} sources (PID: {process.pid})"
            )
            logger.debug(f"Subprocess {process_id} command executed successfully, process created")

        except Exception as e:
            # Clean up temp files on error
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

            logger.error(f"Failed to spawn subprocess {process_id}: {e}")
            self.job_tracker.record_error(
                {
                    "process_id": process_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": time.time(),
                }
            )

    def _receive_cutouts_from_shm(
        self, process_id: str, batch_number: int, timeout_seconds: int
    ) -> tuple:
        """
        Receive cutouts from subprocess via shared memory streaming.

        Reads chunks from shared memory as worker sends them via stdout pipe.
        Sends ACK after each chunk is processed. Accumulates all cutouts and metadata.

        Args:
            process_id: Process identifier
            batch_number: Batch number (0-indexed)
            timeout_seconds: Timeout for the whole operation

        Returns:
            Tuple of (all_cutouts, all_metadata)

        Raises:
            RuntimeError: If process fails or times out
        """
        process = self.active_processes.get(process_id)
        if not process:
            raise RuntimeError(f"Process {process_id} not found in active processes")

        all_cutouts = []
        all_metadata = []
        start_time = time.time()

        logger.info(f"Batch {batch_number + 1}: Receiving cutouts via shared memory")

        try:
            while True:
                # Check timeout to prevent indefinite blocking if subprocess hangs or stalls
                # Terminates the subprocess and raises error if exceeded
                if time.time() - start_time > timeout_seconds:
                    logger.error(f"Batch {batch_number + 1} timed out while receiving cutouts")
                    process.terminate()
                    process.wait(timeout=10.0)
                    self.active_processes.pop(process_id, None)
                    raise RuntimeError(f"Batch {batch_number + 1} timed out receiving cutouts")

                # Check if process is still alive
                process_exit_code = process.poll()
                if process_exit_code is not None:
                    # Process terminated - try to read any remaining output, then break
                    logger.debug(
                        f"Batch {batch_number + 1}: Subprocess terminated with code {process_exit_code}"
                    )
                    # Try one more read in case there's buffered output
                    try:
                        remaining_line = process.stdout.readline()
                        if remaining_line:
                            try:
                                msg = json.loads(remaining_line)
                                if msg.get("type") == "complete":
                                    logger.info(
                                        f"Batch {batch_number + 1}: Got completion message from terminated process"
                                    )
                            except json.JSONDecodeError:
                                pass
                    except Exception:
                        pass
                    break

                # Read line from stdout (non-blocking with timeout)
                # Use select to check if data is available (Unix-like) or just readline with timeout (Windows)
                if sys.platform == "win32":
                    # On Windows, just try to read (will block until data or process ends)
                    # The overall timeout mechanism will handle hung processes
                    line = process.stdout.readline()
                else:
                    # On Unix, use select for proper timeout
                    ready, _, _ = select.select([process.stdout], [], [], 1.0)
                    if not ready:
                        continue
                    line = process.stdout.readline()

                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON line from subprocess: {line[:100]}")
                    continue

                msg_type = msg.get("type")

                if msg_type == "chunk":
                    # Read chunk from shared memory
                    shm_name = msg["shm_name"]
                    shape = tuple(msg["shape"])
                    dtype = np.dtype(msg["dtype"])
                    chunk_metadata = msg["metadata"]

                    # Attach to shared memory
                    shm = None
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)

                        # Create numpy array from shared memory
                        chunk_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

                        # Copy data out of shared memory (important!)
                        chunk_copy = chunk_array.copy()

                        # Extract individual cutouts from chunk
                        for i in range(len(chunk_copy)):
                            all_cutouts.append(chunk_copy[i])

                        all_metadata.extend(chunk_metadata)

                        logger.debug(
                            f"Batch {batch_number + 1}: Received chunk with {len(chunk_copy)} cutouts "
                            f"({msg['nbytes'] / 1024 / 1024:.1f}MB)"
                        )

                    finally:
                        # Close shared memory (worker will unlink it after ACK)
                        if shm is not None:
                            shm.close()

                    # Send ACK to worker so it can cleanup and proceed
                    process.stdin.write("ACK\n")
                    process.stdin.flush()

                elif msg_type == "complete":
                    total_cutouts = msg.get("total_cutouts", 0)
                    logger.info(
                        f"Batch {batch_number + 1}: Received completion message "
                        f"({total_cutouts} total cutouts)"
                    )
                    break

                else:
                    logger.warning(f"Unknown message type from subprocess: {msg_type}")

            # Wait for process to finish
            return_code = process.wait(timeout=10.0)

            if return_code != 0:
                logger.error(f"Batch {batch_number + 1} subprocess exited with code {return_code}")
                raise RuntimeError(
                    f"Batch {batch_number + 1} subprocess failed with code {return_code}"
                )

            # Clean up process from active list
            self.active_processes.pop(process_id, None)

            logger.info(
                f"Batch {batch_number + 1}: Successfully received {len(all_cutouts)} cutouts "
                f"via shared memory (subprocess completed)"
            )

            # Stack all cutouts into single numpy array
            if all_cutouts:
                all_cutouts = np.stack(all_cutouts)
                logger.debug(f"Stacked cutouts into array with shape: {all_cutouts.shape}")

            return all_cutouts, all_metadata

        except Exception as e:
            # Cleanup on error
            if process_id in self.active_processes:
                try:
                    process.terminate()
                    process.wait(timeout=10.0)
                except Exception:
                    pass
                self.active_processes.pop(process_id, None)

            logger.error(f"Error receiving cutouts from shared memory: {e}")
            raise

    def _monitor_processes(self, timeout_seconds: int) -> List[Dict[str, Any]]:
        """
        Monitor active subprocesses and clean up completed ones.

        Simplified version that gets status from job_tracker instead of parsing output files.

        Args:
            timeout_seconds: Maximum time to wait for a process before considering it hung

        Returns:
            List of dictionaries containing process completion info
        """
        completed_processes = []
        current_time = time.time()

        logger.debug(f"Monitoring {len(self.active_processes)} active processes")

        for process_id, process in list(self.active_processes.items()):
            # Check process start time to detect timeouts
            start_time = self.job_tracker.get_process_start_time(process_id)
            if start_time is None:
                start_time = current_time  # Fallback if not found
            runtime = current_time - start_time
            sources_assigned = self.job_tracker.get_sources_assigned_to_process(process_id)

            # Check if subprocess has completed
            return_code = process.poll()
            logger.debug(
                f"Process {process_id}: poll() returned {return_code}, runtime: {runtime:.1f}s"
            )

            # Handle timeout - kill hung processes
            if return_code is None and runtime > timeout_seconds:
                logger.error(f"Subprocess {process_id} timed out after {runtime:.1f}s, terminating")
                completed_processes.append(
                    {"process_id": process_id, "successful": False, "reason": "timeout"}
                )

                try:
                    process.terminate()
                    try:
                        process.wait(timeout=10.0)  # Wait up to 10 seconds for graceful termination
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing hung subprocess {process_id}")
                        process.kill()
                        process.wait()

                    self.job_tracker.complete_process(process_id, 0, sources_assigned)
                    self.job_tracker.record_error(
                        {
                            "process_id": process_id,
                            "error_type": "ProcessTimeout",
                            "error_message": f"Process hung for {runtime:.1f}s and was terminated",
                            "timestamp": current_time,
                        }
                    )

                except Exception as e:
                    logger.error(f"Error terminating hung subprocess {process_id}: {e}")
                    self.job_tracker.complete_process(process_id, 0, sources_assigned)

                # Clean up temp files
                if hasattr(process, "_temp_files"):
                    for temp_file in process._temp_files:
                        try:
                            os.unlink(temp_file)
                        except OSError:
                            pass

                # Clean up (use pop to avoid KeyError if already removed by stop_processing)
                self.active_processes.pop(process_id, None)

                # Update LoadBalancer with new worker count
                self.load_balancer.update_active_worker_count(len(self.active_processes))
                continue

            if return_code is not None:
                # Process has completed normally
                process_completed_info = {
                    "process_id": process_id,
                    "successful": False,
                    "reason": "unknown",
                }

                try:
                    # Wait for process to finish
                    process.wait(timeout=5.0)

                    # Get final status from job tracker instead of parsing files
                    process_details = self.job_tracker.get_process_details()
                    process_detail = process_details.get(process_id)

                    # sources_assigned already retrieved above

                    if return_code == 0:
                        # Success - get completion info from job tracker
                        if process_detail:
                            completed_sources = process_detail["completed_sources"]
                            failed_sources = max(0, sources_assigned - completed_sources)
                        else:
                            # Process completed but no detail in job tracker, assume all completed
                            completed_sources = sources_assigned
                            failed_sources = 0

                        self.job_tracker.complete_process(
                            process_id, completed_sources, failed_sources
                        )
                        logger.info(
                            f"Subprocess {process_id} completed successfully: "
                            f"{completed_sources} processed, {failed_sources} failed"
                        )

                        process_completed_info["successful"] = completed_sources > 0
                        process_completed_info["reason"] = "completed"
                    else:
                        # Process failed
                        logger.error(
                            f"Subprocess {process_id} failed with return code {return_code}"
                        )
                        self.job_tracker.complete_process(process_id, 0, sources_assigned)
                        self.job_tracker.record_error(
                            {
                                "process_id": process_id,
                                "error_type": "ProcessError",
                                "error_message": f"Process exited with code {return_code}",
                                "timestamp": current_time,
                            }
                        )
                        process_completed_info["successful"] = False
                        process_completed_info["reason"] = f"exit_code_{return_code}"

                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout waiting for subprocess {process_id}")
                    self.job_tracker.complete_process(process_id, 0, sources_assigned)
                    process_completed_info["successful"] = False
                    process_completed_info["reason"] = "output_timeout"

                # Add to completed processes list
                completed_processes.append(process_completed_info)

                # Clean up temp files
                if hasattr(process, "_temp_files"):
                    for temp_file in process._temp_files:
                        try:
                            os.unlink(temp_file)
                        except OSError:
                            pass

                # Clean up (use pop to avoid KeyError if already removed by stop_processing)
                self.active_processes.pop(process_id, None)

                # Update LoadBalancer with new worker count
                self.load_balancer.update_active_worker_count(len(self.active_processes))

        return completed_processes

    def _write_source_mapping_parquet(self, output_dir: Path) -> str:
        """
        Write Parquet file mapping source IDs to their zarr file locations.

        Args:
            output_dir: Output directory where Parquet should be written

        Returns:
            Path to created Parquet file
        """
        try:
            if not hasattr(self, "source_to_batch_mapping") or not self.source_to_batch_mapping:
                logger.warning("No source-to-batch mapping available, skipping Parquet creation")
                return None

            parquet_path = output_dir / "source_to_zarr_mapping.parquet"

            # Create DataFrame and write to Parquet
            import pandas as pd

            df = pd.DataFrame(self.source_to_batch_mapping)
            df.to_parquet(parquet_path, index=False)

            logger.info(
                f"Created source mapping Parquet file with {len(df)} entries at: {parquet_path}"
            )
            return str(parquet_path)

        except Exception as e:
            logger.error(f"Failed to write source mapping Parquet file: {e}")
            return None

    def start_processing(self, catalogue_path: str) -> Dict[str, Any]:
        """
        Start the main cutout processing workflow using streaming catalogue loading.

        Uses memory-efficient streaming for catalogues of any size (supports 10M+ sources):
        1. Builds a lightweight index (FITS set to row indices mapping)
        2. Validates a sample of the catalogue
        3. Reads only the specific rows needed for each batch on-demand

        Memory usage: O(index_size) + O(batch_size) instead of O(catalogue_size)

        Args:
            catalogue_path: Path to catalogue file (CSV or Parquet)

        Returns:
            Dictionary containing workflow results and status
        """
        # Validate configuration for processing
        try:
            validate_config_for_processing(self.config, check_paths=False)
            logger.info("Configuration validation for processing completed successfully")
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return {
                "status": "failed",
                "error": f"Configuration validation failed: {e}",
                "error_type": "config_validation_error",
            }

        # Ensure output directory exists
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            return {
                "status": "failed",
                "error": f"Failed to create output directory: {e}",
                "error_type": "output_directory_error",
            }

        # Initialize streaming infrastructure (validates catalogue, builds index, creates reader)
        try:
            catalogue_index, batch_reader = self._init_catalogue_index_and_reader(catalogue_path)
            total_sources = catalogue_index.row_count
        except ValueError as e:
            logger.error(str(e))
            return {
                "status": "failed",
                "error": str(e),
                "error_type": "streaming_init_error",
            }
        except Exception as e:
            logger.error(f"Failed to initialize streaming infrastructure: {e}")
            return {
                "status": "failed",
                "error": f"Failed to initialize streaming infrastructure: {e}",
                "error_type": "streaming_init_error",
            }

        logger.info(f"Starting streaming cutout processing for {total_sources:,} sources")

        # Update config with load balancer recommendations
        self.load_balancer.update_config_with_loadbalancing(self.config, total_sources)

        # Apply load balancer settings
        if self.config.loadbalancer.max_workers != self.config.max_workers:
            logger.info(f"LoadBalancer updated max_workers: {self.config.loadbalancer.max_workers}")
            self.config.max_workers = self.config.loadbalancer.max_workers

        logger.info(
            f"LoadBalancer updated max_sources_per_process: "
            f"{self.config.loadbalancer.max_sources_per_process}"
        )

        self.config.N_batch_cutout_process = self.config.loadbalancer.N_batch_cutout_process
        self.config.memory_limit_gb = self.config.loadbalancer.memory_limit_gb

        # Initialize job tracking
        self.job_tracker.start_job(total_sources)

        # Get optimized batch ranges from the index (lightweight - just row indices)
        # This does NOT load any actual catalogue data yet
        batch_ranges = catalogue_index.get_optimized_batch_ranges(
            max_sources_per_batch=self.config.loadbalancer.max_sources_per_process,
            min_sources_per_batch=500,
            max_fits_sets_per_batch=50,
        )
        total_batches = len(batch_ranges)
        logger.info(
            f"Created {total_batches} optimized batch ranges from "
            f"{len(catalogue_index.fits_set_to_row_indices)} unique FITS sets"
        )

        # Track source to batch mapping
        self.source_to_batch_mapping = []

        try:
            batch_index = 0
            completed_batches = 0
            max_workflow_time = self.config.max_workflow_time_seconds
            workflow_start_time = time.time()
            consecutive_failures = 0
            max_consecutive_failures = 5
            last_progress_update = workflow_start_time
            progress_update_interval = 5.0

            # Send initial UI update
            self._send_ui_update(force=True)

            while (
                batch_index < total_batches or self.active_processes
            ) and not self._stop_requested:
                current_time = time.time()

                # Update LoadBalancer memory tracking
                try:
                    self.load_balancer.update_memory_tracking()
                    self.load_balancer.log_memory_status_if_needed()
                except Exception as e:
                    logger.debug(f"LoadBalancer update error: {e}")

                # Check workflow timeout
                if current_time - workflow_start_time > max_workflow_time:
                    logger.error(
                        f"Workflow timeout after {max_workflow_time}s, "
                        "terminating remaining processes"
                    )
                    self.stop_processing()
                    break

                # Check for too many consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        f"Too many consecutive failures ({consecutive_failures}), aborting workflow"
                    )
                    self.stop_processing()
                    break

                # Periodic progress updates
                if current_time - last_progress_update >= progress_update_interval:
                    status = self.job_tracker.get_status()
                    completed_sources = status.get("completed_sources")
                    self._log_periodic_progress_update(
                        current_time,
                        workflow_start_time,
                        completed_batches,
                        total_batches,
                        completed_sources,
                        total_sources,
                    )
                    last_progress_update = current_time
                    self._send_ui_update(completed_sources=completed_sources)
                else:
                    self._send_ui_update()

                # Use load balancer to decide if we can spawn new processes
                if batch_index < total_batches and not self._stop_requested:
                    pending_batches = total_batches - batch_index
                    recommendation = self.load_balancer.get_spawn_recommendation(
                        self.active_processes, pending_batches
                    )

                    if recommendation["spawn_new"] and not self._stop_requested:
                        import uuid

                        unique_id = str(uuid.uuid4())[:8]
                        process_id = f"cutout_process_{batch_index:03d}_{unique_id}"

                        # Read batch on-demand from catalogue (true streaming)
                        # Only loads this batch's rows, not the entire catalogue
                        row_indices = batch_ranges[batch_index]
                        batch_df = batch_reader.read_rows(row_indices)
                        logger.debug(
                            f"Read batch {batch_index + 1}/{total_batches} "
                            f"({len(row_indices)} rows) on-demand"
                        )

                        # Preprocess the batch
                        batch_df = preprocess_catalogue(batch_df, self.config)

                        # Track source to zarr mapping
                        zarr_file = f"batch_{process_id}/images.zarr"
                        for _, source_row in batch_df.iterrows():
                            self.source_to_batch_mapping.append(
                                {
                                    "SourceID": source_row["SourceID"],
                                    "zarr_file": zarr_file,
                                    "batch_index": batch_index,
                                }
                            )

                        self._spawn_cutout_process(
                            process_id,
                            batch_df,
                            write_to_disk=self.config.write_to_disk,
                        )
                        batch_index += 1

                        logger.debug(f"Spawned streaming process: {recommendation['reason']}")
                    else:
                        if not self._stop_requested:
                            logger.debug(f"Load balancer: no spawn - {recommendation['reason']}")

                # Monitor existing processes
                completed_processes_info = self._monitor_processes(
                    timeout_seconds=self.config.max_workflow_time_seconds
                )

                # Track consecutive failures
                if completed_processes_info:
                    for process_info in completed_processes_info:
                        process_id = process_info.get("process_id")
                        if process_id:
                            self.load_balancer.update_memory_statistics(process_id)

                    any_success = any(p.get("successful", False) for p in completed_processes_info)
                    if any_success:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += len(completed_processes_info)
                        logger.warning(f"Consecutive failures: {consecutive_failures}")

                    completed_batches += len(completed_processes_info)

                # Brief sleep
                if self.active_processes and not self._stop_requested:
                    time.sleep(1.0)
                elif self._stop_requested:
                    time.sleep(0.1)

            # Clean up batch reader
            batch_reader.close()

            # Send final UI update
            logger.info(f"Sending final UI update: {total_sources}/{total_sources} sources")
            self._send_ui_update(force=True, completed_sources=total_sources)

            # Write source to zarr mapping
            output_dir = Path(self.config.output_dir)
            mapping_parquet_path = self._write_source_mapping_parquet(output_dir)

            logger.info(f"Streaming cutout processing completed for {total_sources:,} sources")

            return {
                "status": "completed",
                "total_sources": total_sources,
                "completed_batches": completed_batches,
                "mapping_parquet": mapping_parquet_path,
                "streaming_mode": True,
            }

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in streaming processing workflow: {e}")
            logger.error(f"Full traceback:\n{error_traceback}")

            # Clean up batch reader on error
            try:
                batch_reader.close()
            except Exception:
                pass

            self._send_ui_update(force=True)

            return {
                "status": "failed",
                "error": str(e),
                "error_traceback": error_traceback,
                "streaming_mode": True,
            }

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress and status information.

        Returns:
            Dictionary containing progress information
        """
        return self.job_tracker.get_status()

    def get_progress_for_ui(self, completed_sources: int = None) -> ProgressReport:
        """
        Get progress information optimized for UI display.

        Returns a clean ProgressReport dataclass with all relevant information
        for the status panel to display, including LoadBalancer resource information.

        Args:
            completed_sources: Use specific completed_sources value instead of recalculating

        Returns:
            ProgressReport containing UI-relevant progress information
        """
        full_status = self.get_progress()

        # Override completed_sources if provided to avoid recalculation inconsistencies
        if completed_sources is not None:
            full_status["completed_sources"] = completed_sources
            # Recalculate progress_percent based on consistent total_sources
            if full_status.get("total_sources", 0) > 0:
                full_status["progress_percent"] = (
                    completed_sources / full_status["total_sources"]
                ) * 100.0
            else:
                full_status["progress_percent"] = 0.0

        # Get LoadBalancer resource status for enhanced UI display
        logger.debug("Orchestrator: Getting resource status from LoadBalancer")
        load_balancer_status = self.load_balancer.get_resource_status()
        system_info = load_balancer_status.get("system", {})
        limits_info = load_balancer_status.get("limits", {})
        performance_info = load_balancer_status.get("performance", {})
        logger.debug(
            f"Orchestrator: Received LoadBalancer status - System: CPU {system_info.get('cpu_percent')}%, Memory {system_info.get('memory_total_gb', 0):.1f}GB, Workers: {limits_info.get('cpu_limit')}, Performance: {performance_info.get('processes_measured')} processes, Peak: {performance_info.get('peak_memory_mb', 0):.1f}MB, Avg: {performance_info.get('avg_memory_mb', 0):.1f}MB"
        )

        # Check true completion status from progress files
        completion_status = self.job_tracker.check_completion_status()

        # Use the clean factory method to create the progress report
        report = ProgressReport.from_status_components(
            full_status=full_status,
            system_info=system_info,
            limits_info=limits_info,
            performance_info=performance_info,
            completion_status=completion_status,
        )

        logger.debug(
            f"Orchestrator: Returning ProgressReport - {report.completed_sources}/{report.total_sources} sources, {report.active_processes}/{report.max_workers} workers, Memory: {report.memory_used_gb:.1f}/{report.memory_total_gb:.1f}GB"
        )
        return report

    def stop_processing(self) -> Dict[str, Any]:
        """
        Stop all active subprocesses gracefully.

        Returns:
            Dictionary containing stop operation results
        """
        logger.info("Stopping all active subprocesses...")

        # Set stop flag to prevent new processes from being spawned
        self._stop_requested = True
        logger.info("Stop flag set - no new processes will be spawned")

        stopped_processes = []
        # Create a copy of items to avoid dictionary modification during iteration
        processes_to_stop = list(self.active_processes.items())

        for process_id, process in processes_to_stop:
            try:
                process.terminate()
                try:
                    process.wait(timeout=5.0)  # Wait up to 5 seconds
                except subprocess.TimeoutExpired:
                    # Force kill if still alive
                    process.kill()
                    process.wait()

                stopped_processes.append(process_id)
                logger.info(f"Stopped subprocess {process_id}")

            except Exception as e:
                logger.error(f"Error stopping subprocess {process_id}: {e}")

        # Clear active processes
        self.active_processes.clear()

        # Update LoadBalancer that all workers stopped
        self.load_balancer.update_active_worker_count(0)

        # Reset stop flag for potential future runs
        self._stop_requested = False

        logger.info(f"Successfully stopped {len(stopped_processes)} processes")
        return {"status": "stopped", "stopped_processes": stopped_processes}

    def run(self) -> Dict[str, Any]:
        """
        Run the orchestrator main loop. Meant for backend usage to be called after orchestrator creation.

        Uses streaming catalogue loading for memory-efficient processing of large catalogues.

        Returns:
            Dict[str, Any]: The final status report after running the orchestrator.
        """
        catalogue_path = self.config.source_catalogue
        result = self.start_processing(catalogue_path)
        return result

    def cleanup(self):
        """Clean up resources."""
        logger.debug("Starting Orchestrator cleanup")

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            # Ignore errors in destructor
            pass
