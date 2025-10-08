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
import subprocess
import sys
import time
import traceback
import tempfile
import os
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from loguru import logger
from dotmap import DotMap

from .logging_config import setup_logging
from .job_tracker import JobTracker
from .job_creator import JobCreator
from .loadbalancer import LoadBalancer
from .catalogue_preprocessor import (
    preprocess_catalogue,
    load_and_validate_catalogue,
    CatalogueValidationError,
)
from .get_default_config import save_config_toml
from .validate_config import validate_config, validate_config_for_processing
from .progress_report import ProgressReport


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

        # Set up logging in the output directory
        log_dir = Path(self.config.output_dir) / "logs"
        setup_logging(
            log_level=self.config.log_level,
            log_dir=str(log_dir),
            console_level=self.config.console_log_level,
            session_timestamp=self.config.session_timestamp,
        )
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

    def _calculate_eta(
        self, completed_batches: int, total_batches: int, start_time: float
    ) -> Optional[float]:
        """
        Calculate estimated time to completion using JobTracker smoothing.

        Args:
            completed_batches: Number of completed batches
            total_batches: Total number of batches
            start_time: Workflow start time

        Returns:
            Smoothed estimated seconds to completion, or None if not calculable
        """
        # Delegate to JobTracker for smoothed ETA calculation
        return self.job_tracker.calculate_smoothed_eta(completed_batches, total_batches, start_time)

    def _format_time(self, seconds: Optional[float]) -> str:
        """Format time in seconds to human readable string."""
        if seconds is None:
            return "Unknown"

        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

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

    def _delegate_sources_to_processes(self, catalogue_data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Delegate sources to worker processes using optimized job creation.

        Args:
            catalogue_data: DataFrame containing source information

        Returns:
            List of DataFrames, each representing a batch for one process
        """
        total_sources = len(catalogue_data)

        # Handle edge case of empty catalogue
        if total_sources == 0:
            logger.info("Empty catalogue, creating single empty batch")
            return [pd.DataFrame()]

        # Use JobCreator to create optimized jobs based on FITS file usage
        job_creator = JobCreator(
            max_sources_per_process=self.config.loadbalancer.max_sources_per_process
        )
        jobs = job_creator.create_jobs(catalogue_data)

        # Analyze efficiency
        efficiency = job_creator.analyze_job_efficiency(jobs)
        logger.info(
            f"Job creation efficiency: {efficiency.get('fits_load_reduction', 0):.1f}% reduction in FITS loads"
        )
        logger.info(
            f"Average FITS reuse ratio: {efficiency.get('average_fits_reuse_ratio', 0):.1f}"
        )

        logger.info(
            f"Created {len(jobs)} optimized jobs from {total_sources} sources "
            f"(max_sources_per_process: {self.config.loadbalancer.max_sources_per_process})"
        )
        return jobs

    def _spawn_cutout_process(self, process_id: str, source_batch: pd.DataFrame) -> None:
        """
        Spawn a cutout process for a batch of sources using subprocess.

        Uses temporary files to avoid Windows command line length limitations.

        Args:
            process_id: Unique identifier for the process
            source_batch: DataFrame containing sources for this process
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

            # Start subprocess with redirected output to avoid pipe deadlock
            # The subprocess writes debug logs to stderr which can fill pipes and cause deadlock
            # Instead, redirect to files so we can still debug if needed
            log_dir = Path(self.config.output_dir) / "logs" / "subprocesses"
            log_dir.mkdir(parents=True, exist_ok=True)

            stdout_file = log_dir / f"{process_id}_stdout.log"
            stderr_file = log_dir / f"{process_id}_stderr.log"

            logger.debug(f"Starting subprocess with command: {' '.join(cmd)}")
            logger.debug(f"Subprocess logs: stdout={stdout_file}, stderr={stderr_file}")
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

                # Clean up
                del self.active_processes[process_id]

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

                # Clean up
                del self.active_processes[process_id]

                # Update LoadBalancer with new worker count
                self.load_balancer.update_active_worker_count(len(self.active_processes))

        return completed_processes

    def _write_source_mapping_csv(self, output_dir: Path) -> str:
        """
        Write CSV file mapping source IDs to their zarr file locations.

        Args:
            output_dir: Output directory where CSV should be written

        Returns:
            Path to created CSV file
        """
        try:
            if not hasattr(self, "source_to_batch_mapping") or not self.source_to_batch_mapping:
                logger.warning("No source-to-batch mapping available, skipping CSV creation")
                return None

            csv_path = output_dir / "source_to_zarr_mapping.csv"

            # Create DataFrame and write to CSV
            import pandas as pd

            df = pd.DataFrame(self.source_to_batch_mapping)
            df.to_csv(csv_path, index=False)

            logger.info(f"Created source mapping CSV with {len(df)} entries at: {csv_path}")
            return str(csv_path)

        except Exception as e:
            logger.error(f"Failed to write source mapping CSV: {e}")
            return None

    def start_processing(self, catalogue_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Start the main cutout processing workflow.

        Args:
            catalogue_data: DataFrame containing source catalogue

        Returns:
            Dictionary containing workflow results and status
        """
        # Validate configuration for processing
        # Skip path checking since we're passing DataFrame directly
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

        # Ensure output directory exists now that we're starting processing
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

        # Validate and preprocess catalogue
        logger.info("Validating and preprocessing catalogue...")
        try:
            catalogue_data = preprocess_catalogue(catalogue_data, self.config)
            logger.info("Catalogue validation and preprocessing completed successfully")
        except CatalogueValidationError as e:
            logger.error(f"Catalogue validation failed: {e}")
            return {
                "status": "failed",
                "error": f"Catalogue validation failed: {e}",
                "error_type": "validation_error",
            }
        except Exception as e:
            logger.error(f"Catalogue preprocessing failed: {e}")
            return {
                "status": "failed",
                "error": f"Catalogue preprocessing failed: {e}",
                "error_type": "preprocessing_error",
            }

        total_sources = len(catalogue_data)
        logger.info(f"Starting cutout processing for {total_sources} sources")

        # Update config with load balancer recommendations
        self.load_balancer.update_config_with_loadbalancing(self.config, total_sources)

        # Apply load balancer settings if not explicitly overridden
        if self.config.loadbalancer.max_workers != self.config.max_workers:
            logger.info(f"LoadBalancer updated max_workers: {self.config.loadbalancer.max_workers}")
            self.config.max_workers = self.config.loadbalancer.max_workers

        logger.info(
            f"LoadBalancer updated max_sources_per_process: {self.config.loadbalancer.max_sources_per_process}"
        )

        self.config.N_batch_cutout_process = self.config.loadbalancer.N_batch_cutout_process
        logger.info(
            f"LoadBalancer updated N_batch_cutout_process: {self.config.N_batch_cutout_process}"
        )

        self.config.memory_limit_gb = self.config.loadbalancer.memory_limit_gb
        logger.info(f"LoadBalancer updated memory_limit_gb: {self.config.memory_limit_gb:.1f}GB")

        # Initialize job tracking
        self.job_tracker.start_job(total_sources)

        # Create batches
        batches = self._delegate_sources_to_processes(catalogue_data)

        # Track source to batch mapping for CSV output
        self.source_to_batch_mapping = []

        try:
            batch_index = 0
            completed_batches = 0
            max_workflow_time = self.config.max_workflow_time_seconds
            workflow_start_time = time.time()
            consecutive_failures = 0
            max_consecutive_failures = 5

            # Track last progress update time
            last_progress_update = workflow_start_time
            progress_update_interval = 5.0  # seconds

            # Send initial UI update
            self._send_ui_update(force=True)

            while (
                batch_index < len(batches) or self.active_processes
            ) and not self._stop_requested:
                current_time = time.time()

                # Update LoadBalancer memory tracking and logging synchronously
                try:
                    self.load_balancer.update_memory_tracking()
                    self.load_balancer.log_memory_status_if_needed()
                except Exception as e:
                    logger.debug(f"LoadBalancer update error: {e}")

                # Check overall workflow timeout
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

                # Periodic progress updates (every 5 seconds)
                if current_time - last_progress_update >= progress_update_interval:
                    # Get completed sources from job tracker
                    status = self.job_tracker.get_status()
                    completed_sources = status.get("completed_sources")
                    self._log_periodic_progress_update(
                        current_time,
                        workflow_start_time,
                        completed_batches,
                        len(batches),
                        completed_sources,
                        total_sources,
                    )
                    last_progress_update = current_time

                    # Send UI update using the same completed_sources value to avoid inconsistency
                    self._send_ui_update(completed_sources=completed_sources)
                else:
                    # Send UI update without forcing recalculation (rate-limited internally)
                    self._send_ui_update()

                # Use load balancer to decide if we can spawn new processes
                if batch_index < len(batches) and not self._stop_requested:
                    pending_batches = len(batches) - batch_index
                    recommendation = self.load_balancer.get_spawn_recommendation(
                        self.active_processes, pending_batches
                    )

                    if recommendation["spawn_new"] and not self._stop_requested:
                        # Spawn new process with unique ID
                        import uuid

                        unique_id = str(uuid.uuid4())[:8]
                        process_id = f"cutout_process_{batch_index:03d}_{unique_id}"

                        # write/ append for the zarr output mapping
                        zarr_file = f"batch_{process_id}/images.zarr"
                        for _, source_row in batches[batch_index].iterrows():
                            self.source_to_batch_mapping.append(
                                {
                                    "SourceID": source_row["SourceID"],
                                    "zarr_file": zarr_file,
                                    "batch_index": batch_index,
                                }
                            )

                        self._spawn_cutout_process(process_id, batches[batch_index])
                        batch_index += 1

                        logger.debug(
                            f"Spawned new process based on load balancer recommendation: {recommendation['reason']}"
                        )
                    else:
                        if self._stop_requested:
                            logger.debug("Stop requested - not spawning new processes")
                        else:
                            logger.debug(
                                f"Load balancer recommendation: no spawn - {recommendation['reason']}"
                            )

                # Monitor existing processes
                completed_processes_info = self._monitor_processes(
                    timeout_seconds=self.config.max_workflow_time_seconds
                )

                # Track consecutive failures using results from monitoring
                if completed_processes_info:
                    # Update load balancer memory statistics from completed processes
                    for process_info in completed_processes_info:
                        process_id = process_info.get("process_id")
                        if process_id:
                            logger.debug(
                                f"Orchestrator: Updating LoadBalancer memory statistics for completed process: {process_id}"
                            )
                            self.load_balancer.update_memory_statistics(process_id)

                    # Check if any completed processes had successful results
                    any_success = False
                    for process_info in completed_processes_info:
                        if process_info.get("successful", False):
                            any_success = True
                            break

                    if any_success:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += len(completed_processes_info)
                        logger.warning(f"Consecutive failures: {consecutive_failures}")

                    completed_batches += len(completed_processes_info)

                # Brief sleep to avoid busy waiting - reduced for more responsive UI
                if self.active_processes and not self._stop_requested:
                    time.sleep(1.0)
                elif self._stop_requested:
                    # When stop is requested, sleep briefly but check more frequently
                    time.sleep(0.1)

                # Log progress periodically
                if completed_batches > 0 and completed_batches % 5 == 0:
                    runtime = current_time - workflow_start_time
                    logger.info(
                        f"Completed {completed_batches}/{len(batches)} batches "
                        f"(runtime: {runtime:.1f}s)"
                    )

            # Send final UI update - ALWAYS force 100% completion when workflow completes
            logger.info(
                f"Sending final UI update with 100% completion: {total_sources}/{total_sources} sources"
            )
            self._send_ui_update(force=True, completed_sources=total_sources)

            # Write source to zarr mapping CSV
            output_dir = Path(self.config.output_dir)
            mapping_csv_path = self._write_source_mapping_csv(output_dir)

            logger.info(f"Cutout processing completed for {total_sources} sources")

            return {
                "status": "completed",
                "total_sources": total_sources,
                "completed_batches": completed_batches,
                "mapping_csv": mapping_csv_path,
            }

        except Exception as e:

            error_traceback = traceback.format_exc()
            logger.error(f"Error in processing workflow: {e}")
            logger.error(f"Full traceback:\n{error_traceback}")

            # Send final UI update showing error
            self._send_ui_update(force=True)

            return {
                "status": "failed",
                "error": str(e),
                "error_traceback": error_traceback,
            }

    def _parse_fits_file_paths(self, fits_paths_str: str) -> List[str]:
        """
        Parse FITS file paths from string representation.
        This is a port from job tracker
        Args:
            fits_paths_str: String containing FITS file paths (list format or single path)
        Returns:
            List of normalized FITS file paths
        """
        try:
            if isinstance(fits_paths_str, str):
                # Handle different string formats
                if fits_paths_str.startswith("[") and fits_paths_str.endswith("]"):
                    # String representation of list like "['path1', 'path2']"
                    try:
                        fits_paths = ast.literal_eval(fits_paths_str)
                    except (ValueError, SyntaxError):
                        logger.warning("Failed to parse fits_file_paths with ast.literal_eval")
                        # Fallback: try to extract paths manually
                        fits_paths = [
                            path.strip().strip("'\"")
                            for path in fits_paths_str.strip("[]").split(",")
                        ]
                else:
                    # Single path string
                    fits_paths = [fits_paths_str]
            else:
                fits_paths = fits_paths_str

            # Normalize paths to handle Windows path separators properly
            normalized_paths = [os.path.normpath(path) for path in fits_paths]
            return normalized_paths

        except Exception as e:
            logger.error(f"Error parsing FITS paths '{fits_paths_str}': {e}")
            return []

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

        Returns:
            Dict[str, Any]: The final status report after running the orchestrator.
        """

        try:
            catalogue_df = load_and_validate_catalogue(self.config.source_catalogue)
            logger.info(f"Loaded and validated catalogue with {len(catalogue_df)} sources")
        except CatalogueValidationError as e:
            logger.error(f"Catalogue validation failed: {e}")
            return {
                "status": "error",
                "error": f"Catalogue validation failed: {e}",
                "error_type": "validation_error",
            }
        result = self.start_processing(catalogue_df)
        return result

    def cleanup(self):
        """Clean up resources including logging handlers."""
        try:
            logger.debug("Starting Orchestrator cleanup")
            from .logging_config import cleanup_logging

            cleanup_logging()
        except Exception as e:
            # Use print since logger might be cleaned up
            print(f"Warning: Orchestrator cleanup failed: {e}")

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            # Ignore errors in destructor
            pass
