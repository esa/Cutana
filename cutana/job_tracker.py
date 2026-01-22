#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Job tracker for Cutana - manages overall job progress and resource utilization.

This module handles:
- Overall job progress tracking across multiple processes
- Resource monitoring and constraint checking
- Integration with ProcessStatusReader/Writer for file operations
- Status aggregation and completion detection
- Error recording and reporting
"""

import tempfile
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from .process_status_reader import ProcessStatusReader
from .process_status_writer import ProcessStatusWriter


class JobTracker:
    """
    Tracks overall job progress and manages system resources.

    This class coordinates overall job status by:
    - Managing job-level state (total sources, completion counts)
    - Monitoring system resources (CPU, memory)
    - Delegating process-level operations to ProcessStatusReader/Writer
    - Aggregating status from multiple processes
    - Detecting job completion
    """

    def __init__(
        self,
        tracking_file: str = "job_tracking.json",
        progress_dir: str = None,
        session_id: str = None,
    ):
        """
        Initialize the job tracker.

        Args:
            tracking_file: Path to file for persisting job-level tracking data
            progress_dir: Directory for progress files (default: system temp)
            session_id: Session ID for progress file isolation (auto-generated if None)
        """
        self.tracking_file = tracking_file
        self.progress_dir = progress_dir or tempfile.gettempdir()

        # Session ID to avoid conflicts between different job runs
        import uuid

        self.session_id = session_id if session_id else str(uuid.uuid4())[:8]

        # Job-level state
        self.total_sources = 0
        self.completed_sources = 0
        self.failed_sources = 0
        self.start_time: Optional[float] = None

        # Process tracking (in-memory registry for orchestrator coordination)
        self.active_processes: Dict[str, Dict[str, Any]] = {}

        # Error tracking
        self.errors: List[Dict[str, Any]] = []

        # Process status components
        self.process_writer = ProcessStatusWriter(
            progress_dir=self.progress_dir, session_id=self.session_id
        )
        self.process_reader = ProcessStatusReader(
            progress_dir=self.progress_dir, session_id=self.session_id
        )

        # ETA smoothing
        self.eta_history = []
        self.eta_smoothing_factor = 0.3  # Lower = more smoothing

        logger.debug(f"JobTracker initialized with session ID: {self.session_id}")
        logger.debug(f"Progress files directory: {self.progress_dir}")

    def start_job(self, total_sources: int) -> None:
        """
        Start tracking a new job.

        Args:
            total_sources: Total number of sources to process
        """
        self.total_sources = total_sources
        self.completed_sources = 0
        self.failed_sources = 0
        self.start_time = time.time()
        self.active_processes.clear()
        self.errors.clear()
        # Reset ETA smoothing for new job
        self.eta_history.clear()

        logger.info(f"Started job tracking for {total_sources} sources")

    def register_process(self, process_id: str, sources_assigned: int) -> bool:
        """
        Register a new process and create its progress file.

        Args:
            process_id: Unique identifier for the process
            sources_assigned: Number of sources assigned to this process

        Returns:
            True if registration was successful
        """
        current_time = time.time()

        # Register in active processes dict for orchestrator coordination
        self.active_processes[process_id] = {
            "sources_assigned": sources_assigned,
            "completed_sources": 0,
            "status": "running",
            "start_time": current_time,
            "last_update": current_time,
        }

        # Delegate file creation to process writer
        success = self.process_writer.register_process(process_id, sources_assigned)

        if success:
            logger.debug(f"Registered process {process_id} with {sources_assigned} sources")
        else:
            # Clean up in-memory registration on failure
            if process_id in self.active_processes:
                del self.active_processes[process_id]

        return success

    def update_process_progress(self, process_id: str, progress_update: Dict[str, Any]) -> bool:
        """
        Update progress for a specific process.

        Args:
            process_id: Process identifier
            progress_update: Dictionary with progress information

        Returns:
            True if update was successful
        """
        completed_count = progress_update["completed_sources"]

        # Update in-memory tracking if process is registered
        if process_id in self.active_processes:
            self.active_processes[process_id]["completed_sources"] = completed_count
            self.active_processes[process_id]["last_update"] = time.time()

        # Delegate file update to process writer
        return self.process_writer.update_process_progress(process_id, progress_update)

    def complete_process(self, process_id: str, completed_count: int, failed_count: int) -> bool:
        """
        Mark a process as completed and update overall counters.

        Args:
            process_id: Process identifier
            completed_count: Final count of successfully processed sources
            failed_count: Count of failed sources

        Returns:
            True if completion was recorded successfully
        """
        if process_id in self.active_processes:
            # Remove from active processes tracking
            del self.active_processes[process_id]

            logger.info(
                f"Process {process_id} completed: {completed_count} success, {failed_count} failed"
            )

        # Delegate file operations to process writer
        success = self.process_writer.complete_process(process_id, completed_count, failed_count)

        # DON'T clean up progress files - let them accumulate for accurate totals
        # The orchestrator can clean them up at the end if needed

        return success

    def calculate_smoothed_eta(
        self, completed_batches: int, total_batches: int, start_time: float
    ) -> Optional[float]:
        """
        Calculate estimated time to completion with exponential smoothing.

        Args:
            completed_batches: Number of completed batches
            total_batches: Total number of batches
            start_time: Workflow start time

        Returns:
            Smoothed estimated seconds to completion, or None if not calculable
        """
        if completed_batches == 0:
            return None

        elapsed_time = time.time() - start_time
        completion_rate = completed_batches / elapsed_time  # batches per second

        if completion_rate > 0:
            remaining_batches = total_batches - completed_batches
            raw_eta = remaining_batches / completion_rate

            # Apply exponential smoothing
            if len(self.eta_history) == 0:
                # First calculation
                smoothed_eta = raw_eta
            else:
                # Exponential smoothing: new_value = α * raw + (1-α) * previous
                previous_eta = self.eta_history[-1]
                smoothed_eta = (
                    self.eta_smoothing_factor * raw_eta
                    + (1 - self.eta_smoothing_factor) * previous_eta
                )

            # Store for next calculation (limit history size)
            self.eta_history.append(smoothed_eta)
            if len(self.eta_history) > 10:  # Keep last 10 values
                self.eta_history = self.eta_history[-10:]

            return smoothed_eta

        return None

    def report_process_progress(
        self, process_id: str, completed_sources: int, total_sources: int = None
    ) -> bool:
        """
        Simplified progress reporting method for worker processes.

        Args:
            process_id: Process identifier
            completed_sources: Number of sources completed so far
            total_sources: Total sources (optional, will read from file if not provided)

        Returns:
            True if report was successful
        """
        # If total_sources is provided, use it to update the progress directly
        if total_sources is not None:
            progress_update = {
                "completed_sources": completed_sources,
                "total_sources": total_sources,
                "progress_percent": (completed_sources / total_sources * 100),
            }
            return self.process_writer.update_process_progress(process_id, progress_update)

        # Otherwise delegate to process writer's report method
        return self.process_writer.report_process_progress(process_id, completed_sources)

    def update_process_stage(self, process_id: str, stage: str) -> bool:
        """
        Update the current processing stage for a process.

        Args:
            process_id: Process identifier
            stage: Current processing stage

        Returns:
            True if update was successful
        """
        # Read current progress to get completed_sources
        process_data = self.process_reader.read_progress_file(process_id)
        if not process_data:
            logger.warning(f"Cannot update stage for {process_id} - no progress file found")
            return False

        completed_sources = process_data["completed_sources"]

        # Delegate to process writer with required completed_sources
        stage_update = {
            "completed_sources": completed_sources,
            "current_stage": stage,
            "stage_timestamp": time.time(),
        }
        return self.process_writer.update_process_progress(process_id, stage_update)

    def has_process_progress_file(self, process_id: str) -> bool:
        """
        Check if a process has a progress file.

        Args:
            process_id: Process identifier

        Returns:
            True if progress file exists
        """
        progress_file_path = self.process_writer._get_progress_file_path(process_id)
        return progress_file_path.exists()

    def get_process_start_time(self, process_id: str) -> Optional[float]:
        """
        Get the start time for a process.

        Args:
            process_id: Process identifier

        Returns:
            Process start time or None if not found
        """
        # First try to get from progress files (more accurate)
        process_data = self.process_reader.read_progress_file(process_id)
        if process_data:
            return process_data.get("start_time")

        # Fallback to in-memory data
        if process_id in self.active_processes:
            return self.active_processes[process_id].get("start_time")

        return None

    def record_error(self, error_info: Dict[str, Any]) -> None:
        """
        Record an error for tracking.

        Args:
            error_info: Dictionary containing error information
        """
        error_info["timestamp"] = error_info.get("timestamp", time.time())
        self.errors.append(error_info)

        logger.error(f"Recorded error: {error_info}")

    def get_process_details(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about active processes.

        Returns:
            Dictionary mapping process IDs to their details
        """
        # Delegate to process reader, passing in-memory active processes as fallback
        return self.process_reader.get_process_details(self.active_processes)

    def check_completion_status(self) -> Dict[str, Any]:
        """
        Check completion status by aggregating all progress files.

        Returns:
            Dictionary containing aggregated completion status
        """
        # Delegate to process reader
        return self.process_reader.check_completion_status(self.total_sources)

    def get_status(self) -> Dict[str, Any]:
        """
        Get job status information (job-level only, no system resources).

        Returns:
            Dictionary containing job progress and process information
        """
        # Get base status from process reader
        status = self.process_reader.get_aggregated_status(
            total_sources=self.total_sources,
            completed_sources=self.completed_sources,
            failed_sources=self.failed_sources,
            start_time=self.start_time,
        )

        # Override ETA with smoothed version to reduce jumping
        if self.start_time and status.get("completed_sources", 0) > 0:
            smoothed_eta = self.calculate_smoothed_eta(
                completed_batches=status.get("completed_sources", 0),
                total_batches=status.get("total_sources", 0),
                start_time=self.start_time,
            )
            if smoothed_eta is not None:
                status["eta_seconds"] = smoothed_eta
                logger.debug(
                    f"JobTracker: Applied smoothed ETA: {smoothed_eta:.1f}s (was: {status.get('eta_seconds', 'None')})"
                )

        return status

    def cleanup_stale_processes(self, timeout: int = 1800) -> List[str]:
        """
        Clean up processes that haven't updated in a while.

        Args:
            timeout: Timeout in seconds for considering a process stale

        Returns:
            List of cleaned up process IDs
        """
        # Get stale processes from file reader
        stale_processes = self.process_reader.cleanup_stale_processes(timeout)

        # Clean up from in-memory tracking
        for process_id in stale_processes:
            if process_id in self.active_processes:
                del self.active_processes[process_id]
                logger.warning(f"Cleaned up stale process from memory: {process_id}")

        return stale_processes

    def cleanup_all_progress_files(self) -> int:
        """
        Clean up all progress files for this job.

        Returns:
            Number of files cleaned up
        """
        return self.process_writer.cleanup_all_progress_files()

    def get_sources_assigned_to_process(self, process_id: str) -> int:
        """
        Get the number of sources assigned to a process.

        Args:
            process_id: Process identifier

        Returns:
            Number of sources assigned to the process
        """
        # First try to get from progress files (more accurate)
        process_data = self.process_reader.read_progress_file(process_id)
        if process_data:
            return process_data.get("total_sources", 0)

        # Fallback to in-memory data
        if process_id in self.active_processes:
            return self.active_processes[process_id].get("sources_assigned", 0)

        return 0
