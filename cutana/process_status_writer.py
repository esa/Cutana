#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Process status writer for Cutana - handles individual process progress file writing.

This module is responsible for:
- Writing individual process progress files with atomic operations
- Process registration and lifecycle management at file level
- Progress updates from worker processes
- File cleanup and session management
"""

import json
import sys
import time
import tempfile
import portalocker
import os
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ProcessStatusWriter:
    """
    Handles writing individual process progress to files with atomic operations and session isolation.

    This class provides all file-writing operations for individual process progress tracking,
    including process registration, progress updates, and cleanup.
    """

    def __init__(self, progress_dir: str = None, session_id: str = None):
        """
        Initialize the job status writer.

        Args:
            progress_dir: Directory for progress files (default: system temp)
            session_id: Session ID for progress file isolation (auto-generated if None)
        """
        self.progress_dir = Path(progress_dir) if progress_dir else Path(tempfile.gettempdir())

        # Ensure progress directory exists
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        # Session ID to avoid conflicts between different job runs
        self.session_id = session_id if session_id else f"{int(time.time())}_{os.getpid()}"

        # Clean up old progress files from previous runs
        self._cleanup_old_progress_files()

        logger.debug(f"ProcessStatusWriter initialized with session ID: {self.session_id}")
        logger.debug(f"Progress files directory: {self.progress_dir}")

    def _cleanup_old_progress_files(self, max_age_hours: int = 96) -> None:
        """
        Clean up old progress files from previous runs.

        Args:
            max_age_hours: Maximum age in hours for progress files to keep
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleanup_count = 0

            # Clean up old cutana progress files
            for progress_file in self.progress_dir.glob("cutana_progress_*.json"):
                try:
                    file_age = current_time - progress_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        progress_file.unlink()
                        cleanup_count += 1
                        logger.debug(f"Cleaned up old progress file: {progress_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up progress file {progress_file}: {e}")

            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} old progress files")

        except Exception as e:
            logger.error(f"Error during old progress files cleanup: {e}")

    def _get_progress_file_path(self, process_id: str) -> Path:
        """
        Get the progress file path for a given process ID with session isolation.

        Args:
            process_id: Unique process identifier

        Returns:
            Path to the progress file with session ID to avoid conflicts
        """
        return self.progress_dir / f"cutana_progress_{self.session_id}_{process_id}.json"

    def write_progress_file(self, process_id: str, progress_data: Dict[str, Any]) -> bool:
        """
        Write progress data to file with atomic file locking.

        Args:
            process_id: Process identifier
            progress_data: Progress data to write

        Returns:
            True if write was successful, False otherwise
        """
        progress_file = self._get_progress_file_path(process_id)

        try:
            # Write to temporary file first, then atomic move
            temp_file = progress_file.with_suffix(".tmp")

            with open(temp_file, "w") as f:
                # Use file locking for atomic operations - skip in tests
                if "pytest" not in sys.modules:
                    portalocker.lock(f, portalocker.LOCK_EX)

                json.dump(progress_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Atomic move
            temp_file.replace(progress_file)

            logger.debug(f"Updated progress file for {process_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to write progress file for {process_id}: {e}")
            # Clean up temp file if it exists
            if "temp_file" in locals() and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            return False

    def register_process(self, process_id: str, sources_assigned: int) -> bool:
        """
        Register a new process and create its initial progress file.

        Args:
            process_id: Unique identifier for the process
            sources_assigned: Number of sources assigned to this process

        Returns:
            True if registration was successful, False otherwise
        """
        current_time = time.time()

        # Create initial progress file
        initial_progress = {
            "process_id": process_id,
            "status": "starting",
            "total_sources": sources_assigned,
            "completed_sources": 0,
            "progress_percent": 0.0,
            "memory_footprint_mb": 0.0,
            "memory_footprint_samples": [],
            "current_source_id": None,
            "current_stage": "Starting",
            "last_update": current_time,
            "start_time": current_time,
            "errors": 0,
            "warnings": 0,
        }

        success = self.write_progress_file(process_id, initial_progress)
        if success:
            logger.debug(f"Registered process {process_id} with {sources_assigned} sources")
        return success

    def update_process_progress(self, process_id: str, progress_update: Dict[str, Any]) -> bool:
        """
        Update progress for a specific process.

        Args:
            process_id: Process identifier
            progress_update: Dictionary with progress information

        Returns:
            True if update was successful, False otherwise
        """
        try:
            current_time = time.time()
            completed_count = progress_update["completed_sources"]

            # Read current progress file to get existing data
            from .process_status_reader import ProcessStatusReader

            reader = ProcessStatusReader(str(self.progress_dir), self.session_id)
            current_progress = reader.read_progress_file(process_id) or {}

            # Get existing data - total_sources should come from progress_update if provided
            if "total_sources" in progress_update:
                sources_assigned = progress_update["total_sources"]
            else:
                sources_assigned = current_progress["total_sources"]  # Required, no fallback

            start_time = current_progress.get("start_time", current_time)

            # Calculate progress percentage if not provided
            if "progress_percent" in progress_update:
                progress_percent = progress_update["progress_percent"]
            else:
                # Calculate progress percentage from completed vs total
                progress_percent = completed_count / sources_assigned * 100

        except KeyError as e:
            logger.error(f"Process {process_id}: Missing required field {e} for progress update")
            return False

        # Update progress file with new data
        updated_progress = {
            "process_id": process_id,
            "status": progress_update.get("status", "running"),
            "total_sources": sources_assigned,
            "completed_sources": completed_count,
            "progress_percent": progress_percent,
            "memory_footprint_mb": progress_update.get("memory_footprint_mb", 0.0),
            "memory_footprint_samples": progress_update.get("memory_footprint_samples", []),
            "current_source_id": progress_update.get("current_source_id"),
            "current_stage": progress_update.get(
                "current_stage", current_progress["current_stage"]
            ),
            "last_update": current_time,
            "start_time": start_time,
            "errors": progress_update.get("errors", current_progress.get("errors", 0)),
            "warnings": progress_update.get("warnings", current_progress.get("warnings", 0)),
        }

        success = self.write_progress_file(process_id, updated_progress)
        if success:
            logger.debug(f"Process {process_id}: {completed_count} sources completed")
        return success

    def complete_process(self, process_id: str, completed_count: int, failed_count: int) -> bool:
        """
        Mark a process as completed with final counts.

        Args:
            process_id: Process identifier
            completed_count: Final count of successfully processed sources
            failed_count: Count of failed sources

        Returns:
            True if completion was recorded successfully, False otherwise
        """
        # Read current progress to get existing data
        from .process_status_reader import ProcessStatusReader

        reader = ProcessStatusReader(str(self.progress_dir), self.session_id)
        current_progress = reader.read_progress_file(process_id) or {}

        # Update with completion data
        completion_progress = {
            "process_id": process_id,
            "status": "completed",
            "total_sources": current_progress.get("total_sources", completed_count + failed_count),
            "completed_sources": completed_count,
            "failed_sources": failed_count,
            "progress_percent": 100.0,
            "memory_footprint_mb": current_progress.get("memory_footprint_mb", 0.0),
            "memory_footprint_samples": current_progress.get("memory_footprint_samples", []),
            "current_source_id": None,
            "last_update": time.time(),
            "start_time": current_progress.get("start_time", time.time()),
            "completion_time": time.time(),
            "errors": current_progress.get("errors", 0),
            "warnings": current_progress.get("warnings", 0),
            "current_stage": "Completed",
        }

        success = self.write_progress_file(process_id, completion_progress)
        if success:
            logger.info(
                f"Process {process_id} completed: {completed_count} success, {failed_count} failed"
            )
        return success

    def cleanup_progress_file(self, process_id: str) -> bool:
        """
        Remove progress file for completed process.

        Args:
            process_id: Process identifier

        Returns:
            True if cleanup was successful, False otherwise
        """
        progress_file = self._get_progress_file_path(process_id)

        try:
            if progress_file.exists():
                progress_file.unlink()
                logger.debug(f"Cleaned up progress file for {process_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cleanup progress file for {process_id}: {e}")
            return False

    def cleanup_all_progress_files(self) -> int:
        """
        Clean up all progress files for this session.

        Returns:
            Number of files cleaned up
        """
        cleanup_count = 0

        try:
            # Clean up session-specific progress files
            for progress_file in self.progress_dir.glob(
                f"cutana_progress_{self.session_id}_*.json"
            ):
                try:
                    progress_file.unlink()
                    cleanup_count += 1
                    logger.debug(f"Cleaned up progress file: {progress_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up progress file {progress_file}: {e}")
        except Exception as e:
            logger.error(f"Error during progress files cleanup: {e}")

        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} progress files for session {self.session_id}")

        return cleanup_count

    def report_process_progress(self, process_id: str, completed_sources: int) -> bool:
        """
        Simplified progress reporting method for cutout processes.

        This method handles the common case where worker processes need to report
        their current progress without complex data structures.

        Args:
            process_id: Process identifier
            completed_sources: Number of sources completed so far

        Returns:
            True if report was successful, False otherwise
        """
        try:
            # Read current progress to get total sources
            from .process_status_reader import ProcessStatusReader

            reader = ProcessStatusReader(str(self.progress_dir), self.session_id)

            # Retry logic to handle race condition with progress file creation
            max_retries = 3
            retry_delay = 0.5  # seconds
            total_sources = None

            for attempt in range(max_retries):
                progress_data = reader.read_progress_file(process_id)
                if progress_data:
                    total_sources = progress_data.get("total_sources", 0)
                    logger.debug(
                        f"Retrieved total_sources={total_sources} from progress file for {process_id} (attempt {attempt + 1})"
                    )
                    break

                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    logger.debug(
                        f"Progress file not found for {process_id}, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)

            if total_sources is None:
                logger.error(
                    f"Process {process_id}: Cannot report progress - no progress file found after {max_retries} attempts"
                )
                return False

            if total_sources == 0:
                logger.warning(
                    f"Total sources is 0 for process {process_id}, cannot calculate progress"
                )
                return False

            # Calculate progress percentage
            progress_percent = (
                (completed_sources / total_sources * 100) if total_sources > 0 else 0.0
            )

            # Get memory usage from system monitor
            from .system_monitor import SystemMonitor

            system_monitor = SystemMonitor()
            memory_mb = system_monitor.get_current_process_memory_mb()

            # Build progress update data
            progress_update = {
                "completed_sources": completed_sources,
                "progress_percent": progress_percent,
                "memory_footprint_mb": memory_mb,
                "memory_footprint_samples": [memory_mb],  # Single sample
                "current_source_id": None,  # Could be enhanced later if needed
                "status": "processing" if completed_sources < total_sources else "completed",
                "errors": 0,  # Could be enhanced to track actual errors
                "warnings": 0,  # Could be enhanced to track actual warnings
            }

            # Use existing update method to handle file operations
            return self.update_process_progress(process_id, progress_update)

        except Exception as e:
            logger.error(f"Failed to report progress for {process_id}: {e}")
            return False
