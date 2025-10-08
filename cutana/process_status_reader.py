#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Process status reader for Cutana - handles individual process progress file reading.

This module is responsible for:
- Reading individual process progress files with error handling
- Process details retrieval from files
- Completion status detection at process level
- File-based status aggregation
"""

import json
import sys
import time
import tempfile
import portalocker
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class ProcessStatusReader:
    """
    Handles reading individual process progress from files.

    This class provides all file-reading operations for individual process progress tracking,
    including process details retrieval, completion detection, and file-based status aggregation.
    """

    def __init__(self, progress_dir: str = None, session_id: str = None):
        """
        Initialize the job status reader.

        Args:
            progress_dir: Directory for progress files (default: system temp)
            session_id: Session ID for progress file isolation (required for reading specific session)
        """
        self.progress_dir = Path(progress_dir) if progress_dir else Path(tempfile.gettempdir())
        self.session_id = session_id

        if not self.session_id:
            logger.warning(
                "ProcessStatusReader initialized without session_id - some operations may not work correctly"
            )

    def _get_progress_file_path(self, process_id: str) -> Path:
        """
        Get the progress file path for a given process ID with session isolation.

        Args:
            process_id: Unique process identifier

        Returns:
            Path to the progress file with session ID
        """
        if not self.session_id:
            raise ValueError("Session ID is required to get progress file path")
        return self.progress_dir / f"cutana_progress_{self.session_id}_{process_id}.json"

    def read_progress_file(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Read progress data from file with error handling.

        Args:
            process_id: Process identifier

        Returns:
            Progress data or None if file doesn't exist/is corrupted
        """
        progress_file = self._get_progress_file_path(process_id)

        if not progress_file.exists():
            return None

        try:
            with open(progress_file, "r") as f:
                # Use file locking for atomic read operations - skip in tests
                if "pytest" not in sys.modules:
                    portalocker.lock(f, portalocker.LOCK_SH)

                data = json.load(f)
                return data

        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            logger.warning(f"Failed to read progress file for {process_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading progress file for {process_id}: {e}")
            return None

    def get_session_progress_files(self) -> List[Path]:
        """
        Get all progress files for the current session.

        Returns:
            List of progress file paths for this session
        """
        if not self.session_id:
            logger.warning("No session ID provided - returning all progress files")
            return list(self.progress_dir.glob("cutana_progress_*.json"))

        return list(self.progress_dir.glob(f"cutana_progress_{self.session_id}_*.json"))

    def get_process_details(
        self, active_processes: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about processes from progress files.

        Args:
            active_processes: Optional dict of active process info (for fallback data)

        Returns:
            Dictionary mapping process IDs to their details
        """
        details = {}
        session_files = self.get_session_progress_files()

        for progress_file in session_files:
            try:
                # Extract process ID from filename
                process_id = progress_file.stem.replace(f"cutana_progress_{self.session_id}_", "")

                # Try to get latest data from progress file
                progress_data = self.read_progress_file(process_id)

                if progress_data:
                    # Use data from progress file - no fallback defaults to avoid corruption
                    sources_assigned = progress_data.get("total_sources")
                    completed = progress_data.get("completed_sources")
                    progress_percent = progress_data.get("progress_percent")
                    status = progress_data.get("status")
                    start_time = progress_data.get("start_time")

                    # Skip process if essential data is missing (indicates corrupted read)
                    if sources_assigned is None or completed is None or start_time is None:
                        logger.debug(
                            f"Skipping process {process_id} due to incomplete progress data"
                        )
                        continue

                    details[process_id] = {
                        "sources_assigned": sources_assigned,
                        "completed_sources": completed,
                        "progress_percent": progress_percent or 0.0,
                        "status": status or "unknown",
                        "start_time": start_time,
                        "last_update": progress_data.get("last_update") or start_time,
                        "runtime": time.time() - start_time,
                        "memory_footprint_mb": progress_data.get("memory_footprint_mb") or 0.0,
                        "current_source_id": progress_data.get("current_source_id"),
                        "errors": progress_data.get("errors") or 0,
                        "warnings": progress_data.get("warnings") or 0,
                        "current_stage": progress_data["current_stage"],
                    }
                elif active_processes and process_id in active_processes:
                    # Fall back to in-memory data if available
                    info = active_processes[process_id]
                    sources_assigned = info.get("sources_assigned")
                    completed = info.get("completed_sources")
                    start_time = info.get("start_time")

                    # Skip process if essential in-memory data is missing
                    if sources_assigned is None or completed is None or start_time is None:
                        logger.error(
                            f"Process {process_id}: Missing critical data in fallback - assigned={sources_assigned}, completed={completed}, start_time={start_time}"
                        )
                        continue

                    progress_percent = 0.0
                    if sources_assigned > 0:
                        progress_percent = (completed / sources_assigned) * 100

                    details[process_id] = {
                        "sources_assigned": sources_assigned,
                        "completed_sources": completed,
                        "progress_percent": progress_percent,
                        "status": info.get("status") or "unknown",
                        "start_time": start_time,
                        "last_update": info.get("last_update") or start_time,
                        "runtime": time.time() - start_time,
                        "memory_footprint_mb": info.get("memory_footprint_mb") or 0.0,
                        "current_source_id": info.get("current_source_id"),
                        "errors": info.get("errors") or 0,
                        "warnings": info.get("warnings") or 0,
                        "current_stage": info["current_stage"],
                    }

            except Exception as e:
                logger.warning(f"Error processing progress file {progress_file}: {e}")
                continue

        return details

    def check_completion_status(self, total_sources: int = 0) -> Dict[str, Any]:
        """
        Check completion status by aggregating all session progress files.

        This method reads all progress files for this session to get the true
        completion status, which may be more up-to-date than in-memory counters.

        Args:
            total_sources: Expected total sources (for validation)

        Returns:
            Dictionary containing aggregated completion status
        """
        try:
            session_progress_files = self.get_session_progress_files()

            if not session_progress_files:
                # No progress files - either not started or completed and cleaned up
                return {
                    "has_progress_files": False,
                    "total_processes": 0,
                    "completed_processes": 0,
                    "total_sources_from_files": 0,
                    "completed_sources_from_files": 0,
                    "is_fully_completed": total_sources
                    > 0,  # If we expect sources but have no files, assume cleanup after completion
                }

            total_sources_from_files = 0
            all_sources_completed = 0  # From all processes (active and completed)
            failed_sources_from_files = 0
            completed_processes = 0
            total_processes = len(session_progress_files)

            for progress_file in session_progress_files:
                try:
                    process_id = progress_file.stem.replace(
                        f"cutana_progress_{self.session_id}_", ""
                    )
                    progress_data = self.read_progress_file(process_id)

                    if progress_data:
                        file_total = progress_data.get("total_sources", 0)
                        file_completed = progress_data.get("completed_sources", 0)
                        file_failed = progress_data.get("failed_sources", 0)
                        file_status = progress_data.get("status", "")

                        total_sources_from_files += file_total

                        # FIXED: Count completed sources from ALL processes to prevent jumping
                        all_sources_completed += file_completed

                        failed_sources_from_files += file_failed

                        # Count as completed if status is "completed" or if progress is 100%
                        progress_percent = progress_data.get("progress_percent", 0.0)
                        if (
                            file_status == "completed"
                            or (file_total > 0 and file_completed >= file_total)
                            or progress_percent >= 100.0
                        ):
                            completed_processes += 1

                except Exception as e:
                    logger.warning(f"Error reading progress file {progress_file}: {e}")
                    continue

            # Check if all processes are completed based on file data
            all_processes_completed = (
                completed_processes == total_processes
            ) and total_processes > 0

            # Also check if total sources match what we expect
            expected_total = total_sources if total_sources > 0 else total_sources_from_files

            # Check if all processes are completed based on process status
            is_fully_completed = all_processes_completed and total_processes > 0

            logger.debug(
                f"Completion check: {completed_processes}/{total_processes} processes completed, "
                f"all sources completed: {all_sources_completed}, "
                f"total sources from files: {total_sources_from_files}, "
                f"expected total: {expected_total}, is_fully_completed: {is_fully_completed}"
            )

            return {
                "has_progress_files": True,
                "total_processes": total_processes,
                "completed_processes": completed_processes,
                "total_sources_from_files": total_sources_from_files,
                "completed_sources_from_files": all_sources_completed,  # All completed sources from files
                "failed_sources_from_files": failed_sources_from_files,
                "expected_total_sources": expected_total,
                "is_fully_completed": is_fully_completed,
                "completion_percent": (
                    (completed_processes / total_processes * 100) if total_processes > 0 else 0.0
                ),
            }

        except Exception as e:
            logger.error(f"Error checking completion status: {e}")
            return {
                "has_progress_files": False,
                "total_processes": 0,
                "completed_processes": 0,
                "total_sources_from_files": 0,
                "completed_sources_from_files": 0,
                "failed_sources_from_files": 0,
                "is_fully_completed": False,
                "error": str(e),
            }

    def get_aggregated_status(
        self,
        total_sources: int = 0,
        completed_sources: int = 0,
        failed_sources: int = 0,
        start_time: float = None,
        system_resources: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive status information aggregated from progress files.

        Args:
            total_sources: Total sources being processed (fallback)
            completed_sources: Completed sources count (fallback)
            failed_sources: Failed sources count (fallback)
            start_time: Job start time (fallback)
            system_resources: System resource information

        Returns:
            Dictionary containing complete status
        """
        # Get process details from files
        process_details = self.get_process_details()

        # FIXED: Count completed sources from ALL processes to avoid jumping numbers
        # The race condition occurred when processes transitioned from "active" to "completed"
        # causing temporary decreases in the total. Now we count all completed sources.

        # Count completed sources from ALL processes (both active and completed)
        all_process_completed = sum(
            detail["completed_sources"] for detail in process_details.values()
        )

        effective_completed = all_process_completed
        effective_failed = failed_sources

        # For total sources, prefer the JobTracker's authoritative count
        if total_sources > 0:
            effective_total = total_sources
        else:
            # Fallback: sum assigned sources from all process files
            effective_total = sum(
                detail.get("sources_assigned", 0) for detail in process_details.values()
            )

        # Calculate derived metrics
        progress_percent = 0.0
        if effective_total > 0:
            progress_percent = (effective_completed / effective_total) * 100

        # Calculate throughput (still needed for display)
        eta_seconds = None
        throughput = 0.0
        if start_time and effective_completed > 0:
            elapsed_time = time.time() - start_time
            throughput = effective_completed / elapsed_time if elapsed_time > 0 else 0.0

            # Calculate simple ETA for fallback (smoothed ETA should be provided by JobTracker)
            if throughput > 0:
                remaining_sources = effective_total - effective_completed - effective_failed
                eta_seconds = remaining_sources / throughput if remaining_sources > 0 else 0

        # Aggregate process-level stats from progress files
        total_process_errors = sum(detail.get("errors", 0) for detail in process_details.values())
        total_process_warnings = sum(
            detail.get("warnings", 0) for detail in process_details.values()
        )
        total_memory_footprint = sum(
            detail.get("memory_footprint_mb", 0.0) for detail in process_details.values()
        )

        status = {
            # Job progress
            "total_sources": effective_total,
            "completed_sources": effective_completed,
            "failed_sources": effective_failed,
            "progress_percent": progress_percent,
            # Process information - only count truly active processes (not completed ones)
            "active_processes": len(
                [
                    detail
                    for _, detail in process_details.items()
                    if not self._is_process_completed(detail)
                ]
            ),
            "process_details": process_details,
            # Performance metrics
            "throughput": throughput,
            "eta_seconds": eta_seconds,
            "total_memory_footprint_mb": total_memory_footprint,
            "process_errors": total_process_errors,
            "process_warnings": total_process_warnings,
            # Timing
            "start_time": start_time,
            "current_time": time.time(),
        }

        # Only include system resources if provided
        if system_resources is not None:
            status["system_resources"] = system_resources

        return status

    def cleanup_stale_processes(self, timeout: int = 1800) -> List[str]:
        """
        Identify stale processes based on their progress file timestamps.

        Args:
            timeout: Timeout in seconds for considering a process stale

        Returns:
            List of process IDs that are considered stale
        """
        current_time = time.time()
        stale_processes = []
        session_files = self.get_session_progress_files()

        for progress_file in session_files:
            try:
                process_id = progress_file.stem.replace(f"cutana_progress_{self.session_id}_", "")
                progress_data = self.read_progress_file(process_id)

                if progress_data:
                    last_update = progress_data.get(
                        "last_update", progress_data.get("start_time", current_time)
                    )
                    if current_time - last_update > timeout:
                        stale_processes.append(process_id)
                        logger.warning(f"Identified stale process: {process_id}")

            except Exception as e:
                logger.warning(f"Error checking staleness for {progress_file}: {e}")
                continue

        return stale_processes

    def _is_process_completed(self, process_detail: Dict[str, Any]) -> bool:
        """
        Check if a process is completed based on its details.

        Uses the same logic as check_completion_status method to determine
        if a process should be considered completed and not counted as active.

        Args:
            process_detail: Dictionary containing process details

        Returns:
            True if process is completed, False if still active
        """
        status = process_detail.get("status", "")
        sources_assigned = process_detail.get("sources_assigned", 0)
        completed_sources = process_detail.get("completed_sources", 0)
        progress_percent = process_detail.get("progress_percent", 0.0)

        # Same logic as used in check_completion_status method (lines 249-255)
        return (
            status == "completed"
            or (sources_assigned > 0 and completed_sources >= sources_assigned)
            or progress_percent >= 100.0
        )
