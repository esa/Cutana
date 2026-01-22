#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Progress reporting dataclass for Cutana pipeline.

This module provides a clean, type-safe way to handle progress information
instead of manually constructing large dictionaries.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProgressReport:
    """Clean dataclass for progress reporting throughout the Cutana pipeline."""

    # Core progress metrics
    total_sources: int = 0
    completed_sources: int = 0
    failed_sources: int = 0
    progress_percent: float = 0.0

    # Performance metrics
    throughput: float = 0.0
    eta_seconds: Optional[float] = None

    # System resources (from LoadBalancer)
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    memory_available_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0

    # Worker information (from LoadBalancer)
    active_processes: int = 0
    max_workers: int = 0
    worker_memory_limit_gb: float = 0.0

    # Enhanced memory fields (from LoadBalancer)
    worker_memory_allocation_mb: float = 0.0
    worker_memory_peak_mb: float = 0.0
    worker_memory_remaining_mb: float = 0.0
    main_process_memory_mb: float = 0.0

    # Legacy memory fields (for compatibility)
    avg_worker_memory_mb: float = 0.0
    peak_worker_memory_mb: float = 0.0
    processes_measured: int = 0
    calibration_completed: bool = False
    resource_source: str = "system"
    total_memory_footprint_mb: float = 0.0

    # Error tracking
    process_errors: int = 0
    process_warnings: int = 0
    total_errors: int = 0

    # Timing information
    start_time: Optional[float] = None
    current_time: Optional[float] = None

    # Status indicators
    is_processing: bool = False

    @classmethod
    def empty(cls) -> "ProgressReport":
        """Create an empty progress report with safe defaults."""
        return cls()

    @classmethod
    def from_status_components(
        cls,
        full_status: Dict[str, Any],
        system_info: Dict[str, Any] = None,
        limits_info: Dict[str, Any] = None,
        performance_info: Dict[str, Any] = None,
        completion_status: Dict[str, Any] = None,
    ) -> "ProgressReport":
        """
        Create ProgressReport from orchestrator status components.

        This outsources the boilerplate of extracting data from the various
        status dictionaries into a clean, reusable method.

        Args:
            full_status: Main status from JobTracker
            system_info: System resources from LoadBalancer
            limits_info: Resource limits from LoadBalancer
            performance_info: Performance metrics from LoadBalancer
            completion_status: File-based completion status from JobTracker

        Returns:
            ProgressReport with all fields populated
        """
        # Initialize with defaults
        system_info = system_info or {}
        limits_info = limits_info or {}
        performance_info = performance_info or {}
        completion_status = completion_status or {}

        report = cls()

        # Always use the total_sources from JobTracker (full_status) to avoid jumping
        # This is the authoritative source set at job initialization
        report.total_sources = full_status.get("total_sources", 0)

        # Use file-based completion data if available and more recent
        if completion_status.get("has_progress_files", False):
            file_completed = completion_status.get("completed_sources_from_files", 0)
            is_fully_completed = completion_status.get("is_fully_completed", False)

            # Use file-based completion data if it's more recent than in-memory data
            if file_completed > full_status.get("completed_sources", 0):
                report.completed_sources = file_completed
                report.failed_sources = max(0, report.total_sources - file_completed)
                # Calculate progress percent based on consistent total_sources
                if report.total_sources > 0:
                    report.progress_percent = (file_completed / report.total_sources) * 100.0
                else:
                    report.progress_percent = 0.0
                report.is_processing = (
                    not is_fully_completed and len(full_status.get("process_details", {})) > 0
                )
                report.eta_seconds = 0 if is_fully_completed else full_status.get("eta_seconds")
                # Ensure 100% progress when fully completed
                if is_fully_completed:
                    report.progress_percent = 100.0
            else:
                # Use in-memory data
                report.completed_sources = full_status.get("completed_sources", 0)
                report.failed_sources = full_status.get("failed_sources", 0)
                report.progress_percent = full_status.get("progress_percent", 0.0)
                report.eta_seconds = full_status.get("eta_seconds")
                report.is_processing = (
                    report.total_sources > 0
                    and (report.completed_sources + report.failed_sources) < report.total_sources
                ) or len(full_status.get("process_details", {})) > 0

                # Ensure 100% progress when all sources are completed
                if report.total_sources > 0 and report.completed_sources >= report.total_sources:
                    report.progress_percent = 100.0
                    report.is_processing = False
        else:
            # Use in-memory data
            report.completed_sources = full_status.get("completed_sources", 0)
            report.failed_sources = full_status.get("failed_sources", 0)
            report.progress_percent = full_status.get("progress_percent", 0.0)
            report.eta_seconds = full_status.get("eta_seconds")
            report.is_processing = (
                report.total_sources > 0
                and (report.completed_sources + report.failed_sources) < report.total_sources
            ) or len(full_status.get("process_details", {})) > 0

            # Ensure 100% progress when all sources are completed
            if report.total_sources > 0 and report.completed_sources >= report.total_sources:
                report.progress_percent = 100.0
                report.is_processing = False

        # Fill in common fields
        report.throughput = full_status.get("throughput", 0.0)

        # System resources from LoadBalancer
        report.memory_percent = system_info.get("memory_percent", 0.0)
        report.cpu_percent = system_info.get("cpu_percent", 0.0)
        report.memory_available_gb = system_info.get("memory_available_gb", 0.0)
        report.memory_total_gb = system_info.get("memory_total_gb", 0.0)
        report.memory_used_gb = max(0.0, report.memory_total_gb - report.memory_available_gb)

        # LoadBalancer worker information
        report.active_processes = full_status.get("active_processes", 0)
        report.max_workers = limits_info.get("cpu_limit", 0)
        report.worker_memory_limit_gb = limits_info.get("memory_limit_gb", 0.0)

        # Enhanced memory fields from LoadBalancer
        report.worker_memory_allocation_mb = performance_info.get("worker_allocation_mb", 0.0)
        report.worker_memory_peak_mb = performance_info.get("worker_peak_mb", 0.0)
        report.worker_memory_remaining_mb = performance_info.get("worker_remaining_mb", 0.0)
        report.main_process_memory_mb = performance_info.get("main_process_memory_mb", 0.0)

        # Legacy memory fields for compatibility
        report.avg_worker_memory_mb = performance_info.get("avg_memory_mb", 0.0)
        report.peak_worker_memory_mb = performance_info.get("peak_memory_mb", 0.0)
        report.processes_measured = performance_info.get("processes_measured", 0)
        report.calibration_completed = performance_info.get("calibration_completed", False)
        report.resource_source = system_info.get("resource_source", "system")
        report.total_memory_footprint_mb = full_status.get("total_memory_footprint_mb", 0.0)

        # Error tracking
        report.process_errors = full_status.get("process_errors", 0)
        report.process_warnings = full_status.get("process_warnings", 0)
        report.total_errors = full_status.get("total_errors", 0)

        # Timing
        report.start_time = full_status.get("start_time")
        report.current_time = full_status.get("current_time")

        # Validate and fix any invalid values
        return report.validate_and_fix()

    def to_dict(self) -> Dict[str, Any]:
        """Convert ProgressReport back to dictionary for compatibility."""
        from dataclasses import asdict

        return asdict(self)

    def safe_float(self, value, default: float = 0.0) -> float:
        """Ensure value is a valid float."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def safe_int(self, value, default: int = 0) -> int:
        """Ensure value is a valid integer."""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def validate_and_fix(self) -> "ProgressReport":
        """Validate and fix any invalid values in the progress report."""
        # Ensure numeric fields are valid
        self.total_sources = self.safe_int(self.total_sources)
        self.completed_sources = self.safe_int(self.completed_sources)
        self.failed_sources = self.safe_int(self.failed_sources)
        self.progress_percent = self.safe_float(self.progress_percent)

        self.throughput = self.safe_float(self.throughput)

        self.memory_percent = self.safe_float(self.memory_percent)
        self.cpu_percent = self.safe_float(self.cpu_percent)
        self.memory_available_gb = self.safe_float(self.memory_available_gb)
        self.memory_total_gb = self.safe_float(self.memory_total_gb)
        self.memory_used_gb = self.safe_float(self.memory_used_gb)

        self.active_processes = self.safe_int(self.active_processes)
        self.max_workers = self.safe_int(self.max_workers)
        self.worker_memory_limit_gb = self.safe_float(self.worker_memory_limit_gb)

        self.worker_memory_allocation_mb = self.safe_float(self.worker_memory_allocation_mb)
        self.worker_memory_peak_mb = self.safe_float(self.worker_memory_peak_mb)
        self.worker_memory_remaining_mb = self.safe_float(self.worker_memory_remaining_mb)
        self.main_process_memory_mb = self.safe_float(self.main_process_memory_mb)

        self.avg_worker_memory_mb = self.safe_float(self.avg_worker_memory_mb)
        self.peak_worker_memory_mb = self.safe_float(self.peak_worker_memory_mb)
        self.processes_measured = self.safe_int(self.processes_measured)
        self.total_memory_footprint_mb = self.safe_float(self.total_memory_footprint_mb)

        self.process_errors = self.safe_int(self.process_errors)
        self.process_warnings = self.safe_int(self.process_warnings)
        self.total_errors = self.safe_int(self.total_errors)

        # Ensure resource_source is a string
        if not isinstance(self.resource_source, str):
            self.resource_source = "system"

        return self
