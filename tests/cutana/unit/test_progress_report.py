#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for ProgressReport dataclass.
"""

from cutana.progress_report import ProgressReport


class TestProgressReport:
    """Test suite for ProgressReport dataclass."""

    def test_empty_progress_report(self):
        """Test creating an empty progress report."""
        report = ProgressReport.empty()

        assert report.total_sources == 0
        assert report.completed_sources == 0
        assert report.failed_sources == 0
        assert report.progress_percent == 0.0
        assert report.throughput == 0.0
        assert report.eta_seconds is None
        assert not report.is_processing
        assert report.resource_source == "system"

    def test_to_dict_conversion(self):
        """Test converting ProgressReport back to dictionary."""
        report = ProgressReport(
            total_sources=50,
            completed_sources=25,
            progress_percent=50.0,
            memory_total_gb=8.0,
            cpu_percent=30.0,
        )

        result_dict = report.to_dict()

        assert result_dict["total_sources"] == 50
        assert result_dict["completed_sources"] == 25
        assert result_dict["progress_percent"] == 50.0
        assert result_dict["memory_total_gb"] == 8.0
        assert result_dict["cpu_percent"] == 30.0
        assert result_dict["failed_sources"] == 0  # Default value
        assert result_dict["resource_source"] == "system"  # Default value

    def test_validate_and_fix(self):
        """Test validation and fixing of invalid values."""
        report = ProgressReport(
            total_sources="100",  # Should be converted to int
            completed_sources=75.7,  # Should be converted to int
            progress_percent="invalid",  # Should become 0.0
            memory_total_gb=None,  # Should become 0.0
            cpu_percent=-5.0,  # Negative value should be preserved (might be valid)
            resource_source=123,  # Should become "system"
        )

        fixed_report = report.validate_and_fix()

        assert fixed_report.total_sources == 100
        assert fixed_report.completed_sources == 75
        assert fixed_report.progress_percent == 0.0
        assert fixed_report.memory_total_gb == 0.0
        assert fixed_report.cpu_percent == -5.0  # Preserved
        assert fixed_report.resource_source == "system"

    def test_from_status_components_basic(self):
        """Test creating ProgressReport from status components."""
        full_status = {
            "total_sources": 200,
            "completed_sources": 150,
            "failed_sources": 10,
            "progress_percent": 75.0,
            "throughput": 12.5,
            "eta_seconds": 60.0,
            "process_errors": 2,
            "process_warnings": 5,
            "total_errors": 3,
            "start_time": 1625000000.0,
            "current_time": 1625000060.0,
            "active_processes": 3,
            "total_memory_footprint_mb": 2048.0,
            "process_details": {"proc1": {}, "proc2": {}, "proc3": {}},
        }

        system_info = {
            "memory_percent": 65.0,
            "cpu_percent": 80.0,
            "memory_available_gb": 8.0,
            "memory_total_gb": 16.0,
            "resource_source": "kubernetes_pod",
        }

        limits_info = {"cpu_limit": 8, "memory_limit_gb": 12.0}

        performance_info = {
            "worker_allocation_mb": 1500.0,
            "worker_peak_mb": 1200.0,
            "worker_remaining_mb": 300.0,
            "main_process_memory_mb": 256.0,
            "avg_memory_mb": 1000.0,
            "peak_memory_mb": 1200.0,
            "processes_measured": 5,
        }

        report = ProgressReport.from_status_components(
            full_status=full_status,
            system_info=system_info,
            limits_info=limits_info,
            performance_info=performance_info,
        )

        # Check basic progress fields
        assert report.total_sources == 200
        assert report.completed_sources == 150
        assert report.failed_sources == 10
        assert report.progress_percent == 75.0
        assert report.throughput == 12.5
        assert report.eta_seconds == 60.0

        # Check system resources
        assert report.memory_percent == 65.0
        assert report.cpu_percent == 80.0
        assert report.memory_available_gb == 8.0
        assert report.memory_total_gb == 16.0
        assert report.memory_used_gb == 8.0  # 16 - 8
        assert report.resource_source == "kubernetes_pod"

        # Check worker information
        assert report.active_processes == 3
        assert report.max_workers == 8
        assert report.worker_memory_limit_gb == 12.0

        # Check enhanced memory fields
        assert report.worker_memory_allocation_mb == 1500.0
        assert report.worker_memory_peak_mb == 1200.0
        assert report.worker_memory_remaining_mb == 300.0
        assert report.main_process_memory_mb == 256.0

        # Check legacy fields
        assert report.avg_worker_memory_mb == 1000.0
        assert report.peak_worker_memory_mb == 1200.0
        assert report.processes_measured == 5
        assert report.total_memory_footprint_mb == 2048.0

        # Check error tracking
        assert report.process_errors == 2
        assert report.process_warnings == 5
        assert report.total_errors == 3

        # Check timing
        assert report.start_time == 1625000000.0
        assert report.current_time == 1625000060.0

        # Should be processing since we have active process_details
        assert report.is_processing is True

    def test_from_status_components_with_file_completion(self):
        """Test creating ProgressReport with file-based completion status."""
        full_status = {
            "total_sources": 100,
            "completed_sources": 50,  # Lower than file-based
            "failed_sources": 0,
            "progress_percent": 50.0,
            "process_details": {"proc1": {}},
        }

        completion_status = {
            "has_progress_files": True,
            "completed_sources_from_files": 80,  # Higher than in-memory
            "expected_total_sources": 100,
            "is_fully_completed": False,
            "completion_percent": 80.0,
        }

        report = ProgressReport.from_status_components(
            full_status=full_status, completion_status=completion_status
        )

        # Should use file-based data since it's more complete
        assert report.total_sources == 100
        assert report.completed_sources == 80
        assert report.failed_sources == 20  # 100 - 80
        assert report.progress_percent == 80.0
        assert report.is_processing is True  # Not fully completed and has process_details

    def test_from_status_components_empty_inputs(self):
        """Test creating ProgressReport with empty/None inputs."""
        report = ProgressReport.from_status_components(
            full_status={},
            system_info=None,
            limits_info=None,
            performance_info=None,
            completion_status=None,
        )

        # Should create valid report with all defaults
        assert report.total_sources == 0
        assert report.completed_sources == 0
        assert report.failed_sources == 0
        assert report.progress_percent == 0.0
        assert report.throughput == 0.0
        assert report.memory_total_gb == 0.0
        assert report.cpu_percent == 0.0
        assert report.max_workers == 0
        assert report.resource_source == "system"
        assert report.is_processing is False
