#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the JobTracker module using TDD approach.

Tests cover:
- Progress tracking and statistics
- Process lifecycle management
- Error recording and reporting
- Persistence of tracking data
"""

import time
from unittest.mock import patch

import pytest

from cutana.job_tracker import JobTracker


class TestJobTracker:
    """Test suite for JobTracker class."""

    @pytest.fixture
    def job_tracker(self, tmp_path):
        """Create JobTracker instance for testing."""
        progress_dir = str(tmp_path)
        return JobTracker(progress_dir=progress_dir)

    def test_job_tracker_initialization(self, tmp_path):
        """Test JobTracker initialization with progress directory."""
        progress_dir = str(tmp_path)
        tracker = JobTracker(progress_dir=progress_dir)

        assert tracker.progress_dir == progress_dir
        assert tracker.total_sources == 0
        assert tracker.completed_sources == 0
        assert tracker.failed_sources == 0
        assert tracker.active_processes == {}
        assert tracker.errors == []
        assert tracker.start_time is None  # Only set when start_job() is called

    def test_start_job_tracking(self, job_tracker):
        """Test starting job tracking with source count."""
        total_sources = 1000
        job_tracker.start_job(total_sources)

        assert job_tracker.total_sources == total_sources
        assert job_tracker.completed_sources == 0
        assert job_tracker.failed_sources == 0
        assert job_tracker.start_time is not None

    def test_register_process(self, job_tracker):
        """Test registering a new process."""
        process_id = "cutout_process_001"
        sources_assigned = 25

        job_tracker.register_process(process_id, sources_assigned)

        assert process_id in job_tracker.active_processes
        process_info = job_tracker.active_processes[process_id]
        assert process_info["sources_assigned"] == sources_assigned
        assert process_info["completed_sources"] == 0
        assert "start_time" in process_info

    def test_update_process_progress(self, job_tracker):
        """Test updating process progress."""
        process_id = "cutout_process_001"
        job_tracker.register_process(process_id, 50)

        # Update progress
        progress_update = {"completed_sources": 25}
        success = job_tracker.update_process_progress(process_id, progress_update)

        assert success is True
        assert job_tracker.active_processes[process_id]["completed_sources"] == 25

    def test_complete_process(self, job_tracker):
        """Test process completion."""
        process_id = "cutout_process_001"
        job_tracker.register_process(process_id, 50)

        # Complete the process
        completed = 45
        failed = 5
        success = job_tracker.complete_process(process_id, completed, failed)

        assert success is True

        # Verify the process completion was recorded in the progress file
        completion_status = job_tracker.process_reader.check_completion_status()
        assert completion_status["completed_sources_from_files"] == completed
        assert completion_status["failed_sources_from_files"] == failed

        # Process should be removed from active processes
        assert process_id not in job_tracker.active_processes

    def test_record_error(self, job_tracker):
        """Test error recording."""
        error_info = {"process_id": "proc1", "error": "test error", "timestamp": time.time()}

        job_tracker.record_error(error_info)

        assert len(job_tracker.errors) == 1
        assert job_tracker.errors[0] == error_info

    def test_get_detailed_process_info(self, job_tracker):
        """Test getting detailed information about processes."""
        # Register processes
        job_tracker.register_process("proc1", 50)
        job_tracker.register_process("proc2", 25)

        # Mock progress file data - ProcessStatusReader expects total_sources and current_stage
        mock_progress_data = {
            "process_id": "proc1",
            "total_sources": 50,
            "completed_sources": 30,
            "status": "running",
            "start_time": time.time(),
            "current_stage": "Processing",
        }

        with patch.object(job_tracker.process_reader, "read_progress_file") as mock_read:
            mock_read.return_value = mock_progress_data

            details = job_tracker.get_process_details()

            assert "proc1" in details
            assert details["proc1"]["sources_assigned"] == 50
            assert details["proc1"]["completed_sources"] == 30

    def test_cleanup_stale_processes(self, job_tracker):
        """Test cleanup of stale/dead processes."""
        # Register processes
        job_tracker.register_process("proc1", 50)
        job_tracker.register_process("proc2", 50)

        # Mock the progress file timestamps to simulate stale process
        old_time = time.time() - 3600  # 1 hour ago

        # Create a mock progress data with old timestamp
        mock_progress_data = {
            "process_id": "proc1",
            "last_update": old_time,
            "start_time": old_time,
            "total_sources": 50,
            "completed_sources": 10,
        }

        # Mock the read_progress_file to return stale data for proc1
        with patch.object(job_tracker.process_reader, "read_progress_file") as mock_read:

            def mock_read_func(process_id):
                if process_id == "proc1":
                    return mock_progress_data
                else:
                    return {
                        "process_id": process_id,
                        "last_update": time.time(),
                        "start_time": time.time(),
                    }

            mock_read.side_effect = mock_read_func

            stale_processes = job_tracker.cleanup_stale_processes(timeout=1800)  # 30 min timeout

            assert "proc1" in stale_processes
            assert "proc2" not in stale_processes
            # Note: JobTracker's cleanup_stale_processes only identifies stale processes,
            # it doesn't remove them from active_processes - that would be done by caller
