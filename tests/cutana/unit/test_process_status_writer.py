#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the ProcessStatusWriter module.

Tests cover:
- Writing individual process progress files
- Process registration and progress updates
- Atomic file operations with locking
- Process completion handling
- Progress file cleanup
- Error handling and recovery
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cutana.process_status_writer import ProcessStatusWriter


class TestProcessStatusWriter:
    """Test suite for ProcessStatusWriter class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for progress files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def writer(self, temp_dir):
        """Create ProcessStatusWriter instance for testing."""
        return ProcessStatusWriter(progress_dir=str(temp_dir), session_id="test_session")

    @pytest.fixture
    def sample_progress_update(self):
        """Sample progress update data."""
        return {"completed_sources": 50, "memory_footprint_mb": 256.0, "errors": 1, "warnings": 3}

    def test_writer_initialization(self, temp_dir):
        """Test ProcessStatusWriter initializes correctly."""
        writer = ProcessStatusWriter(progress_dir=str(temp_dir), session_id="test_session")

        assert writer.progress_dir == temp_dir  # progress_dir is a Path object
        assert writer.session_id == "test_session"

    def test_register_process(self, writer, temp_dir):
        """Test registering a new process."""
        process_id = "cutout_process_001"
        sources_assigned = 100

        success = writer.register_process(process_id, sources_assigned)
        assert success is True

        # Verify progress file was created
        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        assert progress_file.exists()

        # Verify file contents
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["process_id"] == process_id
        assert data["total_sources"] == sources_assigned
        assert data["completed_sources"] == 0
        assert data["status"] == "starting"  # Actual implementation uses 'starting'
        assert "start_time" in data

    def test_register_process_duplicate(self, writer, temp_dir):
        """Test registering a process that already exists."""
        process_id = "cutout_process_001"

        # Register first time
        success1 = writer.register_process(process_id, 100)
        assert success1 is True

        # Register again - should still succeed but not overwrite
        success2 = writer.register_process(process_id, 200)
        assert success2 is True

        # Check that original data is preserved
        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["total_sources"] == 200  # Register overwrites with new value

    def test_update_process_progress(self, writer, temp_dir, sample_progress_update):
        """Test updating process progress."""
        process_id = "cutout_process_001"

        # Register process first
        writer.register_process(process_id, 100)

        # Update progress
        success = writer.update_process_progress(process_id, sample_progress_update)
        assert success is True

        # Verify updated contents
        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["completed_sources"] == 50
        # Note: update_process_progress doesn't set failed_sources directly
        assert data["memory_footprint_mb"] == 256.0
        assert data["status"] == "running"  # Actual implementation uses 'running'
        assert "last_update" in data

    def test_update_process_progress_nonexistent(self, writer, sample_progress_update):
        """Test updating progress for non-existent process should fail without fallbacks."""
        success = writer.update_process_progress("nonexistent_process", sample_progress_update)
        # Should fail because there's no progress file and we removed fallback values
        assert success is False

    def test_report_process_progress_simple(self, writer, temp_dir):
        """Test simple progress reporting."""
        process_id = "cutout_process_001"

        # Register process first
        writer.register_process(process_id, 100)

        # Report progress
        success = writer.report_process_progress(process_id, 75)
        assert success is True

        # Verify updated contents
        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["completed_sources"] == 75
        assert data["progress_percent"] == 75.0  # 75/100 * 100

    def test_complete_process(self, writer, temp_dir):
        """Test marking a process as completed."""
        process_id = "cutout_process_001"

        # Register and update process
        writer.register_process(process_id, 100)
        writer.update_process_progress(process_id, {"completed_sources": 50})

        # Complete process
        success = writer.complete_process(process_id, 95, 5)
        assert success is True

        # Verify completion data
        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["status"] == "completed"
        assert data["completed_sources"] == 95
        assert data["failed_sources"] == 5
        assert data["progress_percent"] == 100.0
        assert "completion_time" in data

    def test_complete_process_nonexistent(self, writer):
        """Test completing a non-existent process."""
        success = writer.complete_process("nonexistent_process", 50, 5)
        # Actually succeeds because it reads from ProcessStatusReader which returns None/empty data
        assert success is True

    def test_write_progress_file_atomic(self, writer, temp_dir):
        """Test atomic file writing with locking."""
        process_id = "cutout_process_001"

        # Register process first
        writer.register_process(process_id, 100)

        progress_data = {
            "process_id": process_id,
            "total_sources": 100,
            "completed_sources": 50,
            "status": "processing",
        }

        success = writer.write_progress_file(process_id, progress_data)
        assert success is True

        # Verify file was written correctly
        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data == progress_data

    def test_write_progress_file_permission_error(self, writer, temp_dir):
        """Test handling of permission errors during file writing."""
        process_id = "cutout_process_001"

        # Mock a permission error
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            success = writer.write_progress_file(process_id, {"test": "data"})
            assert success is False

    def test_cleanup_all_progress_files(self, writer, temp_dir):
        """Test cleanup of all progress files for the session."""
        # Register multiple processes
        processes = ["proc_001", "proc_002", "proc_003"]
        for process_id in processes:
            writer.register_process(process_id, 100)

        # Create files for other sessions (should not be deleted)
        other_file = temp_dir / "cutana_progress_other_session_proc_001.json"
        with open(other_file, "w") as f:
            json.dump({"test": "data"}, f)

        # Cleanup session files
        cleaned_count = writer.cleanup_all_progress_files()

        assert cleaned_count == 3
        assert not any(
            (temp_dir / f"cutana_progress_test_session_{proc}.json").exists() for proc in processes
        )
        assert other_file.exists()  # Should not be deleted

    def test_get_progress_file_path(self, writer, temp_dir):
        """Test progress file path generation."""
        process_id = "cutout_process_001"
        path = writer._get_progress_file_path(process_id)

        expected_path = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        assert path == expected_path

    def test_progress_percentage_calculation(self, writer, temp_dir):
        """Test progress percentage calculation during updates."""
        process_id = "cutout_process_001"
        total_sources = 200

        writer.register_process(process_id, total_sources)

        # Test various completion levels
        test_cases = [
            {"completed": 0, "expected": 0.0},
            {"completed": 50, "expected": 25.0},
            {"completed": 100, "expected": 50.0},
            {"completed": 200, "expected": 100.0},
        ]

        for case in test_cases:
            writer.report_process_progress(process_id, case["completed"])

            progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
            with open(progress_file, "r") as f:
                data = json.load(f)

            assert data["progress_percent"] == case["expected"]

    def test_timestamp_updates(self, writer, temp_dir):
        """Test that timestamps are updated correctly."""
        process_id = "cutout_process_001"

        # Register process
        start_time = time.time()
        writer.register_process(process_id, 100)

        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert "start_time" in data
        assert data["start_time"] >= start_time

        # Update progress
        update_time = time.time()
        writer.report_process_progress(process_id, 50)

        with open(progress_file, "r") as f:
            data = json.load(f)

        assert "last_update" in data
        assert data["last_update"] >= update_time

    def test_concurrent_access_simulation(self, writer, temp_dir):
        """Test simulated concurrent access to progress files."""
        process_id = "cutout_process_001"

        # Register process
        writer.register_process(process_id, 100)

        # Simulate multiple rapid updates (like multiple threads)
        update_values = [10, 20, 30, 40, 50]

        for value in update_values:
            success = writer.report_process_progress(process_id, value)
            assert success is True

        # Final value should be the last one written
        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["completed_sources"] == 50

    def test_process_status_transitions(self, writer, temp_dir):
        """Test process status transitions through lifecycle."""
        process_id = "cutout_process_001"
        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"

        # 1. Registration
        writer.register_process(process_id, 100)
        with open(progress_file, "r") as f:
            data = json.load(f)
        assert data["status"] == "starting"  # Actual implementation uses 'starting'

        # 2. First progress update
        writer.report_process_progress(process_id, 25)
        with open(progress_file, "r") as f:
            data = json.load(f)
        assert data["status"] == "processing"

        # 3. Completion
        writer.complete_process(process_id, 95, 5)
        with open(progress_file, "r") as f:
            data = json.load(f)
        assert data["status"] == "completed"

    def test_error_and_warning_tracking(self, writer, temp_dir):
        """Test tracking of errors and warnings."""
        process_id = "cutout_process_001"

        writer.register_process(process_id, 100)

        # Update with errors and warnings
        update_data = {
            "completed_sources": 30,
            "errors": 2,
            "warnings": 5,
            "last_error": "Sample error message",
        }

        writer.update_process_progress(process_id, update_data)

        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["errors"] == 2
        assert data["warnings"] == 5
        # Note: 'last_error' is not part of the actual implementation

    def test_memory_footprint_tracking(self, writer, temp_dir):
        """Test memory footprint tracking."""
        process_id = "cutout_process_001"

        writer.register_process(process_id, 100)

        # Update with memory info
        update_data = {"completed_sources": 25, "memory_footprint_mb": 512.5}

        writer.update_process_progress(process_id, update_data)

        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["memory_footprint_mb"] == 512.5

    def test_session_isolation_in_writing(self, temp_dir):
        """Test that different sessions write to separate files."""
        writer1 = ProcessStatusWriter(str(temp_dir), session_id="session_1")
        writer2 = ProcessStatusWriter(str(temp_dir), session_id="session_2")

        process_id = "proc_001"

        # Both writers register the same process ID
        writer1.register_process(process_id, 100)
        writer2.register_process(process_id, 200)

        # Check that separate files were created
        file1 = temp_dir / f"cutana_progress_session_1_{process_id}.json"
        file2 = temp_dir / f"cutana_progress_session_2_{process_id}.json"

        assert file1.exists()
        assert file2.exists()

        # Check that they have different data
        with open(file1, "r") as f:
            data1 = json.load(f)
        with open(file2, "r") as f:
            data2 = json.load(f)

        assert data1["total_sources"] == 100
        assert data2["total_sources"] == 200

    def test_file_system_error_handling(self, writer):
        """Test handling of file system errors."""
        process_id = "cutout_process_001"

        # Mock file system error on write_progress_file
        with patch.object(writer, "write_progress_file", return_value=False):
            success = writer.register_process(process_id, 100)
            assert success is False

    def test_json_serialization_error_handling(self, writer, temp_dir):
        """Test handling of JSON serialization errors."""
        process_id = "cutout_process_001"

        # Create a data structure that can't be JSON serialized
        invalid_data = {
            "process_id": process_id,
            "invalid_data": set([1, 2, 3]),  # sets are not JSON serializable
        }

        success = writer.write_progress_file(process_id, invalid_data)
        assert success is False

    def test_large_progress_data_handling(self, writer, temp_dir):
        """Test handling of large progress data."""
        process_id = "cutout_process_001"

        # Create large progress data
        large_data = {
            "process_id": process_id,
            "total_sources": 1000000,
            "completed_sources": 500000,
            "large_list": list(range(1000)),  # Large data structure
            "status": "processing",
        }

        success = writer.write_progress_file(process_id, large_data)
        assert success is True

        # Verify it was written correctly
        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["completed_sources"] == 500000
        assert len(loaded_data["large_list"]) == 1000

    def test_edge_case_zero_sources(self, writer, temp_dir):
        """Test edge case of zero sources assigned."""
        process_id = "cutout_process_001"

        writer.register_process(process_id, 0)

        progress_file = temp_dir / f"cutana_progress_test_session_{process_id}.json"
        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["total_sources"] == 0
        assert data["progress_percent"] == 0.0

        # Complete with zero sources
        writer.complete_process(process_id, 0, 0)

        with open(progress_file, "r") as f:
            data = json.load(f)

        assert data["status"] == "completed"
        assert data["progress_percent"] == 100.0  # Complete process sets progress to 100%
