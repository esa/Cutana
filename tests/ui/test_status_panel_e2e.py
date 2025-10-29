#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
End-to-end tests for status panel integration with backend and progress tracking.

This test module validates the complete flow:
UI StatusPanel → BackendInterface → Orchestrator → JobTracker → Progress Files
"""

import time
from unittest.mock import Mock, patch
import pytest
import pandas as pd

from cutana_ui.main_screen.status_panel import StatusPanel
from cutana_ui.utils.backend_interface import BackendInterface
from cutana.orchestrator import Orchestrator
from cutana.job_tracker import JobTracker
from cutana.progress_report import ProgressReport
from cutana import get_default_config


class TestStatusPanelE2E:
    """End-to-end tests for status panel progress monitoring."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = get_default_config()
        config.num_sources = 10
        config.max_workers = 2
        config.source_catalogue = "/tmp/test_catalogue.csv"
        config.output_dir = "/tmp/cutouts"
        config.output_format = "zarr"
        config.target_resolution = 256
        config.data_type = "float32"
        config.fits_extensions = ["VIS"]
        config.selected_extensions = [{"name": "VIS", "ext": "PRIMARY"}]
        config.normalisation_method = "linear"
        config.max_sources_per_process = 1000  # Use valid minimum value
        config.N_batch_cutout_process = 100  # Use valid value <= max_sources_per_process
        config.max_workflow_time_seconds = 600  # Minimum allowed timeout for tests
        return config

    @pytest.fixture
    def mock_catalogue(self):
        """Create mock catalogue data."""
        data = [
            {
                "SourceID": f"MockSource_{i:05d}",
                "RA": 149.95 + i * 0.01,
                "Dec": 1.95 + i * 0.01,
                "diameter_pixel": 64,
                "fits_file_paths": "['/mock/data/euclid_tile_001.fits']",
            }
            for i in range(10)
        ]
        return pd.DataFrame(data)

    @pytest.fixture
    def status_panel(self, config):
        """Create status panel for testing."""
        return StatusPanel(config)

    @pytest.fixture(autouse=True)
    def clear_backend_orchestrator(self):
        """Clear backend orchestrator before each test."""
        BackendInterface._current_orchestrator = None
        yield
        BackendInterface._current_orchestrator = None

    @pytest.mark.asyncio
    async def test_status_panel_no_active_session(self, status_panel):
        """Test status panel behavior when no processing session is active."""
        # Initially, there should be no active processing session
        status = await BackendInterface.get_processing_status()

        # CORRECTED: No active session is not an error, just normal state
        assert status["is_processing"] is False
        assert status["total_sources"] == 0
        assert status["completed_sources"] == 0

    @pytest.mark.asyncio
    async def test_status_panel_with_mock_orchestrator(self, status_panel, config):
        """Test status panel with a mock orchestrator providing progress updates."""

        # Create a mock orchestrator that simulates processing
        mock_orchestrator = Mock(spec=Orchestrator)
        mock_progress_data = {
            "total_sources": 10,
            "completed_sources": 0,
            "failed_sources": 0,
            "progress_percent": 0.0,
            "throughput": 0.0,
            "eta_seconds": None,
            "memory_percent": 25.0,
            "cpu_percent": 50.0,
            "memory_available_gb": 8.0,
            "memory_total_gb": 16.0,
            "active_processes": 1,
            "total_memory_footprint_mb": 512.0,
            "process_errors": 0,
            "process_warnings": 0,
            "start_time": time.time(),
            "is_processing": True,
        }

        # Use ProgressReport for mock return
        mock_progress_report = ProgressReport.from_dict(mock_progress_data)
        mock_orchestrator.get_progress_for_ui.return_value = mock_progress_report

        # Set mock orchestrator in backend
        BackendInterface._current_orchestrator = mock_orchestrator

        # Get status from backend
        status = await BackendInterface.get_processing_status()

        # Verify status is correctly retrieved and formatted
        assert status["is_processing"] is True
        assert status["total_sources"] == 10
        assert status["completed_sources"] == 0
        assert status["active_processes"] == 1
        assert status["memory_percent"] == 25.0

        # Simulate progress
        mock_progress_data["completed_sources"] = 5
        mock_progress_data["progress_percent"] = 50.0
        mock_progress_report = ProgressReport.from_dict(mock_progress_data)
        mock_orchestrator.get_progress_for_ui.return_value = mock_progress_report

        status = await BackendInterface.get_processing_status()
        assert status["completed_sources"] == 5
        assert status["progress_percent"] == 50.0

        # Clean up
        BackendInterface._current_orchestrator = None

    @pytest.mark.asyncio
    async def test_status_panel_direct_updates(self, config, status_panel):
        """Test that status panel receives direct updates from orchestrator."""
        # Test initial state
        assert status_panel.is_processing is False

        # Simulate starting processing
        status_panel.start_processing()
        assert status_panel.is_processing is True

        # Create a test progress report
        progress_report = ProgressReport(
            total_sources=100,
            completed_sources=50,
            progress_percent=50.0,
            is_processing=True,
            active_processes=2,
            max_workers=4,
            memory_used_gb=8.0,
            memory_total_gb=16.0,
        )

        # Send update directly to status panel
        status_panel.receive_status_UI_update(progress_report)

        # Check that progress bar was updated
        assert status_panel.progress_bar.value == 50.0

        # Send completion update
        completion_report = ProgressReport(
            total_sources=100,
            completed_sources=100,
            progress_percent=100.0,
            is_processing=False,
            active_processes=0,
            max_workers=4,
        )

        status_panel.receive_status_UI_update(completion_report)

        # Check completion state
        assert status_panel.progress_bar.value == 100.0

    @pytest.mark.asyncio
    async def test_status_panel_progress_file_integration(self, config, tmp_path):
        """Test that status panel correctly reflects progress from file-based tracking."""

        # Create temp directory for progress files
        progress_dir = tmp_path / "progress"
        progress_dir.mkdir()

        # Create a job tracker with temp directory
        job_tracker = JobTracker(progress_dir=str(progress_dir))
        job_tracker.start_job(total_sources=100)

        # Register and update a process
        job_tracker.register_process("test_process_001", sources_assigned=50)
        job_tracker.update_process_progress("test_process_001", {"completed_sources": 25})

        # Get status from job tracker
        status = job_tracker.get_status()

        # Verify status contains expected data
        assert status["total_sources"] == 100
        assert status["completed_sources"] == 25
        assert status["progress_percent"] == 25.0
        assert status["active_processes"] == 1

    @pytest.mark.asyncio
    async def test_backend_error_handling(self, status_panel, config):
        """Test that backend errors are handled gracefully."""

        # No orchestrator available initially
        status = await BackendInterface.get_processing_status()
        assert status["is_processing"] is False

        # Create mock orchestrator that raises an error
        mock_orchestrator = Mock(spec=Orchestrator)
        mock_orchestrator.get_progress_for_ui.side_effect = RuntimeError("Test error")

        BackendInterface._current_orchestrator = mock_orchestrator

        # Should return safe defaults on error
        status = await BackendInterface.get_processing_status()
        assert "error" in status
        assert status["is_processing"] is True  # Assumes still processing on error

        # Clean up
        BackendInterface._current_orchestrator = None

    @pytest.mark.asyncio
    async def test_orchestrator_with_status_panel_integration(
        self, config, status_panel, mock_catalogue
    ):
        """Test orchestrator sending direct updates to status panel."""

        # Create orchestrator with status panel reference
        orchestrator = Orchestrator(config, status_panel=status_panel)

        # Check that orchestrator has status panel reference
        assert orchestrator.status_panel == status_panel

        # Mock the _spawn_cutout_process to avoid actual subprocess creation
        with patch.object(orchestrator, "_spawn_cutout_process"):
            # Test _send_ui_update method
            orchestrator._send_ui_update(force=True)

            # Verify last update time was set
            assert orchestrator.last_ui_update_time > 0

    def test_status_panel_handles_missing_receive_method(self, config):
        """Test graceful handling when status panel doesn't have receive method."""

        # Create a mock status panel without receive_status_UI_update
        mock_panel = Mock()
        del mock_panel.receive_status_UI_update  # Ensure method doesn't exist

        # Create orchestrator with mock panel
        orchestrator = Orchestrator(config, status_panel=mock_panel)

        # This should not raise an error
        orchestrator._send_ui_update(force=True)

        # Check that warning was logged (method returns without error)
        assert orchestrator.status_panel == mock_panel
