#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Tests for stop/kill functionality in the UI and orchestrator.

This test module validates that:
1. The orchestrator can properly stop and kill running subprocesses
2. The UI can request stops through the backend interface
3. The stop functionality works correctly in both normal and error conditions
"""

import asyncio
import subprocess
from unittest.mock import Mock
import pytest

from cutana_ui.main_screen.status_panel import StatusPanel
from cutana_ui.utils.backend_interface import BackendInterface
from cutana.orchestrator import Orchestrator
from cutana import get_default_config


class TestStopKillFunctionality:
    """Tests for stop/kill functionality."""

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
        config.max_sources_per_process = 1000
        config.N_batch_cutout_process = 100
        config.max_workflow_time_seconds = 600
        return config

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
    async def test_backend_stop_no_active_session(self):
        """Test BackendInterface.stop_processing when no active session exists."""
        # No orchestrator should be active initially
        result = await BackendInterface.stop_processing()

        assert result["status"] == "no_active_session"
        assert "No active processing session" in result["message"]

    @pytest.mark.asyncio
    async def test_backend_stop_with_active_orchestrator(self, config):
        """Test BackendInterface.stop_processing with an active orchestrator."""
        # Create a mock orchestrator
        mock_orchestrator = Mock(spec=Orchestrator)
        mock_stop_result = {
            "status": "stopped",
            "stopped_processes": ["cutout_process_001", "cutout_process_002"],
        }
        mock_orchestrator.stop_processing.return_value = mock_stop_result

        # Set the mock orchestrator as active
        BackendInterface._current_orchestrator = mock_orchestrator

        # Test stop processing
        result = await BackendInterface.stop_processing()

        # Verify results
        assert result["status"] == "success"
        assert result["result"] == mock_stop_result

        # Verify orchestrator's stop method was called
        mock_orchestrator.stop_processing.assert_called_once()

        # Verify orchestrator reference was cleared
        assert BackendInterface._current_orchestrator is None

    @pytest.mark.asyncio
    async def test_backend_stop_with_error(self, config):
        """Test BackendInterface.stop_processing when orchestrator stop fails."""
        # Create a mock orchestrator that raises an exception
        mock_orchestrator = Mock(spec=Orchestrator)
        mock_orchestrator.stop_processing.side_effect = Exception("Stop failed")

        # Set the mock orchestrator as active
        BackendInterface._current_orchestrator = mock_orchestrator

        # Test stop processing
        result = await BackendInterface.stop_processing()

        # Verify error handling
        assert result["status"] == "error"
        assert "Failed to stop processing" in result["error"]
        assert "Stop failed" in result["error"]

        # Verify orchestrator reference was cleared even on error
        assert BackendInterface._current_orchestrator is None

    def test_orchestrator_stop_processing(self, config, tmp_path):
        """Test orchestrator stop_processing method directly."""
        # Set up temp files
        config.tracking_file = str(tmp_path / "tracking.json")

        # Create orchestrator
        orchestrator = Orchestrator(config)

        # Create mock processes
        mock_process_1 = Mock(spec=subprocess.Popen)
        mock_process_1.terminate = Mock()
        mock_process_1.wait = Mock(return_value=0)
        mock_process_1.kill = Mock()

        mock_process_2 = Mock(spec=subprocess.Popen)
        mock_process_2.terminate = Mock()
        mock_process_2.wait = Mock(return_value=0)
        mock_process_2.kill = Mock()

        # Add mock processes to orchestrator
        orchestrator.active_processes = {
            "cutout_process_001": mock_process_1,
            "cutout_process_002": mock_process_2,
        }

        # Test stop processing
        result = orchestrator.stop_processing()

        # Verify results
        assert result["status"] == "stopped"
        assert len(result["stopped_processes"]) == 2
        assert "cutout_process_001" in result["stopped_processes"]
        assert "cutout_process_002" in result["stopped_processes"]

        # Verify processes were terminated
        mock_process_1.terminate.assert_called_once()
        mock_process_1.wait.assert_called_once()
        mock_process_2.terminate.assert_called_once()
        mock_process_2.wait.assert_called_once()

        # Verify active processes were cleared
        assert len(orchestrator.active_processes) == 0

    def test_orchestrator_stop_processing_with_hanging_process(self, config, tmp_path):
        """Test orchestrator stop_processing with a hanging process that needs force kill."""
        # Set up temp files
        config.tracking_file = str(tmp_path / "tracking.json")

        # Create orchestrator
        orchestrator = Orchestrator(config)

        # Create mock hanging process
        mock_hanging_process = Mock(spec=subprocess.Popen)
        mock_hanging_process.terminate = Mock()
        mock_hanging_process.wait = Mock(
            side_effect=[subprocess.TimeoutExpired("test", 5), None]
        )  # First wait times out, second succeeds
        mock_hanging_process.kill = Mock()

        # Add mock process to orchestrator
        orchestrator.active_processes = {"hanging_process": mock_hanging_process}

        # Test stop processing
        result = orchestrator.stop_processing()

        # Verify results
        assert result["status"] == "stopped"
        assert "hanging_process" in result["stopped_processes"]

        # Verify process was terminated, then killed when hanging
        mock_hanging_process.terminate.assert_called_once()
        assert mock_hanging_process.wait.call_count == 2  # Called twice due to timeout
        mock_hanging_process.kill.assert_called_once()

    def test_orchestrator_stop_processing_with_error(self, config, tmp_path):
        """Test orchestrator stop_processing when a process fails to stop."""
        # Set up temp files
        config.tracking_file = str(tmp_path / "tracking.json")

        # Create orchestrator
        orchestrator = Orchestrator(config)

        # Create mock process that fails to terminate
        mock_error_process = Mock(spec=subprocess.Popen)
        mock_error_process.terminate = Mock(side_effect=Exception("Terminate failed"))

        # Add mock process to orchestrator
        orchestrator.active_processes = {"error_process": mock_error_process}

        # Test stop processing (should not raise exception)
        result = orchestrator.stop_processing()

        # Verify results - should still report stopped status even with errors
        assert result["status"] == "stopped"
        # Process that errored should not be in stopped_processes list
        assert "error_process" not in result["stopped_processes"]

        # Verify active processes were still cleared
        assert len(orchestrator.active_processes) == 0

    def test_status_panel_stop_processing(self, status_panel):
        """Test status panel stop_processing method with simplified architecture."""
        # Set processing state
        status_panel.is_processing = True

        # Test stop processing - no mocking needed for the simple method
        status_panel.stop_processing()

        # Verify UI state changes
        assert status_panel.is_processing is False

    def test_status_panel_early_completion_detection(self, status_panel, tmp_path):
        """Test that status panel properly detects completion based on backend status."""
        # Using already imported modules: asyncio, Mock, Orchestrator

        # Create output directory with recent files
        output_dir = tmp_path / "cutouts"
        output_dir.mkdir()

        # Create recent output file (simulating a previous run)
        recent_file = output_dir / "test_cutouts.zarr"
        recent_file.touch()

        # Set config output directory
        status_panel.config.output_dir = str(output_dir)

        # TEST CASE 1: No orchestrator means not processing
        BackendInterface._current_orchestrator = None

        # Get status synchronously for testing
        loop = asyncio.new_event_loop()
        status = loop.run_until_complete(BackendInterface.get_processing_status())
        loop.close()

        assert status["is_processing"] is False, "Should not be processing without orchestrator"
        assert status["total_sources"] == 0, "Should have 0 sources without orchestrator"

        # TEST CASE 2: Active orchestrator processing
        mock_orchestrator = Mock(spec=Orchestrator)
        # Create a complete mock object with all required attributes and to_dict method
        progress_mock = Mock()
        progress_mock.is_processing = True
        progress_mock.total_sources = 50000
        progress_mock.completed_sources = 100
        progress_mock.progress_percent = 0.2
        progress_mock.active_processes = 1
        progress_mock.max_workers = 4
        progress_mock.memory_used_gb = 2.5
        progress_mock.memory_total_gb = 16.0
        progress_mock.to_dict.return_value = {
            "is_processing": True,
            "total_sources": 50000,
            "completed_sources": 100,
            "progress_percent": 0.2,
            "active_processes": 1,
            "max_workers": 4,
            "memory_used_gb": 2.5,
            "memory_total_gb": 16.0,
        }
        mock_orchestrator.get_progress_for_ui.return_value = progress_mock
        BackendInterface._current_orchestrator = mock_orchestrator

        loop = asyncio.new_event_loop()
        status = loop.run_until_complete(BackendInterface.get_processing_status())
        loop.close()

        assert status["is_processing"] is True, "Should be processing when orchestrator is active"
        assert status["total_sources"] == 50000, "Should report correct total sources"
        assert status["completed_sources"] == 100, "Should report progress"

        # TEST CASE 3: Orchestrator reports completion
        completion_mock = Mock()
        completion_mock.is_processing = False
        completion_mock.total_sources = 50000
        completion_mock.completed_sources = 50000
        completion_mock.progress_percent = 100.0
        completion_mock.active_processes = 0
        completion_mock.max_workers = 4
        completion_mock.memory_used_gb = 2.5
        completion_mock.memory_total_gb = 16.0
        completion_mock.to_dict.return_value = {
            "is_processing": False,
            "total_sources": 50000,
            "completed_sources": 50000,
            "progress_percent": 100.0,
            "active_processes": 0,
            "max_workers": 4,
            "memory_used_gb": 2.5,
            "memory_total_gb": 16.0,
        }
        mock_orchestrator.get_progress_for_ui.return_value = completion_mock

        loop = asyncio.new_event_loop()
        status = loop.run_until_complete(BackendInterface.get_processing_status())
        loop.close()

        assert status["is_processing"] is False, "Should not be processing when complete"
        assert status["completed_sources"] == 50000, "Should show all sources completed"
        assert status["progress_percent"] == 100.0, "Should show 100% progress"

        # Clean up
        BackendInterface._current_orchestrator = None

    def test_status_panel_fast_completion_ui_update(self, status_panel):
        """Test status panel UI update for completion via handler."""
        # Set initial processing state
        status_panel.is_processing = True

        # Test completion handler method
        status_panel._handle_completion()

        # Verify UI state changes
        assert status_panel.is_processing is False
        assert status_panel.progress_bar.value == 100
        assert "completed successfully" in status_panel.processing_indicator.value

    def test_end_to_end_stop_flow(self, status_panel):
        """Test StatusPanel stop_processing method updates UI state correctly."""
        # Set panel to processing state
        status_panel.is_processing = True

        # Trigger stop from UI
        status_panel.stop_processing()

        # Verify UI state was updated
        assert status_panel.is_processing is False
