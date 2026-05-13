#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""E2E tests for status panel integration with backend and progress tracking."""

from unittest.mock import Mock

import pytest

from cutana import get_default_config
from cutana.job_tracker import JobTracker
from cutana.orchestrator import Orchestrator
from cutana.progress_report import ProgressReport
from cutana_ui.main_screen.status_panel import StatusPanel
from cutana_ui.utils.backend_interface import BackendInterface


class TestStatusPanelE2E:
    """End-to-end tests for status panel progress monitoring."""

    @pytest.fixture
    def config(self):
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
        return StatusPanel(config)

    @pytest.fixture(autouse=True)
    def clear_backend_orchestrator(self):
        BackendInterface._current_orchestrator = None
        yield
        BackendInterface._current_orchestrator = None

    @pytest.mark.asyncio
    async def test_status_panel_direct_updates(self, config, status_panel):
        """Test that status panel receives direct updates from orchestrator."""
        assert status_panel.is_processing is False

        status_panel.start_processing()
        assert status_panel.is_processing is True

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
        status_panel.receive_status_UI_update(progress_report)
        assert status_panel.progress_bar.value == 50.0

        completion_report = ProgressReport(
            total_sources=100,
            completed_sources=100,
            progress_percent=100.0,
            is_processing=False,
            active_processes=0,
            max_workers=4,
        )
        status_panel.receive_status_UI_update(completion_report)
        assert status_panel.progress_bar.value == 100.0

    @pytest.mark.asyncio
    async def test_status_panel_progress_file_integration(self, config, tmp_path):
        """Test status panel correctly reflects progress from file-based tracking."""
        progress_dir = tmp_path / "progress"
        progress_dir.mkdir()

        job_tracker = JobTracker(progress_dir=str(progress_dir))
        job_tracker.start_job(total_sources=100)
        job_tracker.register_process("test_process_001", sources_assigned=50)
        job_tracker.update_process_progress("test_process_001", {"completed_sources": 25})

        status = job_tracker.get_status()
        assert status["total_sources"] == 100
        assert status["completed_sources"] == 25
        assert status["progress_percent"] == 25.0
        assert status["active_processes"] == 1

    @pytest.mark.asyncio
    async def test_backend_error_handling(self, status_panel, config):
        """Test that backend errors are handled gracefully."""
        mock_orchestrator = Mock(spec=Orchestrator)
        mock_orchestrator.get_progress_for_ui.side_effect = RuntimeError("Test error")
        BackendInterface._current_orchestrator = mock_orchestrator

        status = await BackendInterface.get_processing_status()
        assert "error" in status
        assert status["is_processing"] is True
