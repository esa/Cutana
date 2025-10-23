#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Tests for enhanced progress display functionality with LoadBalancer integration.

Tests the complete data flow from LoadBalancer → Orchestrator → BackendInterface → StatusPanel
to ensure the enhanced progress display shows detailed LoadBalancer information.
"""

import pytest
from unittest.mock import Mock, patch

from cutana_ui.main_screen.status_panel import StatusPanel
from cutana_ui.utils.backend_interface import BackendInterface
from cutana.orchestrator import Orchestrator
from cutana import get_default_config


class TestEnhancedProgressDisplay:
    """Tests for enhanced progress display functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = get_default_config()
        config.num_sources = 100000
        config.max_workers = 4
        config.source_catalogue = "/tmp/test_catalogue.csv"
        config.output_dir = "/tmp/cutouts"
        config.output_format = "zarr"
        config.target_resolution = 256
        config.data_type = "float32"
        config.fits_extensions = ["VIS"]
        config.selected_extensions = [{"name": "VIS", "ext": "PRIMARY"}]
        config.normalisation_method = "linear"
        config.max_sources_per_process = 100000
        config.N_batch_cutout_process = 1000
        config.max_workflow_time_seconds = 600
        return config

    @pytest.fixture
    def mock_load_balancer_status(self):
        """Create mock LoadBalancer status data."""
        return {
            "system": {
                "cpu_count": 8,
                "cpu_percent": 45.0,
                "memory_total_gb": 32.0,
                "memory_available_gb": 18.5,
                "memory_percent": 42.2,
                "resource_source": "system",
            },
            "limits": {"cpu_limit": 7, "memory_limit_gb": 28.8, "safety_margin": 0.1},
            "performance": {
                "processes_measured": 3,
                "avg_memory_mb": 6400.0,
                "peak_memory_mb": 6800.0,
                "memory_samples": 3,
            },
        }

    @pytest.fixture
    def mock_orchestrator(self, config, mock_load_balancer_status):
        """Create mock orchestrator with LoadBalancer status."""
        with patch.object(Orchestrator, "__init__", return_value=None):
            orchestrator = Orchestrator.__new__(Orchestrator)
            orchestrator.config = config

            # Mock LoadBalancer
            mock_lb = Mock()
            mock_lb.get_resource_status.return_value = mock_load_balancer_status
            orchestrator.load_balancer = mock_lb

            # Mock JobTracker
            mock_job_tracker = Mock()
            mock_job_tracker.check_completion_status.return_value = {
                "has_progress_files": False,
                "is_fully_completed": False,
            }
            mock_job_tracker.get_status.return_value = {
                "total_sources": 100000,
                "completed_sources": 25000,
                "failed_sources": 0,
                "progress_percent": 25.0,
                "throughput": 85.3,
                "eta_seconds": 555,  # 9.25 minutes = 555 seconds
                "active_processes": 3,
                "process_details": {"proc1": {}, "proc2": {}, "proc3": {}},
                "total_memory_footprint_mb": 19200.0,
                "process_errors": 0,
                "process_warnings": 0,
                "total_errors": 0,
                "start_time": 1234567890,
                "current_time": 1234568000,
            }
            orchestrator.job_tracker = mock_job_tracker

            return orchestrator

    def test_orchestrator_get_progress_for_ui_includes_loadbalancer_fields(self, mock_orchestrator):
        """Test that orchestrator includes LoadBalancer fields in UI progress data."""
        # Call the method and convert to dict for testing
        ui_status_obj = mock_orchestrator.get_progress_for_ui()
        ui_status = ui_status_obj.to_dict()

        # Verify core progress metrics are present
        assert ui_status["total_sources"] == 100000
        assert ui_status["completed_sources"] == 25000
        assert ui_status["progress_percent"] == 25.0
        assert ui_status["throughput"] == 85.3
        assert ui_status["eta_seconds"] == 555

        # Verify LoadBalancer system information is present
        assert ui_status["memory_percent"] == 42.2
        assert ui_status["cpu_percent"] == 45.0
        assert ui_status["memory_total_gb"] == 32.0
        assert ui_status["memory_available_gb"] == 18.5
        assert ui_status["memory_used_gb"] == 13.5  # 32.0 - 18.5
        assert ui_status["resource_source"] == "system"

        # Verify LoadBalancer worker information is present
        assert ui_status["active_processes"] == 3
        assert ui_status["max_workers"] == 7
        assert ui_status["worker_memory_limit_gb"] == 28.8
        assert ui_status["avg_worker_memory_mb"] == 6400.0
        assert ui_status["peak_worker_memory_mb"] == 6800.0
        assert ui_status["processes_measured"] == 3

        # Verify processing status
        assert ui_status["is_processing"] is True

    @pytest.mark.asyncio
    async def test_backend_interface_passes_through_loadbalancer_fields(self, mock_orchestrator):
        """Test that BackendInterface passes through all LoadBalancer fields."""
        # Set up the backend interface with mock orchestrator
        BackendInterface._current_orchestrator = mock_orchestrator

        try:
            # Get status from backend interface
            status = await BackendInterface.get_processing_status()

            # Verify all LoadBalancer fields are passed through
            assert status["memory_used_gb"] == 13.5
            assert status["max_workers"] == 7
            assert status["worker_memory_limit_gb"] == 28.8
            assert status["avg_worker_memory_mb"] == 6400.0
            assert status["peak_worker_memory_mb"] == 6800.0
            assert status["processes_measured"] == 3
            assert status["resource_source"] == "system"

            # Verify core fields are still present
            assert status["is_processing"] is True
            assert status["total_sources"] == 100000
            assert status["completed_sources"] == 25000
            assert status["throughput"] == 85.3

        finally:
            BackendInterface._current_orchestrator = None

    @pytest.mark.asyncio
    async def test_backend_interface_no_orchestrator_includes_new_fields(self):
        """Test that BackendInterface returns new fields even when no orchestrator is active."""
        # Ensure no orchestrator is active
        BackendInterface._current_orchestrator = None

        # Get status
        status = await BackendInterface.get_processing_status()

        # Verify new LoadBalancer fields are present with default values
        assert status["memory_used_gb"] == 0.0
        assert status["max_workers"] == 0
        assert status["worker_memory_limit_gb"] == 0.0
        assert status["avg_worker_memory_mb"] == 0.0
        assert status["peak_worker_memory_mb"] == 0.0
        assert status["processes_measured"] == 0
        assert status["resource_source"] == "system"

        # Verify core fields
        assert status["is_processing"] is False
        assert status["total_sources"] == 0
        assert status["completed_sources"] == 0

    def test_status_panel_format_eta_hhmmss(self, config):
        """Test ETA formatting to HH:MM:SS format."""
        status_panel = StatusPanel(config)

        # Test various ETA values
        assert status_panel._format_eta_hhmmss(None) == "--:--:--"
        assert status_panel._format_eta_hhmmss(0) == "--:--:--"
        assert status_panel._format_eta_hhmmss(-10) == "--:--:--"
        assert status_panel._format_eta_hhmmss(45) == "00:00:45"  # 45 seconds
        assert status_panel._format_eta_hhmmss(125) == "00:02:05"  # 2 minutes 5 seconds
        assert status_panel._format_eta_hhmmss(3661) == "01:01:01"  # 1 hour 1 minute 1 second
        assert status_panel._format_eta_hhmmss(555) == "00:09:15"  # 9 minutes 15 seconds

    def test_status_panel_enhanced_display_format(self, config):
        """Test the enhanced progress display format."""
        status_panel = StatusPanel(config)

        # Test with realistic data
        completed = 25000
        total = 100000
        throughput = 85.3
        memory_used_gb = 13.5
        memory_total_gb = 32.0
        memory_pct = 42.2
        active_processes = 3
        max_workers = 7
        avg_worker_memory_mb = 6400.0
        eta = "00:09:15"

        # Call the display update method with correct arguments
        worker_allocation_mb = 6400.0
        worker_peak_mb = 6800.0
        worker_remaining_mb = 2000.0
        status_panel._update_stats_display(
            completed,
            total,
            throughput,
            memory_used_gb,
            memory_total_gb,
            memory_pct,
            active_processes,
            max_workers,
            worker_allocation_mb,
            worker_peak_mb,
            worker_remaining_mb,
            eta,
        )

        # Check that the HTML contains the expected format elements
        html_content = status_panel.stats_html.value

        # Verify the format contains expected elements (pipe-separated)
        assert "25,000 / 100,000" in html_content  # Progress with commas
        assert "85.3 sources/sec" in html_content  # Throughput
        assert "13.5GB / 32.0GB (42.2%) RAM" in html_content  # Memory info
        assert "3/7 workers" in html_content  # Worker count
        assert "worker alloc" in html_content  # Worker memory allocation info
        assert "ETA 00:09:15" in html_content  # ETA in HH:MM:SS

    def test_status_panel_enhanced_display_edge_cases(self, config):
        """Test enhanced display format handles edge cases."""
        status_panel = StatusPanel(config)

        # Test with zero/missing values - add missing arguments for correct signature
        status_panel._update_stats_display(
            0, 1000, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, "--:--:--"
        )

        html_content = status_panel.stats_html.value
        assert "0 / 1,000" in html_content
        assert "-- |" in html_content  # Throughput shows as "--" when zero
        assert "-- RAM" in html_content
        assert "-- workers" in html_content
        assert "worker alloc" in html_content
        assert "ETA --:--:--" in html_content

    def test_status_panel_handle_progress_update_with_loadbalancer_data(self, config):
        """Test that progress update handler correctly processes LoadBalancer data."""
        status_panel = StatusPanel(config)

        # Mock status data with all LoadBalancer fields
        mock_status = {
            "completed_sources": 25000,
            "total_sources": 100000,
            "progress_percent": 25.0,
            "throughput": 85.3,
            "memory_used_gb": 13.5,
            "memory_total_gb": 32.0,
            "memory_percent": 42.2,
            "active_processes": 3,
            "max_workers": 7,
            "avg_worker_memory_mb": 6400.0,
            "eta_seconds": 555,
        }

        # Call the handler
        status_panel._handle_progress_update(mock_status)

        # Verify progress bar was updated
        assert status_panel.progress_bar.value == 25.0

        # Verify stats display contains enhanced information
        html_content = status_panel.stats_html.value
        assert "25,000 / 100,000" in html_content
        assert "85.3 sources/sec" in html_content
        assert "13.5GB / 32.0GB (42.2%) RAM" in html_content
        assert "3/7 workers" in html_content
        assert "worker alloc" in html_content  # Worker allocation info in new format
        assert "ETA 00:09:15" in html_content
