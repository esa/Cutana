#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for LoadBalancer module.
"""

import tempfile
from unittest.mock import Mock, patch

import pytest

from cutana.get_default_config import get_default_config
from cutana.loadbalancer import LoadBalancer


class TestLoadBalancer:
    """Test suite for LoadBalancer class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.load_balancer = LoadBalancer(progress_dir=self.temp_dir, session_id="test_session")
        # Prevent monitoring thread from starting during tests
        self.load_balancer.start_monitoring = Mock()

    def teardown_method(self):
        """Clean up after tests."""
        # Clean up temp files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test LoadBalancer initialization."""
        assert self.load_balancer.system_monitor is not None
        assert self.load_balancer.process_reader is not None
        assert self.load_balancer.memory_safety_margin == 0.1
        assert len(self.load_balancer.main_memory_samples) == 0
        assert len(self.load_balancer.worker_memory_history) == 0
        assert self.load_balancer.worker_memory_peak_mb is None
        assert self.load_balancer.main_process_memory_mb is None

    @patch("cutana.loadbalancer.SystemMonitor")
    def test_update_config_with_loadbalancing(self, mock_monitor):
        """Test configuration update with load balancing."""

        # Mock system resources
        mock_instance = Mock()
        mock_instance.get_system_resources.return_value = {
            "memory_total": 16 * 1024**3,  # 16GB
            "memory_available": 12 * 1024**3,  # 12GB
            "cpu_percent": 20.0,
            "memory_percent": 25.0,
            "resource_source": "system",
        }
        mock_instance.get_cpu_count.return_value = 8
        mock_instance._get_kubernetes_pod_limits.return_value = (None, None)  # No K8s limits
        mock_monitor.return_value = mock_instance

        lb = LoadBalancer(progress_dir=self.temp_dir, session_id="test")
        lb.system_monitor = mock_instance  # Use mocked instance

        # Test with large job
        config = get_default_config()
        config.max_workers = 16  # Set high so system resources are the limiting factor
        lb.update_config_with_loadbalancing(config, total_sources=1e6)

        assert config.loadbalancer.max_workers == 7  # 8 cores - 1
        assert config.loadbalancer.max_sources_per_process == 1e5  # Large job
        assert config.loadbalancer.N_batch_cutout_process == 1000
        assert config.loadbalancer.memory_limit_gb == pytest.approx(10.8, 0.1)  # 12GB * 0.9
        assert config.loadbalancer.cpu_count == 8

        # Test with small job
        config_small = get_default_config()
        config_small.max_workers = 16  # Set high so system resources are the limiting factor
        lb.update_config_with_loadbalancing(config_small, total_sources=50000)
        assert config_small.loadbalancer.max_sources_per_process == 12500  # Small job (<1M sources)

        # Test with unknown job size
        config_unknown = get_default_config()
        config_unknown.max_workers = 16  # Set high so system resources are the limiting factor
        lb.update_config_with_loadbalancing(config_unknown, total_sources=None)
        assert config_unknown.loadbalancer.max_sources_per_process == 1e5  # Default for unknown

    @patch("cutana.loadbalancer.SystemMonitor")
    def test_update_config_with_loadbalancing_kubernetes(self, mock_monitor):
        """Test configuration update with Kubernetes resource limits."""

        # Mock Kubernetes environment
        mock_instance = Mock()
        mock_instance.get_system_resources.return_value = {
            "memory_total": 8 * 1024**3,  # 8GB Kubernetes limit
            "memory_available": 6 * 1024**3,  # 6GB available
            "cpu_percent": 20.0,
            "memory_percent": 25.0,
            "resource_source": "kubernetes_pod",
        }
        mock_instance.get_cpu_count.return_value = 4
        mock_instance._get_kubernetes_pod_limits.return_value = (
            8 * 1024**3,
            2000,
        )  # 8GB memory, 2000 millicores (2 cores)
        mock_monitor.return_value = mock_instance

        lb = LoadBalancer(progress_dir=self.temp_dir, session_id="test")
        lb.system_monitor = mock_instance  # Use mocked instance
        config = get_default_config()
        config.max_workers = 16  # Set high so system resources are the limiting factor
        lb.update_config_with_loadbalancing(config, total_sources=100000)

        assert config.loadbalancer.resource_source == "kubernetes_pod"
        assert config.loadbalancer.max_workers == 1  # 2000 millicores (2 cores) - 1
        assert config.loadbalancer.memory_limit_gb == pytest.approx(5.4, 0.1)  # 6GB * 0.9

    def test_update_memory_statistics(self):
        """Test memory statistics update from process data."""
        # Create mock progress file data
        with patch.object(self.load_balancer.process_reader, "read_progress_file") as mock_read:
            mock_read.return_value = {
                "process_id": "test_process",
                "memory_footprint_mb": 1024.5,
                "memory_footprint_samples": [1000.0, 1020.0, 1024.5, 1015.0],
            }

            self.load_balancer.update_memory_statistics("test_process")

            # Check worker memory history was updated
            assert len(self.load_balancer.worker_memory_history) >= 1
            # Peak should be tracked
            assert self.load_balancer.worker_memory_peak_mb == 1024.5
            assert self.load_balancer.processes_measured == 1

    def test_update_memory_statistics_multiple(self):
        """Test memory statistics with multiple process updates."""
        # First process
        with patch.object(self.load_balancer.process_reader, "read_progress_file") as mock_read:
            mock_read.return_value = {"memory_footprint_samples": [1000.0, 1100.0, 1050.0]}
            self.load_balancer.update_memory_statistics("process1")

        # Second process
        with patch.object(self.load_balancer.process_reader, "read_progress_file") as mock_read:
            mock_read.return_value = {"memory_footprint_samples": [1200.0, 1250.0, 1225.0]}
            self.load_balancer.update_memory_statistics("process2")

        # Check worker memory history includes both processes
        assert len(self.load_balancer.worker_memory_history) >= 2
        assert self.load_balancer.worker_memory_peak_mb == 1250.0
        assert self.load_balancer.processes_measured == 2

    def test_memory_samples_limit(self):
        """Test that worker memory history manages size appropriately."""
        with patch.object(self.load_balancer.process_reader, "read_progress_file") as mock_read:
            for i in range(15):
                mock_read.return_value = {"memory_footprint_samples": [1000.0 + i * 10]}
                self.load_balancer.update_memory_statistics(f"process{i}")

        # Worker memory history tracks all updates but may be cleaned by window
        assert len(self.load_balancer.worker_memory_history) >= 1
        assert self.load_balancer.processes_measured == 15

    @patch("cutana.loadbalancer.SystemMonitor")
    def test_can_spawn_new_process_cpu_limit(self, mock_monitor):
        """Test spawn decision when CPU limit is reached."""
        mock_instance = Mock()
        mock_instance.get_system_resources.return_value = {
            "memory_available": 8 * 1024**3,
            "cpu_percent": 50.0,
        }
        mock_monitor.return_value = mock_instance

        lb = LoadBalancer(progress_dir=self.temp_dir, session_id="test")
        lb.cpu_limit = 4

        can_spawn, reason = lb.can_spawn_new_process(active_process_count=4)
        assert not can_spawn
        assert "CPU limit reached" in reason

    @patch("cutana.loadbalancer.SystemMonitor")
    def test_can_spawn_new_process_memory_limit(self, mock_monitor):
        """Test spawn decision when memory is insufficient."""
        mock_instance = Mock()
        mock_instance.get_system_resources.return_value = {
            "memory_available": 1 * 1024**3,  # Only 1GB available
            "cpu_percent": 20.0,
        }
        mock_monitor.return_value = mock_instance

        lb = LoadBalancer(progress_dir=self.temp_dir, session_id="test")
        lb.cpu_limit = 4
        lb.memory_limit_bytes = 8 * 1024**3
        lb.worker_memory_peak_mb = 1500.0  # 1.5GB peak measurements
        lb.processes_measured = 1  # Simulate having measurements from first worker
        lb.calibration_completed = True  # Mark calibration as completed

        can_spawn, reason = lb.can_spawn_new_process(active_process_count=2)
        assert not can_spawn
        assert "Insufficient worker memory" in reason

    def test_can_spawn_new_process_high_cpu(self):
        """Test spawn decision when CPU usage is too high."""
        lb = LoadBalancer(progress_dir=self.temp_dir, session_id="test")

        # Mock only the specific method that returns CPU usage
        with patch.object(lb.system_monitor, "get_system_resources") as mock_resources:
            mock_resources.return_value = {
                "memory_available": 8 * 1024**3,
                "memory_total": 16 * 1024**3,
                "cpu_percent": 95.0,  # Very high CPU
                "memory_percent": 50.0,
                "resource_source": "system",
            }

            lb.cpu_limit = 4
            lb.memory_limit_bytes = 8 * 1024**3
            lb.worker_memory_peak_mb = 500.0
            lb.processes_measured = 1
            lb.calibration_completed = True

            can_spawn, reason = lb.can_spawn_new_process(active_process_count=2)
            assert not can_spawn
            assert "CPU usage too high" in reason

    def test_can_spawn_new_process_success(self):
        """Test successful spawn decision."""
        lb = LoadBalancer(progress_dir=self.temp_dir, session_id="test")

        # Mock only the system resources method
        with patch.object(lb.system_monitor, "get_system_resources") as mock_resources:
            mock_resources.return_value = {
                "memory_available": 8 * 1024**3,
                "memory_total": 16 * 1024**3,
                "cpu_percent": 40.0,
                "memory_percent": 50.0,
                "resource_source": "system",
            }

            lb.cpu_limit = 4
            lb.memory_limit_bytes = 8 * 1024**3

            # For additional workers, need real memory measurements first
            can_spawn, reason = lb.can_spawn_new_process(active_process_count=0)
            assert can_spawn  # First worker always allowed

            # Second worker requires memory measurements
            can_spawn, reason = lb.can_spawn_new_process(active_process_count=1)
            assert not can_spawn
            assert "first worker to complete" in reason.lower()

            # Add memory measurements and mock job tracker to show progress
            lb.worker_memory_peak_mb = 1000.0
            lb.processes_measured = 1

            # Mock JobTracker to indicate progress has been made
            with patch("cutana.job_tracker.JobTracker") as mock_job_tracker:
                mock_tracker_instance = Mock()
                mock_tracker_instance.get_process_details.return_value = {
                    "test_process": {"completed_sources": 1}  # Some progress made
                }
                mock_job_tracker.return_value = mock_tracker_instance

                can_spawn, reason = lb.can_spawn_new_process(active_process_count=1)
                assert can_spawn
            assert "Resources available" in reason

    def test_get_spawn_recommendation(self):
        """Test spawn recommendation logic."""
        lb = LoadBalancer(progress_dir=self.temp_dir, session_id="test")

        # Mock only the system resources method
        with patch.object(lb.system_monitor, "get_system_resources") as mock_resources:
            mock_resources.return_value = {
                "memory_available": 8 * 1024**3,
                "memory_total": 16 * 1024**3,
                "cpu_percent": 30.0,
                "memory_percent": 50.0,
                "resource_source": "system",
            }

            lb.cpu_limit = 4
            lb.memory_limit_bytes = 8 * 1024**3
            lb.worker_memory_peak_mb = 1000.0
            lb.processes_measured = 1
            lb.calibration_completed = True

            # Test with pending work and resources available
            recommendation = lb.get_spawn_recommendation(
                active_processes={"proc1": {}, "proc2": {}}, pending_batches=10
            )
            assert recommendation["spawn_new"] is True
            assert recommendation["active_processes"] == 2
            assert recommendation["pending_batches"] == 10
            assert "system_resources" in recommendation
            assert "memory_stats" in recommendation

            # Test with no pending work
            recommendation = lb.get_spawn_recommendation(
                active_processes={"proc1": {}}, pending_batches=0
            )
            assert recommendation["spawn_new"] is False
            assert "No pending batches" in recommendation["reason"]

    @patch("cutana.loadbalancer.SystemMonitor")
    def test_get_resource_status(self, mock_monitor):
        """Test resource status reporting."""
        mock_instance = Mock()
        mock_instance.get_system_resources.return_value = {
            "memory_total": 16 * 1024**3,
            "memory_available": 8 * 1024**3,
            "cpu_percent": 45.0,
            "memory_percent": 50.0,
            "resource_source": "system",
        }
        mock_instance.get_cpu_count.return_value = 8
        mock_instance._get_kubernetes_pod_limits.return_value = (None, None)  # No K8s limits
        mock_monitor.return_value = mock_instance

        lb = LoadBalancer(progress_dir=self.temp_dir, session_id="test")
        lb.system_monitor = mock_instance  # Use mocked instance
        # No threading in new implementation

        try:
            lb.cpu_limit = 7
            lb.memory_limit_bytes = 12 * 1024**3
            lb.worker_memory_peak_mb = 1024.0
            lb.worker_memory_allocation_mb = 8000.0
            lb.main_process_memory_mb = 500.0
            lb.processes_measured = 5

            status = lb.get_resource_status()

            assert status["system"]["cpu_count"] == 8
            assert status["system"]["cpu_percent"] == 45.0
            assert status["system"]["memory_total_gb"] == pytest.approx(16.0, 0.1)
            assert status["system"]["resource_source"] == "system"
            assert status["limits"]["cpu_limit"] == 7
            assert status["limits"]["memory_limit_gb"] == pytest.approx(12.0, 0.1)
            assert status["performance"]["processes_measured"] == 5
            assert status["performance"]["worker_peak_mb"] == 1024.0
            assert status["performance"]["worker_allocation_mb"] == 8000.0
            assert status["performance"]["main_process_memory_mb"] == 500.0
        finally:
            # No monitoring thread cleanup needed
            pass
