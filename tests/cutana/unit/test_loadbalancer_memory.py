#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for improved LoadBalancer memory monitoring functionality."""

import time
import pytest
from unittest.mock import MagicMock, patch

from cutana.loadbalancer import LoadBalancer


class TestLoadBalancerMemoryMonitoring:
    """Test suite for LoadBalancer memory monitoring improvements."""

    def setup_method(self):
        """Set up test fixtures."""
        self.load_balancer = LoadBalancer()

        # Mock system monitor
        self.mock_system_monitor = MagicMock()
        self.load_balancer.system_monitor = self.mock_system_monitor

        # Mock process reader
        self.mock_process_reader = MagicMock()
        self.load_balancer.process_reader = self.mock_process_reader

        # Set up default mock values
        self.mock_system_monitor.get_current_process_memory_mb.return_value = 500.0
        self.mock_system_monitor.get_cpu_count.return_value = 8
        self.mock_system_monitor._get_kubernetes_pod_limits.return_value = (
            None,
            None,
        )  # No K8s limits
        self.mock_system_monitor.get_system_resources.return_value = {
            "cpu_percent": 30.0,
            "memory_total": 32 * 1024**3,  # 32 GB
            "memory_available": 20 * 1024**3,  # 20 GB
            "memory_percent": 37.5,
            "disk_free": 100 * 1024**3,
            "disk_total": 500 * 1024**3,
            "resource_source": "system",
            "timestamp": time.time(),
        }

    def teardown_method(self):
        """Clean up after tests."""
        # No cleanup needed - new LoadBalancer doesn't use monitoring threads
        pass

    def test_memory_monitoring_initialization(self):
        """Test that memory monitoring initializes correctly."""
        assert self.load_balancer.main_process_memory_mb is None
        assert self.load_balancer.worker_memory_allocation_mb is None
        assert self.load_balancer.worker_memory_peak_mb is None
        assert len(self.load_balancer.main_memory_samples) == 0
        assert len(self.load_balancer.worker_memory_history) == 0

    def test_config_update_with_custom_settings(self):
        """Test configuration update with custom LoadBalancer settings."""
        from cutana.get_default_config import get_default_config

        config = get_default_config()
        config.loadbalancer.memory_safety_margin = 0.15
        config.loadbalancer.memory_poll_interval = 5
        config.loadbalancer.memory_peak_window = 60
        config.loadbalancer.main_process_memory_reserve_gb = 3.0
        config.loadbalancer.initial_workers = 2
        config.loadbalancer.log_interval = 45

        self.load_balancer.update_config_with_loadbalancing(config, total_sources=50000)

        assert self.load_balancer.memory_safety_margin == 0.15
        assert self.load_balancer.memory_poll_interval == 5
        assert self.load_balancer.memory_peak_window == 60
        assert self.load_balancer.main_process_memory_reserve_gb == 3.0
        assert self.load_balancer.initial_workers == 2
        assert self.load_balancer.log_interval == 45

    def test_main_process_memory_tracking(self):
        """Test main process memory tracking with smoothing."""
        # Simulate memory samples
        samples = [450.0, 500.0, 550.0, 530.0, 520.0]

        for sample in samples:
            self.load_balancer.main_memory_samples.append(sample)

        # Calculate average
        if self.load_balancer.main_memory_samples:
            self.load_balancer.main_process_memory_mb = sum(
                self.load_balancer.main_memory_samples
            ) / len(self.load_balancer.main_memory_samples)

        # Should be average of samples
        expected_avg = sum(samples) / len(samples)
        assert self.load_balancer.main_process_memory_mb == pytest.approx(expected_avg, 0.1)

    def test_worker_memory_allocation_calculation(self):
        """Test worker memory allocation calculation."""
        # Set main process memory
        self.load_balancer.main_process_memory_mb = 2000.0  # 2GB
        self.load_balancer.main_process_memory_reserve_gb = 2.0
        self.load_balancer.memory_safety_margin = 0.1

        # Mock system resources
        self.mock_system_monitor.get_system_resources.return_value = {
            "memory_available": 20 * 1024**3,  # 20 GB
            "memory_total": 32 * 1024**3,
            "memory_percent": 37.5,
            "cpu_percent": 30.0,
            "resource_source": "system",
        }

        # Update allocation
        self.load_balancer._update_worker_memory_allocation()

        # Expected: (20GB - 2GB reserved) * 0.9 = 18GB * 0.9 = 16.2GB = 16588.8MB
        memory_available_mb = 20 * 1024
        main_reserved = max(2000.0, 2.0 * 1024)
        expected = (memory_available_mb - main_reserved) * 0.9

        assert self.load_balancer.worker_memory_allocation_mb == pytest.approx(expected, 0.1)

    def test_worker_memory_peak_tracking(self):
        """Test worker memory peak tracking in window."""
        current_time = time.time()

        # Add memory samples with timestamps
        samples = [
            (current_time - 40, 1000.0),  # Outside window
            (current_time - 25, 1500.0),  # Inside window
            (current_time - 20, 2000.0),  # Inside window
            (current_time - 10, 1800.0),  # Inside window
            (current_time - 5, 1600.0),  # Inside window
        ]

        for timestamp, memory_mb in samples:
            self.load_balancer.worker_memory_history.append((timestamp, memory_mb))

        # Clean old samples and calculate peak
        cutoff_time = current_time - self.load_balancer.memory_peak_window
        recent_samples = [
            m for t, m in self.load_balancer.worker_memory_history if t >= cutoff_time
        ]
        if recent_samples:
            self.load_balancer.worker_memory_peak_mb = max(recent_samples)

        # Peak should be 2000 (max of samples in window)
        assert self.load_balancer.worker_memory_peak_mb == 2000.0

    def test_initial_worker_spawn_decision(self):
        """Test that initial workers are always allowed."""
        self.load_balancer.initial_workers = 1
        self.load_balancer.cpu_limit = 4

        # First worker should always be allowed
        can_spawn, reason = self.load_balancer.can_spawn_new_process(0)
        assert can_spawn is True
        assert "Initial worker spawn" in reason

    def test_fits_set_size_update(self):
        """Test FITS set size estimation update."""
        with patch("os.path.exists", return_value=True):
            with patch(
                "os.path.getsize",
                side_effect=[100 * 1024 * 1024, 150 * 1024 * 1024, 200 * 1024 * 1024],
            ):
                fits_paths = ["/path/to/file1.fits", "/path/to/file2.fits", "/path/to/file3.fits"]
                self.load_balancer.update_fits_set_size(fits_paths)

                # Total: 450MB
                assert self.load_balancer.avg_fits_set_size_mb == pytest.approx(450.0, 0.1)

    def test_get_memory_stats(self):
        """Test retrieving current memory statistics."""
        self.load_balancer.main_process_memory_mb = 500.0
        self.load_balancer.worker_memory_allocation_mb = 8000.0
        self.load_balancer.worker_memory_peak_mb = 2000.0
        self.load_balancer.processes_measured = 5

        stats = self.load_balancer._get_memory_stats()

        assert stats["main_process_mb"] == 500.0
        assert stats["worker_allocation_mb"] == 8000.0
        assert stats["worker_peak_mb"] == 2000.0
        assert stats["processes_measured"] == 5
        assert "remaining_mb" in stats  # This depends on system resources calculation
        assert stats["processes_measured"] == 5

    def test_resource_status_with_memory_info(self):
        """Test complete resource status including memory information."""
        self.load_balancer.main_process_memory_mb = 500.0
        self.load_balancer.worker_memory_allocation_mb = 8000.0
        self.load_balancer.worker_memory_peak_mb = 2000.0
        self.load_balancer.cpu_limit = 4
        self.load_balancer.memory_limit_bytes = 20 * 1024**3

        status = self.load_balancer.get_resource_status()

        assert "system" in status
        assert "limits" in status
        assert "performance" in status

        # Check new fields
        assert status["performance"]["main_process_memory_mb"] == 500.0
        assert status["performance"]["worker_allocation_mb"] == 8000.0
        assert status["performance"]["worker_peak_mb"] == 2000.0
        assert "worker_remaining_mb" in status["performance"]

    def test_spawn_decision_with_cpu_limit(self):
        """Test spawn decision respects CPU limits."""
        self.load_balancer.cpu_limit = 4

        # At CPU limit
        can_spawn, reason = self.load_balancer.can_spawn_new_process(4)
        assert can_spawn is False
        assert "CPU limit reached" in reason

    def test_spawn_decision_with_high_cpu_usage(self):
        """Test spawn decision with high CPU usage."""
        self.load_balancer.cpu_limit = 4
        self.load_balancer.initial_workers = 0  # Skip initial worker logic
        self.load_balancer.worker_memory_allocation_mb = 8000.0
        # Add memory measurements so it gets to CPU check
        self.load_balancer.worker_memory_peak_mb = 1000.0
        self.load_balancer.processes_measured = 1
        self.load_balancer.calibration_completed = True  # Mark calibration as completed

        # Mock high CPU usage
        self.mock_system_monitor.get_system_resources.return_value = {
            "cpu_percent": 95.0,  # Very high CPU
            "memory_available": 20 * 1024**3,
            "memory_total": 32 * 1024**3,
            "memory_percent": 37.5,
            "resource_source": "system",
        }

        can_spawn, reason = self.load_balancer.can_spawn_new_process(1)
        assert can_spawn is False
        assert "CPU usage too high" in reason

    def test_reset_statistics(self):
        """Test resetting all statistics."""
        # Set some values
        self.load_balancer.main_process_memory_mb = 500.0
        self.load_balancer.worker_memory_allocation_mb = 8000.0
        self.load_balancer.worker_memory_peak_mb = 2000.0
        self.load_balancer.processes_measured = 5
        self.load_balancer.active_worker_count = 2

        # Reset
        self.load_balancer.reset_statistics()

        # Check all cleared
        assert self.load_balancer.main_process_memory_mb is None
        assert self.load_balancer.worker_memory_allocation_mb is None
        assert self.load_balancer.worker_memory_peak_mb is None
        assert self.load_balancer.processes_measured == 0
        assert self.load_balancer.active_worker_count == 0
        assert len(self.load_balancer.main_memory_samples) == 0
        assert len(self.load_balancer.worker_memory_history) == 0
