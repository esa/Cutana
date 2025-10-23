#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
End-to-end tests for LoadBalancer integration with orchestrator.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np

from cutana.orchestrator import Orchestrator
from cutana.loadbalancer import LoadBalancer
from cutana.get_default_config import get_default_config


class TestE2ELoadBalancer:
    """End-to-end tests for LoadBalancer with full orchestrator integration."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = get_default_config()
        self.config.output_dir = str(self.temp_dir)
        self.config.log_level = "DEBUG"
        # Add dummy source catalogue path for validation
        self.config.source_catalogue = str(self.temp_dir / "test_catalogue.csv")
        # Fix channel_weights to have proper content
        self.config.channel_weights = {"PRIMARY": [1.0]}
        self.config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_catalogue(self, num_sources: int) -> pd.DataFrame:
        """Create a test catalogue with specified number of sources."""
        sources = []
        for i in range(num_sources):
            sources.append(
                {
                    "SourceID": f"TEST_{i:06d}",
                    "RA": np.random.uniform(0, 360),
                    "Dec": np.random.uniform(-90, 90),
                    "diameter_pixel": 64,
                    "fits_file_paths": f"['/fake/path/test_{i % 10}.fits']",  # Reuse 10 FITS files
                }
            )
        return pd.DataFrame(sources)

    @patch("cutana.orchestrator.subprocess.Popen")
    @patch("cutana.fits_reader.load_fits_file")
    def test_loadbalancer_initial_config_integration(self, mock_fits, mock_popen):
        """Test that orchestrator uses load balancer for initial configuration."""
        # Mock FITS loading
        mock_fits.return_value = (Mock(), {})

        # Mock subprocess
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process running
        mock_process.pid = 12345
        mock_process._temp_files = []
        mock_popen.return_value = mock_process

        # Create orchestrator
        orchestrator = Orchestrator(self.config)

        # Create small catalogue
        catalogue = self.create_test_catalogue(50)

        # Verify load balancer was created
        assert orchestrator.load_balancer is not None
        assert isinstance(orchestrator.load_balancer, LoadBalancer)

        # Mock system resources for predictable behavior
        with patch.object(
            orchestrator.load_balancer.system_monitor, "get_system_resources"
        ) as mock_resources:
            with patch.object(
                orchestrator.load_balancer.system_monitor, "get_cpu_count"
            ) as mock_cpu:
                mock_resources.return_value = {
                    "memory_total": 16 * 1024**3,
                    "memory_available": 12 * 1024**3,
                    "cpu_percent": 20.0,
                    "memory_percent": 25.0,
                    "resource_source": "system",
                }
                mock_cpu.return_value = 8

                # Create test config with proper defaults and update with load balancer
                from cutana.get_default_config import get_default_config

                test_config = get_default_config()
                test_config.max_workers = 8  # Set to 8 so the test can verify it gets reduced to 7
                orchestrator.load_balancer.update_config_with_loadbalancing(
                    test_config, len(catalogue)
                )

                assert test_config.loadbalancer.max_workers == 7  # 8 - 1
                assert test_config.loadbalancer.N_batch_cutout_process == 1000
                assert test_config.loadbalancer.memory_limit_gb > 0

    @patch("cutana.orchestrator.subprocess.Popen")
    @patch("cutana.fits_reader.load_fits_file")
    def test_loadbalancer_dynamic_spawning(self, mock_fits, mock_popen):
        """Test that load balancer controls dynamic process spawning."""
        # Mock FITS loading
        mock_fits.return_value = (Mock(), {})

        # Create orchestrator
        orchestrator = Orchestrator(self.config)
        catalogue = self.create_test_catalogue(20)

        # Mock system resources
        with patch.object(
            orchestrator.load_balancer.system_monitor, "get_system_resources"
        ) as mock_resources:
            with patch.object(
                orchestrator.load_balancer.system_monitor, "get_cpu_count"
            ) as mock_cpu:
                mock_resources.return_value = {
                    "memory_total": 16 * 1024**3,
                    "memory_available": 12 * 1024**3,
                    "cpu_percent": 20.0,
                    "memory_percent": 25.0,
                    "resource_source": "system",
                }
                mock_cpu.return_value = 4

                # Initialize load balancer limits by creating test config
                from cutana.get_default_config import get_default_config

                test_config = get_default_config()
                orchestrator.load_balancer.update_config_with_loadbalancing(
                    test_config, len(catalogue)
                )

                # Test spawn recommendation with no memory measurements
                recommendation = orchestrator.load_balancer.get_spawn_recommendation(
                    active_processes={}, pending_batches=5
                )
                assert recommendation["spawn_new"] is True
                assert "Initial worker spawn" in recommendation["reason"]

                # Simulate memory measurement from first process
                orchestrator.load_balancer.memory_samples = [2048.0]  # 2GB
                orchestrator.load_balancer.worker_memory_peak_mb = 2048.0
                orchestrator.load_balancer.worker_memory_allocation_mb = 8000.0
                orchestrator.load_balancer.processes_measured = 1

                # Test spawn with active processes
                recommendation = orchestrator.load_balancer.get_spawn_recommendation(
                    active_processes={"proc1": {}, "proc2": {}}, pending_batches=3
                )

                # Should allow spawn if resources permit
                if recommendation["spawn_new"]:
                    assert "Resources available" in recommendation["reason"]
                else:
                    assert (
                        "CPU limit" in recommendation["reason"]
                        or "memory" in recommendation["reason"].lower()
                    )

    @patch("cutana.orchestrator.subprocess.Popen")
    @patch("cutana.fits_reader.load_fits_file")
    def test_loadbalancer_memory_tracking(self, mock_fits, mock_popen):
        """Test that load balancer tracks memory usage from processes."""
        # Mock FITS loading
        mock_fits.return_value = (Mock(), {})

        # Create orchestrator
        orchestrator = Orchestrator(self.config)

        # Create mock process data in a progress file
        process_data = {
            "process_id": "test_process_001",
            "memory_footprint_mb": 1536.0,
            "memory_footprint_samples": [1400.0, 1500.0, 1536.0, 1520.0],
            "status": "completed",
            "total_sources": 100,
            "completed_sources": 100,
        }

        # Mock reading progress file
        with patch.object(
            orchestrator.load_balancer.process_reader, "read_progress_file"
        ) as mock_read:
            mock_read.return_value = process_data

            # Update memory statistics
            orchestrator.load_balancer.update_memory_statistics("test_process_001")

            # Verify memory tracking
            assert orchestrator.load_balancer.worker_memory_peak_mb == 1536.0
            assert orchestrator.load_balancer.processes_measured == 1
            assert len(orchestrator.load_balancer.worker_memory_history) >= 1

    def test_loadbalancer_kubernetes_detection(self):
        """Test that load balancer properly detects Kubernetes environment."""
        with patch("cutana.system_monitor.SystemMonitor._is_datalabs_environment") as mock_datalabs:
            with patch(
                "cutana.system_monitor.SystemMonitor._get_kubernetes_pod_limits"
            ) as mock_k8s:
                mock_datalabs.return_value = True
                mock_k8s.return_value = (4 * 1024**3, 2000)  # 4GB memory, 2 CPU cores

                from cutana.system_monitor import SystemMonitor

                monitor = SystemMonitor()

                # Mock system memory to be larger than k8s limit
                with patch("psutil.virtual_memory") as mock_mem:
                    with patch("psutil.cpu_percent") as mock_cpu:
                        mock_mem.return_value = Mock(
                            total=16 * 1024**3, available=12 * 1024**3, percent=25.0
                        )
                        mock_cpu.return_value = 20.0

                        resources = monitor.get_system_resources()

                        # Should use Kubernetes limits
                        assert resources["resource_source"] == "kubernetes_pod"
                        assert resources["memory_total"] == 4 * 1024**3

    def test_loadbalancer_reset_statistics(self):
        """Test that load balancer can reset statistics for new job."""
        lb = LoadBalancer(progress_dir=str(self.temp_dir), session_id="test")

        # Add some statistics
        lb.memory_samples = [1000, 2000, 3000]
        lb.worker_memory_peak_mb = 3000
        lb.main_process_memory_mb = 500
        lb.processes_measured = 3

        # Reset for new job
        lb.reset_statistics()

        # Verify reset
        assert len(lb.worker_memory_history) == 0
        assert lb.worker_memory_peak_mb is None
        assert lb.main_process_memory_mb is None
        assert lb.processes_measured == 0

    def test_loadbalancer_spawn_decision_scenarios(self):
        """Test various spawn decision scenarios."""
        lb = LoadBalancer(progress_dir=str(self.temp_dir), session_id="test")
        lb.cpu_limit = 4
        lb.memory_limit_bytes = 8 * 1024**3

        # Mock system monitor
        with patch.object(lb.system_monitor, "get_system_resources") as mock_resources:
            # Scenario 1: First worker - always allowed
            mock_resources.return_value = {
                "memory_available": 6 * 1024**3,
                "memory_total": 8 * 1024**3,
                "cpu_percent": 30.0,
            }
            can_spawn, reason = lb.can_spawn_new_process(0)
            assert can_spawn is True

            # Scenario 1b: Second worker without calibration - wait for first worker to complete sources
            can_spawn, reason = lb.can_spawn_new_process(1)
            assert can_spawn is False
            assert (
                "waiting for first worker to complete at least one source" in reason.lower()
                and "memory" in reason.lower()
            )

            # Mark calibration as completed (simulating first worker has completed sources)
            lb.calibration_completed = True
            lb.worker_memory_peak_mb = 1000.0
            lb.worker_memory_allocation_mb = 6000.0
            lb.processes_measured = 1
            can_spawn, reason = lb.can_spawn_new_process(1)
            assert can_spawn is True

            # Scenario 2: CPU limit reached
            can_spawn, reason = lb.can_spawn_new_process(4)
            assert can_spawn is False
            assert "CPU limit" in reason

            # Scenario 3: High CPU usage (with memory measurements)
            lb.worker_memory_peak_mb = 1000.0  # Add memory measurements first
            mock_resources.return_value = {
                "memory_available": 6 * 1024**3,
                "memory_total": 8 * 1024**3,
                "cpu_percent": 95.0,
            }
            can_spawn, reason = lb.can_spawn_new_process(2)
            assert can_spawn is False
            assert "CPU usage too high" in reason

            # Scenario 4: Low memory with measurements
            lb.worker_memory_peak_mb = 3000.0  # 3GB
            mock_resources.return_value = {
                "memory_available": 2 * 1024**3,  # Only 2GB left
                "memory_total": 8 * 1024**3,
                "cpu_percent": 30.0,
            }
            can_spawn, reason = lb.can_spawn_new_process(2)
            assert can_spawn is False
            assert "Insufficient" in reason
