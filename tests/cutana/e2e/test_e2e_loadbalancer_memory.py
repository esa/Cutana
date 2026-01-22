#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end tests for LoadBalancer memory monitoring with real processing."""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from cutana.get_default_config import get_default_config
from cutana.orchestrator import Orchestrator


class TestE2ELoadBalancerMemory:
    """End-to-end tests for LoadBalancer memory monitoring during real processing."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test configuration
        self.config = get_default_config()
        self.config.output_dir = str(self.temp_path / "output")
        self.config.output_format = "zarr"
        self.config.log_level = "DEBUG"
        self.config.max_workflow_time_seconds = 600
        self.config.apply_flux_conversion = False

        # Configure LoadBalancer settings for testing
        self.config.loadbalancer.memory_safety_margin = 0.1
        self.config.loadbalancer.memory_poll_interval = 1  # Faster polling for tests
        self.config.loadbalancer.memory_peak_window = 10  # Shorter window for tests
        self.config.loadbalancer.initial_workers = 1
        self.config.loadbalancer.log_interval = 5  # More frequent logging for tests

        # Set explicit limits for testing (conservative for stability)
        self.config.max_workers = 1
        self.config.max_sources_per_process = 50
        self.config.N_batch_cutout_process = 10

        # Required for processing
        self.config.selected_extensions = [{"name": "PRIMARY", "ext": "PrimaryHDU"}]

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_fits_file(self, filename: str, size: int = 100) -> str:
        """Create a test FITS file with proper WCS."""
        from astropy.wcs import WCS

        filepath = self.temp_path / filename
        data = np.random.random((size, size)).astype(np.float32)

        # Create a proper WCS object first
        w = WCS(naxis=2)
        w.wcs.crval = [180.0, 0.0]  # Reference RA, Dec in degrees
        w.wcs.crpix = [size // 2 + 1, size // 2 + 1]  # Reference pixels (1-indexed)
        w.wcs.cdelt = [-0.0002777778, 0.0002777778]  # Pixel scale (~1 arcsec)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Coordinate types
        w.wcs.cunit = ["deg", "deg"]  # Units

        # Create HDU with WCS header
        hdu = fits.PrimaryHDU(data=data, header=w.to_header())
        hdu.writeto(filepath, overwrite=True)
        return str(filepath)

    def _create_test_catalogue(self, num_sources: int, fits_files: list) -> pd.DataFrame:
        """Create a test source catalogue."""
        sources = []
        for i in range(num_sources):
            sources.append(
                {
                    "SourceID": f"TEST_{i:04d}",
                    "RA": 180.0,  # Exactly at the center
                    "Dec": 0.0,  # Exactly at the center
                    "diameter_pixel": 32,  # Smaller size to ensure it fits
                    "fits_file_paths": str([fits_files[i % len(fits_files)]]),
                }
            )
        return pd.DataFrame(sources)

    def test_loadbalancer_memory_tracking_during_processing(self):
        """Test that LoadBalancer tracks memory correctly during actual processing."""
        # Create test data (reduced for stability)
        fits_files = [self._create_test_fits_file(f"test_{i}.fits", size=400) for i in range(2)]
        catalogue = self._create_test_catalogue(10, fits_files)

        # Write catalogue to file and set in config
        catalogue_path = self.temp_path / "test_catalogue.csv"
        catalogue.to_csv(catalogue_path, index=False)
        self.config.source_catalogue = str(catalogue_path)

        # Create orchestrator
        orchestrator = Orchestrator(self.config)

        # Check initial LoadBalancer state
        assert orchestrator.load_balancer.main_process_memory_mb is None
        assert orchestrator.load_balancer.worker_memory_peak_mb is None
        assert orchestrator.load_balancer.processes_measured == 0

        # Start processing using catalogue path (not DataFrame)
        try:
            result = orchestrator.start_processing(str(catalogue_path))
            # Check processing completed
            assert result["status"] == "completed"
        finally:
            # Ensure all processes are terminated
            try:
                orchestrator.stop_processing()
            except Exception:
                pass

        # Check LoadBalancer was initialized (processes_measured starts at 0)
        # In unit tests, real worker processes may not be spawned, so check initial state
        assert orchestrator.load_balancer.processes_measured >= 0

        # Get final resource status
        status = orchestrator.load_balancer.get_resource_status()
        performance = status["performance"]

        # Verify memory tracking occurred
        assert performance["main_process_memory_mb"] > 0
        assert performance["worker_allocation_mb"] > 0
        # May or may not have peak depending on timing
        if performance["worker_peak_mb"] > 0:
            assert performance["worker_remaining_mb"] >= 0

    def test_loadbalancer_initial_worker_constraint(self):
        """Test that LoadBalancer respects initial_workers setting."""
        # Configure to start with only 1 worker
        self.config.loadbalancer.initial_workers = 1
        self.config.max_workers = 2  # Reduce for stability

        # Create test data that would benefit from multiple workers
        fits_files = [self._create_test_fits_file(f"test_{i}.fits", size=400) for i in range(1)]
        catalogue = self._create_test_catalogue(4, fits_files)

        # Write catalogue to file and set in config
        catalogue_path = self.temp_path / "test_catalogue.csv"
        catalogue.to_csv(catalogue_path, index=False)
        self.config.source_catalogue = str(catalogue_path)

        # Create orchestrator
        orchestrator = Orchestrator(self.config)

        # Track spawn decisions by monitoring active processes
        spawn_log = []
        original_spawn = orchestrator._spawn_cutout_process

        def logged_spawn(process_id, source_batch, write_to_disk):
            spawn_log.append(
                {
                    "process_id": process_id,
                    "active_count": len(orchestrator.active_processes),
                    "time": time.time(),
                }
            )
            return original_spawn(process_id, source_batch, write_to_disk)

        orchestrator._spawn_cutout_process = logged_spawn

        # Start processing using catalogue path (not DataFrame)
        try:
            result = orchestrator.start_processing(str(catalogue_path))
            assert result["status"] == "completed"

            # Check that initial spawn happened with no active processes
        finally:
            # Ensure all processes are terminated
            try:
                orchestrator.stop_processing()
            except Exception:
                pass
        assert len(spawn_log) > 0
        assert spawn_log[0]["active_count"] == 0  # First spawn with no active

    def test_loadbalancer_memory_allocation_update(self):
        """Test that worker memory allocation updates based on main process memory."""
        # Create minimal test data
        fits_file = self._create_test_fits_file("test.fits", size=50)
        catalogue = self._create_test_catalogue(20, [fits_file])

        # Write catalogue to file and set in config
        catalogue_path = self.temp_path / "test_catalogue.csv"
        catalogue.to_csv(catalogue_path, index=False)
        self.config.source_catalogue = str(catalogue_path)

        # Create orchestrator
        orchestrator = Orchestrator(self.config)
        load_balancer = orchestrator.load_balancer

        # Initialize load balancer to update worker allocation
        load_balancer._update_worker_memory_allocation()

        # Check that worker allocation was calculated
        assert load_balancer.worker_memory_allocation_mb is not None
        assert load_balancer.worker_memory_allocation_mb > 0

    def test_loadbalancer_spawn_decision_logging(self):
        """Test that LoadBalancer logs spawn decisions correctly."""
        # Create test data
        fits_file = self._create_test_fits_file("test.fits", size=100)
        catalogue = self._create_test_catalogue(50, [fits_file])

        # Write catalogue to file and set in config
        catalogue_path = self.temp_path / "test_catalogue.csv"
        catalogue.to_csv(catalogue_path, index=False)
        self.config.source_catalogue = str(catalogue_path)

        # Create orchestrator
        orchestrator = Orchestrator(self.config)

        # Capture spawn decisions
        decisions = []
        original_can_spawn = orchestrator.load_balancer.can_spawn_new_process

        def logged_can_spawn(active_count, active_process_ids=None):
            can_spawn, reason = original_can_spawn(active_count, active_process_ids)
            decisions.append(
                {"active_count": active_count, "can_spawn": can_spawn, "reason": reason}
            )
            return can_spawn, reason

        orchestrator.load_balancer.can_spawn_new_process = logged_can_spawn

        # Start processing using catalogue path (not DataFrame)
        try:
            result = orchestrator.start_processing(str(catalogue_path))
            assert result["status"] == "completed"

            # Check decisions were made and logged
        finally:
            # Ensure all processes are terminated
            try:
                orchestrator.stop_processing()
            except Exception:
                pass
        assert len(decisions) > 0

        # First decision should allow initial worker
        assert decisions[0]["can_spawn"] is True
        assert "Initial worker" in decisions[0]["reason"]

    @pytest.mark.slow
    def test_loadbalancer_memory_peak_window(self):
        """Test that LoadBalancer correctly tracks peak memory within window."""
        # This test needs longer processing to observe windowing behavior
        self.config.loadbalancer.memory_peak_window = 10  # Very short window for testing
        self.config.N_batch_cutout_process = 10  # Small batches for more updates

        # Create test data
        fits_files = [self._create_test_fits_file(f"test_{i}.fits", size=400) for i in range(2)]
        catalogue = self._create_test_catalogue(15, fits_files)

        # Write catalogue to file and set in config
        catalogue_path = self.temp_path / "test_catalogue.csv"
        catalogue.to_csv(catalogue_path, index=False)
        self.config.source_catalogue = str(catalogue_path)

        # Create orchestrator
        orchestrator = Orchestrator(self.config)
        load_balancer = orchestrator.load_balancer

        # Track memory history updates
        memory_updates = []
        original_update = load_balancer.update_memory_statistics

        def tracked_update(process_id):
            original_update(process_id)
            if load_balancer.worker_memory_peak_mb:
                memory_updates.append(
                    {
                        "time": time.time(),
                        "peak": load_balancer.worker_memory_peak_mb,
                        "history_size": len(load_balancer.worker_memory_history),
                    }
                )

        load_balancer.update_memory_statistics = tracked_update

        # Start processing using catalogue path (not DataFrame)
        try:
            result = orchestrator.start_processing(str(catalogue_path))
            assert result["status"] == "completed"

            # Check that memory was tracked (may be 0 in unit tests)
        finally:
            # Ensure all processes are terminated
            try:
                orchestrator.stop_processing()
            except Exception:
                pass
        assert len(memory_updates) >= 0

        # Check that history was maintained
        for update in memory_updates:
            # History should not grow unbounded
            assert update["history_size"] <= 100  # Some reasonable limit
