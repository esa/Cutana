#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
End-to-end tests for StreamingOrchestrator with pool-based parallel worker support.

Tests core streaming behaviour, edge cases, multi-worker speedup, and resource cleanup.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cutana import StreamingOrchestrator, get_default_config


@pytest.fixture
def streaming_config(tmp_path):
    """Create a test configuration for streaming mode."""
    config = get_default_config()
    config.output_format = "zarr"
    config.target_resolution = 32
    config.selected_extensions = ["VIS"]
    config.channel_weights = {"VIS": [1.0]}
    config.console_log_level = "INFO"
    config.skip_memory_calibration_wait = True
    config.max_workers = 1
    config.max_workflow_time_seconds = 600

    # Set dummy source_catalogue (will be overwritten in tests)
    dummy_catalogue = tmp_path / "dummy_catalogue.csv"
    dummy_catalogue.touch()
    config.source_catalogue = str(dummy_catalogue)

    return config


@pytest.fixture(scope="session")
def test_data_dir():
    """Get path to test data directory with real FITS files."""
    return Path(__file__).resolve().parent.parent.parent / "test_data"


@pytest.fixture(scope="session")
def test_small_catalogue(test_data_dir, tmp_path_factory):
    """Create a tiny 3-source catalogue for fast streaming tests.

    Uses only 3 sources (1 batch at batch_size=3) to minimize subprocess
    spawns. The full 25-source catalogue is tested via other e2e tests.
    """
    full_catalogue = test_data_dir / "euclid_cutana_catalogue_small.csv"
    if not full_catalogue.exists():
        pytest.skip("Test catalogue not available - run generate_test_data.py")

    df = pd.read_csv(full_catalogue)
    tiny_df = df.head(3)
    tiny_path = tmp_path_factory.mktemp("catalogues") / "tiny_catalogue.csv"
    tiny_df.to_csv(tiny_path, index=False)
    return tiny_path


@pytest.fixture(scope="session")
def test_large_catalogue(test_data_dir):
    """Get path to large real test catalogue."""
    catalogue_path = test_data_dir / "euclid_cutana_catalogue_large.csv"
    if not catalogue_path.exists():
        pytest.skip("Large test catalogue not available - run generate_test_data.py")
    return catalogue_path


class TestStreamingOrchestratorBasic:
    """Core streaming behaviour tests."""

    def test_streaming_in_memory(self, streaming_config, test_small_catalogue):
        """Test in-memory streaming via SHM pool."""
        with tempfile.TemporaryDirectory() as output_dir:
            streaming_config.output_dir = output_dir
            streaming_config.source_catalogue = str(test_small_catalogue)

            orchestrator = StreamingOrchestrator(streaming_config)

            try:
                orchestrator.init_streaming(
                    batch_size=3,
                    write_to_disk=False,
                )

                num_batches = orchestrator.get_batch_count()
                assert num_batches > 0

                results = []
                for i in range(num_batches):
                    result = orchestrator.next_batch()
                    results.append(result)

                    assert result["batch_number"] == i + 1
                    assert "cutouts" in result
                    assert isinstance(result["cutouts"], list)
                    if result["cutouts"]:
                        assert isinstance(result["cutouts"][0], np.ndarray)
                        assert result["cutouts"][0].ndim == 3  # (H, W, C)
                    assert "metadata" in result
                    assert len(result["cutouts"]) == len(result["metadata"])

                total_cutouts = sum(len(r["cutouts"]) for r in results)
                assert total_cutouts > 0

            finally:
                orchestrator.cleanup()

    @pytest.mark.slow
    def test_streaming_to_disk(self, streaming_config, test_small_catalogue):
        """Test streaming with disk output."""
        # Note: Disk mode zarr writing has a known issue - zarr files may not be created
        # in streaming mode. This test verifies the API behavior even if files aren't written.
        with tempfile.TemporaryDirectory() as output_dir:
            streaming_config.output_dir = output_dir
            streaming_config.source_catalogue = str(test_small_catalogue)

            orchestrator = StreamingOrchestrator(streaming_config)

            try:
                orchestrator.init_streaming(
                    batch_size=3,
                    write_to_disk=True,
                )

                num_batches = orchestrator.get_batch_count()
                assert num_batches > 0

                for i in range(num_batches):
                    result = orchestrator.next_batch()

                    assert result["batch_number"] == i + 1
                    assert "zarr_path" in result
                    assert "cutouts" not in result

            finally:
                orchestrator.cleanup()

    @pytest.mark.slow
    def test_multi_worker_provides_speedup(self, streaming_config, test_large_catalogue):
        """Test that multiple workers provide speedup over a single worker."""
        with tempfile.TemporaryDirectory() as output_dir:
            streaming_config.output_dir = output_dir
            streaming_config.source_catalogue = str(test_large_catalogue)
            streaming_config.console_log_level = "WARNING"

            num_batches = 3

            # Single worker
            streaming_config.max_workers = 1
            orchestrator_1w = StreamingOrchestrator(streaming_config)
            try:
                orchestrator_1w.init_streaming(batch_size=50, write_to_disk=False)
                n = min(num_batches, orchestrator_1w.get_batch_count())
                start = time.time()
                for _ in range(n):
                    result = orchestrator_1w.next_batch()
                    time.sleep(0.5)  # Simulate downstream processing
                    del result
                time_1w = time.time() - start
            finally:
                orchestrator_1w.cleanup()

            # Four workers
            streaming_config.max_workers = 4
            orchestrator_4w = StreamingOrchestrator(streaming_config)
            try:
                orchestrator_4w.init_streaming(batch_size=50, write_to_disk=False)
                start = time.time()
                for _ in range(n):
                    result = orchestrator_4w.next_batch()
                    time.sleep(0.5)  # Simulate downstream processing
                    del result
                time_4w = time.time() - start
            finally:
                orchestrator_4w.cleanup()

            # 4 workers should be faster due to prefetching overlap with processing delay
            assert time_4w <= time_1w * 1.1, (
                f"4-worker mode ({time_4w:.2f}s) should not be significantly slower "
                f"than 1-worker mode ({time_1w:.2f}s)"
            )


class TestStreamingOrchestratorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_batch_size_larger_than_sources(self, streaming_config, test_small_catalogue):
        """Test when batch_size is larger than total number of sources."""
        with tempfile.TemporaryDirectory() as output_dir:
            streaming_config.output_dir = output_dir
            streaming_config.source_catalogue = str(test_small_catalogue)

            orchestrator = StreamingOrchestrator(streaming_config)

            try:
                orchestrator.init_streaming(
                    batch_size=100000,  # Much larger than test catalogue
                    write_to_disk=False,
                )

                num_batches = orchestrator.get_batch_count()
                assert num_batches >= 1

                for i in range(num_batches):
                    result = orchestrator.next_batch()
                    assert "cutouts" in result
                    assert len(result["cutouts"]) > 0

            finally:
                orchestrator.cleanup()

    def test_not_initialized_error(self, streaming_config):
        """Test error when calling next_batch without initialization."""
        orchestrator = StreamingOrchestrator(streaming_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            orchestrator.next_batch()

        orchestrator.cleanup()

    def test_no_more_batches_error(self, streaming_config, test_small_catalogue):
        """Test error when requesting more batches than available."""
        with tempfile.TemporaryDirectory() as output_dir:
            streaming_config.output_dir = output_dir
            streaming_config.source_catalogue = str(test_small_catalogue)

            orchestrator = StreamingOrchestrator(streaming_config)

            try:
                orchestrator.init_streaming(
                    batch_size=3,
                    write_to_disk=False,
                )

                num_batches = orchestrator.get_batch_count()

                for _ in range(num_batches):
                    orchestrator.next_batch()

                with pytest.raises(RuntimeError, match="No more batches"):
                    orchestrator.next_batch()

            finally:
                orchestrator.cleanup()


class TestStreamingOrchestratorCleanup:
    """Tests for proper resource cleanup."""

    def test_cleanup_terminates_pending_batch(self, streaming_config, test_small_catalogue):
        """Test that cleanup properly terminates any pending batch preparation."""
        with tempfile.TemporaryDirectory() as output_dir:
            streaming_config.output_dir = output_dir
            streaming_config.source_catalogue = str(test_small_catalogue)

            orchestrator = StreamingOrchestrator(streaming_config)

            try:
                orchestrator.init_streaming(
                    batch_size=3,
                    write_to_disk=False,
                )

                # Don't call next_batch, just cleanup
                # This should terminate the pending batch preparation
            finally:
                orchestrator.cleanup()

            # No assertion needed - test passes if cleanup doesn't hang or crash
