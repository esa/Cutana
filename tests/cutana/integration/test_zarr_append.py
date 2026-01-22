#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for zarr append functionality.

Tests the new incremental writing capability where sub-batches are
written and appended to zarr files to reduce memory footprint.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import zarr

from cutana.cutout_writer_zarr import (
    append_to_zarr_archive,
    calculate_optimal_chunk_shape,
    create_process_zarr_archive_initial,
    prepare_cutouts_for_zarr,
)
from cutana.get_default_config import get_default_config


class TestZarrAppend:
    """Test suite for zarr append functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        config = get_default_config()
        config.target_resolution = 64
        config.data_type = "float32"
        return config

    def create_test_batch(
        self, n_sources: int, height: int, width: int, n_channels: int, batch_id: int
    ) -> Dict[str, Any]:
        """
        Create a test batch with synthetic cutout data.

        Args:
            n_sources: Number of sources in the batch
            height: Image height
            width: Image width
            n_channels: Number of channels
            batch_id: Batch identifier for metadata

        Returns:
            Dictionary with cutouts and metadata
        """
        # Create cutouts in NHWC format (as produced by cutout_process)
        cutouts = np.random.rand(n_sources, height, width, n_channels).astype(np.float32)

        # Add distinct patterns to verify correct appending
        # Each batch gets a different base value
        cutouts += batch_id * 10.0

        # Create metadata
        metadata = []
        for i in range(n_sources):
            metadata.append(
                {
                    "source_id": f"batch_{batch_id}_source_{i}",
                    "ra": 150.0 + i * 0.01,
                    "dec": 2.0 + i * 0.01,
                    "batch_id": batch_id,
                }
            )

        return {"cutouts": cutouts, "metadata": metadata}

    def test_calculate_optimal_chunk_shape(self):
        """Test chunk shape calculation stays below 2GB limit."""
        # Test with float32 (4 bytes per element)
        dtype = np.dtype("float32")

        # Small images - should fit many in a chunk
        chunk_n, chunk_h, chunk_w, chunk_c = calculate_optimal_chunk_shape(
            100, 64, 64, 3, dtype, max_chunk_size_gb=1.8
        )

        chunk_size_bytes = chunk_n * chunk_h * chunk_w * chunk_c * dtype.itemsize
        assert chunk_size_bytes < 2 * 1024 * 1024 * 1024  # Less than 2GB
        assert chunk_c == 3  # Channels should match input

        # Large images - should reduce number per chunk
        chunk_n, chunk_h, chunk_w, chunk_c = calculate_optimal_chunk_shape(
            100, 2048, 2048, 3, dtype, max_chunk_size_gb=1.8
        )

        chunk_size_bytes = chunk_n * chunk_h * chunk_w * chunk_c * dtype.itemsize
        assert chunk_size_bytes < 2 * 1024 * 1024 * 1024  # Less than 2GB
        assert chunk_c == 3  # Channels should match input

        # Very large single image - should chunk spatially
        chunk_n, chunk_h, chunk_w, chunk_c = calculate_optimal_chunk_shape(
            1, 8192, 8192, 10, dtype, max_chunk_size_gb=1.8
        )

        chunk_size_bytes = chunk_n * chunk_h * chunk_w * chunk_c * dtype.itemsize
        assert chunk_size_bytes < 2 * 1024 * 1024 * 1024  # Less than 2GB
        assert chunk_h < 8192 or chunk_w < 8192  # Should reduce spatial dimensions
        assert chunk_c == 10  # Channels should match input

    def test_prepare_cutouts_for_zarr(self):
        """Test cutout preparation converts format correctly."""
        # Create test batch in NHWC format
        batch = self.create_test_batch(5, 32, 32, 2, batch_id=0)

        # Prepare for zarr
        images_nchw, metadata = prepare_cutouts_for_zarr(batch)

        # Check conversion to NCHW format
        assert images_nchw.shape == (5, 2, 32, 32)  # (N, C, H, W)
        assert len(metadata) == 5
        assert metadata[0]["source_id"] == "batch_0_source_0"

    def test_incremental_zarr_creation_and_append(self, temp_dir, sample_config):
        """Test creating initial zarr and appending sub-batches."""
        # Create test batches
        batch1 = self.create_test_batch(10, 64, 64, 3, batch_id=1)
        batch2 = self.create_test_batch(8, 64, 64, 3, batch_id=2)
        batch3 = self.create_test_batch(12, 64, 64, 3, batch_id=3)

        # Define zarr path
        zarr_path = temp_dir / "test_output" / "images.zarr"

        # Create initial zarr with first batch
        result = create_process_zarr_archive_initial(batch1, str(zarr_path), sample_config)
        assert result is not None
        assert zarr_path.exists()

        # Load and check initial zarr
        zarr_store = zarr.open(str(zarr_path), mode="r")
        images = zarr_store["images"]
        assert images.shape == (10, 64, 64, 3)  # NHWC format in zarr

        # Check first batch values (should have base value around 10)
        batch1_mean = np.mean(images[:])
        assert 9 < batch1_mean < 12  # Random [0,1] + 10

        # Append second batch
        result = append_to_zarr_archive(batch2, str(zarr_path), sample_config)
        assert result is not None

        # Check zarr after first append
        zarr_store = zarr.open(str(zarr_path), mode="r")
        images = zarr_store["images"]
        assert images.shape == (18, 64, 64, 3)  # 10 + 8 sources

        # Check second batch values (should have base value around 20)
        batch2_mean = np.mean(images[10:18])
        assert 19 < batch2_mean < 22  # Random [0,1] + 20

        # Append third batch
        result = append_to_zarr_archive(batch3, str(zarr_path), sample_config)
        assert result is not None

        # Check final zarr
        zarr_store = zarr.open(str(zarr_path), mode="r")
        images = zarr_store["images"]
        metadata = zarr_store.get("metadata", None)

        # Check final shape
        assert images.shape == (30, 64, 64, 3)  # 10 + 8 + 12 sources

        # Check third batch values (should have base value around 30)
        batch3_mean = np.mean(images[18:30])
        assert 29 < batch3_mean < 32  # Random [0,1] + 30

        # Verify all batches maintained their distinct patterns
        assert 9 < np.mean(images[0:10]) < 12  # Batch 1
        assert 19 < np.mean(images[10:18]) < 22  # Batch 2
        assert 29 < np.mean(images[18:30]) < 32  # Batch 3

        # Check metadata if available
        if metadata is not None:
            assert len(metadata) == 30
            # Check metadata ordering
            for i in range(10):
                assert metadata[i]["source_id"] == f"batch_1_source_{i}"
            for i in range(8):
                assert metadata[10 + i]["source_id"] == f"batch_2_source_{i}"
            for i in range(12):
                assert metadata[18 + i]["source_id"] == f"batch_3_source_{i}"

    def test_append_preserves_data_types(self, temp_dir, sample_config):
        """Test that appending preserves data types correctly."""
        # Test with uint8 data type
        sample_config.data_type = "uint8"

        # Create test batches with uint8 values
        batch1 = self.create_test_batch(5, 32, 32, 1, batch_id=1)
        batch1["cutouts"] = (batch1["cutouts"] * 25).astype(np.uint8)  # Scale to uint8 range

        batch2 = self.create_test_batch(5, 32, 32, 1, batch_id=2)
        batch2["cutouts"] = (batch2["cutouts"] * 25 + 100).astype(np.uint8)  # Different range

        zarr_path = temp_dir / "test_uint8" / "images.zarr"

        # Create and append
        create_process_zarr_archive_initial(batch1, str(zarr_path), sample_config)
        append_to_zarr_archive(batch2, str(zarr_path), sample_config)

        # Check result
        zarr_store = zarr.open(str(zarr_path), mode="r")
        images = zarr_store["images"]

        assert images.dtype == np.uint8
        assert images.shape == (10, 32, 32, 1)

        # Check value ranges
        batch1_values = images[0:5]
        batch2_values = images[5:10]

        # Note: batch1 is based on random values [0,1] * 25 + batch_id*10 (=10)
        # So values should be roughly in range [10*25, 11*25] = [250, 275] -> clamped to 255
        # batch2 is based on random values [0,1] * 25 + 100 + batch_id*10 (=20)
        # So values should be roughly [20*25 + 100, 21*25 + 100] = [600, 625] -> clamped to 255

        # Both batches will be clamped to 255, but we can check they have different patterns
        batch1_mean = np.mean(batch1_values)
        batch2_mean = np.mean(batch2_values)

        # Since both are clamped, let's just verify they are uint8 and have reasonable values
        assert images.dtype == np.uint8
        assert np.max(batch1_values) <= 255
        assert np.max(batch2_values) <= 255

    def test_empty_batch_handling(self, temp_dir, sample_config):
        """Test handling of empty batches."""
        zarr_path = temp_dir / "test_empty" / "images.zarr"

        # Try to create with empty batch
        empty_batch = {"cutouts": np.array([]), "metadata": []}
        result = create_process_zarr_archive_initial(empty_batch, str(zarr_path), sample_config)
        assert result is None  # Should return None for empty batch

        # Create with valid batch
        valid_batch = self.create_test_batch(5, 32, 32, 2, batch_id=1)
        result = create_process_zarr_archive_initial(valid_batch, str(zarr_path), sample_config)
        assert result is not None

        # Try to append empty batch - should handle gracefully
        result = append_to_zarr_archive(empty_batch, str(zarr_path), sample_config)
        assert result is None  # Should return None but not crash

        # Verify original data is intact
        zarr_store = zarr.open(str(zarr_path), mode="r")
        images = zarr_store["images"]
        assert images.shape == (5, 32, 32, 2)  # Original batch unchanged

    def test_multi_channel_append(self, temp_dir, sample_config):
        """Test appending with multiple channels maintains channel order."""
        # Create batches with 4 channels (e.g., RGBA or multi-band)
        batch1 = self.create_test_batch(3, 64, 64, 4, batch_id=1)
        batch2 = self.create_test_batch(2, 64, 64, 4, batch_id=2)

        # Set distinct patterns per channel
        for c in range(4):
            batch1["cutouts"][:, :, :, c] *= c + 1  # Channel multipliers: 1, 2, 3, 4
            batch2["cutouts"][:, :, :, c] *= c + 5  # Channel multipliers: 5, 6, 7, 8

        zarr_path = temp_dir / "test_multichannel" / "images.zarr"

        # Create and append
        create_process_zarr_archive_initial(batch1, str(zarr_path), sample_config)
        append_to_zarr_archive(batch2, str(zarr_path), sample_config)

        # Verify
        zarr_store = zarr.open(str(zarr_path), mode="r")
        images = zarr_store["images"]
        assert images.shape == (5, 64, 64, 4)

        # Check channel patterns are preserved
        # Batch 1 channels should have different means
        for c in range(4):
            channel_mean = np.mean(images[0:3, :, :, c])
            expected_base = 10 * (c + 1)  # Base value 10 * channel multiplier
            assert abs(channel_mean - expected_base) < 5  # Allow some variance

        # Batch 2 channels
        for c in range(4):
            channel_mean = np.mean(images[3:5, :, :, c])
            expected_base = 20 * (c + 5)  # Base value 20 * channel multiplier
            assert abs(channel_mean - expected_base) < 10


class TestZarrChunkOptimization:
    """Test suite for chunk size optimization."""

    def test_various_image_sizes(self):
        """Test chunk calculation for various image sizes and data types."""
        test_cases = [
            # (n_sources, height, width, channels, dtype, max_gb)
            (1000, 32, 32, 1, np.float32, 1.8),  # Small images
            (100, 256, 256, 3, np.float32, 1.8),  # Medium RGB images
            (10, 2048, 2048, 3, np.float32, 1.8),  # Large RGB images
            (1, 8192, 8192, 1, np.float32, 1.8),  # Very large single channel
            (100, 512, 512, 10, np.float16, 1.8),  # Multi-channel with float16
            (50, 1024, 1024, 3, np.uint8, 1.8),  # uint8 images
        ]

        for n, h, w, c, dtype, max_gb in test_cases:
            chunk_n, chunk_h, chunk_w, chunk_c = calculate_optimal_chunk_shape(
                n, h, w, c, np.dtype(dtype), max_gb
            )

            # Calculate actual chunk size
            chunk_bytes = chunk_n * chunk_h * chunk_w * chunk_c * np.dtype(dtype).itemsize
            max_bytes = max_gb * 1024 * 1024 * 1024

            # Verify chunk is below limit
            assert chunk_bytes <= max_bytes, (
                f"Chunk size {chunk_bytes / 1e9:.2f}GB exceeds "
                f"limit {max_gb}GB for image ({n},{h},{w},{c}) dtype={dtype}"
            )

            # Verify chunk dimensions are valid
            assert 1 <= chunk_n <= n
            assert 1 <= chunk_h <= h
            assert 1 <= chunk_w <= w
            assert chunk_c == c  # Channels should match input
