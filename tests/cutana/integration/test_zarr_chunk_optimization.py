#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Integration tests for zarr chunk optimization.

Tests verify that zarr archives are created with proper chunk sizes
that don't unnecessarily split small datasets.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr
from dotmap import DotMap

from cutana.cutout_writer_zarr import (
    calculate_optimal_chunk_shape,
    create_zarr_from_memory,
)


class TestZarrChunkOptimization:
    """Integration tests for zarr chunk optimization."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "zarr_chunk_test"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = DotMap()
        config.target_resolution = 64
        config.data_type = "uint8"
        return config

    def test_small_uint8_dataset_single_chunk(self, temp_output_dir, mock_config):
        """Test that small uint8 dataset creates single chunk in zarr."""
        # Create small dataset: 500 images of 64x64x4 uint8 = 8.2 MB
        n_sources = 500
        height = 64
        width = 64
        n_channels = 4
        dtype = np.uint8

        # Create test data in NCHW format
        images = np.random.randint(0, 255, (n_sources, n_channels, height, width), dtype=dtype)

        # Create metadata
        metadata_list = [
            {
                "source_id": f"Source_{i:03d}",
                "ra": 150.0 + i * 0.01,
                "dec": 2.0 + i * 0.01,
            }
            for i in range(n_sources)
        ]

        # Create zarr archive
        zarr_path = temp_output_dir / "small_test.zarr"
        result = create_zarr_from_memory(
            images, metadata_list, str(zarr_path), mock_config, append=False
        )

        # Open the created zarr and check its chunking
        z = zarr.open(str(result), mode="r")

        # images_to_zarr uses 'images' as the key for the data array
        assert "images" in z, f"Available arrays: {list(z.keys())}"
        data_array = z["images"]

        # Check shape matches expected (NHWC format after conversion)
        assert data_array.shape == (
            n_sources,
            height,
            width,
            n_channels,
        ), f"Expected shape {(n_sources, height, width, n_channels)}, got {data_array.shape}"

        # Check chunk shape - should be single chunk for small data
        chunk_shape = data_array.chunks

        # Verify we got a single chunk (or very few chunks)
        # The chunk shape should match the data shape for such small data
        expected_chunk_shape = calculate_optimal_chunk_shape(
            n_sources, height, width, n_channels, np.dtype(dtype)
        )

        # Check that chunking is not overly aggressive
        # For 8.2MB of data, we should have at most a few chunks, not hundreds
        total_chunks = 1
        for data_dim, chunk_dim in zip(data_array.shape, chunk_shape):
            total_chunks *= (data_dim + chunk_dim - 1) // chunk_dim  # Ceiling division

        assert total_chunks <= 4, (
            f"Small dataset (8.2MB) created {total_chunks} chunks. "
            f"Data shape: {data_array.shape}, Chunk shape: {chunk_shape}"
        )

    def test_medium_float32_dataset_chunking(self, temp_output_dir, mock_config):
        """Test medium-sized float32 dataset creates appropriate chunks."""
        # Create medium dataset: 100 images of 256x256x3 float32 = 78.6 MB
        n_sources = 100
        height = 256
        width = 256
        n_channels = 3
        dtype = np.float32

        # Create test data in NCHW format
        images = np.random.random((n_sources, n_channels, height, width)).astype(dtype)

        # Create metadata
        metadata_list = [
            {
                "source_id": f"Source_{i:03d}",
                "ra": 150.0 + i * 0.01,
                "dec": 2.0 + i * 0.01,
            }
            for i in range(n_sources)
        ]

        mock_config.target_resolution = 256
        mock_config.data_type = "float32"

        # Create zarr archive
        zarr_path = temp_output_dir / "medium_test.zarr"
        result = create_zarr_from_memory(
            images, metadata_list, str(zarr_path), mock_config, append=False
        )

        # Open the created zarr and check its chunking
        z = zarr.open(str(result), mode="r")

        # images_to_zarr uses 'images' as the key for the data array
        assert "images" in z, f"Available arrays: {list(z.keys())}"
        data_array = z["images"]

        # For 78.6MB, should still be a single chunk or very few
        chunk_shape = data_array.chunks
        total_chunks = 1
        for data_dim, chunk_dim in zip(data_array.shape, chunk_shape):
            total_chunks *= (data_dim + chunk_dim - 1) // chunk_dim

        assert total_chunks <= 4, (
            f"Medium dataset (78.6MB) created {total_chunks} chunks. "
            f"Data shape: {data_array.shape}, Chunk shape: {chunk_shape}"
        )

    def test_large_dataset_appropriate_chunking(self, temp_output_dir, mock_config):
        """Test that large datasets get appropriately chunked."""
        # Create large dataset: 1000 images of 512x512x3 float32 = 3.14 GB
        n_sources = 1000
        height = 512
        width = 512
        n_channels = 3
        dtype = np.float32

        # For memory efficiency, we'll create a smaller test but calculate as if full size
        test_n_sources = 10  # Use fewer for actual test to avoid memory issues
        images = np.random.random((test_n_sources, n_channels, height, width)).astype(dtype)

        metadata_list = [
            {
                "source_id": f"Source_{i:03d}",
                "ra": 150.0 + i * 0.01,
                "dec": 2.0 + i * 0.01,
            }
            for i in range(test_n_sources)
        ]

        mock_config.target_resolution = 512
        mock_config.data_type = "float32"

        # Calculate what chunk shape should be for the full dataset
        expected_chunk_shape = calculate_optimal_chunk_shape(
            n_sources, height, width, n_channels, np.dtype(dtype)
        )

        # Verify the calculation says we need chunking
        bytes_per_image = height * width * n_channels * np.dtype(dtype).itemsize
        total_bytes = n_sources * bytes_per_image

        assert total_bytes > 2 * 1024**3, "Dataset should be > 2GB"
        assert expected_chunk_shape[0] < n_sources, (
            f"Large dataset should chunk along source dimension. "
            f"Got chunk shape {expected_chunk_shape} for {n_sources} sources"
        )

    def test_actual_zarr_structure_inspection(self, temp_output_dir, mock_config):
        """Inspect actual zarr file structure to understand chunking."""
        # Create a controlled small dataset
        n_sources = 50
        height = 32
        width = 32
        n_channels = 2
        dtype = np.uint8

        images = np.random.randint(0, 255, (n_sources, n_channels, height, width), dtype=dtype)

        metadata_list = [
            {"source_id": f"Source_{i:03d}", "ra": 150.0 + i * 0.01, "dec": 2.0 + i * 0.01}
            for i in range(n_sources)
        ]

        mock_config.target_resolution = 32
        mock_config.data_type = "uint8"

        zarr_path = temp_output_dir / "inspect_test.zarr"
        result = create_zarr_from_memory(
            images, metadata_list, str(zarr_path), mock_config, append=False
        )

        # Directly check the zarr directory structure
        zarr_dir = Path(result)
        assert zarr_dir.exists(), f"Zarr directory doesn't exist at {zarr_dir}"

        # List all subdirectories to see chunk structure
        subdirs = [d for d in zarr_dir.rglob("*") if d.is_dir()]

        # Print diagnostic info
        total_data_size = n_sources * height * width * n_channels * 1  # uint8 is 1 byte
        print(f"\nDataset size: {total_data_size / 1024:.1f} KB")
        print(f"Number of subdirectories in zarr: {len(subdirs)}")
        if subdirs:
            print(f"Subdirectories: {[str(d.relative_to(zarr_dir)) for d in subdirs[:10]]}")

        # For a 50KB dataset, we should have minimal subdirectories
        # images_to_zarr might create some structure, but it shouldn't be excessive
        # Let's just ensure we're not creating hundreds of chunks
        assert len(subdirs) < 20, (
            f"Small dataset (50KB) created {len(subdirs)} subdirectories/chunks. "
            f"This suggests over-chunking."
        )
