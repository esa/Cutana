#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the cutout_writer_zarr module using TDD approach.

Tests cover:
- Zarr archive creation and management
- Efficient batch writing of cutouts
- Metadata storage and indexing
- Compression and chunking strategies
- Error handling and recovery
- Direct memory-to-zarr conversion
"""

from unittest.mock import patch
import pytest
import numpy as np
from dotmap import DotMap
from cutana.cutout_writer_zarr import (
    generate_process_subfolder,
    create_zarr_from_memory,
    prepare_cutouts_for_zarr,
    create_process_zarr_archive_initial,
)


class TestCutoutWriterZarrFunctions:
    """Test suite for zarr writer functions."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = DotMap()
        config.target_resolution = 256
        config.fits_extensions = ["PRIMARY"]
        config.target_channels = None
        config.channel_weights = None
        config.data_type = "float32"
        return config

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "zarr_output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def mock_cutout_data(self):
        """Create mock processed cutout data in new batch format."""
        # Create 3-channel cutout data
        vis_cutout = np.random.random((256, 256)).astype(np.float32)
        nir_y_cutout = np.random.random((256, 256)).astype(np.float32)
        nir_h_cutout = np.random.random((256, 256)).astype(np.float32)

        # Stack into batch tensor: (N_sources, H, W, N_channels)
        cutouts_batch = np.stack([vis_cutout, nir_y_cutout, nir_h_cutout], axis=-1)[
            np.newaxis, ...
        ]  # Shape: (1, 256, 256, 3)

        return {
            "cutouts": cutouts_batch,
            "metadata": [
                {
                    "source_id": "MockSource_00001",
                    "ra": 150.0,
                    "dec": 2.0,
                    "diameter_pixel": 256,
                    "fits_file_paths": "['path1.fits', 'path2.fits', 'path3.fits']",
                    "custom_field": "test_value",
                    "processing_timestamp": 1642678800.0,
                    "wcs_info": {"dummy": "value"},
                }
            ],
        }

    def test_generate_process_subfolder(self):
        """Test subfolder generation for processes."""
        process_id = "cutout_process_001_unique_id"

        subfolder = generate_process_subfolder(process_id)

        assert subfolder == "batch_cutout_process_001_unique_id"

        # Test with different process ID formats
        assert (
            generate_process_subfolder("cutout_process_000_abc123")
            == "batch_cutout_process_000_abc123"
        )
        assert generate_process_subfolder("proc_123_xyz") == "batch_proc_123_xyz"
        assert generate_process_subfolder("simple_id") == "batch_simple_id"

    def test_prepare_cutouts_for_zarr(self, mock_cutout_data):
        """Test preparation of cutout data for zarr conversion."""
        # Now prepare_cutouts_for_zarr takes a single batch dict, not a list
        images_array, metadata_list = prepare_cutouts_for_zarr(mock_cutout_data)

        # Should return 4D array in NCHW format
        assert images_array.ndim == 4
        assert images_array.shape[0] == 1  # 1 source from the mock data
        assert images_array.shape[1] == 3  # 3 channels per source
        assert images_array.shape[2:] == (256, 256)

        # Should preserve original catalogue metadata exactly
        assert len(metadata_list) == 1
        metadata = metadata_list[0]

        # Should preserve all original catalogue fields exactly
        assert metadata["source_id"] == "MockSource_00001"
        assert metadata["ra"] == 150.0
        assert metadata["dec"] == 2.0
        assert metadata["diameter_pixel"] == 256
        assert metadata["fits_file_paths"] == "['path1.fits', 'path2.fits', 'path3.fits']"
        assert metadata["custom_field"] == "test_value"

        # Should not add any inferred fields
        assert "channels" not in metadata  # No longer added by zarr writer
        assert "num_channels" not in metadata

    @patch("cutana.cutout_writer_zarr.convert")
    def test_create_zarr_from_memory(self, mock_convert, temp_output_dir, mock_config):
        """Test direct memory-to-zarr conversion."""
        mock_convert.return_value = str(temp_output_dir / "test.zarr")

        # Create test data in NCHW format
        num_images = 5
        images = np.random.random((num_images, 1, 128, 128)).astype(np.float32)

        # Original catalogue metadata (should be preserved exactly)
        metadata_list = [
            {
                "SourceID": f"Source_{i:03d}",
                "RA": 150.0 + i * 0.01,
                "Dec": 2.0 + i * 0.01,
                "diameter_pixel": 128,
                "fits_file_paths": f"['file_{i}.fits']",
            }
            for i in range(num_images)
        ]

        # Update config for the test
        mock_config.target_resolution = 128

        zarr_path = create_zarr_from_memory(
            images, metadata_list, str(temp_output_dir), mock_config
        )

        assert zarr_path.endswith(".zarr")
        mock_convert.assert_called_once()

        # Verify convert was called with correct parameters
        call_args = mock_convert.call_args
        # create_zarr_from_memory passes parent of full path, so we need to check parent
        assert call_args.kwargs["output_dir"] == str(temp_output_dir.parent)
        assert call_args.kwargs["chunk_shape"] == (
            5,
            128,
            128,
            1,
        )  # Should match num_images and channels
        assert call_args.kwargs["compressor"] == "lz4"
        assert call_args.kwargs["overwrite"] is True

        # Verify original metadata passed through unchanged
        passed_metadata = call_args.kwargs["image_metadata"]
        assert len(passed_metadata) == 5
        assert passed_metadata[0]["SourceID"] == "Source_000"
        assert passed_metadata[0]["diameter_pixel"] == 128

    def test_create_process_zarr_archive_initial(self, temp_output_dir, mock_config):
        """Test creating initial zarr archive for a process batch."""
        # Create batch data in new format with 10 sources
        cutouts_list = []
        metadata_list = []

        for i in range(10):
            # Create 2-channel cutout for each source
            vis_cutout = np.random.random((64, 64)).astype(np.float32)
            nir_cutout = np.random.random((64, 64)).astype(np.float32)
            # Stack channels: (H, W, C)
            source_cutout = np.stack([vis_cutout, nir_cutout], axis=-1)
            cutouts_list.append(source_cutout)

            metadata_list.append(
                {
                    "source_id": f"BatchSource_{i:03d}",
                    "ra": 150.0 + i * 0.01,
                    "dec": 2.0 + i * 0.01,
                    "diameter_pixel": 128,
                    "fits_file_paths": f"['vis_{i}.fits', 'nir_{i}.fits']",
                    "processing_timestamp": 1642678800.0,
                    "wcs_info": {"dummy": "value"},
                }
            )

        # Stack into batch tensor: (N_sources, H, W, N_channels)
        cutouts_batch = np.stack(cutouts_list, axis=0)  # Shape: (10, 64, 64, 2)

        batch_data = {
            "cutouts": cutouts_batch,
            "metadata": metadata_list,
        }

        mock_config.target_resolution = 64
        zarr_path = temp_output_dir / "test_process" / "images.zarr"

        with patch("cutana.cutout_writer_zarr.create_zarr_from_memory") as mock_create:
            mock_create.return_value = str(zarr_path)

            result = create_process_zarr_archive_initial(batch_data, str(zarr_path), mock_config)

            assert result is not None
            mock_create.assert_called_once()

            # Check that images were prepared correctly
            call_args = mock_create.call_args
            # Arguments are passed as keyword arguments
            images_array = call_args.kwargs["images"]
            metadata_list = call_args.kwargs["metadata"]

            # Should have 10 sources with 2 channels each
            assert images_array.shape[0] == 10  # 10 sources
            assert images_array.shape[1] == 2  # 2 channels per source
            assert len(metadata_list) == 10  # Original catalogue metadata for each source

    def test_error_handling_empty_batch(self, temp_output_dir, mock_config):
        """Test handling of empty batch data."""
        empty_batch = {"cutouts": None, "metadata": []}
        zarr_path = temp_output_dir / "empty_process" / "images.zarr"

        result = create_process_zarr_archive_initial(empty_batch, str(zarr_path), mock_config)

        assert result is None

    def test_error_handling_no_valid_cutouts(self, temp_output_dir, mock_config):
        """Test handling when no valid cutouts are found."""
        # Batch with empty cutouts tensor
        batch_data = {
            "cutouts": np.array([]),  # Empty tensor
            "metadata": [
                {
                    "source_id": "NoData",
                    "ra": 150.0,
                    "dec": 2.0,
                    "processing_timestamp": 1642678800.0,
                }
            ],
        }

        zarr_path = temp_output_dir / "no_cutouts_process" / "images.zarr"

        result = create_process_zarr_archive_initial(batch_data, str(zarr_path), mock_config)

        assert result is None

    def test_multi_channel_cutout_handling(self, mock_config):
        """Test handling of multi-channel cutouts."""
        # Create 3-channel cutout data in new batch format
        rgb_cutout = np.random.random((32, 32, 3)).astype(np.float32)  # 3 channels
        cutouts_batch = rgb_cutout[np.newaxis, ...]  # Shape: (1, 32, 32, 3)

        batch_data = {
            "cutouts": cutouts_batch,
            "metadata": [
                {
                    "source_id": "MultiChannel",
                    "ra": 150.0,
                    "dec": 2.0,
                    "diameter_pixel": 64,
                    "processing_timestamp": 1642678800.0,
                }
            ],
        }

        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        # Should handle 3-channel input correctly
        assert images_array.shape == (1, 3, 32, 32)  # 1 source, 3 channels
        assert len(metadata_list) == 1
        # Should preserve original metadata exactly
        assert metadata_list[0]["source_id"] == "MultiChannel"
        assert metadata_list[0]["diameter_pixel"] == 64

    def test_metadata_preservation_diameter_pixel(self, mock_config):
        """Test that original catalogue metadata is preserved exactly."""
        # Create single-channel cutout in new batch format
        vis_cutout = np.random.random((64, 64)).astype(np.float32)
        cutouts_batch = vis_cutout[np.newaxis, :, :, np.newaxis]  # Shape: (1, 64, 64, 1)

        batch_data = {
            "cutouts": cutouts_batch,
            "metadata": [
                {
                    "source_id": "TestSource_001",
                    "ra": 150.0,
                    "dec": 2.0,
                    "diameter_pixel": 128,  # Should be preserved exactly
                    "custom_metadata": "preserved_value",
                    "fits_file_paths": "['vis.fits']",
                    "processing_timestamp": 1642678800.0,
                }
            ],
        }

        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        # Verify metadata preservation
        assert len(metadata_list) == 1
        metadata = metadata_list[0]

        # Should preserve all original catalogue fields exactly
        assert metadata["source_id"] == "TestSource_001"
        assert metadata["diameter_pixel"] == 128
        assert "diameter_arcsec" not in metadata  # Should not add if not in original
        assert metadata["custom_metadata"] == "preserved_value"
        assert metadata["fits_file_paths"] == "['vis.fits']"
        assert metadata["ra"] == 150.0
        assert metadata["dec"] == 2.0

    def test_metadata_preservation_mixed_fields(self, mock_config):
        """Test metadata preservation with mixed size fields."""
        # Create 2 sources with different cutouts in new batch format
        vis_cutout = np.random.random((32, 32)).astype(np.float32)
        nir_cutout = np.random.random((32, 32)).astype(np.float32)

        # Stack into batch tensor: (N_sources, H, W, N_channels)
        cutouts_batch = np.stack(
            [
                vis_cutout[:, :, np.newaxis],  # Add channel dimension
                nir_cutout[:, :, np.newaxis],  # Add channel dimension
            ],
            axis=0,
        )  # Shape: (2, 32, 32, 1)

        batch_data = {
            "cutouts": cutouts_batch,
            "metadata": [
                {
                    "source_id": "TestSource_pixel",
                    "ra": 150.0,
                    "dec": 2.0,
                    "diameter_pixel": 64,
                    "telescope": "Euclid",
                    "observation_date": "2024-01-01",
                    "processing_timestamp": 1642678800.0,
                },
                {
                    "source_id": "TestSource_arcsec",
                    "ra": 151.0,
                    "dec": 3.0,
                    "diameter_arcsec": 5.2,
                    "filter_name": "H_band",
                    "processing_timestamp": 1642678800.0,
                },
            ],
        }

        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        # Verify both sources preserved their original fields exactly
        assert len(metadata_list) == 2

        # First source should have diameter_pixel
        meta1 = metadata_list[0]
        assert meta1["source_id"] == "TestSource_pixel"
        assert "diameter_pixel" in meta1
        assert meta1["diameter_pixel"] == 64
        assert "diameter_arcsec" not in meta1
        assert meta1["telescope"] == "Euclid"
        assert meta1["observation_date"] == "2024-01-01"

        # Second source should have diameter_arcsec
        meta2 = metadata_list[1]
        assert meta2["source_id"] == "TestSource_arcsec"
        assert "diameter_arcsec" in meta2
        assert meta2["diameter_arcsec"] == 5.2
        assert "diameter_pixel" not in meta2
        assert meta2["filter_name"] == "H_band"


class TestCalculateOptimalChunkShape:
    """Test suite for chunk shape calculation."""

    def test_small_uint8_images_should_not_chunk(self):
        """Test that small uint8 images don't get unnecessarily chunked."""
        # 500 images of 64x64x4 uint8 = 8.2 MB total (well below 2GB)
        n_sources = 500
        height = 64
        width = 64
        n_channels = 4
        dtype = np.dtype("uint8")

        from cutana.cutout_writer_zarr import calculate_optimal_chunk_shape

        chunk_shape = calculate_optimal_chunk_shape(n_sources, height, width, n_channels, dtype)

        # Should return full dimensions since it's way below 2GB
        assert chunk_shape == (
            500,
            64,
            64,
            4,
        ), f"Small dataset (8.2MB) should not be chunked, got {chunk_shape}"

    def test_large_float32_images_should_chunk_sources(self):
        """Test that large float32 images chunk along source dimension."""
        # 1000 images of 512x512x3 float32 = 3.14 GB total
        n_sources = 1000
        height = 512
        width = 512
        n_channels = 3
        dtype = np.dtype("float32")

        from cutana.cutout_writer_zarr import calculate_optimal_chunk_shape

        chunk_shape = calculate_optimal_chunk_shape(n_sources, height, width, n_channels, dtype)

        # Calculate expected chunk size
        bytes_per_image = height * width * n_channels * dtype.itemsize  # 3.145 MB
        max_chunk_bytes = 1.8 * 1024**3  # 1.8 GB
        expected_max_images = int(max_chunk_bytes / bytes_per_image)  # ~573

        assert chunk_shape[0] <= expected_max_images
        assert chunk_shape[1:] == (height, width, n_channels)

        # Verify chunk size is below 2GB
        chunk_bytes = (
            chunk_shape[0] * chunk_shape[1] * chunk_shape[2] * chunk_shape[3] * dtype.itemsize
        )
        assert chunk_bytes < 2 * 1024**3, f"Chunk size {chunk_bytes / 1024**3:.2f}GB exceeds 2GB"

    def test_single_huge_image_should_chunk_spatially(self):
        """Test that a single image larger than 2GB chunks spatially."""
        # 1 image of 32768x32768x4 float32 = 17.2 GB (way over 2GB)
        n_sources = 1
        height = 32768
        width = 32768
        n_channels = 4
        dtype = np.dtype("float32")

        from cutana.cutout_writer_zarr import calculate_optimal_chunk_shape

        chunk_shape = calculate_optimal_chunk_shape(
            n_sources, height, width, n_channels, dtype, max_chunk_size_gb=1.8
        )

        assert chunk_shape[0] == 1  # Should still be 1 source
        assert chunk_shape[1] < height  # Should reduce height
        assert chunk_shape[2] < width  # Should reduce width
        assert chunk_shape[3] == n_channels  # Channels unchanged

        # Verify chunk size is below 2GB
        chunk_bytes = (
            chunk_shape[0] * chunk_shape[1] * chunk_shape[2] * chunk_shape[3] * dtype.itemsize
        )
        assert chunk_bytes < 2 * 1024**3, f"Chunk size {chunk_bytes / 1024**3:.2f}GB exceeds 2GB"

    def test_medium_uint16_images(self):
        """Test medium-sized uint16 images."""
        # 200 images of 256x256x8 uint16 = 209.7 MB (below 2GB)
        n_sources = 200
        height = 256
        width = 256
        n_channels = 8
        dtype = np.dtype("uint16")

        from cutana.cutout_writer_zarr import calculate_optimal_chunk_shape

        chunk_shape = calculate_optimal_chunk_shape(n_sources, height, width, n_channels, dtype)

        # Should not chunk this small dataset
        assert chunk_shape == (
            200,
            256,
            256,
            8,
        ), f"Medium dataset (209.7MB) should not be chunked, got {chunk_shape}"

    def test_edge_case_exactly_2gb(self):
        """Test edge case where data is exactly at the limit."""
        # Calculate dimensions for exactly 1.8GB with float32
        n_channels = 3
        height = 512
        width = 512
        dtype = np.dtype("float32")
        bytes_per_image = height * width * n_channels * dtype.itemsize
        n_sources = int((1.8 * 1024**3) / bytes_per_image)  # Exactly at limit

        from cutana.cutout_writer_zarr import calculate_optimal_chunk_shape

        chunk_shape = calculate_optimal_chunk_shape(
            n_sources, height, width, n_channels, dtype, max_chunk_size_gb=1.8
        )

        # Should fit exactly or chunk minimally
        assert chunk_shape[0] <= n_sources
        chunk_bytes = (
            chunk_shape[0] * chunk_shape[1] * chunk_shape[2] * chunk_shape[3] * dtype.itemsize
        )
        assert chunk_bytes <= 1.8 * 1024**3

    def test_single_channel_images(self):
        """Test single channel images."""
        # 1000 images of 128x128x1 float64 = 131 MB
        n_sources = 1000
        height = 128
        width = 128
        n_channels = 1
        dtype = np.dtype("float64")

        from cutana.cutout_writer_zarr import calculate_optimal_chunk_shape

        chunk_shape = calculate_optimal_chunk_shape(n_sources, height, width, n_channels, dtype)

        # Should not chunk this small dataset
        expected_bytes = n_sources * height * width * n_channels * dtype.itemsize
        if expected_bytes < 1.8 * 1024**3:
            assert chunk_shape == (n_sources, height, width, n_channels)
