#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Integration test for multi-channel Zarr output.

Tests that multi-channel sources are properly grouped in Zarr archives
instead of being flattened to individual single-channel images.
"""

import numpy as np
from dotmap import DotMap

from cutana.cutout_writer_zarr import prepare_cutouts_for_zarr


class TestMultiChannelZarrIntegration:
    """Test multi-channel Zarr output integration."""

    def mock_config(self):
        """Create mock configuration."""
        config = DotMap()
        config.target_resolution = 150
        config.fits_extensions = ["PRIMARY"]
        config.target_channels = None
        config.channel_weights = None
        config.data_type = "float32"
        return config

    def test_multi_channel_cutout_preparation(self):
        """Test that multi-channel cutouts are prepared correctly for Zarr."""
        # Configuration available via self.mock_config() if needed

        # Create mock multi-channel cutout data in new batch format
        h_cutout = np.random.random((150, 150)).astype(np.float32)
        j_cutout = np.random.random((150, 150)).astype(np.float32)
        y_cutout = np.random.random((150, 150)).astype(np.float32)

        # Create batch tensors: (N_sources, H, W, N_channels)
        cutouts_batch = np.stack(
            [
                np.stack([h_cutout, j_cutout, y_cutout], axis=-1),  # source_001: (150, 150, 3)
                np.stack(
                    [h_cutout * 0.8, j_cutout * 0.8, y_cutout * 0.8], axis=-1
                ),  # source_002: (150, 150, 3)
            ],
            axis=0,
        )  # Shape: (2, 150, 150, 3)

        batch_data = {
            "cutouts": cutouts_batch,
            "metadata": [
                {
                    # Original catalogue metadata - should be preserved exactly
                    "source_id": "source_001",
                    "ra": 10.0,
                    "dec": 20.0,
                    "diameter_pixel": 150,
                    "fits_file_paths": "['h.fits', 'j.fits', 'y.fits']",
                    "telescope": "Euclid",
                    "processing_timestamp": 1642678800.0,
                },
                {
                    # Original catalogue metadata - should be preserved exactly
                    "source_id": "source_002",
                    "ra": 11.0,
                    "dec": 21.0,
                    "diameter_pixel": 150,
                    "fits_file_paths": "['h.fits', 'j.fits', 'y.fits']",
                    "observation_id": "obs_002",
                    "processing_timestamp": 1642678800.0,
                },
            ],
        }

        # Prepare cutouts for Zarr
        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        # FIXED behavior: 2 sources with 3 channels each = (2, 3, 150, 150)
        assert images_array.shape == (
            2,
            3,
            150,
            150,
        ), f"Got shape {images_array.shape}, expected (2, 3, 150, 150)"
        assert len(metadata_list) == 2  # One metadata entry per source

        # Verify metadata preserves original catalogue data exactly
        assert metadata_list[0]["source_id"] == "source_001"
        assert metadata_list[1]["source_id"] == "source_002"
        assert metadata_list[0]["diameter_pixel"] == 150
        assert metadata_list[1]["diameter_pixel"] == 150
        assert metadata_list[0]["telescope"] == "Euclid"
        assert metadata_list[1]["observation_id"] == "obs_002"

        # Should not add inferred fields
        assert "channels" not in metadata_list[0]
        assert "num_channels" not in metadata_list[0]

    def test_single_channel_cutout_preparation(self):
        """Test that single-channel cutouts work correctly."""
        # Configuration available via self.mock_config() if needed

        # Create single-channel cutout in new batch format
        primary_cutout = np.random.random((150, 150)).astype(np.float32)
        cutouts_batch = primary_cutout[np.newaxis, :, :, np.newaxis]  # Shape: (1, 150, 150, 1)

        batch_data = {
            "cutouts": cutouts_batch,
            "metadata": [
                {
                    # Original catalogue metadata
                    "source_id": "source_001",
                    "ra": 10.0,
                    "dec": 20.0,
                    "diameter_pixel": 150,
                    "fits_file_paths": "['single.fits']",
                    "processing_timestamp": 1642678800.0,
                }
            ],
        }

        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        assert images_array.shape == (1, 1, 150, 150)
        assert len(metadata_list) == 1
        assert metadata_list[0]["source_id"] == "source_001"
        assert metadata_list[0]["diameter_pixel"] == 150
        assert metadata_list[0]["fits_file_paths"] == "['single.fits']"

    def test_multi_channel_grouping_integration(self):
        """Test complete multi-channel grouping behavior - now implemented!"""
        # Configuration available via self.mock_config() if needed

        # Create multi-channel cutouts in new batch format
        h_cutout = np.random.random((150, 150)).astype(np.float32)
        j_cutout = np.random.random((150, 150)).astype(np.float32)
        y_cutout = np.random.random((150, 150)).astype(np.float32)

        cutouts_batch = np.stack(
            [
                np.stack([h_cutout, j_cutout, y_cutout], axis=-1),  # source_001: (150, 150, 3)
                np.stack(
                    [h_cutout * 0.9, j_cutout * 0.9, y_cutout * 0.9], axis=-1
                ),  # source_002: (150, 150, 3)
            ],
            axis=0,
        )  # Shape: (2, 150, 150, 3)

        batch_data = {
            "cutouts": cutouts_batch,
            "metadata": [
                {
                    "source_id": "source_001",
                    "ra": 10.0,
                    "dec": 20.0,
                    "diameter_pixel": 150,
                    "fits_file_paths": "['h.fits', 'j.fits', 'y.fits']",
                    "processing_timestamp": 1642678800.0,
                },
                {
                    "source_id": "source_002",
                    "ra": 11.0,
                    "dec": 21.0,
                    "diameter_pixel": 150,
                    "fits_file_paths": "['h.fits', 'j.fits', 'y.fits']",
                    "processing_timestamp": 1642678800.0,
                },
            ],
        }

        # Test the IMPLEMENTED behavior - channels are properly grouped by source
        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        # Now we get the correct shape: 2 sources, 3 channels each
        assert images_array.shape == (2, 3, 150, 150)  # 2 sources, 3 channels each
        assert len(metadata_list) == 2  # One metadata entry per source
        assert metadata_list[0]["source_id"] == "source_001"
        assert metadata_list[1]["source_id"] == "source_002"

    def test_empty_batch_handling(self):
        """Test handling of empty batch data."""
        # Configuration available via self.mock_config() if needed
        batch_data = {"cutouts": np.array([]), "metadata": []}

        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        assert images_array.size == 0
        assert len(metadata_list) == 0

    def test_invalid_cutout_data_handling(self):
        """Test handling of invalid cutout data."""
        # Configuration available via self.mock_config() if needed

        # Create invalid batch data in new format
        batch_data = {
            "cutouts": np.array([]),  # Empty tensor
            "metadata": [
                {
                    "source_id": "source_001",
                    "ra": 10.0,
                    "dec": 20.0,
                    "processing_timestamp": 1642678800.0,
                }
            ],
        }

        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        assert images_array.size == 0
        assert len(metadata_list) == 0

    def test_metadata_preservation_diameter_pixel(self):
        """Test that diameter_pixel metadata is preserved in multi-channel zarr processing."""
        # Configuration available via self.mock_config() if needed

        # Create 2-channel cutout in new batch format
        h_cutout = np.random.random((100, 100)).astype(np.float32)
        j_cutout = np.random.random((100, 100)).astype(np.float32)
        cutouts_batch = np.stack([h_cutout, j_cutout], axis=-1)[
            np.newaxis, ...
        ]  # Shape: (1, 100, 100, 2)

        batch_data = {
            "cutouts": cutouts_batch,
            "metadata": [
                {
                    # Original catalogue metadata - should be preserved exactly
                    "source_id": "source_001",
                    "ra": 10.0,
                    "dec": 20.0,
                    "diameter_pixel": 200,  # Should be preserved
                    "custom_field": "preserved_value",
                    "instrument": "NISP",
                    "fits_file_paths": "['h.fits', 'j.fits']",
                    "processing_timestamp": 1642678800.0,
                }
            ],
        }

        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        # Verify metadata preservation
        assert len(metadata_list) == 1
        metadata = metadata_list[0]

        # Should preserve diameter_pixel and custom fields exactly
        assert "diameter_pixel" in metadata
        assert metadata["diameter_pixel"] == 200
        assert "diameter_arcsec" not in metadata  # Should not be added

        # Should preserve all custom metadata exactly
        assert metadata["custom_field"] == "preserved_value"
        assert metadata["instrument"] == "NISP"
        assert metadata["fits_file_paths"] == "['h.fits', 'j.fits']"

        # Should preserve coordinates
        assert metadata["ra"] == 10.0
        assert metadata["dec"] == 20.0

        # Should not have inferred channel info (that's handled elsewhere)
        assert "channels" not in metadata
        assert "num_channels" not in metadata

    def test_mixed_metadata_fields(self):
        """Test sources with different metadata fields are all preserved."""
        # Configuration available via self.mock_config() if needed

        # Create cutouts for 2 sources in new batch format
        vis_cutout = np.random.random((80, 80)).astype(np.float32)
        nir_cutout = np.random.random((80, 80)).astype(np.float32)

        cutouts_batch = np.stack(
            [
                vis_cutout[:, :, np.newaxis],  # source_pixel: (80, 80, 1)
                nir_cutout[:, :, np.newaxis],  # source_arcsec: (80, 80, 1)
            ],
            axis=0,
        )  # Shape: (2, 80, 80, 1)

        batch_data = {
            "cutouts": cutouts_batch,
            "metadata": [
                {
                    "source_id": "source_pixel",
                    "ra": 10.0,
                    "dec": 20.0,
                    "diameter_pixel": 160,
                    "survey": "Euclid_Wide",
                    "quality_flag": "A",
                    "processing_timestamp": 1642678800.0,
                },
                {
                    "source_id": "source_arcsec",
                    "ra": 11.0,
                    "dec": 21.0,
                    "diameter_arcsec": 3.5,
                    "field_id": "COSMOS_001",
                    "exposure_time": 565.0,
                    "processing_timestamp": 1642678800.0,
                },
            ],
        }

        images_array, metadata_list = prepare_cutouts_for_zarr(batch_data)

        assert len(metadata_list) == 2

        # First source metadata - should preserve exactly
        meta1 = metadata_list[0]
        assert meta1["source_id"] == "source_pixel"
        assert meta1["diameter_pixel"] == 160
        assert "diameter_arcsec" not in meta1
        assert meta1["survey"] == "Euclid_Wide"
        assert meta1["quality_flag"] == "A"

        # Second source metadata - should preserve exactly
        meta2 = metadata_list[1]
        assert meta2["source_id"] == "source_arcsec"
        assert meta2["diameter_arcsec"] == 3.5
        assert "diameter_pixel" not in meta2
        assert meta2["field_id"] == "COSMOS_001"
        assert meta2["exposure_time"] == 565.0
