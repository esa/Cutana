#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Integration tests for multi-channel processing with different resolutions.

Tests the full pipeline from cutout extraction through resizing and channel combination
when different FITS files return cutouts of different sizes.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dotmap import DotMap

from cutana.cutout_process_utils import _process_sources_batch_vectorized_with_fits_set


class TestMultiResolutionChannelProcessing:
    """Integration tests for processing channels with different resolutions."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing."""
        from cutana.get_default_config import get_default_config

        config = get_default_config()
        config.target_resolution = 64
        config.data_type = "float32"
        config.normalisation_method = "linear"
        config.interpolation = "bilinear"
        return config

    @patch("cutana.cutout_process_utils.extract_cutouts_batch_vectorized")
    def test_multi_resolution_processing_without_channel_combination(
        self, mock_extract_cutouts, base_config
    ):
        """
        Test that the full processing pipeline handles different resolution cutouts correctly.

        This integration test verifies that when different FITS files return cutouts of
        different sizes, they are properly resized to the target resolution.
        """
        # Create mock cutouts with different resolutions
        small_cutout = np.random.random((64, 64)).astype(np.float32) * 100
        medium_cutout = np.random.random((128, 128)).astype(np.float32) * 100
        large_cutout = np.random.random((256, 256)).astype(np.float32) * 100

        # Source data with multiple FITS files (3 channels)
        sources_batch = [
            {
                "SourceID": "multi_res_test_001",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "['/mock/ch1.fits', '/mock/ch2.fits', '/mock/ch3.fits']",
            }
        ]

        # Mock WCS object
        mock_wcs = MagicMock()

        # Mock the extract_cutouts_batch_vectorized to return different sized cutouts
        # Returns 5 values: (combined_cutouts, combined_wcs, source_ids, pixel_scale, combined_offsets)
        mock_pixel_scale = 0.1  # arcsec/pixel

        def mock_extract_side_effect(
            sources, hdul, wcs_dict, extensions, padding_factor=1.0, config=None
        ):
            # Simulate different FITS files returning different sized cutouts
            fits_name = getattr(hdul, "_mock_name", "unknown")
            source_id = sources[0]["SourceID"]

            # Mock offsets (pixel offset from cutout center to target center)
            mock_offsets = {source_id: {"x": 0.0, "y": 0.0}}

            # Return PRIMARY extension (default) with different sizes per file
            if "ch1" in fits_name:
                return (
                    {source_id: {"PRIMARY": large_cutout}},
                    {source_id: {"PRIMARY": mock_wcs}},
                    [source_id],
                    mock_pixel_scale,
                    mock_offsets,
                )
            elif "ch2" in fits_name:
                return (
                    {source_id: {"PRIMARY": medium_cutout}},
                    {source_id: {"PRIMARY": mock_wcs}},
                    [source_id],
                    mock_pixel_scale,
                    mock_offsets,
                )
            elif "ch3" in fits_name:
                return (
                    {source_id: {"PRIMARY": small_cutout}},
                    {source_id: {"PRIMARY": mock_wcs}},
                    [source_id],
                    mock_pixel_scale,
                    mock_offsets,
                )
            else:
                return {}, {}, [], 0.1, {}

        mock_extract_cutouts.side_effect = mock_extract_side_effect

        # Configure with default channel_weights (no combination, preserve all channels)
        config = DotMap(base_config.copy())
        config.target_resolution = (64, 64)
        config.fits_extensions = ["PRIMARY"]
        # Set channel_weights to pass through all channels without combination
        # Using single weight [1.0] for each channel preserves them separately
        config.channel_weights = {"PRIMARY": [1.0, 1.0, 1.0]}  # 3 output channels from PRIMARY

        # Create mock loaded FITS data
        mock_loaded_fits_data = {}
        for fits_path in ["/mock/ch1.fits", "/mock/ch2.fits", "/mock/ch3.fits"]:
            mock_hdul = MagicMock()
            mock_hdul._mock_name = fits_path
            mock_hdul.close = MagicMock()
            mock_wcs_dict = {"PRIMARY": mock_wcs}
            mock_loaded_fits_data[fits_path] = (mock_hdul, mock_wcs_dict)

        # Call the processing function
        results = _process_sources_batch_vectorized_with_fits_set(
            sources_batch,
            mock_loaded_fits_data,
            config,
            profiler=None,
            process_name=None,
            job_tracker=None,
        )

        # Verify results structure
        assert len(results) == 1, "Should return one batch result"
        batch_result = results[0]
        assert "cutouts" in batch_result, "Result should contain cutouts"
        assert "metadata" in batch_result, "Result should contain metadata"
        assert len(batch_result["metadata"]) == 1, "Should have metadata for one source"

        # Verify metadata
        result_metadata = batch_result["metadata"][0]
        assert result_metadata["source_id"] == "multi_res_test_001"

        # Verify cutouts tensor shape
        cutouts_tensor = batch_result["cutouts"]
        assert cutouts_tensor.shape[0] == 1, "Should have 1 source"

        # Without channel combination, should have 3 separate channels
        assert cutouts_tensor.shape[-1] == 3, "Should have 3 separate channels"

        # Verify all dimensions are resized to target resolution
        assert cutouts_tensor.shape[1] == 64, "Height should be 64"
        assert cutouts_tensor.shape[2] == 64, "Width should be 64"

        # Verify data is not all zeros (actual processing occurred)
        assert cutouts_tensor.max() > 0, "Cutouts should contain actual data"
        assert cutouts_tensor.min() >= 0, "Cutout values should be non-negative"

        # Verify extract was called for each FITS file (3 files)
        assert mock_extract_cutouts.call_count == 3, "Should extract from all 3 FITS files"
