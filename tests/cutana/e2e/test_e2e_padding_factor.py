#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end test for padding_factor functionality."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from astropy.wcs import WCS

from cutana.cutout_extraction import extract_cutouts_vectorized_from_extension


class TestPaddingFactorE2E:
    """Test padding_factor in extraction function."""

    @pytest.fixture
    def mock_fits_data(self):
        """Create mock FITS data for testing."""
        # Create a 512x512 test image with a gradient pattern
        image_size = 512
        image_data = np.zeros((image_size, image_size), dtype=np.float32)

        # Add a gradient pattern for testing
        for i in range(image_size):
            for j in range(image_size):
                image_data[i, j] = (i + j) / (2 * image_size)

        # Add some features in the center
        center = image_size // 2
        image_data[center - 50 : center + 50, center - 50 : center + 50] = 1.0

        # Create mock HDU
        hdu = MagicMock()
        hdu.data = image_data

        # Create mock WCS
        wcs_obj = MagicMock(spec=WCS)

        # Mock world_to_pixel to return center coordinates
        def mock_world_to_pixel(coords):
            # Return center of image for any input
            return np.array([256.0]), np.array([256.0])

        wcs_obj.world_to_pixel = MagicMock(side_effect=mock_world_to_pixel)

        return hdu, wcs_obj, image_data

    def test_padding_factor_zoom_in(self, mock_fits_data):
        """Test padding_factor < 1.0 (zoom-in effect)."""
        hdu, wcs_obj, image_data = mock_fits_data

        # Test parameters
        target_size = 128
        padding_factor = 0.5  # Zoom in
        expected_size = int(target_size * padding_factor)  # Should be 64

        # Extract cutout with padding factor
        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_source"],
            padding_factor=padding_factor,
        )

        # Check extraction succeeded
        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0] is not None, "Cutout should not be None"

        # Check output size matches extraction size (not target size)
        assert cutouts[0].shape == (
            expected_size,
            expected_size,
        ), f"Output should be {expected_size}x{expected_size}, got {cutouts[0].shape}"

        # With zoom-in, we extract a smaller area (64x64)
        # The center should show the bright square
        center = expected_size // 2
        central_value = cutouts[0][center, center]
        assert central_value > 0.5, "Center should show bright feature when zoomed in"

    def test_padding_factor_no_padding(self, mock_fits_data):
        """Test padding_factor = 1.0 (no padding, default behavior)."""
        hdu, wcs_obj, image_data = mock_fits_data

        # Test parameters
        target_size = 128
        padding_factor = 1.0  # No padding
        expected_size = int(target_size * padding_factor)  # Should be 128

        # Extract cutout with padding factor
        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_source"],
            padding_factor=padding_factor,
        )

        # Check extraction succeeded
        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0] is not None, "Cutout should not be None"

        # Check output size is extraction size
        assert cutouts[0].shape == (
            expected_size,
            expected_size,
        ), f"Output should be {expected_size}x{expected_size}, got {cutouts[0].shape}"

    def test_padding_factor_zoom_out(self, mock_fits_data):
        """Test padding_factor > 1.0 (zoom-out effect with padding)."""
        hdu, wcs_obj, image_data = mock_fits_data

        # Test parameters
        target_size = 128
        padding_factor = 2.0  # Zoom out
        expected_size = int(target_size * padding_factor)  # Should be 256

        # Extract cutout with padding factor
        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_source"],
            padding_factor=padding_factor,
        )

        # Check extraction succeeded
        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0] is not None, "Cutout should not be None"

        # Check output size is extraction size
        assert cutouts[0].shape == (
            expected_size,
            expected_size,
        ), f"Output should be {expected_size}x{expected_size}, got {cutouts[0].shape}"

        # With zoom-out, we extract a larger area (256x256)
        # Should capture more of the surrounding gradient
        corner_value = cutouts[0][0, 0]
        assert corner_value < 0.5, "Corners should show gradient when zoomed out"

    def test_padding_factor_large_zoom_out(self, mock_fits_data):
        """Test large padding_factor with edge sources (should add black padding)."""
        hdu, wcs_obj, image_data = mock_fits_data

        # Test parameters
        target_size = 128
        padding_factor = 5.0  # Large zoom out
        expected_size = int(target_size * padding_factor)  # Should be 640

        # Extract cutout with padding factor
        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_source"],
            padding_factor=padding_factor,
        )

        # Check extraction succeeded
        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0] is not None, "Cutout should not be None"

        # Check output size is extraction size
        assert cutouts[0].shape == (
            expected_size,
            expected_size,
        ), f"Output should be {expected_size}x{expected_size}, got {cutouts[0].shape}"

        # With large zoom-out (5.0), we try to extract 640x640 from a 512x512 image
        # This will require padding with zeros at the edges
        # Check that edges have zeros (black padding)
        edge_sum = (
            np.sum(cutouts[0][0, :])
            + np.sum(cutouts[0][-1, :])
            + np.sum(cutouts[0][:, 0])
            + np.sum(cutouts[0][:, -1])
        )

        # Some edge values should be zero due to padding
        assert edge_sum < 100, "Edges should have padding (zeros) for large zoom-out"
