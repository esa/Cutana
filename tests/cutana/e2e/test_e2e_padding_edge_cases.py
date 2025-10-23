#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end test for padding_factor edge cases and boundary conditions."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from astropy.wcs import WCS

from cutana.cutout_extraction import extract_cutouts_vectorized_from_extension


class TestPaddingEdgeCases:
    """Test edge cases for padding_factor in extraction function."""

    @pytest.fixture
    def mock_wcs(self):
        """Create a mock WCS that properly handles coordinate transformations."""
        wcs_obj = MagicMock(spec=WCS)

        def mock_world_to_pixel(coords):
            """Return pixel coordinates based on RA/Dec."""
            # This will be overridden in individual tests
            return np.array([256.0]), np.array([256.0])

        wcs_obj.world_to_pixel = MagicMock(side_effect=mock_world_to_pixel)
        return wcs_obj

    def test_even_sized_cutout_no_padding(self):
        """Test even-sized cutout with padding_factor=1.0."""
        image_size = 512
        image_data = np.ones((image_size, image_size), dtype=np.float32) * 0.5
        # Add distinct pattern
        for i in range(image_size):
            image_data[i, i] = 1.0  # Diagonal line

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([256.0]), np.array([256.0]))
        )

        # Test with even target size
        target_size = 128
        padding_factor = 1.0
        expected_size = int(target_size * padding_factor)  # Should be 128

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_even"],
            padding_factor=padding_factor,
        )

        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0].shape == (expected_size, expected_size)
        # Check that diagonal pattern is preserved
        center = expected_size // 2
        assert cutouts[0][center, center] > 0.9, "Center diagonal should be bright"

    def test_odd_sized_cutout_no_padding(self):
        """Test odd-sized cutout with padding_factor=1.0."""
        image_size = 512
        image_data = np.ones((image_size, image_size), dtype=np.float32) * 0.5
        # Add distinct pattern
        for i in range(image_size):
            image_data[i, i] = 1.0  # Diagonal line

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([256.0]), np.array([256.0]))
        )

        # Test with odd target size
        target_size = 127
        padding_factor = 1.0
        expected_size = int(target_size * padding_factor)  # Should be 127

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_odd"],
            padding_factor=padding_factor,
        )

        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0].shape == (expected_size, expected_size)
        # Check that diagonal pattern is preserved
        center = expected_size // 2
        assert cutouts[0][center, center] > 0.9, "Center diagonal should be bright"

    def test_source_at_image_edge_top_left(self):
        """Test source at top-left corner of image."""
        image_size = 512
        image_data = np.ones((image_size, image_size), dtype=np.float32)
        # Mark specific region near top-left
        image_data[0:50, 0:50] = 2.0

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)
        # Place source at top-left corner
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([25.0]), np.array([25.0]))
        )

        target_size = 128
        padding_factor = 1.0
        expected_size = int(target_size * padding_factor)

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_edge_tl"],
            padding_factor=padding_factor,
        )

        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0].shape == (expected_size, expected_size)
        # Check that bright region is captured
        assert np.any(cutouts[0] > 1.5), "Should capture bright corner region"
        # Check that padding with zeros occurs (not reflection/imputation)
        # Bottom right should be padded with zeros since source is at top-left
        assert np.all(cutouts[0][-10:, -10:] == 0.0), "Edge padding should be zeros"

    def test_source_at_image_edge_bottom_right(self):
        """Test source at bottom-right corner of image."""
        image_size = 512
        image_data = np.ones((image_size, image_size), dtype=np.float32)
        # Mark specific region near bottom-right
        image_data[-50:, -50:] = 2.0

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)
        # Place source at bottom-right corner
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([487.0]), np.array([487.0]))
        )

        target_size = 128
        padding_factor = 1.0
        expected_size = int(target_size * padding_factor)

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_edge_br"],
            padding_factor=padding_factor,
        )

        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0].shape == (expected_size, expected_size)
        # Check that bright region is captured
        assert np.any(cutouts[0] > 1.5), "Should capture bright corner region"
        # Check that padding with zeros occurs
        # Top left should be padded with zeros since source is at bottom-right
        assert np.all(cutouts[0][:10, :10] == 0.0), "Edge padding should be zeros"

    def test_padding_factor_small_zoom_in(self):
        """Test padding_factor=0.25 (minimum allowed zoom-in)."""
        image_size = 512
        # Create gradient pattern for testing
        image_data = np.zeros((image_size, image_size), dtype=np.float32)
        for i in range(image_size):
            for j in range(image_size):
                image_data[i, j] = (i + j) / (2 * image_size)
        # Add bright center square
        image_data[230:280, 230:280] = 2.0

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([256.0]), np.array([256.0]))
        )

        target_size = 128
        padding_factor = 0.25  # Maximum zoom-in
        expected_size = int(target_size * padding_factor)  # Should be 32

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_zoom_in_max"],
            padding_factor=padding_factor,
        )

        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0].shape == (
            expected_size,
            expected_size,
        ), f"Should be {expected_size}x{expected_size}"
        # With 0.25 padding, we extract 32x32 directly
        # The center bright square should be visible in this smaller extraction
        center = expected_size // 2
        assert cutouts[0][center, center] > 1.5, "Center should show bright feature"

    def test_padding_factor_large_zoom_out(self):
        """Test padding_factor=10.0 (maximum allowed zoom-out)."""
        image_size = 512
        # Create a pattern that's easy to verify
        image_data = np.ones((image_size, image_size), dtype=np.float32)
        # Add bright center square
        image_data[206:306, 206:306] = 3.0

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([256.0]), np.array([256.0]))
        )

        target_size = 64  # Smaller target to test extreme zoom
        padding_factor = 10.0  # Maximum zoom-out
        expected_size = int(target_size * padding_factor)  # Should be 640

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_zoom_out_max"],
            padding_factor=padding_factor,
        )

        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0].shape == (
            expected_size,
            expected_size,
        ), f"Should be {expected_size}x{expected_size}"
        # With 10.0 padding on 64px target, we try to extract 640x640 from 512x512
        # This requires padding, edges should be zeros
        assert np.any(cutouts[0] == 0.0), "Should have zero padding at edges"

    def test_fractional_coordinates(self):
        """Test extraction with fractional pixel coordinates."""
        image_size = 512
        # Create checkerboard pattern
        image_data = np.zeros((image_size, image_size), dtype=np.float32)
        for i in range(0, image_size, 20):
            for j in range(0, image_size, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image_data[i : i + 20, j : j + 20] = 1.0

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)
        # Use fractional coordinates
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([256.7]), np.array([255.3]))
        )

        target_size = 100
        padding_factor = 1.0
        expected_size = int(target_size * padding_factor)

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_fractional"],
            padding_factor=padding_factor,
        )

        assert success_mask[0], "Extraction should succeed"
        assert cutouts[0].shape == (expected_size, expected_size)
        # Check that checkerboard pattern is preserved
        assert np.any(cutouts[0] > 0.5) and np.any(
            cutouts[0] < 0.5
        ), "Checkerboard pattern should be preserved"

    def test_multiple_sources_different_padding(self):
        """Test multiple sources with different padding factors in batch."""
        image_size = 512
        image_data = np.ones((image_size, image_size), dtype=np.float32)
        # Add features at different locations
        image_data[100:150, 100:150] = 2.0  # Top-left feature
        image_data[350:400, 350:400] = 3.0  # Bottom-right feature

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)

        # Mock world_to_pixel to return both coordinates as arrays
        def mock_world_to_pixel(coords):
            # Return arrays with coordinates for both sources at once
            return (np.array([125.0, 375.0]), np.array([125.0, 375.0]))

        wcs_obj.world_to_pixel = MagicMock(side_effect=mock_world_to_pixel)

        # Test with different target sizes
        target_sizes = np.array([64, 128])
        padding_factor = 1.0
        expected_sizes = (target_sizes * padding_factor).astype(int)

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0, 180.1]),
            dec_array=np.array([0.0, 0.1]),
            size_pixels_array=target_sizes,
            source_ids=["source1", "source2"],
            padding_factor=padding_factor,
        )

        assert np.all(success_mask), "All extractions should succeed"
        assert cutouts[0].shape == (
            expected_sizes[0],
            expected_sizes[0],
        ), f"First cutout should be {expected_sizes[0]}x{expected_sizes[0]}"
        assert cutouts[1].shape == (
            expected_sizes[1],
            expected_sizes[1],
        ), f"Second cutout should be {expected_sizes[1]}x{expected_sizes[1]}"
        # Check that features are captured
        assert np.max(cutouts[0]) >= 2.0, "First cutout should capture bright feature"
        assert np.max(cutouts[1]) >= 3.0, "Second cutout should capture bright feature"

    def test_source_outside_image_bounds(self):
        """Test source completely outside image bounds."""
        image_size = 512
        image_data = np.ones((image_size, image_size), dtype=np.float32)

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)
        # Place source way outside image
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([1000.0]), np.array([1000.0]))
        )

        target_size = 128
        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["test_outside"],
            padding_factor=1.0,
        )

        # Should fail gracefully
        assert not success_mask[0], "Extraction should fail for out-of-bounds source"
        assert cutouts[0] is None, "Cutout should be None for failed extraction"

    def test_mixed_padding_factors_edge_sources(self):
        """Test sources at edges with different padding factors."""
        image_size = 256  # Smaller for edge testing
        image_data = np.ones((image_size, image_size), dtype=np.float32)
        # Create distinct patterns at corners
        image_data[0:30, 0:30] = 2.0  # Top-left
        image_data[-30:, -30:] = 3.0  # Bottom-right

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)

        # Test corner source with zoom-in
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([15.0]), np.array([15.0]))
        )

        target_size = 64
        padding_factor_zoom_in = 0.5  # Zoom in
        expected_size_zoom_in = int(target_size * padding_factor_zoom_in)  # 32

        # Zoom-in on corner (should capture more detail of corner feature)
        cutouts_zoom_in, success_mask_zoom_in = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["corner_zoom_in"],
            padding_factor=padding_factor_zoom_in,
        )

        # Reset mock for next test
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([15.0]), np.array([15.0]))
        )

        padding_factor_zoom_out = 2.0  # Zoom out
        expected_size_zoom_out = int(target_size * padding_factor_zoom_out)  # 128

        # Zoom-out on corner (should need padding)
        cutouts_zoom_out, success_mask_zoom_out = extract_cutouts_vectorized_from_extension(
            hdu=hdu,
            wcs_obj=wcs_obj,
            ra_array=np.array([180.0]),
            dec_array=np.array([0.0]),
            size_pixels_array=np.array([target_size]),
            source_ids=["corner_zoom_out"],
            padding_factor=padding_factor_zoom_out,
        )

        assert success_mask_zoom_in[0], "Zoom-in extraction should succeed"
        assert success_mask_zoom_out[0], "Zoom-out extraction should succeed"

        # Check extracted sizes match expectations
        assert cutouts_zoom_in[0].shape == (
            expected_size_zoom_in,
            expected_size_zoom_in,
        ), f"Should be {expected_size_zoom_in}x{expected_size_zoom_in}"
        assert cutouts_zoom_out[0].shape == (
            expected_size_zoom_out,
            expected_size_zoom_out,
        ), f"Should be {expected_size_zoom_out}x{expected_size_zoom_out}"

        # Zoom-in should show more concentrated bright region
        bright_pixels_zoom_in = np.sum(cutouts_zoom_in[0] > 1.5)
        # Zoom-out should show bright region but also padding
        bright_pixels_zoom_out = np.sum(cutouts_zoom_out[0] > 1.5)
        zero_pixels_zoom_out = np.sum(cutouts_zoom_out[0] == 0.0)

        assert bright_pixels_zoom_in > 0, "Zoom-in should capture bright corner"
        assert bright_pixels_zoom_out > 0, "Zoom-out should capture bright corner"
        assert zero_pixels_zoom_out > 0, "Zoom-out at edge should have zero padding"

    def test_very_small_cutout_sizes(self):
        """Test extraction with very small target sizes."""
        image_size = 512
        image_data = np.random.random((image_size, image_size)).astype(np.float32)

        hdu = MagicMock()
        hdu.data = image_data

        wcs_obj = MagicMock(spec=WCS)
        wcs_obj.world_to_pixel = MagicMock(
            side_effect=lambda coords: (np.array([256.0]), np.array([256.0]))
        )

        # Test very small sizes, skip 1 as it's an edge case that may not be supported
        for target_size in [2, 4, 8, 16]:
            cutouts, success_mask = extract_cutouts_vectorized_from_extension(
                hdu=hdu,
                wcs_obj=wcs_obj,
                ra_array=np.array([180.0]),
                dec_array=np.array([0.0]),
                size_pixels_array=np.array([target_size]),
                source_ids=[f"test_size_{target_size}"],
                padding_factor=1.0,
            )

            assert success_mask[0], f"Extraction should succeed for size {target_size}"
            assert cutouts[0].shape == (
                target_size,
                target_size,
            ), f"Should be {target_size}x{target_size}"
            # Since we're using random data, just verify structure
            assert cutouts[0] is not None, f"Cutout should not be None for size {target_size}"
            assert not np.any(np.isnan(cutouts[0])), "Should not contain NaN values"
