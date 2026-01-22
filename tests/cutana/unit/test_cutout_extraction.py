#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for cutout extraction module.

Tests cover:
- Odd/even size extraction correctness
- Edge case handling (boundaries, corners)
- Flux conversion application
- Padding behavior verification
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from cutana.cutout_extraction import (
    extract_cutouts_vectorized_from_extension,
)


class TestCutoutExtraction:
    """Test suite for cutout extraction functions."""

    @pytest.fixture
    def mock_wcs(self):
        """Create a mock WCS object with basic functionality."""
        wcs = WCS(naxis=2)
        wcs.wcs.crval = [150.0, 2.0]  # Reference RA, Dec
        wcs.wcs.crpix = [50, 50]  # Reference pixel (center of 100x100 image)
        wcs.wcs.cdelt = [-0.0002777778, 0.0002777778]  # ~1 arcsec/pixel
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ["deg", "deg"]
        return wcs

    @pytest.fixture
    def mock_hdu_ones(self, mock_wcs):
        """Create a mock HDU with data filled with ones."""
        # Create a 100x100 image filled with ones
        data = np.ones((100, 100), dtype=np.float32)
        hdu = fits.ImageHDU(data=data)
        hdu.header.update(mock_wcs.to_header())
        hdu.header["MAGZERO"] = 25.0  # For flux conversion testing
        return hdu

    @pytest.fixture
    def mock_hdu_gradient(self, mock_wcs):
        """Create a mock HDU with gradient data for verification."""
        # Create a 100x100 image with gradient (values = x + y)
        y, x = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
        data = (x + y).astype(np.float32)
        hdu = fits.ImageHDU(data=data)
        hdu.header.update(mock_wcs.to_header())
        hdu.header["MAGZERO"] = 25.0
        return hdu

    def test_odd_size_extraction_no_padding(self, mock_hdu_ones, mock_wcs):
        """Test that odd-sized cutouts extract correct amount of data without unwanted padding."""
        # Test various odd sizes
        odd_sizes = [5, 7, 9, 11, 13, 15, 21, 31]

        # Extract from center of image (50, 50)
        ra, dec = 150.0, 2.0  # These should map to approximately center

        for size in odd_sizes:
            cutouts, success_mask, offset_x, offset_y = extract_cutouts_vectorized_from_extension(
                mock_hdu_ones,
                mock_wcs,
                np.array([ra]),
                np.array([dec]),
                np.array([size]),
                source_ids=["test_source"],
                padding_factor=1.0,
                config=None,
            )

            assert success_mask[0], f"Extraction failed for size {size}"
            cutout = cutouts[0]
            assert cutout is not None, f"Cutout is None for size {size}"
            assert cutout.shape == (
                size,
                size,
            ), f"Expected shape ({size}, {size}), got {cutout.shape}"

            # All values should be 1.0 (no zeros from padding) if extracted from center
            assert np.all(
                cutout == 1.0
            ), f"Found non-one values in cutout of size {size}: min={cutout.min()}, max={cutout.max()}"

    def test_even_size_extraction_no_padding(self, mock_hdu_ones, mock_wcs):
        """Test that even-sized cutouts extract correct amount of data without unwanted padding."""
        # Test various even sizes
        even_sizes = [4, 6, 8, 10, 12, 14, 20, 30]

        # Extract from center of image
        ra, dec = 150.0, 2.0

        for size in even_sizes:
            cutouts, success_mask, offset_x, offset_y = extract_cutouts_vectorized_from_extension(
                mock_hdu_ones,
                mock_wcs,
                np.array([ra]),
                np.array([dec]),
                np.array([size]),
                source_ids=["test_source"],
                padding_factor=1.0,
                config=None,
            )

            assert success_mask[0], f"Extraction failed for size {size}"
            cutout = cutouts[0]
            assert cutout is not None, f"Cutout is None for size {size}"
            assert cutout.shape == (
                size,
                size,
            ), f"Expected shape ({size}, {size}), got {cutout.shape}"

            # All values should be 1.0 (no zeros from padding) if extracted from center
            assert np.all(
                cutout == 1.0
            ), f"Found non-one values in cutout of size {size}: min={cutout.min()}, max={cutout.max()}"

    def test_edge_extraction_with_padding(self, mock_hdu_ones, mock_wcs):
        """Test extraction near edges correctly pads with zeros."""
        # Extract a 20x20 cutout from near the edge (position 5,5 in pixel coords)
        # This should require padding since we can't extract full 20x20

        # Mock the world_to_pixel to return edge coordinates
        with patch.object(mock_wcs, "world_to_pixel") as mock_world_to_pixel:
            mock_world_to_pixel.return_value = (5.0, 5.0)  # Near top-left corner

            size = 20
            cutouts, success_mask, offset_x, offset_y = extract_cutouts_vectorized_from_extension(
                mock_hdu_ones,
                mock_wcs,
                np.array([150.0]),
                np.array([2.0]),
                np.array([size]),
                source_ids=["edge_source"],
                padding_factor=1.0,
                config=None,
            )

            assert success_mask[0], "Extraction failed for edge source"
            cutout = cutouts[0]
            assert cutout is not None, "Cutout is None for edge source"
            assert cutout.shape == (
                size,
                size,
            ), f"Expected shape ({size}, {size}), got {cutout.shape}"

            # Should have some zeros from padding and some ones from data
            assert np.sum(cutout == 0.0) > 0, "Expected some zero padding for edge extraction"
            assert np.sum(cutout == 1.0) > 0, "Expected some data values for edge extraction"

    def test_flux_conversion_applied(self, mock_hdu_ones, mock_wcs):
        """Test that flux conversion is correctly applied when configured."""
        # Create a config that enables flux conversion
        config = Mock()
        config.apply_flux_conversion = True

        # Mock the flux conversion function
        with patch("cutana.cutout_extraction.apply_flux_conversion") as mock_flux_conv:
            # Make flux conversion multiply by 2 for testing
            mock_flux_conv.return_value = np.ones((10, 10)) * 2.0

            cutouts, success_mask, offset_x, offset_y = extract_cutouts_vectorized_from_extension(
                mock_hdu_ones,
                mock_wcs,
                np.array([150.0]),
                np.array([2.0]),
                np.array([10]),
                source_ids=["flux_test"],
                padding_factor=1.0,
                config=config,
            )

            # Verify flux conversion was called
            mock_flux_conv.assert_called_once()

            # Verify the result has the converted values
            assert success_mask[0], "Extraction failed"
            cutout = cutouts[0]
            assert np.all(cutout == 2.0), "Flux conversion not applied correctly"

    def test_flux_conversion_with_edge_padding(self, mock_hdu_ones, mock_wcs):
        """Test that flux conversion is applied even when edge padding occurs."""
        config = Mock()
        config.apply_flux_conversion = True

        with patch.object(mock_wcs, "world_to_pixel") as mock_world_to_pixel:
            mock_world_to_pixel.return_value = (5.0, 5.0)  # Near edge

            with patch("cutana.cutout_extraction.apply_flux_conversion") as mock_flux_conv:
                # Flux conversion should be called on the padded array
                def flux_conv_side_effect(config, data, header):
                    # Convert only non-zero values (simulating real behavior)
                    result = data.copy()
                    result[data != 0] *= 2.0
                    return result

                mock_flux_conv.side_effect = flux_conv_side_effect

                size = 20
                cutouts, success_mask, offset_x, offset_y = (
                    extract_cutouts_vectorized_from_extension(
                        mock_hdu_ones,
                        mock_wcs,
                        np.array([150.0]),
                        np.array([2.0]),
                        np.array([size]),
                        source_ids=["edge_flux_test"],
                        padding_factor=1.0,
                        config=config,
                    )
                )

                assert success_mask[0], "Extraction failed"
                cutout = cutouts[0]

                # Should have zeros (padding) and 2.0 (converted data)
                unique_values = np.unique(cutout)
                assert 0.0 in unique_values, "Should have zero padding"
                assert 2.0 in unique_values, "Should have flux-converted values"
                assert 1.0 not in unique_values, "Should not have unconverted values"

    def test_padding_factor_zoom_in(self, mock_hdu_gradient, mock_wcs):
        """Test that padding_factor < 1.0 correctly zooms in."""
        # Request size 20, but with padding_factor=0.5, should extract 10x10
        size = 20
        padding_factor = 0.5
        expected_extraction_size = int(size * padding_factor)

        cutouts, success_mask, offset_x, offset_y = extract_cutouts_vectorized_from_extension(
            mock_hdu_gradient,
            mock_wcs,
            np.array([150.0]),
            np.array([2.0]),
            np.array([size]),
            source_ids=["zoom_in_test"],
            padding_factor=padding_factor,
            config=None,
        )

        assert success_mask[0], "Extraction failed"
        cutout = cutouts[0]
        assert cutout.shape == (
            expected_extraction_size,
            expected_extraction_size,
        ), f"Expected shape ({expected_extraction_size}, {expected_extraction_size}), got {cutout.shape}"

    def test_padding_factor_zoom_out(self, mock_hdu_gradient, mock_wcs):
        """Test that padding_factor > 1.0 correctly zooms out."""
        # Request size 10, but with padding_factor=2.0, should extract 20x20
        size = 10
        padding_factor = 2.0
        expected_extraction_size = int(size * padding_factor)

        cutouts, success_mask, offset_x, offset_y = extract_cutouts_vectorized_from_extension(
            mock_hdu_gradient,
            mock_wcs,
            np.array([150.0]),
            np.array([2.0]),
            np.array([size]),
            source_ids=["zoom_out_test"],
            padding_factor=padding_factor,
            config=None,
        )

        assert success_mask[0], "Extraction failed"
        cutout = cutouts[0]
        assert cutout.shape == (
            expected_extraction_size,
            expected_extraction_size,
        ), f"Expected shape ({expected_extraction_size}, {expected_extraction_size}), got {cutout.shape}"

    def test_gradient_preservation(self, mock_hdu_gradient, mock_wcs):
        """Test that gradient data is preserved correctly during extraction."""
        # Extract a small cutout from a known position
        with patch.object(mock_wcs, "world_to_pixel") as mock_world_to_pixel:
            # Place at pixel position (30, 40)
            mock_world_to_pixel.return_value = (30.0, 40.0)

            size = 5  # Small size to manually verify
            cutouts, success_mask, offset_x, offset_y = extract_cutouts_vectorized_from_extension(
                mock_hdu_gradient,
                mock_wcs,
                np.array([150.0]),
                np.array([2.0]),
                np.array([size]),
                source_ids=["gradient_test"],
                padding_factor=1.0,
                config=None,
            )

            assert success_mask[0], "Extraction failed"
            cutout = cutouts[0]

            # The gradient should be preserved
            # At position (30, 40), with size 5, we extract:
            # x: 28-33 (30 - 2 to 30 + 2 for size 5)
            # y: 38-43 (40 - 2 to 40 + 2 for size 5)
            # So cutout[0, 0] should be gradient[38, 28] = 38 + 28 = 66
            expected_top_left = 38 + 28
            assert (
                cutout[0, 0] == expected_top_left
            ), f"Expected top-left to be {expected_top_left}, got {cutout[0, 0]}"

    def test_batch_extraction_consistency(self, mock_hdu_ones, mock_wcs):
        """Test that batch extraction gives same results as individual extraction."""
        ra_array = np.array([150.0, 150.0, 150.0])
        dec_array = np.array([2.0, 2.0, 2.0])
        size_array = np.array([5, 11, 20])

        # Batch extraction
        batch_cutouts, batch_success, batch_offset_x, batch_offset_y = (
            extract_cutouts_vectorized_from_extension(
                mock_hdu_ones,
                mock_wcs,
                ra_array,
                dec_array,
                size_array,
                source_ids=["source1", "source2", "source3"],
                padding_factor=1.0,
                config=None,
            )
        )

        # Individual extractions
        for i, size in enumerate(size_array):
            single_cutouts, single_success, single_offset_x, single_offset_y = (
                extract_cutouts_vectorized_from_extension(
                    mock_hdu_ones,
                    mock_wcs,
                    np.array([ra_array[i]]),
                    np.array([dec_array[i]]),
                    np.array([size]),
                    source_ids=[f"source{i+1}"],
                    padding_factor=1.0,
                    config=None,
                )
            )

            assert batch_success[i] == single_success[0], f"Success mismatch for source {i+1}"
            if batch_success[i]:
                assert np.array_equal(
                    batch_cutouts[i], single_cutouts[0]
                ), f"Cutout mismatch for source {i+1}"

    def test_flux_conversion_bug_regression(self, mock_hdu_ones, mock_wcs):
        """Regression test for flux conversion bug where it wasn't applied in edge padding cases."""
        config = Mock()
        config.apply_flux_conversion = True

        # Mock the flux conversion to double values
        with patch("cutana.cutout_extraction.apply_flux_conversion") as mock_flux_conv:
            mock_flux_conv.side_effect = lambda cfg, data, header: data * 2.0

            # Test various sizes (especially odd ones that had the bug)
            test_sizes = [5, 11, 15, 21]

            for size in test_sizes:
                cutouts, success_mask, offset_x, offset_y = (
                    extract_cutouts_vectorized_from_extension(
                        mock_hdu_ones,
                        mock_wcs,
                        np.array([150.0]),
                        np.array([2.0]),
                        np.array([size]),
                        source_ids=[f"flux_regression_{size}"],
                        padding_factor=1.0,
                        config=config,
                    )
                )

                assert success_mask[0], f"Extraction failed for size {size}"
                cutout = cutouts[0]
                assert cutout.shape == (size, size), f"Wrong shape for size {size}"

                # All actual data should be 2.0 (flux converted), no 1.0 should remain
                assert np.all(
                    cutout == 2.0
                ), f"Flux conversion not applied correctly for size {size}: got values {np.unique(cutout)}"

            # Verify flux conversion was called for each size
            assert mock_flux_conv.call_count == len(
                test_sizes
            ), f"Flux conversion not called expected number of times"


class TestPixelOffsetAccuracy:
    """Test suite for pixel offset tracking and WCS accuracy."""

    @staticmethod
    def compute_expected_offset(pixel_coord: float, cutout_size: int) -> float:
        """Compute expected pixel offset using the same algorithm as cutout_extraction."""
        half_size_left = cutout_size // 2
        coord_min = int(pixel_coord - half_size_left)
        cutout_center = coord_min + cutout_size / 2.0
        return pixel_coord - cutout_center

    @pytest.fixture
    def mock_wcs_precise(self):
        """Create a WCS with 0.36 arcsec/pixel scale."""
        wcs = WCS(naxis=2)
        wcs.wcs.crval = [180.0, 0.0]
        wcs.wcs.crpix = [50.5, 50.5]
        wcs.wcs.cdelt = [-0.0001, 0.0001]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ["deg", "deg"]
        return wcs

    @pytest.fixture
    def mock_hdu(self, mock_wcs_precise):
        """Create a 100x100 zero-filled HDU."""
        data = np.zeros((100, 100), dtype=np.float32)
        hdu = fits.ImageHDU(data=data)
        hdu.header.update(mock_wcs_precise.to_header())
        return hdu

    def _extract_and_verify_offset(self, hdu, wcs, target_x, target_y, cutout_size):
        """Helper to extract cutout and verify offset matches expected value."""
        target_ra, target_dec = wcs.pixel_to_world_values(target_x, target_y)

        cutouts, success_mask, offset_x, offset_y = extract_cutouts_vectorized_from_extension(
            hdu,
            wcs,
            np.array([target_ra]),
            np.array([target_dec]),
            np.array([cutout_size]),
            source_ids=["test"],
            padding_factor=1.0,
            config=None,
        )
        assert success_mask[0], "Extraction failed"

        actual_px, actual_py = wcs.world_to_pixel_values(target_ra, target_dec)
        expected_x = self.compute_expected_offset(actual_px, cutout_size)
        expected_y = self.compute_expected_offset(actual_py, cutout_size)

        assert (
            abs(offset_x[0] - expected_x) < 1e-10
        ), f"offset_x: got {offset_x[0]}, expected {expected_x}"
        assert (
            abs(offset_y[0] - expected_y) < 1e-10
        ), f"offset_y: got {offset_y[0]}, expected {expected_y}"

        return cutouts, offset_x[0], offset_y[0]

    @pytest.mark.parametrize(
        "target_pos,cutout_size",
        [
            ((50.0, 50.0), 10),  # Even size, integer pixel
            ((50.0, 50.0), 9),  # Odd size, integer pixel
            ((50.5, 50.5), 10),  # Even size, half pixel
            ((50.5, 50.5), 9),  # Odd size, half pixel
        ],
    )
    def test_pixel_offset_matches_expected(
        self, mock_hdu, mock_wcs_precise, target_pos, cutout_size
    ):
        """Test that pixel offset matches expected value for various positions and sizes."""
        self._extract_and_verify_offset(mock_hdu, mock_wcs_precise, *target_pos, cutout_size)

    def test_pixel_offset_sign_convention(self, mock_hdu, mock_wcs_precise):
        """Test that positive offset = target toward top-right (larger pixel indices)."""
        _, offset_x, offset_y = self._extract_and_verify_offset(
            mock_hdu, mock_wcs_precise, 50.7, 50.3, 10
        )
        assert offset_x > 0, f"Expected positive offset_x, got {offset_x}"
        assert offset_y > 0, f"Expected positive offset_y, got {offset_y}"

    @pytest.mark.parametrize("cutout_size", [20, 21])
    def test_cutout_with_3x3_target(self, mock_wcs_precise, cutout_size):
        """Test that computed offset correctly places target at cutout center.

        Physical verification: the 3x3 target centroid + offset correction
        should equal the geometric center of the cutout.
        """
        data = np.zeros((100, 100), dtype=np.float32)
        target_y, target_x = 60, 40
        data[target_y - 1 : target_y + 2, target_x - 1 : target_x + 2] = 1000.0
        hdu = fits.ImageHDU(data=data)
        hdu.header.update(mock_wcs_precise.to_header())

        cutouts, offset_x, offset_y = self._extract_and_verify_offset(
            hdu, mock_wcs_precise, target_x, target_y, cutout_size
        )

        # Physical verification: measure actual target position in cutout
        cutout = cutouts[0]
        bright_pixels = np.where(cutout > 500)
        assert len(bright_pixels[0]) == 9, "Expected 3x3=9 bright pixels"

        centroid_x = np.mean(bright_pixels[1])  # Column index = x
        centroid_y = np.mean(bright_pixels[0])  # Row index = y

        # Geometric center of cutout (in 0-indexed pixel coordinates)
        geometric_center = cutout_size / 2.0

        # The offset tells us: target is at (center + offset) in pixel coords
        # So: corrected_position = centroid - offset should equal geometric_center
        corrected_x = centroid_x - offset_x
        corrected_y = centroid_y - offset_y

        # Verify the corrected position matches the geometric center
        # Tolerance accounts for discrete 3x3 target (centroid is exact, but target spans 3 pixels)
        assert abs(corrected_x - geometric_center) < 0.01, (
            f"X mismatch: centroid={centroid_x:.3f}, offset={offset_x:.3f}, "
            f"corrected={corrected_x:.3f}, expected_center={geometric_center:.3f}"
        )
        assert abs(corrected_y - geometric_center) < 0.01, (
            f"Y mismatch: centroid={centroid_y:.3f}, offset={offset_y:.3f}, "
            f"corrected={corrected_y:.3f}, expected_center={geometric_center:.3f}"
        )

    @pytest.mark.parametrize("diameter_arcsec,expected_pixels", [(3.7, 10), (3.2, 9)])
    def test_non_integer_arcsec_diameter(
        self, mock_hdu, mock_wcs_precise, diameter_arcsec, expected_pixels
    ):
        """Test with diameter_arcsec that doesn't equal exact pixel multiple."""
        from cutana.cutout_extraction import arcsec_to_pixels

        actual_pixels = arcsec_to_pixels(diameter_arcsec, mock_wcs_precise)
        assert (
            actual_pixels == expected_pixels
        ), f"Expected {expected_pixels} pixels, got {actual_pixels}"

        self._extract_and_verify_offset(mock_hdu, mock_wcs_precise, 50.3, 50.7, expected_pixels)

    def test_batch_mixed_even_odd_sizes(self, mock_hdu, mock_wcs_precise):
        """Test batch extraction with mixed even/odd sizes all match expected offsets."""
        positions = [(50.0, 50.0), (40.3, 60.7), (55.5, 45.5), (30.2, 70.8)]
        sizes = [10, 9, 12, 11]

        ra_array, dec_array = [], []
        for px, py in positions:
            ra, dec = mock_wcs_precise.pixel_to_world_values(px, py)
            ra_array.append(ra)
            dec_array.append(dec)

        cutouts, success_mask, offset_x, offset_y = extract_cutouts_vectorized_from_extension(
            mock_hdu,
            mock_wcs_precise,
            np.array(ra_array),
            np.array(dec_array),
            np.array(sizes),
            source_ids=[f"src{i}" for i in range(4)],
            padding_factor=1.0,
            config=None,
        )
        assert np.all(success_mask), "Some extractions failed"

        for i in range(len(positions)):
            actual_px, actual_py = mock_wcs_precise.world_to_pixel_values(ra_array[i], dec_array[i])
            expected_x = self.compute_expected_offset(actual_px, sizes[i])
            expected_y = self.compute_expected_offset(actual_py, sizes[i])
            assert abs(offset_x[i] - expected_x) < 1e-10, f"offset_x[{i}] mismatch"
            assert abs(offset_y[i] - expected_y) < 1e-10, f"offset_y[{i}] mismatch"
            assert cutouts[i].shape == (sizes[i], sizes[i])

    @pytest.mark.parametrize("size", [8, 9, 10, 11, 12, 13, 20, 21])
    def test_offset_consistency_across_sizes(self, mock_hdu, mock_wcs_precise, size):
        """Test that pixel offsets match expected values for various cutout sizes."""
        self._extract_and_verify_offset(mock_hdu, mock_wcs_precise, 50.37, 50.63, size)
