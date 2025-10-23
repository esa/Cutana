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

import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from unittest.mock import Mock, patch

from cutana.cutout_extraction import (
    extract_cutouts_vectorized_from_extension,
    extract_cutout_from_extension,
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
            cutouts, success_mask = extract_cutouts_vectorized_from_extension(
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
            cutouts, success_mask = extract_cutouts_vectorized_from_extension(
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
            cutouts, success_mask = extract_cutouts_vectorized_from_extension(
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

            cutouts, success_mask = extract_cutouts_vectorized_from_extension(
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
                cutouts, success_mask = extract_cutouts_vectorized_from_extension(
                    mock_hdu_ones,
                    mock_wcs,
                    np.array([150.0]),
                    np.array([2.0]),
                    np.array([size]),
                    source_ids=["edge_flux_test"],
                    padding_factor=1.0,
                    config=config,
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

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
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

        cutouts, success_mask = extract_cutouts_vectorized_from_extension(
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
            cutouts, success_mask = extract_cutouts_vectorized_from_extension(
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
        batch_cutouts, batch_success = extract_cutouts_vectorized_from_extension(
            mock_hdu_ones,
            mock_wcs,
            ra_array,
            dec_array,
            size_array,
            source_ids=["source1", "source2", "source3"],
            padding_factor=1.0,
            config=None,
        )

        # Individual extractions
        for i, size in enumerate(size_array):
            single_cutouts, single_success = extract_cutouts_vectorized_from_extension(
                mock_hdu_ones,
                mock_wcs,
                np.array([ra_array[i]]),
                np.array([dec_array[i]]),
                np.array([size]),
                source_ids=[f"source{i+1}"],
                padding_factor=1.0,
                config=None,
            )

            assert batch_success[i] == single_success[0], f"Success mismatch for source {i+1}"
            if batch_success[i]:
                assert np.array_equal(
                    batch_cutouts[i], single_cutouts[0]
                ), f"Cutout mismatch for source {i+1}"

    def test_single_wrapper_function(self, mock_hdu_ones, mock_wcs):
        """Test the single-source wrapper function."""
        cutout = extract_cutout_from_extension(
            mock_hdu_ones,
            mock_wcs,
            ra=150.0,
            dec=2.0,
            size_pixels=11,
            padding_factor=1.0,
            config=None,
        )

        assert cutout is not None, "Single extraction returned None"
        assert cutout.shape == (11, 11), f"Expected shape (11, 11), got {cutout.shape}"
        assert np.all(cutout == 1.0), "Expected all ones in cutout"

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
                cutouts, success_mask = extract_cutouts_vectorized_from_extension(
                    mock_hdu_ones,
                    mock_wcs,
                    np.array([150.0]),
                    np.array([2.0]),
                    np.array([size]),
                    source_ids=[f"flux_regression_{size}"],
                    padding_factor=1.0,
                    config=config,
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
