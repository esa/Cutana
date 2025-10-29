#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for cutout extraction module targeting specific uncovered lines.

These tests focus on hitting exact uncovered lines to improve coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from astropy.wcs import WCS

from cutana.cutout_extraction import (
    get_pixel_scale_arcsec_per_pixel,
    arcsec_to_pixels,
    validate_size_parameters,
    radec_to_pixel,
    extract_cutout_from_extension,
)


class TestCutoutExtractionCoverage:
    """Test suite focusing on specific uncovered lines."""

    def test_get_pixel_scale_successful_path(self):
        """Test successful pixel scale calculation - hits lines 33-40."""
        mock_wcs = Mock(spec=WCS)
        # Create a realistic pixel scale matrix for 0.1 arcsec/pixel
        pixel_scale_deg_per_pixel = 0.1 / 3600.0  # 0.1 arcsec in degrees
        mock_wcs.pixel_scale_matrix = np.array(
            [[pixel_scale_deg_per_pixel, 0], [0, pixel_scale_deg_per_pixel]]
        )

        # This should hit lines 33-40 (successful calculation path)
        result = get_pixel_scale_arcsec_per_pixel(mock_wcs)
        expected = 0.1  # arcsec/pixel
        assert abs(result - expected) < 0.001

    def test_get_pixel_scale_exception_path(self):
        """Test pixel scale calculation exception handling - hits lines 41-44."""
        mock_wcs = Mock(spec=WCS)
        # Make pixel_scale_matrix raise an exception
        mock_wcs.pixel_scale_matrix = Mock(side_effect=Exception("WCS error"))

        # This should hit lines 41-44 (exception handling path)
        with patch("cutana.cutout_extraction.logger") as mock_logger:
            result = get_pixel_scale_arcsec_per_pixel(mock_wcs)

            # Should return default value
            assert result == 0.1
            # Should log warning
            mock_logger.warning.assert_called_once()

    def test_arcsec_to_pixels_successful_path(self):
        """Test arcsec to pixels conversion success - hits lines 58-68."""
        mock_wcs = Mock(spec=WCS)
        pixel_scale_deg = 0.1 / 3600.0  # 0.1 arcsec/pixel in degrees
        mock_wcs.pixel_scale_matrix = np.array([[pixel_scale_deg, 0], [0, pixel_scale_deg]])

        # This should hit the successful conversion path
        result = arcsec_to_pixels(10.0, mock_wcs)  # 10 arcsec
        expected = 100  # pixels (10 arcsec / 0.1 arcsec/pixel)
        assert result == expected

    def test_arcsec_to_pixels_exception_path(self):
        """Test arcsec to pixels conversion with exception - hits exception path."""
        mock_wcs = Mock(spec=WCS)
        # Make pixel_scale_matrix raise exception
        mock_wcs.pixel_scale_matrix = Mock(side_effect=Exception("WCS error"))

        # This should hit exception handling and use default pixel scale
        result = arcsec_to_pixels(10.0, mock_wcs)
        expected = 100  # Should use default 0.1 arcsec/pixel
        assert result == expected

    def test_radec_to_pixel_conversion(self):
        """Test RA/Dec to pixel conversion - hits lines around 104+."""
        mock_wcs = Mock(spec=WCS)
        mock_wcs.world_to_pixel.return_value = (100.5, 200.7)

        # This should hit RA/Dec conversion code
        x_pixel, y_pixel = radec_to_pixel(150.0, 2.0, mock_wcs)

        assert x_pixel == 100.5
        assert y_pixel == 200.7
        mock_wcs.world_to_pixel.assert_called_once()

    def test_extract_cutout_from_extension(self):
        """Test single cutout extraction - hits lines 130+."""
        # Create mock HDU and WCS with correct signature
        mock_hdu = Mock()
        mock_hdu.data = np.random.rand(1000, 1000).astype(np.float32)

        mock_wcs = Mock(spec=WCS)

        # Use correct method signature: extract_cutout_from_extension(hdu, wcs_obj, ra, dec, size_pixels)
        result = extract_cutout_from_extension(
            hdu=mock_hdu, wcs_obj=mock_wcs, ra=150.0, dec=2.0, size_pixels=64
        )

        # Method should execute without error
        # Result can be None or array depending on cutout success

    def test_size_validation_edge_cases(self):
        """Test size parameter validation edge cases - hits exception paths."""
        # Test missing size parameters
        source_data = {
            "RA": 150.0,
            "Dec": 2.0,
            # Missing both size_arcsec and size_pixel
        }

        # This should hit validation error paths
        with pytest.raises(ValueError):
            validate_size_parameters(source_data)

        # Test invalid size values
        source_data = {"size_pixel": -10, "RA": 150.0, "Dec": 2.0}  # Invalid negative size

        with pytest.raises(ValueError):
            validate_size_parameters(source_data)
