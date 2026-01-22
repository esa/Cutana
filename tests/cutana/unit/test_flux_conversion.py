#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the preprocessing module.

Tests cover:
- Default preprocessing functionality
- Configuration setup and validation
- Manual preprocessing function validation
- Header keyword configuration
- Error handling for preprocessing
"""

import numpy as np
import pytest
from astropy.io import fits
from dotmap import DotMap

from cutana.flux_conversion import (
    apply_flux_conversion,
    convert_mosaic_to_flux,
)
from cutana.get_default_config import get_default_config
from cutana.validate_config import (
    _validate_flux_conversion_config as validate_flux_conversion_config,
)


class TestPreprocessingFunctions:
    """Test suite for preprocessing functions."""

    @pytest.fixture
    def mock_image_data(self):
        """Create mock image data for testing."""
        return np.random.random((100, 100)).astype(np.float32)

    @pytest.fixture
    def mock_header_with_magzero(self):
        """Create mock FITS header with MAGZERO keyword."""
        header = fits.Header()
        header["MAGZERO"] = 25.0
        header["OBJECT"] = "Test Object"
        return header

    @pytest.fixture
    def mock_header_with_zp(self):
        """Create mock FITS header with ZP keyword."""
        header = fits.Header()
        header["ZP"] = 23.5
        header["OBJECT"] = "Test Object"
        return header

    @pytest.fixture
    def mock_header_no_zeropoint(self):
        """Create mock FITS header without zeropoint keyword."""
        header = fits.Header()
        header["OBJECT"] = "Test Object"
        return header

    def test_convert_mosaic_to_flux(self, mock_image_data):
        """Test the basic flux conversion function."""
        zp = 25.0

        # Test flux conversion
        flux = convert_mosaic_to_flux(mock_image_data, zp)

        # Check output properties
        assert isinstance(flux, np.ndarray)
        assert flux.shape == mock_image_data.shape
        assert flux.dtype == mock_image_data.dtype

        # Check conversion formula
        expected_flux = mock_image_data * 10 ** (-0.4 * zp) * 3631.0
        np.testing.assert_array_almost_equal(flux, expected_flux)

    def test_apply_flux_conversion_disabled(self, mock_image_data, mock_header_with_magzero):
        """Test preprocessing when apply_flux_conversion is False."""
        config = get_default_config()
        config.apply_flux_conversion = False

        original_data = mock_image_data.copy()
        result = apply_flux_conversion(config, mock_image_data, mock_header_with_magzero)

        # Data should be unchanged
        np.testing.assert_array_equal(result, original_data)

    def test_apply_flux_conversion_enabled_with_magzero(
        self, mock_image_data, mock_header_with_magzero
    ):
        """Test preprocessing when apply_flux_conversion is True with MAGZERO header."""
        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = None
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        result = apply_flux_conversion(config, mock_image_data, mock_header_with_magzero)

        # Should apply flux conversion
        expected = convert_mosaic_to_flux(mock_image_data, 25.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_flux_conversion_enabled_with_zp(self, mock_image_data, mock_header_with_zp):
        """Test preprocessing with different zeropoint keyword (ZP)."""
        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = None
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "ZP"})

        result = apply_flux_conversion(config, mock_image_data, mock_header_with_zp)

        # Should apply flux conversion with ZP value
        expected = convert_mosaic_to_flux(mock_image_data, 23.5)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_flux_conversion_missing_zeropoint(
        self, mock_image_data, mock_header_no_zeropoint
    ):
        """Test preprocessing when zeropoint keyword is missing from header."""
        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = None
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        # Should raise ValueError when header keyword is missing
        with pytest.raises(
            ValueError, match="Zeropoint keyword 'MAGZERO' not found in FITS header"
        ):
            apply_flux_conversion(config, mock_image_data, mock_header_no_zeropoint)

    def test_apply_flux_conversion_missing_zeropoint_with_instrument(self, mock_image_data):
        """Test preprocessing when zeropoint keyword is missing regardless of instrument."""
        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = None
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        # Create header with instrument but no MAGZERO
        header = fits.Header()
        header["INSTRUME"] = "NIR-H"
        header["OBJECT"] = "Test Object"

        # Should raise ValueError when header keyword is missing, regardless of instrument
        with pytest.raises(
            ValueError, match="Zeropoint keyword 'MAGZERO' not found in FITS header"
        ):
            apply_flux_conversion(config, mock_image_data, header)

    def test_apply_flux_conversion_instrument_detection(self, mock_image_data):
        """Test flux conversion with different instruments when MAGZERO is present."""
        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = None
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        # Test each instrument type with MAGZERO present
        instruments = [("VIS", 24.6), ("NIR_H", 29.9), ("NIR_J", 30.0), ("NIR_Y", 29.8)]

        for instrument, expected_zp in instruments:
            header = fits.Header()
            header["INSTRUME"] = instrument
            header["MAGZERO"] = expected_zp  # Provide the zeropoint in header

            result = apply_flux_conversion(config, mock_image_data, header)
            expected = convert_mosaic_to_flux(mock_image_data, expected_zp)
            np.testing.assert_array_almost_equal(result, expected)

        # Test that missing MAGZERO raises error regardless of instrument
        header = fits.Header()
        header["INSTRUME"] = "VIS"
        # No MAGZERO header

        with pytest.raises(
            ValueError, match="Zeropoint keyword 'MAGZERO' not found in FITS header"
        ):
            apply_flux_conversion(config, mock_image_data, header)

    def test_user_flux_conversion_function_simple_square(
        self, mock_image_data, mock_header_with_magzero
    ):
        """Test preprocessing with manual function that squares the data."""

        def square_function(img, header):
            return img**2

        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = square_function
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        result = apply_flux_conversion(config, mock_image_data, mock_header_with_magzero)

        # Should apply the square function
        expected = mock_image_data**2
        np.testing.assert_array_almost_equal(result, expected)

    def test_user_flux_conversion_function_one_input_error(self):
        """Test that manual preprocessing function with one input raises ValueError."""

        def bad_function(img):
            return img * 2

        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = bad_function
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        # Should raise ValueError during config validation
        with pytest.raises(
            ValueError, match="user_flux_conversion_function must take exactly 2 arguments"
        ):
            validate_flux_conversion_config(config)

    def test_user_flux_conversion_function_three_inputs_error(self):
        """Test that manual preprocessing function with three inputs raises ValueError."""

        def bad_function(img, header, extra):
            return img * 2

        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = bad_function
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        # Should raise ValueError during config validation
        with pytest.raises(
            ValueError, match="user_flux_conversion_function must take exactly 2 arguments"
        ):
            validate_flux_conversion_config(config)

    def test_user_flux_conversion_function_not_callable(self):
        """Test that non-callable manual preprocessing function raises ValueError."""
        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = "not_a_function"
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        # Should raise ValueError during config validation
        with pytest.raises(ValueError, match="user_flux_conversion_function must be callable"):
            validate_flux_conversion_config(config)

    def test_flux_conversion_default_config_values(self):
        """Test that default config has correct flux conversion values."""
        config = get_default_config()

        # Should have correct defaults
        assert config.apply_flux_conversion is True
        assert config.user_flux_conversion_function is None
        assert config.flux_conversion_keywords.AB_zeropoint == "MAGZERO"

    def test_flux_conversion_config_validation_succeeds(self):
        """Test that flux conversion config validation succeeds with valid config."""
        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = None
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "ZP_CUSTOM"})

        # Should not raise any errors
        validate_flux_conversion_config(config)

    def test_flux_conversion_different_zeropoints(self, mock_image_data):
        """Test flux conversion with different zeropoint values."""
        test_zeropoints = [20.0, 25.0, 30.0]

        for zp in test_zeropoints:
            flux = convert_mosaic_to_flux(mock_image_data, zp)

            # Check that different zeropoints give different results
            expected = mock_image_data * 10 ** (-0.4 * zp) * 3631.0
            np.testing.assert_array_almost_equal(flux, expected)

            # Check that flux values are positive
            assert np.all(flux >= 0)

    def test_preprocessing_preserves_data_type(self, mock_header_with_magzero):
        """Test that preprocessing preserves the input data type."""
        for dtype in [np.float32, np.float64, np.int32]:
            data = np.random.random((50, 50)).astype(dtype)

            config = get_default_config()
            config.apply_flux_conversion = True
            config.user_flux_conversion_function = None
            config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

            result = apply_flux_conversion(config, data, mock_header_with_magzero)

            # Result should maintain input dtype for float types
            if dtype in [np.float32, np.float64]:
                assert result.dtype == dtype
            else:
                # Integer types might be promoted to float
                assert result.dtype in [np.float32, np.float64]

    def test_manual_function_with_header_access(self, mock_image_data):
        """Test manual preprocessing function that uses header information."""

        def header_aware_function(img, header):
            # Use header value for scaling
            scale_factor = header.get("SCALE", 1.0)
            return img * scale_factor

        header = fits.Header()
        header["SCALE"] = 2.5
        header["MAGZERO"] = 25.0

        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = header_aware_function
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        result = apply_flux_conversion(config, mock_image_data, header)

        # Should apply the scale factor from header
        expected = mock_image_data * 2.5
        np.testing.assert_array_almost_equal(result, expected)

    def test_preprocessing_error_handling(self, mock_image_data, mock_header_with_magzero):
        """Test error handling in preprocessing function."""

        def error_function(img, header):
            raise RuntimeError("Preprocessing error")

        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = error_function
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        # Should propagate the error from manual function
        with pytest.raises(RuntimeError, match="Preprocessing error"):
            apply_flux_conversion(config, mock_image_data, mock_header_with_magzero)

    def test_preprocessing_with_zero_data(self, mock_header_with_magzero):
        """Test preprocessing with zero-valued data."""
        zero_data = np.zeros((10, 10), dtype=np.float32)

        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = None
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        result = apply_flux_conversion(config, zero_data, mock_header_with_magzero)

        # Result should still be zeros (scaled by conversion factor)
        assert np.all(result >= 0)
        assert result.shape == zero_data.shape

    def test_preprocessing_with_negative_data(self, mock_header_with_magzero):
        """Test preprocessing with negative data values."""
        negative_data = np.array([[-1.0, -0.5], [0.5, 1.0]], dtype=np.float32)

        config = get_default_config()
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = None
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})

        result = apply_flux_conversion(config, negative_data, mock_header_with_magzero)

        # Should handle negative values appropriately
        assert result.shape == negative_data.shape
        # Negative input values should result in negative flux values
        assert np.any(result < 0)
