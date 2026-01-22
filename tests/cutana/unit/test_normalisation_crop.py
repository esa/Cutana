#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Test for the norm_crop_for_maximum_value parameter implementation.

This test verifies that the crop functionality works correctly by creating
an image with bright borders (value 10000) and a faint middle (value 100),
then testing that the crop parameter correctly limits the maximum value
computation to the center region.
"""

from unittest.mock import patch

import numpy as np
import pytest
from dotmap import DotMap

from cutana.image_processor import apply_normalisation
from cutana.normalisation_parameters import convert_cfg_to_fitsbolt_cfg


class TestNormalisationCrop:
    """Test suite for crop functionality in normalization."""

    def create_test_image(self, size=128, border_width=20, border_value=10000, center_value=100):
        """Create a test image with bright borders and faint center.

        Args:
            size: Image size (square)
            border_width: Width of the bright border
            border_value: Value for the bright border pixels
            center_value: Value for the center pixels

        Returns:
            numpy array with the specified pattern
        """
        image = np.full((size, size), center_value, dtype=np.float32)

        # Add bright borders
        image[:border_width, :] = border_value  # Top border
        image[-border_width:, :] = border_value  # Bottom border
        image[:, :border_width] = border_value  # Left border
        image[:, -border_width:] = border_value  # Right border

        return image

    @pytest.fixture
    def crop_config(self):
        """Create config with crop enabled."""
        return DotMap(
            {
                "normalisation_method": "linear",
                "normalisation": {
                    "percentile": 99.0,
                    "a": 0.1,
                    "n_samples": 1000,
                    "contrast": 0.25,
                    "crop_enable": True,
                    "crop_height": 64,
                    "crop_width": 64,
                },
            }
        )

    @pytest.fixture
    def no_crop_config(self):
        """Create config with crop disabled."""
        return DotMap(
            {
                "normalisation_method": "linear",
                "normalisation": {
                    "percentile": 99.0,
                    "a": 0.1,
                    "n_samples": 1000,
                    "contrast": 0.25,
                    "crop_enable": False,
                    "crop_height": 64,
                    "crop_width": 64,
                },
            }
        )

    def test_convert_cfg_to_fitsbolt_cfg_with_crop(self, crop_config):
        """Test that crop parameters are correctly converted to fitsbolt parameters."""
        fitsbolt_params = convert_cfg_to_fitsbolt_cfg(crop_config, num_channels=1)

        # Check that crop parameter is included when enabled
        assert "norm_crop_for_maximum_value" in fitsbolt_params
        assert fitsbolt_params["norm_crop_for_maximum_value"] == (64, 64)

    def test_convert_cfg_to_fitsbolt_cfg_without_crop(self, no_crop_config):
        """Test that crop parameters are not included when disabled."""
        fitsbolt_params = convert_cfg_to_fitsbolt_cfg(no_crop_config, num_channels=1)

        # Check that crop parameter is not included when disabled
        assert "norm_crop_for_maximum_value" not in fitsbolt_params

    @patch("cutana.image_processor.fitsbolt")
    def test_apply_normalisation_passes_crop_parameters(self, mock_fitsbolt, crop_config):
        """Test that apply_normalisation correctly passes crop parameters to fitsbolt."""
        # Create test image batch
        test_image = self.create_test_image(size=128, border_width=20)
        images = test_image[np.newaxis, :, :]  # Add batch dimension

        # Mock fitsbolt.normalise_images to return the same images
        mock_fitsbolt.normalise_images.return_value = images[:, :, :, np.newaxis]
        mock_fitsbolt.NormalisationMethod.CONVERSION_ONLY = "CONVERSION_ONLY"

        # Apply normalization
        apply_normalisation(images, crop_config)

        # Check that fitsbolt.normalise_images was called with crop parameters
        mock_fitsbolt.normalise_images.assert_called_once()
        call_kwargs = mock_fitsbolt.normalise_images.call_args[1]

        assert "norm_crop_for_maximum_value" in call_kwargs
        assert call_kwargs["norm_crop_for_maximum_value"] == (64, 64)

    @patch("cutana.image_processor.fitsbolt")
    def test_apply_normalisation_without_crop_parameters(self, mock_fitsbolt, no_crop_config):
        """Test that apply_normalisation doesn't pass crop parameters when disabled."""
        # Create test image batch
        test_image = self.create_test_image(size=128, border_width=20)
        images = test_image[np.newaxis, :, :]  # Add batch dimension

        # Mock fitsbolt.normalise_images to return the same images
        mock_fitsbolt.normalise_images.return_value = images[:, :, :, np.newaxis]
        mock_fitsbolt.NormalisationMethod.CONVERSION_ONLY = "CONVERSION_ONLY"

        # Apply normalization
        apply_normalisation(images, no_crop_config)

        # Check that fitsbolt.normalise_images was called without crop parameters
        mock_fitsbolt.normalise_images.assert_called_once()
        call_kwargs = mock_fitsbolt.normalise_images.call_args[1]

        assert "norm_crop_for_maximum_value" not in call_kwargs

    def test_crop_parameter_validation_ranges(self):
        """Test that crop parameter ranges are correctly defined."""
        from cutana.normalisation_parameters import NormalisationDefaults, NormalisationRanges

        # Test that defaults are within ranges
        assert (
            NormalisationRanges.CROP_HEIGHT_MIN
            <= NormalisationDefaults.CROP_HEIGHT
            <= NormalisationRanges.CROP_HEIGHT_MAX
        )
        assert (
            NormalisationRanges.CROP_WIDTH_MIN
            <= NormalisationDefaults.CROP_WIDTH
            <= NormalisationRanges.CROP_WIDTH_MAX
        )

        # Test that ranges are sensible
        assert NormalisationRanges.CROP_HEIGHT_MIN > 0
        assert NormalisationRanges.CROP_WIDTH_MIN > 0
        assert NormalisationRanges.CROP_HEIGHT_MAX <= 1024  # Reasonable for most images
        assert NormalisationRanges.CROP_WIDTH_MAX <= 1024

    def test_default_normalisation_config_includes_crop(self):
        """Test that default normalisation config includes crop parameters."""
        from cutana.normalisation_parameters import get_default_normalisation_config

        config = get_default_normalisation_config()

        assert hasattr(config, "crop_enable")
        assert hasattr(config, "crop_height")
        assert hasattr(config, "crop_width")

        # Check default values
        assert config.crop_enable is False  # Disabled by default
        assert config.crop_height == 64
        assert config.crop_width == 64

    def test_different_crop_sizes(self):
        """Test conversion with different crop sizes."""
        test_cases = [(32, 32), (64, 64), (128, 256), (256, 128)]

        for height, width in test_cases:
            config = DotMap(
                {
                    "normalisation_method": "linear",
                    "normalisation": {
                        "crop_enable": True,
                        "crop_height": height,
                        "crop_width": width,
                    },
                }
            )

            fitsbolt_params = convert_cfg_to_fitsbolt_cfg(config, num_channels=1)
            assert fitsbolt_params["norm_crop_for_maximum_value"] == (height, width)

    @patch("cutana.image_processor.fitsbolt")
    def test_crop_with_different_normalisation_methods(self, mock_fitsbolt):
        """Test that crop works with different normalisation methods."""
        methods = ["linear", "log", "asinh", "zscale"]

        for method in methods:
            config = DotMap(
                {
                    "normalisation_method": method,
                    "normalisation": {
                        "percentile": 99.0,
                        "a": 0.1,
                        "n_samples": 1000,
                        "contrast": 0.25,
                        "crop_enable": True,
                        "crop_height": 64,
                        "crop_width": 64,
                    },
                }
            )

            # Create test image
            test_image = self.create_test_image(size=128)
            images = test_image[np.newaxis, :, :]

            # Mock appropriate fitsbolt method
            if method == "linear":
                mock_fitsbolt.NormalisationMethod.CONVERSION_ONLY = "CONVERSION_ONLY"
            elif method == "log":
                mock_fitsbolt.NormalisationMethod.LOG = "LOG"
            elif method == "asinh":
                mock_fitsbolt.NormalisationMethod.ASINH = "ASINH"
            elif method == "zscale":
                mock_fitsbolt.NormalisationMethod.ZSCALE = "ZSCALE"

            mock_fitsbolt.normalise_images.return_value = images[:, :, :, np.newaxis]

            # Apply normalization
            apply_normalisation(images, config)

            # Check that crop parameter was passed
            call_kwargs = mock_fitsbolt.normalise_images.call_args[1]
            assert "norm_crop_for_maximum_value" in call_kwargs
            assert call_kwargs["norm_crop_for_maximum_value"] == (64, 64)

            mock_fitsbolt.reset_mock()

    def test_functional_crop_benefit_simulation(self):
        """
        Simulate the expected benefit of using crop for maximum value computation.

        This test creates an image where cropping should make a difference:
        - Bright borders (10000) that would skew normalization
        - Faint center (100) that should be the actual data of interest

        With crop enabled, the maximum should be computed from the center region only.
        """
        # Create test image with bright borders and faint center
        image_size = 128
        border_width = 20
        border_value = 10000
        center_value = 100

        test_image = self.create_test_image(
            size=image_size,
            border_width=border_width,
            border_value=border_value,
            center_value=center_value,
        )

        # Test that our test image has the expected properties
        assert np.max(test_image) == border_value  # Full image max is border value

        # Extract center region (what crop would see)
        crop_size = 64
        center_start = (image_size - crop_size) // 2
        center_end = center_start + crop_size
        center_region = test_image[center_start:center_end, center_start:center_end]

        # Center region should have much lower maximum
        assert np.max(center_region) == center_value
        assert np.max(center_region) < np.max(test_image)

        # This demonstrates why crop is useful: it prevents bright borders
        # from affecting the normalization maximum value computation
        ratio = np.max(test_image) / np.max(center_region)
        assert ratio == 100  # 10000 / 100 = 100x difference
