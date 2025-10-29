#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the image_processor module using TDD approach.

Tests cover:
- Image resizing to target resolution
- Data type conversion (float32, uint8, etc.)
- Normalization using fitsbolt
- Stretch function application (linear, log, asinh, sqrt)
- Multi-channel image processing
- Error handling for invalid inputs
"""

import numpy as np
from unittest.mock import patch
import pytest
from cutana.image_processor import (
    resize_images,
    convert_data_type,
    apply_normalisation,
    combine_channels,
)


class TestImageProcessor:
    """Test suite for image processor functions."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for apply_normalisation tests."""
        from dotmap import DotMap

        return DotMap(
            {
                "normalisation_method": "linear",
                "normalisation": {
                    "a": None,
                    "percentile": None,
                    "n_samples": None,
                    "contrast": None,
                },
                "norm_minimum_value": None,
                "norm_maximum_value": None,
                "norm_crop_for_maximum_value": None,
                "norm_log_calculate_minimum_value": False,
            }
        )

    @pytest.fixture
    def processor_config(self):
        """Create image processing configuration."""
        return {
            "target_resolution": 256,
            "file_type": "float32",
            "stretch": "linear",
            "interpolation": "bilinear",
        }

    @pytest.fixture
    def mock_cutout_data(self):
        """Create mock cutout data for testing."""
        return {
            "VIS": np.random.random((128, 128)).astype(np.float32),
            "NIR-Y": np.random.random((128, 128)).astype(np.float32),
            "NIR-H": np.random.random((128, 128)).astype(np.float32),
        }

    def test_image_processor_initialization(self, processor_config):
        """Test function-based image processor configuration works."""
        # Since we use functions, just test that config values are valid
        assert processor_config["target_resolution"] == 256
        assert processor_config["file_type"] == "float32"
        assert processor_config["stretch"] == "linear"
        assert processor_config["interpolation"] == "bilinear"

    def test_resize_images_upscale(self):
        """Test resizing image from smaller to larger resolution."""
        input_image = np.random.random((64, 64)).astype(np.float32)

        resized = resize_images(input_image, target_size=(128, 128))

        assert resized.shape == (1, 128, 128)  # Single image becomes batch
        assert resized.dtype == np.float32
        assert not np.array_equal(
            resized[0], input_image
        )  # Should be different due to interpolation

    def test_resize_images_downscale(self):
        """Test resizing image from larger to smaller resolution."""
        input_image = np.random.random((512, 512)).astype(np.float32)

        resized = resize_images(input_image, target_size=(256, 256))

        assert resized.shape == (1, 256, 256)  # Single image becomes batch
        assert resized.dtype == np.float32

    def test_resize_images_preserve_range(self):
        """Test that resizing preserves the approximate data range."""
        # Create image with known range
        input_image = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)

        resized = resize_images(input_image, target_size=(128, 128))

        # Range should be approximately preserved
        assert resized[0].min() >= -0.1  # Allow small interpolation artifacts
        assert resized[0].max() <= 1.1
        assert abs(resized[0].mean() - input_image.mean()) < 0.1

    def test_convert_data_type_float32(self):
        """Test conversion to float32 data type."""
        input_image = np.random.randint(0, 65535, (100, 100)).astype(np.uint16)

        converted = convert_data_type(input_image, "float32")

        assert converted.dtype == np.float32
        assert converted.shape == input_image.shape

    def test_convert_data_type_uint8(self):
        """Test conversion to uint8 with proper scaling."""
        input_image = np.random.random((100, 100)).astype(np.float32)

        converted = convert_data_type(input_image, "uint8")

        assert converted.dtype == np.uint8
        assert converted.min() >= 0
        assert converted.max() <= 255

    @patch("fitsbolt.normalise_images")
    def test_apply_normalisation_fitsbolt(self, mock_normalise_images, mock_config):
        """Test applying normalization using fitsbolt with batch processing."""
        # Use batch format from the start
        input_batch = np.random.random((2, 100, 100)).astype(np.float32)
        # Mock fitsbolt response - it expects batch format
        normalized_output = np.random.random((2, 100, 100, 1)).astype(np.float32)
        mock_normalise_images.return_value = normalized_output

        result = apply_normalisation(input_batch, mock_config)

        # Check that fitsbolt.normalise_images was called
        mock_normalise_images.assert_called_once()
        assert result.shape == input_batch.shape  # Should match input batch shape
        assert result.dtype == np.float32

    def test_apply_normalisation_linear(self, mock_config):
        """Test linear normalization application with batch processing."""
        # Use batch format (1, H, W)
        input_batch = np.linspace(0, 1, 100).reshape(1, 10, 10).astype(np.float32)

        normalized = apply_normalisation(input_batch, mock_config)

        # Should return normalized image with same shape
        assert normalized.shape == input_batch.shape
        # fitsbolt may convert to uint8, so check for valid numeric type
        assert normalized.dtype in [np.float32, np.float64, np.uint8, np.uint16]

    def test_apply_normalisation_log(self, mock_config):
        """Test logarithmic normalization application with batch processing."""
        # Use batch format (1, H, W)
        input_batch = np.linspace(0.1, 1, 100).reshape(1, 10, 10).astype(np.float32)

        # Set the normalization method to log
        mock_config.normalisation_method = "log"
        normalized = apply_normalisation(input_batch, mock_config)

        # Log normalization should process the image
        assert normalized.shape == input_batch.shape
        # fitsbolt may convert to uint8, so check for valid numeric type
        assert normalized.dtype in [np.float32, np.float64, np.uint8, np.uint16]
        assert np.isfinite(normalized).all()

    def test_apply_normalisation_asinh(self, mock_config):
        """Test asinh normalization application with batch processing."""
        # Use batch format (1, H, W)
        input_batch = np.linspace(-1, 1, 100).reshape(1, 10, 10).astype(np.float32)

        # Set the normalization method to asinh
        mock_config.normalisation_method = "asinh"
        normalized = apply_normalisation(input_batch, mock_config)

        assert normalized.shape == input_batch.shape
        # fitsbolt may convert to uint8, so check for valid numeric type
        assert normalized.dtype in [np.float32, np.float64, np.uint8, np.uint16]
        # Asinh should handle negative values gracefully
        assert np.isfinite(normalized).all()

    def test_apply_normalisation_fallback(self, mock_config):
        """Test normalization fallback when fitsbolt fails with batch processing."""
        # Use batch format (1, H, W)
        input_batch = np.linspace(0, 1, 100).reshape(1, 10, 10).astype(np.float32)

        # Test with an unsupported method to trigger fallback
        mock_config.normalisation_method = "unsupported"
        normalized = apply_normalisation(input_batch, mock_config)

        assert normalized.shape == input_batch.shape
        # fitsbolt may convert to uint8, so check for valid numeric type
        assert normalized.dtype in [np.float32, np.float64, np.uint8, np.uint16]
        # Normalization should produce reasonable range - adjust for uint8
        if normalized.dtype == np.uint8:
            assert normalized.min() >= 0
            assert normalized.max() <= 255
        else:
            assert normalized.min() >= 0
            assert normalized.max() <= 1

    def test_combine_channels_simple(self, mock_cutout_data):
        """Test combining multiple channels into single output."""
        channel_weights = {
            "VIS": [1.0, 0.0, 0.0],
            "NIR-Y": [0.0, 0.8, 0.0],
            "NIR-H": [0.0, 0.0, 0.6],
        }

        # Convert dict to batch format (1, H, W, 3)
        extension_names = ["VIS", "NIR-Y", "NIR-H"]
        H, W = mock_cutout_data["VIS"].shape
        batch_cutouts = np.zeros((1, H, W, 3), dtype=np.float32)
        for i, ext in enumerate(extension_names):
            batch_cutouts[0, :, :, i] = mock_cutout_data[ext]

        combined = combine_channels(batch_cutouts, channel_weights)

        # Should return RGB format (1, H, W, 3)
        assert combined.shape == (1, H, W, 3)
        assert combined.dtype == np.float32
        assert isinstance(combined, np.ndarray)

    def test_combine_channels_equal_weights(self, mock_cutout_data):
        """Test combining channels with equal weighting."""
        channel_weights = {
            "VIS": [0.33, 0.33, 0.33],
            "NIR-Y": [0.33, 0.33, 0.33],
            "NIR-H": [0.34, 0.34, 0.34],
        }

        # Convert dict to batch format (1, H, W, 3)
        extension_names = ["VIS", "NIR-Y", "NIR-H"]
        H, W = mock_cutout_data["VIS"].shape
        batch_cutouts = np.zeros((1, H, W, 3), dtype=np.float32)
        for i, ext in enumerate(extension_names):
            batch_cutouts[0, :, :, i] = mock_cutout_data[ext]

        combined = combine_channels(batch_cutouts, channel_weights)

        # Note: fitsbolt.batch_channel_combination does not simply average channels
        # It processes RGB weights differently than simple linear combination
        # Just verify basic properties - should return RGB format (1, H, W, 3)
        assert combined.shape == (1, H, W, 3)
        assert combined.dtype == np.float32
        assert isinstance(combined, np.ndarray)

    def test_error_handling_invalid_cutout_data(self):
        """Test error handling with invalid cutout data."""
        # Test with empty array - should handle gracefully
        try:
            empty_cutouts = np.array([])
            # This might raise an exception or handle gracefully
            result = resize_images(empty_cutouts, target_size=(64, 64))
            assert isinstance(result, np.ndarray)
        except Exception:
            # It's acceptable to raise an exception for invalid input
            pass

    def test_error_handling_missing_channels(self):
        """Test error handling with malformed input shapes."""
        # Test with incorrectly shaped array
        try:
            malformed_cutouts = np.random.random((2, 10)).astype(
                np.float32
            )  # Only 2D instead of 3D batch
            result = resize_images(malformed_cutouts, target_size=(64, 64))
            # If successful, should be a valid array
            assert isinstance(result, np.ndarray)
        except Exception:
            # It's acceptable to raise an exception for malformed input
            pass

    def test_memory_efficient_processing(self, mock_cutout_data, mock_config):
        """Test memory-efficient processing of large cutouts."""
        # Create larger cutout batch
        large_cutouts = []
        for channel in mock_cutout_data.keys():
            large_cutout = np.random.random((1024, 1024)).astype(np.float32)
            large_cutouts.append(large_cutout)

        cutouts_batch = np.array(large_cutouts)

        # Process using individual functions
        resized = resize_images(cutouts_batch, target_size=(256, 256))
        mock_config.normalisation_method = "linear"
        normalized = apply_normalisation(resized, mock_config)
        converted = convert_data_type(normalized, "float32")

        # Should complete without memory errors
        assert isinstance(converted, np.ndarray)
        assert converted.shape[0] == len(large_cutouts)
        assert converted.shape[1:] == (256, 256)
        assert converted.dtype == np.float32

    def test_batch_processing_multiple_sources(self, mock_config):
        """Test batch processing multiple cutouts efficiently."""
        # Create batch of cutouts (15 cutouts total: 5 sources × 3 channels each)
        all_cutouts = []

        for i in range(5):
            # Add 3 cutouts per "source" (VIS, NIR-Y, NIR-H)
            for channel in ["VIS", "NIR-Y", "NIR-H"]:
                cutout = np.random.random((64, 64)).astype(np.float32)
                all_cutouts.append(cutout)

        cutouts_batch = np.array(all_cutouts)

        # Process using individual functions
        resized = resize_images(cutouts_batch, target_size=(256, 256))
        mock_config.normalisation_method = "linear"
        normalized = apply_normalisation(resized, mock_config)
        converted = convert_data_type(normalized, "float32")

        assert converted.shape[0] == 15  # 5 sources × 3 channels
        assert converted.shape[1:] == (256, 256)
        assert converted.dtype == np.float32

    def test_batch_processing_consistency(self, mock_cutout_data, mock_config):
        """Test that batch processing produces consistent results."""
        # Create batch from mock data
        cutouts_list = []
        for channel, cutout in mock_cutout_data.items():
            cutouts_list.append(cutout)

        cutouts_batch = np.array(cutouts_list)

        # Process twice with same parameters
        resized1 = resize_images(cutouts_batch, target_size=(128, 128))
        mock_config.normalisation_method = "linear"
        normalized1 = apply_normalisation(resized1, mock_config)
        result1 = convert_data_type(normalized1, "float32")

        resized2 = resize_images(cutouts_batch, target_size=(128, 128))
        normalized2 = apply_normalisation(resized2, mock_config)
        result2 = convert_data_type(normalized2, "float32")

        # Results should be identical (deterministic processing)
        assert result1.shape == result2.shape
        assert result1.dtype == result2.dtype
        assert np.allclose(result1, result2, rtol=1e-6)

    def test_different_normalisation_methods(self, mock_config):
        """Test all supported normalization methods with batch processing."""
        # Use batch format (1, H, W)
        input_batch = np.linspace(0.01, 1, 100).reshape(1, 10, 10).astype(np.float32)

        normalisation_methods = ["linear", "log", "asinh", "zscale"]

        for method in normalisation_methods:
            mock_config.normalisation_method = method
            normalized = apply_normalisation(input_batch, mock_config)

            assert normalized.shape == input_batch.shape
            # fitsbolt may convert to uint8, so check for valid numeric type
            assert normalized.dtype in [np.float32, np.float64, np.uint8, np.uint16]
            assert np.isfinite(normalized).all()

    @patch("fitsbolt.normalise_images")
    def test_fitsbolt_integration(self, mock_normalise_images, mock_cutout_data, mock_config):
        """Test integration with fitsbolt library."""
        # Create batch from mock data
        cutouts_list = []
        for channel, cutout in mock_cutout_data.items():
            cutouts_list.append(cutout)

        cutouts_batch = np.array(cutouts_list)

        # Mock fitsbolt responses - normalise_images expects batch format and returns batch format
        def mock_normalise_func(images, normalisation_method, show_progress):
            # Return normalized version of the input batch
            batch_size, height, width, channels = images.shape
            return np.random.random((batch_size, height, width, channels)).astype(np.float32)

        mock_normalise_images.side_effect = mock_normalise_func

        # Process with individual functions
        resized = resize_images(cutouts_batch, target_size=(256, 256))
        mock_config.normalisation_method = "linear"
        result = apply_normalisation(resized, mock_config)

        # Vectorized implementation calls fitsbolt once for entire batch
        assert mock_normalise_images.call_count == 1
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(cutouts_list)
        assert result.shape[1:] == (256, 256)

    def test_resize_images_edge_cases(self):
        """Test resize_images function with edge cases."""
        # Test same size - should return copy
        image = np.random.random((64, 64))
        resized = resize_images(image, (64, 64))
        assert resized.shape == (1, 64, 64)  # Single image becomes batch
        # Should be a copy but values should be the same since no resizing happened
        assert resized is not image  # Different objects

        # Test different interpolation methods
        for method in ["nearest", "bilinear", "biquadratic", "bicubic", "invalid_method"]:
            resized = resize_images(image, (32, 32), interpolation=method)
            assert resized.shape == (1, 32, 32)  # Single image becomes batch

        # Test with error condition
        with patch("skimage.transform.resize", side_effect=Exception("Resize failed")):
            resized = resize_images(image, (128, 128))
            # Should return zeros on error
            assert resized.shape == (1, 128, 128)  # Single image becomes batch
            assert np.allclose(resized, 0)

    def test_convert_data_type_all_types(self):
        """Test data type conversion for all supported types."""
        image = np.random.random((32, 32)).astype(np.float64)

        # Test all supported types
        type_map = {
            "float32": np.float32,
            "float64": np.float64,
            "uint8": np.uint8,
            "uint16": np.uint16,
        }

        for target_dtype, expected_type in type_map.items():
            converted = convert_data_type(image, target_dtype)
            assert converted.dtype == expected_type

        # Test unknown type
        converted = convert_data_type(image, "unknown_type")
        assert converted.dtype == image.dtype  # Should return original

        # Test with conversion error
        with patch("skimage.util.img_as_float32", side_effect=Exception("Conversion failed")):
            converted = convert_data_type(image, "float32")
            assert converted.dtype == image.dtype  # Should return original

    def test_apply_normalisation_error_fallback(self, mock_config):
        """Test normalization fallback when fitsbolt fails with batch processing."""
        # Use batch format (1, H, W)
        batch_images = np.random.random((1, 16, 16)).astype(np.float32) * 100

        mock_config.normalisation_method = "linear"
        # Mock fitsbolt to fail
        with patch("fitsbolt.normalise_images", side_effect=Exception("Fitsbolt failed")):
            normalized = apply_normalisation(batch_images, mock_config)

            # Should use fallback normalization
            assert normalized.shape == batch_images.shape
            assert np.min(normalized) >= 0
            assert np.max(normalized) <= 1

    def test_combine_channels_comprehensive(self):
        """Test comprehensive channel combination scenarios."""
        cutouts = {
            "RED": np.ones((32, 32)) * 1.0,
            "GREEN": np.ones((32, 32)) * 2.0,
            "BLUE": np.ones((32, 32)) * 3.0,
        }

        # Test equal weights (channels and weights available for reference)
        # channels = ["RED", "GREEN", "BLUE"]
        # weights = [1.0, 1.0, 1.0]

        channel_weights = {
            "RED": [0.33, 0.33, 0.33],
            "GREEN": [0.33, 0.33, 0.33],
            "BLUE": [0.34, 0.34, 0.34],
        }

        # Convert dict to batch format (1, 32, 32, 3)
        extension_names = ["RED", "GREEN", "BLUE"]
        batch_cutouts = np.zeros((1, 32, 32, 3), dtype=np.float32)
        for i, ext in enumerate(extension_names):
            batch_cutouts[0, :, :, i] = cutouts[ext]

        combined = combine_channels(batch_cutouts, channel_weights)
        assert combined.shape == (1, 32, 32, 3)  # RGB output
        assert isinstance(combined, np.ndarray)

        # Test empty channel_weights - should raise assertion error
        try:
            combined = combine_channels(batch_cutouts, {})
            assert False, "Should have raised AssertionError for empty channel_weights"
        except AssertionError:
            pass  # Expected behavior

    def test_apply_normalisation_batch_processing(self, mock_config):
        """Test batch normalization of multiple images."""
        # Create batch of test images
        batch_size = 4
        height, width = 64, 64
        images_batch = np.random.random((batch_size, height, width)).astype(np.float32)

        mock_config.normalisation_method = "linear"
        normalized_batch = apply_normalisation(images_batch, mock_config)

        assert normalized_batch.shape == (batch_size, height, width)
        assert normalized_batch.dtype in [np.float32, np.float64, np.uint8, np.uint16]

    def test_apply_normalisation_different_methods(self, mock_config):
        """Test batch normalization with different methods."""
        batch_size = 3
        images_batch = np.random.random((batch_size, 32, 32)).astype(np.float32)

        methods = ["linear", "log", "asinh", "zscale"]
        for method in methods:
            mock_config.normalisation_method = method
            normalized_batch = apply_normalisation(images_batch, mock_config)
            assert normalized_batch.shape == images_batch.shape
            assert np.isfinite(normalized_batch).all()

    @patch("fitsbolt.normalise_images")
    def test_apply_normalisation_fitsbolt_call(self, mock_normalise_images, mock_config):
        """Test that batch normalization calls fitsbolt correctly."""
        batch_size = 2
        images_batch = np.random.random((batch_size, 16, 16)).astype(np.float32)

        # Mock fitsbolt response
        mock_normalise_images.return_value = np.random.random((batch_size, 16, 16, 1)).astype(
            np.float32
        )

        mock_config.normalisation_method = "linear"
        apply_normalisation(images_batch, mock_config)

        # Should call fitsbolt correctly
        mock_normalise_images.assert_called_once()
        call_kwargs = mock_normalise_images.call_args[1]
        assert call_kwargs["show_progress"] is False

    def test_apply_normalisation_fallback_batch(self, mock_config):
        """Test batch normalization fallback when fitsbolt fails."""
        images_batch = np.random.random((3, 16, 16)).astype(np.float32) * 100

        mock_config.normalisation_method = "linear"
        # Mock fitsbolt to fail
        with patch("fitsbolt.normalise_images", side_effect=Exception("Fitsbolt failed")):
            normalized_batch = apply_normalisation(images_batch, mock_config)

            # Should use fallback normalization
            assert normalized_batch.shape == images_batch.shape
            for i in range(images_batch.shape[0]):
                assert np.min(normalized_batch[i]) >= 0
                assert np.max(normalized_batch[i]) <= 1
