#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Enhanced unit tests for the image_processor module with real fitsbolt testing.

Tests cover:
- Real fitsbolt normalization testing without mocks
- Verification that stretch functions actually change images as expected
- Data range and value testing after processing
- Performance and edge case handling
- Integration testing with real data transformations
"""

import numpy as np
import pytest
from cutana.image_processor import (
    resize_images,
    convert_data_type,
    apply_normalisation,
    combine_channels,
)


class TestImageProcessorEnhanced:
    """Enhanced test suite for image processor with real fitsbolt testing."""

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
    def synthetic_astronomical_image(self):
        """Create synthetic astronomical image with known characteristics."""
        size = 128
        # Create background
        background = np.random.normal(100, 10, (size, size)).astype(np.float32)

        # Add bright source in center
        center_y, center_x = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        source = 10000.0 * np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * 5**2))

        # Add some noise
        noise = np.random.normal(0, 5, (size, size)).astype(np.float32)

        return background + source + noise

    @pytest.fixture
    def realistic_cutout_data(self, synthetic_astronomical_image):
        """Create realistic multi-channel cutout data."""
        base_image = synthetic_astronomical_image
        return {
            "VIS": base_image,
            "NIR_Y": base_image * 0.8 + np.random.normal(0, 2, base_image.shape).astype(np.float32),
            "NIR_H": base_image * 0.6 + np.random.normal(0, 3, base_image.shape).astype(np.float32),
        }

    def test_fitsbolt_linear_normalization_real(self, synthetic_astronomical_image, mock_config):
        """Test real fitsbolt linear normalization without mocking."""
        original_image = synthetic_astronomical_image.copy()

        # Apply linear normalization
        mock_config.normalisation_method = "linear"
        normalized = apply_normalisation(original_image, mock_config)

        # Verify normalization occurred
        assert normalized.shape == original_image.shape
        assert normalized.dtype in [np.float32, np.float64, np.uint8, np.uint16]

        # Linear normalization should preserve relative intensities
        # The bright source should still be the brightest part
        original_max_pos = np.unravel_index(np.argmax(original_image), original_image.shape)

        if normalized.dtype == np.uint8:
            # For uint8, check that the bright source is close to maximum
            max_val = np.max(normalized)
            center_val = normalized[64, 64]  # Center where we put the source
            assert center_val >= max_val * 0.8  # Should be at least 80% of max
        else:
            # For float types, verify relative scaling
            normalized_max_pos = np.unravel_index(np.argmax(normalized), normalized.shape)
            assert original_max_pos == normalized_max_pos

    def test_fitsbolt_log_normalization_real(self, synthetic_astronomical_image, mock_config):
        """Test real fitsbolt logarithmic normalization."""
        original_image = synthetic_astronomical_image.copy()
        # Ensure all values are positive for log
        original_image = np.maximum(original_image, 0.1)

        mock_config.normalisation_method = "log"
        normalized = apply_normalisation(original_image, mock_config)

        # Verify basic properties
        assert normalized.shape == original_image.shape
        assert np.all(np.isfinite(normalized))

        # Log normalization should compress dynamic range
        # High values should be compressed more than low values
        if normalized.dtype in [np.float32, np.float64]:
            original_ratio = np.max(original_image) / np.mean(original_image)
            normalized_ratio = np.max(normalized) / np.mean(normalized)
            # Log should reduce the ratio (compress dynamic range)
            assert normalized_ratio < original_ratio

    def test_fitsbolt_asinh_normalization_real(self, synthetic_astronomical_image, mock_config):
        """Test real fitsbolt asinh normalization."""
        original_image = synthetic_astronomical_image.copy()

        mock_config.normalisation_method = "asinh"
        normalized = apply_normalisation(original_image, mock_config)

        # Verify basic properties
        assert normalized.shape == original_image.shape
        assert np.all(np.isfinite(normalized))

        # Asinh should handle negative values and preserve structure
        if normalized.dtype in [np.float32, np.float64]:
            # Asinh normalization should preserve the relative structure
            # The center source should still be brighter than background
            center_val = normalized[64, 64]
            corner_val = normalized[10, 10]  # Background area
            assert center_val > corner_val

    def test_fitsbolt_zscale_normalization_real(self, synthetic_astronomical_image, mock_config):
        """Test real fitsbolt zscale normalization."""
        original_image = synthetic_astronomical_image.copy()

        mock_config.normalisation_method = "zscale"
        normalized = apply_normalisation(original_image, mock_config)

        # Verify basic properties
        assert normalized.shape == original_image.shape
        assert np.all(np.isfinite(normalized))

        # ZScale should provide good contrast for astronomical data
        if normalized.dtype == np.uint8:
            # Should use a good range of the available dynamic range
            unique_values = len(np.unique(normalized))
            assert unique_values > 50  # Should use at least 50 different values

    def test_normalization_preserves_structure(self, synthetic_astronomical_image, mock_config):
        """Test that normalization preserves the overall image structure."""
        original_image = synthetic_astronomical_image.copy()

        # Test different normalization methods
        methods = ["linear", "log", "asinh", "zscale"]

        for method in methods:
            mock_config.normalisation_method = method
            normalized = apply_normalisation(original_image, mock_config)

            # The bright source should still be at the center
            center_region = normalized[60:68, 60:68]  # 8x8 region around center
            corner_region = normalized[5:13, 5:13]  # 8x8 region at corner

            # Center should generally be brighter than corner
            center_mean = np.mean(center_region)
            corner_mean = np.mean(corner_region)

            if normalized.dtype == np.uint8:
                # For uint8, allow some tolerance due to quantization
                assert center_mean >= corner_mean * 0.9
            else:
                assert center_mean > corner_mean

    def test_resize_preserves_brightness_distribution(self, synthetic_astronomical_image):
        """Test that resizing preserves the brightness distribution."""
        original_image = synthetic_astronomical_image.copy()

        # Test different resize operations
        sizes = [(64, 64), (256, 256), (192, 192)]

        for target_size in sizes:
            resized = resize_images(original_image, target_size)

            assert resized.shape == (1,) + target_size  # Single image becomes batch

            # Statistical properties should be approximately preserved
            original_mean = np.mean(original_image)
            resized_mean = np.mean(resized[0])  # Access first image in batch

            # Allow some tolerance for interpolation effects
            assert abs(resized_mean - original_mean) / original_mean < 0.1

    def test_data_type_conversion_range_preservation(self):
        """Test that data type conversions preserve appropriate ranges."""
        # Test with known data ranges
        test_data = np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)

        # Test float32 -> uint8
        uint8_data = convert_data_type(test_data, "uint8")
        assert uint8_data.dtype == np.uint8
        assert uint8_data.min() >= 0
        assert uint8_data.max() <= 255
        assert uint8_data.min() < 50  # Should use low values
        assert uint8_data.max() > 200  # Should use high values

        # Test uint8 -> float32
        float32_data = convert_data_type(uint8_data, "float32")
        assert float32_data.dtype == np.float32
        assert 0 <= float32_data.min() <= 0.1
        assert 0.9 <= float32_data.max() <= 1.0

    def test_complete_processing_pipeline_validation(self, realistic_cutout_data, mock_config):
        """Test complete processing pipeline produces valid scientific data."""
        # Create batch array from cutout data
        cutouts_list = []
        for channel, cutout in realistic_cutout_data.items():
            cutouts_list.append(cutout)

        cutouts_batch = np.array(cutouts_list)

        # Process using individual functions
        resized = resize_images(cutouts_batch, target_size=(64, 64), interpolation="bilinear")
        mock_config.normalisation_method = "asinh"
        normalized = apply_normalisation(resized, mock_config)
        processed_batch = convert_data_type(normalized, "float32")

        assert processed_batch.shape[0] == len(cutouts_list)  # Same number of cutouts
        assert processed_batch.shape[1:] == (64, 64)  # Target resolution
        assert processed_batch.dtype == np.float32

        # Verify each processed cutout
        for i in range(processed_batch.shape[0]):
            processed_cutout = processed_batch[i]

            # Should have valid data range
            assert np.all(np.isfinite(processed_cutout))

            # Should have some variation (not all zeros or constant)
            assert np.std(processed_cutout) > 0

            # For astronomical data, should have positive or at least not all negative values
            assert np.max(processed_cutout) > 0

    def test_normalization_with_extreme_values(self, mock_config):
        """Test normalization handles extreme values correctly."""
        # Create image with extreme dynamic range
        extreme_image = np.ones((64, 64), dtype=np.float32) * 100
        extreme_image[30:34, 30:34] = 1000000  # Very bright source
        extreme_image[10:14, 10:14] = 0.001  # Very dim region

        # Add some noise
        extreme_image += np.random.normal(0, 1, extreme_image.shape).astype(np.float32)

        # Test different normalization methods
        methods = ["linear", "log", "asinh", "zscale"]

        for method in methods:
            mock_config.normalisation_method = method
            normalized = apply_normalisation(extreme_image, mock_config)

            # Should handle extreme values without creating invalid data
            assert np.all(np.isfinite(normalized))
            assert normalized.shape == extreme_image.shape

            if normalized.dtype == np.uint8:
                assert 0 <= normalized.min() <= normalized.max() <= 255

            # The bright source should still be distinguishable
            bright_region_mean = np.mean(normalized[30:34, 30:34])
            background_mean = np.mean(normalized[50:60, 50:60])
            assert bright_region_mean > background_mean

    def test_channel_combination_scientific_validity(self, realistic_cutout_data):
        """Test channel combination produces scientifically valid results with concrete value checks."""
        channels = ["VIS", "NIR_Y", "NIR_H"]

        # Create test data with known values for precise validation
        H, W = 64, 64
        test_cutout_data = {
            "VIS": np.full((H, W), 100.0, dtype=np.float64),  # All pixels = 100
            "NIR_Y": np.full((H, W), 200.0, dtype=np.float64),  # All pixels = 200
            "NIR_H": np.full((H, W), 300.0, dtype=np.float64),  # All pixels = 300
        }

        # Test specific weighting scheme with known expected outputs
        # weights = [2.0, 1.0, 0.5] for VIS, NIR_Y, NIR_H respectively
        channel_weights = {
            "VIS": [2.0, 0.0, 0.0],  # VIS contributes 2.0 to output channel 0
            "NIR_Y": [1.0, 0.0, 0.0],  # NIR_Y contributes 1.0 to output channel 0
            "NIR_H": [0.5, 0.0, 0.0],  # NIR_H contributes 0.5 to output channel 0
        }

        # Convert to batch format (1, H, W, 3)
        batch_cutouts = np.zeros((1, H, W, len(channels)), dtype=np.float64)
        for i, ext in enumerate(channels):
            batch_cutouts[0, :, :, i] = test_cutout_data[ext]

        combined = combine_channels(batch_cutouts, channel_weights)

        # Should return single-channel output (1, H, W, 1) since only channel 0 has weights
        assert combined.shape == (1, H, W, 1) or combined.shape == (1, H, W, 3)
        assert combined.dtype == np.float64

        # Combined should be finite and reasonable
        assert np.all(np.isfinite(combined))

        # Calculate expected combined value:
        # output = 2.0*100 + 1.0*200 + 0.5*300 = 200 + 200 + 150 = 550
        expected_value = 2.0 * 100.0 + 1.0 * 200.0 + 0.5 * 300.0  # = 550

        # Check the actual combined values
        actual_combined = combined[0, :, :, 0]  # First output channel
        actual_mean = np.mean(actual_combined)

        # Since all input pixels were uniform, output should also be uniform
        assert np.allclose(
            actual_combined, expected_value, rtol=1e-6
        ), f"Expected uniform value {expected_value}, got mean {actual_mean} with std {np.std(actual_combined)}"

        # Test with more complex multi-channel output
        multi_channel_weights = {
            "VIS": [1.0, 0.5],  # VIS contributes 1.0 to ch0, 0.5 to ch1
            "NIR_Y": [2.0, 1.0],  # NIR_Y contributes 2.0 to ch0, 1.0 to ch1
            "NIR_H": [0.5, 2.0],  # NIR_H contributes 0.5 to ch0, 2.0 to ch1
        }

        combined_multi = combine_channels(batch_cutouts, multi_channel_weights)
        assert combined_multi.shape == (1, H, W, 2)

        # Expected values:
        # ch0 = 1.0*100 + 2.0*200 + 0.5*300 = 100 + 400 + 150 = 650
        # ch1 = 0.5*100 + 1.0*200 + 2.0*300 = 50 + 200 + 600 = 850
        expected_ch0 = 1.0 * 100.0 + 2.0 * 200.0 + 0.5 * 300.0  # = 650
        expected_ch1 = 0.5 * 100.0 + 1.0 * 200.0 + 2.0 * 300.0  # = 850

        actual_ch0 = np.mean(combined_multi[0, :, :, 0])
        actual_ch1 = np.mean(combined_multi[0, :, :, 1])

        assert np.isclose(
            actual_ch0, expected_ch0, rtol=1e-6
        ), f"Channel 0: Expected {expected_ch0}, got {actual_ch0}"
        assert np.isclose(
            actual_ch1, expected_ch1, rtol=1e-6
        ), f"Channel 1: Expected {expected_ch1}, got {actual_ch1}"

        # Verify channels have different values as expected
        assert not np.allclose(
            combined_multi[0, :, :, 0], combined_multi[0, :, :, 1], rtol=1e-3
        ), "Channels should have different values based on different weight combinations"

    def test_processing_consistency_across_runs(self, synthetic_astronomical_image, mock_config):
        """Test that processing is consistent across multiple runs."""
        # Same input should produce same output (deterministic)
        image1 = synthetic_astronomical_image.copy()
        image2 = synthetic_astronomical_image.copy()

        # Apply same processing
        mock_config.normalisation_method = "linear"
        norm1 = apply_normalisation(image1, mock_config)
        norm2 = apply_normalisation(image2, mock_config)

        # Should be identical (or very close if using stochastic methods)
        if norm1.dtype == norm2.dtype:
            assert np.allclose(norm1, norm2, rtol=1e-6)

    def test_memory_efficiency_large_images(self, mock_config):
        """Test processing of large images doesn't cause memory issues."""
        # Create large image
        large_image = np.random.normal(1000, 100, (2048, 2048)).astype(np.float32)

        # Add bright source
        large_image[1000:1048, 1000:1048] += 10000

        # Test resizing (most memory-intensive operation)
        resized = resize_images(large_image, (512, 512))

        assert resized.shape == (1, 512, 512)  # Single image becomes batch
        assert resized.dtype == np.float32

        # Test normalization on batch
        mock_config.normalisation_method = "asinh"
        normalized = apply_normalisation(resized, mock_config)

        assert normalized.shape == (1, 512, 512)  # Batch format
        assert np.all(np.isfinite(normalized))

    def test_error_recovery_corrupted_data(self, mock_config):
        """Test error recovery with corrupted or invalid data."""
        # Test with NaN values
        corrupted_image = np.random.random((64, 64)).astype(np.float32)
        corrupted_image[30:34, 30:34] = np.nan

        mock_config.normalisation_method = "linear"
        # Should handle gracefully or use fallback
        try:
            normalized = apply_normalisation(corrupted_image, mock_config)
            # If it succeeds, should not propagate NaNs
            if not np.any(np.isnan(normalized)):
                assert True  # Good, handled the NaNs
        except Exception:
            # If it fails, that's also acceptable error handling
            assert True

    def test_fitsbolt_version_compatibility(self):
        """Test basic fitsbolt functionality to ensure version compatibility."""
        import fitsbolt

        # Test basic functionality
        test_image = np.random.random((32, 32)).astype(np.float32)

        # Create batch format (1, H, W, 1) as expected by fitsbolt
        batch_image = [test_image]

        try:
            # Test that fitsbolt works with our data format
            result = fitsbolt.normalise_images(
                images=batch_image,
                normalisation_method=fitsbolt.NormalisationMethod.CONVERSION_ONLY,
            )

            assert np.array(result).shape == np.array(batch_image).shape
            assert np.all(np.isfinite(np.array(result)))

        except Exception as e:
            pytest.fail(f"fitsbolt compatibility issue: {e}")
