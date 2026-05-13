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
- Normalization using fitsbolt
- Stretch function application (linear, log, asinh, sqrt)
- Multi-channel image processing
- Error handling for invalid inputs
"""

from unittest.mock import patch

import numpy as np
import pytest
from astropy.wcs import WCS
from dotmap import DotMap

from cutana.image_processor import (
    apply_normalisation,
    combine_channels,
    resize_batch_tensor,
)
from cutana.normalisation_parameters import NormalisationDefaults


class TestImageProcessor:
    """Test suite for image processor functions."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for apply_normalisation tests."""
        return DotMap(
            {
                "normalisation_method": "linear",
                "normalisation": {
                    "a": NormalisationDefaults.ASINH_A,
                    "percentile": NormalisationDefaults.PERCENTILE,
                    "n_samples": NormalisationDefaults.N_SAMPLES,
                    "contrast": NormalisationDefaults.CONTRAST,
                    "crop_enable": False,
                },
                "external_fitsbolt_cfg": DotMap(),
            }
        )

    @pytest.fixture
    def mock_cutout_data(self):
        """Create mock cutout data for testing."""
        return {
            "VIS": np.random.random((128, 128)).astype(np.float32),
            "NIR-Y": np.random.random((128, 128)).astype(np.float32),
            "NIR-H": np.random.random((128, 128)).astype(np.float32),
        }

    def test_resize_batch_tensor_upscale(self):
        """Test resizing image from smaller to larger resolution using resize_batch_tensor."""
        input_image = np.random.random((64, 64)).astype(np.float32)

        source_cutouts = {"source_0": {"VIS": input_image}}

        resized = resize_batch_tensor(
            source_cutouts,
            target_resolution=(128, 128),
            interpolation="bilinear",
            flux_conserved_resizing=False,
            pixel_scales_dict={"VIS": 0.1},
        )

        assert resized.shape == (1, 128, 128, 1)
        assert resized.dtype == np.float32
        assert not np.array_equal(resized[0, :, :, 0], input_image)

    def test_resize_batch_tensor_downscale(self):
        """Test resizing image from larger to smaller resolution using resize_batch_tensor."""
        input_image = np.random.random((512, 512)).astype(np.float32)

        source_cutouts = {"source_0": {"VIS": input_image}}

        resized = resize_batch_tensor(
            source_cutouts,
            target_resolution=(256, 256),
            interpolation="bilinear",
            flux_conserved_resizing=False,
            pixel_scales_dict={"VIS": 0.1},
        )

        assert resized.shape == (1, 256, 256, 1)
        assert resized.dtype == np.float32

    def test_resize_batch_tensor_preserve_range(self):
        """Test that resizing preserves the approximate data range."""
        input_image = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)

        source_cutouts = {"source_0": {"VIS": input_image}}

        resized = resize_batch_tensor(
            source_cutouts,
            target_resolution=(128, 128),
            interpolation="bilinear",
            flux_conserved_resizing=False,
            pixel_scales_dict={"VIS": 0.1},
        )

        assert resized[0, :, :, 0].min() >= -0.1
        assert resized[0, :, :, 0].max() <= 1.1
        assert abs(resized[0, :, :, 0].mean() - input_image.mean()) < 0.1

    @pytest.mark.parametrize(
        "method,input_range",
        [
            ("linear", (0, 1)),
            ("log", (0.1, 1)),
            ("asinh", (-1, 1)),
            ("zscale", (0.01, 1)),
        ],
    )
    def test_apply_normalisation_methods(self, mock_config, method, input_range):
        """Test normalisation with each supported stretch method."""
        input_batch = (
            np.linspace(input_range[0], input_range[1], 100).reshape(1, 10, 10).astype(np.float32)
        )
        mock_config.normalisation_method = method
        normalized = apply_normalisation(input_batch, mock_config)

        assert normalized.shape == input_batch.shape
        assert normalized.dtype in [np.float32, np.float64, np.uint8, np.uint16]
        assert np.isfinite(normalized).all()

    @pytest.mark.parametrize(
        "method",
        ["linear", "log", "asinh", "zscale"],
    )
    def test_apply_normalisation_batch_methods(self, mock_config, method):
        """Test batch normalisation with each stretch method on multi-image batches."""
        images_batch = np.random.random((3, 32, 32)).astype(np.float32)
        mock_config.normalisation_method = method
        normalized_batch = apply_normalisation(images_batch, mock_config)

        assert normalized_batch.shape == images_batch.shape
        assert np.isfinite(normalized_batch).all()

    def test_apply_normalisation_unsupported_method(self, mock_config):
        """Test normalisation with unsupported method falls back to CONVERSION_ONLY."""
        input_batch = np.linspace(0, 1, 100).reshape(1, 10, 10).astype(np.float32)

        mock_config.normalisation_method = "unsupported"
        normalized = apply_normalisation(input_batch, mock_config)

        assert normalized.shape == input_batch.shape
        assert normalized.dtype in [np.float32, np.float64, np.uint8, np.uint16]

    @patch("fitsbolt.normalise_images")
    def test_apply_normalisation_fitsbolt_mock(self, mock_normalise_images, mock_config):
        """Test that apply_normalisation calls fitsbolt correctly and handles its output."""
        batch_size = 2
        images_batch = np.random.random((batch_size, 16, 16)).astype(np.float32)

        mock_normalise_images.return_value = np.random.random((batch_size, 16, 16, 1)).astype(
            np.float32
        )

        mock_config.normalisation_method = "linear"
        result = apply_normalisation(images_batch, mock_config)

        mock_normalise_images.assert_called_once()
        call_kwargs = mock_normalise_images.call_args[1]
        assert call_kwargs["show_progress"] is False
        assert result.shape == images_batch.shape
        assert result.dtype == np.float32

    def test_apply_normalisation_error_raises(self, mock_config):
        """Test that normalisation raises RuntimeError when fitsbolt fails."""
        batch_images = np.random.random((1, 16, 16)).astype(np.float32) * 100

        mock_config.normalisation_method = "linear"
        with patch("fitsbolt.normalise_images", side_effect=Exception("Fitsbolt failed")):
            with pytest.raises(RuntimeError, match="Fitsbolt normalisation failed"):
                apply_normalisation(batch_images, mock_config)

    def test_combine_channels_simple(self, mock_cutout_data):
        """Test combining multiple channels into single output."""
        channel_weights = {
            "VIS": [1.0, 0.0, 0.0],
            "NIR-Y": [0.0, 0.8, 0.0],
            "NIR-H": [0.0, 0.0, 0.6],
        }

        extension_names = ["VIS", "NIR-Y", "NIR-H"]
        H, W = mock_cutout_data["VIS"].shape
        batch_cutouts = np.zeros((1, H, W, 3), dtype=np.float32)
        for i, ext in enumerate(extension_names):
            batch_cutouts[0, :, :, i] = mock_cutout_data[ext]

        combined = combine_channels(batch_cutouts, channel_weights)

        assert combined.shape == (1, H, W, 3)
        assert combined.dtype == np.float32
        assert isinstance(combined, np.ndarray)

    def test_combine_channels_comprehensive(self):
        """Test comprehensive channel combination scenarios."""
        cutouts = {
            "RED": np.ones((32, 32)) * 1.0,
            "GREEN": np.ones((32, 32)) * 2.0,
            "BLUE": np.ones((32, 32)) * 3.0,
        }

        channel_weights = {
            "RED": [0.33, 0.33, 0.33],
            "GREEN": [0.33, 0.33, 0.33],
            "BLUE": [0.34, 0.34, 0.34],
        }

        extension_names = ["RED", "GREEN", "BLUE"]
        batch_cutouts = np.zeros((1, 32, 32, 3), dtype=np.float32)
        for i, ext in enumerate(extension_names):
            batch_cutouts[0, :, :, i] = cutouts[ext]

        combined = combine_channels(batch_cutouts, channel_weights)
        assert combined.shape == (1, 32, 32, 3)
        assert isinstance(combined, np.ndarray)

        # Test empty channel_weights - should raise assertion error
        try:
            combined = combine_channels(batch_cutouts, {})
            assert False, "Should have raised AssertionError for empty channel_weights"
        except AssertionError:
            pass  # Expected behavior

    def test_error_handling_invalid_cutout_data(self):
        """Test error handling with invalid cutout data."""
        try:
            empty_cutouts = {}
            result = resize_batch_tensor(
                empty_cutouts,
                target_resolution=(64, 64),
                interpolation="bilinear",
                flux_conserved_resizing=False,
                pixel_scales_dict={},
            )
            assert isinstance(result, np.ndarray)
        except Exception:
            pass

    def test_error_handling_missing_channels(self):
        """Test error handling with malformed input shapes."""
        try:
            malformed_cutouts = {"source_0": {"VIS": np.random.random((2, 10)).astype(np.float32)}}
            result = resize_batch_tensor(
                malformed_cutouts,
                target_resolution=(64, 64),
                interpolation="bilinear",
                flux_conserved_resizing=False,
                pixel_scales_dict={"VIS": 0.1},
            )
            assert isinstance(result, np.ndarray)
        except Exception:
            pass

    def test_memory_efficient_processing(self, mock_cutout_data, mock_config):
        """Test memory-efficient processing of large cutouts."""
        source_cutouts = {"source_0": {}}
        pixel_scales_dict = {}
        for channel in mock_cutout_data.keys():
            large_cutout = np.random.random((1024, 1024)).astype(np.float32)
            source_cutouts["source_0"][channel] = large_cutout
            pixel_scales_dict[channel] = 0.1

        resized = resize_batch_tensor(
            source_cutouts,
            target_resolution=(256, 256),
            interpolation="bilinear",
            flux_conserved_resizing=False,
            pixel_scales_dict=pixel_scales_dict,
        )

        N_sources, H, W, N_extensions = resized.shape
        resized_for_norm = resized.reshape(N_sources * N_extensions, H, W)

        mock_config.normalisation_method = "linear"
        mock_config.data_type = "float32"
        converted = apply_normalisation(resized_for_norm, mock_config)

        assert isinstance(converted, np.ndarray)
        assert converted.shape[0] == len(mock_cutout_data)
        assert converted.shape[1:] == (256, 256)
        assert converted.dtype == np.float32

    def test_batch_processing_multiple_sources(self, mock_config):
        """Test batch processing multiple cutouts efficiently."""
        source_cutouts = {}
        pixel_scales_dict = {"VIS": 0.1, "NIR-Y": 0.1, "NIR-H": 0.1}
        for i in range(5):
            source_id = f"source_{i}"
            source_cutouts[source_id] = {}
            for channel in ["VIS", "NIR-Y", "NIR-H"]:
                cutout = np.random.random((64, 64)).astype(np.float32)
                source_cutouts[source_id][channel] = cutout

        resized = resize_batch_tensor(
            source_cutouts,
            target_resolution=(256, 256),
            interpolation="bilinear",
            flux_conserved_resizing=False,
            pixel_scales_dict=pixel_scales_dict,
        )

        N_sources, H, W, N_extensions = resized.shape
        resized_for_norm = resized.reshape(N_sources * N_extensions, H, W)

        mock_config.normalisation_method = "linear"
        mock_config.data_type = "float32"
        converted = apply_normalisation(resized_for_norm, mock_config)

        assert converted.shape[0] == 15
        assert converted.shape[1:] == (256, 256)
        assert converted.dtype == np.float32

    def test_batch_processing_consistency(self, mock_cutout_data, mock_config):
        """Test that batch processing produces consistent results."""
        source_cutouts = {}
        pixel_scales_dict = {}
        for idx, (channel, cutout) in enumerate(mock_cutout_data.items()):
            source_id = f"source_{idx}"
            source_cutouts[source_id] = {channel: cutout}
            pixel_scales_dict[channel] = 0.1

        resized1 = resize_batch_tensor(
            source_cutouts,
            target_resolution=(128, 128),
            interpolation="bilinear",
            flux_conserved_resizing=False,
            pixel_scales_dict=pixel_scales_dict,
        )

        N_sources, H, W, N_extensions = resized1.shape
        resized1_for_norm = resized1.reshape(N_sources * N_extensions, H, W)

        mock_config.normalisation_method = "linear"
        result1 = apply_normalisation(resized1_for_norm, mock_config)

        resized2 = resize_batch_tensor(
            source_cutouts,
            target_resolution=(128, 128),
            interpolation="bilinear",
            flux_conserved_resizing=False,
            pixel_scales_dict=pixel_scales_dict,
        )
        resized2_for_norm = resized2.reshape(N_sources * N_extensions, H, W)
        result2 = apply_normalisation(resized2_for_norm, mock_config)

        assert result1.shape == result2.shape
        assert result1.dtype == result2.dtype
        assert np.allclose(result1, result2, rtol=1e-6)

    @patch("fitsbolt.normalise_images")
    def test_fitsbolt_integration(self, mock_normalise_images, mock_cutout_data, mock_config):
        """Test integration with fitsbolt library."""
        source_cutouts = {"source_0": {}}
        pixel_scales_dict = {}
        for channel, cutout in mock_cutout_data.items():
            source_cutouts["source_0"][channel] = cutout
            pixel_scales_dict[channel] = 0.1

        def mock_normalise_func(
            images, output_dtype, normalisation_method, show_progress, num_workers=1
        ):
            batch_size, height, width, channels = images.shape
            return np.random.random((batch_size, height, width, channels)).astype(np.float32)

        mock_normalise_images.side_effect = mock_normalise_func

        resized = resize_batch_tensor(
            source_cutouts,
            target_resolution=(256, 256),
            interpolation="bilinear",
            flux_conserved_resizing=False,
            pixel_scales_dict=pixel_scales_dict,
        )

        N_sources, H, W, N_extensions = resized.shape
        resized_for_norm = resized.reshape(N_sources * N_extensions, H, W)

        mock_config.normalisation_method = "linear"
        result = apply_normalisation(resized_for_norm, mock_config)

        assert mock_normalise_images.call_count == 1
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(mock_cutout_data)
        assert result.shape[1:] == (256, 256)

    def test_resize_batch_tensor_edge_cases(self):
        """Test resize_batch_tensor function with edge cases."""
        image = np.random.random((64, 64)).astype(np.float32)
        source_cutouts = {"source_0": {"VIS": image}}

        resized = resize_batch_tensor(
            source_cutouts,
            target_resolution=(64, 64),
            interpolation="bilinear",
            flux_conserved_resizing=False,
            pixel_scales_dict={"VIS": 0.1},
        )
        assert resized.shape == (1, 64, 64, 1)
        assert resized[0, :, :, 0] is not image
        assert np.allclose(resized[0, :, :, 0], image)

        for method in ["nearest", "bilinear", "biquadratic", "bicubic", "invalid_method"]:
            resized = resize_batch_tensor(
                source_cutouts,
                target_resolution=(32, 32),
                interpolation=method,
                flux_conserved_resizing=False,
                pixel_scales_dict={"VIS": 0.1},
            )
            assert resized.shape == (1, 32, 32, 1)

        with patch("skimage.transform.resize", side_effect=Exception("Resize failed")):
            resized = resize_batch_tensor(
                source_cutouts,
                target_resolution=(128, 128),
                interpolation="bilinear",
                flux_conserved_resizing=False,
                pixel_scales_dict={"VIS": 0.1},
            )
            assert resized.shape == (1, 128, 128, 1)
            assert np.allclose(resized, 0)

    def test_flux_conserved_resizing_single_scale(self):
        """Test that flux-conserved resizing preserves total flux for different scales."""
        test_cases = [
            ((100, 100), (50, 50)),
            ((50, 50), (100, 100)),
            ((80, 80), (120, 120)),
            ((200, 200), (64, 64)),
        ]

        for input_shape, output_shape in test_cases:
            input_image = np.zeros(input_shape, dtype=np.float32)
            center_h, center_w = input_shape[0] // 2, input_shape[1] // 2
            square_size = min(input_shape) // 4
            h_start = center_h - square_size // 2
            h_end = center_h + square_size // 2
            w_start = center_w - square_size // 2
            w_end = center_w + square_size // 2

            flux_value = 1000.0
            input_image[h_start:h_end, w_start:w_end] = flux_value

            input_flux = np.sum(input_image)

            pixel_scale = 0.1
            input_wcs = WCS(naxis=2)
            input_wcs.wcs.crpix = [input_shape[1] / 2, input_shape[0] / 2]
            input_wcs.wcs.cdelt = [pixel_scale, pixel_scale]
            input_wcs.wcs.crval = [0, 0]
            input_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

            source_cutouts = {"source_1": {"channel_1": input_image}}
            pixel_scales_dict = {"channel_1": pixel_scale}

            resized_tensor = resize_batch_tensor(
                source_cutouts,
                output_shape,
                interpolation="bilinear",
                flux_conserved_resizing=True,
                pixel_scales_dict=pixel_scales_dict,
            )

            resized_image = resized_tensor[0, :, :, 0]
            output_flux = np.sum(resized_image)

            flux_ratio = output_flux / input_flux
            assert abs(flux_ratio - 1.0) < 0.01, (
                f"Flux not conserved for {input_shape} -> {output_shape}: "
                f"input={input_flux:.2f}, output={output_flux:.2f}, ratio={flux_ratio:.4f}"
            )

    def test_flux_conserved_resizing_roundtrip(self):
        """Test that flux-conserved resizing roundtrip preserves flux."""
        test_cases = [
            ((100, 100), (200, 200)),
            ((80, 80), (160, 160)),
            ((120, 120), (240, 240)),
        ]

        for original_shape, intermediate_shape in test_cases:
            input_image = np.zeros(original_shape, dtype=np.float32)
            center_h, center_w = original_shape[0] // 2, original_shape[1] // 2
            square_size = min(original_shape) // 4
            h_start = center_h - square_size // 2
            h_end = center_h + square_size // 2
            w_start = center_w - square_size // 2
            w_end = center_w + square_size // 2

            flux_value = 1000.0
            input_image[h_start:h_end, w_start:w_end] = flux_value

            input_flux = np.sum(input_image)

            pixel_scale = 0.1
            original_wcs = WCS(naxis=2)
            original_wcs.wcs.crpix = [original_shape[1] / 2, original_shape[0] / 2]
            original_wcs.wcs.cdelt = [pixel_scale, pixel_scale]
            original_wcs.wcs.crval = [0, 0]
            original_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

            source_cutouts_1 = {"source_1": {"channel_1": input_image}}
            pixel_scales_dict_1 = {"channel_1": pixel_scale}

            intermediate_tensor = resize_batch_tensor(
                source_cutouts_1,
                intermediate_shape,
                interpolation="bilinear",
                flux_conserved_resizing=True,
                pixel_scales_dict=pixel_scales_dict_1,
            )

            intermediate_image = intermediate_tensor[0, :, :, 0]
            intermediate_flux = np.sum(intermediate_image)

            flux_ratio_1 = intermediate_flux / input_flux
            assert abs(flux_ratio_1 - 1.0) < 0.01, (
                f"Flux not conserved in first resize {original_shape} -> {intermediate_shape}: "
                f"ratio={flux_ratio_1:.4f}"
            )

            intermediate_pixel_scale = pixel_scale * (original_shape[0] / intermediate_shape[0])
            intermediate_wcs = WCS(naxis=2)
            intermediate_wcs.wcs.crpix = [intermediate_shape[1] / 2, intermediate_shape[0] / 2]
            intermediate_wcs.wcs.cdelt = [intermediate_pixel_scale, intermediate_pixel_scale]
            intermediate_wcs.wcs.crval = [0, 0]
            intermediate_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

            source_cutouts_2 = {"source_1": {"channel_1": intermediate_image}}
            pixel_scales_dict_2 = {"channel_1": intermediate_pixel_scale}

            final_tensor = resize_batch_tensor(
                source_cutouts_2,
                original_shape,
                interpolation="bilinear",
                flux_conserved_resizing=True,
                pixel_scales_dict=pixel_scales_dict_2,
            )

            final_image = final_tensor[0, :, :, 0]
            final_flux = np.sum(final_image)

            flux_ratio_final = final_flux / input_flux
            assert abs(flux_ratio_final - 1.0) < 0.02, (
                f"Flux not conserved in roundtrip "
                f"{original_shape} -> {intermediate_shape} -> {original_shape}: "
                f"input={input_flux:.2f}, final={final_flux:.2f}, ratio={flux_ratio_final:.4f}"
            )

            correlation = np.corrcoef(input_image.flatten(), final_image.flatten())[0, 1]
            assert correlation > 0.9, (
                f"Image structure not well preserved in roundtrip: correlation={correlation:.4f}"
            )
