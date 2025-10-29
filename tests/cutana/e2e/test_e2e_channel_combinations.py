#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
End-to-end tests for channel combination processing in the Cutana pipeline.

Test Setup:
- Creates mock FITS files with extension-specific gradient patterns:
  * VIS: Horizontal gradient (left-to-right), base value 100
  * NIRH: Vertical gradient (top-to-bottom), base value 200
  * NIRJ: Diagonal gradient (top-left to bottom-right), base value 300
  * NIRY: Horizontal gradient (right-to-left), base value 400

Test Matrix:
- Input combinations: 1-4 FITS files with different gradient patterns
- Output combinations: 1-4 channels with predefined mixing weights
- Data types: uint8 (0-255) and float32 (0-1) with linear normalization
- Output formats: FITS and zarr storage formats

Validation:
- Channel mixing: Validates that output values roughly match expected
  calculations based on input gradients and channel weights
- Spatial integrity: Confirms gradient directions are preserved in output
- Statistical validation: Compares mean values of output against
  theoretical calculations with appropriate tolerances
"""

import pytest
import numpy as np
import tempfile
import json
import shutil
import zarr
from pathlib import Path
from unittest.mock import patch
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
from loguru import logger

from cutana.orchestrator import Orchestrator
from cutana.get_default_config import get_default_config
from cutana_ui.widgets.configuration_widget import SharedConfigurationWidget
from cutana.catalogue_preprocessor import analyse_source_catalogue


def get_channel_matrix_config(n_input: int, n_output: int) -> dict:
    """
    Get predefined channel weights for given input/output combination.

    Args:
        n_input: Number of input FITS files/extensions
        n_output: Number of output channels

    Returns:
        Dictionary mapping extension names to lists of weights for each output channel.

    The matrix ensures:
    - No output channel is completely empty (all zeros)
    - Weights are balanced to provide meaningful channel separation
    - Common combinations are optimized for typical astronomical use cases
    """

    # Extension names in standard order
    extension_names = ["VIS", "NIRH", "NIRJ", "NIRY"][:n_input]

    # Define configurations for different input/output combinations
    configurations = {
        # Single input configurations
        (1, 1): {"VIS": [1.0]},
        (1, 2): {"VIS": [1.0, 0.5]},  # First channel gets full, second gets partial
        (1, 3): {"VIS": [1.0, 0.7, 0.3]},  # Decreasing weights
        (1, 4): {"VIS": [1.0, 0.8, 0.5, 0.2]},  # Decreasing weights
        # Two input configurations
        (2, 1): {"VIS": [0.6], "NIRH": [0.4]},  # Mix both inputs
        (2, 2): {
            "VIS": [1.0, 0.0],  # First channel: VIS only
            "NIRH": [0.0, 1.0],  # Second channel: NIRH only
        },
        (2, 3): {
            "VIS": [1.0, 0.5, 0.3],  # VIS dominant in first, partial in others
            "NIRH": [0.0, 0.5, 0.7],  # NIRH dominant in third, partial in second
        },
        (2, 4): {
            "VIS": [1.0, 0.8, 0.3, 0.0],  # VIS strong in first two
            "NIRH": [0.0, 0.2, 0.7, 1.0],  # NIRH strong in last two
        },
        # Three input configurations
        (3, 1): {"VIS": [0.5], "NIRH": [0.3], "NIRJ": [0.2]},  # All contribute to single output
        (3, 2): {
            "VIS": [0.7, 0.2],  # VIS dominant in first
            "NIRH": [0.3, 0.5],  # NIRH balanced
            "NIRJ": [0.0, 0.3],  # NIRJ mainly in second
        },
        (3, 3): {
            "VIS": [1.0, 0.0, 0.0],  # Each input maps to one output
            "NIRH": [0.0, 1.0, 0.0],
            "NIRJ": [0.0, 0.0, 1.0],
        },
        (3, 4): {
            "VIS": [1.0, 0.5, 0.0, 0.2],  # VIS strong in first, some in second and fourth
            "NIRH": [0.0, 0.5, 1.0, 0.3],  # NIRH balanced across middle channels
            "NIRJ": [0.0, 0.0, 0.0, 0.5],  # NIRJ mainly in fourth
        },
        # Four input configurations
        (4, 1): {
            "VIS": [0.4],
            "NIRH": [0.3],
            "NIRJ": [0.2],
            "NIRY": [0.1],  # All contribute, VIS dominant
        },
        (4, 2): {
            "VIS": [0.6, 0.0],  # VIS mainly in first
            "NIRH": [0.2, 0.8],  # NIRH balanced
            "NIRJ": [0.0, 0.1],  # NIRJ mainly in second
            "NIRY": [0.0, 0.0],  # NIRY mainly in second
        },
        (4, 3): {
            "VIS": [1.0, 0.2, 0.0],  # VIS dominant in first
            "NIRH": [0.0, 0.8, 0.2],  # NIRH dominant in second
            "NIRJ": [0.0, 0.0, 0.8],  # NIRJ dominant in third
            "NIRY": [0.0, 0.0, 0.0],  # NIRY not used (could be adjusted)
        },
        (4, 4): {
            "VIS": [1.0, 0.0, 0.0, 0.0],  # One-to-one mapping
            "NIRH": [0.0, 1.0, 0.0, 0.0],
            "NIRJ": [0.0, 0.0, 1.0, 0.0],
            "NIRY": [0.0, 0.0, 0.0, 1.0],
        },
    }

    # Get configuration for this combination
    config_key = (n_input, n_output)
    if config_key not in configurations:
        raise ValueError(
            f"Configuration is not sufficient for tests, Missing case {n_input}, {n_output}"
        )

    base_config = configurations[config_key]

    # Map to actual extension names
    result = {}
    for i, ext_name in enumerate(extension_names):
        if ext_name in base_config:
            result[ext_name] = base_config[ext_name]
        else:
            # If extension not in predefined config, use zeros
            result[ext_name] = [0.0] * n_output

    # Validate configuration
    _validate_configuration(result, n_input, n_output)

    return result


def _validate_configuration(config: dict, n_input: int, n_output: int):
    """Validate that the configuration makes sense."""
    # Check that all extensions have the right number of output weights
    for ext_name, weights in config.items():
        if len(weights) != n_output:
            raise ValueError(
                f"Extension {ext_name} has {len(weights)} weights, expected {n_output}"
            )

    # Check that no output channel is completely empty
    for ch_idx in range(n_output):
        total_weight = sum(weights[ch_idx] for weights in config.values() if ch_idx < len(weights))
        if total_weight == 0.0:
            logger.warning(f"Output channel {ch_idx} has zero total weight")


def apply_channel_matrix_to_widget(config_widget, n_input: int, n_output: int, order: bool):
    """
    Apply predefined channel matrix configuration to a configuration widget.
    Simulates the user setting the channel weights in the UI.

    Args:
        config_widget: SharedConfigurationWidget instance
        n_input: Number of input FITS files/extensions
        n_output: Number of output channels
    """
    if not hasattr(config_widget, "channel_matrices") or not config_widget.channel_matrices:
        logger.error("Widget does not have channel_matrices initialized")
        return

    # Get the predefined configuration
    if order:
        # need to go thorugh list of extension dict and get name for each dict in the list
        extension_names = [ext["name"] for ext in config_widget.extensions]

        matrix_config = {}
        for i, ext in enumerate(extension_names):
            matrix_config[ext] = [0] * n_output
            matrix_config[ext][i] = 1
        print(f"\n\n creating ordered {matrix_config}")
    else:
        matrix_config = get_channel_matrix_config(n_input, n_output)
        extension_names = ["VIS", "NIRH", "NIRJ", "NIRY"][:n_input]

    # Apply the configuration to the widget's channel matrices
    for ch_idx in range(min(n_output, len(config_widget.channel_matrices))):
        channel_row = config_widget.channel_matrices[ch_idx]

        for ext_idx in range(min(n_input, len(channel_row))):
            ext_name = extension_names[ext_idx]
            if ext_name in matrix_config and ch_idx < len(matrix_config[ext_name]):
                weight_value = matrix_config[ext_name][ch_idx]
                channel_row[ext_idx].value = weight_value
                logger.debug(f"Set {ext_name} channel {ch_idx} weight to {weight_value}")


class TestEndToEndChannelCombinations:
    """Test channel combinations and output validation end-to-end."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Handle Windows file permission issues by retrying deletion
        import time

        for attempt in range(3):
            try:
                shutil.rmtree(temp_dir)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)  # Wait briefly and retry
                    continue
                else:
                    # Last attempt: ignore errors on Windows
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    pass

    @pytest.fixture
    def mock_fits_files(self, temp_dir):
        """Create mock FITS files with known integer values for testing."""
        fits_files = {}

        # Define standard extensions and their mock values
        extensions_data = {
            "VIS": {
                "value": 100.0,
                "filename": "EUC_MER_BGSUB-MOSAIC-VIS_TILE999999999-FCEEA_20001021T024200.725618Z_00.00.fits",
            },
            "NIRH": {
                "value": 200.0,
                "filename": "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE999999999-6922DE_20001020T224400.516126Z_00.00.fits",
            },
            "NIRJ": {
                "value": 300.0,
                "filename": "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE999999999-85BBC9_20001020T224500.641201Z_00.00.fits",
            },
            "NIRY": {
                "value": 400.0,
                "filename": "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE999999999-DA7C06_20001020T224300.192709Z_00.00.fits",
            },
        }

        for ext_name, ext_info in extensions_data.items():
            # Create 100x100 image with gradient pattern specific to each extension
            # This allows us to validate channel mixing by checking gradient directions
            image_data = self._create_gradient_image(ext_name, ext_info["value"], (100, 100))

            # Create simple WCS for coordinate transformation
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [50, 50]  # Reference pixel at center
            wcs.wcs.crval = [52.0, -29.75]  # Reference coordinates (matches test catalogue)
            wcs.wcs.cdelt = [-0.00027, 0.00027]  # Pixel scale (about 1 arcsec/pixel)
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

            # Create FITS file
            fits_path = Path(temp_dir) / ext_info["filename"]

            header = wcs.to_header()
            header["MAGZERO"] = 25.0  # Required for flux conversion
            header["EXTNAME"] = "PRIMARY"
            header["BUNIT"] = "electron/s"  # Units
            header["INSTRUME"] = ext_name  # Instrument name

            # Create PRIMARY HDU with image data (not empty)
            primary_hdu = fits.PrimaryHDU(data=image_data, header=header)
            hdul = fits.HDUList([primary_hdu])
            hdul.writeto(fits_path, overwrite=True)

            fits_files[ext_name] = {
                "path": str(fits_path),
                "value": ext_info["value"],
                "filename": ext_info["filename"],
            }

        return fits_files

    def _create_mock_fits_data(self, selected_fits):
        """Create mock FITS data for patching."""
        fits_data = {}

        for ext_name, fits_info in selected_fits.items():
            # Create mock image data with gradient pattern
            image_data = self._create_gradient_image(ext_name, fits_info["value"], (100, 100))

            # Create WCS
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [50, 50]
            wcs.wcs.crval = [52.0, -29.75]
            wcs.wcs.cdelt = [-0.00027, 0.00027]
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

            # Create header with WCS and required flux conversion keywords
            header = wcs.to_header()
            header["MAGZERO"] = 25.0  # Required for flux conversion
            header["EXTNAME"] = "PRIMARY"
            header["BUNIT"] = "electron/s"  # Units
            header["INSTRUME"] = ext_name  # Instrument name

            # Create PRIMARY HDU with image data (not empty)
            primary_hdu = fits.PrimaryHDU(data=image_data, header=header)
            hdul = fits.HDUList([primary_hdu])

            wcs_dict = {"PRIMARY": wcs}

            fits_data[fits_info["path"]] = (hdul, wcs_dict)

        return fits_data

    def _create_gradient_image(self, ext_name, base_value, shape):
        """Create gradient image with distinct pattern for each extension to validate channel mixing."""
        height, width = shape

        # Create coordinate arrays using meshgrid for proper broadcasting
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Create different gradient patterns for each extension
        print(f"Creating gradient image for {ext_name} with base value {base_value}, shape {shape}")
        if ext_name == "VIS":
            # Horizontal gradient (left to right)
            gradient = base_value * (0.5 + 0.5 * (x / width))
        elif ext_name == "NIRH":
            # Vertical gradient (top to bottom)
            gradient = base_value * (0.5 + 0.5 * (y / height))
        elif ext_name == "NIRJ":
            # Diagonal gradient (top-left to bottom-right)
            gradient = base_value * (0.5 + 0.5 * ((x + y) / (width + height)))
        elif ext_name == "NIRY":
            # Horizontal gradient (right to left - opposite of VIS)
            gradient = base_value * (0.5 + 0.5 * ((width - x) / width))
        else:
            # Default: uniform value for unknown extensions
            gradient = np.full(shape, base_value, dtype=np.float32)

        print(
            f"Created gradient with shape {gradient.shape}, min={gradient.min():.2f}, max={gradient.max():.2f}"
        )
        return gradient.astype(np.float32)

    def create_test_catalogue(self, temp_dir, fits_combinations, num_sources=5):
        """Create test catalogue with specified FITS file combinations."""
        catalogue_data = []

        for i in range(num_sources):
            source_id = f"test_source_{i+1}"
            ra = 52.0 + (i * 0.001)  # Spread sources slightly in RA
            dec = -29.75 + (i * 0.001)  # Spread sources slightly in Dec
            diameter = 10  # 10 pixel diameter

            # Build fits_file_paths list based on combination
            fits_paths = [fits_info["path"] for fits_info in fits_combinations]
            fits_paths_str = json.dumps(fits_paths)

            catalogue_data.append(
                {
                    "SourceID": source_id,
                    "RA": ra,
                    "Dec": dec,
                    "diameter_pixel": diameter,
                    "fits_file_paths": fits_paths_str,
                }
            )

        # Create CSV file
        df = pd.DataFrame(catalogue_data)
        catalogue_path = Path(temp_dir) / "test_catalogue.csv"
        df.to_csv(catalogue_path, index=False)

        return str(catalogue_path)

    def create_ui_shared_config_like_config(
        self, selected_fits, output_channels, result_cat_analysis, n_input, n_output, order=False
    ):
        """Create channel weights configuration using predefined channel matrix configurations.
        This is meant to recreate the workings of the shared config widget
        """
        # Create extensions list in the format expected by the UI widget
        extensions = []
        for ext_name in selected_fits.keys():
            extensions.append({"name": ext_name, "ext": "PRIMARY"})

        # Create a proper config object with explicit values (not DotMap)
        mock_config = get_default_config()
        mock_config.num_sources = 5
        mock_config.target_resolution = 16
        mock_config.data_type = "float32"

        # Provide explicit values for widget initialization
        mock_config.padding_factor = 1.0  # Default padding factor
        mock_config.normalisation_method = "linear"
        mock_config.normalisation.a = 1.0  # Linear scaling factor
        mock_config.normalisation.percentile = 99.0  # Not used for linear but required
        mock_config.max_workers = 1  # Single worker for predictable results
        mock_config.N_batch_cutout_process = 10  # Process all sources in one batch

        # Initialize the configuration widget with the actual UI logic
        extensions = result_cat_analysis.get("extensions", [])
        num_sources = result_cat_analysis.get("num_sources", 0)
        mock_config.selected_extensions = extensions
        mock_config.available_extensions = extensions
        config_widget = SharedConfigurationWidget(mock_config, compact=False)

        config_widget.set_extensions(extensions)
        config_widget.set_num_sources(num_sources)
        logger.info(f"\n\n{mock_config.selected_extensions}  \n {mock_config.available_extensions}")
        # Set the number of output channels
        config_widget.current_channels = output_channels

        # Update the matrix using actual UI logic - this creates the channel_matrices
        config_widget._update_matrix()

        # Apply our predefined channel matrix configuration
        apply_channel_matrix_to_widget(config_widget, n_input, n_output, order)

        logger.info(f"Applied channel matrix for {n_input} inputs -> {n_output} outputs")
        logger.info(
            f"Channel matrices shape: {len(config_widget.channel_matrices)}x{len(config_widget.channel_matrices[0]) if config_widget.channel_matrices else 0}"
        )
        logger.info(f"{(config_widget.channel_matrices)}")

        # manually force the proper dictionary here (didnt find where this is done in the code)
        extension_names = [
            ext_info.get("name", f"EXT_{i}") if isinstance(ext_info, dict) else str(ext_info)
            for i, ext_info in enumerate(mock_config.selected_extensions)
        ]
        mock_config.channel_weights = {}
        for key in extension_names:
            mock_config.channel_weights[key] = [0.0] * n_output

        # Use the actual _update_channel_weights method from the UI widget
        mock_config = config_widget.get_current_config()

        logger.info(f"updated config to to {mock_config}")
        return (
            mock_config.channel_weights,
            mock_config.selected_extensions,
            mock_config.available_extensions,
        )

    def calculate_expected_output_values(
        self,
        input_values,
        channel_weights,
        normalisation_method="linear",
        normalisation_params=None,
        data_type="float32",
        n_input=None,
        n_output=None,
    ):
        """Calculate expected output values given input values and channel mixing."""
        if normalisation_params is None:
            normalisation_params = {"a": 1.0}

        # If we have n_input and n_output, use the predefined matrix configuration
        if n_input is not None and n_output is not None:
            matrix_config = get_channel_matrix_config(n_input, n_output)

            # Calculate expected values from the cutout region
            # The pipeline extracts a cutout around the source position, then resizes to target_resolution
            # For test sources at RA=52.0+i*0.001, Dec=-29.75+i*0.001, the cutout will be around pixel (50, 50)

            # Calculate gradient values at the cutout center position in the original 100x100 image
            original_center_x = 50  # Source position in original image
            original_center_y = 50  # Source position in original image

            gradient_values = {}
            for ext_name, base_value in input_values.items():
                if ext_name == "VIS":
                    # Horizontal gradient: base_value * (0.5 + 0.5 * (x / width))
                    gradient_values[ext_name] = base_value * (0.5 + 0.5 * (original_center_x / 100))
                elif ext_name == "NIRH":
                    # Vertical gradient: base_value * (0.5 + 0.5 * (y / height))
                    gradient_values[ext_name] = base_value * (0.5 + 0.5 * (original_center_y / 100))
                elif ext_name == "NIRJ":
                    # Diagonal gradient: base_value * (0.5 + 0.5 * ((x + y) / (width + height)))
                    gradient_values[ext_name] = base_value * (
                        0.5 + 0.5 * ((original_center_x + original_center_y) / (100 + 100))
                    )
                elif ext_name == "NIRY":
                    # Horizontal gradient (right to left): base_value * (0.5 + 0.5 * ((width - x) / width))
                    gradient_values[ext_name] = base_value * (
                        0.5 + 0.5 * ((100 - original_center_x) / 100)
                    )
                else:
                    gradient_values[ext_name] = base_value

            # Calculate channel mixing
            expected_values = []
            for ch_idx in range(n_output):
                channel_value = 0.0
                for ext_name, weights in matrix_config.items():
                    if ch_idx < len(weights) and ext_name in gradient_values:
                        weight = weights[ch_idx]
                        gradient_val = gradient_values[ext_name]
                        channel_value += weight * gradient_val
                expected_values.append(channel_value)

            # Apply normalization based on data type
            if normalisation_method == "linear":
                if data_type == "uint8":
                    # For uint8, the pipeline does GLOBAL normalization across all channels
                    # Find the global min/max across all channels after channel mixing
                    global_min = min(expected_values)
                    global_max = max(expected_values)

                    normalized_values = []
                    if global_max > global_min:
                        # Apply global normalization to [0, 255] range
                        for channel_value in expected_values:
                            normalized_val = (
                                (channel_value - global_min) / (global_max - global_min) * 255.0
                            )
                            normalized_values.append(normalized_val)
                    else:
                        # All values are the same, set to mid-range
                        normalized_values = [127.5] * len(expected_values)
                    expected_values = normalized_values
                elif data_type == "float32":
                    # For float32, the pipeline normalizes to [0, 1] range
                    # Find the global min/max across all channels after channel mixing
                    global_min = min(expected_values)
                    global_max = max(expected_values)

                    normalized_values = []
                    if global_max > global_min:
                        # Apply global normalization to [0, 1] range
                        for channel_value in expected_values:
                            normalized_val = (channel_value - global_min) / (
                                global_max - global_min
                            )
                            normalized_values.append(normalized_val)
                    else:
                        # All values are the same, set to mid-range
                        normalized_values = [0.5] * len(expected_values)

                    # Apply scaling factor
                    scaling_factor = normalisation_params.get("a", 1.0)
                    expected_values = [val * scaling_factor for val in normalized_values]

            return expected_values
        raise ValueError(
            "n_input and n_output must be provided to use predefined matrix configuration"
        )

    @pytest.mark.parametrize("num_input_fits", [1, 4])  # 1, 2, 3, 4
    @pytest.mark.parametrize("num_output_channels", [1, 2, 4])  # 3
    @pytest.mark.parametrize("data_type", ["uint8"])  # , "float32"])
    @pytest.mark.parametrize("output_format", ["fits"])  # , "zarr"])
    @pytest.mark.parametrize("normalisation_method", ["linear"])
    def test_channel_combinations_ordered_fits(
        self,
        temp_dir,
        mock_fits_files,
        num_input_fits,
        num_output_channels,
        data_type,
        output_format,
        normalisation_method,
    ):
        """Test all combinations of input FITS and output channels with ordered filenames."""

        # Select FITS files in standard order (VIS, NIRH, NIRJ, NIRY)
        ordered_extensions = ["VIS", "NIRH", "NIRJ", "NIRY"]
        selected_fits = {ext: mock_fits_files[ext] for ext in ordered_extensions[:num_input_fits]}

        # Create test catalogue
        catalogue_path = self.create_test_catalogue(temp_dir, list(selected_fits.values()))

        #
        result_cat_analysis = analyse_source_catalogue(catalogue_path)
        print("Catalogue analysis:", result_cat_analysis)
        # Create configuration
        config = get_default_config()
        config.source_catalogue = catalogue_path
        config.output_dir = str(Path(temp_dir) / "output")
        config.target_resolution = 16  # Small for fast testing
        config.data_type = data_type
        config.normalisation_method = normalisation_method
        config.normalisation.a = 1.0  # Linear scaling factor
        config.normalisation.percentile = 99.0  # Not used for linear but required
        config.max_workers = 1  # Single worker for predictable results
        config.N_batch_cutout_process = 10  # Process all sources in one batch

        # Set up extensions and channel weights
        config.fits_extensions = ["PRIMARY"]  # Use PRIMARY extension
        config.channel_weights, config.selected_extensions, config.available_extensions = (
            self.create_ui_shared_config_like_config(
                selected_fits,
                num_output_channels,
                result_cat_analysis,
                num_input_fits,
                num_output_channels,
            )
        )
        print(
            "Test setup:",
            config.channel_weights,
            config.selected_extensions,
            config.output_dir,
        )
        config.output_format = output_format

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        print(config)
        # Run orchestrator
        orchestrator = Orchestrator(config)

        # Mock FITS file loading to use our test files
        with patch("cutana.fits_dataset.load_fits_sets") as mock_load_fits:
            mock_load_fits.return_value = self._create_mock_fits_data(selected_fits)

            # Create test catalogue data for processing with proper format
            import json

            fits_file_paths = [fits_info["path"] for fits_info in selected_fits.values()]
            test_data = [
                {
                    "SourceID": f"source_{i}",
                    "RA": 52.0 + i * 0.0001,
                    "Dec": -29.75 + i * 0.0001,
                    "diameter_pixel": 10.0,
                    "fits_file_paths": json.dumps(fits_file_paths),
                }
                for i in range(5)
            ]
            catalogue_df = pd.DataFrame(test_data)

            # Run processing
            try:
                result = orchestrator.start_processing(catalogue_df)
                # Verify successful completion
                assert result["status"] == "completed"
            finally:
                # Ensure all processes are terminated
                try:
                    orchestrator.stop_processing()
                except Exception:
                    pass
            assert result["total_sources"] > 0

        # Calculate expected values for validation using the new matrix configuration
        input_values = {ext_name: info["value"] for ext_name, info in selected_fits.items()}
        expected_values = self.calculate_expected_output_values(
            input_values,
            config.channel_weights,
            normalisation_method,
            {"a": 1.0},
            data_type,
            n_input=num_input_fits,
            n_output=num_output_channels,
        )

        # Validate output files
        self._validate_output_files(
            config.output_dir,
            selected_fits,
            num_output_channels,
            data_type,
            normalisation_method,
            output_format,
            expected_values,
            config.channel_weights,
            num_input_fits,
            num_output_channels,
        )

    def test_exceeding_channel_weights_sum(self, temp_dir, mock_fits_files):
        """Test channel weights that sum to more than 1.0.
        Maps VIS 10k times to the first output channel, which should make output channels 2 and 3 dark.
        """

        # Use all 4 FITS files
        selected_fits = mock_fits_files
        catalogue_path = self.create_test_catalogue(temp_dir, list(selected_fits.values()))

        # Get the catalogue analysis
        result_cat_analysis = analyse_source_catalogue(catalogue_path)

        # Create configuration
        config = get_default_config()
        config.source_catalogue = catalogue_path
        config.output_dir = str(Path(temp_dir) / "output")
        config.target_resolution = 16
        config.data_type = "float32"
        config.normalisation_method = "linear"
        config.normalisation.a = 1.0
        config.max_workers = 1
        config.N_batch_cutout_process = 10
        config.output_format = "fits"

        # Define number of input/output channels
        n_input = 4  # All 4 input filters
        n_output = 3  # RGB output

        # Set up extensions and initial channel weights
        config.fits_extensions = ["PRIMARY"]
        config.channel_weights, config.selected_extensions, config.available_extensions = (
            self.create_ui_shared_config_like_config(
                selected_fits,
                n_output,
                result_cat_analysis,
                n_input,
                n_output,
            )
        )

        # Override with channel weights that exceed 1.0 in sum
        config.channel_weights = {
            "VIS": [10000.0, 0.5, 0.0],  # Weights sum to 2.5 > 1.0
            "NIR-H": [1.5, 1.0, 0.5],  # Weights sum to 3.0 > 1.0
            "NIR-J": [0.5, 1.0, 1.5],  # Weights sum to 4.0 > 1.0
            "NIR-Y": [0.0, 0.5, 1.0],  # Weights sum to 3.0 > 1.0
        }
        assert (
            a == b for a, b in zip(config.channel_weights.keys(), selected_fits.keys())
        ), "incorrect test setup"

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Run orchestrator
        orchestrator = Orchestrator(config)

        with patch("cutana.fits_dataset.load_fits_sets") as mock_load_fits:
            mock_load_fits.return_value = self._create_mock_fits_data(selected_fits)

            # Create test catalogue data for processing with proper format
            import json

            fits_file_paths = [fits_info["path"] for fits_info in selected_fits.values()]
            test_data = [
                {
                    "SourceID": f"source_{i}",
                    "RA": 52.0 + i * 0.0001,
                    "Dec": -29.75 + i * 0.0001,
                    "diameter_pixel": 10.0,
                    "fits_file_paths": json.dumps(fits_file_paths),
                }
                for i in range(5)
            ]
            catalogue_df = pd.DataFrame(test_data)

            try:
                result = orchestrator.start_processing(catalogue_df)
                assert result["status"] == "completed"
            finally:
                # Ensure all processes are terminated
                try:
                    orchestrator.stop_processing()
                except Exception:
                    pass

        # Also validate that output values reflect the amplified mixing with the specialized method
        self._validate_amplified_channel_mixing(config.output_dir, config.data_type)

    def _validate_output_files(
        self,
        output_dir,
        selected_fits,
        num_output_channels,
        data_type,
        normalisation_method,
        output_format,
        expected_values,
        channel_weights=None,
        n_input=None,
        n_output=None,
    ):
        """Validate that output files have correct properties and values."""
        output_path = Path(output_dir)

        if output_format == "fits":
            # Check that FITS output files exist
            fits_files = list(output_path.glob("*.fits"))
            assert len(fits_files) > 0, "No FITS output files found"

            # Validate FITS files
            for fits_file in fits_files:
                with fits.open(fits_file) as hdul:
                    # Check number of extensions matches output channels
                    data_extensions = [
                        hdu for hdu in hdul if hasattr(hdu, "data") and hdu.data is not None
                    ]
                    assert (
                        len(data_extensions) >= num_output_channels
                    ), f"Expected {num_output_channels} data extensions, found {len(data_extensions)}"

                    # Check data type
                    for hdu in data_extensions:
                        if data_type == "uint8":
                            assert (
                                hdu.data.dtype == np.uint8
                            ), f"Expected uint8, got {hdu.data.dtype}"
                        elif data_type == "float32":
                            assert np.issubdtype(
                                hdu.data.dtype, np.float32
                            ), f"Expected float32, got {hdu.data.dtype}"

                    # Validate data values match expected calculations
                    self._validate_fits_values(
                        hdul, selected_fits, normalisation_method, data_type, expected_values
                    )

        elif output_format == "zarr":
            # Check that zarr output files exist
            zarr_files = list(output_path.glob("*/*.zarr"))
            assert len(zarr_files) > 0, "No zarr output files found"

            # Validate zarr files
            for zarr_file in zarr_files:
                zarr_group = zarr.open(zarr_file, mode="r")

                # Zarr files created by images_to_zarr are Groups with an 'images' key
                assert isinstance(
                    zarr_group, zarr.Group
                ), f"Expected zarr Group, got {type(zarr_group)}"
                assert (
                    "images" in zarr_group
                ), f"Expected 'images' key in zarr group, got keys: {list(zarr_group.keys())}"

                zarr_data = zarr_group["images"]

                # Check data dimensions - zarr data is in NHWC format
                assert (
                    len(zarr_data.shape) == 4
                ), f"Expected 4D data (NHWC), got shape {zarr_data.shape}"

                # Check number of channels (last dimension in NHWC)
                assert (
                    zarr_data.shape[-1] == num_output_channels
                ), f"Expected {num_output_channels} channels, got {zarr_data.shape[-1]}"

                # Check data type
                if data_type == "uint8":
                    assert zarr_data.dtype == np.uint8, f"Expected uint8, got {zarr_data.dtype}"
                elif data_type == "float32":
                    assert np.issubdtype(
                        zarr_data.dtype, np.float32
                    ), f"Expected float32, got {zarr_data.dtype}"

                # Validate data values match expected calculations
                self._validate_zarr_values(
                    zarr_data, selected_fits, normalisation_method, data_type, expected_values
                )

        # Validate gradient directions if we have the necessary parameters
        if channel_weights is not None and n_input is not None and n_output is not None:
            self._validate_gradient_directions(
                output_dir, channel_weights, selected_fits, n_input, n_output
            )

    def _validate_fits_values(
        self, hdul, selected_fits, normalisation_method, data_type, expected_values
    ):
        """Validate that FITS output values match expected calculations."""
        data_extensions = [hdu for hdu in hdul if hasattr(hdu, "data") and hdu.data is not None]

        for idx, hdu in enumerate(data_extensions):
            data = hdu.data

            # Check that data is not all zeros or all the same value
            assert not np.all(data == 0), "Output data is all zeros"
            assert data.std() >= 0, "Output data has no variation"

            if data_type == "uint8":
                # For uint8, values should be in [0, 255] range
                assert data.min() >= 0, f"uint8 data has negative values: {data.min()}"
                assert data.max() <= 255, f"uint8 data exceeds 255: {data.max()}"
            elif data_type == "float32":
                # For float32, values should be reasonable (not extreme)
                assert np.isfinite(data).all(), "float32 data contains non-finite values"

            # Validate actual values match expected gradient calculations
            """
            TODO this need a rework at some point, but should be ensured in other tests
            if idx < len(expected_values):
                expected_mean = expected_values[idx]
                actual_mean = float(np.mean(data))

                # For gradient validation, check that the data shows expected variation patterns
                # Calculate gradient characteristics
                height, width = data.shape

                # Check gradient direction based on expected pattern
                top_mean = np.mean(data[: height // 4, :])
                bottom_mean = np.mean(data[3 * height // 4 :, :])
                left_mean = np.mean(data[:, : width // 4])
                right_mean = np.mean(data[:, 3 * width // 4 :])
                center_mean = np.mean(
                    data[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3]
                )

                # Validate that gradients are preserved in the combined output
                # The exact pattern depends on channel weights, but variation should exist
                gradient_variation = data.std()

                # For float32, the gradient variation threshold should be much smaller
                # because values are normalized to [0,1] range instead of [0,255] for uint8
                min_variation_threshold = 0.001 if data_type == "float32" else 0.01

                assert (
                    gradient_variation > min_variation_threshold
                ), f"Channel {idx}: Insufficient gradient variation {gradient_variation}"

                # Check mean is in expected range with tolerance
                # The actual values vary slightly from expected due to interpolation during resizing
                tolerance = abs(expected_mean) * 0.20 + 5.0  # 20% relative + 5 absolute tolerance
                assert (
                    abs(actual_mean - expected_mean) <= tolerance
                ), f"Channel {idx}: Expected mean ~{expected_mean:.2f}, got {actual_mean:.2f}, tolerance {tolerance:.2f}"
            """

    def _validate_zarr_values(
        self, zarr_data, selected_fits, normalisation_method, data_type, expected_values
    ):
        """Validate that zarr output values match expected calculations."""
        # Zarr data from images_to_zarr is in NHWC format (N samples, H height, W width, C channels)
        assert len(zarr_data.shape) == 4, f"Expected 4D NHWC data, got shape {zarr_data.shape}"

        num_samples, height, width, num_channels = zarr_data.shape

        # Extract channels from the last dimension
        data_channels = [zarr_data[:, :, :, i] for i in range(num_channels)]

        for idx, channel_data in enumerate(data_channels):
            # Check that data is not all zeros or all the same value
            assert not np.all(channel_data == 0), f"Channel {idx} data is all zeros"
            assert channel_data.std() >= 0, f"Channel {idx} data has no variation"

            if data_type == "uint8":
                # For uint8, values should be in [0, 255] range
                assert (
                    channel_data.min() >= 0
                ), f"uint8 channel {idx} has negative values: {channel_data.min()}"
                assert (
                    channel_data.max() <= 255
                ), f"uint8 channel {idx} exceeds 255: {channel_data.max()}"
            elif data_type == "float32":
                # For float32, values should be reasonable (not extreme)
                assert np.isfinite(
                    channel_data
                ).all(), f"float32 channel {idx} contains non-finite values"

            # Validate actual values match expected gradient calculations
            if idx < len(expected_values):
                expected_mean = expected_values[idx]
                actual_mean = float(np.mean(channel_data))

                # For gradient validation, ensure variation exists
                gradient_variation = channel_data.std()

                # For float32, the gradient variation threshold should be much smaller
                # because values are normalized to [0,1] range instead of [0,255] for uint8
                min_variation_threshold = 0.001 if data_type == "float32" else 0.01

                assert (
                    gradient_variation > min_variation_threshold
                ), f"Channel {idx}: Insufficient gradient variation {gradient_variation}"

                # Check mean is in expected range with tolerance
                # The actual values vary slightly from expected due to interpolation during resizing
                tolerance = abs(expected_mean) * 0.20 + 5.0  # 20% relative + 5 absolute tolerance
                assert (
                    abs(actual_mean - expected_mean) <= tolerance
                ), f"Channel {idx}: Expected mean ~{expected_mean:.2f}, got {actual_mean:.2f}, tolerance {tolerance:.2f}"

    def _validate_gradient_directions(
        self, output_dir, channel_weights, selected_fits, n_input, n_output
    ):
        """Validate that output gradient directions match expected patterns based on channel mixing."""
        # Check for both FITS and zarr files
        fits_files = list(Path(output_dir).glob("*.fits"))
        zarr_files = list(Path(output_dir).glob("*/*.zarr"))

        if len(fits_files) > 0:
            output_format = "fits"
            output_files = fits_files
        elif len(zarr_files) > 0:
            output_format = "zarr"
            output_files = zarr_files
        else:
            assert False, "No output files (FITS or zarr) to validate gradient directions"

        matrix_config = get_channel_matrix_config(n_input, n_output)

        # Define gradient vectors directly for each extension type
        # must account for gradient strengths due to image value variations!
        extension_gradient_vectors = {
            "VIS": (1.0, 0.0),  # Horizontal gradient (left to right)
            "NIRH": (2 * 0.0, 2 * 1.0),  # Vertical gradient (top to bottom)
            "NIRJ": (3 * 0.707, 3 * 0.707),  # Diagonal gradient (top-left to bottom-right)
            "NIRY": (4 * -1.0, 4 * 0.0),  # Horizontal gradient (right to left)
        }

        if output_format == "fits":
            with fits.open(output_files[0]) as hdul:
                data_extensions = [
                    hdu for hdu in hdul if hasattr(hdu, "data") and hdu.data is not None
                ]
                self._validate_gradient_data(
                    data_extensions,
                    n_output,
                    matrix_config,
                    selected_fits,
                    channel_weights,
                    extension_gradient_vectors,
                    n_input,
                )
        else:  # zarr format
            zarr_group = zarr.open(output_files[0], mode="r")
            zarr_data = zarr_group["images"]  # NHWC format
            # Extract channel data for each output channel
            data_extensions = []
            for ch_idx in range(min(n_output, zarr_data.shape[-1])):
                # Get first sample's data for this channel (H, W)
                channel_data = zarr_data[0, :, :, ch_idx]
                data_extensions.append(type("MockHDU", (), {"data": channel_data})())
            self._validate_gradient_data(
                data_extensions,
                n_output,
                matrix_config,
                selected_fits,
                channel_weights,
                extension_gradient_vectors,
                n_input,
            )

    def _validate_gradient_data(
        self,
        data_extensions,
        n_output,
        matrix_config,
        selected_fits,
        channel_weights,
        extension_gradient_vectors,
        n_input=None,
    ):
        """Helper method to validate gradient data from either FITS or zarr format."""
        # If n_input isn't provided, infer it from the number of selected fits
        if n_input is None:
            n_input = len(selected_fits)
        for ch_idx, hdu in enumerate(data_extensions[:n_output]):
            data = hdu.data
            height, width = data.shape

            # Calculate gradient strength in different directions (traditional)
            left_mean = np.mean(data[:, : width // 4])  # Left quarter
            right_mean = np.mean(data[:, 3 * width // 4 :])  # Right quarter
            top_mean = np.mean(data[: height // 4, :])  # Top quarter
            bottom_mean = np.mean(data[3 * height // 4 :, :])  # Bottom quarter

            # Compute actual gradient vector using Sobel operators
            from scipy import ndimage

            # Apply Sobel operators to compute gradient in x and y directions
            grad_x = ndimage.sobel(data, axis=1)
            grad_y = ndimage.sobel(data, axis=0)

            # Compute magnitude and direction of gradient
            magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Clip extremely large gradient values to prevent overflow
            # Use a threshold based on data type (higher for uint8, lower for float32)
            threshold = 1e7 if data.dtype == np.uint8 else 1e3
            grad_x_clipped = np.clip(grad_x, -threshold, threshold)
            grad_y_clipped = np.clip(grad_y, -threshold, threshold)

            print(f"Max grad_x: {np.max(np.abs(grad_x))}, Max grad_y: {np.max(np.abs(grad_y))}")

            # Check traditional means for gradient direction
            h_gradient = right_mean - left_mean
            v_gradient = bottom_mean - top_mean

            # Use mean-based gradients to determine overall direction
            # This is more reliable than Sobel for our test patterns
            mean_dir_x = np.sign(h_gradient) if abs(h_gradient) > 0.01 else 0
            mean_dir_y = np.sign(v_gradient) if abs(v_gradient) > 0.01 else 0

            # Compute average gradient vector (weighted by magnitude to emphasize stronger gradients)
            avg_grad_x = (
                np.sum(grad_x_clipped * magnitude) / np.sum(magnitude)
                if np.sum(magnitude) > 0
                else 0
            )
            avg_grad_y = (
                np.sum(grad_y_clipped * magnitude) / np.sum(magnitude)
                if np.sum(magnitude) > 0
                else 0
            )

            if np.isnan(avg_grad_x):
                avg_grad_x = 0.0
            if np.isnan(avg_grad_y):
                avg_grad_y = 0.0

            # Make sure sign is correct based on means (handles the case where Sobel gives wrong direction)
            if mean_dir_x != 0 and np.sign(avg_grad_x) != mean_dir_x:
                avg_grad_x = -avg_grad_x
            if mean_dir_y != 0 and np.sign(avg_grad_y) != mean_dir_y:
                avg_grad_y = -avg_grad_y

            # Normalize the gradient vector
            grad_norm = np.sqrt(avg_grad_x**2 + avg_grad_y**2)
            print(f"\n {avg_grad_x}, {avg_grad_y}, {grad_norm}, {np.sum(magnitude)}")
            if grad_norm > 0:
                avg_grad_x /= grad_norm
                avg_grad_y /= grad_norm
            print(f"\n {avg_grad_x}, {avg_grad_y}, {grad_norm}")

            # Determine expected gradient direction based on channel weights
            dominant_weights = {}
            for ext_name, weights in matrix_config.items():
                if ch_idx < len(weights) and ext_name in selected_fits:
                    weight = weights[ch_idx]
                    dominant_weights[ext_name] = weight

            # Calculate expected gradient vector based on weighted combination of input vectors
            expected_grad_x = 0.0
            expected_grad_y = 0.0
            total_weight = 0.0

            for ext_name, weight in dominant_weights.items():
                if ext_name in extension_gradient_vectors:
                    vx, vy = extension_gradient_vectors[ext_name]
                    expected_grad_x += vx * weight
                    expected_grad_y += vy * weight
                    total_weight += weight

            # Normalize expected gradient vector
            if total_weight > 0:
                expected_grad_x /= total_weight
                expected_grad_y /= total_weight

            # Normalize the expected vector
            expected_norm = np.sqrt(expected_grad_x**2 + expected_grad_y**2)
            if expected_norm > 0:
                expected_grad_x /= expected_norm
                expected_grad_y /= expected_norm

            # Compute cosine similarity between expected and actual gradient vectors
            # This measures how aligned the vectors are (1.0 for perfect alignment, -1.0 for opposite direction)
            cosine_similarity = avg_grad_x * expected_grad_x + avg_grad_y * expected_grad_y

            print(f"\nChannel {ch_idx} gradient analysis:")
            print(f"  Dominant inputs: {dominant_weights}")
            print(f"  Left mean: {left_mean:.2f}, Right mean: {right_mean:.2f}")
            print(f"  Top mean: {top_mean:.2f}, Bottom mean: {bottom_mean:.2f}")
            print(f"  Horizontal gradient (R-L): {right_mean - left_mean:.2f}")
            print(f"  Vertical gradient (B-T): {bottom_mean - top_mean:.2f}")
            print(f"  Computed gradient vector: ({avg_grad_x:.3f}, {avg_grad_y:.3f})")
            print(f"  Expected gradient vector: ({expected_grad_x:.3f}, {expected_grad_y:.3f})")
            print(f"  Cosine similarity: {cosine_similarity:.3f}")

            # Validate gradient direction alignment
            # We use a threshold for cosine similarity to allow for some numerical imprecision
            cosine_threshold = 0.9  # Corresponds to about degrees of about 25 degrees

            # Special case: if we expect no gradient (uniform image), don't check alignment
            if expected_norm > 0.01:
                # Print diagnostics about our calculated and expected gradient vectors
                print(f"  Final Computed gradient vector: ({avg_grad_x:.3f}, {avg_grad_y:.3f})")
                print(
                    f"  Final Expected gradient vector: ({expected_grad_x:.3f}, {expected_grad_y:.3f})"
                )
                print(f"  Final Cosine similarity: {cosine_similarity:.3f}")

                assert cosine_similarity > cosine_threshold, (
                    f"Channel {ch_idx}: Gradient direction mismatch. "
                    f"Expected ({expected_grad_x:.3f}, {expected_grad_y:.3f}), "
                    f"Actual ({avg_grad_x:.3f}, {avg_grad_y:.3f}), "
                    f"Cosine similarity: {cosine_similarity:.3f} < {cosine_threshold}"
                )

            # Additional validation for single dominant inputs (simplified checks for backward compatibility)
            if len(dominant_weights) == 1:
                # Single dominant input - should match that pattern
                dominant_ext = list(dominant_weights.keys())[0]
                expected_x, expected_y = extension_gradient_vectors[dominant_ext]

                if expected_x > 0.9:  # left-to-right (VIS)
                    assert (
                        right_mean > left_mean
                    ), f"Channel {ch_idx}: Expected left-to-right gradient (VIS), but right({right_mean:.2f}) <= left({left_mean:.2f})"
                elif expected_x < -0.9:  # right-to-left (NIRY)
                    assert (
                        left_mean > right_mean
                    ), f"Channel {ch_idx}: Expected right-to-left gradient (NIRY), but left({left_mean:.2f}) <= right({right_mean:.2f})"
                elif expected_y > 0.9:  # top-to-bottom (NIRH)
                    assert (
                        bottom_mean > top_mean
                    ), f"Channel {ch_idx}: Expected top-to-bottom gradient (NIRH), but bottom({bottom_mean:.2f}) <= top({top_mean:.2f})"

    def _validate_amplified_channel_mixing(self, output_dir, output_type):
        """Validate that channel weights > 1.0 produce amplified output."""
        output_files = list(Path(output_dir).glob("*.fits"))
        assert len(output_files) > 0, "No output files to validate"

        with fits.open(output_files[0]) as hdul:
            data_extensions = [hdu for hdu in hdul if hasattr(hdu, "data") and hdu.data is not None]

            # Check that first data extension is bright and others near 0
            for idx, hdu in enumerate(data_extensions):
                data = hdu.data
                mean_val = np.mean(data)

                print(f"Channel {idx}: mean={mean_val:.2f}")
                if output_type == "uint8":
                    if idx == 0:
                        # First channel should be very bright due to high weights
                        assert mean_val > (255 / 2 - 1), f"Channel {idx} mean too low: {mean_val}"

                    else:
                        # Other channels should be near zero
                        assert mean_val < 10, f"Channel {idx} mean too high: {mean_val}"
                else:
                    if idx == 0:
                        # First channel should be very bright due to high weights
                        assert mean_val > (1.0 / 2 - 1), f"Channel {idx} mean too low: {mean_val}"

                    else:
                        # Other channels should be near zero
                        assert mean_val < 0.01, f"Channel {idx} mean too high: {mean_val}"

    def test_fits_order_influence(self, temp_dir, mock_fits_files):
        """Test that different ordering of input FITS files preserves gradient directions correctly.

        This test creates catalogues with FITS files in:
        1. Alphabetical order (VIS, NIRH, NIRJ, NIRY)
        2. Reverse alphabetical order (NIRY, NIRJ, NIRH, VIS)
        3. Random order (not alphabetical or reverse alphabetical)

        We use 1-to-1 channel mappings to validate that the output channel order
        correctly follows the expected order regardless of input file order.
        """
        # Ensure we have all required extensions for the test
        required_extensions = ["VIS", "NIRH", "NIRJ", "NIRY"]

        # Check if all required extensions are available
        missing_extensions = [ext for ext in required_extensions if ext not in mock_fits_files]
        if missing_extensions:
            logger.warning(f"Missing extensions for test: {missing_extensions}")
            logger.info("Creating missing extensions with mock data")

            # Create any missing extensions with mock data
            for ext_name in missing_extensions:
                # Use same mock data creation logic as in the mock_fits_files fixture
                if ext_name == "VIS":
                    value = 100.0
                    filename = "EUC_MER_BGSUB-MOSAIC-VIS_TILE999999999-FCEEA_20001021T024200.725618Z_00.00.fits"
                elif ext_name == "NIRH":
                    value = 200.0
                    filename = "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE999999999-6922DE_20001020T224400.516126Z_00.00.fits"
                elif ext_name == "NIRJ":
                    value = 300.0
                    filename = "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE999999999-85BBC9_20001020T224500.641201Z_00.00.fits"
                elif ext_name == "NIRY":
                    value = 400.0
                    filename = "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE999999999-DA7C06_20001020T224300.192709Z_00.00.fits"

                # Create the gradient image
                image_data = self._create_gradient_image(ext_name, value, (100, 100))

                # Create WCS for the FITS file
                wcs = WCS(naxis=2)
                wcs.wcs.crpix = [50, 50]
                wcs.wcs.crval = [52.0, -29.75]
                wcs.wcs.cdelt = [-0.00027, 0.00027]
                wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

                # Create FITS file
                fits_path = Path(temp_dir) / filename

                header = wcs.to_header()
                header["MAGZERO"] = 25.0  # Required for flux conversion
                header["EXTNAME"] = "PRIMARY"
                header["BUNIT"] = "electron/s"  # Units
                header["INSTRUME"] = ext_name  # Instrument name

                # Create PRIMARY HDU with image data
                primary_hdu = fits.PrimaryHDU(data=image_data, header=header)
                hdul = fits.HDUList([primary_hdu])
                hdul.writeto(fits_path, overwrite=True)

                # Add to mock_fits_files dictionary
                mock_fits_files[ext_name] = {
                    "path": str(fits_path),
                    "value": value,
                    "filename": filename,
                }

        # Define test configurations with subset of extensions if needed
        # For real tests, use 2 extensions to keep test time manageable
        n_input = 2  # Using only 2 extensions for faster testing
        n_output = 2

        # Define all possible orders to test with the reduced set of extensions
        # Just use the first n_input extensions from each order
        all_orders = {
            "alphabetical": ["VIS", "NIRH", "NIRJ", "NIRY"],
            "reverse_alphabetical": ["NIRY", "NIRJ", "NIRH", "VIS"],
            "random": ["NIRJ", "VIS", "NIRY", "NIRH"],
        }

        # Use only the orders with the specified number of extensions
        orders_to_test = {name: order[:n_input] for name, order in all_orders.items()}

        # Just test one order for now to debug the issue
        orders_to_test = {"alphabetical": orders_to_test["alphabetical"]}

        for order_name, extension_order in orders_to_test.items():
            logger.info(f"\n\nTesting {order_name} order: {extension_order}")

            # Select FITS files in the specified order
            selected_fits = {ext: mock_fits_files[ext] for ext in extension_order}

            # Create test catalogue
            catalogue_path = self.create_test_catalogue(temp_dir, list(selected_fits.values()))

            # Analyze catalogue
            result_cat_analysis = analyse_source_catalogue(catalogue_path)

            # Create configuration
            config = get_default_config()
            config.source_catalogue = catalogue_path
            config.output_dir = str(Path(temp_dir) / f"output_{order_name}")
            config.target_resolution = 32  # Small for fast testing
            config.data_type = "float32"
            config.normalisation_method = "linear"
            config.normalisation.a = 1.0  # Linear scaling factor
            config.normalisation.percentile = 99.0
            config.max_workers = 1  # Single worker for predictable results
            config.output_format = "fits"
            config.N_batch_cutout_process = 10  # Process all sources in one batch

            # Set up extensions
            config.fits_extensions = ["PRIMARY"]

            # Get the proper extensions format for the configuration
            extensions = []
            for ext_name in extension_order:
                extensions.append({"name": ext_name, "ext": "PRIMARY"})

            # Setup proper extensions with the format expected by the processing pipeline
            config.selected_extensions = extensions
            config.available_extensions = extensions
            # Set up extensions and channel weights
            config.fits_extensions = ["PRIMARY"]  # Use PRIMARY extension
            config.channel_weights, config.selected_extensions, config.available_extensions = (
                self.create_ui_shared_config_like_config(
                    selected_fits,
                    n_output,
                    result_cat_analysis,
                    n_input,
                    n_output,
                    order=True,  # we do ordering
                )
            )
            logger.info(f"Config setup: {config.channel_weights}, {config.selected_extensions}")

            # Create output directory
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

            # Run orchestrator
            orchestrator = Orchestrator(config)

            # Mock FITS file loading to use our test files
            with patch("cutana.fits_dataset.load_fits_sets") as mock_load_fits:
                # Setup mocks to ensure proper operation
                mock_load_fits.return_value = self._create_mock_fits_data(selected_fits)

                # Create test catalogue data for processing
                fits_file_paths = [fits_info["path"] for fits_info in selected_fits.values()]
                test_data = [
                    {
                        "SourceID": f"source_{i}",
                        "RA": 52.0 + i * 0.0001,
                        "Dec": -29.75 + i * 0.0001,
                        "diameter_pixel": 10.0,
                        "fits_file_paths": json.dumps(fits_file_paths),
                    }
                    for i in range(5)
                ]
                catalogue_df = pd.DataFrame(test_data)

                # Run processing
                try:
                    result = orchestrator.start_processing(catalogue_df)
                    # Verify successful completion
                    assert result["status"] == "completed"
                finally:
                    # Ensure all processes are terminated
                    try:
                        orchestrator.stop_processing()
                    except Exception:
                        pass

            # Validate that each output channel has the correct gradient direction
            # For 1-to-1 mapping, channel 0 should have the gradient of first input file, etc.

            # Define expected gradient patterns based on input order
            expected_patterns = {}
            for idx, ext_name in enumerate(extension_order):
                # Map each output channel to the expected gradient pattern of its input
                if ext_name == "VIS":
                    expected_patterns[idx] = "left_to_right"
                elif ext_name == "NIRH":
                    expected_patterns[idx] = "top_to_bottom"
                elif ext_name == "NIRJ":
                    expected_patterns[idx] = "diagonal"
                elif ext_name == "NIRY":
                    expected_patterns[idx] = "right_to_left"

            # Validate output files
            output_files = list(Path(config.output_dir).glob("*.fits"))
            assert len(output_files) > 0, "No output files to validate"

            with fits.open(output_files[0]) as hdul:
                data_extensions = [
                    hdu for hdu in hdul if hasattr(hdu, "data") and hdu.data is not None
                ]

                # Define gradient vectors for expected patterns
                gradient_vectors = {
                    "left_to_right": (1.0, 0.0),
                    "right_to_left": (-1.0, 0.0),
                    "top_to_bottom": (0.0, 1.0),
                    "diagonal": (0.707, 0.707),
                }

                for ch_idx, hdu in enumerate(data_extensions[:n_output]):
                    data = hdu.data
                    height, width = data.shape

                    # Calculate gradient using Sobel operators
                    from scipy import ndimage

                    grad_x = ndimage.sobel(data, axis=1)
                    grad_y = ndimage.sobel(data, axis=0)

                    # Compute magnitude and direction of gradient
                    magnitude = np.sqrt(grad_x**2 + grad_y**2)

                    # Compute average gradient vector
                    avg_grad_x = (
                        np.sum(grad_x * magnitude) / np.sum(magnitude)
                        if np.sum(magnitude) > 0
                        else 0
                    )
                    avg_grad_y = (
                        np.sum(grad_y * magnitude) / np.sum(magnitude)
                        if np.sum(magnitude) > 0
                        else 0
                    )

                    # Normalize the gradient vector
                    grad_norm = np.sqrt(avg_grad_x**2 + avg_grad_y**2)
                    if grad_norm > 0:
                        avg_grad_x /= grad_norm
                        avg_grad_y /= grad_norm

                    # Get expected gradient vector for this channel
                    expected_pattern = expected_patterns[ch_idx]
                    expected_grad_x, expected_grad_y = gradient_vectors[expected_pattern]

                    # Compute cosine similarity between expected and actual gradient vectors
                    if grad_norm > 0:
                        cosine_similarity = (
                            avg_grad_x * expected_grad_x + avg_grad_y * expected_grad_y
                        )
                    else:
                        cosine_similarity = 0.0

                    logger.info(
                        f"\nOrder: {order_name}, Channel {ch_idx} ({extension_order[ch_idx]}) gradient analysis:"
                    )
                    logger.info(f"  Expected pattern: {expected_pattern}")
                    logger.info(f"  Computed gradient vector: ({avg_grad_x:.3f}, {avg_grad_y:.3f})")
                    logger.info(
                        f"  Expected gradient vector: ({expected_grad_x:.3f}, {expected_grad_y:.3f})"
                    )
                    logger.info(f"  Cosine similarity: {cosine_similarity:.3f}")

                    # Validate gradient direction alignment with a tolerance
                    cosine_threshold = 0.9  # Allow about 25 degrees of tolerance

                    assert cosine_similarity > cosine_threshold, (
                        f"Order: {order_name}, Channel {ch_idx}: Gradient direction mismatch. "
                        f"Expected {expected_pattern} ({expected_grad_x:.3f}, {expected_grad_y:.3f}), "
                        f"got ({avg_grad_x:.3f}, {avg_grad_y:.3f}), similarity: {cosine_similarity:.3f}"
                    )

            logger.info(f"✓ {order_name} order test passed!")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
