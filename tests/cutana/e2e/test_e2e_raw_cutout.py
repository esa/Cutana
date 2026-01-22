#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
End-to-end tests for raw cutout extraction (do_only_cutout_extraction=True).

This module tests the raw cutout extraction functionality where:
- No resizing is applied
- No normalization is applied
- No channel combination is applied
- Output format is FITS with original data type preserved

Test Setup:
- Creates a small 10x10 FITS file with known values
- Extracts a 9x9 cutout centered on a source
- Validates that output matches expected input data
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger

from cutana.get_default_config import get_default_config
from cutana.orchestrator import Orchestrator


class TestEndToEndRawCutout:
    """Test raw cutout extraction end-to-end (do_only_cutout_extraction=True)."""

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

    @pytest.fixture
    def small_fits_file(self, temp_dir):
        """Create a small 10x10 FITS file with known float32 values for testing."""
        # Create a 10x10 image with a specific pattern that we can verify
        # Use a simple pattern: pixel value = row_index * 10 + col_index
        image_size = 10
        image_data = np.zeros((image_size, image_size), dtype=np.float32)

        for row in range(image_size):
            for col in range(image_size):
                image_data[row, col] = row * 10 * 1e-7 + 1 * 10 ** (-col)

        # Expected values in the 10x10 grid:
        # Row 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        # Row 1: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        # ...
        # Row 9: 90, 91, 92, 93, 94, 95, 96, 97, 98, 99

        logger.info(f"Created 10x10 test image with values from 0 to 99")
        logger.info(f"Image data shape: {image_data.shape}, dtype: {image_data.dtype}")
        logger.info(f"Image min: {image_data.min()}, max: {image_data.max()}")

        # Create simple WCS for coordinate transformation
        # Place center of image at RA=180, Dec=0
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [5.5, 5.5]  # Reference pixel at center (1-indexed)
        wcs.wcs.crval = [180.0, 0.0]  # Reference coordinates
        wcs.wcs.cdelt = [-0.0001, 0.0001]  # Pixel scale (~0.36 arcsec/pixel)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Create FITS file
        fits_filename = "test_10x10_float32.fits"
        fits_path = Path(temp_dir) / fits_filename

        header = wcs.to_header()
        header["MAGZERO"] = 25.0  # Required for flux conversion
        header["EXTNAME"] = "PRIMARY"
        header["BUNIT"] = "electron/s"
        header["INSTRUME"] = "TEST"

        # Create PRIMARY HDU with image data
        primary_hdu = fits.PrimaryHDU(data=image_data, header=header)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(fits_path, overwrite=True)

        logger.info(f"Created test FITS file: {fits_path}")

        return {
            "path": str(fits_path),
            "filename": fits_filename,
            "image_data": image_data,
            "wcs": wcs,
        }

    def create_test_catalogue(self, temp_dir, fits_path, ra=180.0, dec=0.0, diameter_pixel=9):
        """Create test catalogue with a single source at specified coordinates.

        Args:
            temp_dir: Temporary directory for output
            fits_path: Path to the FITS file
            ra: Right ascension of source (default: center of image)
            dec: Declination of source (default: center of image)
            diameter_pixel: Cutout size in pixels (default: 9 for 9x9 cutout)

        Returns:
            Path to the catalogue CSV file
        """
        catalogue_data = [
            {
                "SourceID": "test_source_1",
                "RA": ra,
                "Dec": dec,
                "diameter_pixel": diameter_pixel,
                "fits_file_paths": json.dumps([fits_path]),
            }
        ]

        df = pd.DataFrame(catalogue_data)
        catalogue_path = Path(temp_dir) / "test_catalogue.csv"
        df.to_csv(catalogue_path, index=False)

        logger.info(f"Created test catalogue: {catalogue_path}")
        logger.info(f"Source at RA={ra}, Dec={dec}, diameter={diameter_pixel}px")

        return str(catalogue_path)

    def test_raw_cutout_extraction_9x9_from_10x10(self, temp_dir, small_fits_file):
        """
        Test raw cutout extraction: extract 9x9 cutout from 10x10 FITS image.

        This test verifies:
        1. do_only_cutout_extraction=True produces unresized cutouts
        2. Output is in FITS format
        3. Output data type is float32
        4. Cutout values exactly match the expected region from the input using np.allclose
        """
        # Create output directory
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)

        # Create test catalogue with source at center of image
        catalogue_path = self.create_test_catalogue(
            temp_dir,
            small_fits_file["path"],
            ra=180.0,  # Center of image
            dec=0.0,  # Center of image
            diameter_pixel=9,  # 9x9 cutout
        )

        # Configure for raw cutout extraction
        config = get_default_config()
        config.source_catalogue = catalogue_path
        config.output_dir = str(output_dir)
        config.output_format = "fits"
        config.data_type = "float32"
        config.do_only_cutout_extraction = True
        config.apply_flux_conversion = False  # No flux conversion for this test
        config.max_workers = 1
        config.N_batch_cutout_process = 10
        config.padding_factor = 1.0  # No padding

        # Set channel weights for single input
        config.channel_weights = {"PRIMARY": [1.0]}
        config.fits_extensions = ["PRIMARY"]

        # Set required selected_extensions (mimics UI configuration)
        config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]
        config.available_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]

        logger.info("Starting raw cutout extraction test")
        logger.info(f"Config: do_only_cutout_extraction={config.do_only_cutout_extraction}")
        logger.info(f"Config: output_format={config.output_format}")
        logger.info(f"Config: data_type={config.data_type}")

        # Run the orchestrator
        orchestrator = Orchestrator(config)
        result = orchestrator.run()

        # Verify successful completion
        assert result is not None, "Orchestrator should return a result"
        logger.info(f"Orchestrator result: {result}")

        # Find the output FITS file
        output_fits_files = list(output_dir.glob("*.fits"))
        assert (
            len(output_fits_files) == 1
        ), f"Expected 1 output FITS file, found {len(output_fits_files)}"

        output_fits_path = output_fits_files[0]
        logger.info(f"Found output FITS file: {output_fits_path}")

        # Read and validate the output
        with fits.open(output_fits_path) as hdul:
            logger.info(f"Output FITS has {len(hdul)} HDUs")

            # Find the data HDU (could be PRIMARY or extension)
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.size > 0:
                    data_hdu = hdu
                    break

            assert data_hdu is not None, "Output FITS should contain data"

            output_data = data_hdu.data
            logger.info(f"Output data shape: {output_data.shape}")
            logger.info(f"Output data dtype: {output_data.dtype}")
            logger.info(f"Output data min: {output_data.min()}, max: {output_data.max()}")

            # Verify data type is float32 (may have different byte order like >f4 for big-endian)
            assert np.issubdtype(
                output_data.dtype, np.floating
            ), f"Expected floating point, got {output_data.dtype}"
            assert (
                output_data.dtype.itemsize == 4
            ), f"Expected 4-byte float (float32), got {output_data.dtype.itemsize}-byte"

            # Get the input image for comparison
            input_data = small_fits_file["image_data"]

            # Handle potential channel dimension to get 2D output
            if len(output_data.shape) == 3:
                if output_data.shape[2] == 1:
                    output_2d = output_data[:, :, 0]
                elif output_data.shape[0] == 1:
                    output_2d = output_data[0, :, :]
                else:
                    output_2d = output_data
            else:
                output_2d = output_data

            logger.info(f"Output 2D shape for comparison: {output_2d.shape}")

            # Verify output shape is 9x9
            assert output_2d.shape == (9, 9), f"Expected (9, 9), got {output_2d.shape}"

            # The 9x9 cutout centered on a 10x10 image should extract rows 0-8 and cols 0-8
            # Since the image center is at pixel (4.5, 4.5) in 0-indexed coords (CRPIX=[5.5,5.5] is 1-indexed)
            # and we request a 9x9 cutout, it should extract [0:9, 0:9]
            expected_cutout = input_data[0:9, 0:9]

            logger.info(f"Expected cutout shape: {expected_cutout.shape}")
            logger.info(f"Expected cutout values:\n{expected_cutout}")
            logger.info(f"Output cutout values:\n{output_2d}")

            # Direct comparison using np.allclose
            assert np.allclose(output_2d, expected_cutout, rtol=1e-9, atol=1e-9), (
                f"Output cutout does not match expected input region.\n"
                f"Max difference: {np.max(np.abs(output_2d - expected_cutout))}"
            )

            logger.info("Raw cutout extraction test PASSED - output exactly matches input region")
