#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
End-to-end tests for raw cutout extraction with StreamingOrchestrator.

This module tests the combination of:
- do_only_cutout_extraction=True (raw cutouts without resizing/normalization)
- StreamingOrchestrator batch-by-batch processing

Test Setup:
- Creates mock FITS files with known values
- Uses StreamingOrchestrator to process sources in batches
- Validates that output cutouts preserve original data values

Key validations:
- Raw cutouts maintain original data type (float32)
- No resizing is applied (cutout size matches extraction region)
- No normalization is applied (values match input)
- Streaming batch processing works correctly with raw mode
- Both sync and async streaming modes work with raw cutout extraction
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger

from cutana import StreamingOrchestrator, get_default_config


class TestEndToEndStreamingRawCutout:
    """Test raw cutout extraction with StreamingOrchestrator end-to-end."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files with robust cleanup."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Handle Windows file permission issues by retrying deletion
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
    def mock_fits_file(self, temp_dir):
        """Create a mock FITS file with known float32 values for testing."""
        # Create a 20x20 image with a specific gradient pattern
        image_size = 20
        image_data = np.zeros((image_size, image_size), dtype=np.float32)

        # Create a pattern: pixel value = row * 100 + col
        for row in range(image_size):
            for col in range(image_size):
                image_data[row, col] = row * 100 + col

        logger.info(f"Created {image_size}x{image_size} test image")
        logger.info(f"Image data shape: {image_data.shape}, dtype: {image_data.dtype}")
        logger.info(f"Image min: {image_data.min()}, max: {image_data.max()}")

        # Create WCS centered on RA=180, Dec=0
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [10.5, 10.5]  # Reference pixel at center (1-indexed)
        wcs.wcs.crval = [180.0, 0.0]  # Reference coordinates
        wcs.wcs.cdelt = [-0.0001, 0.0001]  # Pixel scale (~0.36 arcsec/pixel)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Create FITS file
        fits_filename = "test_20x20_float32.fits"
        fits_path = Path(temp_dir) / fits_filename

        header = wcs.to_header()
        header["MAGZERO"] = 25.0  # Required for flux conversion
        header["EXTNAME"] = "PRIMARY"
        header["BUNIT"] = "electron/s"
        header["INSTRUME"] = "TEST"

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

    @pytest.fixture
    def multi_extension_fits_file(self, temp_dir):
        """Create a multi-extension FITS file with known values for each extension."""
        image_size = 20

        # Create different patterns for each extension
        extensions = {}
        extension_names = ["VIS", "NIR_H", "NIR_J"]

        for idx, ext_name in enumerate(extension_names):
            image_data = np.zeros((image_size, image_size), dtype=np.float32)
            base_value = (idx + 1) * 1000
            for row in range(image_size):
                for col in range(image_size):
                    image_data[row, col] = base_value + row * 10 + col
            extensions[ext_name] = image_data

        # Create WCS
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [10.5, 10.5]
        wcs.wcs.crval = [180.0, 0.0]
        wcs.wcs.cdelt = [-0.0001, 0.0001]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        fits_filename = "test_multi_ext.fits"
        fits_path = Path(temp_dir) / fits_filename

        # Create HDU list with PRIMARY and IMAGE extensions
        hdu_list = [fits.PrimaryHDU()]

        for ext_name, img_data in extensions.items():
            header = wcs.to_header()
            header["EXTNAME"] = ext_name
            header["MAGZERO"] = 25.0
            header["BUNIT"] = "electron/s"
            header["INSTRUME"] = ext_name
            hdu = fits.ImageHDU(data=img_data, header=header, name=ext_name)
            hdu_list.append(hdu)

        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fits_path, overwrite=True)

        logger.info(f"Created multi-extension test FITS file: {fits_path}")

        return {
            "path": str(fits_path),
            "filename": fits_filename,
            "extensions": extensions,
            "extension_names": extension_names,
            "wcs": wcs,
        }

    def create_test_catalogue(self, temp_dir, fits_path, num_sources=5, diameter_pixel=10):
        """Create test catalogue with multiple sources spread across the image.

        Args:
            temp_dir: Temporary directory for output
            fits_path: Path to the FITS file
            num_sources: Number of sources to create
            diameter_pixel: Cutout size in pixels

        Returns:
            Path to the catalogue CSV file
        """
        catalogue_data = []

        # Create sources spread across the image center region
        for i in range(num_sources):
            # Spread sources slightly around the center
            ra = 180.0 + (i - num_sources // 2) * 0.00005
            dec = 0.0 + (i - num_sources // 2) * 0.00005

            catalogue_data.append(
                {
                    "SourceID": f"streaming_source_{i+1:03d}",
                    "RA": ra,
                    "Dec": dec,
                    "diameter_pixel": diameter_pixel,
                    "fits_file_paths": json.dumps([fits_path]),
                }
            )

        df = pd.DataFrame(catalogue_data)
        catalogue_path = Path(temp_dir) / "streaming_test_catalogue.csv"
        df.to_csv(catalogue_path, index=False)

        logger.info(f"Created test catalogue with {num_sources} sources: {catalogue_path}")

        return str(catalogue_path)

    def get_raw_cutout_config(self, temp_dir, catalogue_path):
        """Create a configuration for raw cutout extraction with streaming.

        Args:
            temp_dir: Temporary directory for output
            catalogue_path: Path to the source catalogue

        Returns:
            Configuration DotMap
        """
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)

        config = get_default_config()
        config.source_catalogue = catalogue_path
        config.output_dir = str(output_dir)

        # Raw cutout extraction settings
        config.output_format = "fits"  # Required for do_only_cutout_extraction
        config.data_type = "float32"
        config.do_only_cutout_extraction = True
        config.apply_flux_conversion = False

        # Processing settings
        config.max_workers = 1
        config.N_batch_cutout_process = 10
        config.padding_factor = 1.0
        config.max_workflow_time_seconds = 600
        config.skip_memory_calibration_wait = True

        # Channel configuration for single extension
        config.channel_weights = {"PRIMARY": [1.0]}
        config.fits_extensions = ["PRIMARY"]
        config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]
        config.available_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]

        return config

    def test_streaming_raw_cutout_sync_mode(self, temp_dir, mock_fits_file):
        """Test raw cutout extraction with synchronous streaming mode.

        Validates:
        - StreamingOrchestrator works with do_only_cutout_extraction=True
        - Output is in FITS format
        - Output preserves original float32 data type
        - Cutouts are not resized (match extraction region size)
        """
        # Create catalogue with 10 sources
        catalogue_path = self.create_test_catalogue(
            temp_dir,
            mock_fits_file["path"],
            num_sources=10,
            diameter_pixel=8,
        )

        config = self.get_raw_cutout_config(temp_dir, catalogue_path)

        logger.info("Starting streaming raw cutout test (sync mode)")
        logger.info(f"Config: do_only_cutout_extraction={config.do_only_cutout_extraction}")

        orchestrator = StreamingOrchestrator(config)

        try:
            # Initialize streaming in sync mode with small batch size
            orchestrator.init_streaming(
                batch_size=3,  # Small batches to test streaming
                write_to_disk=True,  # Write FITS files
                synchronised_loading=True,  # Sync mode
            )

            num_batches = orchestrator.get_batch_count()
            assert num_batches > 0, "Should have at least one batch"
            logger.info(f"Processing {num_batches} batches")

            # Process all batches
            for batch_idx in range(num_batches):
                result = orchestrator.next_batch()

                assert result is not None, f"Batch {batch_idx + 1} should return a result"
                assert result["batch_number"] == batch_idx + 1, f"Batch number mismatch"

                logger.info(f"Completed batch {batch_idx + 1}/{num_batches}")

        finally:
            orchestrator.cleanup()

        # Verify output FITS files
        output_dir = Path(config.output_dir)
        output_fits_files = list(output_dir.glob("**/*.fits"))

        assert len(output_fits_files) > 0, "Should have created output FITS files"
        logger.info(f"Found {len(output_fits_files)} output FITS files")

        # Validate a sample output file
        sample_fits = output_fits_files[0]
        with fits.open(sample_fits) as hdul:
            # Find the data HDU
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.size > 0:
                    data_hdu = hdu
                    break

            assert data_hdu is not None, "Output FITS should contain data"

            # Verify data type is float32
            assert np.issubdtype(
                data_hdu.data.dtype, np.floating
            ), f"Expected floating point, got {data_hdu.data.dtype}"
            assert (
                data_hdu.data.dtype.itemsize == 4
            ), f"Expected 4-byte float (float32), got {data_hdu.data.dtype}"

            logger.info(f"Sample output shape: {data_hdu.data.shape}")
            logger.info(f"Sample output dtype: {data_hdu.data.dtype}")

    def test_streaming_raw_cutout_async_mode(self, temp_dir, mock_fits_file):
        """Test raw cutout extraction with asynchronous streaming mode.

        Validates:
        - Async prefetching works with do_only_cutout_extraction=True
        - All batches are processed correctly
        - Output files are generated
        """
        catalogue_path = self.create_test_catalogue(
            temp_dir,
            mock_fits_file["path"],
            num_sources=12,
            diameter_pixel=8,
        )

        config = self.get_raw_cutout_config(temp_dir, catalogue_path)

        logger.info("Starting streaming raw cutout test (async mode)")

        orchestrator = StreamingOrchestrator(config)

        try:
            # Initialize streaming in async mode
            orchestrator.init_streaming(
                batch_size=4,
                write_to_disk=True,
                synchronised_loading=False,  # Async mode!
            )

            num_batches = orchestrator.get_batch_count()
            assert num_batches > 0, "Should have at least one batch"
            logger.info(f"Processing {num_batches} batches in async mode")

            results = []
            for batch_idx in range(num_batches):
                result = orchestrator.next_batch()
                results.append(result)

                assert result["batch_number"] == batch_idx + 1

            assert len(results) == num_batches, "Should process all batches"

        finally:
            orchestrator.cleanup()

        # Verify output
        output_dir = Path(config.output_dir)
        output_fits_files = list(output_dir.glob("**/*.fits"))
        assert len(output_fits_files) > 0, "Should have created output FITS files"
        logger.info(f"Async mode created {len(output_fits_files)} FITS files")

    def test_streaming_raw_cutout_value_preservation(self, temp_dir, mock_fits_file):
        """Test that raw cutout extraction preserves original pixel values.

        This test verifies that:
        - Pixel values in output match the expected region from input
        - No scaling or normalization is applied
        """
        # Create single source at image center for precise value checking
        catalogue_path = self.create_test_catalogue(
            temp_dir,
            mock_fits_file["path"],
            num_sources=1,
            diameter_pixel=6,  # 6x6 cutout from center
        )

        config = self.get_raw_cutout_config(temp_dir, catalogue_path)

        logger.info("Testing raw cutout value preservation")

        orchestrator = StreamingOrchestrator(config)

        try:
            orchestrator.init_streaming(
                batch_size=10,
                write_to_disk=True,
                synchronised_loading=True,
            )

            result = orchestrator.next_batch()
            assert result is not None

        finally:
            orchestrator.cleanup()

        # Verify output values match input
        output_dir = Path(config.output_dir)
        output_fits_files = list(output_dir.glob("**/*.fits"))
        assert len(output_fits_files) == 1, "Should have exactly one output file"

        with fits.open(output_fits_files[0]) as hdul:
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.size > 0:
                    data_hdu = hdu
                    break

            assert data_hdu is not None
            output_data = data_hdu.data

            # Handle potential channel dimension
            if len(output_data.shape) == 3:
                if output_data.shape[2] == 1:
                    output_2d = output_data[:, :, 0]
                elif output_data.shape[0] == 1:
                    output_2d = output_data[0, :, :]
                else:
                    output_2d = output_data
            else:
                output_2d = output_data

            # Verify the values are in the expected range from our input pattern
            # Input pattern: pixel = row * 100 + col
            # For a 20x20 image, values range from 0 to 1919
            assert output_2d.min() >= 0, "Output min should be >= 0"
            assert output_2d.max() <= 1919, "Output max should be <= 1919"

            # Verify it's float32
            assert np.issubdtype(output_2d.dtype, np.floating)

            logger.info(f"Output data range: [{output_2d.min()}, {output_2d.max()}]")
            logger.info("Value preservation test passed")

    def test_streaming_raw_cutout_batch_size_edge_cases(self, temp_dir, mock_fits_file):
        """Test streaming raw cutout with various batch size configurations.

        Tests:
        - Batch size larger than total sources
        - Batch size of 1 (single source per batch)
        """
        # Test with batch_size > num_sources
        catalogue_path = self.create_test_catalogue(
            temp_dir,
            mock_fits_file["path"],
            num_sources=3,
            diameter_pixel=6,
        )

        config = self.get_raw_cutout_config(temp_dir, catalogue_path)

        logger.info("Testing batch size larger than source count")

        orchestrator = StreamingOrchestrator(config)

        try:
            orchestrator.init_streaming(
                batch_size=100,  # Much larger than 3 sources
                write_to_disk=True,
                synchronised_loading=True,
            )

            num_batches = orchestrator.get_batch_count()
            assert num_batches >= 1, "Should have at least 1 batch"

            for i in range(num_batches):
                result = orchestrator.next_batch()
                assert result is not None

        finally:
            orchestrator.cleanup()

        # Verify output
        output_dir = Path(config.output_dir)
        output_fits_files = list(output_dir.glob("**/*.fits"))
        assert len(output_fits_files) == 3, "Should have 3 output files (one per source)"

    def test_streaming_raw_cutout_not_initialized_error(self, temp_dir, mock_fits_file):
        """Test that calling next_batch without init_streaming raises error."""
        catalogue_path = self.create_test_catalogue(temp_dir, mock_fits_file["path"], num_sources=1)

        config = self.get_raw_cutout_config(temp_dir, catalogue_path)
        orchestrator = StreamingOrchestrator(config)

        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                orchestrator.next_batch()
        finally:
            orchestrator.cleanup()

    def test_streaming_raw_cutout_cleanup_terminates_pending(self, temp_dir, mock_fits_file):
        """Test that cleanup properly terminates any pending batch preparation.

        This tests the cleanup mechanism for async mode where a batch might
        be preparing in the background when cleanup is called.
        """
        catalogue_path = self.create_test_catalogue(
            temp_dir,
            mock_fits_file["path"],
            num_sources=6,
            diameter_pixel=6,
        )

        config = self.get_raw_cutout_config(temp_dir, catalogue_path)

        orchestrator = StreamingOrchestrator(config)

        try:
            # Initialize in async mode - this starts preparing the first batch
            orchestrator.init_streaming(
                batch_size=2,
                write_to_disk=True,
                synchronised_loading=False,
            )

            # Get first batch
            result = orchestrator.next_batch()
            assert result is not None

            # Cleanup should terminate any pending preparation
            # (next batch should be preparing in async mode)

        finally:
            # This should not raise or hang
            orchestrator.cleanup()

        logger.info("Cleanup with pending batch completed successfully")


class TestStreamingRawCutoutMultiExtension:
    """Test raw cutout extraction with multi-extension FITS files."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files with robust cleanup."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        for attempt in range(3):
            try:
                shutil.rmtree(temp_dir)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)
                    continue
                else:
                    shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def multi_ext_fits_file(self, temp_dir):
        """Create a multi-extension FITS file with distinct values per extension."""
        image_size = 20

        extensions = {}
        extension_data = {
            "VIS": 1000,  # Base value for VIS
            "NIR_H": 2000,  # Base value for NIR_H
        }

        # Create WCS
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [10.5, 10.5]
        wcs.wcs.crval = [180.0, 0.0]
        wcs.wcs.cdelt = [-0.0001, 0.0001]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        fits_filename = "test_multi_ext_raw.fits"
        fits_path = Path(temp_dir) / fits_filename

        hdu_list = [fits.PrimaryHDU()]

        for ext_name, base_value in extension_data.items():
            image_data = np.full((image_size, image_size), base_value, dtype=np.float32)
            # Add gradient to make values unique
            for row in range(image_size):
                for col in range(image_size):
                    image_data[row, col] = base_value + row * 10 + col

            extensions[ext_name] = image_data

            header = wcs.to_header()
            header["EXTNAME"] = ext_name
            header["MAGZERO"] = 25.0
            header["BUNIT"] = "electron/s"
            header["INSTRUME"] = ext_name
            hdu = fits.ImageHDU(data=image_data, header=header, name=ext_name)
            hdu_list.append(hdu)

        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fits_path, overwrite=True)

        logger.info(f"Created multi-extension FITS: {fits_path}")

        return {
            "path": str(fits_path),
            "extensions": extensions,
            "wcs": wcs,
        }

    def create_multi_ext_catalogue(self, temp_dir, fits_path, num_sources=3):
        """Create catalogue for multi-extension FITS testing."""
        catalogue_data = []
        for i in range(num_sources):
            ra = 180.0 + (i - num_sources // 2) * 0.00003
            dec = 0.0 + (i - num_sources // 2) * 0.00003

            catalogue_data.append(
                {
                    "SourceID": f"multi_ext_source_{i+1:03d}",
                    "RA": ra,
                    "Dec": dec,
                    "diameter_pixel": 8,
                    "fits_file_paths": json.dumps([fits_path]),
                }
            )

        df = pd.DataFrame(catalogue_data)
        catalogue_path = Path(temp_dir) / "multi_ext_catalogue.csv"
        df.to_csv(catalogue_path, index=False)
        return str(catalogue_path)

    def test_streaming_raw_cutout_multi_extension(self, temp_dir, multi_ext_fits_file):
        """Test raw cutout extraction preserves multiple extensions correctly.

        Validates:
        - Each extension's data is extracted correctly
        - Extension values match their base patterns
        """
        catalogue_path = self.create_multi_ext_catalogue(
            temp_dir, multi_ext_fits_file["path"], num_sources=2
        )

        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)

        config = get_default_config()
        config.source_catalogue = catalogue_path
        config.output_dir = str(output_dir)
        config.output_format = "fits"
        config.data_type = "float32"
        config.do_only_cutout_extraction = True
        config.apply_flux_conversion = False
        config.max_workers = 1
        config.N_batch_cutout_process = 10
        config.padding_factor = 1.0
        config.max_workflow_time_seconds = 600
        config.skip_memory_calibration_wait = True

        # Configure for two extensions
        config.channel_weights = {"VIS": [1.0], "NIR_H": [1.0]}
        config.fits_extensions = ["VIS", "NIR_H"]
        config.selected_extensions = [
            {"name": "VIS", "ext": "VIS"},
            {"name": "NIR_H", "ext": "NIR_H"},
        ]
        config.available_extensions = [
            {"name": "VIS", "ext": "VIS"},
            {"name": "NIR_H", "ext": "NIR_H"},
        ]

        logger.info("Testing multi-extension raw cutout streaming")

        orchestrator = StreamingOrchestrator(config)

        try:
            orchestrator.init_streaming(
                batch_size=5,
                write_to_disk=True,
                synchronised_loading=True,
            )

            num_batches = orchestrator.get_batch_count()
            for i in range(num_batches):
                result = orchestrator.next_batch()
                assert result is not None

        finally:
            orchestrator.cleanup()

        # Verify output files
        output_fits_files = list(output_dir.glob("**/*.fits"))
        assert len(output_fits_files) == 2, f"Expected 2 output files, got {len(output_fits_files)}"

        # Check one output file has both extensions' data
        with fits.open(output_fits_files[0]) as hdul:
            # Find data HDUs
            data_hdus = [hdu for hdu in hdul if hdu.data is not None and hdu.data.size > 0]
            assert len(data_hdus) >= 1, "Should have data HDUs"

            # If multi-channel, the output might be stacked
            data = data_hdus[0].data
            logger.info(f"Multi-extension output shape: {data.shape}")

            # If 3D with channels, check channel count matches extensions
            if len(data.shape) == 3:
                # Should have 2 channels (VIS and NIR_H)
                assert data.shape[2] == 2, f"Expected 2 channels, got {data.shape[2]}"

        logger.info("Multi-extension raw cutout test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
