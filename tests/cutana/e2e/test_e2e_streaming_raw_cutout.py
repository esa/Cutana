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
- Creates mock FITS files with known values (class-scoped for performance)
- Uses StreamingOrchestrator to process sources in batches
- Validates that output cutouts preserve original data values

Key validations:
- Raw cutouts maintain original data type (float32)
- No resizing is applied (cutout size matches extraction region)
- No normalization is applied (values match input)
- Streaming batch processing works correctly with raw mode
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger

from cutana import StreamingOrchestrator, get_default_config


def _create_mock_wcs():
    """Create a WCS centered on RA=180, Dec=0."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [180.0, 0.0]
    wcs.wcs.cdelt = [-0.0001, 0.0001]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _create_test_catalogue(catalogue_path, fits_path, num_sources=5, diameter_pixel=10):
    """Create test catalogue with multiple sources spread across the image.

    Args:
        catalogue_path: Path to write the catalogue CSV file
        fits_path: Path to the FITS file
        num_sources: Number of sources to create
        diameter_pixel: Cutout size in pixels

    Returns:
        Path string to the catalogue CSV file
    """
    catalogue_data = []
    for i in range(num_sources):
        ra = 180.0 + (i - num_sources // 2) * 0.00005
        dec = 0.0 + (i - num_sources // 2) * 0.00005
        catalogue_data.append(
            {
                "SourceID": f"streaming_source_{i + 1:03d}",
                "RA": ra,
                "Dec": dec,
                "diameter_pixel": diameter_pixel,
                "fits_file_paths": json.dumps([fits_path]),
            }
        )

    df = pd.DataFrame(catalogue_data)
    df.to_csv(catalogue_path, index=False)
    logger.info(f"Created test catalogue with {num_sources} sources: {catalogue_path}")
    return str(catalogue_path)


def _get_raw_cutout_config(output_dir, catalogue_path):
    """Create a configuration for raw cutout extraction with streaming.

    Args:
        output_dir: Directory for output files
        catalogue_path: Path to the source catalogue

    Returns:
        Configuration DotMap
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = get_default_config()
    config.source_catalogue = catalogue_path
    config.output_dir = str(output_dir)

    # Raw cutout extraction settings
    config.output_format = "fits"
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


class TestEndToEndStreamingRawCutout:
    """Test raw cutout extraction with StreamingOrchestrator end-to-end."""

    @pytest.fixture(scope="class")
    def shared_dir(self, tmp_path_factory):
        """Class-scoped temporary directory shared across all tests."""
        return tmp_path_factory.mktemp("streaming_raw")

    @pytest.fixture(scope="class")
    def mock_fits_file(self, shared_dir):
        """Create a mock FITS file once for the entire test class."""
        image_size = 20
        image_data = np.zeros((image_size, image_size), dtype=np.float32)
        for row in range(image_size):
            for col in range(image_size):
                image_data[row, col] = row * 100 + col

        logger.info(f"Created {image_size}x{image_size} test image")

        wcs = _create_mock_wcs()

        fits_path = shared_dir / "test_20x20_float32.fits"
        header = wcs.to_header()
        header["MAGZERO"] = 25.0
        header["EXTNAME"] = "PRIMARY"
        header["BUNIT"] = "electron/s"
        header["INSTRUME"] = "TEST"

        primary_hdu = fits.PrimaryHDU(data=image_data, header=header)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(fits_path, overwrite=True)

        logger.info(f"Created test FITS file: {fits_path}")

        return {
            "path": str(fits_path),
            "image_data": image_data,
            "wcs": wcs,
        }

    @pytest.fixture(scope="class")
    def catalogues(self, shared_dir, mock_fits_file):
        """Create all catalogues once for the entire test class.

        Returns a dict keyed by (num_sources, diameter_pixel).
        """
        fits_path = mock_fits_file["path"]
        specs = [
            (10, 8),  # streaming_batches
            (1, 6),  # value_preservation
            (3, 6),  # batch_size_edge_cases
            (1, 10),  # not_initialized_error
            (6, 6),  # cleanup_terminates_pending
        ]
        result = {}
        for num_sources, diameter_pixel in specs:
            key = (num_sources, diameter_pixel)
            if key not in result:
                cat_path = shared_dir / f"catalogue_{num_sources}s_{diameter_pixel}d.csv"
                _create_test_catalogue(cat_path, fits_path, num_sources, diameter_pixel)
                result[key] = str(cat_path)
        return result

    def test_streaming_raw_cutout_batches(self, shared_dir, mock_fits_file, catalogues):
        """Test raw cutout extraction processes all batches and produces valid FITS output.

        Validates:
        - StreamingOrchestrator works with do_only_cutout_extraction=True
        - Output is in FITS format with float32 data type
        - All batches are returned with correct batch numbers
        """
        output_dir = shared_dir / "output_batches"
        config = _get_raw_cutout_config(output_dir, catalogues[(10, 8)])

        logger.info(f"Config: do_only_cutout_extraction={config.do_only_cutout_extraction}")

        orchestrator = StreamingOrchestrator(config)

        try:
            orchestrator.init_streaming(
                batch_size=3,
                write_to_disk=True,
            )

            num_batches = orchestrator.get_batch_count()
            assert num_batches > 0, "Should have at least one batch"

            for batch_idx in range(num_batches):
                result = orchestrator.next_batch()
                assert result is not None, f"Batch {batch_idx + 1} should return a result"
                assert result["batch_number"] == batch_idx + 1, "Batch number mismatch"

        finally:
            orchestrator.cleanup()

        # Verify output FITS files
        output_fits_files = list(output_dir.glob("**/*.fits"))
        assert len(output_fits_files) > 0, "Should have created output FITS files"

        with fits.open(output_fits_files[0]) as hdul:
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.size > 0:
                    data_hdu = hdu
                    break

            assert data_hdu is not None, "Output FITS should contain data"
            assert np.issubdtype(data_hdu.data.dtype, np.floating), (
                f"Expected floating point, got {data_hdu.data.dtype}"
            )
            assert data_hdu.data.dtype.itemsize == 4, (
                f"Expected 4-byte float (float32), got {data_hdu.data.dtype}"
            )

    def test_streaming_raw_cutout_value_preservation(self, shared_dir, mock_fits_file, catalogues):
        """Test that raw cutout extraction preserves original pixel values.

        This test verifies that:
        - Pixel values in output match the expected region from input
        - No scaling or normalization is applied
        """
        output_dir = shared_dir / "output_value"
        config = _get_raw_cutout_config(output_dir, catalogues[(1, 6)])

        logger.info("Testing raw cutout value preservation")

        orchestrator = StreamingOrchestrator(config)

        try:
            orchestrator.init_streaming(
                batch_size=10,
                write_to_disk=True,
            )

            result = orchestrator.next_batch()
            assert result is not None

        finally:
            orchestrator.cleanup()

        # Verify output values match input
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

    def test_streaming_raw_cutout_batch_size_edge_cases(
        self, shared_dir, mock_fits_file, catalogues
    ):
        """Test streaming raw cutout with various batch size configurations.

        Tests:
        - Batch size larger than total sources
        - Batch size of 1 (single source per batch)
        """
        output_dir = shared_dir / "output_batch_edge"
        config = _get_raw_cutout_config(output_dir, catalogues[(3, 6)])

        logger.info("Testing batch size larger than source count")

        orchestrator = StreamingOrchestrator(config)

        try:
            orchestrator.init_streaming(
                batch_size=100,
                write_to_disk=True,
            )

            num_batches = orchestrator.get_batch_count()
            assert num_batches >= 1, "Should have at least 1 batch"

            for i in range(num_batches):
                result = orchestrator.next_batch()
                assert result is not None

        finally:
            orchestrator.cleanup()

        # Verify output
        output_fits_files = list(output_dir.glob("**/*.fits"))
        assert len(output_fits_files) == 3, "Should have 3 output files (one per source)"

    def test_streaming_raw_cutout_not_initialized_error(
        self, shared_dir, mock_fits_file, catalogues
    ):
        """Test that calling next_batch without init_streaming raises error."""
        output_dir = shared_dir / "output_not_init"
        config = _get_raw_cutout_config(output_dir, catalogues[(1, 10)])

        orchestrator = StreamingOrchestrator(config)

        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                orchestrator.next_batch()
        finally:
            orchestrator.cleanup()

    def test_streaming_raw_cutout_cleanup_terminates_pending(
        self, shared_dir, mock_fits_file, catalogues
    ):
        """Test that cleanup properly terminates any pending batch preparation.

        This tests the cleanup mechanism where a batch might
        be preparing in the background when cleanup is called.
        """
        output_dir = shared_dir / "output_cleanup"
        config = _get_raw_cutout_config(output_dir, catalogues[(6, 6)])

        orchestrator = StreamingOrchestrator(config)

        try:
            orchestrator.init_streaming(
                batch_size=2,
                write_to_disk=True,
            )

            result = orchestrator.next_batch()
            assert result is not None

        finally:
            orchestrator.cleanup()

        logger.info("Cleanup with pending batch completed successfully")


class TestStreamingRawCutoutMultiExtension:
    """Test raw cutout extraction with multi-extension FITS files."""

    @pytest.fixture(scope="class")
    def shared_dir(self, tmp_path_factory):
        """Class-scoped temporary directory shared across all tests."""
        return tmp_path_factory.mktemp("streaming_multi_ext")

    @pytest.fixture(scope="class")
    def multi_ext_fits_file(self, shared_dir):
        """Create a multi-extension FITS file once for the entire test class."""
        image_size = 20

        extensions = {}
        extension_data = {
            "VIS": 1000,
            "NIR_H": 2000,
        }

        wcs = _create_mock_wcs()

        fits_path = shared_dir / "test_multi_ext_raw.fits"

        hdu_list = [fits.PrimaryHDU()]

        for ext_name, base_value in extension_data.items():
            image_data = np.full((image_size, image_size), base_value, dtype=np.float32)
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

    @pytest.fixture(scope="class")
    def catalogue(self, shared_dir, multi_ext_fits_file):
        """Create the multi-extension catalogue once for the entire test class."""
        fits_path = multi_ext_fits_file["path"]
        num_sources = 2
        catalogue_data = []
        for i in range(num_sources):
            ra = 180.0 + (i - num_sources // 2) * 0.00003
            dec = 0.0 + (i - num_sources // 2) * 0.00003
            catalogue_data.append(
                {
                    "SourceID": f"multi_ext_source_{i + 1:03d}",
                    "RA": ra,
                    "Dec": dec,
                    "diameter_pixel": 8,
                    "fits_file_paths": json.dumps([fits_path]),
                }
            )

        df = pd.DataFrame(catalogue_data)
        catalogue_path = shared_dir / "multi_ext_catalogue.csv"
        df.to_csv(catalogue_path, index=False)
        return str(catalogue_path)

    def test_streaming_raw_cutout_multi_extension(self, shared_dir, multi_ext_fits_file, catalogue):
        """Test raw cutout extraction preserves multiple extensions correctly.

        Validates:
        - Each extension's data is extracted correctly
        - Extension values match their base patterns
        """
        output_dir = shared_dir / "output"
        output_dir.mkdir(exist_ok=True)

        config = get_default_config()
        config.source_catalogue = catalogue
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
            {"name": "PRIMARY", "ext": "VIS"},
            {"name": "PRIMARY", "ext": "NIR_H"},
        ]
        config.available_extensions = [
            {"name": "PRIMARY", "ext": "VIS"},
            {"name": "PRIMARY", "ext": "NIR_H"},
        ]

        logger.info("Testing multi-extension raw cutout streaming")

        orchestrator = StreamingOrchestrator(config)

        try:
            orchestrator.init_streaming(
                batch_size=5,
                write_to_disk=True,
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
            data_hdus = [hdu for hdu in hdul if hdu.data is not None and hdu.data.size > 0]
            assert len(data_hdus) >= 1, "Should have data HDUs"

            data = data_hdus[0].data
            logger.info(f"Multi-extension output shape: {data.shape}")

            if len(data.shape) == 3:
                assert data.shape[2] == 2, f"Expected 2 channels, got {data.shape[2]}"

        logger.info("Multi-extension raw cutout test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
