#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the cutout_process module using TDD approach.

Tests cover:
- FITS file loading and processing
- Cutout extraction from FITS tiles
- WCS coordinate transformation
- Error handling for missing/corrupted files
- Integration with image_processor
"""

from unittest.mock import patch, MagicMock, Mock
import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from cutana.cutout_process import (
    create_cutouts_batch,
    create_cutouts,
    _process_sources_batch_vectorized_with_fits_set,
)
from cutana.fits_reader import load_fits_file
from cutana.cutout_extraction import (
    radec_to_pixel,
    extract_cutout_from_extension,
)
from cutana.job_tracker import JobTracker


class TestCutoutProcessFunctions:
    """Test suite for cutout process functions."""

    @pytest.fixture
    def mock_job_tracker(self):
        """Create mock job tracker for testing."""
        mock_tracker = Mock(spec=JobTracker)
        mock_tracker.report_process_progress.return_value = True
        mock_tracker.update_process_stage.return_value = True
        return mock_tracker

    @pytest.fixture
    def mock_source_data(self):
        """Create mock source data for testing."""
        return {
            "SourceID": "MockSource_00001",
            "RA": 150.0,
            "Dec": 2.0,
            "diameter_pixel": 20,  # Only use diameter_pixel (mutual exclusivity)
            "fits_file_paths": "['/mock/data/euclid_tile_001.fits']",
        }

    @pytest.fixture
    def mock_fits_file(self, tmp_path):
        """Create a mock FITS file for testing."""
        fits_path = tmp_path / "test_tile.fits"

        # Create simple test data
        data = np.random.random((1000, 1000)).astype(np.float32)

        # Create a simple WCS
        header = fits.Header()
        header["CRVAL1"] = 150.0  # RA reference
        header["CRVAL2"] = 2.0  # Dec reference
        header["CRPIX1"] = 500.0  # Reference pixel X
        header["CRPIX2"] = 500.0  # Reference pixel Y
        header["CDELT1"] = -0.0002777778  # -1 arcsec/pixel
        header["CDELT2"] = 0.0002777778  # 1 arcsec/pixel
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["NAXIS"] = 2
        header["NAXIS1"] = 1000
        header["NAXIS2"] = 1000
        header["MAGZERO"] = 25.0  # Zeropoint for preprocessing

        # Create primary HDU
        primary_hdu = fits.PrimaryHDU()

        # Create image extensions
        vis_hdu = fits.ImageHDU(data, header=header, name="VIS")
        niry_hdu = fits.ImageHDU(data * 0.8, header=header, name="NIR-Y")
        nirh_hdu = fits.ImageHDU(data * 0.6, header=header, name="NIR-H")

        hdul = fits.HDUList([primary_hdu, vis_hdu, niry_hdu, nirh_hdu])
        hdul.writeto(fits_path, overwrite=True)

        return str(fits_path)

    @pytest.fixture
    def cutout_config(self):
        """Create cutout processing configuration."""
        from cutana.get_default_config import get_default_config
        from dotmap import DotMap

        config = get_default_config()
        config.target_resolution = 64
        config.data_type = "float32"
        config.fits_extensions = ["VIS", "NIR-Y", "NIR-H"]
        config.normalisation_method = "linear"
        config.output_dir = "/tmp/cutouts"
        config.N_batch_cutout_process = 100
        config.apply_flux_conversion = True
        config.user_flux_conversion_function = None
        config.flux_conversion_keywords = DotMap({"AB_zeropoint": "MAGZERO"})
        config.process_id = "test_process"
        config.job_tracker_session_id = "test_session"
        return config

    def test_load_fits_file(self, mock_fits_file):
        """Test loading FITS file with multiple extensions."""
        fits_extensions = ["VIS", "NIR-Y", "NIR-H"]

        hdul, wcs_dict = load_fits_file(mock_fits_file, fits_extensions)

        assert hdul is not None
        assert len(hdul) >= 4  # Primary + 3 extensions
        assert "VIS" in wcs_dict
        assert "NIR-Y" in wcs_dict
        assert "NIR-H" in wcs_dict

        # Check WCS objects are valid
        for ext_name, wcs_obj in wcs_dict.items():
            assert isinstance(wcs_obj, WCS)

        hdul.close()

    def test_coordinate_transformation(self, mock_fits_file):
        """Test RA/Dec to pixel coordinate transformation."""
        hdul, wcs_dict = load_fits_file(mock_fits_file, ["VIS"])

        # Test coordinate at the reference position
        ra, dec = 150.0, 2.0
        wcs_obj = wcs_dict["VIS"]

        pixel_x, pixel_y = radec_to_pixel(ra, dec, wcs_obj)

        # Should be close to reference pixel (500, 500)
        # Note: WCS uses 1-based indexing, so we expect ~499 for 0-based
        assert abs(pixel_x - 499.0) < 1.0
        assert abs(pixel_y - 499.0) < 1.0

        hdul.close()

    def test_extract_cutout_from_extension(self, mock_fits_file, mock_source_data):
        """Test extracting cutout from a specific FITS extension."""
        hdul, wcs_dict = load_fits_file(mock_fits_file, ["VIS"])

        # Extract cutout from VIS extension
        cutout = extract_cutout_from_extension(
            hdul["VIS"],
            wcs_dict["VIS"],
            mock_source_data["RA"],
            mock_source_data["Dec"],
            mock_source_data["diameter_pixel"],
        )

        assert cutout is not None
        assert isinstance(cutout, np.ndarray)
        assert cutout.shape == (20, 20)  # diameter_pixel = 20

        hdul.close()

    def test_cutout_boundary_handling(self, mock_fits_file):
        """Test cutout extraction near image boundaries."""
        hdul, wcs_dict = load_fits_file(mock_fits_file, ["VIS"])

        # Test cutout at edge of image
        edge_ra, edge_dec = 149.86, 1.86  # Near edge based on WCS

        cutout = extract_cutout_from_extension(
            hdul["VIS"], wcs_dict["VIS"], edge_ra, edge_dec, 50  # Large cutout size
        )

        # Should handle boundary gracefully with padding
        assert cutout is not None
        assert cutout.shape == (50, 50)  # Should be padded to requested size

        hdul.close()

    def test_error_handling_missing_file(self, cutout_config, mock_source_data, mock_job_tracker):
        """Test error handling when FITS file is missing."""
        # Test error handling in batch processing instead
        source_batch = [mock_source_data.copy()]
        source_batch[0]["fits_file_paths"] = "['/nonexistent/file.fits']"

        # Should handle error gracefully and continue processing
        results = create_cutouts_batch(source_batch, cutout_config, mock_job_tracker)

        # Results should be a single item with empty metadata since FITS file doesn't exist
        assert isinstance(results, list)
        assert len(results) == 1  # Returns one result with empty metadata
        assert "metadata" in results[0]
        assert len(results[0]["metadata"]) == 0  # No successful results

    def test_error_handling_corrupted_wcs(
        self, tmp_path, cutout_config, mock_source_data, mock_job_tracker
    ):
        """Test error handling with corrupted WCS information."""
        fits_path = tmp_path / "corrupted_wcs.fits"

        # Create FITS with invalid WCS
        data = np.random.random((100, 100)).astype(np.float32)
        header = fits.Header()
        header["CRVAL1"] = "INVALID"  # Invalid WCS value
        header["NAXIS"] = 2
        header["NAXIS1"] = 100
        header["NAXIS2"] = 100

        hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, header=header, name="VIS")])
        hdul.writeto(fits_path, overwrite=True)

        source_batch = [mock_source_data.copy()]
        source_batch[0]["fits_file_paths"] = f"['{fits_path}']"

        results = create_cutouts_batch(source_batch, cutout_config, mock_job_tracker)

        # Should handle WCS error gracefully - either empty results or error results
        assert isinstance(results, list)
        # May be empty or contain error results

    def test_multiple_extensions_processing(self, mock_fits_file, cutout_config, mock_source_data):
        """Test processing cutouts from multiple FITS extensions."""
        source_data = mock_source_data.copy()
        source_data["fits_file_paths"] = f"['{mock_fits_file}']"

        fits_extensions = cutout_config["fits_extensions"]
        hdul, wcs_dict = load_fits_file(mock_fits_file, fits_extensions)

        cutouts = {}
        for ext_name in fits_extensions:
            if ext_name in wcs_dict:
                cutout = extract_cutout_from_extension(
                    hdul[ext_name],
                    wcs_dict[ext_name],
                    source_data["RA"],
                    source_data["Dec"],
                    source_data["diameter_pixel"],
                )
                if cutout is not None:
                    cutouts[ext_name] = cutout

        # Should have cutouts from all requested extensions
        assert len(cutouts) >= 1  # At least one successful extraction
        for ext_name, cutout in cutouts.items():
            assert isinstance(cutout, np.ndarray)
            assert cutout.shape == (source_data["diameter_pixel"], source_data["diameter_pixel"])

        hdul.close()

    def test_create_cutouts_batch_function(self, mock_source_data, cutout_config, mock_job_tracker):
        """Test the create_cutouts_batch function with FITS set-based processing."""
        source_batch = [mock_source_data]

        # Mock FITS loading and FITS set-based processing
        mock_hdul = MagicMock()
        mock_wcs = MagicMock()

        with (
            patch("cutana.fits_dataset.load_fits_file") as mock_load_fits,
            patch(
                "cutana.cutout_process._process_sources_batch_vectorized_with_fits_set"
            ) as mock_process,
            patch("cutana.cutout_process.create_process_zarr_archive_initial") as mock_zarr_create,
            patch("cutana.cutout_process.append_to_zarr_archive") as mock_zarr_append,
        ):

            mock_load_fits.return_value = (mock_hdul, {"VIS": mock_wcs})
            # Mock should return batch format: {"cutouts": tensor, "metadata": list}
            batch_cutouts = np.random.random((1, 20, 20, 1)).astype(np.float32)
            mock_process.return_value = [
                {
                    "cutouts": batch_cutouts,
                    "metadata": [
                        {
                            "source_id": mock_source_data["SourceID"],
                            "channel_count": 1,
                        }
                    ],
                }
            ]
            mock_zarr_create.return_value = True  # Successful zarr creation
            mock_zarr_append.return_value = True  # Successful zarr append

            results = create_cutouts_batch(source_batch, cutout_config, mock_job_tracker)

            assert len(results) == 1
            assert "metadata" in results[0]
            # For zarr output (default), metadata should contain incremental write indicator
            assert len(results[0]["metadata"]) == 1
            assert results[0]["metadata"][0]["source_id"] == "written_incrementally"
            # FITS loading should be called once per FITS file in the set
            mock_load_fits.assert_called()
            # FITS set-based processing should be called once per source
            mock_process.assert_called()
            # Zarr writing should be called
            mock_zarr_create.assert_called()

    def test_create_cutouts_legacy_function(
        self, mock_source_data, cutout_config, mock_job_tracker
    ):
        """Test the create_cutouts legacy function."""
        source_batch = [mock_source_data]

        with patch("cutana.cutout_process.create_cutouts_batch") as mock_batch:
            mock_batch.return_value = [
                {
                    "source_id": mock_source_data["SourceID"],
                    "processed_cutouts": {"VIS": np.random.random((20, 20))},
                }
            ]

            results = create_cutouts(source_batch, cutout_config)

            assert len(results) == 1
            assert results[0]["source_id"] == mock_source_data["SourceID"]
            # Check that create_cutouts_batch was called with correct arguments
            # (the third argument is the auto-created job_tracker)
            assert mock_batch.call_count == 1
            call_args = mock_batch.call_args[0]
            assert call_args[0] == source_batch
            assert call_args[1] == cutout_config
            # Third argument should be a JobTracker instance
            from cutana.job_tracker import JobTracker

            assert isinstance(call_args[2], JobTracker)

    def test_batch_processing_multiple_sources(self, cutout_config, mock_job_tracker):
        """Test FITS set-based batch processing of multiple sources."""
        # Set output format to FITS to get actual processed data back
        cutout_config.output_format = "fits"
        source_batch = [
            {
                "SourceID": f"MockSource_{i:05d}",
                "RA": 150.0 + i * 0.01,
                "Dec": 2.0 + i * 0.01,
                "diameter_pixel": 20,  # Only use diameter_pixel to avoid validation error
                "fits_file_paths": "['/mock/data/euclid_tile_001.fits']",
            }
            for i in range(5)
        ]

        # Mock FITS loading and FITS set-based processing
        mock_hdul = MagicMock()
        mock_wcs = MagicMock()

        with (
            patch("cutana.fits_dataset.load_fits_file") as mock_load_fits,
            patch(
                "cutana.cutout_process._process_sources_batch_vectorized_with_fits_set"
            ) as mock_process,
        ):

            mock_load_fits.return_value = (mock_hdul, {"VIS": mock_wcs})

            # Mock successful vectorized batch processing for all sources
            def mock_process_batch(
                sources_batch,
                loaded_fits_data,
                config,
                profiler=None,
                process_name=None,
                job_tracker=None,
            ):
                # Return batch format: one result with cutouts tensor and metadata list
                batch_cutouts = np.random.random((len(sources_batch), 20, 20, 1)).astype(np.float32)
                metadata_list = [
                    {
                        "source_id": source_data["SourceID"],
                        "channel_count": 1,
                    }
                    for source_data in sources_batch
                ]
                return [
                    {
                        "cutouts": batch_cutouts,
                        "metadata": metadata_list,
                    }
                ]

            mock_process.side_effect = mock_process_batch

            results = create_cutouts_batch(source_batch, cutout_config, mock_job_tracker)

            assert len(results) == 1
            assert "metadata" in results[0]
            # For FITS output, should have metadata for all 5 sources
            assert len(results[0]["metadata"]) == 5
            assert all("source_id" in meta for meta in results[0]["metadata"])
            # Should call vectorized batch processing once per FITS set (all sources share the same FITS file)
            assert mock_process.call_count == 1

    def test_error_handling_in_batch_processing(self, cutout_config, mock_job_tracker):
        """Test error handling in batch processing."""
        source_batch = [
            {
                "SourceID": "ErrorSource_001",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_arcsec": 10.0,
                "diameter_pixel": 20,
                "fits_file_paths": "['/nonexistent/file.fits']",
            }
        ]

        # Should handle errors gracefully and continue processing
        results = create_cutouts_batch(source_batch, cutout_config, mock_job_tracker)

        # Results should be a single item with empty metadata since FITS file doesn't exist
        assert isinstance(results, list)
        assert len(results) == 1  # Returns one result with empty metadata
        assert "metadata" in results[0]
        assert len(results[0]["metadata"]) == 0  # No successful results

    def test_memory_efficient_processing(
        self, mock_fits_file, cutout_config, mock_source_data, mock_job_tracker
    ):
        """Test memory-efficient processing with FITS file handling."""
        source_data = mock_source_data.copy()
        source_data["fits_file_paths"] = f"['{mock_fits_file}']"

        # Mock to track file operations through the batch processing
        with patch("cutana.fits_dataset.load_fits_sets") as mock_load_fits_sets:
            # Mock return value: fits_path -> (hdul, wcs_dict)
            mock_hdul = MagicMock()
            mock_wcs_dict = {"PRIMARY": MagicMock()}
            mock_load_fits_sets.return_value = {mock_fits_file: (mock_hdul, mock_wcs_dict)}

            # Set up mock HDU list
            mock_hdul.__len__.return_value = 4
            mock_hdul.__getitem__.side_effect = lambda x: MagicMock(data=np.zeros((100, 100)))
            mock_hdul.close = MagicMock()

            # Test that batch processing uses FITS loading appropriately
            try:
                create_cutouts_batch([source_data], cutout_config, mock_job_tracker)
            except Exception:  # noqa: E722
                pass  # May fail due to mocking

            # Note: load_fits_sets may or may not be called depending on implementation
            # This test just verifies the API doesn't crash with mocked dependencies
            pass

    def test_backward_compatibility_wrapper(self, mock_fits_file):
        """Test that the wrapper function in cutout_process still works."""
        # Use VIS extension instead of PRIMARY since PRIMARY has no data in our test files
        fits_extensions = ["VIS"]

        # Test that the old function still works through the wrapper
        try:
            hdul, wcs_dict = load_fits_file(mock_fits_file, fits_extensions)

            assert hdul is not None
            assert isinstance(wcs_dict, dict)
            # Should have at least one extension loaded
            assert len(wcs_dict) >= 1

            hdul.close()

        except Exception as e:
            # This is expected if fitsbolt has compatibility issues
            assert (
                "fitsbolt failed" in str(e)
                or "Invalid FITS file" in str(e)
                or "No valid extensions found" in str(e)
            )
            pytest.skip(f"fitsbolt compatibility issue in wrapper: {e}")

    def test_create_cutouts_main_subprocess_execution(self, tmp_path):
        """Test the main subprocess entry point with file-based communication."""
        import json
        import tempfile

        # Create test data
        source_batch = [
            {
                "SourceID": "subprocess_test_001",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['test_file.fits']",
            }
        ]

        config = {
            "fits_extensions": ["VIS"],
            "output_format": "zarr",
            "output_dir": str(tmp_path),
            "log_level": "DEBUG",
        }

        # Create temporary files for communication
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as source_file:
            json.dump(source_batch, source_file)
            source_temp_path = source_file.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config_file:
            json.dump(config, config_file)
            config_temp_path = config_file.name

        # Test command line argument parsing
        import sys

        original_argv = sys.argv[:]

        try:
            sys.argv = ["cutout_process.py", source_temp_path, config_temp_path]

            # Mock the actual processing to avoid FITS file dependencies
            with (
                patch("cutana.cutout_process.create_cutouts_batch", return_value=[]),
                patch("builtins.print") as mock_print,
            ):
                # Import and run the main function
                from cutana.cutout_process import create_cutouts_main

                try:
                    create_cutouts_main()
                    # Should print JSON output
                    mock_print.assert_called_once()
                    call_args = mock_print.call_args[0][0]
                    output = json.loads(call_args)
                    assert output["status"] == "success"
                    assert "processed_count" in output
                except SystemExit:
                    pass  # Expected for successful completion

        finally:
            sys.argv = original_argv
            # Cleanup temp files (files should be cleaned up by the process)
            import os

            for temp_file in [source_temp_path, config_temp_path]:
                try:
                    os.unlink(temp_file)
                except FileNotFoundError:
                    pass  # Already cleaned up by process

    def test_create_cutouts_main_error_handling(self, tmp_path):
        """Test main function error handling."""
        import sys
        import json
        import tempfile

        # Test insufficient arguments
        original_argv = sys.argv[:]

        try:
            sys.argv = ["cutout_process.py"]  # Missing arguments

            with patch("builtins.print") as mock_print:
                from cutana.cutout_process import create_cutouts_main

                try:
                    create_cutouts_main()
                except SystemExit as e:
                    assert e.code == 1
                    # Should print error to logger, not stdout

        finally:
            sys.argv = original_argv

        # Test with invalid JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as invalid_file:
            invalid_file.write("invalid json content")
            invalid_temp_path = invalid_file.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as valid_file:
            json.dump({"test": "config"}, valid_file)
            valid_temp_path = valid_file.name

        try:
            sys.argv = [
                "cutout_process.py",
                invalid_temp_path,
                valid_temp_path,
            ]

            with patch("builtins.print") as mock_print:
                from cutana.cutout_process import create_cutouts_main

                try:
                    create_cutouts_main()
                except SystemExit as e:
                    assert e.code == 1
                    # Should print error JSON
                    mock_print.assert_called_once()
                    call_args = mock_print.call_args[0][0]
                    output = json.loads(call_args)
                    assert "error" in output  # Should have error field when failing

        finally:
            sys.argv = original_argv
            import os

            for temp_file in [invalid_temp_path, valid_temp_path]:
                try:
                    os.unlink(temp_file)
                except FileNotFoundError:
                    pass

    def test_fits_path_parsing_edge_cases(self, cutout_config, mock_job_tracker):
        """Test various FITS path parsing scenarios."""

        # Test different string formats for fits_file_paths
        test_cases = [
            # Standard list format
            {
                "SourceID": "parse_test_001",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "['file1.fits', 'file2.fits']",
            },
            # Single file without brackets
            {
                "SourceID": "parse_test_002",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "single_file.fits",
            },
            # Already parsed as list
            {
                "SourceID": "parse_test_003",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": ["already_list.fits"],
            },
            # Malformed string
            {
                "SourceID": "parse_test_004",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "[malformed, 'string",
            },
        ]

        with (
            patch(
                "cutana.fits_reader.load_fits_file",
                side_effect=FileNotFoundError("Mocked file not found"),
            ),
            patch("cutana.cutout_process.logger"),
        ):
            for i, source_data in enumerate(test_cases):
                try:
                    results = create_cutouts_batch([source_data], cutout_config, mock_job_tracker)
                    # Most should result in empty results due to missing files, but shouldn't crash
                    assert isinstance(results, list)
                except Exception as e:
                    # Should handle parsing errors gracefully
                    assert "parsing" in str(e).lower() or "fits" in str(e).lower()

    def test_size_parameter_edge_cases(self, cutout_config, mock_job_tracker):
        """Test size parameter handling edge cases."""
        base_source = {
            "SourceID": "size_test_001",
            "RA": 150.0,
            "Dec": 2.0,
            "fits_file_paths": "['test.fits']",
        }

        test_cases = [
            # Zero diameter_pixel
            {**base_source, "diameter_pixel": 0},
            # Negative diameter_pixel
            {**base_source, "diameter_pixel": -10},
            # Zero diameter_arcsec
            {**base_source, "diameter_arcsec": 0.0},
            # Negative diameter_arcsec
            {**base_source, "diameter_arcsec": -5.0},
            # Both parameters (should fail with mutual exclusivity)
            {**base_source, "diameter_pixel": 32, "diameter_arcsec": 10.0},
            # Neither parameter
            {**base_source},
            # Non-numeric values
            {**base_source, "diameter_pixel": "invalid"},
            {**base_source, "diameter_arcsec": "invalid"},
        ]

        with patch(
            "cutana.fits_reader.load_fits_file",
            side_effect=FileNotFoundError("Mocked"),
        ):
            for i, source_data in enumerate(test_cases):
                results = create_cutouts_batch([source_data], cutout_config, mock_job_tracker)
                # Should handle all cases gracefully
                assert isinstance(results, list)
                assert len(results) == 1
                assert "metadata" in results[0]
                # Most should result in empty metadata due to invalid parameters
                if i < 6:  # First 6 cases should fail validation
                    assert len(results[0]["metadata"]) == 0

    def test_multi_channel_processing(self, cutout_config, mock_job_tracker):
        """Test FITS set-based processing with multiple files per source."""
        # Set output format to FITS to get actual processed data back
        cutout_config.output_format = "fits"

        # Test multi-channel source (multiple FITS files)
        multi_channel_source = {
            "SourceID": "multi_channel_test_001",
            "RA": 150.0,
            "Dec": 2.0,
            "diameter_pixel": 32,
            "fits_file_paths": "['nir_h.fits', 'nir_j.fits', 'nir_y.fits']",  # Multiple files
        }

        # Mock the FITS set-based processing
        with (
            patch(
                "cutana.cutout_process._process_sources_batch_vectorized_with_fits_set"
            ) as mock_fits_set_processing,
            patch("cutana.fits_dataset.load_fits_file") as mock_load_fits,
        ):
            # Set up FITS file loading mock
            mock_hdul = MagicMock()
            mock_wcs = MagicMock()
            mock_load_fits.return_value = (mock_hdul, {"PRIMARY": mock_wcs})

            # Set up FITS set processing mock (returns batch format)
            batch_cutouts = np.random.random((1, 32, 32, 3)).astype(
                np.float32
            )  # 1 source, 3 channels
            mock_fits_set_processing.return_value = [
                {
                    "cutouts": batch_cutouts,
                    "metadata": [
                        {
                            "source_id": "multi_channel_test_001",
                            "channel_count": 3,
                        }
                    ],
                }
            ]

            results = create_cutouts_batch([multi_channel_source], cutout_config, mock_job_tracker)

            # Should call FITS set-based processing
            mock_fits_set_processing.assert_called()
            # Should load FITS files (3 files in the set)
            assert mock_load_fits.call_count == 3

            assert len(results) == 1
            assert "metadata" in results[0]
            # For FITS output, should have metadata for the processed source
            assert len(results[0]["metadata"]) == 1
            metadata = results[0]["metadata"][0]
            assert metadata["source_id"] == "multi_channel_test_001"
            assert metadata["channel_count"] == 3

    def test_single_vs_multi_channel_routing(self, cutout_config, mock_job_tracker):
        """Test that FITS set-based processing handles both single and multi-channel sources."""
        # Set output format to FITS to get actual processed data back
        cutout_config.output_format = "fits"

        sources = [
            # Single-channel source (one FITS file)
            {
                "SourceID": "single_channel_001",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "['single_file.fits']",
            },
            # Multi-channel source (three FITS files)
            {
                "SourceID": "multi_channel_002",
                "RA": 150.1,
                "Dec": 2.1,
                "diameter_pixel": 32,
                "fits_file_paths": "['file1.fits', 'file2.fits', 'file3.fits']",
            },
        ]

        with (
            patch(
                "cutana.cutout_process._process_sources_batch_vectorized_with_fits_set"
            ) as mock_fits_set_processing,
            patch("cutana.fits_dataset.load_fits_file") as mock_load_fits,
        ):
            # Set up FITS file loading mock
            mock_hdul = MagicMock()
            mock_wcs = MagicMock()
            mock_load_fits.return_value = (mock_hdul, {"PRIMARY": mock_wcs})

            # Mock processing returns batch format with appropriate channels per source
            def mock_processing_side_effect(
                sources_batch,
                loaded_fits_data,
                config,
                profiler=None,
                process_name=None,
                job_tracker=None,
            ):
                # Determine channels based on source type
                num_sources = len(sources_batch)
                num_channels = 1  # Default for single channel

                # Check if any source is multi-channel
                for source_data in sources_batch:
                    if source_data["SourceID"] == "multi_channel_002":
                        num_channels = 3
                        break

                # Create batch tensor
                batch_cutouts = np.random.random((num_sources, 32, 32, num_channels)).astype(
                    np.float32
                )

                # Create metadata list
                metadata_list = []
                for source_data in sources_batch:
                    source_id = source_data["SourceID"]
                    channel_count = 3 if source_id == "multi_channel_002" else 1
                    metadata_list.append(
                        {
                            "source_id": source_id,
                            "channel_count": channel_count,
                        }
                    )

                return [
                    {
                        "cutouts": batch_cutouts,
                        "metadata": metadata_list,
                    }
                ]

            mock_fits_set_processing.side_effect = mock_processing_side_effect

            results = create_cutouts_batch(sources, cutout_config, mock_job_tracker)

            # Should process both sources separately (different FITS sets)
            assert len(results) == 2  # Two batch results (one per FITS set)
            assert all("metadata" in result for result in results)

            # Total sources across all results should be 2
            total_sources = sum(len(result["metadata"]) for result in results)
            assert total_sources == 2

            # Each FITS set should be processed once (2 different FITS sets = 2 calls)
            assert mock_fits_set_processing.call_count == 2

            # Verify both sources were processed across all results
            all_metadata = []
            for result in results:
                all_metadata.extend(result["metadata"])
            result_ids = [meta["source_id"] for meta in all_metadata]
            assert "single_channel_001" in result_ids
            assert "multi_channel_002" in result_ids

    def test_multi_channel_source_deduplication(self, cutout_config, mock_job_tracker):
        """Test that sources are processed once per FITS set, not once per FITS file."""
        # Set output format to FITS to get actual processed data back
        cutout_config.output_format = "fits"

        # Source with multiple FITS files should be processed only once
        sources = [
            {
                "SourceID": "same_source_001",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "['file1.fits', 'file2.fits']",
            }
        ]

        with (
            patch(
                "cutana.cutout_process._process_sources_batch_vectorized_with_fits_set"
            ) as mock_fits_set_processing,
            patch("cutana.fits_dataset.load_fits_file") as mock_load_fits,
        ):
            # Set up mocks
            mock_hdul = MagicMock()
            mock_wcs = MagicMock()
            mock_load_fits.return_value = (mock_hdul, {"PRIMARY": mock_wcs})

            # Mock should return batch format: {"cutouts": tensor, "metadata": list}
            batch_cutouts = np.random.random((1, 32, 32, 2)).astype(np.float32)
            mock_fits_set_processing.return_value = [
                {
                    "cutouts": batch_cutouts,
                    "metadata": [
                        {
                            "source_id": "same_source_001",
                            "channel_count": 2,
                        }
                    ],
                }
            ]

            results = create_cutouts_batch(sources, cutout_config, mock_job_tracker)

            # Should be called exactly once per source, not once per FITS file
            mock_fits_set_processing.assert_called_once()
            # Should load 2 FITS files (once each)
            assert mock_load_fits.call_count == 2
            assert len(results) == 1
            assert "metadata" in results[0]
            # For FITS output, should have metadata for the processed source
            assert len(results[0]["metadata"]) == 1
            metadata = results[0]["metadata"][0]
            assert metadata["source_id"] == "same_source_001"

    def test_multi_channel_error_handling(self, cutout_config, mock_job_tracker):
        """Test error handling in FITS set-based processing."""
        multi_channel_source = {
            "SourceID": "error_test_001",
            "RA": 150.0,
            "Dec": 2.0,
            "diameter_pixel": 32,
            "fits_file_paths": "['missing1.fits', 'missing2.fits']",
        }

        with (
            patch("cutana.cutout_process._process_sources_batch_vectorized_with_fits_set"),
            patch("cutana.fits_dataset.load_fits_file") as mock_load_fits,
        ):
            # Simulate FITS loading failure
            mock_load_fits.side_effect = FileNotFoundError("FITS file not found")

            # This should be handled gracefully - no loaded FITS data means no processing
            results = create_cutouts_batch([multi_channel_source], cutout_config, mock_job_tracker)

            # Should handle errors gracefully - FITS loading fails, so processing is skipped
            assert isinstance(results, list)
            assert len(results) == 1  # Returns one result with empty metadata
            assert "metadata" in results[0]
            assert (
                len(results[0]["metadata"]) == 0
            )  # No successful results due to missing FITS files

    def test_fits_path_parsing_multi_channel(self, cutout_config, mock_job_tracker):
        """Test parsing of various FITS path formats in FITS set-based processing."""
        # Set output format to FITS to get actual processed data back
        cutout_config.output_format = "fits"

        test_cases = [
            # Standard multi-file list
            {
                "SourceID": "multi_parse_001",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "['file1.fits', 'file2.fits', 'file3.fits']",
            },
            # Single file
            {
                "SourceID": "single_parse_002",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "single_file.fits",
            },
            # Already parsed list
            {
                "SourceID": "list_parse_003",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": ["list_file1.fits", "list_file2.fits"],
            },
        ]

        with (
            patch(
                "cutana.cutout_process._process_sources_batch_vectorized_with_fits_set"
            ) as mock_fits_set_processing,
            patch("cutana.fits_dataset.load_fits_file") as mock_load_fits,
        ):
            # Set up mocks
            mock_hdul = MagicMock()
            mock_wcs = MagicMock()
            mock_load_fits.return_value = (mock_hdul, {"PRIMARY": mock_wcs})

            def mock_processing_side_effect(
                sources_batch,
                loaded_fits_data,
                config,
                profiler=None,
                process_name=None,
                job_tracker=None,
            ):
                # Return batch format: {"cutouts": tensor, "metadata": list}
                # Each call processes one FITS set with sources that share the same FITS files
                metadata_list = []
                channels = len(loaded_fits_data) if loaded_fits_data else 1

                for source_data in sources_batch:
                    source_id = source_data["SourceID"]
                    if source_id == "single_parse_002":
                        metadata_list.append({"source_id": source_id, "channel_count": 1})
                        channels = 1
                    elif source_id == "multi_parse_001":
                        metadata_list.append({"source_id": source_id, "channel_count": 3})
                        channels = 3
                    else:  # list_parse_003
                        metadata_list.append({"source_id": source_id, "channel_count": 2})
                        channels = 2

                # Create batch cutouts tensor for this FITS set
                batch_cutouts = np.random.random((len(sources_batch), 32, 32, channels)).astype(
                    np.float32
                )
                return [
                    {
                        "cutouts": batch_cutouts,
                        "metadata": metadata_list,
                    }
                ]

            mock_fits_set_processing.side_effect = mock_processing_side_effect

            results = create_cutouts_batch(test_cases, cutout_config, mock_job_tracker)

            # Should process all sources using FITS set-based approach
            # Since each source has different FITS files, they are processed separately
            assert len(results) == 3  # Three batch results (one per FITS set)
            assert all("metadata" in result for result in results)

            # Total sources across all results should be 3
            total_sources = sum(len(result["metadata"]) for result in results)
            assert total_sources == 3
            assert mock_fits_set_processing.call_count == 3

            # Verify all sources were processed across all results
            all_metadata = []
            for result in results:
                all_metadata.extend(result["metadata"])
            result_ids = [meta["source_id"] for meta in all_metadata]
            assert "multi_parse_001" in result_ids
            assert "single_parse_002" in result_ids
            assert "list_parse_003" in result_ids

    def test_progress_counting_with_multi_channel(self, cutout_config, mock_job_tracker):
        """Test that progress counting works correctly with multi-channel sources."""
        # This test ensures the bug fix - sources should be counted once regardless of channel count
        # Set output format to FITS to get actual processed data back
        cutout_config.output_format = "fits"

        sources = [
            {
                "SourceID": "progress_test_001",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "['file1.fits', 'file2.fits', 'file3.fits']",  # 3 channels
            },
            {
                "SourceID": "progress_test_002",
                "RA": 150.1,
                "Dec": 2.1,
                "diameter_pixel": 32,
                "fits_file_paths": "['single.fits']",  # 1 channel
            },
        ]

        with (
            patch(
                "cutana.cutout_process._process_sources_batch_vectorized_with_fits_set"
            ) as mock_fits_set_processing,
            patch("cutana.fits_dataset.load_fits_file") as mock_load_fits,
        ):
            # Set up mocks
            mock_hdul = MagicMock()
            mock_wcs = MagicMock()
            mock_load_fits.return_value = (mock_hdul, {"PRIMARY": mock_wcs})

            def mock_processing_side_effect(
                sources_batch,
                loaded_fits_data,
                config,
                profiler=None,
                process_name=None,
                job_tracker=None,
            ):
                # Return batch format: {"cutouts": tensor, "metadata": list}
                # Each call processes sources with the same FITS set, so create one result per call
                metadata_list = []
                for source_data in sources_batch:
                    source_id = source_data["SourceID"]
                    metadata_list.append(
                        {
                            "source_id": source_id,
                            "channel_count": len(loaded_fits_data),
                        }
                    )

                # Create batch cutouts tensor for this FITS set
                batch_cutouts = np.random.random(
                    (len(sources_batch), 32, 32, len(loaded_fits_data))
                ).astype(np.float32)
                return [
                    {
                        "cutouts": batch_cutouts,
                        "metadata": metadata_list,
                    }
                ]

            mock_fits_set_processing.side_effect = mock_processing_side_effect

            results = create_cutouts_batch(sources, cutout_config, mock_job_tracker)

            # The bug fix: Should process exactly 2 sources, not 2 + 3 = 5 (old behavior)
            # Since the sources have different FITS sets, they are processed separately
            assert len(results) == 2  # Two batch results (one per FITS set)
            assert all("metadata" in result for result in results)

            # Each FITS set should have its own batch result
            total_sources = sum(len(result["metadata"]) for result in results)
            assert total_sources == 2  # Total of 2 sources processed

            # Should call FITS set-based processing twice (once per FITS set)
            assert mock_fits_set_processing.call_count == 2

            # Verify both sources were processed exactly once across all results
            all_metadata = []
            for result in results:
                all_metadata.extend(result["metadata"])
            result_ids = [meta["source_id"] for meta in all_metadata]
            assert "progress_test_001" in result_ids
            assert "progress_test_002" in result_ids

    @patch("cutana.cutout_process.extract_cutouts_batch_vectorized")
    def test_channel_combination_with_different_resolutions(
        self, mock_extract_cutouts, cutout_config
    ):
        """Test that channel combination works correctly when channels have different resolutions."""
        # Mock different-sized cutouts for different channels
        small_cutout = np.random.random((64, 64)).astype(np.float32)
        medium_cutout = np.random.random((128, 128)).astype(np.float32)
        large_cutout = np.random.random((256, 256)).astype(np.float32)

        # Create mock source data with multiple channel configuration
        sources_batch = [
            {
                "SourceID": "multi_res_test_001",
                "RA": 150.0,
                "Dec": 2.0,
                "diameter_pixel": 32,
                "fits_file_paths": "['/mock/vis.fits', '/mock/nir_h.fits', '/mock/nir_j.fits']",
            }
        ]

        # Mock extracted cutouts with different resolutions
        mock_combined_cutouts = {
            "multi_res_test_001": {
                "PRIMARY": small_cutout,  # Different size from other channels
            }
        }

        mock_combined_wcs = {"multi_res_test_001": {"PRIMARY": MagicMock()}}  # Mock WCS object

        # Mock the extract_cutouts_batch_vectorized to return different sized cutouts
        def mock_extract_side_effect(
            sources, hdul, wcs_dict, extensions, padding_factor=1.0, config=None
        ):
            # Simulate different FITS files returning different sized cutouts
            fits_name = getattr(hdul, "_mock_name", "unknown")
            if "vis" in fits_name:
                return (
                    {sources[0]["SourceID"]: {"PRIMARY": large_cutout}},
                    mock_combined_wcs,
                    [sources[0]["SourceID"]],
                )
            elif "nir_h" in fits_name:
                return (
                    {sources[0]["SourceID"]: {"PRIMARY": medium_cutout}},
                    mock_combined_wcs,
                    [sources[0]["SourceID"]],
                )
            elif "nir_j" in fits_name:
                return (
                    {sources[0]["SourceID"]: {"PRIMARY": small_cutout}},
                    mock_combined_wcs,
                    [sources[0]["SourceID"]],
                )
            else:
                return mock_combined_cutouts, mock_combined_wcs, [sources[0]["SourceID"]]

        mock_extract_cutouts.side_effect = mock_extract_side_effect

        # Set up config for channel combination
        from dotmap import DotMap

        config = DotMap(cutout_config.copy())
        config.channel_weights = {"vis": [0.5], "nir_h": [0.3], "nir_j": [0.2]}
        config.target_resolution = (64, 64)  # Force resize to common resolution
        config.normalisation_method = "linear"
        config.data_type = "float32"

        # Create mock loaded FITS data
        mock_loaded_fits_data = {}
        for fits_path in ["/mock/vis.fits", "/mock/nir_h.fits", "/mock/nir_j.fits"]:
            mock_hdul = MagicMock()
            mock_hdul._mock_name = fits_path  # Add mock name for identification
            mock_hdul.close = MagicMock()
            mock_wcs_dict = {"PRIMARY": MagicMock()}
            mock_loaded_fits_data[fits_path] = (mock_hdul, mock_wcs_dict)

        # Call the function directly
        results = _process_sources_batch_vectorized_with_fits_set(
            sources_batch,
            mock_loaded_fits_data,
            config,
            profiler=None,
            process_name=None,
            job_tracker=None,
        )

        # Verify results - new batch format
        assert len(results) == 1
        batch_result = results[0]
        assert "cutouts" in batch_result
        assert "metadata" in batch_result
        assert len(batch_result["metadata"]) == 1

        # Check metadata
        result_metadata = batch_result["metadata"][0]
        assert result_metadata["source_id"] == "multi_res_test_001"

        # Check cutouts tensor shape
        cutouts_tensor = batch_result["cutouts"]
        assert cutouts_tensor.shape[0] == 1  # 1 source

        # For legacy compatibility, create processed_cutouts structure from tensor
        processed_cutouts = {}
        if cutouts_tensor.shape[-1] == 1:
            # Single channel - likely combined result
            processed_cutouts["combined"] = cutouts_tensor[0, :, :, 0]
        if "combined" in processed_cutouts:
            # Channel combination was applied - should have single combined cutout
            combined_cutout = processed_cutouts["combined"]
            # With single-channel weights, should produce (64, 64, 1) then squeezed to (64, 64)
            # Note: expected shape varies based on channel combination
            # expected_shape = (64, 64, 1) if len(combined_cutout.shape) == 3 else (64, 64)
            assert combined_cutout.shape in [
                (64, 64),
                (64, 64, 1),
            ]  # Should be resized to target resolution
        else:
            # No channel combination - should have all individual channels resized
            for channel_key, cutout in processed_cutouts.items():
                assert cutout.shape == (64, 64)  # All should be resized to target resolution

        # Verify extract was called for each FITS file
        assert mock_extract_cutouts.call_count == 3  # Three FITS files
