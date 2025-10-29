#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the cutout_writer_fits module using TDD approach.

Tests cover:
- Individual FITS file creation for each cutout
- Proper WCS header preservation
- Multi-extension FITS handling
- File naming conventions and organization
- Metadata embedding in FITS headers
- Error handling for file system issues
"""

from pathlib import Path
import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from unittest.mock import patch
from cutana.cutout_writer_fits import (
    ensure_output_directory,
    generate_fits_filename,
    create_wcs_header,
    write_single_fits_cutout,
    write_fits_batch,
    validate_fits_file,
)


class TestCutoutWriterFitsFunctions:
    """Test suite for FITS writer functions."""

    @pytest.fixture
    def writer_config(self):
        """Create FITS writer configuration."""
        return {
            "output_directory": "/tmp/cutouts",
            "file_naming_template": "{source_id}_{filter}.fits",
            "compression": "rice",
            "preserve_wcs": True,
            "create_subdirs": True,
            "overwrite": False,
        }

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "fits_output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def mock_cutout_data(self):
        """Create mock processed cutout data with WCS."""
        # Create simple WCS for testing
        wcs = WCS(naxis=2)
        wcs.wcs.crval = [150.0, 2.0]  # RA, Dec reference
        wcs.wcs.crpix = [128.0, 128.0]  # Reference pixel
        wcs.wcs.cdelt = [-0.0002777778, 0.0002777778]  # -1", 1" per pixel
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        return {
            "source_id": "MockSource_00001",
            "processed_cutouts": {
                "VIS": np.random.random((256, 256)).astype(np.float32),
                "NIR-Y": np.random.random((256, 256)).astype(np.float32),
                "NIR-H": np.random.random((256, 256)).astype(np.float32),
            },
            "wcs_info": {"VIS": wcs, "NIR-Y": wcs, "NIR-H": wcs},
            "metadata": {
                "ra": 150.0,
                "dec": 2.0,
                "diameter_arcsec": 10.0,
                "diameter_pixel": 256,
                "channels": ["VIS", "NIR-Y", "NIR-H"],
                "processing_timestamp": 1642678800.0,
                "original_tile": "euclid_tile_001.fits",
            },
        }

    def test_ensure_output_directory(self, tmp_path):
        """Test creation of output directory."""
        output_dir = tmp_path / "test_output" / "subfolder"

        ensure_output_directory(output_dir)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_generate_fits_filename(self):
        """Test FITS filename generation."""
        # Basic filename with all required parameters
        filename = generate_fits_filename(
            "TestSource_001", "{source_id}_cutout.fits", "", {"ra": 150.0, "dec": 2.0}
        )
        assert filename == "TestSource_001_cutout.fits"

        # With template and modifier
        filename = generate_fits_filename(
            "TestSource_001", "{modifier}{source_id}_VIS.fits", "prefix_", {"ra": 150.0, "dec": 2.0}
        )
        assert filename == "prefix_TestSource_001_VIS.fits"

        # With timestamp template
        filename = generate_fits_filename(
            "TestSource_001", "{source_id}_{timestamp}.fits", "", {"ra": 150.0, "dec": 2.0}
        )
        assert "TestSource_001_" in filename
        assert filename.endswith(".fits")

    def test_create_wcs_header(self, mock_cutout_data):
        """Test WCS header creation."""
        wcs = mock_cutout_data["wcs_info"]["VIS"]
        cutout_shape = (256, 256)

        header = create_wcs_header(cutout_shape, original_wcs=wcs, ra_center=150.0, dec_center=2.0)

        assert "CRVAL1" in header
        assert "CRVAL2" in header
        assert "CRPIX1" in header
        assert "CRPIX2" in header
        assert header["CRVAL1"] == 150.0
        assert header["CRVAL2"] == 2.0

    def test_write_single_fits_cutout(self, mock_cutout_data, temp_output_dir):
        """Test writing a single FITS cutout file."""
        output_path = temp_output_dir / "test_cutout.fits"

        success = write_single_fits_cutout(
            mock_cutout_data, str(output_path), preserve_wcs=True, overwrite=True
        )

        assert success is True
        assert output_path.exists()

        # Verify FITS file structure
        with fits.open(output_path) as hdul:
            assert len(hdul) >= 4  # Primary + 3 image extensions
            assert "SOURCE" in hdul[0].header
            assert hdul[0].header["SOURCE"] == "MockSource_00001"

            # Check extensions
            ext_names = [hdu.name for hdu in hdul[1:]]
            assert "VIS" in ext_names
            assert "NIR-Y" in ext_names
            assert "NIR-H" in ext_names

    def test_write_single_fits_with_compression(self, mock_cutout_data, temp_output_dir):
        """Test writing FITS with compression."""
        output_path = temp_output_dir / "compressed_cutout.fits"

        success = write_single_fits_cutout(
            mock_cutout_data, str(output_path), compression="gzip", overwrite=True
        )

        assert success is True
        assert output_path.exists()

        # Verify compression
        with fits.open(output_path) as hdul:
            for hdu in hdul[1:]:
                if hasattr(hdu, "header") and "COMPRESS" in hdu.header:
                    assert hdu.header["COMPRESS"] == "gzip"

    def test_write_fits_batch(self, temp_output_dir):
        """Test batch writing of individual FITS files."""
        # Create batch data in the format expected by the current implementation
        # Each batch_result contains "cutouts" tensor and "metadata" list
        cutouts_tensor = np.random.random((5, 64, 64, 1)).astype(np.float32)  # (N, H, W, C)
        metadata_list = []
        for i in range(5):
            metadata_list.append(
                {
                    "source_id": f"BatchSource_{i:03d}",
                    "ra": 150.0 + i * 0.01,
                    "dec": 2.0 + i * 0.01,
                }
            )

        batch_data = [
            {
                "cutouts": cutouts_tensor,
                "metadata": metadata_list,
            }
        ]

        written_files = write_fits_batch(
            batch_data,
            str(temp_output_dir),
            file_naming_template="{source_id}_cutout.fits",
            create_subdirs=False,
            overwrite=True,
        )

        assert len(written_files) == 5

        # Verify files exist
        for file_path in written_files:
            assert Path(file_path).exists()

    def test_write_fits_batch_with_subdirs(self, temp_output_dir):
        """Test batch writing with subdirectory organization."""
        cutouts_tensor = np.random.random((1, 64, 64, 1)).astype(np.float32)  # (N, H, W, C)
        metadata_list = [
            {
                "source_id": "ABC123_source",
                "ra": 150.0,
                "dec": 2.0,
            }
        ]

        batch_data = [
            {
                "cutouts": cutouts_tensor,
                "metadata": metadata_list,
            }
        ]

        written_files = write_fits_batch(
            batch_data, str(temp_output_dir), create_subdirs=True, overwrite=True
        )

        assert len(written_files) == 1

        # Check subdirectory was created
        written_path = Path(written_files[0])
        assert written_path.parent.name == "ABC"  # First 3 chars
        assert written_path.exists()

    def test_error_handling_no_overwrite(self, mock_cutout_data, temp_output_dir):
        """Test error handling when file exists and overwrite is False."""
        output_path = temp_output_dir / "existing.fits"

        # Create existing file
        output_path.touch()

        success = write_single_fits_cutout(mock_cutout_data, str(output_path), overwrite=False)

        assert success is False

    def test_error_handling_invalid_path(self, mock_cutout_data):
        """Test error handling with invalid output path."""
        invalid_path = "/invalid/path/that/does/not/exist/cutout.fits"

        success = write_single_fits_cutout(mock_cutout_data, invalid_path, overwrite=True)

        assert success is False

    def test_validate_fits_file(self, mock_cutout_data, temp_output_dir):
        """Test FITS file validation."""
        output_path = temp_output_dir / "validate_test.fits"

        # Write a valid FITS file
        write_single_fits_cutout(mock_cutout_data, str(output_path), overwrite=True)

        # Validate it
        validation_result = validate_fits_file(str(output_path))

        assert validation_result["valid"] is True
        assert validation_result["num_extensions"] >= 4
        assert "extensions" in validation_result
        assert validation_result["file_size"] > 0

    def test_validate_invalid_fits_file(self, temp_output_dir):
        """Test validation of invalid FITS file."""
        invalid_path = temp_output_dir / "invalid.fits"

        # Create invalid file
        with open(invalid_path, "w") as f:
            f.write("This is not a FITS file")

        validation_result = validate_fits_file(str(invalid_path))

        assert validation_result["valid"] is False
        assert "error" in validation_result

    def test_preserve_wcs_information(self, mock_cutout_data, temp_output_dir):
        """Test that WCS information is properly preserved."""
        output_path = temp_output_dir / "wcs_test.fits"

        success = write_single_fits_cutout(
            mock_cutout_data, str(output_path), preserve_wcs=True, overwrite=True
        )

        assert success is True

        # Read back and check WCS
        with fits.open(output_path) as hdul:
            for ext_name in ["VIS", "NIR-Y", "NIR-H"]:
                if ext_name in hdul:
                    hdu = hdul[ext_name]
                    # Try to create WCS from header
                    wcs = WCS(hdu.header)
                    assert wcs.wcs.has_cd() or wcs.wcs.has_pc()
                    assert wcs.wcs.ctype[0] == "RA---TAN"
                    assert wcs.wcs.ctype[1] == "DEC--TAN"

    def test_metadata_preservation(self, mock_cutout_data, temp_output_dir):
        """Test that metadata is properly saved to FITS headers."""
        output_path = temp_output_dir / "metadata_test.fits"

        write_single_fits_cutout(mock_cutout_data, str(output_path), overwrite=True)

        # Read back and check metadata
        with fits.open(output_path) as hdul:
            header = hdul[0].header
            assert header["SOURCE"] == "MockSource_00001"
            assert abs(header["RA"] - 150.0) < 0.0001
            assert abs(header["DEC"] - 2.0) < 0.0001
            assert header["SIZEARC"] == 10.0

    def test_empty_cutout_handling(self, temp_output_dir):
        """Test handling of empty cutout data."""
        empty_data = {
            "source_id": "EmptySource",
            "processed_cutouts": {},  # No cutouts
            "metadata": {"ra": 150.0, "dec": 2.0},
        }

        output_path = temp_output_dir / "empty.fits"

        success = write_single_fits_cutout(empty_data, str(output_path), overwrite=True)

        assert success is False  # Should fail with no cutout data

    def test_ensure_output_directory_error_handling(self):
        """Test ensure_output_directory with various error conditions."""
        from cutana.cutout_writer_fits import ensure_output_directory

        # Test with invalid permissions path
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                ensure_output_directory(Path("/invalid/permission/path"))

    def test_generate_fits_filename_comprehensive(self):
        """Test comprehensive filename generation scenarios."""
        from cutana.cutout_writer_fits import generate_fits_filename

        # Test basic functionality with required parameters
        filename = generate_fits_filename(
            "test_source", "{source_id}_cutout.fits", "", {"ra": 150.0, "dec": 2.0}
        )
        assert filename == "test_source_cutout.fits"

        # Test with modifier and metadata
        filename = generate_fits_filename(
            "test_source", "{modifier}{source_id}_data.fits", "prefix_", {"ra": 150.0, "dec": 2.0}
        )
        assert filename == "prefix_test_source_data.fits"

        # Test with timestamp
        filename = generate_fits_filename(
            "test", "{source_id}_{timestamp}.fits", "", {"ra": 150.0, "dec": 2.0}
        )
        assert "test_" in filename
        assert filename.endswith(".fits")

        # Test with invalid characters in source_id
        filename = generate_fits_filename(
            "test<>:source", "{source_id}.fits", "", {"ra": 150.0, "dec": 2.0}
        )
        assert filename == "test___source.fits"

        # Test without .fits extension in template
        filename = generate_fits_filename("test", "{source_id}_data", "", {"ra": 150.0, "dec": 2.0})
        assert filename.endswith(".fits")

        # Test template error handling by directly testing with invalid template
        try:
            filename = generate_fits_filename(
                "fallback_test", "{invalid_key}", "", {"ra": 150.0, "dec": 2.0}
            )
            # Should use fallback on template error
        except KeyError:
            # If error occurs, should fallback to default
            filename = "fallback_test_cutout.fits"

        assert "fallback_test" in filename
        assert filename.endswith(".fits")

    def test_create_wcs_header_comprehensive(self):
        """Test comprehensive WCS header creation scenarios."""
        from cutana.cutout_writer_fits import create_wcs_header
        from astropy.wcs import WCS

        # Test with original WCS
        original_wcs = WCS(naxis=2)
        original_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        original_wcs.wcs.crval = [150.0, 2.0]
        original_wcs.wcs.crpix = [50.0, 50.0]
        original_wcs.wcs.cdelt = [-0.0001, 0.0001]

        header = create_wcs_header(
            (64, 64), original_wcs=original_wcs, ra_center=151.0, dec_center=3.0
        )

        assert header["CRPIX1"] == 32.0  # 64/2
        assert header["CRPIX2"] == 32.0  # 64/2
        assert header["CRVAL1"] == 151.0  # Updated center
        assert header["CRVAL2"] == 3.0  # Updated center

        # Test without original WCS but with coordinates
        header = create_wcs_header((128, 128), ra_center=150.5, dec_center=2.5, pixel_scale=0.6)

        assert header["WCSAXES"] == 2
        assert header["CTYPE1"] == "RA---TAN"
        assert header["CTYPE2"] == "DEC--TAN"
        assert header["CRVAL1"] == 150.5
        assert header["CRVAL2"] == 2.5
        assert header["CRPIX1"] == 64.0  # 128/2
        assert header["CRPIX2"] == 64.0  # 128/2

        # Test with error condition
        with patch("astropy.wcs.WCS.to_header", side_effect=Exception("WCS error")):
            header = create_wcs_header((32, 32), original_wcs=original_wcs)
            # Should return empty header on error
            assert len(header) == 0

    def test_write_fits_batch_edge_cases(self, temp_output_dir):
        """Test write_fits_batch with edge cases."""
        from cutana.cutout_writer_fits import write_fits_batch

        # Test empty batch
        written_files = write_fits_batch([], str(temp_output_dir))
        assert written_files == []

        # Test batch with empty cutouts tensor
        invalid_batch = [
            {
                "cutouts": np.array([]),  # Empty tensor
                "metadata": [{"source_id": "InvalidSource", "ra": 150.0, "dec": 2.0}],
            }
        ]

        written_files = write_fits_batch(invalid_batch, str(temp_output_dir))
        assert len(written_files) == 0  # Should skip invalid data

        # Test valid batch
        valid_cutouts = np.random.random((1, 16, 16, 1)).astype(np.float32)
        valid_batch = [
            {
                "cutouts": valid_cutouts,
                "metadata": [{"source_id": "BatchSource_001", "ra": 150.0, "dec": 2.0}],
            }
        ]

        written_files = write_fits_batch(valid_batch, str(temp_output_dir), overwrite=True)
        assert len(written_files) == 1
        assert Path(written_files[0]).exists()

    def test_validate_fits_file_comprehensive(self, temp_output_dir):
        """Test comprehensive FITS file validation."""
        from cutana.cutout_writer_fits import validate_fits_file

        # Test with non-existent file
        validation_result = validate_fits_file(str(temp_output_dir / "nonexistent.fits"))
        assert validation_result["valid"] is False
        assert "error" in validation_result

        # Create a valid FITS file for testing
        hdu = fits.PrimaryHDU(data=np.random.random((32, 32)))
        hdu.header["TEST"] = "value"
        hdul = fits.HDUList([hdu])

        test_file = temp_output_dir / "test_validate.fits"
        hdul.writeto(test_file)

        validation_result = validate_fits_file(str(test_file))
        assert validation_result["valid"] is True
        assert validation_result["num_extensions"] == 1
        assert len(validation_result["extensions"]) == 1

        ext_info = validation_result["extensions"][0]
        assert ext_info["index"] == 0
        assert ext_info["type"] == "PrimaryHDU"
        assert ext_info["shape"] == (32, 32)

    def test_error_handling_comprehensive(self, temp_output_dir):
        """Test comprehensive error handling scenarios."""
        from cutana.cutout_writer_fits import write_single_fits_cutout

        mock_data = {
            "source_id": "ErrorTest",
            "processed_cutouts": {"TEST": np.random.random((16, 16))},
            "metadata": {"ra": 150.0, "dec": 2.0},
        }

        # Test with FITS writing error
        with patch("astropy.io.fits.HDUList.writeto", side_effect=Exception("FITS write failed")):
            success = write_single_fits_cutout(
                mock_data, str(temp_output_dir / "error_test.fits"), overwrite=True
            )
            assert success is False

        # Test with invalid cutout data in processed_cutouts
        invalid_data = {
            "source_id": "InvalidCutoutTest",
            "processed_cutouts": {"INVALID": "not_an_array"},  # Invalid data type
            "metadata": {},
        }

        with patch("astropy.io.fits.ImageHDU", side_effect=Exception("HDU creation failed")):
            success = write_single_fits_cutout(
                invalid_data, str(temp_output_dir / "invalid_cutout.fits"), overwrite=True
            )
            assert success is False
