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
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from dotmap import DotMap

from cutana import cutout_writer_fits
from cutana.cutout_extraction import extract_cutouts_vectorized_from_extension
from cutana.cutout_writer_fits import (
    create_wcs_header,
    ensure_output_directory,
    generate_fits_filename,
    write_fits_batch,
    write_single_fits_cutout,
)


@pytest.fixture(autouse=True)
def _clear_wcs_header_cache():
    """Isolate the module-level WCS header cache between tests.

    ``cutout_writer_fits._wcs_header_cache`` is keyed on ``id(wcs)``; across tests a
    freed WCS object's id can be reused, returning a stale cached header. Clearing it
    per test keeps WCS assertions deterministic regardless of execution order.
    """
    cutout_writer_fits._wcs_header_cache.clear()
    yield
    cutout_writer_fits._wcs_header_cache.clear()


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
                "tile": "euclid_tile_001.fits",
                # Extraction origin/size threaded from cutout_extraction (unresized 256 px).
                "extraction_origin_x": 0,
                "extraction_origin_y": 0,
                "extraction_size": 256,
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
        """CRVAL is inherited from the parent tile, not re-tangented at the source."""
        wcs = mock_cutout_data["wcs_info"]["VIS"]  # parent CRVAL = [150.0, 2.0]
        cutout_shape = (256, 256)

        # Deliberately offset the source from the parent CRVAL so a re-tangenting
        # regression (CRVAL <- source) would be caught. The extraction origin/size are
        # threaded in (as they are from cutout_extraction in the real pipeline).
        header = create_wcs_header(
            cutout_shape,
            original_wcs=wcs,
            ra_source=150.05,
            dec_source=2.05,
            extraction_origin_x=0,
            extraction_origin_y=0,
            extraction_size=256,
        )

        assert "CRPIX1" in header
        assert "CRPIX2" in header
        # CRVAL must stay at the parent tile reference, NOT the source position.
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

    def test_write_single_fits_unit_and_consvflx_headers(self, mock_cutout_data, temp_output_dir):
        """UNIT and CONSVFLX from cutout_data are written to the primary header."""
        output_path = temp_output_dir / "unit_header.fits"

        cutout_data = {**mock_cutout_data, "unit": "Jy", "conserved_flux": True}
        success = write_single_fits_cutout(cutout_data, str(output_path), overwrite=True)

        assert success is True
        with fits.open(output_path) as hdul:
            assert hdul[0].header["UNIT"] == "Jy"
            assert bool(hdul[0].header["CONSVFLX"]) is True

    @pytest.mark.parametrize(
        ("flux_conserved_resizing", "apply_flux_conversion", "expected_unit", "expected_consvflx"),
        [
            (True, True, "Jy", True),
            (True, False, "OriginalUnit", True),
            (False, True, "approx Jy", False),
            (False, False, "approx OriginalUnit", False),
        ],
    )
    def test_write_fits_batch_unit_mapping(
        self,
        temp_output_dir,
        flux_conserved_resizing,
        apply_flux_conversion,
        expected_unit,
        expected_consvflx,
    ):
        """write_fits_batch maps flux config to the UNIT/CONSVFLX headers it writes."""
        cutouts_tensor = np.random.random((1, 32, 32, 1)).astype(np.float32)
        batch_data = [
            {
                "cutouts": cutouts_tensor,
                "metadata": [
                    {
                        "source_id": "UnitSource_001",
                        "ra": 150.0,
                        "dec": 2.0,
                        "tile": "euclid_tile_001.fits",
                    }
                ],
            }
        ]

        written_files = write_fits_batch(
            batch_data,
            str(temp_output_dir),
            config=DotMap(
                {
                    "do_only_cutout_extraction": False,
                    "flux_conserved_resizing": flux_conserved_resizing,
                    "apply_flux_conversion": apply_flux_conversion,
                }
            ),
            file_naming_template="{source_id}_cutout.fits",
            create_subdirs=False,
            overwrite=True,
        )

        assert len(written_files) == 1
        with fits.open(written_files[0]) as hdul:
            assert hdul[0].header["UNIT"] == expected_unit
            assert bool(hdul[0].header["CONSVFLX"]) == expected_consvflx

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
                    "tile": "euclid_tile_001.fits",
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
            config=DotMap({"do_only_cutout_extraction": False}),
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
                "tile": "euclid_tile_001.fits",
            }
        ]

        batch_data = [
            {
                "cutouts": cutouts_tensor,
                "metadata": metadata_list,
            }
        ]

        written_files = write_fits_batch(
            batch_data,
            str(temp_output_dir),
            config=DotMap({"do_only_cutout_extraction": False}),
            create_subdirs=True,
            overwrite=True,
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
        # Test with invalid permissions path
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                ensure_output_directory(Path("/invalid/permission/path"))

    def test_generate_fits_filename_comprehensive(self):
        """Test comprehensive filename generation scenarios."""
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
        # Test with original WCS
        original_wcs = WCS(naxis=2)
        original_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        original_wcs.wcs.crval = [150.0, 2.0]
        original_wcs.wcs.crpix = [50.0, 50.0]
        original_wcs.wcs.cdelt = [-0.0001, 0.0001]

        px, py = original_wcs.world_to_pixel_values(151.0, 3.0)
        x_min = int(px - 64 // 2)
        y_min = int(py - 64 // 2)
        header = create_wcs_header(
            (64, 64),
            original_wcs=original_wcs,
            ra_source=151.0,
            dec_source=3.0,
            extraction_origin_x=int(x_min),
            extraction_origin_y=int(y_min),
            extraction_size=64,
        )

        # CRVAL/CTYPE are inherited from the parent tile unchanged. The projection
        # is NOT re-tangented at the source position (doing so keeps the parent CD
        # matrix at the wrong tangent point and rotates the cutout frame).
        assert header["CRVAL1"] == 150.0  # parent CRVAL preserved
        assert header["CRVAL2"] == 2.0  # parent CRVAL preserved
        # CRPIX is shifted to the extraction origin so the cutout reproduces the
        # parent sky mapping exactly. Verify that round-trip agreement directly.
        cut_wcs = WCS(header)
        for cx, cy in [(0, 0), (63, 63), (10, 50)]:
            sky_parent = original_wcs.pixel_to_world_values(x_min + cx, y_min + cy)
            sky_cut = cut_wcs.pixel_to_world_values(cx, cy)
            assert np.allclose(sky_parent, sky_cut, atol=1e-10)

        # Test without original WCS but with coordinates
        header = create_wcs_header((128, 128), ra_source=150.5, dec_source=2.5, pixel_scale=0.6)

        assert header["WCSAXES"] == 2
        assert header["CTYPE1"] == "RA---TAN"
        assert header["CTYPE2"] == "DEC--TAN"
        assert header["CRVAL1"] == 150.5
        assert header["CRVAL2"] == 2.5
        assert header["CRPIX1"] == 64.5  # 128/2 + 0.5 (FITS 1-based center)
        assert header["CRPIX2"] == 64.5  # 128/2 + 0.5 (FITS 1-based center)

        # Test with error condition - use a new WCS object that hasn't been cached
        cutout_writer_fits._wcs_header_cache.clear()  # Clear cache so the mock will be invoked
        new_wcs = WCS(naxis=2)
        new_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        with patch("astropy.wcs.WCS.to_header", side_effect=Exception("WCS error")):
            header = create_wcs_header((32, 32), original_wcs=new_wcs)
            # Should return empty header on error
            assert len(header) == 0

    def test_write_fits_batch_edge_cases(self, temp_output_dir):
        """Test write_fits_batch with edge cases."""
        # Test empty batch
        written_files = write_fits_batch(
            [], str(temp_output_dir), config=DotMap({"do_only_cutout_extraction": False})
        )
        assert written_files == []

        # Test batch with empty cutouts tensor
        invalid_batch = [
            {
                "cutouts": np.array([]),  # Empty tensor
                "metadata": [{"source_id": "InvalidSource", "ra": 150.0, "dec": 2.0}],
            }
        ]

        written_files = write_fits_batch(
            invalid_batch,
            str(temp_output_dir),
            config=DotMap({"do_only_cutout_extraction": False}),
        )
        assert len(written_files) == 0  # Should skip invalid data

        # Test valid batch
        valid_cutouts = np.random.random((1, 16, 16, 1)).astype(np.float32)
        valid_batch = [
            {
                "cutouts": valid_cutouts,
                "metadata": [
                    {
                        "source_id": "BatchSource_001",
                        "ra": 150.0,
                        "dec": 2.0,
                        "tile": "euclid_tile_001.fits",
                    }
                ],
            }
        ]

        written_files = write_fits_batch(
            valid_batch,
            str(temp_output_dir),
            config=DotMap({"do_only_cutout_extraction": False}),
            overwrite=True,
        )
        assert len(written_files) == 1
        assert Path(written_files[0]).exists()

    def test_error_handling_comprehensive(self, temp_output_dir):
        """Test comprehensive error handling scenarios."""
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


class TestCutoutWcsFidelity:
    """Regression tests for cutout WCS fidelity against the parent tile.

    Guards against the re-tangenting bug where the cutout WCS set CRVAL to the
    source RA/Dec and CRPIX to the geometric centre while keeping the parent tile's
    CD matrix. That rotates the cutout frame by the meridian convergence between the
    tile centre and the source, giving a WCS error that GROWS with distance from the
    cutout centre (order ~1" at a few-arcmin FOV for sources far from the tile centre).

    The correct construction inherits the parent CRVAL/CD and only shifts CRPIX to the
    extraction origin, so every cutout pixel maps to the same sky position as the
    parent tile (to numerical precision).
    """

    # Declination near the pole: meridian convergence (~ tan(Dec)) is large there,
    # so the re-tangenting error the fix removes is at its most pronounced.
    _TILE_DEC = -85.0

    @staticmethod
    def _euclid_like_tile_wcs():
        """A MER-VIS-like tile WCS: 19200^2, TAN, 0.1"/px, CD at centre, near the pole.

        Placed at Dec = -85 deg so meridian convergence — and hence the frame
        rotation a re-tangented WCS would introduce — is strong.
        """
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [57.9990741, TestCutoutWcsFidelity._TILE_DEC]
        wcs.wcs.crpix = [9600.0, 9600.0]  # tile centre, FITS 1-based
        wcs.wcs.cd = [[-2.777777777778e-05, 0.0], [0.0, 2.777777777778e-05]]
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.pixel_shape = (19200, 19200)
        return wcs

    @staticmethod
    def _max_sky_error_arcsec(header, tile_wcs, x_min, y_min, resize, final_size):
        """Max separation (arcsec) between cutout-WCS and parent-tile sky positions.

        Ground truth uses the cv2.resize half-pixel-centre convention:
        parent_pixel = origin + (cutout_pixel + 0.5) / resize - 0.5.
        """
        cut_wcs = WCS(header)
        max_err = 0.0
        for frac in np.linspace(0.0, 1.0, 11):
            c = frac * (final_size - 1)
            parent_x = x_min + (c + 0.5) / resize - 0.5
            parent_y = y_min + (c + 0.5) / resize - 0.5
            truth = tile_wcs.pixel_to_world(parent_x, parent_y)
            got = cut_wcs.pixel_to_world(c, c)
            max_err = max(max_err, truth.separation(got).to(u.arcsec).value)
        return max_err

    def test_far_from_centre_no_growing_error(self):
        """A far-off-centre source cutout must agree with the parent tile to < 1 mas."""
        tile = self._euclid_like_tile_wcs()
        # Far off-centre (~0.23 deg) but the 1800 px window stays fully on-tile
        # (centre in [900, 18300]) so this isolates the rotation term, not clipping.
        tx, ty = 17800.3, 17800.7
        target = tile.pixel_to_world(tx, ty)
        ra_c, dec_c = target.ra.deg, target.dec.deg

        requested = 1800  # 3 arcmin at 0.1"/px
        # On-tile, unresized window: origin is simply the integer window start.
        px, py = tile.world_to_pixel_values(ra_c, dec_c)
        x_min = int(px - requested // 2)
        y_min = int(py - requested // 2)
        header = create_wcs_header(
            (requested, requested),
            original_wcs=tile,
            ra_source=ra_c,
            dec_source=dec_c,
            extraction_origin_x=int(x_min),
            extraction_origin_y=int(y_min),
            extraction_size=requested,
        )
        # Parent CRVAL/CD inherited, not re-tangented at the source.
        assert header["CRVAL1"] == 57.9990741
        assert header["CRVAL2"] == self._TILE_DEC

        err = self._max_sky_error_arcsec(header, tile, x_min, y_min, 1.0, requested)
        assert err < 1e-3, f'cutout WCS error {err:.4f}" exceeds 1 mas tolerance'

    @pytest.mark.parametrize(
        "requested,padding,final",
        [
            (400, 1.0, 400),  # no resize, no padding
            (400, 1.5, 224),  # padding > 1 + downsize
            (400, 0.8, 224),  # padding < 1 + downsize
            (300, 2.0, 600),  # padding + upsize
            (128, 1.0, 128),  # small, exact
        ],
    )
    def test_padding_and_resize_combinations(self, requested, padding, final):
        """WCS stays exact across padding factors and resize ratios."""
        tile = self._euclid_like_tile_wcs()
        tx, ty = 18500.3, 18500.7
        target = tile.pixel_to_world(tx, ty)
        ra_c, dec_c = target.ra.deg, target.dec.deg

        # On-tile window: origin is the integer window start (no clip/pad here).
        px, py = tile.world_to_pixel_values(ra_c, dec_c)
        ext_size = int(requested * padding)
        x_min = int(px - ext_size // 2)
        y_min = int(py - ext_size // 2)
        resize = final / ext_size

        header = create_wcs_header(
            (final, final),
            original_wcs=tile,
            ra_source=ra_c,
            dec_source=dec_c,
            extraction_origin_x=int(x_min),
            extraction_origin_y=int(y_min),
            extraction_size=ext_size,
        )

        err = self._max_sky_error_arcsec(header, tile, x_min, y_min, resize, final)
        assert err < 1e-3, f'padding={padding} resize={resize:.3f}: WCS error {err:.4f}" too large'

    @staticmethod
    def _synthetic_tile_hdu(width, height):
        """Small on-disk-style tile HDU with a high-Dec TAN WCS and known pixel_shape."""
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = width
        hdr["NAXIS2"] = height
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRVAL1"] = 57.999
        hdr["CRVAL2"] = -51.5
        hdr["CRPIX1"] = width / 2.0
        hdr["CRPIX2"] = height / 2.0
        hdr["CD1_1"] = -2.7778e-05
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 2.7778e-05
        hdr["CUNIT1"] = "deg"
        hdr["CUNIT2"] = "deg"
        data = np.zeros((height, width), dtype=np.float32)
        return fits.PrimaryHDU(data=data, header=hdr), WCS(hdr)

    @pytest.mark.parametrize(
        "src_x,src_y,requested,padding",
        [
            (184.0, 178.0, 60, 1.0),  # window overruns the tile edge -> clipped + padded
            (183.0, 177.0, 50, 1.5),  # heavier clip with padding
            (185.0, 178.0, 20, 1.0),  # fully on-tile (no clip)
            (182.0, 176.0, 50, 2.0),  # clip + padding upsize
        ],
    )
    def test_real_extraction_marker_roundtrip(self, src_x, src_y, requested, padding):
        """Drive the REAL extraction path, incl. edge clipping, and check WCS fidelity.

        A marker pixel at a known parent location is extracted through the actual
        ``extract_cutouts_vectorized_from_extension`` (which clips and centre-pads at
        tile edges). The cutout WCS built by ``create_wcs_header`` must map the marker's
        cutout pixel back to its true sky position. This is not self-confirming: the
        oracle is the parent tile WCS and the real extracted data, not the CRPIX formula.
        """
        cutout_writer_fits._wcs_header_cache.clear()
        width = height = 200
        hdu, tile = self._synthetic_tile_hdu(width, height)
        marker_px, marker_py = 185, 178  # 0-based parent pixel
        hdu.data[marker_py, marker_px] = 1000.0
        marker_sky = tile.pixel_to_world(marker_px, marker_py)

        target = tile.pixel_to_world(src_x, src_y)
        ra = np.array([target.ra.deg])
        dec = np.array([target.dec.deg])
        # Capture the extraction origin the way the real pipeline threads it to the writer.
        cutouts, success, _, _, origin_x, origin_y = extract_cutouts_vectorized_from_extension(
            hdu, tile, ra, dec, np.array([requested], dtype=int), ["s0"], padding_factor=padding
        )
        assert cutouts[0] is not None and bool(success[0])
        cut = cutouts[0]

        # The marker must have been captured for this to test WCS placement.
        assert cut.max() > 0, "marker not inside extracted window; adjust test params"

        header = create_wcs_header(
            cut.shape,
            original_wcs=tile,
            ra_source=float(ra[0]),
            dec_source=float(dec[0]),
            extraction_origin_x=int(origin_x[0]),
            extraction_origin_y=int(origin_y[0]),
            extraction_size=int(requested * padding),
        )
        cut_wcs = WCS(header)
        cj, ci = np.unravel_index(int(np.argmax(cut)), cut.shape)  # (row=y, col=x)
        got = cut_wcs.pixel_to_world(int(ci), int(cj))
        err = marker_sky.separation(got).to(u.arcsec).value
        assert err < 1e-3, (
            f'real-extraction marker WCS error {err:.4f}" (clip/pad handling regressed)'
        )
