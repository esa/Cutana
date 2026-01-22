#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Tests for cutana.catalogue_preprocessor module.

Tests the catalogue preprocessing, validation, and analysis functionality including
FITS file inspection, data validation, coordinate checking, and comprehensive
catalogue metadata extraction.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cutana.catalogue_preprocessor import (  # noqa: E402
    CatalogueValidationError,
    analyse_source_catalogue,
    analyze_fits_file,
    check_fits_files_exist,
    extract_filter_name,
    extract_fits_sets,
    load_and_validate_catalogue,
    parse_fits_file_paths,
    preprocess_catalogue,
    validate_catalogue_columns,
    validate_coordinate_ranges,
    validate_resolution_ratios,
)


class TestFilterNameExtraction:
    """Test filter name extraction from FITS filenames."""

    def test_extract_filter_vis(self):
        """Test VIS filter detection."""
        assert extract_filter_name("EUC_MER_BGSUB-MOSAIC-VIS_TILE001.fits") == "VIS"
        assert extract_filter_name("/path/to/vis_image.fits") == "VIS"
        assert extract_filter_name("data_vis_processed.fits") == "VIS"

    def test_extract_filter_nir(self):
        """Test NIR filter detection."""
        assert extract_filter_name("EUC_MER_BGSUB-MOSAIC-NIR-H_TILE001.fits") == "NIR-H"
        assert extract_filter_name("NIR-Y_data.fits") == "NIR-Y"
        assert extract_filter_name("NIRJ_processed.fits") == "NIR-J"
        assert extract_filter_name("nir_h_final.fits") == "NIR-H"

    def test_extract_filter_single_letters(self):
        """Test single letter filter detection."""
        assert extract_filter_name("H_band.fits") == "H"
        assert extract_filter_name("Y_filter.fits") == "Y"
        assert extract_filter_name("J_observation.fits") == "J"

    def test_extract_filter_unknown(self):
        """Test unknown filter detection."""
        assert extract_filter_name("unknown_filter.fits") == "UNKNOWN"
        assert extract_filter_name("random_data.fits") == "UNKNOWN"
        assert extract_filter_name("no_filter_info.fits") == "UNKNOWN"


class TestFITSAnalysis:
    """Test FITS file analysis functionality."""

    def test_analyze_fits_nonexistent(self):
        """Test analysis of non-existent FITS file."""
        result = analyze_fits_file("/nonexistent/path.fits")

        assert result["path"] == "/nonexistent/path.fits"
        assert result["exists"] is False
        assert result["filter"] == "UNKNOWN"
        assert result["extensions"] == []
        assert result["num_extensions"] == 0
        assert "File not found" in result["error"]

    @patch("cutana.catalogue_preprocessor.fits")
    def test_analyze_fits_with_mock(self, mock_fits):
        """Test FITS analysis with mocked astropy.fits."""
        # Create mock HDU structure
        mock_hdu1 = MagicMock()
        mock_hdu1.name = "PRIMARY"
        mock_hdu1.data = None

        mock_hdu2 = MagicMock()
        mock_hdu2.name = "IMAGE"
        mock_hdu2.data = MagicMock()  # Has data

        mock_hdul = [mock_hdu1, mock_hdu2]
        mock_fits.open.return_value.__enter__.return_value = mock_hdul

        # Mock path existence
        with patch("pathlib.Path.exists", return_value=True):
            result = analyze_fits_file("/mock/vis_tile.fits")

        assert result["path"] == "/mock/vis_tile.fits"
        assert result["exists"] is True
        assert result["filter"] == "VIS"
        assert result["num_extensions"] == 2
        assert len(result["extensions"]) == 2

        # Check first extension
        assert result["extensions"][0]["index"] == 0
        assert result["extensions"][0]["name"] == "PRIMARY"
        assert result["extensions"][0]["has_data"] is False

        # Check second extension
        assert result["extensions"][1]["index"] == 1
        assert result["extensions"][1]["name"] == "IMAGE"
        assert result["extensions"][1]["has_data"] is True

    @patch("cutana.catalogue_preprocessor.fits")
    def test_analyze_fits_exception(self, mock_fits):
        """Test FITS analysis with exception."""
        mock_fits.open.side_effect = Exception("FITS read error")

        with patch("pathlib.Path.exists", return_value=True):
            result = analyze_fits_file("/mock/bad_file.fits")

        assert result["exists"] is False
        assert "FITS read error" in result["error"]


class TestFITSPathParsing:
    """Test FITS file path parsing from CSV strings.

    Note: parse_fits_file_paths now normalizes paths by default using os.path.normpath.
    Tests must account for platform-specific path separators.
    """

    def test_parse_fits_list_string(self):
        """Test parsing string representation of list."""
        paths_str = "['/path/to/file1.fits', '/path/to/file2.fits']"
        result = parse_fits_file_paths(paths_str)

        # Paths are normalized, so use os.path.normpath for expected values
        expected = [
            os.path.normpath("/path/to/file1.fits"),
            os.path.normpath("/path/to/file2.fits"),
        ]
        assert result == expected

    def test_parse_fits_single_string(self):
        """Test parsing single file path."""
        paths_str = "/path/to/single_file.fits"
        result = parse_fits_file_paths(paths_str)

        expected = [os.path.normpath("/path/to/single_file.fits")]
        assert result == expected

    def test_parse_fits_actual_list(self):
        """Test parsing actual Python list."""
        paths_list = ["/path/to/file1.fits", "/path/to/file2.fits"]
        result = parse_fits_file_paths(paths_list)

        # Paths are normalized
        expected = [os.path.normpath(p) for p in paths_list]
        assert result == expected

    def test_parse_fits_without_normalization(self):
        """Test parsing without path normalization preserves original format."""
        paths_str = "/path/to/file.fits"
        result = parse_fits_file_paths(paths_str, normalize=False)
        assert result == ["/path/to/file.fits"]

    def test_parse_fits_empty_string(self):
        """Test parsing empty string."""
        result = parse_fits_file_paths("")
        assert result == []

    def test_parse_fits_malformed_string(self):
        """Test parsing malformed string raises ValueError."""
        with pytest.raises(ValueError, match="unbalanced brackets"):
            parse_fits_file_paths("[malformed string")

    def test_parse_fits_whitespace(self):
        """Test parsing string with whitespace."""
        paths_str = "  ['/path/to/file.fits']  "
        result = parse_fits_file_paths(paths_str)

        expected = [os.path.normpath("/path/to/file.fits")]
        assert result == expected


class TestCatalogueAnalysis:
    """Test complete catalogue analysis functionality."""

    def create_mock_csv(self, num_sources=10):
        """Create a mock CSV file for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        # Write CSV header
        temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")

        # Write mock data
        for i in range(num_sources):
            source_id = f"MockSource_{i:03d}"
            ra = 150.0 + i * 0.01
            dec = 2.0 + i * 0.01
            diameter_pixel = 128
            fits_paths = f"['/mock/vis_tile_{i:03d}.fits', '/mock/nir_h_tile_{i:03d}.fits']"

            temp_file.write(f'{source_id},{ra},{dec},{diameter_pixel},"{fits_paths}"\n')

        temp_file.close()
        return temp_file.name

    def test_analyze_catalogue_basic(self):
        """Test basic catalogue analysis."""
        csv_path = self.create_mock_csv(num_sources=25)

        try:
            # Mock the FITS analysis to avoid actual file access
            with (
                patch("cutana.catalogue_preprocessor.analyze_fits_file") as mock_analyze,
                patch("cutana.catalogue_preprocessor.load_and_validate_catalogue") as mock_load,
            ):

                # Create mock DataFrame
                mock_df = pd.DataFrame(
                    {
                        "SourceID": [f"MockSource_{i:03d}" for i in range(25)],
                        "RA": [150.0 + i * 0.01 for i in range(25)],
                        "Dec": [2.0 + i * 0.01 for i in range(25)],
                        "diameter_pixel": [128] * 25,
                        "fits_file_paths": [
                            f"['/mock/vis_tile_{i:03d}.fits', '/mock/nir_h_tile_{i:03d}.fits']"
                            for i in range(25)
                        ],
                    }
                )

                mock_load.return_value = mock_df
                mock_analyze.return_value = {
                    "path": "/mock/file.fits",
                    "exists": True,
                    "filter": "VIS",
                    "extensions": [{"name": "PRIMARY", "type": "PrimaryHDU"}],
                    "num_extensions": 1,
                    "error": None,
                }

                result = analyse_source_catalogue(csv_path)

            # Check basic results
            assert result["num_sources"] == 25
            assert result["sample_analysis_size"] == 5  # Min of 5 or num_sources
            assert isinstance(result["fits_files"], list)
            assert isinstance(result["extensions"], list)
            assert "catalogue_columns" in result
            assert "SourceID" in result["catalogue_columns"]
            assert "fits_file_paths" in result["catalogue_columns"]

        finally:
            os.unlink(csv_path)

    def test_analyze_catalogue_large(self):
        """Test catalogue analysis with large dataset."""
        csv_path = self.create_mock_csv(num_sources=1000)

        try:
            with (
                patch("cutana.catalogue_preprocessor.analyze_fits_file") as mock_analyze,
                patch("cutana.catalogue_preprocessor.load_and_validate_catalogue") as mock_load,
            ):

                # Create mock DataFrame
                mock_df = pd.DataFrame(
                    {
                        "SourceID": [f"MockSource_{i:03d}" for i in range(1000)],
                        "RA": [150.0 + i * 0.01 for i in range(1000)],
                        "Dec": [2.0 + i * 0.01 for i in range(1000)],
                        "diameter_pixel": [128] * 1000,
                        "fits_file_paths": [
                            f"['/mock/vis_tile_{i:03d}.fits', '/mock/nir_h_tile_{i:03d}.fits']"
                            for i in range(1000)
                        ],
                    }
                )

                mock_load.return_value = mock_df
                mock_analyze.return_value = {
                    "path": "/mock/file.fits",
                    "exists": True,
                    "filter": "NIR-H",
                    "extensions": [
                        {"name": "PRIMARY", "type": "PrimaryHDU"},
                        {"name": "IMAGE", "type": "ImageHDU"},
                    ],
                    "num_extensions": 2,
                    "error": None,
                }

                result = analyse_source_catalogue(csv_path)

            assert result["num_sources"] == 1000
            assert result["sample_analysis_size"] == 5  # Still samples first 5

        finally:
            os.unlink(csv_path)

    def test_analyze_catalogue_nonexistent(self):
        """Test analysis of non-existent catalogue."""
        with pytest.raises(Exception):
            analyse_source_catalogue("/nonexistent/catalogue.csv")

    def test_analyze_catalogue_multi_channel(self):
        """Test analysis of multi-channel catalogue."""
        # Create a multi-channel CSV
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")

        # Write sources with different channel combinations
        temp_file.write("Source1,150.0,2.0,128,\"['/data/vis.fits']\"\n")
        temp_file.write("Source2,150.1,2.1,128,\"['/data/vis.fits', '/data/nir_h.fits']\"\n")
        temp_file.write(
            "Source3,150.2,2.2,128,\"['/data/vis.fits', '/data/nir_h.fits', '/data/nir_y.fits']\"\n"
        )
        temp_file.close()

        csv_path = temp_file.name

        try:
            with (
                patch("cutana.catalogue_preprocessor.analyze_fits_file") as mock_analyze,
                patch("cutana.catalogue_preprocessor.load_and_validate_catalogue") as mock_load,
            ):

                # Create mock DataFrame
                mock_df = pd.DataFrame(
                    {
                        "SourceID": ["Source1", "Source2", "Source3"],
                        "RA": [150.0, 150.1, 150.2],
                        "Dec": [2.0, 2.1, 2.2],
                        "diameter_pixel": [128, 128, 128],
                        "fits_file_paths": [
                            "['/data/vis.fits']",
                            "['/data/vis.fits', '/data/nir_h.fits']",
                            "['/data/vis.fits', '/data/nir_h.fits', '/data/nir_y.fits']",
                        ],
                    }
                )

                mock_load.return_value = mock_df

                # Return different results based on filename
                def mock_fits_analysis(path):
                    if "vis" in path:
                        return {
                            "path": path,
                            "exists": True,
                            "filter": "VIS",
                            "extensions": [{"name": "IMAGE", "type": "ImageHDU"}],
                            "num_extensions": 1,
                            "error": None,
                        }
                    elif "nir_h" in path:
                        return {
                            "path": path,
                            "exists": True,
                            "filter": "NIR-H",
                            "extensions": [{"name": "IMAGE", "type": "ImageHDU"}],
                            "num_extensions": 1,
                            "error": None,
                        }
                    else:
                        return {
                            "path": path,
                            "exists": True,
                            "filter": "NIR-Y",
                            "extensions": [{"name": "IMAGE", "type": "ImageHDU"}],
                            "num_extensions": 1,
                            "error": None,
                        }

                mock_analyze.side_effect = mock_fits_analysis
                result = analyse_source_catalogue(csv_path)

            assert result["num_sources"] == 3
            assert len(result["fits_files"]) >= 3  # Should find VIS, NIR-H, NIR-Y

            # Check that multiple filters are detected
            extension_names = [ext["name"] for ext in result["extensions"]]
            assert len(extension_names) >= 2  # Should have multiple filter types

        finally:
            os.unlink(csv_path)

    def test_analyze_catalogue_empty(self):
        """Test analysis of empty catalogue."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")  # Header only
        temp_file.close()

        csv_path = temp_file.name

        try:
            with patch("cutana.catalogue_preprocessor.load_and_validate_catalogue") as mock_load:
                # Create empty mock DataFrame
                mock_df = pd.DataFrame(
                    {
                        "SourceID": [],
                        "RA": [],
                        "Dec": [],
                        "diameter_pixel": [],
                        "fits_file_paths": [],
                    }
                )

                mock_load.return_value = mock_df
                result = analyse_source_catalogue(csv_path)

            assert result["num_sources"] == 0
            assert result["sample_analysis_size"] == 0
            assert result["fits_files"] == []
            assert result["extensions"] == []

        finally:
            os.unlink(csv_path)


class TestColumnValidation:
    """Test catalogue column validation functionality."""

    def test_validate_columns_valid(self):
        """Test validation of valid catalogue columns."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001", "S002", "S003"],
                "RA": [150.0, 150.1, 150.2],
                "Dec": [2.0, 2.1, 2.2],
                "diameter_pixel": [128, 256, 128],
                "fits_file_paths": [
                    "['/path/file1.fits']",
                    "['/path/file2.fits']",
                    "['/path/file3.fits']",
                ],
            }
        )

        errors = validate_catalogue_columns(df)
        assert len(errors) == 0

    def test_validate_columns_missing_required(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001", "S002"],
                "RA": [150.0, 150.1],
                # Missing Dec, size, and fits_file_paths
            }
        )

        errors = validate_catalogue_columns(df)
        assert len(errors) > 0
        assert any("Missing required columns" in error for error in errors)

    def test_validate_columns_missing_size(self):
        """Test validation with missing size columns."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001", "S002"],
                "RA": [150.0, 150.1],
                "Dec": [2.0, 2.1],
                "fits_file_paths": ["['/path/file1.fits']", "['/path/file2.fits']"],
                # Missing both diameter_pixel and diameter_arcsec
            }
        )

        errors = validate_catalogue_columns(df)
        assert len(errors) > 0
        assert any("diameter_pixel" in error and "diameter_arcsec" in error for error in errors)

    def test_validate_columns_invalid_types(self):
        """Test validation with invalid data types."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001", "S002"],
                "RA": ["not_numeric", "also_not_numeric"],  # Should be numeric
                "Dec": [2.0, 2.1],
                "diameter_pixel": [128, 256],
                "fits_file_paths": ["['/path/file1.fits']", "['/path/file2.fits']"],
            }
        )

        errors = validate_catalogue_columns(df)
        assert len(errors) > 0
        assert any("RA and Dec" in error and "numeric" in error for error in errors)

    def test_validate_columns_sourceid_convertible(self):
        """Test validation accepts SourceID values that can be converted to strings."""
        # Test with integer SourceIDs (should be acceptable)
        df = pd.DataFrame(
            {
                "SourceID": [1001, 2002, 3003],  # Integer SourceIDs
                "RA": [150.0, 150.1, 150.2],
                "Dec": [2.0, 2.1, 2.2],
                "diameter_pixel": [128, 256, 128],
                "fits_file_paths": [
                    "['/path/file1.fits']",
                    "['/path/file2.fits']",
                    "['/path/file3.fits']",
                ],
            }
        )

        errors = validate_catalogue_columns(df)
        assert len(errors) == 0  # Should be valid

    def test_validate_columns_sourceid_mixed_types(self):
        """Test validation accepts mixed SourceID types that can be converted to strings."""
        df = pd.DataFrame(
            {
                "SourceID": [123, "Source_456", 789.0],  # Mixed types
                "RA": [150.0, 150.1, 150.2],
                "Dec": [2.0, 2.1, 2.2],
                "diameter_pixel": [128, 256, 128],
                "fits_file_paths": [
                    "['/path/file1.fits']",
                    "['/path/file2.fits']",
                    "['/path/file3.fits']",
                ],
            }
        )

        errors = validate_catalogue_columns(df)
        assert len(errors) == 0  # Should be valid


class TestCoordinateValidation:
    """Test coordinate range validation functionality."""

    def test_validate_coordinates_valid(self):
        """Test validation of valid coordinates."""
        df = pd.DataFrame(
            {
                "RA": [150.0, 180.0, 0.0, 359.9],
                "Dec": [-89.9, 0.0, 45.0, 89.9],
                "diameter_pixel": [128, 256, 64, 512],
            }
        )

        errors = validate_coordinate_ranges(df)
        assert len(errors) == 0

    def test_validate_coordinates_invalid_ra(self):
        """Test validation with invalid RA values."""
        df = pd.DataFrame(
            {
                "RA": [-10.0, 370.0],  # Invalid RA values
                "Dec": [45.0, 45.0],
                "diameter_pixel": [128, 128],
            }
        )

        errors = validate_coordinate_ranges(df)
        assert len(errors) > 0
        assert any("RA values must be between 0 and 360" in error for error in errors)

    def test_validate_coordinates_invalid_dec(self):
        """Test validation with invalid Dec values."""
        df = pd.DataFrame(
            {
                "RA": [150.0, 150.0],
                "Dec": [-95.0, 95.0],  # Invalid Dec values
                "diameter_pixel": [128, 128],
            }
        )

        errors = validate_coordinate_ranges(df)
        assert len(errors) > 0
        assert any("Dec values must be between -90 and +90" in error for error in errors)

    def test_validate_coordinates_invalid_sizes(self):
        """Test validation with invalid size values."""
        df = pd.DataFrame(
            {
                "RA": [150.0, 150.1],
                "Dec": [45.0, 45.1],
                "diameter_pixel": [-10, 20000],  # Invalid size values
            }
        )

        errors = validate_coordinate_ranges(df)
        assert len(errors) > 0
        assert any("diameter_pixel values must be between 1 and 10000" in error for error in errors)


class TestFITSFileChecking:
    """Test FITS file existence checking functionality."""

    def create_mock_fits_files(self, file_paths):
        """Create temporary mock FITS files for testing."""
        temp_files = []
        for path in file_paths:
            temp_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
            temp_file.close()
            temp_files.append(temp_file.name)
        return temp_files

    def test_check_fits_files_exist_all_present(self):
        """Test FITS checking when all files exist."""
        # Create temporary FITS files
        temp_files = self.create_mock_fits_files(["file1.fits", "file2.fits"])

        try:
            # Use raw strings and proper escaping for Windows paths
            # Use actual lists instead of string representations
            df = pd.DataFrame({"fits_file_paths": [[temp_files[0]], [temp_files[1]]]})

            errors, warnings = check_fits_files_exist(df)
            assert len(errors) == 0

        finally:
            # Clean up
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    def test_check_fits_files_missing(self):
        """Test FITS checking when files are missing."""
        df = pd.DataFrame(
            {"fits_file_paths": ["['/nonexistent/file1.fits']", "['/nonexistent/file2.fits']"]}
        )

        errors, warnings = check_fits_files_exist(df)
        assert len(errors) > 0
        assert any("Missing FITS files" in error for error in errors)

    def test_check_fits_files_parse_errors(self):
        """Test FITS checking with parsing errors."""
        df = pd.DataFrame({"fits_file_paths": ["malformed_list[", "another_bad_format"]})

        errors, warnings = check_fits_files_exist(df)
        # Should have parsing errors
        assert len(errors) > 0


class TestPreprocessing:
    """Test catalogue preprocessing functionality."""

    def test_preprocess_catalogue_reset_index(self):
        """Test that preprocessing resets non-contiguous indices."""
        # Create DataFrame with non-contiguous index
        df = pd.DataFrame(
            {
                "SourceID": ["S001", "S002", "S003"],
                "RA": [150.0, 150.1, 150.2],
                "Dec": [2.0, 2.1, 2.2],
            }
        )
        df.index = [5, 10, 15]  # Non-contiguous index

        processed_df = preprocess_catalogue(df)

        # Check that index is now contiguous
        expected_index = pd.RangeIndex(len(df))
        assert processed_df.index.equals(expected_index)

    def test_preprocess_catalogue_preserves_data(self):
        """Test that preprocessing preserves all data."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001", "S002", "S003"],
                "RA": [150.0, 150.1, 150.2],
                "Dec": [2.0, 2.1, 2.2],
                "diameter_pixel": [128, 256, 128],
            }
        )

        processed_df = preprocess_catalogue(df)

        # Check data is preserved
        pd.testing.assert_frame_equal(df.reset_index(drop=True), processed_df)

    def test_preprocess_catalogue_sourceid_conversion(self):
        """Test that SourceID values are converted to strings."""
        # Test with various SourceID types
        df = pd.DataFrame(
            {
                "SourceID": [1001, 2002, 3003],  # Integer SourceIDs
                "RA": [150.0, 150.1, 150.2],
                "Dec": [2.0, 2.1, 2.2],
                "diameter_pixel": [128, 256, 128],
            }
        )

        processed_df = preprocess_catalogue(df)

        # Check that SourceID is now string type
        assert processed_df["SourceID"].dtype == object  # String type
        assert all(isinstance(sid, str) for sid in processed_df["SourceID"])
        assert processed_df["SourceID"].tolist() == ["1001", "2002", "3003"]

    def test_preprocess_catalogue_sourceid_mixed_types(self):
        """Test SourceID conversion with mixed data types."""
        df = pd.DataFrame(
            {
                "SourceID": [123, "Source_456", 789.0, "MockSource_001"],  # Mixed types
                "RA": [150.0, 150.1, 150.2, 150.3],
                "Dec": [2.0, 2.1, 2.2, 2.3],
                "diameter_pixel": [128, 256, 128, 64],
            }
        )

        processed_df = preprocess_catalogue(df)

        # All should be converted to strings
        assert processed_df["SourceID"].dtype == object
        assert all(isinstance(sid, str) for sid in processed_df["SourceID"])
        expected = ["123", "Source_456", "789.0", "MockSource_001"]
        assert processed_df["SourceID"].tolist() == expected

    def test_preprocess_catalogue_sourceid_already_strings(self):
        """Test that string SourceIDs are preserved."""
        df = pd.DataFrame(
            {
                "SourceID": ["Source_001", "Source_002", "Source_003"],
                "RA": [150.0, 150.1, 150.2],
                "Dec": [2.0, 2.1, 2.2],
                "diameter_pixel": [128, 256, 128],
            }
        )

        processed_df = preprocess_catalogue(df)

        # Should remain strings
        assert processed_df["SourceID"].dtype == object
        assert processed_df["SourceID"].tolist() == ["Source_001", "Source_002", "Source_003"]


class TestLoadAndValidate:
    """Test comprehensive load and validation functionality."""

    def create_valid_csv(self):
        """Create a valid test CSV file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
        temp_file.write("S001,150.0,2.0,128,\"['/mock/file1.fits']\"\n")
        temp_file.write("S002,150.1,2.1,256,\"['/mock/file2.fits']\"\n")
        temp_file.close()
        return temp_file.name

    def create_valid_parquet(self):
        """Create a valid test Parquet file."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        df = pd.DataFrame(
            {
                "SourceID": ["S001", "S002"],
                "RA": [150.0, 150.1],
                "Dec": [2.0, 2.1],
                "diameter_pixel": [128, 256],
                "fits_file_paths": ["['/mock/file1.fits']", "['/mock/file2.fits']"],
            }
        )
        df.to_parquet(temp_file.name)
        temp_file.close()
        return temp_file.name

    def create_invalid_csv(self, error_type="missing_columns"):
        """Create an invalid test CSV file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        if error_type == "missing_columns":
            temp_file.write("SourceID,RA\n")  # Missing required columns
            temp_file.write("S001,150.0\n")
        elif error_type == "invalid_coordinates":
            temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
            temp_file.write("S001,400.0,100.0,128,\"['/mock/file1.fits']\"\n")  # Invalid RA/Dec
        elif error_type == "invalid_types":
            temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
            temp_file.write(
                "S001,not_a_number,2.0,128,\"['/mock/file1.fits']\"\n"
            )  # Invalid RA type

        temp_file.close()
        return temp_file.name

    def test_load_and_validate_valid_catalogue(self):
        """Test loading and validating a valid catalogue."""
        csv_path = self.create_valid_csv()
        parquet_path = self.create_valid_parquet()
        try:
            # Skip FITS file checking for this test
            df_csv = load_and_validate_catalogue(csv_path, skip_fits_check=True)
            df_parquet = load_and_validate_catalogue(parquet_path, skip_fits_check=True)

            assert len(df_csv) == 2
            assert "SourceID" in df_csv.columns
            assert df_csv.index.equals(pd.RangeIndex(len(df_csv)))  # Index should be reset

            assert len(df_parquet) == 2
            assert "SourceID" in df_parquet.columns
            assert df_parquet.index.equals(pd.RangeIndex(len(df_parquet)))  # Index should be reset

        finally:
            os.unlink(csv_path)
            os.unlink(parquet_path)

    def test_load_and_validate_missing_columns(self):
        """Test loading catalogue with missing columns."""
        csv_path = self.create_invalid_csv("missing_columns")

        try:
            with pytest.raises(CatalogueValidationError) as exc_info:
                load_and_validate_catalogue(csv_path, skip_fits_check=True)

            assert "Missing required columns" in str(exc_info.value)

        finally:
            os.unlink(csv_path)

    def test_load_and_validate_invalid_coordinates(self):
        """Test loading catalogue with invalid coordinates."""
        csv_path = self.create_invalid_csv("invalid_coordinates")

        try:
            with pytest.raises(CatalogueValidationError) as exc_info:
                load_and_validate_catalogue(csv_path, skip_fits_check=True)

            error_msg = str(exc_info.value)
            assert (
                "RA values must be between 0 and 360" in error_msg
                or "Dec values must be between -90 and +90" in error_msg
            )

        finally:
            os.unlink(csv_path)

    def test_load_and_validate_invalid_types(self):
        """Test loading catalogue with invalid data types."""
        csv_path = self.create_invalid_csv("invalid_types")

        try:
            with pytest.raises(CatalogueValidationError) as exc_info:
                load_and_validate_catalogue(csv_path, skip_fits_check=True)

            assert "RA and Dec columns must be numeric" in str(exc_info.value)

        finally:
            os.unlink(csv_path)

    def test_load_and_validate_ill_formatted_parquet(self):
        """Test loading ill-formatted parquet file shows meaningful error."""
        # Create a file with .parquet extension but invalid content (plain text)
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False)
        temp_file.write("This is not a valid parquet file\n")
        temp_file.write("Just plain text content\n")
        temp_file.close()
        fake_parquet_path = temp_file.name

        try:
            with pytest.raises(Exception) as exc_info:
                load_and_validate_catalogue(fake_parquet_path, skip_fits_check=True)

            # Verify the error message is meaningful (not a generic error)
            error_message = str(exc_info.value).lower()
            # Should indicate parquet-related error (pyarrow/fastparquet will fail with specific errors)
            assert any(
                keyword in error_message
                for keyword in ["parquet", "arrow", "magic", "corrupt", "invalid", "file"]
            ), f"Expected meaningful parquet error, got: {exc_info.value}"

        finally:
            os.unlink(fake_parquet_path)

    def test_load_and_validate_truncated_parquet(self):
        """Test loading truncated/corrupted parquet file shows meaningful error."""
        # First create a valid parquet file
        valid_temp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        df = pd.DataFrame(
            {
                "SourceID": ["S001", "S002"],
                "RA": [150.0, 150.1],
                "Dec": [2.0, 2.1],
                "diameter_pixel": [128, 256],
                "fits_file_paths": ["['/mock/file1.fits']", "['/mock/file2.fits']"],
            }
        )
        df.to_parquet(valid_temp.name)
        valid_temp.close()

        # Read and truncate the file to create a corrupted parquet
        with open(valid_temp.name, "rb") as f:
            valid_content = f.read()

        truncated_temp = tempfile.NamedTemporaryFile(mode="wb", suffix=".parquet", delete=False)
        # Write only first 100 bytes (truncated)
        truncated_temp.write(valid_content[:100])
        truncated_temp.close()

        try:
            with pytest.raises(Exception) as exc_info:
                load_and_validate_catalogue(truncated_temp.name, skip_fits_check=True)

            # Verify the error message is meaningful
            error_message = str(exc_info.value).lower()
            assert any(
                keyword in error_message
                for keyword in ["parquet", "arrow", "corrupt", "truncat", "eof", "file", "invalid"]
            ), f"Expected meaningful parquet error, got: {exc_info.value}"

        finally:
            os.unlink(valid_temp.name)
            os.unlink(truncated_temp.name)


class TestExtractFitsSets:
    """Test the extract_fits_sets function."""

    def test_extract_fits_sets_single_file(self):
        """Test extract_fits_sets with single FITS file."""
        fits_files = ["/path/to/vis_image.fits"]
        fits_set_dict, resolution_ratios = extract_fits_sets(fits_files)

        assert len(fits_set_dict) == 1
        assert len(resolution_ratios) == 0  # No multi-filter scenario

    def test_extract_fits_sets_multiple_files(self):
        """Test extract_fits_sets with multiple FITS files."""
        fits_files = ["/path/to/nir_h.fits", "/path/to/nir_j.fits", "/path/to/vis.fits"]
        fits_set_dict, resolution_ratios = extract_fits_sets(fits_files, ["NIR-H", "NIR-J", "VIS"])

        assert len(fits_set_dict) == 1
        fits_set = list(fits_set_dict.keys())[0]
        assert len(fits_set) == 3
        # Resolution ratios would be computed if FITS files existed


class TestResolutionValidation:
    """Test resolution ratio validation for diameter_pixel usage."""

    def test_validate_resolution_ratios_single_filter(self):
        """Test resolution validation with single filter - should pass."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001"],
                "RA": [150.0],
                "Dec": [2.0],
                "diameter_pixel": [128],
                "fits_file_paths": ["['/path/to/vis.fits']"],
            }
        )

        errors = validate_resolution_ratios(df)
        assert len(errors) == 0  # No errors for single filter

    def test_validate_resolution_ratios_no_diameter_pixel(self):
        """Test resolution validation when using diameter_arcsec - should pass."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001"],
                "RA": [150.0],
                "Dec": [2.0],
                "diameter_arcsec": [5.0],
                "fits_file_paths": ["['/path/to/vis.fits', '/path/to/nir_h.fits']"],
            }
        )

        errors = validate_resolution_ratios(df)
        assert len(errors) == 0  # No errors when using diameter_arcsec

    @patch("cutana.catalogue_preprocessor.extract_fits_sets")
    def test_validate_resolution_ratios_different_resolutions(self, mock_extract_fits_sets):
        """Test resolution validation with different resolutions - should error."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001"],
                "RA": [150.0],
                "Dec": [2.0],
                "diameter_pixel": [128],
                "fits_file_paths": ["['/path/to/vis.fits', '/path/to/nir_h.fits']"],
            }
        )

        # Mock extract_fits_sets to return different resolution ratios
        mock_extract_fits_sets.return_value = (
            {
                ("/path/to/nir_h.fits", "/path/to/vis.fits"): [
                    "/path/to/vis.fits",
                    "/path/to/nir_h.fits",
                ]
            },
            {"VIS": 1.0, "NIR-H": 1.05},  # 5% difference - should trigger error
        )

        errors = validate_resolution_ratios(df)
        assert len(errors) > 0
        assert "Resolution ratio difference" in errors[0]
        assert "diameter_arcsec" in errors[0]

    @patch("cutana.catalogue_preprocessor.extract_fits_sets")
    def test_validate_resolution_ratios_similar_resolutions(self, mock_extract_fits_sets):
        """Test resolution validation with similar resolutions - should pass."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001"],
                "RA": [150.0],
                "Dec": [2.0],
                "diameter_pixel": [128],
                "fits_file_paths": ["['/path/to/vis.fits', '/path/to/nir_h.fits']"],
            }
        )

        # Mock extract_fits_sets to return similar resolution ratios
        mock_extract_fits_sets.return_value = (
            {
                ("/path/to/nir_h.fits", "/path/to/vis.fits"): [
                    "/path/to/vis.fits",
                    "/path/to/nir_h.fits",
                ]
            },
            {"VIS": 1.0, "NIR-H": 1.00005},  # 0.005% difference - should pass
        )

        errors = validate_resolution_ratios(df)
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
