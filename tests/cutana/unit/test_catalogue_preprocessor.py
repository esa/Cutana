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

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("EUC_MER_BGSUB-MOSAIC-VIS_TILE001.fits", "VIS"),
            ("/path/to/vis_image.fits", "VIS"),
            ("data_vis_processed.fits", "VIS"),
            ("EUC_MER_BGSUB-MOSAIC-NIR-H_TILE001.fits", "NIR-H"),
            ("NIR-Y_data.fits", "NIR-Y"),
            ("NIRJ_processed.fits", "NIR-J"),
            ("nir_h_final.fits", "NIR-H"),
            ("H_band.fits", "H"),
            ("Y_filter.fits", "Y"),
            ("J_observation.fits", "J"),
            ("unknown_filter.fits", "UNKNOWN"),
            ("random_data.fits", "UNKNOWN"),
            ("no_filter_info.fits", "UNKNOWN"),
        ],
    )
    def test_extract_filter(self, filename, expected):
        """Test filter name extraction from various FITS filenames."""
        assert extract_filter_name(filename) == expected


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
        mock_hdu1 = MagicMock()
        mock_hdu1.name = "PRIMARY"
        mock_hdu1.data = None

        mock_hdu2 = MagicMock()
        mock_hdu2.name = "IMAGE"
        mock_hdu2.data = MagicMock()

        mock_hdul = [mock_hdu1, mock_hdu2]
        mock_fits.open.return_value.__enter__.return_value = mock_hdul

        with patch("pathlib.Path.exists", return_value=True):
            result = analyze_fits_file("/mock/vis_tile.fits")

        assert result["path"] == "/mock/vis_tile.fits"
        assert result["exists"] is True
        assert result["filter"] == "VIS"
        assert result["num_extensions"] == 2
        assert len(result["extensions"]) == 2
        assert result["extensions"][0]["index"] == 0
        assert result["extensions"][0]["name"] == "PRIMARY"
        assert result["extensions"][0]["has_data"] is False
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

    @pytest.mark.parametrize(
        "input_val, normalize, expected_paths",
        [
            # List string with two paths
            (
                "['/path/to/file1.fits', '/path/to/file2.fits']",
                True,
                ["/path/to/file1.fits", "/path/to/file2.fits"],
            ),
            # Single file path
            ("/path/to/single_file.fits", True, ["/path/to/single_file.fits"]),
            # Actual Python list
            (
                ["/path/to/file1.fits", "/path/to/file2.fits"],
                True,
                ["/path/to/file1.fits", "/path/to/file2.fits"],
            ),
            # Without normalization
            ("/path/to/file.fits", False, None),
            # Empty string
            ("", True, []),
            # Whitespace-padded
            ("  ['/path/to/file.fits']  ", True, ["/path/to/file.fits"]),
        ],
    )
    def test_parse_fits_file_paths(self, input_val, normalize, expected_paths):
        """Test parsing various FITS path formats."""
        result = parse_fits_file_paths(input_val, normalize=normalize)
        if normalize and expected_paths is not None:
            expected = [os.path.normpath(p) for p in expected_paths]
        elif expected_paths is not None:
            expected = expected_paths
        else:
            # Without normalization, preserve original format
            expected = ["/path/to/file.fits"]
        assert result == expected

    def test_parse_fits_malformed_string(self):
        """Test parsing malformed string raises ValueError."""
        with pytest.raises(ValueError, match="unbalanced brackets"):
            parse_fits_file_paths("[malformed string")


class TestCatalogueAnalysis:
    """Test complete catalogue analysis functionality."""

    def _create_mock_csv(self, num_sources=10):
        """Create a mock CSV file for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
        for i in range(num_sources):
            source_id = f"MockSource_{i:03d}"
            ra = 150.0 + i * 0.01
            dec = 2.0 + i * 0.01
            fits_paths = f"['/mock/vis_tile_{i:03d}.fits', '/mock/nir_h_tile_{i:03d}.fits']"
            temp_file.write(f'{source_id},{ra},{dec},128,"{fits_paths}"\n')
        temp_file.close()
        return temp_file.name

    @pytest.mark.parametrize("num_sources", [25, 1000])
    def test_analyze_catalogue_sizes(self, num_sources):
        """Test catalogue analysis with different dataset sizes."""
        csv_path = self._create_mock_csv(num_sources=num_sources)
        try:
            with (
                patch("cutana.catalogue_preprocessor.analyze_fits_file") as mock_analyze,
                patch("cutana.catalogue_preprocessor.load_and_validate_catalogue") as mock_load,
            ):
                mock_df = pd.DataFrame(
                    {
                        "SourceID": [f"MockSource_{i:03d}" for i in range(num_sources)],
                        "RA": [150.0 + i * 0.01 for i in range(num_sources)],
                        "Dec": [2.0 + i * 0.01 for i in range(num_sources)],
                        "diameter_pixel": [128] * num_sources,
                        "fits_file_paths": [
                            f"['/mock/vis_tile_{i:03d}.fits', '/mock/nir_h_tile_{i:03d}.fits']"
                            for i in range(num_sources)
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

            assert result["num_sources"] == num_sources
            assert result["sample_analysis_size"] == 5
            assert isinstance(result["fits_files"], list)
            assert isinstance(result["extensions"], list)
            assert "catalogue_columns" in result
            assert "SourceID" in result["catalogue_columns"]
            assert "fits_file_paths" in result["catalogue_columns"]
        finally:
            os.unlink(csv_path)

    def test_analyze_catalogue_nonexistent(self):
        """Test analysis of non-existent catalogue."""
        with pytest.raises(Exception):
            analyse_source_catalogue("/nonexistent/catalogue.csv")

    def test_analyze_catalogue_multi_channel(self):
        """Test analysis of multi-channel catalogue."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
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

                def mock_fits_analysis(path):
                    if "vis" in path:
                        filter_name = "VIS"
                    elif "nir_h" in path:
                        filter_name = "NIR-H"
                    else:
                        filter_name = "NIR-Y"
                    return {
                        "path": path,
                        "exists": True,
                        "filter": filter_name,
                        "extensions": [{"name": "IMAGE", "type": "ImageHDU"}],
                        "num_extensions": 1,
                        "error": None,
                    }

                mock_analyze.side_effect = mock_fits_analysis
                result = analyse_source_catalogue(csv_path)

            assert result["num_sources"] == 3
            assert len(result["fits_files"]) >= 3
            extension_names = [ext["name"] for ext in result["extensions"]]
            assert len(extension_names) >= 2
        finally:
            os.unlink(csv_path)

    def test_analyze_catalogue_empty(self):
        """Test analysis of empty catalogue."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
        temp_file.close()
        csv_path = temp_file.name

        try:
            with patch("cutana.catalogue_preprocessor.load_and_validate_catalogue") as mock_load:
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

    @pytest.mark.parametrize(
        "source_ids",
        [
            ["S001", "S002", "S003"],
            [1001, 2002, 3003],
            [123, "Source_456", 789.0],
        ],
        ids=["string_ids", "integer_ids", "mixed_ids"],
    )
    def test_validate_columns_valid(self, source_ids):
        """Test validation of valid catalogue columns with various SourceID types."""
        df = pd.DataFrame(
            {
                "SourceID": source_ids,
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

    @pytest.mark.parametrize(
        "columns, error_substring",
        [
            # Missing Dec, size, and fits_file_paths
            (
                {"SourceID": ["S001", "S002"], "RA": [150.0, 150.1]},
                "Missing required columns",
            ),
            # Missing both diameter_pixel and diameter_arcsec
            (
                {
                    "SourceID": ["S001", "S002"],
                    "RA": [150.0, 150.1],
                    "Dec": [2.0, 2.1],
                    "fits_file_paths": ["['/path/file1.fits']", "['/path/file2.fits']"],
                },
                "diameter_pixel",
            ),
            # Invalid RA type
            (
                {
                    "SourceID": ["S001", "S002"],
                    "RA": ["not_numeric", "also_not_numeric"],
                    "Dec": [2.0, 2.1],
                    "diameter_pixel": [128, 256],
                    "fits_file_paths": ["['/path/file1.fits']", "['/path/file2.fits']"],
                },
                "RA and Dec",
            ),
        ],
        ids=["missing_required", "missing_size", "invalid_types"],
    )
    def test_validate_columns_errors(self, columns, error_substring):
        """Test validation catches various column errors."""
        df = pd.DataFrame(columns)
        errors = validate_catalogue_columns(df)
        assert len(errors) > 0
        assert any(error_substring in error for error in errors)


class TestCoordinateValidation:
    """Test coordinate range validation functionality."""

    @pytest.mark.parametrize(
        "ra_vals, dec_vals, diameters, error_substring",
        [
            # Valid coordinates
            ([150.0, 180.0, 0.0, 359.9], [-89.9, 0.0, 45.0, 89.9], [128, 256, 64, 512], None),
            # Invalid RA
            ([-10.0, 370.0], [45.0, 45.0], [128, 128], "RA values must be between 0 and 360"),
            # Invalid Dec
            ([150.0, 150.0], [-95.0, 95.0], [128, 128], "Dec values must be between -90 and +90"),
            # Invalid sizes
            (
                [150.0, 150.1],
                [45.0, 45.1],
                [-10, 20000],
                "diameter_pixel values must be between 1 and 10000",
            ),
        ],
        ids=["valid", "invalid_ra", "invalid_dec", "invalid_sizes"],
    )
    def test_validate_coordinate_ranges(self, ra_vals, dec_vals, diameters, error_substring):
        """Test coordinate range validation for valid and invalid inputs."""
        df = pd.DataFrame({"RA": ra_vals, "Dec": dec_vals, "diameter_pixel": diameters})
        errors = validate_coordinate_ranges(df)
        if error_substring is None:
            assert len(errors) == 0
        else:
            assert len(errors) > 0
            assert any(error_substring in error for error in errors)


class TestFITSFileChecking:
    """Test FITS file existence checking functionality."""

    def _create_mock_fits_files(self, file_paths):
        """Create temporary mock FITS files for testing."""
        temp_files = []
        for path in file_paths:
            temp_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
            temp_file.close()
            temp_files.append(temp_file.name)
        return temp_files

    def test_check_fits_files_exist_all_present(self):
        """Test FITS checking when all files exist."""
        temp_files = self._create_mock_fits_files(["file1.fits", "file2.fits"])
        try:
            df = pd.DataFrame({"fits_file_paths": [[temp_files[0]], [temp_files[1]]]})
            errors, warnings = check_fits_files_exist(df)
            assert len(errors) == 0
        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    @pytest.mark.parametrize(
        "paths_data",
        [
            ["['/nonexistent/file1.fits']", "['/nonexistent/file2.fits']"],
            ["malformed_list[", "another_bad_format"],
        ],
        ids=["missing_files", "parse_errors"],
    )
    def test_check_fits_files_errors(self, paths_data):
        """Test FITS checking with missing files or parse errors."""
        df = pd.DataFrame({"fits_file_paths": paths_data})
        errors, warnings = check_fits_files_exist(df)
        assert len(errors) > 0


class TestPreprocessing:
    """Test catalogue preprocessing functionality."""

    def test_preprocess_catalogue_reset_index(self):
        """Test that preprocessing resets non-contiguous indices."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001", "S002", "S003"],
                "RA": [150.0, 150.1, 150.2],
                "Dec": [2.0, 2.1, 2.2],
            }
        )
        df.index = [5, 10, 15]
        processed_df = preprocess_catalogue(df)
        expected_index = pd.RangeIndex(len(df))
        assert processed_df.index.equals(expected_index)

    @pytest.mark.parametrize(
        "source_ids, expected_ids",
        [
            (["S001", "S002", "S003"], ["S001", "S002", "S003"]),
            (["UNIQUE_A", "UNIQUE_B", "UNIQUE_C"], ["UNIQUE_A", "UNIQUE_B", "UNIQUE_C"]),
        ],
        ids=["preserves_data", "no_duplicates_unchanged"],
    )
    def test_preprocess_catalogue_unique_ids_preserved(self, source_ids, expected_ids):
        """Test that unique SourceIDs are preserved unchanged."""
        df = pd.DataFrame(
            {
                "SourceID": source_ids,
                "RA": [150.0 + i * 10 for i in range(len(source_ids))],
                "Dec": [2.0 + i for i in range(len(source_ids))],
                "diameter_pixel": [64] * len(source_ids),
            }
        )
        processed_df = preprocess_catalogue(df)
        assert processed_df["SourceID"].tolist() == expected_ids

    def test_preprocess_catalogue_small_catalogue_duplicates_reformatted(self):
        """Duplicate SourceIDs are reformatted to ``<orig>_<RA>_<Dec>`` with the
        row's actual RA/Dec embedded — not just any 4-token string."""
        original_ids = ["SOURCE_A", "SOURCE_A", "SOURCE_B", "SOURCE_B"]
        ras = [54.447980, 54.447955, 59.309802, 59.309844]
        decs = [-29.010482, -29.228961, -49.797120, -49.800558]
        df = pd.DataFrame(
            {
                "SourceID": original_ids,
                "RA": ras,
                "Dec": decs,
                "diameter_pixel": [17, 33, 41, 47],
            }
        )
        processed_df = preprocess_catalogue(df)

        assert len(processed_df["SourceID"].unique()) == 4
        for orig_base, ra, dec, reformatted in zip(
            original_ids, ras, decs, processed_df["SourceID"]
        ):
            base, ra_str, dec_str = reformatted.rsplit("_", 2)
            assert base == orig_base, (
                f"Reformatted ID {reformatted!r} does not preserve original base "
                f"{orig_base!r} — row order may have been corrupted."
            )
            assert float(ra_str) == pytest.approx(ra, abs=1e-9), (
                f"Reformatted ID {reformatted!r} embeds RA={ra_str} but the row's RA is {ra}."
            )
            assert float(dec_str) == pytest.approx(dec, abs=1e-9), (
                f"Reformatted ID {reformatted!r} embeds Dec={dec_str} but the row's Dec is {dec}."
            )

    def test_preprocess_catalogue_large_catalogue_duplicates_not_reformatted(self):
        """Test that duplicate SourceIDs in a large catalogue are not reformatted."""
        n = 100_000
        source_ids = [f"SOURCE_{i % 10}" for i in range(n)]
        df = pd.DataFrame(
            {
                "SourceID": source_ids,
                "RA": [float(i % 360) for i in range(n)],
                "Dec": [float(i % 180 - 90) for i in range(n)],
                "diameter_pixel": [64] * n,
            }
        )
        processed_df = preprocess_catalogue(df)
        assert processed_df["SourceID"].tolist() == source_ids


class TestLoadAndValidate:
    """Test comprehensive load and validation functionality."""

    def _create_valid_csv(self):
        """Create a valid test CSV file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
        temp_file.write("S001,150.0,2.0,128,\"['/mock/file1.fits']\"\n")
        temp_file.write("S002,150.1,2.1,256,\"['/mock/file2.fits']\"\n")
        temp_file.close()
        return temp_file.name

    def _create_valid_parquet(self):
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

    def test_load_and_validate_valid_catalogue(self):
        """Test loading and validating a valid catalogue (CSV and Parquet)."""
        csv_path = self._create_valid_csv()
        parquet_path = self._create_valid_parquet()
        try:
            df_csv = load_and_validate_catalogue(csv_path, skip_fits_check=True)
            df_parquet = load_and_validate_catalogue(parquet_path, skip_fits_check=True)

            assert len(df_csv) == 2
            assert "SourceID" in df_csv.columns
            assert df_csv.index.equals(pd.RangeIndex(len(df_csv)))

            assert len(df_parquet) == 2
            assert "SourceID" in df_parquet.columns
            assert df_parquet.index.equals(pd.RangeIndex(len(df_parquet)))
        finally:
            os.unlink(csv_path)
            os.unlink(parquet_path)

    @pytest.mark.parametrize(
        "error_type, error_substring",
        [
            ("missing_columns", "Missing required columns"),
            ("invalid_coordinates", "values must be between"),
            ("invalid_types", "RA and Dec columns must be numeric"),
        ],
        ids=["missing_columns", "invalid_coordinates", "invalid_types"],
    )
    def test_load_and_validate_invalid_csv(self, error_type, error_substring):
        """Test loading catalogue with various CSV errors."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        if error_type == "missing_columns":
            temp_file.write("SourceID,RA\n")
            temp_file.write("S001,150.0\n")
        elif error_type == "invalid_coordinates":
            temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
            temp_file.write("S001,400.0,100.0,128,\"['/mock/file1.fits']\"\n")
        elif error_type == "invalid_types":
            temp_file.write("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
            temp_file.write("S001,not_a_number,2.0,128,\"['/mock/file1.fits']\"\n")
        temp_file.close()
        csv_path = temp_file.name

        try:
            with pytest.raises(CatalogueValidationError) as exc_info:
                load_and_validate_catalogue(csv_path, skip_fits_check=True)
            assert error_substring in str(exc_info.value)
        finally:
            os.unlink(csv_path)

    @pytest.mark.parametrize(
        "content_factory",
        [
            # Plain text pretending to be parquet
            lambda: b"This is not a valid parquet file\nJust plain text content\n",
            # Truncated valid parquet (first 100 bytes)
            "truncated",
        ],
        ids=["ill_formatted", "truncated"],
    )
    def test_load_and_validate_invalid_parquet(self, content_factory):
        """Test loading ill-formatted or truncated parquet files."""
        if content_factory == "truncated":
            # Create a valid parquet, then truncate it
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
            with open(valid_temp.name, "rb") as f:
                content = f.read()[:100]
            os.unlink(valid_temp.name)
        else:
            content = content_factory()

        temp_file = tempfile.NamedTemporaryFile(mode="wb", suffix=".parquet", delete=False)
        temp_file.write(content)
        temp_file.close()
        parquet_path = temp_file.name

        try:
            with pytest.raises(Exception) as exc_info:
                load_and_validate_catalogue(parquet_path, skip_fits_check=True)
            error_message = str(exc_info.value).lower()
            assert any(
                keyword in error_message
                for keyword in [
                    "parquet",
                    "arrow",
                    "magic",
                    "corrupt",
                    "invalid",
                    "file",
                    "eof",
                    "truncat",
                ]
            ), f"Expected meaningful parquet error, got: {exc_info.value}"
        finally:
            os.unlink(parquet_path)


class TestExtractFitsSets:
    """Test the extract_fits_sets function."""

    @pytest.mark.parametrize(
        "fits_files, filters, expected_set_count, expected_set_size",
        [
            (["/path/to/vis_image.fits"], None, 1, None),
            (
                ["/path/to/nir_h.fits", "/path/to/nir_j.fits", "/path/to/vis.fits"],
                ["NIR-H", "NIR-J", "VIS"],
                1,
                3,
            ),
        ],
        ids=["single_file", "multiple_files"],
    )
    def test_extract_fits_sets(self, fits_files, filters, expected_set_count, expected_set_size):
        """Test extract_fits_sets with single and multiple FITS files."""
        if filters:
            fits_set_dict, resolution_ratios = extract_fits_sets(fits_files, filters)
        else:
            fits_set_dict, resolution_ratios = extract_fits_sets(fits_files)
            assert len(resolution_ratios) == 0
        assert len(fits_set_dict) == expected_set_count
        if expected_set_size is not None:
            fits_set = list(fits_set_dict.keys())[0]
            assert len(fits_set) == expected_set_size


class TestResolutionValidation:
    """Test resolution ratio validation for diameter_pixel usage."""

    @pytest.mark.parametrize(
        "columns, expected_no_errors",
        [
            # Single filter with diameter_pixel
            (
                {
                    "SourceID": ["S001"],
                    "RA": [150.0],
                    "Dec": [2.0],
                    "diameter_pixel": [128],
                    "fits_file_paths": ["['/path/to/vis.fits']"],
                },
                True,
            ),
            # Using diameter_arcsec instead
            (
                {
                    "SourceID": ["S001"],
                    "RA": [150.0],
                    "Dec": [2.0],
                    "diameter_arcsec": [5.0],
                    "fits_file_paths": ["['/path/to/vis.fits', '/path/to/nir_h.fits']"],
                },
                True,
            ),
        ],
        ids=["single_filter", "diameter_arcsec"],
    )
    def test_validate_resolution_ratios_no_mock(self, columns, expected_no_errors):
        """Test resolution validation without mocked fits sets."""
        df = pd.DataFrame(columns)
        errors = validate_resolution_ratios(df)
        assert (len(errors) == 0) == expected_no_errors

    @pytest.mark.parametrize(
        "ratios, expect_error",
        [
            ({"VIS": 1.0, "NIR-H": 1.05}, True),
            ({"VIS": 1.0, "NIR-H": 1.00005}, False),
        ],
        ids=["different_resolutions", "similar_resolutions"],
    )
    @patch("cutana.catalogue_preprocessor.extract_fits_sets")
    def test_validate_resolution_ratios_with_mock(
        self, mock_extract_fits_sets, ratios, expect_error
    ):
        """Test resolution validation with mocked resolution ratios."""
        df = pd.DataFrame(
            {
                "SourceID": ["S001"],
                "RA": [150.0],
                "Dec": [2.0],
                "diameter_pixel": [128],
                "fits_file_paths": ["['/path/to/vis.fits', '/path/to/nir_h.fits']"],
            }
        )
        mock_extract_fits_sets.return_value = (
            {
                ("/path/to/nir_h.fits", "/path/to/vis.fits"): [
                    "/path/to/vis.fits",
                    "/path/to/nir_h.fits",
                ]
            },
            ratios,
        )
        errors = validate_resolution_ratios(df)
        if expect_error:
            assert len(errors) > 0
            assert "Resolution ratio difference" in errors[0]
            assert "diameter_arcsec" in errors[0]
        else:
            assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
