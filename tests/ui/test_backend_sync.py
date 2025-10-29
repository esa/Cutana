#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Synchronous tests for backend interface functionality."""

import pytest
import asyncio
from pathlib import Path
import csv
import tempfile

from cutana_ui.utils.backend_interface import BackendInterface
from cutana.preview_generator import clear_preview_cache, get_cache_status


class TestBackendInterfaceMockData:
    """Test backend interface with mock data (synchronous only)."""

    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory."""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def mock_catalogue_small(self, test_data_dir):
        """Path to small mock catalogue."""
        return str(test_data_dir / "euclid_cutana_catalogue_small.csv")

    def test_mock_catalogue_structure(self, mock_catalogue_small):
        """Test that mock catalogue has correct structure."""
        # This is a synchronous test that verifies the mock data structure
        # without requiring async functionality

        with open(mock_catalogue_small, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check we have expected number of sources
        assert len(rows) == 25

        # Check required columns exist
        expected_cols = ["SourceID", "RA", "Dec", "diameter_pixel", "fits_file_paths"]
        first_row = rows[0]
        for col in expected_cols:
            assert col in first_row, f"Missing column: {col}"

        # Check data types and ranges
        assert first_row["SourceID"].startswith("MockSource_")
        assert 149.0 <= float(first_row["RA"]) <= 151.0  # Around 150 degrees
        assert 1.0 <= float(first_row["Dec"]) <= 3.0  # Around 2 degrees
        assert int(first_row["diameter_pixel"]) > 0

    def test_catalogue_processing_simulation(self, mock_catalogue_small):
        """Test catalogue processing simulation (synchronous)."""
        # Simulate what the backend would do synchronously
        with open(mock_catalogue_small, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Simulate analysis results
        num_sources = len(rows)

        # Extract FITS files from the catalogue
        fits_files = []
        for row in rows[:5]:  # Check first 5 sources
            file_paths = eval(row["fits_file_paths"])  # Convert string back to list
            fits_files.extend(file_paths)
        fits_files = list(set(fits_files))  # Remove duplicates

        # Expected results structure
        mock_result = {
            "num_sources": num_sources,
            "fits_files": fits_files,
            "extensions": [
                {"name": "VIS", "ext": "IMAGE"},
                {"name": "NIR-Y", "ext": "IMAGE"},
                {"name": "NIR-H", "ext": "IMAGE"},
            ],
        }

        # Verify results
        assert mock_result["num_sources"] == 25
        assert len(mock_result["fits_files"]) > 0
        assert len(mock_result["extensions"]) == 3

        # Check extension names
        ext_names = [ext["name"] for ext in mock_result["extensions"]]
        assert "VIS" in ext_names
        assert "NIR-Y" in ext_names
        assert "NIR-H" in ext_names

    def test_configuration_merging_simulation(self):
        """Test configuration merging simulation (synchronous)."""
        # Simulate default configuration
        default_config = {
            "max_workers": 4,
            "batch_size": 128,
            "output_format": "zarr",
            "data_format": "float32",
        }

        # Simulate user configuration from UI
        user_config = {
            "source_catalogue": "/test/catalogue.csv",
            "output_dir": "/test/output",
            "selected_extensions": [{"name": "VIS", "ext": "IMAGE"}],
            "target_resolution": 256,
            "stretch": "log",
        }

        # Simulate analysis results
        analysis_result = {
            "num_sources": 25,
            "fits_files": ["test.fits"],
            "available_extensions": [{"name": "VIS", "ext": "IMAGE"}],
        }

        # Merge configurations (simulating what happens in StartScreen)
        full_config = default_config.copy()
        full_config.update(analysis_result)
        full_config.update(user_config)

        # Verify merged configuration
        assert full_config["max_workers"] == 4  # From default
        assert full_config["source_catalogue"] == "/test/catalogue.csv"  # From user
        assert full_config["num_sources"] == 25  # From analysis
        assert full_config["target_resolution"] == 256  # From user

        # Check all required keys are present
        required_keys = [
            "source_catalogue",
            "output_dir",
            "selected_extensions",
            "num_sources",
            "fits_files",
            "available_extensions",
            "target_resolution",
            "stretch",
        ]
        for key in required_keys:
            assert key in full_config, f"Missing required key: {key}"

    def test_validation_scenarios(self):
        """Test validation scenarios (synchronous)."""
        # Test empty extensions validation
        invalid_config = {"selected_extensions": [], "output_dir": "/test/output"}

        # Simulate validation logic
        validation_errors = []

        if not invalid_config.get("selected_extensions"):
            validation_errors.append("Please select at least one FITS extension")

        if not invalid_config.get("output_dir"):
            validation_errors.append("Please select an output directory")

        assert len(validation_errors) == 1
        assert "select at least one FITS extension" in validation_errors[0]

        # Test missing output directory validation
        invalid_config2 = {
            "selected_extensions": [{"name": "VIS", "ext": "IMAGE"}],
            "output_dir": None,
        }

        validation_errors2 = []
        if not invalid_config2.get("selected_extensions"):
            validation_errors2.append("Please select at least one FITS extension")
        if not invalid_config2.get("output_dir"):
            validation_errors2.append("Please select an output directory")

        assert len(validation_errors2) == 1
        assert "select an output directory" in validation_errors2[0]

        # Test valid configuration
        valid_config = {
            "selected_extensions": [{"name": "VIS", "ext": "IMAGE"}],
            "output_dir": "/test/output",
        }

        validation_errors3 = []
        if not valid_config.get("selected_extensions"):
            validation_errors3.append("Please select at least one FITS extension")
        if not valid_config.get("output_dir"):
            validation_errors3.append("Please select an output directory")

        assert len(validation_errors3) == 0


class TestBackendInterfaceReal:
    """Test backend interface with real data and functions."""

    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory."""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def small_catalogue(self, test_data_dir):
        """Path to small real catalogue."""
        return str(test_data_dir / "euclid_cutana_catalogue_small.csv")

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp(prefix="cutana_test_")
        yield temp_dir
        # Cleanup handled by tempfile

    def test_check_source_catalogue_data_structure(self, small_catalogue):
        """Test that check_source_catalogue returns expected data structure."""
        # This test verifies the actual return structure of the backend analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                BackendInterface.check_source_catalogue(small_catalogue)
            )

            # Verify expected keys are present
            assert "num_sources" in result
            assert "fits_files" in result or "num_unique_fits_files" in result
            assert "extensions" in result

            # Verify data types
            assert isinstance(result["num_sources"], int)
            assert result["num_sources"] > 0

            # Check extensions structure
            extensions = result["extensions"]
            assert isinstance(extensions, list)
            assert len(extensions) > 0

            # Each extension should have name and ext
            for ext in extensions:
                assert "name" in ext
                assert "ext" in ext
                assert isinstance(ext["name"], str)
                assert isinstance(ext["ext"], str)

            print("\nBackend analysis result structure:")
            print(f"   Sources: {result['num_sources']}")
            print(f"   Extensions: {[ext['name'] for ext in extensions]}")
            print(f"   Full result keys: {list(result.keys())}")

            # Store result for other tests to use
            TestBackendInterfaceReal._analysis_result = result

        finally:
            loop.close()

    def test_preview_cutouts_data_structure(self, small_catalogue):
        """Test that generate_previews returns expected data structure."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # First get analysis to understand available extensions
            analysis_result = loop.run_until_complete(
                BackendInterface.check_source_catalogue(small_catalogue)
            )

            # Create a config with selected extensions using default config
            from cutana.get_default_config import get_default_config

            config = get_default_config()
            config.source_catalogue = small_catalogue
            config.selected_extensions = analysis_result["extensions"][:1]  # Use first extension
            config.target_resolution = 64  # Small for faster testing
            config.normalisation_method = "linear"
            config.apply_flux_conversion = False  # Disable for test data without MAGZERO header

            # Load sources into cache first (this is what generate_previews expects)
            loop.run_until_complete(
                BackendInterface.load_sources_for_previews(
                    catalogue_path=small_catalogue,
                    config=config,
                )
            )

            # Test preview generation using generate_previews instead of get_preview_cutouts
            cutouts = loop.run_until_complete(
                BackendInterface.generate_previews(
                    num_samples=2,  # Small number for testing
                    size=64,
                    config=config,
                )
            )

            # Verify structure
            assert isinstance(cutouts, list)
            assert len(cutouts) > 0
            assert len(cutouts) <= 2  # Should not exceed requested samples

            # Each cutout should be (ra, dec, array)
            for ra, dec, cutout_array in cutouts:
                assert isinstance(ra, float)
                assert isinstance(dec, float)
                assert hasattr(cutout_array, "shape")  # Should be numpy-like array
                print(f"\nPreview cutout: RA={ra:.2f}, Dec={dec:.2f}, Shape={cutout_array.shape}")

            print(f"\nSuccessfully generated {len(cutouts)} preview cutouts")

        finally:
            loop.close()

    def test_processing_config_structure(self, small_catalogue, temp_output_dir):
        """Test the processing configuration structure expected by start_processing."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Get analysis result
            analysis_result = loop.run_until_complete(
                BackendInterface.check_source_catalogue(small_catalogue)
            )

            # Create a complete config as would be generated by the UI
            processing_config = {
                "source_catalogue": small_catalogue,
                "output_dir": temp_output_dir,
                "output_format": "zarr",
                "data_type": "float32",
                "max_workers": 1,  # Single worker for testing
                "target_resolution": 64,  # Small for faster testing
                "selected_extensions": analysis_result["extensions"][:1],  # Use first extension
                "normalisation_method": "linear",
                "interpolation": "bilinear",
                "num_sources": analysis_result["num_sources"],
            }

            print("\nProcessing configuration structure:")
            for key, value in processing_config.items():
                if isinstance(value, str) and len(value) > 50:
                    print(f"   {key}: ...{value[-40:]}")
                else:
                    print(f"   {key}: {value}")

            # Verify all required keys are present
            required_keys = [
                "source_catalogue",
                "output_dir",
                "output_format",
                "data_type",
                "max_workers",
                "target_resolution",
                "selected_extensions",
            ]

            for key in required_keys:
                assert key in processing_config, f"Missing required config key: {key}"

            print("\nAll required configuration keys present")

            # Note: We don't actually run start_processing here as it's slow
            # But we verify the config structure is complete

        finally:
            loop.close()

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Test with non-existent file
            with pytest.raises(FileNotFoundError):
                loop.run_until_complete(
                    BackendInterface.check_source_catalogue("/non/existent/file.csv")
                )

            print("\nError handling works for non-existent files")

        finally:
            loop.close()

    def test_load_sources_for_previews_data_structure(self, small_catalogue):
        """Test that load_sources_for_previews returns expected data structure."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # First get analysis to understand available extensions
            analysis_result = loop.run_until_complete(
                BackendInterface.check_source_catalogue(small_catalogue)
            )

            # Create a config with selected extensions using default config
            from cutana.get_default_config import get_default_config

            config = get_default_config()
            config.source_catalogue = small_catalogue
            config.selected_extensions = analysis_result["extensions"][:1]
            config.target_resolution = 64
            config.normalisation_method = "linear"
            config.apply_flux_conversion = False  # Disable for test data without MAGZERO header

            # Test source loading
            cache_info = loop.run_until_complete(
                BackendInterface.load_sources_for_previews(
                    catalogue_path=small_catalogue,
                    config=config,
                )
            )

            # Verify cache info structure
            assert isinstance(cache_info, dict)
            assert "status" in cache_info
            assert cache_info["status"] == "success"
            assert "num_cached_sources" in cache_info
            assert "num_cached_fits" in cache_info
            assert "fits_file_sets" in cache_info
            assert "source_size_range" in cache_info

            # Verify cache contains data
            assert cache_info["num_cached_sources"] > 0
            assert cache_info["num_cached_fits"] > 0
            assert cache_info["fits_file_sets"] > 0

            print(
                f"\nCache info: {cache_info['num_cached_sources']} sources, {cache_info['num_cached_fits']} FITS files"
            )

        finally:
            loop.close()

    def test_generate_previews_with_cache(self, small_catalogue):
        """Test that generate_previews works with cached data."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # First get analysis to understand available extensions
            analysis_result = loop.run_until_complete(
                BackendInterface.check_source_catalogue(small_catalogue)
            )

            # Create a config with selected extensions using default config
            from cutana.get_default_config import get_default_config

            config = get_default_config()
            config.source_catalogue = small_catalogue
            config.selected_extensions = analysis_result["extensions"][:1]
            config.target_resolution = 64
            config.normalisation_method = "linear"
            config.apply_flux_conversion = False  # Disable for test data without MAGZERO header

            # Load sources into cache
            loop.run_until_complete(
                BackendInterface.load_sources_for_previews(
                    catalogue_path=small_catalogue,
                    config=config,
                )
            )

            # Generate previews using cache
            cutouts = loop.run_until_complete(
                BackendInterface.generate_previews(
                    num_samples=3,  # Small number for testing
                    size=64,
                    config=config,
                )
            )

            # Verify structure
            assert isinstance(cutouts, list)
            assert len(cutouts) > 0
            assert len(cutouts) <= 3  # Should not exceed requested samples

            # Each cutout should be (ra, dec, array)
            for ra, dec, cutout_array in cutouts:
                assert isinstance(ra, float)
                assert isinstance(dec, float)
                assert hasattr(cutout_array, "shape")  # Should be numpy-like array
                print(
                    f"\nCached preview cutout: RA={ra:.2f}, Dec={dec:.2f}, Shape={cutout_array.shape}"
                )

            print(f"\nSuccessfully generated {len(cutouts)} preview cutouts from cache")

        finally:
            loop.close()

    def test_generate_previews_without_cache(self, small_catalogue):
        """Test that generate_previews falls back to full pipeline without cache."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Clear any existing cache
            clear_preview_cache()

            # First get analysis to understand available extensions
            analysis_result = loop.run_until_complete(
                BackendInterface.check_source_catalogue(small_catalogue)
            )

            # Create a config with selected extensions using default config
            from cutana.get_default_config import get_default_config

            config = get_default_config()
            config.source_catalogue = small_catalogue
            config.selected_extensions = analysis_result["extensions"][:1]
            config.target_resolution = 64
            config.normalisation_method = "linear"
            config.apply_flux_conversion = False  # Disable for test data without MAGZERO header

            # Generate previews without cache (should fall back)
            cutouts = loop.run_until_complete(
                BackendInterface.generate_previews(
                    num_samples=2,  # Small number for testing
                    size=64,
                    config=config,
                )
            )

            # Verify structure (should work the same)
            assert isinstance(cutouts, list)
            assert len(cutouts) > 0
            assert len(cutouts) <= 2

            print(f"\nFallback preview generation worked: {len(cutouts)} cutouts")

        finally:
            loop.close()

    def test_cache_persistence(self, small_catalogue):
        """Test that cache persists between calls."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # First get analysis to understand available extensions
            analysis_result = loop.run_until_complete(
                BackendInterface.check_source_catalogue(small_catalogue)
            )

            # Create a config with selected extensions using default config
            from cutana.get_default_config import get_default_config

            config = get_default_config()
            config.source_catalogue = small_catalogue
            config.selected_extensions = analysis_result["extensions"][:1]
            config.target_resolution = 64
            config.normalisation_method = "linear"
            config.apply_flux_conversion = False  # Disable for test data without MAGZERO header

            # Load sources into cache
            loop.run_until_complete(
                BackendInterface.load_sources_for_previews(
                    catalogue_path=small_catalogue,
                    config=config,
                )
            )

            # Verify cache is available
            cache_status = get_cache_status()
            assert cache_status["cached"] is True
            assert cache_status["num_sources"] > 0
            assert cache_status["num_fits"] > 0

            # Generate previews using cache (first call)
            cutouts1 = loop.run_until_complete(
                BackendInterface.generate_previews(
                    num_samples=2,
                    size=64,
                    config=config,
                )
            )

            # Generate previews using cache (second call)
            cutouts2 = loop.run_until_complete(
                BackendInterface.generate_previews(
                    num_samples=2,
                    size=64,
                    config=config,
                )
            )

            # Both calls should succeed
            assert len(cutouts1) > 0
            assert len(cutouts2) > 0

            print(f"\nCache persistence test passed: {len(cutouts1)} + {len(cutouts2)} cutouts")

        finally:
            loop.close()
