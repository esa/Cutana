#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Integration tests for create_cutouts_direct with real FITS data from test_data/."""

from pathlib import Path

import numpy as np
import pytest

from cutana.catalogue_preprocessor import load_catalogue
from cutana.direct_cutout import create_cutouts_direct
from cutana.get_default_config import get_default_config

TEST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "test_data"


@pytest.fixture
def test_config():
    """Create config suitable for integration testing."""
    cfg = get_default_config()
    cfg.target_resolution = 64
    cfg.normalisation_method = "linear"
    cfg.fits_extensions = ["PRIMARY"]
    cfg.selected_extensions = [{"ext": "PRIMARY", "name": "VIS"}]
    cfg.channel_weights = {"PRIMARY": [1.0]}
    cfg.data_type = "float32"
    cfg.padding_factor = 1.0
    cfg.interpolation = "bilinear"
    cfg.flux_conserved_resizing = False
    cfg.do_only_cutout_extraction = False
    cfg.apply_flux_conversion = False
    cfg.output_format = "zarr"
    cfg.process_id = "test_integration"
    return cfg


@pytest.fixture
def small_catalogue():
    """Load the small test catalogue."""
    catalogue_path = TEST_DATA_DIR / "euclid_cutana_catalogue_small.csv"
    if not catalogue_path.exists():
        pytest.skip("Test data not available - run generate_test_data.py")
    return load_catalogue(str(catalogue_path))


class TestDirectCutoutIntegration:
    """Integration tests using real test FITS files."""

    def test_single_channel_output(self, small_catalogue, test_config):
        """Test that single-channel processing produces correct output shape."""
        results = create_cutouts_direct(small_catalogue, test_config)

        assert len(results) >= 1
        for result in results:
            cutouts = result["cutouts"]
            metadata = result["metadata"]
            assert isinstance(cutouts, np.ndarray)
            assert cutouts.ndim == 4  # (N, H, W, C)
            assert cutouts.shape[1] == 64  # target_resolution
            assert cutouts.shape[2] == 64
            assert cutouts.shape[3] == 1  # single channel
            assert len(metadata) == cutouts.shape[0]

    def test_different_target_resolutions(self, small_catalogue, test_config):
        """Test that different target resolutions produce correct shapes."""
        for resolution in [32, 128]:
            test_config.target_resolution = resolution
            results = create_cutouts_direct(small_catalogue, test_config)

            for result in results:
                assert result["cutouts"].shape[1] == resolution
                assert result["cutouts"].shape[2] == resolution

    @pytest.mark.parametrize("norm_method", ["linear", "log", "asinh", "zscale"])
    def test_normalisation_methods(self, small_catalogue, test_config, norm_method):
        """Test that different normalisation methods produce valid output."""
        test_config.normalisation_method = norm_method
        results = create_cutouts_direct(small_catalogue, test_config)

        for result in results:
            cutouts = result["cutouts"]
            assert not np.all(np.isnan(cutouts)), f"All NaN for {norm_method}"
            assert cutouts.shape[1] == 64

    def test_do_only_cutout_extraction(self, small_catalogue, test_config):
        """Test raw cutout extraction mode (no resizing)."""
        test_config.do_only_cutout_extraction = True
        results = create_cutouts_direct(small_catalogue, test_config)

        assert len(results) >= 1
        for result in results:
            # In raw mode, cutouts is a list not a tensor
            cutouts = result["cutouts"]
            assert isinstance(cutouts, list)
            assert len(cutouts) > 0

    def test_metadata_contains_required_fields(self, small_catalogue, test_config):
        """Test that metadata has the expected fields."""
        results = create_cutouts_direct(small_catalogue, test_config)

        for result in results:
            for meta in result["metadata"]:
                assert "source_id" in meta
                assert "ra" in meta
                assert "dec" in meta

    def test_subset_of_catalogue(self, small_catalogue, test_config):
        """Test processing a subset of the catalogue."""
        subset = small_catalogue.head(3)
        results = create_cutouts_direct(subset, test_config)

        total_cutouts = sum(len(r["metadata"]) for r in results)
        assert total_cutouts == 3
