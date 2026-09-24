#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Integration tests for create_cutouts_direct with real FITS data from test_data/."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger

from cutana.catalogue_preprocessor import load_catalogue
from cutana.direct_cutout import create_cutouts_direct
from cutana.get_default_config import get_default_config

TEST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "test_data"


def _write_tiny_tile(path: Path) -> str:
    """Write a minimal VIS-like FITS tile (256x256, 0.1"/px) for self-contained tests.

    CI has no real mosaic data, and the size-conversion regression this guards is
    independent of pixel content, so a synthetic tile is sufficient and portable.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [180.0, 0.0]
    wcs.wcs.crpix = [128, 128]
    wcs.wcs.cd = [[-0.1 / 3600.0, 0.0], [0.0, 0.1 / 3600.0]]  # 0.1 arcsec/pixel
    data = np.random.default_rng(0).normal(0.0, 1.0, (256, 256)).astype(np.float32)
    fits.PrimaryHDU(data, header=wcs.to_header()).writeto(path, overwrite=True)
    return path.as_posix()


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


class TestDiameterSizeHandling:
    """Regression tests for diameter→pixel size conversion in metadata.

    Self-contained: a synthetic tile is generated per test, so these run in CI
    without the real Euclid mosaics.
    """

    @pytest.fixture
    def tile(self, tmp_path):
        return _write_tiny_tile(tmp_path / "tiny_VIS.fits")

    def _catalogue(self, tile, size_col, size_val, n=3):
        # Coordinates within a few pixels of the tile centre (RA=180, Dec=0).
        coords = [(180.0, 0.0), (180.0002, 0.0002), (179.9998, -0.0002)]
        return pd.DataFrame(
            [
                {
                    "SourceID": f"S{i}",
                    "RA": coords[i][0],
                    "Dec": coords[i][1],
                    size_col: size_val,
                    "fits_file_paths": f"['{tile}']",
                }
                for i in range(n)
            ]
        )

    def test_diameter_pixel_only_no_cast_warning(self, test_config, tile):
        """diameter_pixel-only catalogues must not raise the cast RuntimeWarning.

        The size computation builds an all-NaN diameter_arcsec array here; the
        masked-out int cast used to emit "invalid value encountered in cast" for
        every source. Promote RuntimeWarning to an error to catch a regression.
        """
        df = self._catalogue(tile, "diameter_pixel", 64)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            results = create_cutouts_direct(df, test_config, max_workers=1)

        sizes = [m["original_cutout_size"] for r in results for m in r["metadata"]]
        assert sizes == [64, 64, 64]

    def test_subpixel_diameter_arcsec_is_graceful(self, test_config, tile):
        """diameter_arcsec smaller than one pixel yields a None size, not a crash."""
        # 0.05" at 0.1"/px rounds to 0 px; cutout still extracted (clamped to >=1 px).
        df = self._catalogue(tile, "diameter_arcsec", 0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            results = create_cutouts_direct(df, test_config, max_workers=1)

        sizes = [m["original_cutout_size"] for r in results for m in r["metadata"]]
        assert all(s is None for s in sizes)

    def test_subpixel_emits_one_aggregated_warning(self, test_config, tile):
        """Sub-pixel sources are flagged once per batch with a count, not per source."""
        # cutana disables its logger by default (see cutana/__init__.py); enable it
        # so the per-batch warning reaches our sink, then restore the default.
        messages = []
        logger.enable("cutana")
        sink_id = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            df = self._catalogue(tile, "diameter_arcsec", 0.05, n=3)  # 3 sub-pixel sources
            create_cutouts_direct(df, test_config, max_workers=1)
        finally:
            logger.remove(sink_id)
            logger.disable("cutana")

        subpixel = [str(m) for m in messages if "below one pixel" in str(m)]
        assert len(subpixel) == 1  # one aggregated warning, not three
        assert "3/3" in subpixel[0]

    def test_normal_diameter_arcsec_converts_to_pixels(self, test_config, tile):
        """A normal diameter_arcsec converts to the expected pixel size."""
        # 6.4" at 0.1"/px -> 64 px.
        df = self._catalogue(tile, "diameter_arcsec", 6.4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            results = create_cutouts_direct(df, test_config, max_workers=1)

        sizes = [m["original_cutout_size"] for r in results for m in r["metadata"]]
        assert sizes == [64, 64, 64]
