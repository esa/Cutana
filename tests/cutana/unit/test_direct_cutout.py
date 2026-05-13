#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for the direct_cutout module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cutana.direct_cutout import create_cutouts_direct
from cutana.get_default_config import get_default_config


@pytest.fixture
def simple_config():
    """Create a minimal valid config for testing."""
    cfg = get_default_config()
    cfg.target_resolution = 64
    cfg.normalisation_method = "linear"
    cfg.fits_extensions = ["PRIMARY"]
    cfg.channel_weights = {"PRIMARY": [1.0]}
    cfg.data_type = "float32"
    cfg.padding_factor = 1.0
    cfg.interpolation = "bilinear"
    cfg.flux_conserved_resizing = False
    cfg.do_only_cutout_extraction = False
    cfg.apply_flux_conversion = False
    cfg.output_format = "zarr"
    cfg.process_id = "test_direct"
    return cfg


@pytest.fixture
def simple_catalogue():
    """Create a minimal valid catalogue DataFrame."""
    return pd.DataFrame(
        [
            {
                "SourceID": "SRC_001",
                "RA": 180.0,
                "Dec": 0.0,
                "diameter_pixel": 32,
                "fits_file_paths": "['test_file.fits']",
            },
            {
                "SourceID": "SRC_002",
                "RA": 180.001,
                "Dec": 0.001,
                "diameter_pixel": 48,
                "fits_file_paths": "['test_file.fits']",
            },
        ]
    )


class TestCreateCutoutsDirect:
    """Unit tests for create_cutouts_direct."""

    def test_empty_dataframe_raises_value_error(self, simple_config):
        """Empty DataFrame should raise ValueError."""
        empty_df = pd.DataFrame(
            columns=["SourceID", "RA", "Dec", "diameter_pixel", "fits_file_paths"]
        )
        with pytest.raises(ValueError, match="Empty catalogue"):
            create_cutouts_direct(empty_df, simple_config)

    def test_missing_required_columns_raises_key_error(self, simple_config):
        """Missing required columns should raise KeyError."""
        df = pd.DataFrame([{"SourceID": "S1", "RA": 1.0}])
        with pytest.raises(KeyError, match="Missing required columns"):
            create_cutouts_direct(df, simple_config)

    def test_missing_size_columns_raises_key_error(self, simple_config):
        """Missing both diameter columns should raise KeyError."""
        df = pd.DataFrame(
            [
                {
                    "SourceID": "S1",
                    "RA": 1.0,
                    "Dec": 1.0,
                    "fits_file_paths": "['f.fits']",
                }
            ]
        )
        with pytest.raises(KeyError, match="diameter"):
            create_cutouts_direct(df, simple_config)

    def test_calls_processing_pipeline(self, simple_catalogue, simple_config):
        """Verify that the function calls the core processing functions."""
        with (
            patch("cutana.direct_cutout.prepare_fits_sets_and_sources") as mock_prepare,
            patch("cutana.direct_cutout.load_fits_sets") as mock_load,
            patch(
                "cutana.direct_cutout._process_sources_batch_vectorized_with_fits_set"
            ) as mock_process,
        ):
            mock_hdul = MagicMock()
            mock_prepare.return_value = {("test_file.fits",): simple_catalogue.to_dict("records")}
            mock_load.return_value = {"test_file.fits": (mock_hdul, {"PRIMARY": MagicMock()})}
            mock_process.return_value = [
                {
                    "cutouts": np.zeros((2, 64, 64, 1), dtype=np.float32),
                    "metadata": [
                        {"source_id": "SRC_001", "ra": 180.0, "dec": 0.0},
                        {"source_id": "SRC_002", "ra": 180.001, "dec": 0.001},
                    ],
                }
            ]

            results = create_cutouts_direct(simple_catalogue, simple_config)

            mock_prepare.assert_called_once()
            mock_load.assert_called_once()
            mock_process.assert_called_once()
            # FITS files should be closed
            mock_hdul.close.assert_called_once()

            assert len(results) == 1
            assert results[0]["cutouts"].shape == (2, 64, 64, 1)
            assert len(results[0]["metadata"]) == 2

    def test_config_not_mutated(self, simple_catalogue, simple_config):
        """Config should not be modified by the function."""
        original_target_res = simple_config.target_resolution

        with (
            patch("cutana.direct_cutout.prepare_fits_sets_and_sources") as mock_prepare,
            patch("cutana.direct_cutout.load_fits_sets") as mock_load,
            patch(
                "cutana.direct_cutout._process_sources_batch_vectorized_with_fits_set"
            ) as mock_process,
        ):
            mock_prepare.return_value = {("f.fits",): []}
            mock_load.return_value = {"f.fits": (MagicMock(), {})}
            mock_process.return_value = [
                {
                    "cutouts": np.zeros((1, 64, 64, 1)),
                    "metadata": [{"source_id": "S1"}],
                }
            ]

            create_cutouts_direct(simple_catalogue, simple_config)

        assert simple_config.target_resolution == original_target_res

    def test_fits_cleanup_on_error(self, simple_catalogue, simple_config):
        """FITS files should be closed even when processing fails."""
        mock_hdul = MagicMock()

        with (
            patch("cutana.direct_cutout.prepare_fits_sets_and_sources") as mock_prepare,
            patch("cutana.direct_cutout.load_fits_sets") as mock_load,
            patch(
                "cutana.direct_cutout._process_sources_batch_vectorized_with_fits_set"
            ) as mock_process,
        ):
            mock_prepare.return_value = {("test_file.fits",): simple_catalogue.to_dict("records")}
            mock_load.return_value = {"test_file.fits": (mock_hdul, {"PRIMARY": MagicMock()})}
            mock_process.side_effect = Exception("Processing failed")

            with pytest.raises(Exception, match="Processing failed"):
                create_cutouts_direct(simple_catalogue, simple_config)

        # FITS should still be closed despite the error
        mock_hdul.close.assert_called_once()

    def test_multiple_fits_sets(self, simple_config):
        """Sources using different FITS sets should all be processed."""
        df = pd.DataFrame(
            [
                {
                    "SourceID": "A1",
                    "RA": 180.0,
                    "Dec": 0.0,
                    "diameter_pixel": 32,
                    "fits_file_paths": "['setA.fits']",
                },
                {
                    "SourceID": "B1",
                    "RA": 181.0,
                    "Dec": 1.0,
                    "diameter_pixel": 32,
                    "fits_file_paths": "['setB.fits']",
                },
            ]
        )

        with (
            patch("cutana.direct_cutout.prepare_fits_sets_and_sources") as mock_prepare,
            patch("cutana.direct_cutout.load_fits_sets") as mock_load,
            patch(
                "cutana.direct_cutout._process_sources_batch_vectorized_with_fits_set"
            ) as mock_process,
        ):
            mock_prepare.return_value = {
                ("setA.fits",): [df.iloc[0].to_dict()],
                ("setB.fits",): [df.iloc[1].to_dict()],
            }
            mock_load.return_value = {
                "setA.fits": (MagicMock(), {"PRIMARY": MagicMock()}),
                "setB.fits": (MagicMock(), {"PRIMARY": MagicMock()}),
            }
            mock_process.return_value = [
                {
                    "cutouts": np.zeros((1, 64, 64, 1)),
                    "metadata": [{"source_id": "X"}],
                }
            ]

            results = create_cutouts_direct(df, simple_config)

            # Should be called once per FITS set
            assert mock_process.call_count == 2
            assert len(results) == 2
