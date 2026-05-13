#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end tests for create_cutouts_direct, including output equivalence
with the StreamingOrchestrator subprocess pipeline.

Uses real FITS data from tests/test_data/.
"""

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cutana import StreamingOrchestrator, get_default_config
from cutana.catalogue_preprocessor import load_catalogue
from cutana.direct_cutout import create_cutouts_direct
from cutana.preview_generator import (
    clear_preview_cache,
    generate_previews,
    load_sources_for_previews,
)

TEST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "test_data"


def _make_config(output_dir: str):
    """Create a consistent config for both direct and streaming paths."""
    cfg = get_default_config()
    cfg.target_resolution = 32
    cfg.normalisation_method = "linear"
    cfg.fits_extensions = ["PRIMARY"]
    cfg.selected_extensions = ["VIS"]
    cfg.channel_weights = {"VIS": [1.0]}
    cfg.data_type = "float32"
    cfg.padding_factor = 1.0
    cfg.interpolation = "bilinear"
    cfg.flux_conserved_resizing = False
    cfg.do_only_cutout_extraction = False
    cfg.apply_flux_conversion = False
    cfg.output_format = "zarr"
    cfg.output_dir = output_dir
    cfg.log_level = "WARNING"
    cfg.console_log_level = "WARNING"
    cfg.skip_catalogue_validation = True
    cfg.max_workflow_time_seconds = 600
    cfg.process_id = "test_equiv"
    return cfg


@pytest.fixture
def small_catalogue_path():
    """Path to the small test catalogue."""
    path = TEST_DATA_DIR / "euclid_cutana_catalogue_small.csv"
    if not path.exists():
        pytest.skip("Test data not available - run generate_test_data.py")
    return path


@pytest.fixture
def medium_catalogue_path():
    """Path to the medium test catalogue."""
    path = TEST_DATA_DIR / "euclid_cutana_catalogue_medium.csv"
    if not path.exists():
        pytest.skip("Test data not available - run generate_test_data.py")
    return path


class TestOutputEquivalence:
    """Test that create_cutouts_direct produces identical output to StreamingOrchestrator."""

    def test_direct_vs_streaming_identical_cutouts(self, small_catalogue_path):
        """Core equivalence test: same sources must produce identical cutout arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            cfg.source_catalogue = str(small_catalogue_path)

            # --- Direct path ---
            catalogue_df = load_catalogue(str(small_catalogue_path))
            direct_results = create_cutouts_direct(catalogue_df, cfg)

            # Collect all direct cutouts and metadata, keyed by source_id
            direct_cutouts = {}
            for result in direct_results:
                for i, meta in enumerate(result["metadata"]):
                    sid = meta["source_id"]
                    direct_cutouts[sid] = result["cutouts"][i]

            # --- Streaming path (subprocess) ---
            orchestrator = StreamingOrchestrator(cfg)
            try:
                orchestrator.init_streaming(
                    batch_size=len(catalogue_df),
                    write_to_disk=False,
                )
                streaming_cutouts = {}
                for _ in range(orchestrator.get_batch_count()):
                    batch = orchestrator.next_batch()
                    for i, meta in enumerate(batch["metadata"]):
                        sid = meta["source_id"]
                        streaming_cutouts[sid] = batch["cutouts"][i]
            finally:
                orchestrator.cleanup()

            # --- Compare ---
            # Both paths should have the same source IDs
            direct_ids = set(direct_cutouts.keys())
            streaming_ids = set(streaming_cutouts.keys())
            assert direct_ids == streaming_ids, (
                f"Source ID mismatch: direct has {len(direct_ids)}, "
                f"streaming has {len(streaming_ids)}, "
                f"diff: {direct_ids.symmetric_difference(streaming_ids)}"
            )

            # Cutout arrays must be numerically identical
            for sid in direct_ids:
                np.testing.assert_array_equal(
                    direct_cutouts[sid],
                    streaming_cutouts[sid],
                    err_msg=f"Cutout mismatch for source {sid}",
                )


class TestDirectCutoutE2E:
    """End-to-end tests for create_cutouts_direct with real data."""

    def test_e2e_with_small_catalogue(self, small_catalogue_path):
        """Full end-to-end test with the small test catalogue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            catalogue_df = load_catalogue(str(small_catalogue_path))

            results = create_cutouts_direct(catalogue_df, cfg)

            total_cutouts = sum(len(r["metadata"]) for r in results)
            assert total_cutouts == len(catalogue_df)

            for result in results:
                cutouts = result["cutouts"]
                assert isinstance(cutouts, np.ndarray)
                assert cutouts.ndim == 4
                assert cutouts.shape[1:3] == (32, 32)

    def test_e2e_with_medium_catalogue(self, medium_catalogue_path):
        """End-to-end test with the medium test catalogue (100 sources)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            catalogue_df = load_catalogue(str(medium_catalogue_path))

            results = create_cutouts_direct(catalogue_df, cfg)

            total_cutouts = sum(len(r["metadata"]) for r in results)
            assert total_cutouts == len(catalogue_df)

    def test_preview_generator_regression(self, small_catalogue_path):
        """Regression test: preview_generator still works after refactoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            cfg.selected_extensions = [{"ext": "PRIMARY", "name": "VIS"}]

            clear_preview_cache()

            loop = asyncio.new_event_loop()
            try:
                # Load sources
                cache_info = loop.run_until_complete(
                    load_sources_for_previews(str(small_catalogue_path), cfg)
                )
                assert cache_info["status"] == "success"
                assert cache_info["num_cached_sources"] > 0

                # Generate previews
                previews = loop.run_until_complete(
                    generate_previews(num_samples=3, size=64, config=cfg)
                )
                assert len(previews) >= 1
                for ra, dec, cutout_array in previews:
                    assert isinstance(cutout_array, np.ndarray)
                    assert cutout_array.dtype == np.uint8
            finally:
                clear_preview_cache()
                loop.close()
