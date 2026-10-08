#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for the direct_cutout module."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from cutana.direct_cutout import (
    _MAX_AUTO_WORKERS,
    _resolve_worker_count,
    create_cutouts_direct,
)
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


class TestResolveWorkerCount:
    """Unit tests for the worker-count resolution helper."""

    def _patch_cpus(self, count):
        """Patch the k8s-aware CPU probe to return a fixed effective core count."""
        mock_sm = MagicMock()
        mock_sm.get_effective_cpu_count.return_value = count
        return patch("cutana.direct_cutout.SystemMonitor", return_value=mock_sm)

    def test_single_set_is_always_serial(self):
        """A single FITS set never spawns threads (no CPU probe needed)."""
        with self._patch_cpus(16) as probe:
            assert _resolve_worker_count(None, 1) == 1
            assert _resolve_worker_count(8, 1) == 1
            assert _resolve_worker_count(1, 0) == 1
            # The single-set short-circuit must not probe the CPU limit.
            probe.return_value.get_effective_cpu_count.assert_not_called()

    def test_auto_bounded_by_sets_cpus_and_cap(self):
        """None auto-selects min(n_sets, effective_cpus, cap)."""
        # Bounded by the number of sets.
        with self._patch_cpus(16):
            assert _resolve_worker_count(None, 2) == 2
        # Bounded by the (k8s-aware) CPU count.
        with self._patch_cpus(3):
            assert _resolve_worker_count(None, 100) == 3
        # Bounded by the hard cap when sets and CPUs are both larger.
        with self._patch_cpus(_MAX_AUTO_WORKERS * 8):
            assert _resolve_worker_count(None, 100) == _MAX_AUTO_WORKERS

    def test_explicit_workers_skip_cpu_probe_and_cap_to_sets(self):
        """An explicit count is capped at n_sets and must not probe the CPU limit."""
        with self._patch_cpus(16) as probe:
            assert _resolve_worker_count(2, 5) == 2
            assert _resolve_worker_count(10, 3) == 3
            probe.return_value.get_effective_cpu_count.assert_not_called()

    def test_invalid_workers_raises(self):
        """max_workers < 1 (with multiple sets) is rejected."""
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            _resolve_worker_count(0, 3)
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            _resolve_worker_count(-1, 3)


class TestParallelTileProcessing:
    """Tests for concurrent processing of multiple FITS sets."""

    def _patches(self, set_to_sources, load_return, process_return):
        return (
            patch(
                "cutana.direct_cutout.prepare_fits_sets_and_sources", return_value=set_to_sources
            ),
            patch("cutana.direct_cutout.load_fits_sets", side_effect=load_return),
            patch(
                "cutana.direct_cutout._process_sources_batch_vectorized_with_fits_set",
                side_effect=process_return,
            ),
        )

    def test_parallel_processes_all_sets_in_order(self, simple_config):
        """With max_workers>1, every set is processed and results keep set order."""
        df = pd.DataFrame(
            [
                {
                    "SourceID": f"S{i}",
                    "RA": 180.0 + i,
                    "Dec": float(i),
                    "diameter_pixel": 32,
                    "fits_file_paths": f"['tile{i}.fits']",
                }
                for i in range(4)
            ]
        )
        set_to_sources = {(f"tile{i}.fits",): [df.iloc[i].to_dict()] for i in range(4)}

        # _process_one_fits_set calls load_fits_sets([fits_set], ...) with exactly one
        # set, so unpack that single (path-tuple) set and return one entry per path.
        def fake_load(fits_sets, _ext):
            (fits_set,) = fits_sets
            return {path: (MagicMock(), {"PRIMARY": MagicMock()}) for path in fits_set}

        # Tag each result with its source so we can assert ordering is preserved.
        def fake_process(sources, _data, _cfg, profiler=None):
            return [{"metadata": [{"source_id": sources[0]["SourceID"]}]}]

        p1, p2, p3 = self._patches(set_to_sources, fake_load, fake_process)
        with p1, p2, p3:
            results = create_cutouts_direct(df, simple_config, max_workers=4)

        assert [r["metadata"][0]["source_id"] for r in results] == ["S0", "S1", "S2", "S3"]

    def test_selected_extensions_narrows_the_set_before_loading(self, simple_config):
        """The fix itself: create_cutouts_direct must load only the selected bands.

        Without the narrowing, the direct path loads every file in the set and
        combine_channels pairs three NISP weights with VIS, NIR-H, NIR-Y — the
        #420 failure. Nothing else in the suite fails if that one line goes.
        """
        euclid_set = (
            "/d/EUC_MER_BGSUB-MOSAIC-VIS_TILE1-A.fits",
            "/d/EUC_MER_BGSUB-MOSAIC-NIR-H_TILE1-B.fits",
            "/d/EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE1-C.fits",
            "/d/EUC_MER_BGSUB-MOSAIC-NIR-J_TILE1-D.fits",
        )
        df = pd.DataFrame(
            [
                {
                    "SourceID": "S0",
                    "RA": 180.0,
                    "Dec": 0.0,
                    "diameter_pixel": 32,
                    "fits_file_paths": str(list(euclid_set)),
                }
            ]
        )
        simple_config.selected_extensions = [
            {"name": "NIR-H", "ext": "PRIMARY"},
            {"name": "NIR-Y", "ext": "PRIMARY"},
            {"name": "NIR-J", "ext": "PRIMARY"},
        ]
        loaded = []

        def fake_load(fits_sets, _ext):
            (fits_set,) = fits_sets
            loaded.append(fits_set)
            return {path: (MagicMock(), {"PRIMARY": MagicMock()}) for path in fits_set}

        def fake_process(sources, _data, _cfg, profiler=None):
            return [{"metadata": [{"source_id": sources[0]["SourceID"]}]}]

        p1, p2, p3 = self._patches({euclid_set: [df.iloc[0].to_dict()]}, fake_load, fake_process)
        with p1, p2, p3:
            create_cutouts_direct(df, simple_config, max_workers=1)

        # The three requested bands, in catalogue order — VIS must not be loaded.
        assert loaded == [euclid_set[1:]]

    def test_a_set_with_no_requested_band_fails_the_call(self, simple_config):
        """The direct path raises per set where the worker path skips the set.

        ``FITSDataset._load_missing_fits_files`` skips it so that one heterogeneous
        tile does not cost the tiles around it, or the sub-batches already written
        to zarr; the worker refuses a selection that misses *every* set up front
        instead. Here it is in-process with a single set, nothing is written yet and
        the caller reads the message, so it stays loud — the split is intentional,
        not an oversight.
        """
        vis_only = ("/d/EUC_MER_BGSUB-MOSAIC-VIS_TILE1-A.fits",)
        df = pd.DataFrame(
            [
                {
                    "SourceID": "S0",
                    "RA": 180.0,
                    "Dec": 0.0,
                    "diameter_pixel": 32,
                    "fits_file_paths": str(list(vis_only)),
                }
            ]
        )
        simple_config.selected_extensions = [{"name": "NIR-J", "ext": "PRIMARY"}]

        p1, p2, p3 = self._patches(
            {vis_only: [df.iloc[0].to_dict()]},
            lambda *_: {},
            lambda *_a, **_k: [],
        )
        with p1, p2, p3, pytest.raises(ValueError, match="match none of the bands"):
            create_cutouts_direct(df, simple_config, max_workers=1)

    def test_serial_and_parallel_agree(self, simple_config):
        """max_workers=1 and max_workers=4 produce identical (ordered) output."""
        df = pd.DataFrame(
            [
                {
                    "SourceID": f"S{i}",
                    "RA": 180.0 + i,
                    "Dec": float(i),
                    "diameter_pixel": 32,
                    "fits_file_paths": f"['tile{i}.fits']",
                }
                for i in range(3)
            ]
        )
        set_to_sources = {(f"tile{i}.fits",): [df.iloc[i].to_dict()] for i in range(3)}

        def fake_load(fits_sets, _ext):
            (fits_set,) = fits_sets  # single set passed by _process_one_fits_set
            return {path: (MagicMock(), {"PRIMARY": MagicMock()}) for path in fits_set}

        def fake_process(sources, _data, _cfg, profiler=None):
            return [{"metadata": [{"source_id": sources[0]["SourceID"]}]}]

        def run(workers):
            p1, p2, p3 = self._patches(set_to_sources, fake_load, fake_process)
            with p1, p2, p3:
                return create_cutouts_direct(df, simple_config, max_workers=workers)

        serial = [r["metadata"][0]["source_id"] for r in run(1)]
        parallel = [r["metadata"][0]["source_id"] for r in run(4)]
        assert serial == parallel == ["S0", "S1", "S2"]

    def test_worker_exception_propagates_in_parallel(self, simple_config):
        """An exception inside a worker surfaces via future.result(), not swallowed."""
        df = pd.DataFrame(
            [
                {
                    "SourceID": f"S{i}",
                    "RA": 180.0 + i,
                    "Dec": float(i),
                    "diameter_pixel": 32,
                    "fits_file_paths": f"['tile{i}.fits']",
                }
                for i in range(3)
            ]
        )
        set_to_sources = {(f"tile{i}.fits",): [df.iloc[i].to_dict()] for i in range(3)}

        def fake_load(fits_sets, _ext):
            (fits_set,) = fits_sets  # single set passed by _process_one_fits_set
            return {path: (MagicMock(), {"PRIMARY": MagicMock()}) for path in fits_set}

        def boom(_sources, _data, _cfg, profiler=None):
            raise RuntimeError("worker blew up")

        p1, p2, p3 = self._patches(set_to_sources, fake_load, boom)
        with p1, p2, p3, pytest.raises(RuntimeError, match="worker blew up"):
            create_cutouts_direct(df, simple_config, max_workers=3)

    @staticmethod
    def _fake_load(fits_sets, _ext):
        (fits_set,) = fits_sets
        return {p: (MagicMock(), {"PRIMARY": MagicMock()}) for p in fits_set}

    @staticmethod
    def _fake_process(sources, _data, _cfg, profiler=None):
        return [{"metadata": [{"source_id": sources[0]["SourceID"]}]}]

    def _df_and_sets(self, n):
        df = pd.DataFrame(
            [
                {
                    "SourceID": f"S{i}",
                    "RA": 180.0 + i,
                    "Dec": float(i),
                    "diameter_pixel": 32,
                    "fits_file_paths": f"['tile{i}.fits']",
                }
                for i in range(n)
            ]
        )
        set_to_sources = {(f"tile{i}.fits",): [df.iloc[i].to_dict()] for i in range(n)}
        return df, set_to_sources

    def _run_capture_progress(
        self, df, set_to_sources, simple_config, workers, *, log_set_progress=True, process=None
    ):
        messages: list[str] = []
        # cutana disables its own logger at import (see cutana/__init__.py);
        # a consumer must opt in to see these lines.
        logger.enable("cutana")
        sink_id = logger.add(
            lambda m, store=messages: store.append(m.record["message"]),
            level="INFO",
            format="{message}",
        )
        try:
            p1, p2, p3 = self._patches(
                set_to_sources, self._fake_load, process or self._fake_process
            )
            with p1, p2, p3:
                results = create_cutouts_direct(
                    df, simple_config, max_workers=workers, log_set_progress=log_set_progress
                )
        finally:
            logger.remove(sink_id)
            logger.disable("cutana")
        progress = [m for m in messages if m.startswith("Processed")]
        return progress, results

    def test_per_set_progress_logged_for_multiple_sets(self, simple_config):
        """A multi-tile request logs a 'Processed k/N FITS sets' completion
        heartbeat 1..N on both the serial and parallel paths."""
        n = 4
        df, set_to_sources = self._df_and_sets(n)
        for workers in (1, 4):
            progress, _ = self._run_capture_progress(df, set_to_sources, simple_config, workers)
            assert progress == [f"Processed {k}/{n} FITS sets" for k in range(1, n + 1)], (
                f"workers={workers}: {progress}"
            )

    def test_single_set_emits_no_progress(self, simple_config):
        """One FITS set has no intermediate progress to report."""
        df, set_to_sources = self._df_and_sets(1)
        progress, _ = self._run_capture_progress(df, set_to_sources, simple_config, workers=1)
        assert progress == []

    def test_caller_can_opt_out_of_progress(self, simple_config):
        """A caller invoking this per group (e.g. once per catalogue) passes
        log_set_progress=False to silence the otherwise-noisy heartbeat."""
        n = 4
        df, set_to_sources = self._df_and_sets(n)
        for workers in (1, 4):
            progress, _ = self._run_capture_progress(
                df, set_to_sources, simple_config, workers, log_set_progress=False
            )
            assert progress == [], f"workers={workers}: {progress}"

    def test_parallel_output_order_preserved_on_out_of_order_completion(self, simple_config):
        """The threaded path reassembles by submission index, so output stays in
        catalogue order even when a later-submitted set finishes first."""
        n = 4
        df, set_to_sources = self._df_and_sets(n)

        def reverse_delay_process(sources, _data, _cfg, profiler=None):
            # Make earlier-submitted sets finish last: S0 sleeps most, S3 least,
            # so completion order is the reverse of submission order.
            idx = int(sources[0]["SourceID"][1:])
            time.sleep(0.02 * (n - idx))
            return [{"metadata": [{"source_id": sources[0]["SourceID"]}]}]

        _progress, results = self._run_capture_progress(
            df, set_to_sources, simple_config, workers=4, process=reverse_delay_process
        )
        order = [r["metadata"][0]["source_id"] for r in results]
        assert order == [f"S{i}" for i in range(n)], order

    def test_missing_fits_set_raises(self, simple_config):
        """A set whose FITS files all fail to load is a hard error, not a silent drop."""
        df = pd.DataFrame(
            [
                {
                    "SourceID": "S0",
                    "RA": 180.0,
                    "Dec": 0.0,
                    "diameter_pixel": 32,
                    "fits_file_paths": "['missing.fits']",
                }
            ]
        )
        set_to_sources = {("missing.fits",): [df.iloc[0].to_dict()]}

        # load_fits_sets returns empty when every file in the set fails to load.
        def empty_load(_fits_sets, _ext):
            return {}

        p1, p2, p3 = self._patches(set_to_sources, empty_load, lambda *a, **k: [])
        with p1, p2, p3, pytest.raises(RuntimeError, match="No FITS data could be loaded"):
            create_cutouts_direct(df, simple_config, max_workers=1)
