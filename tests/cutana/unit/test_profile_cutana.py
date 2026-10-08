#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the Cutana bottleneck profiler's pure logic.

Covers the correctness-critical parts that do not require running the streaming
pipeline: canonical band/channel ordering, scenario parsing, per-stage aggregation
of the lazy-safe CPU/stall/read_bytes split, and the existence-checked catalogue
loader.
"""

import pandas as pd
import pytest

from benchmarking.profile_cutana import (
    BAND_SETS,
    Scenario,
    _parse_scenario_token,
    _summarise_stages,
    aggregate_worker_info,
    channel_weights_for,
    load_tiles,
)
from cutana.profiling_types import WorkerInfo


class TestChannelOrdering:
    """channel_weights must be keyed in the same order as fits_file_paths/selected_extensions.

    combine_channels binds weights positionally, so a mismatch silently corrupts
    channels. These tests pin the canonical order.
    """

    def test_vis1_single_passthrough(self):
        assert channel_weights_for("vis1") == {"VIS": [1.0]}

    def test_visnir3_keys_match_band_set_order(self):
        weights = channel_weights_for("visnir3")
        assert list(weights.keys()) == BAND_SETS["visnir3"]

    def test_visnir3_maps_four_bands_to_three_channels(self):
        weights = channel_weights_for("visnir3")
        assert all(len(vec) == 3 for vec in weights.values())

    def test_unknown_band_set_raises(self):
        with pytest.raises(ValueError, match="Unknown band set"):
            channel_weights_for("nope")


class TestScenarioParsing:
    """Scenario token parsing and naming."""

    def test_valid_token(self):
        assert _parse_scenario_token("visnir3_sparse_warm") == ("visnir3", "sparse", "warm")

    @pytest.mark.parametrize(
        "token", ["vis1_dense", "bogus_dense_cold", "vis1_medium_cold", "vis1_dense_lukewarm"]
    )
    def test_invalid_tokens_raise(self, token):
        with pytest.raises(ValueError):
            _parse_scenario_token(token)

    def test_scenario_name_is_stable(self):
        scenario = Scenario("visnir3", 180, 8, "dense", "cold")
        assert scenario.name == "visnir3_dense_cold_r180_w8"


class TestAggregation:
    """Per-stage CPU/stall/read_bytes aggregation and dominant-stage classification."""

    def _worker(self, start, end, extraction):
        return WorkerInfo(
            process_id="w",
            batch_index=0,
            n_sources=1,
            pool_slot=0,
            start_time=start,
            end_time=end,
            performance={"steps": {"CutoutExtraction": extraction}},
        )

    def test_aggregate_sums_across_workers(self):
        worker_info = {
            "w0": self._worker(
                0.0,
                10.0,
                {"total_time": 6.0, "cpu_time": 5.0, "stall_time": 1.0, "read_bytes": 100},
            ),
            "w1": self._worker(
                0.0, 10.0, {"total_time": 4.0, "cpu_time": 3.0, "stall_time": 1.0, "read_bytes": 50}
            ),
        }
        agg = aggregate_worker_info(worker_info)
        extraction = agg["stages"]["CutoutExtraction"]
        assert extraction["wall"] == pytest.approx(10.0)
        assert extraction["cpu"] == pytest.approx(8.0)
        assert extraction["stall_time"] == pytest.approx(2.0)
        assert extraction["read_bytes"] == 150
        # Both read_bytes were known ints, so the summed total is trustworthy.
        assert extraction["read_bytes_known"] is True
        # Both workers busy the whole 10s span -> perfect efficiency.
        assert agg["parallel_efficiency_estimate"] == pytest.approx(1.0)
        assert agg["n_workers"] == 2

    def test_unknown_read_bytes_marks_stage(self):
        worker_info = {
            "w0": self._worker(
                0.0,
                1.0,
                {"total_time": 1.0, "cpu_time": 1.0, "stall_time": 0.0, "read_bytes": None},
            )
        }
        agg = aggregate_worker_info(worker_info)
        assert agg["stages"]["CutoutExtraction"]["read_bytes_known"] is False

    def test_summarise_stages_picks_dominant_and_nature(self):
        stages = {
            "FitsLoading": {
                "wall": 1.0,
                "cpu": 0.1,
                "stall_time": 0.9,
                "read_bytes": 10,
                "read_bytes_known": True,
            },
            "CutoutExtraction": {
                "wall": 5.0,
                "cpu": 4.5,
                "stall_time": 0.5,
                "read_bytes": 0,
                "read_bytes_known": True,
            },
        }
        summary = _summarise_stages(stages)
        assert summary["dominant_stage"] == "CutoutExtraction"
        assert summary["dominant_nature"] == "cpu-bound"
        assert summary["dominant_cpu_fraction"] == pytest.approx(0.9)

    def test_summarise_stages_stall_bound(self):
        stages = {
            "CutoutExtraction": {
                "wall": 5.0,
                "cpu": 1.0,
                "stall_time": 4.0,
                "read_bytes": 99,
                "read_bytes_known": True,
            },
        }
        summary = _summarise_stages(stages)
        assert summary["dominant_nature"] == "stall-bound"
        # disk_MB/s = read_bytes / total_stall.
        assert summary["disk_mb_per_s"] == pytest.approx((99 / 1024 / 1024) / 4.0)


class TestLoadTilesExistenceCheck:
    """load_tiles must skip tiles whose FITS paths are absent on this machine."""

    def test_skips_tiles_with_missing_paths(self, tmp_path):
        present = tmp_path / "EUC_MER_BGSUB-MOSAIC-VIS_TILE102099999-AAA_x.fits"
        present.write_bytes(b"\x00")
        missing = tmp_path / "EUC_MER_BGSUB-MOSAIC-VIS_TILE102088888-BBB_x.fits"  # never created

        catalogue = tmp_path / "cat.csv"
        pd.DataFrame(
            [
                {
                    "SourceID": "a",
                    "RA": 1.0,
                    "Dec": 2.0,
                    "diameter_pixel": 10,
                    "fits_file_paths": str([str(present)]),
                },
                {
                    "SourceID": "b",
                    "RA": 1.0,
                    "Dec": 2.0,
                    "diameter_pixel": 10,
                    "fits_file_paths": str([str(missing)]),
                },
            ]
        ).to_csv(catalogue, index=False)

        tiles = load_tiles(catalogue, "vis1")
        assert len(tiles) == 1
        assert tiles[0].tile_id == "102099999"
        assert tiles[0].band_paths == [str(present)]
