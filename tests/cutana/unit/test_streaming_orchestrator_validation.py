#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for StreamingOrchestrator.
"""

import json

import numpy as np
import pytest

from cutana import StreamingOrchestrator, get_default_config
from cutana.catalogue_preprocessor import DUPLICATE_CHECK_THRESHOLD
from cutana.profiling_types import WorkerInfo


@pytest.fixture
def base_config(tmp_path):
    config = get_default_config()
    config.output_format = "fits"
    config.target_resolution = 32
    config.selected_extensions = ["VIS"]
    config.channel_weights = {"VIS": [1.0]}
    config.skip_memory_calibration_wait = True
    dummy_catalogue = tmp_path / "dummy.csv"
    dummy_catalogue.touch()
    config.source_catalogue = str(dummy_catalogue)
    return config


class TestInitStreamingValidation:
    """Tests for init_streaming parameter validation."""

    def test_do_only_cutout_extraction_with_write_to_disk_false_raises(self, base_config):
        """do_only_cutout_extraction=True is only meaningful when writing to disk."""
        base_config.do_only_cutout_extraction = True

        orchestrator = StreamingOrchestrator(base_config)
        try:
            with pytest.raises(ValueError, match="do_only_cutout_extraction"):
                orchestrator.init_streaming(batch_size=10, write_to_disk=False)
        finally:
            orchestrator.cleanup()

    def test_do_only_cutout_extraction_with_write_to_disk_true_does_not_raise_early(
        self, base_config, tmp_path
    ):
        """do_only_cutout_extraction=True + write_to_disk=True must not raise the guard."""
        base_config.do_only_cutout_extraction = True
        base_config.output_dir = str(tmp_path)

        orchestrator = StreamingOrchestrator(base_config)
        try:
            # The guard must not fire; subsequent I/O will raise (empty catalogue),
            # but we only care that the ValueError guard is not triggered.
            with pytest.raises(Exception) as exc_info:
                orchestrator.init_streaming(batch_size=10, write_to_disk=True)
            assert "do_only_cutout_extraction" not in str(exc_info.value)
        finally:
            orchestrator.cleanup()


class TestWorkerInfo:
    """Tests for the rich per-worker detail exposed via get_worker_info (issue #354)."""

    def test_get_worker_info_returns_independent_mapping(self, base_config):
        """The returned mapping is a fresh dict; adding/removing keys is local to it.

        The contained WorkerInfo objects are shared (read-only by contract), so the
        contract is that the top-level mapping is a copy, not the records.
        """
        orchestrator = StreamingOrchestrator(base_config)
        record = WorkerInfo(
            process_id="w0", batch_index=0, n_sources=5, pool_slot=0, start_time=1.0
        )
        orchestrator._worker_info["w0"] = record

        snapshot = orchestrator.get_worker_info()
        snapshot["w1"] = WorkerInfo(
            process_id="w1", batch_index=1, n_sources=1, pool_slot=1, start_time=2.0
        )

        assert "w1" not in orchestrator._worker_info
        # Same WorkerInfo instance is handed back (no per-record copy).
        assert snapshot["w0"] is record

    def test_complete_message_merges_batch_info_and_end_time(self, base_config):
        """A worker's complete message merges FITS-set + timing detail and stamps end_time."""
        orchestrator = StreamingOrchestrator(base_config)
        # Seed the spawn-time record as _spawn_next_worker would.
        orchestrator._worker_info["w0"] = WorkerInfo(
            process_id="w0",
            batch_index=0,
            n_sources=3,
            pool_slot=0,
            start_time=100.0,
        )
        # In-memory completion path avoids disk/zarr assembly.
        orchestrator._streaming_write_to_disk = False
        # Completion now cross-checks delivered cutouts against the assignment (#398).
        # This test is about merge_completion, so the worker state just has to be
        # self-consistent: 3 sources assigned, the 3 chunks it would have handed over.
        orchestrator._worker_source_counts["w0"] = 3
        orchestrator._worker_cutouts["w0"] = [np.zeros((3, 2, 2, 1), dtype=np.float32)]
        orchestrator._worker_metadata["w0"] = [{"source_id": str(i)} for i in range(3)]
        orchestrator._worker_to_pool["w0"] = 0
        orchestrator._pool_to_worker[0] = "w0"

        complete_line = json.dumps(
            {
                "type": "complete",
                "total_cutouts": 3,
                "batch_info": {
                    "sources_per_fits_set": {"VIS.fits, NIR-H.fits": 2, "VIS.fits": 1},
                    "performance": {"steps": {"CutoutExtraction": {"total_time": 0.5}}},
                },
            }
        )
        orchestrator._handle_worker_line("w0", complete_line)

        info = orchestrator.get_worker_info()["w0"]
        assert info.sources_per_fits_set == {"VIS.fits, NIR-H.fits": 2, "VIS.fits": 1}
        assert len(info.sources_per_fits_set) == 2
        assert info.performance["steps"]["CutoutExtraction"]["total_time"] == 0.5
        assert info.batch_index == 0  # spawn-time field preserved
        assert info.end_time is not None


class _RecordingIndex:
    """Catalogue index stub that records how batch ranges were requested."""

    def __init__(self, row_count):
        self.row_count = row_count
        self.requested_max_sources_per_batch = None

    def get_optimized_batch_ranges(self, max_sources_per_batch, **_kwargs):
        self.requested_max_sources_per_batch = max_sources_per_batch
        return []


class _StubReader:
    """Batch reader stub; cleanup() closes the reader, so it needs close()."""

    def close(self):
        pass


class TestInternalBatchSizing:
    """Internal batches must be sized by the load balancer, not by the user batch size.

    Every internal batch costs one subprocess spawn (interpreter start plus the whole
    cutana import graph), so sizing them at the user batch size makes startup dominate
    at survey scale.
    """

    def _init_with_stub_index(
        self,
        config,
        tmp_path,
        monkeypatch,
        *,
        batch_size,
        write_to_disk,
        row_count=600_000,
        max_workers=2,
    ):
        config.output_dir = str(tmp_path)
        index = _RecordingIndex(row_count=row_count)
        orchestrator = StreamingOrchestrator(config)
        monkeypatch.setattr(
            type(orchestrator),
            "_init_catalogue_index_and_reader",
            lambda _self, _path: (index, _StubReader()),
        )
        try:
            orchestrator.init_streaming(
                batch_size=batch_size, write_to_disk=write_to_disk, max_workers=max_workers
            )
            return index, orchestrator._internal_batch_size
        finally:
            orchestrator.cleanup()

    @pytest.mark.parametrize("write_to_disk", [True, False])
    def test_internal_batch_uses_max_sources_per_process(
        self, base_config, tmp_path, monkeypatch, write_to_disk
    ):
        """Both streaming modes batch by max_sources_per_process, not by batch_size."""
        index, internal_batch_size = self._init_with_stub_index(
            base_config, tmp_path, monkeypatch, batch_size=1000, write_to_disk=write_to_disk
        )

        expected = int(base_config.loadbalancer.max_sources_per_process)
        assert expected > 1000, "load balancer should pick a batch larger than the user's"
        assert internal_batch_size == expected
        assert index.requested_max_sources_per_batch == expected

    def test_internal_batch_never_shrinks_below_requested_batch_size(
        self, base_config, tmp_path, monkeypatch
    ):
        """A caller asking for more than the load balancer's size keeps its own size.

        Disk-mode batches are one zarr archive each and cannot be re-split, so shrinking
        below the requested size would silently break the requested batching.
        """
        index, internal_batch_size = self._init_with_stub_index(
            base_config, tmp_path, monkeypatch, batch_size=50_000, write_to_disk=True
        )

        assert int(base_config.loadbalancer.max_sources_per_process) < 50_000
        assert internal_batch_size == 50_000
        assert index.requested_max_sources_per_batch == 50_000

    def test_load_balancer_config_is_applied(self, base_config, tmp_path, monkeypatch):
        """init_streaming must run the load balancer, as the non-streaming path does.

        Skipping it left max_sources_per_process unset on the streaming path entirely.
        """
        base_config.loadbalancer.max_sources_per_process = None

        self._init_with_stub_index(
            base_config, tmp_path, monkeypatch, batch_size=10, write_to_disk=False
        )

        assert base_config.loadbalancer.max_sources_per_process is not None

    def test_small_job_is_split_across_the_worker_pool(self, base_config, tmp_path, monkeypatch):
        """A job that fits in one load-balancer batch must still use every worker.

        Sizing purely by max_sources_per_process hands a 10k-source job to a single
        process while the other seven idle, which is slower than the batching it replaced.
        """
        _index, internal_batch_size = self._init_with_stub_index(
            base_config,
            tmp_path,
            monkeypatch,
            batch_size=1,
            write_to_disk=False,
            row_count=10_000,
            max_workers=8,
        )

        assert internal_batch_size == 1250, "10,000 sources over 8 workers is 1,250 each"
        assert 10_000 / internal_batch_size >= 8, "must produce at least one batch per worker"

    def test_internal_batch_stays_below_the_duplicate_check_threshold(
        self, base_config, tmp_path, monkeypatch
    ):
        """Batches must stay small enough that preprocess_catalogue still de-dupes.

        At >=1M sources the load balancer picks max_sources_per_process=100_000, which
        is exactly where preprocess_catalogue stops checking for duplicate SourceIDs.
        Streaming is documented to keep that check (#400), so the batch must stay under.
        """
        _index, internal_batch_size = self._init_with_stub_index(
            base_config,
            tmp_path,
            monkeypatch,
            batch_size=1,
            write_to_disk=True,
            row_count=10_000_000,
            max_workers=1,
        )

        assert int(base_config.loadbalancer.max_sources_per_process) >= DUPLICATE_CHECK_THRESHOLD
        assert internal_batch_size == DUPLICATE_CHECK_THRESHOLD - 1

    def test_in_memory_batch_fits_the_cutout_memory_budget(
        self, base_config, tmp_path, monkeypatch
    ):
        """In-memory batches are bounded by what their cutouts weigh.

        A worker accumulates a whole internal batch before streaming it and the parent
        holds one whole batch per worker, so the in-flight bytes are
        copies * workers * batch * bytes_per_cutout — unbounded by the SHM pool.
        """
        base_config.target_resolution = 150
        base_config.data_type = "float32"
        base_config.channel_weights = {"VIS": [1.0, 1.0, 1.0]}

        _index, internal_batch_size = self._init_with_stub_index(
            base_config,
            tmp_path,
            monkeypatch,
            batch_size=1,
            write_to_disk=False,
            row_count=10_000_000,
            max_workers=8,
        )

        bytes_per_cutout = 150 * 150 * 3 * np.dtype("float32").itemsize
        in_flight = 2 * 8 * internal_batch_size * bytes_per_cutout
        budget = base_config.loadbalancer.memory_limit_bytes
        assert in_flight <= 0.25 * budget, (
            f"{in_flight / 1e9:.1f} GB of cutouts in flight against a {budget / 1e9:.1f} GB budget"
        )

    def test_heavier_cutouts_get_smaller_in_memory_batches(
        self, base_config, tmp_path, monkeypatch
    ):
        """The in-memory bound must actually respond to cutout size, not just exist."""
        base_config.target_resolution = 32
        base_config.data_type = "uint8"
        base_config.channel_weights = {"VIS": [1.0]}
        _index, light = self._init_with_stub_index(
            base_config,
            tmp_path,
            monkeypatch,
            batch_size=1,
            write_to_disk=False,
            row_count=10_000_000,
            max_workers=8,
        )

        base_config.target_resolution = 512
        base_config.data_type = "float32"
        base_config.channel_weights = {"VIS": [1.0, 1.0, 1.0, 1.0]}
        _index, heavy = self._init_with_stub_index(
            base_config,
            tmp_path,
            monkeypatch,
            batch_size=1,
            write_to_disk=False,
            row_count=10_000_000,
            max_workers=8,
        )

        assert heavy < light
