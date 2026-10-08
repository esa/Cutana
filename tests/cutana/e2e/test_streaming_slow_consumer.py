#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Regression tests for silent cutout loss when the streaming consumer is slow.

Background
----------
In in-memory streaming mode a worker writes a chunk of cutouts into its shared
memory pool, announces it on stdout and then blocks until the orchestrator
acknowledges the chunk. The orchestrator only services those acknowledgements
from inside ``next_batch()``, so the delay a worker sees is however long the
*consumer* spends on the batch it was just handed (ML inference, disk writes,
...).

The bug: the worker aborted the acknowledgement wait after a hardcoded 60s and
returned, abandoning every remaining chunk. The abort was logged and swallowed,
so the worker still exited 0 and the orchestrator accepted the short result.
Because ``next_batch()`` derives its batch count from the catalogue row count,
the run then died near the tail with the unrelated message "All workers
completed but no cutouts remain for the expected batch" after silently dropping
a large fraction of the requested cutouts.

Expected behaviour: the acknowledgement wait is backpressure, not a fault. A
worker waits for as long as the consumer needs, and any failure that does cost
cutouts is raised rather than logged and absorbed.
"""

import io
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS
from dotmap import DotMap

from cutana import StreamingOrchestrator, get_default_config
from cutana.cutout_process import stream_cutouts_via_shm_pool
from cutana.shm_pool import ShmPool, ShmPoolConfig

# Make the shared mock-data generator helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "test_data"))
from generate_test_data import create_euclid_mosaic_fits, create_euclid_wcs  # noqa: E402

# The wait that used to be cut short after 60s. The slow-consumer test has to
# stall the orchestrator for longer than that to exercise the regression.
CONSUMER_STALL_SECONDS = 65

N_PER_TILE = 60
N_BATCH = 40
TOTAL_SOURCES = 2 * N_PER_TILE
MOSAIC_SIZE = (400, 400)
DIAMETER_PIXEL = 20
SLOTS_PER_WORKER = 5


def _build_tile(output_path: Path, tile_id: str, ra_center: float, dec_center: float):
    """Create one mock VIS mosaic and N_PER_TILE source rows placed inside it."""
    mosaic_path = create_euclid_mosaic_fits(
        output_path, tile_id, ra_center, dec_center, size=MOSAIC_SIZE, instrument="VIS"
    )
    fits_paths_str = str([Path(mosaic_path).resolve().as_posix()])

    wcs: WCS = create_euclid_wcs(ra_center, dec_center, MOSAIC_SIZE)
    margin = 60
    span = MOSAIC_SIZE[0] - 2 * margin
    side = math.ceil(math.sqrt(N_PER_TILE))
    step = span / side

    rows = []
    for i in range(N_PER_TILE):
        px = margin + step * ((i % side) + 0.5)
        py = margin + step * ((i // side) + 0.5)
        ra, dec = wcs.wcs_pix2world(px, py, 0)
        rows.append(
            {
                "SourceID": f"{tile_id}_src_{i:04d}",
                "RA": float(ra),
                "Dec": float(dec),
                "diameter_pixel": DIAMETER_PIXEL,
                "fits_file_paths": fits_paths_str,
            }
        )
    return rows


@pytest.fixture(scope="module")
def two_tile_catalogue(tmp_path_factory):
    """Build a 2-tile catalogue and return its CSV path."""
    tmp_dir = tmp_path_factory.mktemp("cutana_slow_consumer")
    rows = _build_tile(tmp_dir, "900000101", 150.10, 2.10)
    rows += _build_tile(tmp_dir, "900000102", 151.10, 3.10)

    catalogue_path = tmp_dir / "two_tile_catalogue.csv"
    pd.DataFrame(rows).to_csv(catalogue_path, index=False)
    return catalogue_path


@pytest.fixture
def streaming_config(two_tile_catalogue):
    """Streaming config sized so two workers run concurrently."""
    config = get_default_config()
    config.output_format = "zarr"
    config.target_resolution = 32
    config.selected_extensions = ["VIS"]
    config.channel_weights = {"VIS": [1.0]}
    config.console_log_level = "WARNING"
    config.skip_memory_calibration_wait = True
    config.max_workflow_time_seconds = 600
    config.N_batch_cutout_process = N_BATCH
    config.source_catalogue = str(two_tile_catalogue)
    return config


@pytest.mark.slow
def test_slow_consumer_does_not_lose_cutouts(streaming_config):
    """A consumer that stalls past the old ACK window must still get every cutout.

    Two workers run concurrently, so while the consumer is stalled on the first
    batch the second worker sits blocked waiting for a chunk acknowledgement.
    Under the old 60s abort it gave up mid-stream and its remaining cutouts were
    dropped without an error.
    """
    with tempfile.TemporaryDirectory() as output_dir:
        streaming_config.output_dir = output_dir

        orchestrator = StreamingOrchestrator(streaming_config)
        try:
            # 32x32 float32 single-channel slots, capped so each worker needs
            # several chunks and is therefore certain to block on an ACK.
            slot_bytes = 32 * 32 * 1 * np.dtype(streaming_config.data_type).itemsize
            orchestrator.init_streaming(
                batch_size=N_BATCH,
                write_to_disk=False,
                max_workers=2,
                min_workers=2,
                max_shm_memory_consumption=SLOTS_PER_WORKER * slot_bytes * 2,
            )

            delivered = 0
            source_ids = []
            for batch_number in range(orchestrator.get_batch_count()):
                result = orchestrator.next_batch()
                delivered += len(result["cutouts"])
                source_ids.extend(m["source_id"] for m in result["metadata"])

                # Stall only once: one stall past the old window is enough to
                # strand the concurrently running worker, and each extra stall
                # costs another minute of test runtime.
                if batch_number == 0:
                    time.sleep(CONSUMER_STALL_SECONDS)

            assert delivered == TOTAL_SOURCES, (
                f"Delivered {delivered} cutouts, expected {TOTAL_SOURCES}: a slow "
                "consumer must not cause workers to abandon cutouts"
            )
            assert len(set(source_ids)) == TOTAL_SOURCES
        finally:
            orchestrator.cleanup()


def _pool_streaming_config(pool: ShmPool) -> DotMap:
    """Minimal worker-side config pointing at an existing pool."""
    config = DotMap(_dynamic=False)
    config.shm_pool_name = pool.name
    config.shm_control_name = pool.control_name
    config.shm_pool_config = pool.config.to_dict()
    return config


@pytest.mark.parametrize(
    "ack_line, expected_fragment",
    [
        ("", "closed the ACK pipe"),  # parent died / pipe closed
        ("NACK\n", "expected 'ACK'"),  # protocol violation
    ],
)
def test_failed_acknowledgement_raises_instead_of_dropping(
    monkeypatch, ack_line, expected_fragment
):
    """A missing or malformed ACK must raise, never silently truncate the stream.

    The worker's exception is what makes the parent see a non-zero exit code; the
    old code logged and returned, which left the orchestrator to accept a short
    batch as if the worker had finished normally.
    """
    pool_config = ShmPoolConfig(slot_shape=(4, 4, 1), dtype=np.dtype("float32"), slots_per_worker=2)
    pool = ShmPool(pool_config)
    try:
        # Four cutouts through a two-slot pool: the first chunk needs an ACK
        # before the second can reuse the slots.
        cutouts = [np.full((4, 4, 1), i, dtype=np.float32) for i in range(4)]
        batch_results = [
            {"cutouts": cutouts, "metadata": [{"source_id": str(i)} for i in range(4)]}
        ]

        # StringIO gives readline() the ack once and "" thereafter, which is both
        # cases this test needs; the raise happens on the first read either way.
        monkeypatch.setattr(sys, "stdin", io.StringIO(ack_line))

        with pytest.raises(RuntimeError, match=expected_fragment):
            stream_cutouts_via_shm_pool(batch_results, "test_worker", _pool_streaming_config(pool))
    finally:
        pool.cleanup()
