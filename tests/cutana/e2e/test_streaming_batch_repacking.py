#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Regression test for user-batch repacking across internal batch boundaries.

Background
----------
``StreamingOrchestrator`` decouples *internal* batches (one subprocess per
internal batch, grouped by FITS set / tile) from *user* batches (the batches
returned by ``next_batch()``). In in-memory mode each internal batch result is
split into batches of size ``batch_size`` independently in
``_consume_ready_result`` / ``_pop_user_batch_from_leftover``.

The bug: the trailing remainder of one internal batch is emitted as its own
(short) user batch instead of being carried over and combined with sources from
the next internal batch. So whenever an internal batch size is not a multiple of
the user batch size, a short batch leaks out *mid-stream* rather than only at the
very end. ``get_batch_count()`` (``ceil(total / batch_size)``) then under-reports
the number of batches the orchestrator actually produces, and looping over the
reported count drops the tail batches entirely -> silent source loss.

Scenario (matches the reported regression)
-------------------------------------------
Three tiles with N_1, N_2, N_3 sources, streamed with ``N_batch == N_1`` (where
``N_batch`` is the user-facing ``batch_size`` passed to ``init_streaming``), and
``N_batch << N_2``, ``N_2 % N_batch != 0`` and ``N_batch > N_3``. We force the
internal batch size to equal the user batch size (via ``N_batch_cutout_process``)
so internal batches align to tile boundaries with tiny, cheap-to-process source
counts. Tile 2 then spans multiple internal batches with a non-aligned tail, and
tile 3 forms a sub-``N_batch`` internal batch.

Expected (correct) behaviour: every batch except the last has exactly ``N_batch``
cutouts, the last has exactly ``total % N_batch``, the number of batches is
``ceil(total / N_batch)``, and the total number of delivered cutouts equals the
total number of sources.
"""

import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from astropy.wcs import WCS

from cutana import StreamingOrchestrator, get_default_config

# Make the shared mock-data generator helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "test_data"))
from generate_test_data import create_euclid_mosaic_fits, create_euclid_wcs  # noqa: E402

# Tile source counts chosen so that N_batch == N_1, N_batch << N_2,
# N_2 % N_batch != 0, and N_batch > N_3.
N_1 = 10
N_2 = 24
N_3 = 5
N_BATCH = N_1
TOTAL_SOURCES = N_1 + N_2 + N_3
MOSAIC_SIZE = (400, 400)
DIAMETER_PIXEL = 20


def _build_tile(
    output_path: Path, tile_id: str, ra_center: float, dec_center: float, n_sources: int
):
    """Create one mock VIS mosaic and a list of source rows placed inside it.

    Sources are laid out on a coarse grid well inside the mosaic bounds so every
    cutout window stays in-frame. Returns the source row dicts (Cutana CSV
    schema) pointing at the freshly written mosaic file.
    """
    mosaic_path = create_euclid_mosaic_fits(
        output_path,
        tile_id,
        ra_center,
        dec_center,
        size=MOSAIC_SIZE,
        instrument="VIS",
    )
    fits_paths_str = str([Path(mosaic_path).resolve().as_posix()])

    wcs: WCS = create_euclid_wcs(ra_center, dec_center, MOSAIC_SIZE)

    # Grid of pixel positions inside a safe interior region.
    margin = 60
    span = MOSAIC_SIZE[0] - 2 * margin
    side = math.ceil(math.sqrt(n_sources))
    step = span / max(side, 1)

    rows = []
    for i in range(n_sources):
        gx = i % side
        gy = i // side
        px = margin + step * (gx + 0.5)
        py = margin + step * (gy + 0.5)
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
def three_tile_catalogue():
    """Build a 3-tile catalogue (N_1, N_2, N_3 sources) and return its CSV path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="cutana_repack_"))

    rows = []
    rows += _build_tile(tmp_dir, "900000001", 150.10, 2.10, N_1)
    rows += _build_tile(tmp_dir, "900000002", 151.10, 3.10, N_2)
    rows += _build_tile(tmp_dir, "900000003", 152.10, 4.10, N_3)

    catalogue_path = tmp_dir / "three_tile_catalogue.csv"
    pd.DataFrame(rows).to_csv(catalogue_path, index=False)
    return catalogue_path


@pytest.fixture
def streaming_config(three_tile_catalogue):
    """Streaming config that forces internal batch size == user batch size.

    Setting ``N_batch_cutout_process == N_BATCH`` makes the in-memory internal
    batch size equal the user batch size, so internal batches align to tile
    boundaries without needing thousands of sources. ``max_workers == 1`` keeps
    batch delivery deterministic and ordered for the size-pattern assertions.
    """
    config = get_default_config()
    config.output_format = "zarr"
    config.target_resolution = 32
    config.selected_extensions = ["VIS"]
    config.channel_weights = {"VIS": [1.0]}
    config.console_log_level = "WARNING"
    config.skip_memory_calibration_wait = True
    config.max_workers = 1
    config.max_workflow_time_seconds = 600
    config.N_batch_cutout_process = N_BATCH
    config.source_catalogue = str(three_tile_catalogue)
    return config


class TestStreamingBatchRepacking:
    """Regression: user batches must repack across internal batch boundaries."""

    def test_batches_are_full_until_last_and_no_sources_lost(self, streaming_config):
        with tempfile.TemporaryDirectory() as output_dir:
            streaming_config.output_dir = output_dir

            orchestrator = StreamingOrchestrator(streaming_config)
            try:
                orchestrator.init_streaming(
                    batch_size=N_BATCH,
                    write_to_disk=False,
                    max_workers=1,
                    min_workers=1,
                )

                reported_batches = orchestrator.get_batch_count()
                expected_batches = math.ceil(TOTAL_SOURCES / N_BATCH)

                # 1. Reported batch count must match the mathematical optimum.
                #    Checked before draining so a wrong count fails fast.
                assert reported_batches == expected_batches, (
                    f"get_batch_count() reported {reported_batches}, "
                    f"expected {expected_batches} for {TOTAL_SOURCES} sources "
                    f"at batch_size={N_BATCH}"
                )

                batches = []
                for _ in range(reported_batches):
                    result = orchestrator.next_batch()
                    batches.append(result["cutouts"])

                sizes = [len(b) for b in batches]
                total_delivered = sum(sizes)

                # 2. No sources may be silently dropped.
                assert total_delivered == TOTAL_SOURCES, (
                    f"Delivered {total_delivered} cutouts across {len(batches)} "
                    f"batches (sizes={sizes}), expected {TOTAL_SOURCES}"
                )

                # 3. Every batch but the last must be exactly N_BATCH; the last
                #    must equal TOTAL_SOURCES % N_BATCH. No short batches mid-stream.
                for idx, size in enumerate(sizes[:-1]):
                    assert size == N_BATCH, (
                        f"Batch {idx} has {size} cutouts, expected {N_BATCH} "
                        f"(only the final batch may be smaller). sizes={sizes}"
                    )
                expected_last = TOTAL_SOURCES % N_BATCH or N_BATCH
                assert sizes[-1] == expected_last, (
                    f"Final batch has {sizes[-1]} cutouts, expected {expected_last} "
                    f"(TOTAL_SOURCES % N_BATCH). sizes={sizes}"
                )
            finally:
                orchestrator.cleanup()
