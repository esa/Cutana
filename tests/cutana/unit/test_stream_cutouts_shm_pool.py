#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for stream_cutouts_via_shm_pool in cutout_process.py."""

import json
import sys
from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest
from dotmap import DotMap

from cutana.cutout_process import stream_cutouts_via_shm_pool
from cutana.shm_pool import ShmPool, ShmPoolConfig


@pytest.fixture
def pool_config():
    """Small pool: 5 slots of 4x4x3 float32."""
    return ShmPoolConfig(
        slot_shape=(4, 4, 3),
        dtype=np.dtype("float32"),
        slots_per_worker=5,
    )


@pytest.fixture
def pool(pool_config):
    p = ShmPool(pool_config)
    yield p
    p.cleanup()


def make_config(pool, pool_config):
    """Build a DotMap config with pool info."""
    cfg = DotMap(_dynamic=False)
    cfg.shm_pool_name = pool.name
    cfg.shm_control_name = pool.control_name
    cfg.shm_pool_config = pool_config.to_dict()
    return cfg


def make_batch_results(n_cutouts, shape, dtype):
    """Create mock batch results with random cutouts."""
    cutouts = [np.random.rand(*shape).astype(dtype) for _ in range(n_cutouts)]
    metadata = [{"SourceID": f"src_{i}"} for i in range(n_cutouts)]
    return [{"cutouts": cutouts, "metadata": metadata}]


class TestStreamCutoutsViaShmPool:
    def test_single_chunk_fits_in_pool(self, pool, pool_config):
        """3 cutouts fit in 5-slot pool — one chunk, one ACK."""
        config = make_config(pool, pool_config)
        batch_results = make_batch_results(3, pool_config.slot_shape, pool_config.dtype)
        expected_cutouts = batch_results[0]["cutouts"]

        # Capture stdout, provide ACK on stdin
        captured_stdout = StringIO()
        ack_stdin = StringIO("ACK\n")

        with patch.object(sys, "stdout", captured_stdout), patch.object(sys, "stdin", ack_stdin):
            stream_cutouts_via_shm_pool(batch_results, "test_proc", config)

        # Parse stdout messages
        lines = captured_stdout.getvalue().strip().split("\n")
        messages = [json.loads(line) for line in lines]

        # Should have chunk_ready + complete
        assert len(messages) == 2
        assert messages[0]["type"] == "chunk_ready"
        assert messages[0]["slot_count"] == 3
        assert len(messages[0]["metadata"]) == 3
        assert messages[1]["type"] == "complete"
        assert messages[1]["total_cutouts"] == 3

        # Verify data was written to pool (read from orchestrator side)
        result = pool.read_slots(3)
        for i in range(3):
            np.testing.assert_array_almost_equal(result[i], expected_cutouts[i])

    def test_multiple_chunks(self, pool_config):
        """8 cutouts in a 5-slot pool — needs 2 chunks."""
        pool = ShmPool(pool_config)
        try:
            config = make_config(pool, pool_config)
            batch_results = make_batch_results(8, pool_config.slot_shape, pool_config.dtype)

            captured_stdout = StringIO()
            # Two ACKs needed (one per chunk)
            ack_stdin = StringIO("ACK\nACK\n")

            with (
                patch.object(sys, "stdout", captured_stdout),
                patch.object(sys, "stdin", ack_stdin),
            ):
                stream_cutouts_via_shm_pool(batch_results, "test_proc", config)

            lines = captured_stdout.getvalue().strip().split("\n")
            messages = [json.loads(line) for line in lines]

            # 2 chunk_ready + 1 complete
            assert len(messages) == 3
            assert messages[0]["type"] == "chunk_ready"
            assert messages[0]["slot_count"] == 5  # first chunk fills pool
            assert messages[1]["type"] == "chunk_ready"
            assert messages[1]["slot_count"] == 3  # remaining cutouts
            assert messages[2]["type"] == "complete"
            assert messages[2]["total_cutouts"] == 8
        finally:
            pool.cleanup()

    def test_empty_batch_results(self, pool, pool_config):
        """Empty batch results should send complete with 0 cutouts."""
        config = make_config(pool, pool_config)
        batch_results = [{"cutouts": [], "metadata": []}]

        captured_stdout = StringIO()
        with patch.object(sys, "stdout", captured_stdout):
            stream_cutouts_via_shm_pool(batch_results, "test_proc", config)

        lines = captured_stdout.getvalue().strip().split("\n")
        messages = [json.loads(line) for line in lines]
        assert len(messages) == 1
        assert messages[0]["type"] == "complete"
        assert messages[0]["total_cutouts"] == 0

    def test_metadata_preserved(self, pool, pool_config):
        """Metadata should be passed through in chunk messages."""
        config = make_config(pool, pool_config)
        cutouts = [np.zeros(pool_config.slot_shape, dtype=pool_config.dtype) for _ in range(2)]
        metadata = [{"SourceID": "A", "RA": 1.0}, {"SourceID": "B", "RA": 2.0}]
        batch_results = [{"cutouts": cutouts, "metadata": metadata}]

        captured_stdout = StringIO()
        ack_stdin = StringIO("ACK\n")

        with patch.object(sys, "stdout", captured_stdout), patch.object(sys, "stdin", ack_stdin):
            stream_cutouts_via_shm_pool(batch_results, "test_proc", config)

        lines = captured_stdout.getvalue().strip().split("\n")
        chunk_msg = json.loads(lines[0])
        assert chunk_msg["metadata"] == metadata

    def test_exact_pool_size(self, pool, pool_config):
        """Exactly 5 cutouts in 5-slot pool — one chunk."""
        config = make_config(pool, pool_config)
        batch_results = make_batch_results(5, pool_config.slot_shape, pool_config.dtype)

        captured_stdout = StringIO()
        ack_stdin = StringIO("ACK\n")

        with patch.object(sys, "stdout", captured_stdout), patch.object(sys, "stdin", ack_stdin):
            stream_cutouts_via_shm_pool(batch_results, "test_proc", config)

        lines = captured_stdout.getvalue().strip().split("\n")
        messages = [json.loads(line) for line in lines]
        assert messages[0]["slot_count"] == 5
        assert messages[1]["type"] == "complete"

    def test_uint8_dtype(self):
        """Pool mode works with uint8 data."""
        config_uint8 = ShmPoolConfig(
            slot_shape=(4, 4, 1),
            dtype=np.dtype("uint8"),
            slots_per_worker=3,
        )
        pool = ShmPool(config_uint8)
        try:
            config = make_config(pool, config_uint8)
            cutouts = [np.full(config_uint8.slot_shape, 200, dtype=np.uint8) for _ in range(2)]
            metadata = [{"SourceID": f"s{i}"} for i in range(2)]
            batch_results = [{"cutouts": cutouts, "metadata": metadata}]

            captured_stdout = StringIO()
            ack_stdin = StringIO("ACK\n")

            with (
                patch.object(sys, "stdout", captured_stdout),
                patch.object(sys, "stdin", ack_stdin),
            ):
                stream_cutouts_via_shm_pool(batch_results, "test_proc", config)

            result = pool.read_slots(2)
            np.testing.assert_array_equal(result[0], cutouts[0])
        finally:
            pool.cleanup()

    def test_multiple_batch_results_combined(self, pool, pool_config):
        """Multiple batch_results entries are concatenated."""
        config = make_config(pool, pool_config)
        cutouts1 = [np.ones(pool_config.slot_shape, dtype=pool_config.dtype) * 1.0]
        cutouts2 = [np.ones(pool_config.slot_shape, dtype=pool_config.dtype) * 2.0]
        batch_results = [
            {"cutouts": cutouts1, "metadata": [{"SourceID": "A"}]},
            {"cutouts": cutouts2, "metadata": [{"SourceID": "B"}]},
        ]

        captured_stdout = StringIO()
        ack_stdin = StringIO("ACK\n")

        with patch.object(sys, "stdout", captured_stdout), patch.object(sys, "stdin", ack_stdin):
            stream_cutouts_via_shm_pool(batch_results, "test_proc", config)

        lines = captured_stdout.getvalue().strip().split("\n")
        chunk_msg = json.loads(lines[0])
        assert chunk_msg["slot_count"] == 2
        assert len(chunk_msg["metadata"]) == 2
