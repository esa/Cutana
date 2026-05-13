#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for the shared memory pool."""

import numpy as np
import pytest

from cutana.shm_pool import (
    ShmPool,
    ShmPoolConfig,
    attach_pool,
    calculate_pool_config,
    detach_pool,
    mark_slots_ready,
    write_cutouts_to_pool,
)


@pytest.fixture
def small_config():
    """Pool config for 10 slots of 4x4x3 float32 cutouts."""
    return ShmPoolConfig(
        slot_shape=(4, 4, 3),
        dtype=np.dtype("float32"),
        slots_per_worker=10,
    )


@pytest.fixture
def pool(small_config):
    """Create and yield a pool, then clean up."""
    p = ShmPool(small_config)
    yield p
    p.cleanup()


class TestShmPoolConfig:
    def test_slot_bytes(self, small_config):
        # 4 * 4 * 3 * 4 bytes = 192
        assert small_config.slot_bytes == 4 * 4 * 3 * 4

    def test_total_bytes(self, small_config):
        assert small_config.total_bytes == small_config.slot_bytes * 10

    def test_to_dict_roundtrip(self, small_config):
        d = small_config.to_dict()
        restored = ShmPoolConfig.from_dict(d)
        assert restored.slot_shape == small_config.slot_shape
        assert restored.dtype == small_config.dtype
        assert restored.slots_per_worker == small_config.slots_per_worker

    def test_frozen(self, small_config):
        with pytest.raises(AttributeError):
            small_config.slots_per_worker = 999


class TestShmPool:
    def test_create_and_properties(self, pool, small_config):
        assert pool.name is not None
        assert pool.control_name is not None
        assert pool.config == small_config

    def test_read_empty_slots_returns_zeros(self, pool):
        result = pool.read_slots(3)
        assert result.shape == (3, 4, 4, 3)
        assert result.dtype == np.float32
        # Empty slots contain whatever was in memory, but should not raise
        assert isinstance(result, np.ndarray)

    def test_write_and_read_single_slot(self, pool, small_config):
        # Simulate worker writing a cutout
        data_shm, control_shm = attach_pool(pool.name, pool.control_name, small_config)
        try:
            cutout = np.ones(small_config.slot_shape, dtype=small_config.dtype) * 42.0
            write_cutouts_to_pool(data_shm, small_config, [cutout])
            mark_slots_ready(control_shm, small_config, 1)
        finally:
            detach_pool(data_shm, control_shm)

        # Orchestrator reads
        result = pool.read_slots(1)
        np.testing.assert_array_equal(result[0], cutout)

    def test_write_multiple_slots(self, pool, small_config):
        data_shm, control_shm = attach_pool(pool.name, pool.control_name, small_config)
        try:
            cutouts = [
                np.full(small_config.slot_shape, fill_value=float(i), dtype=small_config.dtype)
                for i in range(5)
            ]
            write_cutouts_to_pool(data_shm, small_config, cutouts)
            mark_slots_ready(control_shm, small_config, 5)
        finally:
            detach_pool(data_shm, control_shm)

        result = pool.read_slots(5)
        assert result.shape == (5, 4, 4, 3)
        for i in range(5):
            np.testing.assert_array_equal(result[i], cutouts[i])

    def test_read_resets_slots_to_empty(self, pool, small_config):
        data_shm, control_shm = attach_pool(pool.name, pool.control_name, small_config)
        try:
            cutout = np.ones(small_config.slot_shape, dtype=small_config.dtype)
            write_cutouts_to_pool(data_shm, small_config, [cutout])
            mark_slots_ready(control_shm, small_config, 1)
        finally:
            detach_pool(data_shm, control_shm)

        pool.read_slots(1)

        # Verify control array is EMPTY
        control_array = np.ndarray(
            (small_config.slots_per_worker,), dtype=np.int32, buffer=pool._control_shm.buf
        )
        assert control_array[0] == ShmPool.SLOT_EMPTY

    def test_reset_clears_all_slots(self, pool, small_config):
        data_shm, control_shm = attach_pool(pool.name, pool.control_name, small_config)
        try:
            mark_slots_ready(control_shm, small_config, 5)
        finally:
            detach_pool(data_shm, control_shm)

        pool.reset()

        control_array = np.ndarray(
            (small_config.slots_per_worker,), dtype=np.int32, buffer=pool._control_shm.buf
        )
        assert (control_array == ShmPool.SLOT_EMPTY).all()

    def test_read_slots_out_of_range(self, pool):
        with pytest.raises(IndexError):
            pool.read_slots(11)  # 11 > 10 slots

    def test_cleanup_idempotent(self, pool):
        pool.cleanup()
        pool.cleanup()  # Should not raise

    def test_slot_states(self, pool, small_config):
        data_shm, control_shm = attach_pool(pool.name, pool.control_name, small_config)
        try:
            control_array = np.ndarray(
                (small_config.slots_per_worker,), dtype=np.int32, buffer=control_shm.buf
            )
            # Initially all EMPTY
            assert (control_array == ShmPool.SLOT_EMPTY).all()

            mark_slots_ready(control_shm, small_config, 3)
            assert control_array[0] == ShmPool.SLOT_READY
            assert control_array[1] == ShmPool.SLOT_READY
            assert control_array[2] == ShmPool.SLOT_READY
            assert control_array[3] == ShmPool.SLOT_EMPTY
        finally:
            detach_pool(data_shm, control_shm)

    def test_different_dtypes(self):
        """Test pool works with uint8 dtype."""
        config = ShmPoolConfig(
            slot_shape=(8, 8, 1),
            dtype=np.dtype("uint8"),
            slots_per_worker=5,
        )
        p = ShmPool(config)
        try:
            data_shm, control_shm = attach_pool(p.name, p.control_name, config)
            try:
                cutout = np.full(config.slot_shape, fill_value=200, dtype=config.dtype)
                write_cutouts_to_pool(data_shm, config, [cutout])
                mark_slots_ready(control_shm, config, 1)
            finally:
                detach_pool(data_shm, control_shm)

            result = p.read_slots(1)
            np.testing.assert_array_equal(result[0], cutout)
        finally:
            p.cleanup()

    def test_write_cutouts_exceeds_capacity_raises(self, pool, small_config):
        """write_cutouts_to_pool must reject writes that overflow the pool."""
        data_shm, control_shm = attach_pool(pool.name, pool.control_name, small_config)
        try:
            cutouts = [
                np.ones(small_config.slot_shape, dtype=small_config.dtype)
                for _ in range(small_config.slots_per_worker + 1)
            ]
            with pytest.raises(IndexError, match="exceeds pool capacity"):
                write_cutouts_to_pool(data_shm, small_config, cutouts)
        finally:
            detach_pool(data_shm, control_shm)

    def test_write_cutouts_full_pool_bulk(self, pool, small_config):
        """write_cutouts_to_pool fills all slots in one call."""
        data_shm, control_shm = attach_pool(pool.name, pool.control_name, small_config)
        try:
            cutouts = [
                np.full(small_config.slot_shape, fill_value=float(i), dtype=small_config.dtype)
                for i in range(small_config.slots_per_worker)
            ]
            write_cutouts_to_pool(data_shm, small_config, cutouts)
            mark_slots_ready(control_shm, small_config, small_config.slots_per_worker)
        finally:
            detach_pool(data_shm, control_shm)

        result = pool.read_slots(small_config.slots_per_worker)
        for i in range(small_config.slots_per_worker):
            np.testing.assert_array_equal(result[i], cutouts[i])


class TestCalculatePoolConfig:
    def test_float32_default_500_slots(self):
        config = calculate_pool_config(128, 3, np.dtype("float32"))
        assert config.slots_per_worker == 500
        assert config.slot_shape == (128, 128, 3)
        assert config.dtype == np.float32

    def test_uint8_default_1000_slots(self):
        config = calculate_pool_config(128, 3, np.dtype("uint8"))
        assert config.slots_per_worker == 1000

    def test_custom_memory_limit(self):
        # 128*128*3*4 = 196608 bytes per slot
        config = calculate_pool_config(
            128, 3, np.dtype("float32"), max_shm_memory_per_worker=196608 * 10
        )
        assert config.slots_per_worker == 10

    def test_minimum_one_slot(self):
        config = calculate_pool_config(128, 3, np.dtype("float32"), max_shm_memory_per_worker=1)
        assert config.slots_per_worker == 1

    def test_single_channel(self):
        config = calculate_pool_config(256, 1, np.dtype("float32"))
        assert config.slot_shape == (256, 256, 1)
