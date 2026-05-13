#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Shared memory pool for streaming orchestrator.

Provides pre-allocated shared memory blocks for zero-copy cutout transfer
between worker subprocesses and the orchestrator. Each worker gets its own
ShmPool instance; pools are reused across worker lifetimes to avoid
repeated allocation/deallocation overhead.

Slot lifecycle:
    EMPTY -> READY (worker writes cutout and marks slot ready)
    READY -> EMPTY (orchestrator reads and releases slot)
"""

import atexit
import sys
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Tuple

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class ShmPoolConfig:
    """Configuration for a single worker's shared memory pool."""

    slot_shape: Tuple[int, ...]  # (H, W, C) per cutout
    dtype: np.dtype
    slots_per_worker: int  # number of cutout slots in this pool

    @property
    def slot_bytes(self) -> int:
        return int(np.prod(self.slot_shape)) * self.dtype.itemsize

    @property
    def total_bytes(self) -> int:
        return self.slots_per_worker * self.slot_bytes

    def to_dict(self) -> dict:
        """Serialize for passing to subprocess via config TOML."""
        return {
            "slot_shape": list(self.slot_shape),
            "dtype": str(self.dtype),
            "slots_per_worker": self.slots_per_worker,
        }

    @staticmethod
    def from_dict(d: dict) -> "ShmPoolConfig":
        """Deserialize from config dict."""
        return ShmPoolConfig(
            slot_shape=tuple(d["slot_shape"]),
            dtype=np.dtype(d["dtype"]),
            slots_per_worker=d["slots_per_worker"],
        )


class ShmPool:
    """
    Pre-allocated shared memory pool for one worker.

    Creates two shared memory blocks:
    - data_shm: holds cutout pixel data (slots_per_worker * slot_bytes)
    - control_shm: int32 array with one state value per slot

    The orchestrator creates the pool; worker subprocesses attach to it
    by name and write cutouts into slots.
    """

    # Slot states stored in the control array (int32 values)
    SLOT_EMPTY = 0
    SLOT_READY = 2

    def __init__(self, config: ShmPoolConfig):
        self.config = config
        self._cleaned_up = False

        # Create data shared memory
        data_size = config.total_bytes
        if data_size <= 0:
            raise ValueError(f"Invalid pool size: {data_size} bytes")

        self._data_shm = shared_memory.SharedMemory(create=True, size=data_size)

        # Create control shared memory (one int32 per slot)
        control_size = config.slots_per_worker * np.dtype(np.int32).itemsize
        self._control_shm = shared_memory.SharedMemory(create=True, size=control_size)

        # Initialize all slots to EMPTY
        control_array = np.ndarray(
            (config.slots_per_worker,), dtype=np.int32, buffer=self._control_shm.buf
        )
        control_array[:] = ShmPool.SLOT_EMPTY

        # Register cleanup on exit as safety net
        atexit.register(self.cleanup)

        logger.debug(
            f"ShmPool created: data={self._data_shm.name} ({data_size / 1024 / 1024:.1f}MB), "
            f"control={self._control_shm.name}, slots={config.slots_per_worker}"
        )

    @property
    def name(self) -> str:
        return self._data_shm.name

    @property
    def control_name(self) -> str:
        return self._control_shm.name

    def reset(self) -> None:
        """Reset all slots to EMPTY for reuse with a new worker."""
        control_array = np.ndarray(
            (self.config.slots_per_worker,), dtype=np.int32, buffer=self._control_shm.buf
        )
        control_array[:] = ShmPool.SLOT_EMPTY

    def read_slots(self, slot_count: int) -> np.ndarray:
        """
        Read cutouts from the first slot_count slots and reset them to EMPTY.

        Args:
            slot_count: Number of slots to read, starting from slot 0

        Returns:
            numpy array of shape (slot_count, *slot_shape) with cutout data copied out
        """
        config = self.config

        if slot_count > config.slots_per_worker:
            raise IndexError(f"slot_count={slot_count} exceeds pool size {config.slots_per_worker}")

        data_length = slot_count * config.slot_bytes
        flat_shape = (data_length // config.dtype.itemsize,)

        data_view = np.ndarray(flat_shape, dtype=config.dtype, buffer=self._data_shm.buf)
        result = data_view.reshape(slot_count, *config.slot_shape).copy()

        control_array = np.ndarray(
            (config.slots_per_worker,), dtype=np.int32, buffer=self._control_shm.buf
        )
        control_array[:slot_count] = ShmPool.SLOT_EMPTY

        return result

    def cleanup(self) -> None:
        """Close and unlink both shared memory blocks."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        for shm, label in [(self._data_shm, "data"), (self._control_shm, "control")]:
            try:
                shm.close()
                shm.unlink()
                logger.debug(f"ShmPool {label} cleaned up: {shm.name}")
            except Exception as e:
                logger.debug(f"ShmPool {label} cleanup error: {e}")

    def __del__(self):
        self.cleanup()


# === Worker-side functions (called from subprocess) ===


def attach_pool(
    pool_name: str, control_name: str, config: ShmPoolConfig
) -> Tuple[shared_memory.SharedMemory, shared_memory.SharedMemory]:
    """
    Attach to an existing SHM pool from a worker subprocess.

    Returns (data_shm, control_shm). The worker must call detach_pool()
    when done — never unlink, as the orchestrator owns the memory.
    """
    data_shm = shared_memory.SharedMemory(name=pool_name, create=False)
    control_shm = shared_memory.SharedMemory(name=control_name, create=False)

    # Unregister from resource tracker so the subprocess doesn't try to
    # unlink SHM it doesn't own when the interpreter shuts down.
    # resource_tracker uses POSIX "/" prefix and is not available on Windows.
    if sys.platform != "win32":
        resource_tracker.unregister("/" + data_shm.name, "shared_memory")
        resource_tracker.unregister("/" + control_shm.name, "shared_memory")

    logger.debug(f"Attached to SHM pool: data={pool_name}, control={control_name}")
    return data_shm, control_shm


def write_cutouts_to_pool(
    data_shm: shared_memory.SharedMemory,
    config: ShmPoolConfig,
    cutouts: list,
) -> None:
    """
    Write a batch of cutouts into the pool starting at slot 0.

    All cutouts must match config.slot_shape and config.dtype.
    len(cutouts) must not exceed config.slots_per_worker.
    """
    n = len(cutouts)
    if n > config.slots_per_worker:
        raise IndexError(
            f"Writing {n} cutouts exceeds pool capacity of {config.slots_per_worker} slots"
        )
    pool_view = np.ndarray((n, *config.slot_shape), dtype=config.dtype, buffer=data_shm.buf)
    for slot_index, cutout in enumerate(cutouts):
        pool_view[slot_index] = cutout


def mark_slots_ready(
    control_shm: shared_memory.SharedMemory,
    config: ShmPoolConfig,
    slot_count: int,
) -> None:
    """Mark the first slot_count slots as READY after writing cutout data."""
    control_array = np.ndarray((config.slots_per_worker,), dtype=np.int32, buffer=control_shm.buf)
    control_array[:slot_count] = ShmPool.SLOT_READY


def detach_pool(
    data_shm: shared_memory.SharedMemory,
    control_shm: shared_memory.SharedMemory,
) -> None:
    """
    Detach from SHM pool (worker side). Closes but does NOT unlink.
    """
    data_shm.close()
    control_shm.close()
    logger.debug("Detached from SHM pool")


def calculate_pool_config(
    target_resolution: int,
    n_channels: int,
    dtype: np.dtype,
    max_shm_memory_per_worker: int | None = None,
) -> ShmPoolConfig:
    """
    Calculate SHM pool configuration from image parameters.

    Args:
        target_resolution: Cutout size (H=W)
        n_channels: Number of image channels
        dtype: Output data type
        max_shm_memory_per_worker: Max bytes per worker pool. None = auto-calculate.

    Returns:
        ShmPoolConfig with computed slots_per_worker
    """
    dtype = np.dtype(dtype)
    slot_shape = (target_resolution, target_resolution, n_channels)
    slot_bytes = int(np.prod(slot_shape)) * dtype.itemsize

    if max_shm_memory_per_worker is not None:
        slots_per_worker = max_shm_memory_per_worker // slot_bytes
    else:
        # Default: 1000 slots for uint8, 500 for float types
        if dtype.itemsize <= 1:
            slots_per_worker = 1000
        else:
            slots_per_worker = 500

    slots_per_worker = max(slots_per_worker, 1)

    return ShmPoolConfig(
        slot_shape=slot_shape,
        dtype=dtype,
        slots_per_worker=slots_per_worker,
    )
