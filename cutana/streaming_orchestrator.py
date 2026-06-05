#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Streaming Orchestrator for Cutana - handles batch-by-batch streaming workflows.

This module provides StreamingOrchestrator, a specialized orchestrator for
processing large catalogues in batches with adaptive parallel processing.

Key features:
- Adaptive parallelism: spawns workers based on consumer demand
- Per-worker shared memory pools for zero-copy cutout transfer
- Decoupled internal/user batch sizes to reduce subprocess overhead
- Unordered result delivery: next_batch() returns whichever batch finishes first
"""

import json
import math
import queue
import select
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from dotmap import DotMap
from loguru import logger

from .catalogue_preprocessor import preprocess_catalogue
from .catalogue_streamer import CatalogueBatchReader, CatalogueIndex
from .orchestrator import Orchestrator
from .shm_pool import ShmPool, calculate_pool_config


class StreamingOrchestrator(Orchestrator):
    """
    Orchestrator specialized for streaming batch-by-batch processing.

    Uses memory-efficient streaming with adaptive parallel workers.
    Each worker gets its own pre-allocated shared memory pool.

    Usage:
        orchestrator = StreamingOrchestrator(config)
        orchestrator.init_streaming(batch_size=500, write_to_disk=False, max_workers=4, min_workers=2)

        for i in range(orchestrator.get_batch_count()):
            result = orchestrator.next_batch()
            # Process result["cutouts"] or result["zarr_path"]
    """

    def __init__(self, config: DotMap, status_panel=None):
        """Initialize the streaming orchestrator."""
        super().__init__(config, status_panel)

        # Streaming state - initialized by init_streaming()
        self._streaming_initialized = False
        self._catalogue_index: Optional[CatalogueIndex] = None
        self._batch_reader: Optional[CatalogueBatchReader] = None
        self._batch_ranges: List[List[int]] = []  # internal batch ranges
        self._streaming_write_to_disk = True

        # Parallel state
        self._shm_pools: List[Optional[ShmPool]] = []  # one per worker slot
        self._pool_to_worker: Dict[int, str] = {}  # pool_idx -> active process_id
        self._worker_to_pool: Dict[str, int] = {}  # process_id -> pool_idx
        self._max_workers = 4
        self._min_workers = 1
        self._next_internal_batch_idx = 0  # next internal batch to assign

        # User/internal batch decoupling
        self._user_batch_size = 0
        self._internal_batch_size = 0
        self._user_batch_count = 0
        self._streaming_batch_index = 0  # user batches returned so far

        # Result accumulation
        self._ready_results: List[Dict[str, Any]] = []  # completed batch results (unordered)
        # Rolling buffer of cutouts/metadata not yet emitted to the user. Internal
        # batch results are appended here and drained in batch_size batches, so
        # remainders carry over across internal batch boundaries instead of leaking
        # out as short batches mid-stream. A deque keeps append/popleft O(1) so
        # emission cost stays O(batch_size) regardless of how large the buffer grows.
        self._pending_cutouts: Deque[np.ndarray] = deque()
        self._pending_metadata: Deque = deque()

        # Per-worker accumulation (chunk arrays received so far from each worker)
        self._worker_cutouts: Dict[str, List[np.ndarray]] = {}
        self._worker_metadata: Dict[str, List] = {}

        # Worker stdout reader state (used on Windows to avoid blocking readline)
        self._worker_stdout_queues: Dict[str, queue.Queue] = {}
        self._worker_reader_threads: Dict[str, threading.Thread] = {}

        # Profiling / Gantt chart data
        self._worker_events: List[Tuple] = []  # (worker_id, event, timestamp)

        # Pool config (computed once in init_streaming)
        self._pool_config = None

    def init_streaming(
        self,
        batch_size: int,
        write_to_disk: bool = True,
        max_workers: int = 4,
        min_workers: int = 1,
        max_shm_memory_consumption: Optional[int] = None,
    ) -> None:
        """
        Initialize streaming mode for batch-by-batch processing.

        Args:
            batch_size: Maximum number of sources per user-facing batch
            write_to_disk: If True, write batches to zarr; if False, return cutouts in memory
            max_workers: Maximum number of parallel workers (default: 4)
            min_workers: Number of workers to pre-spawn at init time so they are
                already processing when next_batch() is first called (default: 1)
            max_shm_memory_consumption: Total SHM memory budget in bytes (None = auto)
        """
        if min_workers < 1:
            raise ValueError(f"min_workers must be >= 1, got {min_workers}")
        if min_workers > max_workers:
            raise ValueError(
                f"min_workers ({min_workers}) cannot exceed max_workers ({max_workers})"
            )

        if self.config.do_only_cutout_extraction and not write_to_disk:
            raise ValueError(
                "do_only_cutout_extraction=True requires write_to_disk=True: "
                "raw cutout extraction has no defined semantics for in-memory streaming."
            )

        catalogue_path = self.config.source_catalogue

        logger.info(
            f"Initializing streaming: batch_size={batch_size}, "
            f"write_to_disk={write_to_disk}, max_workers={max_workers}, min_workers={min_workers}"
        )

        # Use parent's shared catalogue index and reader initialization
        self._catalogue_index, self._batch_reader = self._init_catalogue_index_and_reader(
            catalogue_path
        )
        total_sources = self._catalogue_index.row_count

        # Compute internal batch size
        # For disk mode: use user batch_size (can't split disk output)
        # For in-memory mode: use larger internal batch to reduce subprocess overhead
        self._user_batch_size = batch_size
        if write_to_disk:
            self._internal_batch_size = batch_size
        else:
            self._internal_batch_size = max(batch_size, self.config.N_batch_cutout_process)

        # Get optimized batch ranges using internal batch size
        self._batch_ranges = self._catalogue_index.get_optimized_batch_ranges(
            max_sources_per_batch=self._internal_batch_size,
            min_sources_per_batch=500,
            max_fits_sets_per_batch=50,
        )

        # Compute user batch count
        if write_to_disk:
            # Disk mode: one user batch per internal batch
            self._user_batch_count = len(self._batch_ranges)
        else:
            # In-memory mode: split internal batches into user-sized chunks
            self._user_batch_count = math.ceil(total_sources / batch_size)

        logger.info(
            f"Created {len(self._batch_ranges)} internal batches, "
            f"{self._user_batch_count} user batches"
        )

        # Store streaming state
        self._streaming_batch_index = 0
        self._streaming_write_to_disk = write_to_disk
        self._max_workers = max_workers
        self._min_workers = min_workers
        self._next_internal_batch_idx = 0
        self._streaming_initialized = True

        # Reset parallel state
        self._shm_pools = [None] * max_workers
        self._pool_to_worker = {}
        self._worker_to_pool = {}
        self._ready_results = []
        self._pending_cutouts = deque()
        self._pending_metadata = deque()
        self._worker_cutouts = {}
        self._worker_metadata = {}
        self._worker_stdout_queues = {}
        self._worker_reader_threads = {}
        self._worker_events = []

        # Compute per-worker pool config (only for in-memory, uniform cutout sizes)
        self._pool_config = None
        if not write_to_disk and not self.config.do_only_cutout_extraction:
            n_channels = max(len(w) for w in self.config.channel_weights.values())
            per_worker_budget = (
                max_shm_memory_consumption // max_workers
                if max_shm_memory_consumption is not None
                else None
            )
            self._pool_config = calculate_pool_config(
                self.config.target_resolution,
                n_channels,
                np.dtype(self.config.data_type),
                max_shm_memory_per_worker=per_worker_budget,
            )
            logger.info(
                f"SHM pool config: {self._pool_config.slots_per_worker} slots/worker, "
                f"{self._pool_config.total_bytes / 1024 / 1024:.1f}MB/worker"
            )

        # Initialize job tracking
        self.job_tracker.start_job(total_sources)

        # Pre-spawn min_workers workers so they are already processing on first next_batch() call
        workers_to_spawn = min(min_workers, len(self._batch_ranges))
        for _ in range(workers_to_spawn):
            self._spawn_next_worker()

        logger.info(
            f"Streaming initialized: {len(self._batch_ranges)} internal batches, "
            f"{total_sources} sources, max_workers={max_workers}, "
            f"pre-spawned {workers_to_spawn} workers"
        )

    def _get_batch_df(self, batch_index: int):
        """Get the DataFrame for a specific internal batch."""
        row_indices = self._batch_ranges[batch_index]
        batch_df = self._batch_reader.read_rows(row_indices)
        batch_df = preprocess_catalogue(batch_df)
        return batch_df

    def _allocate_or_reuse_pool(self, pool_idx: int) -> Optional[ShmPool]:
        """Get or create SHM pool for a worker slot."""
        if self._pool_config is None:
            return None

        if self._shm_pools[pool_idx] is not None:
            self._shm_pools[pool_idx].reset()
            return self._shm_pools[pool_idx]

        pool = ShmPool(self._pool_config)
        self._shm_pools[pool_idx] = pool
        logger.debug(f"Created SHM pool {pool_idx}: {pool.name}")
        return pool

    def _find_free_pool_index(self) -> Optional[int]:
        """Find a pool index not currently assigned to an active worker."""
        for i in range(self._max_workers):
            if i not in self._pool_to_worker:
                return i
        return None

    def _use_windows_stdout_thread_mode(self) -> bool:
        """Return True only for Windows runs configured with at least 2 workers."""
        return sys.platform == "win32" and self._max_workers >= 2

    def _spawn_next_worker(self) -> Optional[str]:
        """Spawn the next worker subprocess if batches remain."""
        if self._next_internal_batch_idx >= len(self._batch_ranges):
            return None

        pool_idx = self._find_free_pool_index()
        if pool_idx is None:
            return None

        batch_index = self._next_internal_batch_idx
        self._next_internal_batch_idx += 1

        job_df = self._get_batch_df(batch_index)
        unique_id = str(uuid.uuid4())[:8]
        process_id = f"streaming_{batch_index:03d}_{unique_id}"

        # Allocate or reuse pool
        pool = self._allocate_or_reuse_pool(pool_idx)

        logger.info(
            f"Spawning worker {process_id} for internal batch {batch_index + 1} "
            f"({len(job_df)} sources, pool_slot={pool_idx})"
        )

        self._spawn_cutout_process(
            process_id, job_df, write_to_disk=self._streaming_write_to_disk, shm_pool=pool
        )

        # Track pool-worker mapping
        self._pool_to_worker[pool_idx] = process_id
        self._worker_to_pool[process_id] = pool_idx
        self._worker_cutouts[process_id] = []
        self._worker_metadata[process_id] = []

        if self._use_windows_stdout_thread_mode():
            # On Windows, select() does not support subprocess pipes.
            # Use a dedicated reader thread per worker and drain its queue in _poll_workers().
            self._start_worker_stdout_reader(process_id)

        # Record event for Gantt chart
        self._worker_events.append((process_id, "start", time.time()))

        return process_id

    def _start_worker_stdout_reader(self, process_id: str) -> None:
        """
        Start a background stdout reader thread for one worker.

        Args:
            process_id: Worker process identifier in active_processes.

        Returns:
            None.

        Notes:
            This function is called only for Windows runs configured with
            at least 2 workers. The thread blocks on readline() and pushes
            complete lines to a per-worker queue.
        """

        proc = self.active_processes.get(process_id)
        if proc is None or proc.stdout is None:
            return

        line_queue = queue.Queue()
        self._worker_stdout_queues[process_id] = line_queue

        def _reader() -> None:
            try:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    line_queue.put(line)
            except Exception as e:
                logger.debug(f"stdout reader thread error for {process_id}: {e}")
            finally:
                line_queue.put(None)

        reader_thread = threading.Thread(
            target=_reader,
            name=f"streaming-stdout-reader-{process_id}",
            daemon=True,
        )
        reader_thread.start()
        self._worker_reader_threads[process_id] = reader_thread

    def _drain_worker_stdout_queue(self, proc_id: str) -> None:
        """
        Drain all queued stdout lines for one worker and handle each message.

        Args:
            proc_id: Worker process identifier.

        Returns:
            None.
        """
        line_queue = self._worker_stdout_queues.get(proc_id)
        if line_queue is None:
            return

        while True:
            try:
                line = line_queue.get_nowait()
            except queue.Empty:
                break

            if line is None:
                break

            self._handle_worker_line(proc_id, line)

    def _cleanup_worker_stdout_reader(self, proc_id: str) -> None:
        """
        Clean up reader-thread state for one worker.

        Args:
            proc_id: Worker process identifier.

        Returns:
            None.

        Notes:
            Reader threads run in the parent process (not inside the worker
            subprocess). We therefore clean up this state independently of
            worker terminate()/exit success.
        """
        self._worker_stdout_queues.pop(proc_id, None)

        reader_thread = self._worker_reader_threads.pop(proc_id, None)
        if reader_thread is not None and reader_thread.is_alive():
            try:
                reader_thread.join(timeout=0.1)
            except RuntimeError as e:
                logger.warning(f"Failed joining stdout reader thread for {proc_id}: {e}")
                return

            if reader_thread.is_alive():
                logger.warning(
                    f"stdout reader thread for {proc_id} did not exit within cleanup timeout"
                )

    def _maybe_spawn_workers(self) -> None:
        """Spawn additional workers if capacity allows and batches remain."""
        active_count = len(self._pool_to_worker)
        while active_count < self._max_workers and self._next_internal_batch_idx < len(
            self._batch_ranges
        ):
            result = self._spawn_next_worker()
            if result is None:
                break
            active_count += 1

    def _poll_workers(self) -> None:
        """
        Poll all active workers for results using select on stdout pipes.

        Handles:
        - chunk_ready: read cutouts from SHM pool or parse inline
        - complete: assemble full result and add to _ready_results
        - crashed/finished workers: clean up
        """
        # Separate pipe-based (in-memory) and file-based (disk) workers
        pipe_procs = {}
        disk_proc_ids = []
        for proc_id, proc in list(self.active_processes.items()):
            if proc.stdout is not None:
                pipe_procs[proc.stdout] = proc_id
            else:
                disk_proc_ids.append(proc_id)

        # Poll pipe-based workers via select
        if pipe_procs:
            if self._use_windows_stdout_thread_mode():
                for proc_id in pipe_procs.values():
                    self._drain_worker_stdout_queue(proc_id)
            elif sys.platform == "win32":
                # Single-worker Windows mode: avoid select() on pipes.
                for proc_id in pipe_procs.values():
                    self._try_read_worker(proc_id)
            else:
                ready, _, _ = select.select(list(pipe_procs.keys()), [], [], 0.05)
                for fd in ready:
                    proc_id = pipe_procs[fd]
                    self._try_read_worker(proc_id)

            # Check for completed/crashed pipe workers
            for proc_id in list(self.active_processes.keys()):
                if proc_id in disk_proc_ids:
                    continue
                proc = self.active_processes.get(proc_id)
                if proc is None:
                    continue
                exit_code = proc.poll()
                if exit_code is not None:
                    if exit_code != 0:
                        logger.error(f"Worker {proc_id} crashed with exit code {exit_code}")
                    self._complete_worker(proc_id)

        # Poll disk-mode workers via process.poll()
        for proc_id in disk_proc_ids:
            proc = self.active_processes.get(proc_id)
            if proc is None:
                continue
            exit_code = proc.poll()
            if exit_code is not None:
                if exit_code != 0:
                    logger.error(f"Disk worker {proc_id} failed with exit code {exit_code}")
                else:
                    logger.info(f"Disk worker {proc_id} completed successfully")
                self._complete_worker(proc_id)

    def _try_read_worker(self, proc_id: str) -> None:
        """Try to read a JSON message from a worker's stdout."""
        proc = self.active_processes.get(proc_id)
        if proc is None or proc.stdout is None:
            return

        line = proc.stdout.readline()
        if not line:
            return

        self._handle_worker_line(proc_id, line)

    def _handle_worker_line(self, proc_id: str, line: str) -> None:
        """
        Parse and handle one JSON line emitted by a worker.

        Args:
            proc_id: Worker process identifier.
            line: One newline-delimited JSON message.

        Returns:
            None.
        """

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON from {proc_id}: {line[:100]}")
            return

        msg_type = msg.get("type")

        if msg_type == "chunk_ready":
            self._handle_chunk_ready(proc_id, msg)
        elif msg_type == "chunk":
            raise RuntimeError(
                f"Received legacy 'chunk' message from worker {proc_id}. "
                "StreamingOrchestrator requires pool mode. "
                "Verify shm_pool_name is set in worker config."
            )
        elif msg_type == "complete":
            total = msg.get("total_cutouts", 0)
            logger.info(f"Worker {proc_id} completed: {total} cutouts")
            self._complete_worker(proc_id)

    def _handle_chunk_ready(self, proc_id: str, msg: dict) -> None:
        """Handle chunk_ready message from pool-mode worker."""
        pool_idx = self._worker_to_pool[proc_id]
        pool = self._shm_pools[pool_idx]

        slot_count = msg["slot_count"]
        metadata = msg["metadata"]

        # Read cutouts from pool
        cutouts = pool.read_slots(slot_count)
        self._worker_cutouts[proc_id].append(cutouts)
        self._worker_metadata[proc_id].extend(metadata)

        logger.debug(f"Worker {proc_id}: received {slot_count} cutouts from pool")

        # Send ACK
        proc = self.active_processes[proc_id]
        proc.stdin.write("ACK\n")
        proc.stdin.flush()

    def _complete_worker(self, proc_id: str) -> None:
        """Finalize a completed worker: assemble result, free pool slot."""
        # Record event
        self._worker_events.append((proc_id, "end", time.time()))

        # Free pool slot
        pool_idx = self._worker_to_pool.pop(proc_id, None)
        if pool_idx is not None:
            self._pool_to_worker.pop(pool_idx, None)

        # Clean up process
        proc = self.active_processes.pop(proc_id, None)
        if proc is not None:
            try:
                proc.wait(timeout=5.0)
            except Exception as e:
                logger.warning(f"Worker {proc_id} did not exit cleanly: {e}")

        if self._use_windows_stdout_thread_mode():
            # Reader threads exist only in Windows stdout-thread mode.
            self._cleanup_worker_stdout_reader(proc_id)

        if self._streaming_write_to_disk:
            # Disk mode: build zarr path result
            output_dir = Path(self.config.output_dir)
            zarr_subfolder = f"batch_{proc_id}"
            zarr_path = output_dir / zarr_subfolder / "images.zarr"
            result = {
                "zarr_path": str(zarr_path),
                "metadata": self._worker_metadata.pop(proc_id, []),
            }
            self._worker_cutouts.pop(proc_id, None)
            self._ready_results.append(result)
            logger.info(f"Worker {proc_id} finalized: disk mode, {zarr_path}")
        else:
            # In-memory mode: assemble cutouts
            cutout_chunks = self._worker_cutouts.pop(proc_id, [])
            metadata = self._worker_metadata.pop(proc_id, [])

            # Flatten chunk arrays into a single list of per-source cutout arrays.
            # This avoids the extra full-batch copy from np.concatenate.
            all_cutouts = []
            for cutout_chunk in cutout_chunks:
                all_cutouts.extend(cutout_chunk)

            n_cutouts = len(all_cutouts)
            result = {
                "cutouts": all_cutouts,
                "metadata": metadata,
            }
            self._ready_results.append(result)
            logger.info(f"Worker {proc_id} finalized: {n_cutouts} cutouts")

    def _consume_disk_result(self) -> Dict[str, Any]:
        """Pop a ready disk-mode result and tag it with the next batch number.

        Disk mode keeps a 1:1 mapping between internal batches and user batches
        (zarr output cannot be re-split), so each result passes straight through.
        """
        result = self._ready_results.pop(0)
        self._streaming_batch_index += 1
        result["batch_number"] = self._streaming_batch_index
        return result

    def _absorb_ready_results(self) -> None:
        """Move all ready in-memory results into the rolling pending buffer.

        Appending here (rather than emitting per internal batch) is what lets a
        sub-batch_size remainder carry over and combine with the next
        internal batch instead of being flushed as a short batch.
        """
        while self._ready_results:
            result = self._ready_results.pop(0)
            self._pending_cutouts.extend(result["cutouts"])
            self._pending_metadata.extend(result["metadata"])

    def _work_remaining(self) -> bool:
        """Return True while more cutouts may still arrive from workers."""
        return (
            bool(self._ready_results)
            or bool(self.active_processes)
            or self._next_internal_batch_idx < len(self._batch_ranges)
        )

    def _emit_pending_user_batch(self) -> Dict[str, Any]:
        """Emit up to user_batch_size cutouts from the front of the pending buffer."""
        user_size = self._user_batch_size
        n = min(user_size, len(self._pending_cutouts))

        # popleft (O(n)) rather than slicing (O(buffer)) so a large buffer never
        # forces a full re-copy of the remaining references on every batch.
        cutouts = [self._pending_cutouts.popleft() for _ in range(n)]
        metadata = [self._pending_metadata.popleft() for _ in range(n)]

        self._streaming_batch_index += 1
        return {
            "batch_number": self._streaming_batch_index,
            "cutouts": cutouts,
            "metadata": metadata,
        }

    def next_batch(self) -> Dict[str, Any]:
        """
        Get the next batch of cutouts.

        Returns whichever worker finishes first (unordered). Adaptively spawns
        additional workers if the caller is consuming faster than production.

        In-memory batches are assembled from a rolling buffer so that every batch
        except the last contains exactly ``batch_size`` cutouts; only the final
        batch may be smaller. Disk-mode batches map 1:1 to internal batches.

        Returns:
            Dictionary with:
            - 'batch_number': 1-indexed batch number
            - 'cutouts': list of cutout arrays (if write_to_disk=False)
            - 'metadata': list of source metadata dicts

        Raises:
            RuntimeError: If not initialized or no more batches
        """
        if not self._streaming_initialized:
            raise RuntimeError("Streaming not initialized. Call init_streaming() first.")

        if self._streaming_batch_index >= self._user_batch_count:
            raise RuntimeError("No more batches available.")

        timeout_start = time.time()
        timeout_seconds = self.config.max_workflow_time_seconds

        while True:
            self._poll_workers()

            if self._streaming_write_to_disk:
                if self._ready_results:
                    self._maybe_spawn_workers()
                    return self._consume_disk_result()
            else:
                self._absorb_ready_results()
                # Emit a full batch as soon as one is available. Only emit a
                # smaller (final) batch once no further cutouts can arrive.
                if len(self._pending_cutouts) >= self._user_batch_size:
                    self._maybe_spawn_workers()
                    return self._emit_pending_user_batch()
                if not self._work_remaining():
                    if self._pending_cutouts:
                        return self._emit_pending_user_batch()
                    raise RuntimeError(
                        "All workers completed but no cutouts remain for the "
                        "expected batch. This may indicate worker failures."
                    )

            # Adaptive spawn: consumer is waiting, spawn more workers
            self._maybe_spawn_workers()

            # Disk-mode deadlock guard (in-memory is handled by _work_remaining above)
            if (
                self._streaming_write_to_disk
                and not self.active_processes
                and self._next_internal_batch_idx >= len(self._batch_ranges)
                and not self._ready_results
            ):
                raise RuntimeError(
                    "All workers completed but no results available. "
                    "This may indicate worker failures."
                )

            if time.time() - timeout_start > timeout_seconds:
                raise RuntimeError("Timeout waiting for batch completion")

            time.sleep(0.05)

    def get_batch_count(self) -> int:
        """Get total number of user-facing batches."""
        if not self._streaming_initialized:
            raise RuntimeError("Streaming not initialized. Call init_streaming() first.")
        return self._user_batch_count

    def get_worker_events(self) -> List[Tuple]:
        """Get worker lifecycle events for Gantt chart generation."""
        return list(self._worker_events)

    def cleanup(self):
        """Clean up resources including workers and SHM pools."""
        # Terminate active workers
        for proc_id in list(self.active_processes.keys()):
            logger.info(f"Cleaning up worker {proc_id}")
            self._terminate_process(proc_id)

        # Clean up pool mappings
        self._pool_to_worker.clear()
        self._worker_to_pool.clear()
        self._worker_cutouts.clear()
        self._worker_metadata.clear()
        self._worker_stdout_queues.clear()
        self._worker_reader_threads.clear()

        # Clean up SHM pools
        for i, pool in enumerate(self._shm_pools):
            if pool is not None:
                pool.cleanup()
                self._shm_pools[i] = None

        # Close batch reader
        if self._batch_reader:
            self._batch_reader.close()
            self._batch_reader = None

        super().cleanup()

    def _terminate_process(self, process_id: str) -> None:
        """Terminate a process and clean up."""
        if process_id in self.active_processes:
            try:
                self.active_processes[process_id].terminate()
                self.active_processes[process_id].wait(timeout=10.0)
            except Exception as e:
                logger.error(f"Failed to terminate process {process_id}: {e}")
            finally:
                self.active_processes.pop(process_id, None)

        # Reader threads are parent-side resources. Clean them up regardless
        # of subprocess terminate()/wait() outcome.
        if self._use_windows_stdout_thread_mode():
            self._cleanup_worker_stdout_reader(process_id)

        # Clean pool mapping
        pool_idx = self._worker_to_pool.pop(process_id, None)
        if pool_idx is not None:
            self._pool_to_worker.pop(pool_idx, None)
        self._worker_cutouts.pop(process_id, None)
        self._worker_metadata.pop(process_id, None)
