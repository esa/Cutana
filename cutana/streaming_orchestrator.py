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

from .catalogue_preprocessor import DUPLICATE_CHECK_THRESHOLD, preprocess_catalogue
from .catalogue_streamer import CatalogueBatchReader, CatalogueIndex
from .orchestrator import Orchestrator
from .profiling_types import WorkerInfo
from .shm_pool import ShmPool, calculate_pool_config

# In-memory streaming holds a whole internal batch of cutouts twice: once accumulated in
# the worker before it streams, once in the parent until the worker completes.
_IN_MEMORY_BATCH_COPIES = 2
# Share of the load balancer's memory budget those in-flight cutouts may take. The rest
# is for the FITS tile caches, which dwarf the cutouts -- a Euclid MER tile is ~1.5 GB.
_IN_MEMORY_CUTOUT_BUDGET_FRACTION = 0.25


# Cap on how many per-worker shortfalls are spelled out in an error message before
# the rest are summarised as a count.
_MAX_SHORTFALLS_IN_MESSAGE = 10


class StreamingOrchestrator(Orchestrator):
    """
    Orchestrator specialized for streaming batch-by-batch processing.

    Uses memory-efficient streaming with adaptive parallel workers.
    Each worker gets its own pre-allocated shared memory pool.

    Usage:
        orchestrator = StreamingOrchestrator(config)
        try:
            orchestrator.init_streaming(
                batch_size=500, write_to_disk=False, max_workers=4, min_workers=2
            )

            for i in range(orchestrator.get_batch_count()):
                result = orchestrator.next_batch()
                # Process result["cutouts"] or result["zarr_path"]
        finally:
            # Required: workers block indefinitely waiting to hand over their next
            # chunk, so abandoning the loop without cleanup() leaves them running
            # and holding their shared memory pools.
            orchestrator.cleanup()
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

        # Delivery accounting. next_batch() derives its batch count from the
        # catalogue row count, so any cutout a worker fails to deliver turns into
        # an unexplained "no cutouts remain" failure at the tail of the run.
        # Tracking what each worker was asked for vs. what it handed back lets
        # that failure name the workers actually responsible.
        self._worker_source_counts: Dict[str, int] = {}  # process_id -> sources assigned
        self._worker_reported_totals: Dict[str, int] = {}  # process_id -> cutouts it claims
        self._worker_shortfalls: List[Tuple[str, int, int]] = []  # (id, assigned, delivered)
        # A failing worker prints its exception message before exiting non-zero.
        # Keeping it here is what lets the exit-code failure quote the real cause
        # instead of sending the user off to a log file.
        self._worker_errors: Dict[str, str] = {}

        # Worker stdout reader state (used on Windows to avoid blocking readline)
        self._worker_stdout_queues: Dict[str, queue.Queue] = {}
        self._worker_reader_threads: Dict[str, threading.Thread] = {}

        # Per-worker detail keyed by process_id: batch composition (recorded at
        # spawn) plus per-stage timings reported on completion (issue #354). This
        # carries everything the old (worker_id, event, timestamp) event list
        # held (start/end times) and more, so it is the single profiling source.
        self._worker_info: Dict[str, WorkerInfo] = {}

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
            batch_size: Sources per user-facing batch. In-memory batches are cut to
                exactly this size from a rolling buffer, so it is an exact size there
                and is independent of how sources are split across worker processes.
                In disk mode each batch is one zarr archive written by one worker, so
                it cannot be re-split: there ``batch_size`` is a *minimum*, and
                :meth:`_resolve_internal_batch_size` may enlarge it.
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

        # The load balancer has never been run on the streaming path, so
        # max_sources_per_process was unset here; run it as the non-streaming path does.
        self.load_balancer.update_config_with_loadbalancing(self.config, total_sources)
        self._user_batch_size = batch_size
        self._internal_batch_size = self._resolve_internal_batch_size(
            batch_size=batch_size,
            write_to_disk=write_to_disk,
            total_sources=total_sources,
            max_workers=max_workers,
        )

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
        self._worker_source_counts = {}
        self._worker_reported_totals = {}
        self._worker_shortfalls = []
        self._worker_errors = {}
        self._worker_stdout_queues = {}
        self._worker_reader_threads = {}
        self._worker_info = {}

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

    def _session_log_path(self) -> Path:
        """Return the shared session log every process writes to."""
        return Path(self.config.output_dir) / "logs" / f"cutana_{self.config.session_timestamp}.log"

    def _worker_stderr_log_path(self, proc_id: str) -> Path:
        """Return the file the parent redirects a worker's raw stderr into."""
        return Path(self.config.output_dir) / "logs" / "subprocesses" / f"{proc_id}_stderr.log"

    def _worker_stdout_log_path(self, proc_id: str) -> Path:
        """Return the file a disk-mode worker's stdout is redirected into."""
        return Path(self.config.output_dir) / "logs" / "subprocesses" / f"{proc_id}_stdout.log"

    def _worker_log_hint(self, proc_id: Optional[str] = None) -> str:
        """Name the logs that hold a worker's traceback.

        A worker writes its fatal traceback twice: through loguru into the shared
        session log, and straight to stderr, which the parent redirects into
        ``<id>_stderr.log``. Both are named so the pointer is correct whichever
        one the user opens; without ``proc_id`` (aggregate failures spanning
        several workers) only the shared log can be named.

        Args:
            proc_id: Worker process identifier, when a single worker is implicated.

        Returns:
            Human-readable pointer to the relevant log file(s).
        """
        if proc_id is None:
            return str(self._session_log_path())
        return f"{self._session_log_path()} and {self._worker_stderr_log_path(proc_id)}"

    def _resolve_internal_batch_size(
        self,
        batch_size: int,
        write_to_disk: bool,
        total_sources: int,
        max_workers: int,
    ) -> int:
        """Choose how many sources one worker process handles.

        Every internal batch costs one ``python -m cutana.cutout_process`` spawn --
        interpreter start plus the whole import graph -- so at survey scale small
        batches make startup, not cutouts, the dominant cost. Four limits apply:

        - ``max_sources_per_process`` from the load balancer, the same ceiling the
          non-streaming path uses;
        - one batch per worker, so a job small enough to fit in a single batch does
          not run on one process while the rest of the pool idles;
        - one below ``DUPLICATE_CHECK_THRESHOLD``, because ``preprocess_catalogue``
          skips the duplicate-``SourceID`` check at or above it and streaming is
          documented to keep that check (#400);
        - for in-memory streaming only, what the cutouts of one batch are allowed to
          weigh (see :meth:`_max_in_memory_batch_sources`).

        Args:
            batch_size: Sources per user-facing batch, as passed to init_streaming.
            write_to_disk: True for zarr-per-worker output, False for in-memory.
            total_sources: Rows in the catalogue.
            max_workers: Maximum concurrent worker processes.

        Returns:
            Sources per internal batch, at least 1.
        """
        limits = {
            "max_sources_per_process": self.config.loadbalancer.max_sources_per_process,
            "per_worker_share": math.ceil(total_sources / max_workers),
            "duplicate_check": DUPLICATE_CHECK_THRESHOLD - 1,
        }
        if not write_to_disk:
            limits["in_memory_budget"] = self._max_in_memory_batch_sources(max_workers)

        internal_batch_size = min(limits.values())
        if write_to_disk:
            # A disk batch is one zarr archive and cannot be re-split, so it must not
            # drop below what the caller asked for. In-memory batches are cut from a
            # rolling buffer, so there the user batch size is independent of this.
            internal_batch_size = max(batch_size, internal_batch_size)

        # The in-memory budget is the one limit that can reach zero, on a config whose
        # cutouts are large relative to available memory. One source per batch is
        # miserably slow but correct; refusing to run at all would be worse.
        internal_batch_size = max(1, internal_batch_size)

        logger.info(
            f"Internal batch size: {internal_batch_size} (batch_size={batch_size}, "
            + ", ".join(f"{name}={value}" for name, value in limits.items())
            + ")"
        )
        return internal_batch_size

    def _max_in_memory_batch_sources(self, max_workers: int) -> int:
        """Sources per internal batch that in-memory streaming can afford to hold.

        In-memory streaming holds a whole internal batch of cutouts twice: the worker
        accumulates every sub-batch result before it starts streaming, and the parent
        keeps one whole batch per worker until that worker completes. The shared-memory
        pool bounds only the transfer chunk, not either of those, so the in-flight
        cutout bytes are ``copies * max_workers * internal_batch * bytes_per_cutout``.

        Args:
            max_workers: Maximum concurrent worker processes.

        Returns:
            Largest internal batch whose cutouts fit the in-memory share of the budget.
        """
        resolution = self.config.target_resolution
        height, width = (
            (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
        )
        n_channels = max(len(weights) for weights in self.config.channel_weights.values())
        bytes_per_cutout = height * width * n_channels * np.dtype(self.config.data_type).itemsize
        budget = self.config.loadbalancer.memory_limit_bytes * _IN_MEMORY_CUTOUT_BUDGET_FRACTION
        return int(budget // (_IN_MEMORY_BATCH_COPIES * max_workers * bytes_per_cutout))

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

        # _spawn_cutout_process records spawn errors instead of raising. Without
        # this check a failed spawn would drop the whole internal batch silently
        # *and* leak the pool slot (registered below but never released).
        if process_id not in self.active_processes:
            raise RuntimeError(
                f"Failed to spawn worker {process_id} for internal batch "
                f"{batch_index + 1}/{len(self._batch_ranges)} ({len(job_df)} sources). "
                "See the orchestrator log for the spawn error."
            )

        # Track pool-worker mapping
        self._pool_to_worker[pool_idx] = process_id
        self._worker_to_pool[process_id] = pool_idx
        self._worker_cutouts[process_id] = []
        self._worker_metadata[process_id] = []
        self._worker_source_counts[process_id] = len(job_df)

        if self._use_windows_stdout_thread_mode():
            # On Windows, select() does not support subprocess pipes.
            # Use a dedicated reader thread per worker and drain its queue in _poll_workers().
            self._start_worker_stdout_reader(process_id)

        # Record per-worker detail known at spawn time (issue #354); FITS-set
        # composition and per-stage timings are merged in on completion.
        spawn_time = time.time()
        self._worker_info[process_id] = WorkerInfo(
            process_id=process_id,
            batch_index=batch_index,
            n_sources=len(job_df),
            pool_slot=pool_idx,
            start_time=spawn_time,
        )

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
                        # A non-zero exit means the worker hit a fatal error and
                        # its cutouts never arrived. Continuing here is what made
                        # worker failures invisible until the run died at the tail
                        # with an unrelated message. Read the cause off the pipe
                        # before _complete_worker() closes it.
                        message = self._worker_failure_message(
                            proc_id, exit_code, "its cutouts were not delivered"
                        )
                        self._discard_failed_worker(proc_id)
                        raise RuntimeError(message)
                    # _try_read_worker() takes one line per poll, so a worker that
                    # wrote 'complete' and exited in the same window still has it
                    # unread here. Completing without draining loses total_cutouts
                    # (the reported-vs-delivered check silently never runs) and
                    # batch_info (get_worker_info() drops its FITS-set composition
                    # and timings). The process has exited, so this hits EOF.
                    self._drain_worker_stdout(proc_id)
                    # A drained 'complete' line completes the worker itself, which
                    # releases it from active_processes. Only finish it here if the
                    # worker exited without sending one -- otherwise _record_delivery
                    # pops _worker_source_counts twice and raises KeyError.
                    if proc_id in self.active_processes:
                        self._complete_worker(proc_id)

        # Poll disk-mode workers via process.poll()
        for proc_id in disk_proc_ids:
            proc = self.active_processes.get(proc_id)
            if proc is None:
                continue
            exit_code = proc.poll()
            if exit_code is not None:
                if exit_code != 0:
                    # Same reasoning as the pipe branch: completing the worker would
                    # queue a result advertising a zarr_path the failed worker may
                    # never have written.
                    message = self._worker_failure_message(
                        proc_id, exit_code, "its batch was not written"
                    )
                    self._discard_failed_worker(proc_id)
                    raise RuntimeError(message)
                logger.info(f"Disk worker {proc_id} completed successfully")
                self._complete_worker(proc_id)

    def _drain_worker_stdout(self, proc_id: str) -> None:
        """Feed every buffered stdout line of an exited worker through the handler.

        Only safe once the worker has exited: iteration relies on hitting EOF
        rather than blocking.

        Args:
            proc_id: Worker process identifier.

        Returns:
            None.
        """
        lines = []
        if self._use_windows_stdout_thread_mode():
            line_queue = self._worker_stdout_queues.get(proc_id)
            if line_queue is not None:
                while True:
                    try:
                        line = line_queue.get_nowait()
                    except queue.Empty:
                        break
                    if line is None:
                        break
                    lines.append(line)
        else:
            proc = self.active_processes.get(proc_id)
            if proc is not None and proc.stdout is not None:
                lines = list(proc.stdout)

        for line in lines:
            self._handle_worker_line(proc_id, line)

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
            total = msg["total_cutouts"]
            logger.info(f"Worker {proc_id} completed: {total} cutouts")
            # Merge worker-reported FITS-set composition + per-stage timings into
            # the per-worker detail record (issue #354). The worker always sends a
            # batch_info payload and the record was created at spawn, so both are
            # accessed directly (an empty payload simply merges nothing).
            self._worker_info[proc_id].merge_completion(msg["batch_info"])
            self._worker_reported_totals[proc_id] = total
            self._complete_worker(proc_id)
        elif "error" in msg:
            # A failing worker prints this summary right before exiting non-zero.
            # It carries the real cause, so keep it for the exit-code failure to
            # quote rather than discarding it and citing a log file instead.
            self._worker_errors[proc_id] = msg["error"]
            logger.error(f"Worker {proc_id} reported failure: {msg['error']}")
        elif "processed_count" in msg:
            # Final summary a worker prints after 'complete'. Only reachable in the
            # Windows queue-drain path, which keeps handling buffered lines after
            # the worker was completed; this branch exists so it does not trip the
            # unrecognized-message raise below.
            logger.debug(
                f"Worker {proc_id} summary: {msg['processed_count']}/{msg['total_count']} sources"
            )
        else:
            raise RuntimeError(
                f"Unrecognized message from worker {proc_id}: {line.strip()[:200]}. "
                "Worker and orchestrator protocols are out of sync."
            )

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

        # Send ACK. A broken pipe here means the worker died between announcing
        # the chunk and us acknowledging it, so the rest of its cutouts are gone —
        # surface that instead of letting it escape as a bare BrokenPipeError.
        proc = self.active_processes[proc_id]
        try:
            proc.stdin.write("ACK\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            raise RuntimeError(
                f"Worker {proc_id} exited before its chunk could be acknowledged "
                f"({type(e).__name__}); remaining cutouts for this batch are lost."
            ) from e

    def _read_worker_stdout_summary(self, proc_id: str) -> Optional[Dict[str, Any]]:
        """Return the final JSON summary a disk-mode worker printed, if any.

        Disk-mode workers get no stdout pipe — the parent redirects their stdout
        to ``<id>_stdout.log`` — so the summary line carrying the processed count
        (or the failure message) is only reachable by reading that file back.

        Args:
            proc_id: Worker process identifier.

        Returns:
            The last summary dict the worker printed, or None if it never got
            far enough to print one.
        """
        stdout_log = self._worker_stdout_log_path(proc_id)
        if not stdout_log.exists():
            return None

        summary = None
        # This file is the worker's whole redirected stdout, not just its JSON
        # summary, so it can carry anything else the process printed. Pinning the
        # decode to UTF-8 keeps reading it independent of the parent's locale.
        for line in stdout_log.read_text(encoding="utf-8").splitlines():
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "processed_count" in msg:
                summary = msg
        return summary

    def _collect_worker_error(self, proc_id: str) -> Optional[str]:
        """Return the failure message a dead worker left behind, if it printed one.

        The worker prints its exception right before exiting non-zero, so that
        message is often still in flight when the exit code is noticed. Draining
        it here is what lets the failure quote the real cause instead of only
        naming a log file.

        Args:
            proc_id: Worker process identifier.

        Returns:
            The worker's error message, or None if it died without printing one.
        """
        if proc_id in self._worker_errors:
            return self._worker_errors.pop(proc_id)

        lines = []
        if self._use_windows_stdout_thread_mode():
            line_queue = self._worker_stdout_queues.get(proc_id)
            if line_queue is not None:
                while True:
                    try:
                        line = line_queue.get_nowait()
                    except queue.Empty:
                        break
                    if line is None:
                        break
                    lines.append(line)
        else:
            proc = self.active_processes.get(proc_id)
            if proc is not None and proc.stdout is not None:
                # The worker has exited, so iteration hits EOF instead of blocking.
                lines = list(proc.stdout)

        for line in lines:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "error" in msg:
                return msg["error"]

        summary = self._read_worker_stdout_summary(proc_id)
        if summary is not None and "error" in summary:
            return summary["error"]
        return None

    def _worker_failure_message(self, proc_id: str, exit_code: int, consequence: str) -> str:
        """Build the failure text for a worker that exited non-zero.

        Args:
            proc_id: Worker process identifier.
            exit_code: The worker's exit status.
            consequence: What the run lost because of the failure.

        Returns:
            Failure text quoting the worker's own error when it managed to
            report one, and always naming the logs holding the full traceback.
        """
        error = self._collect_worker_error(proc_id)
        cause = f" Worker error: {error}." if error else ""
        return (
            f"Worker {proc_id} failed with exit code {exit_code}; {consequence}.{cause} "
            f"Full traceback in {self._worker_log_hint(proc_id)}"
        )

    def _release_worker(self, proc_id: str) -> None:
        """Reap a finished worker and free everything it held.

        Args:
            proc_id: Worker process identifier.

        Returns:
            None.
        """
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

    def _discard_failed_worker(self, proc_id: str) -> None:
        """Reap a worker that died without delivering, queueing nothing.

        A failed worker's partial output must not reach the caller: its cutout
        list is short by an unknown amount and its zarr batch may never have been
        written. Freeing the slot without queueing a result is what keeps a caller
        that catches the failure and calls ``next_batch()`` again from being handed
        the wreckage.

        Args:
            proc_id: Worker process identifier.

        Returns:
            None.
        """
        # Without this a crashed worker reports no end_time, so get_worker_info()
        # shows it as still running — the first worker someone profiling a failed
        # run will look at.
        self._worker_info[proc_id].end_time = time.time()
        self._release_worker(proc_id)
        self._worker_cutouts.pop(proc_id, None)
        self._worker_metadata.pop(proc_id, None)
        self._worker_source_counts.pop(proc_id, None)
        self._worker_reported_totals.pop(proc_id, None)

    def _complete_worker(self, proc_id: str) -> None:
        """Finalize a successfully exited worker: assemble result, free pool slot."""
        # Stamp completion time on the worker record created at spawn (#354).
        self._worker_info[proc_id].end_time = time.time()
        self._release_worker(proc_id)

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

            # Disk workers report through their redirected stdout file rather than
            # a pipe, so their delivery count has to be read back from it. Without
            # this the accounting simply did not run in disk mode and every failure
            # claimed "no worker reported a shortfall".
            summary = self._read_worker_stdout_summary(proc_id)
            if summary is None:
                # Only reached for a worker that exited 0, which it can only do
                # after printing this summary. Absorbing it would drop the whole
                # batch out of the accounting unnoticed.
                raise RuntimeError(
                    f"Disk worker {proc_id} exited successfully but wrote no summary to "
                    f"{self._worker_stdout_log_path(proc_id)}; its batch cannot be "
                    "accounted for."
                )
            self._record_delivery(proc_id, summary["processed_count"])

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
            # Account before queueing: a transport mismatch raises, and the result
            # it would have queued describes a handover known to be incomplete.
            self._record_delivery(proc_id, n_cutouts)
            self._ready_results.append(result)
            logger.info(f"Worker {proc_id} finalized: {n_cutouts} cutouts")

    def _record_delivery(self, proc_id: str, n_delivered: int) -> None:
        """Compare what a finished worker delivered against what it was given.

        ``next_batch()`` sizes the run from the catalogue row count, so a worker
        that hands back fewer cutouts than it was assigned silently shortens the
        stream. Recording the discrepancy here is what lets the eventual failure
        point at the responsible workers instead of reporting an empty buffer.

        Args:
            proc_id: Worker process identifier.
            n_delivered: Number of cutouts the orchestrator actually received.

        Returns:
            None.
        """
        # _spawn_next_worker always records the assignment, so a missing entry is a
        # broken invariant — defaulting it would disable the accounting for exactly
        # the bookkeeping bug it exists to catch. A reported total, by contrast, is
        # genuinely absent in disk mode, where workers send no 'complete' message.
        assigned = self._worker_source_counts.pop(proc_id)
        reported = self._worker_reported_totals.pop(proc_id, None)

        if reported is not None and reported != n_delivered:
            # The worker announced N cutouts but we assembled a different number:
            # that is a transport bug, not an un-cuttable source.
            raise RuntimeError(
                f"Worker {proc_id} reported {reported} cutouts but only "
                f"{n_delivered} were transferred through the shared memory pool."
            )

        if n_delivered == assigned:
            return

        # Not necessarily a failure: extraction skips sources whose cutout window
        # falls outside the tile, so an edge source legitimately yields nothing.
        # A shortfall small enough to fit in the final batch never reaches
        # next_batch()'s end-of-stream check, so warn and expose it through
        # get_delivery_report() for the caller to judge.
        self._worker_shortfalls.append((proc_id, assigned, n_delivered))
        logger.warning(
            f"Worker {proc_id} delivered {n_delivered} cutouts for {assigned} assigned "
            f"sources ({assigned - n_delivered} not extractable or not delivered — a "
            f"source outside its tile produces no cutout). See "
            f"{self._worker_log_hint(proc_id)}"
        )

    def _shortfall_summary(self) -> str:
        """Render recorded per-worker shortfalls for inclusion in error messages."""
        if not self._worker_shortfalls:
            return (
                "No worker reported a shortfall, so the deficit appeared between "
                f"workers - inspect {self._worker_log_hint()}."
            )

        missing = sum(assigned - delivered for _, assigned, delivered in self._worker_shortfalls)
        details = ", ".join(
            f"{proc_id}: {delivered}/{assigned}"
            for proc_id, assigned, delivered in self._worker_shortfalls[:_MAX_SHORTFALLS_IN_MESSAGE]
        )
        if len(self._worker_shortfalls) > _MAX_SHORTFALLS_IN_MESSAGE:
            details += f", ... ({len(self._worker_shortfalls) - _MAX_SHORTFALLS_IN_MESSAGE} more)"
        return (
            f"{missing} cutouts were not extracted or not delivered across "
            f"{len(self._worker_shortfalls)} worker(s) — sources whose cutout window "
            f"falls outside their tile produce none. [delivered/assigned] {details}. "
            f"See {self._worker_log_hint()}"
        )

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
                        f"Streaming ended after {self._streaming_batch_index} of "
                        f"{self._user_batch_count} expected batches: all workers finished "
                        f"but produced fewer cutouts than the catalogue has sources. "
                        f"{self._shortfall_summary()}"
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
                    f"Streaming ended after {self._streaming_batch_index} of "
                    f"{self._user_batch_count} expected batches: all workers finished "
                    f"but no results are available. {self._shortfall_summary()}"
                )

            if time.time() - timeout_start > timeout_seconds:
                raise RuntimeError("Timeout waiting for batch completion")

            time.sleep(0.05)

    def get_batch_count(self) -> int:
        """Get total number of user-facing batches."""
        if not self._streaming_initialized:
            raise RuntimeError("Streaming not initialized. Call init_streaming() first.")
        return self._user_batch_count

    def get_worker_info(self) -> Dict[str, WorkerInfo]:
        """Get per-worker detail keyed by process_id (issue #354).

        Each :class:`~cutana.profiling_types.WorkerInfo` combines spawn-time batch
        composition (``batch_index``, ``n_sources``, ``pool_slot``, ``start_time``)
        with completion-time fields (``end_time``, ``sources_per_fits_set``,
        ``performance`` — the per-stage wall/cpu/stall_time/read_bytes breakdown).
        The completion-time fields are ``None`` until the worker reports completion
        via the in-memory streaming path. This supersedes the old worker-event list
        (it carries start/end times and more), so it is the single profiling entry
        point.

        Returns:
            A mapping of process_id to that worker's ``WorkerInfo``.
        """
        return dict(self._worker_info)

    def get_delivery_report(self) -> Dict[str, Any]:
        """Report how many cutouts finished workers delivered against their assignment.

        A source whose cutout window falls outside its tile yields nothing, so a
        run can legitimately produce fewer cutouts than the catalogue has rows.
        Such a deficit is only fatal when it empties a whole batch; otherwise the
        run completes and this is where the caller sees what was lost and which
        worker lost it.

        Returns:
            Dictionary with:
            - 'missing': total cutouts assigned but never delivered so far
            - 'shortfalls': list of (process_id, assigned, delivered) tuples,
              one per worker that delivered fewer cutouts than it was assigned
        """
        return {
            "missing": sum(
                assigned - delivered for _, assigned, delivered in self._worker_shortfalls
            ),
            "shortfalls": list(self._worker_shortfalls),
        }

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
        self._worker_source_counts.clear()
        self._worker_reported_totals.clear()
        self._worker_errors.clear()
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
        self._worker_source_counts.pop(process_id, None)
        self._worker_reported_totals.pop(process_id, None)
        self._worker_errors.pop(process_id, None)
