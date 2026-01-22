#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Streaming Orchestrator for Cutana - handles batch-by-batch streaming workflows.

This module provides StreamingOrchestrator, a specialized orchestrator for
processing large catalogues in batches with optional asynchronous preparation.

Key features:
- Synchronous mode (default): Each batch is prepared when next_batch() is called
- Asynchronous mode: Next batch is prepared in background while current batch is being used
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotmap import DotMap
from loguru import logger

from .catalogue_preprocessor import preprocess_catalogue
from .catalogue_streamer import CatalogueBatchReader, CatalogueIndex
from .orchestrator import Orchestrator


class StreamingOrchestrator(Orchestrator):
    """
    Orchestrator specialized for streaming batch-by-batch processing.

    Uses memory-efficient streaming: builds a lightweight index from the catalogue,
    then reads only the rows needed for each batch on-demand. Supports 10M+ sources.

    Supports two modes:
    - synchronised_loading=True (default): Batches prepared on-demand
    - synchronised_loading=False: Next batch prepared in background while user processes current

    Usage:
        orchestrator = StreamingOrchestrator(config)
        orchestrator.init_streaming(batch_size=10000, synchronised_loading=False)

        for i in range(orchestrator.get_batch_count()):
            result = orchestrator.next_batch()
            # Process result["cutouts"] or result["zarr_path"]
    """

    def __init__(self, config: DotMap, status_panel=None):
        """Initialize the streaming orchestrator."""
        super().__init__(config, status_panel)

        # Streaming state - initialized by init_streaming()
        self._streaming_initialized = False  # True after init_streaming() is called
        self._catalogue_index: Optional[CatalogueIndex] = None  # FITS set to row indices mapping
        self._batch_reader: Optional[CatalogueBatchReader] = None  # Reads specific rows on-demand
        # List of row index lists - each inner list contains the catalogue row indices for one batch
        # e.g., [[0,1,2], [3,4,5,6], [7,8]] means batch 0 has rows 0-2, batch 1 has rows 3-6, etc.
        self._batch_ranges: List[List[int]] = []
        self._streaming_batch_index = 0  # Current position in batch iteration (0-indexed)
        self._streaming_write_to_disk = True  # If True, write to file; if False, return in-memory
        self._synchronised_loading = True  # If True, blocking batch prep; if False, async prep

        # Async batch preparation state (only used when synchronised_loading=False)
        self._pending_batch_index: Optional[int] = (
            None  # Index of batch being prepared in background
        )
        self._pending_process_id: Optional[str] = None  # Process ID of background preparation

    def init_streaming(
        self,
        batch_size: int,
        write_to_disk: bool = True,
        synchronised_loading: bool = True,
    ) -> None:
        """
        Initialize streaming mode for batch-by-batch processing.

        Uses memory-efficient streaming: builds a lightweight index from the catalogue,
        then reads only the rows needed for each batch on-demand.

        Args:
            batch_size: Maximum number of sources per batch
            write_to_disk: If True, write batches to zarr; if False, return cutouts in memory
            synchronised_loading: If True, prepare batches on-demand (blocking).
                                  If False, prepare next batch in background while current is used.

        Raises:
            ValueError: If catalogue cannot be loaded or validated
        """
        catalogue_path = self.config.source_catalogue

        logger.info(
            f"Initializing streaming: batch_size={batch_size}, "
            f"write_to_disk={write_to_disk}, synchronised_loading={synchronised_loading}"
        )

        # Use parent's shared catalogue index and reader initialization
        self._catalogue_index, self._batch_reader = self._init_catalogue_index_and_reader(
            catalogue_path
        )
        total_sources = self._catalogue_index.row_count

        # Get optimized batch ranges from the index
        self._batch_ranges = self._catalogue_index.get_optimized_batch_ranges(
            max_sources_per_batch=batch_size,
            min_sources_per_batch=500,
            max_fits_sets_per_batch=50,
        )

        logger.info(f"Created {len(self._batch_ranges)} optimized batches")

        # Store streaming state
        self._streaming_batch_index = 0
        self._streaming_write_to_disk = write_to_disk
        self._synchronised_loading = synchronised_loading
        self._streaming_initialized = True

        # Reset async state
        self._pending_batch_index = None
        self._pending_process_id = None

        # Initialize job tracking
        self.job_tracker.start_job(total_sources)

        # If async mode, start preparing first batch immediately
        if not synchronised_loading and len(self._batch_ranges) > 0:
            self._start_batch_preparation(0)
            logger.info("Async mode: started preparing batch 1 in background")

        logger.info(
            f"Streaming initialized: {len(self._batch_ranges)} batches, {total_sources} sources"
        )

    def _get_batch_df(self, batch_index: int) -> Any:
        """
        Get the DataFrame for a specific batch by reading rows on-demand.

        Args:
            batch_index: Index of batch (0-indexed)

        Returns:
            DataFrame containing the batch sources
        """
        row_indices = self._batch_ranges[batch_index]
        batch_df = self._batch_reader.read_rows(row_indices)
        # Preprocess the batch
        batch_df = preprocess_catalogue(batch_df, self.config)
        return batch_df

    def _start_batch_preparation(self, batch_index: int) -> str:
        """
        Start preparing a batch in the background (spawns subprocess).

        Args:
            batch_index: Index of batch to prepare (0-indexed)

        Returns:
            Process ID of the spawned subprocess
        """
        if batch_index >= len(self._batch_ranges):
            raise RuntimeError(f"Batch index {batch_index} out of range")

        # Read the batch rows on-demand from the catalogue
        job_df = self._get_batch_df(batch_index)

        unique_id = str(uuid.uuid4())[:8]
        process_id = f"streaming_{batch_index:03d}_{unique_id}"

        logger.info(f"Starting preparation for batch {batch_index + 1} ({len(job_df)} sources)")

        self._spawn_cutout_process(process_id, job_df, write_to_disk=self._streaming_write_to_disk)

        self._pending_batch_index = batch_index
        self._pending_process_id = process_id

        return process_id

    def _wait_for_batch(self, batch_index: int, process_id: str) -> Dict[str, Any]:
        """
        Wait for a batch to complete and collect results.

        Args:
            batch_index: Index of the batch (0-indexed)
            process_id: Process ID to wait for

        Returns:
            Result dictionary with batch data
        """
        row_indices = self._batch_ranges[batch_index]
        timeout_seconds = self.config.max_workflow_time_seconds

        logger.info(f"Waiting for batch {batch_index + 1} to complete...")

        if not self._streaming_write_to_disk:
            # In-memory mode: receive via shared memory
            all_cutouts, all_metadata = self._receive_cutouts_from_shm(
                process_id, batch_index, timeout_seconds
            )

            result = {
                "batch_number": batch_index + 1,
                "cutouts": all_cutouts,
                "metadata": all_metadata,
            }

            num_cutouts = (
                len(all_cutouts) if all_cutouts is not None and len(all_cutouts) > 0 else 0
            )
            logger.info(f"Batch {batch_index + 1} ready: {num_cutouts} cutouts in memory")

        else:
            # Disk mode: wait for subprocess completion
            start_time = time.time()
            completed = False

            while not completed:
                if time.time() - start_time > timeout_seconds:
                    self._terminate_process(process_id)
                    raise RuntimeError(f"Batch {batch_index + 1} timed out")

                completed_processes = self._monitor_processes(timeout_seconds)

                for proc_info in completed_processes:
                    if proc_info["process_id"] == process_id:
                        if proc_info["successful"]:
                            completed = True
                        else:
                            raise RuntimeError(
                                f"Batch {batch_index + 1} failed: {proc_info.get('reason', 'unknown')}"
                            )
                        break

                if not completed:
                    time.sleep(0.5)

            # Build disk mode result
            output_dir = Path(self.config.output_dir)
            zarr_subfolder = f"batch_{process_id}"
            zarr_path = output_dir / zarr_subfolder / "images.zarr"

            # Read batch metadata on-demand
            batch_df = self._batch_reader.read_rows(row_indices)
            result = {
                "batch_number": batch_index + 1,
                "zarr_path": str(zarr_path),
                "metadata": batch_df.to_dict("records"),
            }

            logger.info(f"Batch {batch_index + 1} ready: written to {zarr_path}")

        return result

    def _terminate_process(self, process_id: str) -> None:
        """Terminate a process and clean up."""
        if process_id in self.active_processes:
            try:
                self.active_processes[process_id].terminate()
                self.active_processes[process_id].wait(timeout=10.0)
            except Exception as e:
                logger.error(f"Failed to terminate process {process_id}: {e}")
            finally:
                if process_id in self.active_processes:
                    del self.active_processes[process_id]

    def next_batch(self) -> Dict[str, Any]:
        """
        Get the next batch of cutouts.

        In synchronised mode: Prepares and returns the batch (blocking).
        In async mode: Returns pre-prepared batch, starts preparing next one.

        Returns:
            Dictionary with:
            - 'batch_number': 1-indexed batch number
            - 'cutouts': numpy array of cutouts (if write_to_disk=False)
            - 'zarr_path': path to zarr file (if write_to_disk=True)
            - 'metadata': list of source metadata dicts

        Raises:
            RuntimeError: If not initialized or no more batches
        """
        if not self._streaming_initialized:
            raise RuntimeError("Streaming not initialized. Call init_streaming() first.")

        if self._streaming_batch_index >= len(self._batch_ranges):
            raise RuntimeError("No more batches available.")

        batch_index = self._streaming_batch_index
        self._streaming_batch_index += 1

        # Determine if we need to prepare this batch or if it's already being prepared
        if self._pending_batch_index == batch_index and self._pending_process_id:
            # Batch is already being prepared in background (async mode)
            process_id = self._pending_process_id
            logger.info(f"Batch {batch_index + 1} was prepared in background")
        else:
            # Need to start preparation now (sync mode or first batch)
            process_id = self._start_batch_preparation(batch_index)

        # Wait for the batch to complete
        result = self._wait_for_batch(batch_index, process_id)

        # Clear pending state
        self._pending_batch_index = None
        self._pending_process_id = None

        # In async mode, start preparing next batch
        if not self._synchronised_loading:
            next_index = self._streaming_batch_index
            if next_index < len(self._batch_ranges):
                self._start_batch_preparation(next_index)
                logger.info(f"Async: started preparing batch {next_index + 1} in background")

        return result

    def get_batch_count(self) -> int:
        """
        Get total number of batches.

        Returns:
            Number of batches available

        Raises:
            RuntimeError: If streaming not initialized
        """
        if not self._streaming_initialized:
            raise RuntimeError("Streaming not initialized. Call init_streaming() first.")
        return len(self._batch_ranges)

    def get_batch(self, batch_index: int) -> Dict[str, Any]:
        """
        Get a specific batch by index (random access).

        Note: In async mode, this may not benefit from prefetching if accessed
        out of order.

        Args:
            batch_index: 0-indexed batch index

        Returns:
            Batch result dictionary

        Raises:
            RuntimeError: If not initialized
            IndexError: If batch_index out of range
        """
        if not self._streaming_initialized:
            raise RuntimeError("Streaming not initialized. Call init_streaming() first.")

        if batch_index < 0 or batch_index >= len(self._batch_ranges):
            raise IndexError(
                f"Batch index {batch_index} out of range (0-{len(self._batch_ranges) - 1})"
            )

        # Check if this batch is the pending one
        if self._pending_batch_index == batch_index and self._pending_process_id:
            process_id = self._pending_process_id
        else:
            # Cancel any pending batch preparation if different index
            if self._pending_process_id:
                logger.warning(
                    f"Random access to batch {batch_index + 1}, "
                    f"cancelling pending batch {self._pending_batch_index + 1}"
                )
                self._terminate_process(self._pending_process_id)
                self._pending_batch_index = None
                self._pending_process_id = None

            process_id = self._start_batch_preparation(batch_index)

        result = self._wait_for_batch(batch_index, process_id)

        # Clear pending state
        self._pending_batch_index = None
        self._pending_process_id = None

        return result

    def cleanup(self):
        """Clean up resources including any pending batch preparation."""
        # Terminate any pending batch
        if self._pending_process_id:
            logger.info(f"Cleaning up pending batch {self._pending_batch_index + 1}")
            self._terminate_process(self._pending_process_id)
            self._pending_batch_index = None
            self._pending_process_id = None

        # Close batch reader
        if self._batch_reader:
            self._batch_reader.close()
            self._batch_reader = None

        super().cleanup()
