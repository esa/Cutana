#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Streaming catalogue loading infrastructure for Cutana.

Provides memory-efficient catalogue loading for large catalogues (10M+ sources)
through a two-phase approach:
1. Index building: Stream through catalogue once building FITS-set-to-row-indices mapping
2. Batch reading: Read only specific rows on-demand using pyarrow's take()

This module maintains FITS set optimization (80-90% I/O reduction) while enabling
O(index_size) + O(batch_size) memory usage instead of O(catalogue_size).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from .catalogue_preprocessor import parse_fits_file_paths
from .constants import DEFAULT_CATALOGUE_CHUNK_SIZE


@dataclass
class CatalogueIndex:
    """
    Lightweight index for streaming catalogue access.

    Stores mapping from FITS file sets to row indices, enabling FITS-set-optimized
    batch creation without loading the full catalogue.

    Memory usage: ~100 bytes per source (row_idx + fits_set reference)
    For 10M sources: ~1GB index vs ~100GB for full catalogue
    """

    fits_set_to_row_indices: Dict[tuple, List[int]] = field(default_factory=dict)
    row_count: int = 0
    catalogue_path: str = ""

    @classmethod
    def build_from_path(
        cls,
        path: str,
        batch_size: int = 100000,
    ) -> "CatalogueIndex":
        """
        Build index by streaming through catalogue.

        Extracts only (row_index, fits_file_paths) to minimize memory usage.

        Args:
            path: Path to catalogue file (CSV or Parquet)
            batch_size: Number of rows to process per chunk

        Returns:
            CatalogueIndex with FITS set to row indices mapping
        """
        logger.info(f"Building catalogue index from {path} with batch_size={batch_size}")

        fits_set_to_rows: Dict[tuple, List[int]] = defaultdict(list)
        total_rows = 0

        path_obj = Path(path)
        suffix = path_obj.suffix.lower()

        if suffix == ".parquet":
            total_rows = cls._build_index_from_parquet(path, batch_size, fits_set_to_rows)
        elif suffix == ".csv":
            total_rows = cls._build_index_from_csv(path, batch_size, fits_set_to_rows)
        else:
            raise ValueError(f"Unsupported catalogue format: {suffix}")

        logger.info(f"Built index: {total_rows} rows, {len(fits_set_to_rows)} unique FITS sets")

        return cls(
            fits_set_to_row_indices=dict(fits_set_to_rows),
            row_count=total_rows,
            catalogue_path=path,
        )

    @staticmethod
    def _build_index_from_parquet(
        path: str,
        batch_size: int,
        fits_set_to_rows: Dict[tuple, List[int]],
    ) -> int:
        """Build index from parquet file using pyarrow iter_batches."""
        parquet_file = pq.ParquetFile(path)
        row_offset = 0

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["fits_file_paths"],
        ):
            df = batch.to_pandas()

            for local_idx, fits_paths_str in enumerate(df["fits_file_paths"]):
                row_idx = row_offset + local_idx
                fits_paths = parse_fits_file_paths(fits_paths_str)
                fits_set = tuple(fits_paths)
                fits_set_to_rows[fits_set].append(row_idx)

            row_offset += len(df)

            if row_offset % 500000 == 0:
                logger.info(f"Indexed {row_offset} rows...")

        return row_offset

    @staticmethod
    def _build_index_from_csv(
        path: str,
        batch_size: int,
        fits_set_to_rows: Dict[tuple, List[int]],
    ) -> int:
        """Build index from CSV file using pandas chunked reading."""
        row_offset = 0

        for chunk in pd.read_csv(path, chunksize=batch_size, usecols=["fits_file_paths"]):
            for local_idx, fits_paths_str in enumerate(chunk["fits_file_paths"]):
                row_idx = row_offset + local_idx
                fits_paths = parse_fits_file_paths(fits_paths_str)
                fits_set = tuple(fits_paths)
                fits_set_to_rows[fits_set].append(row_idx)

            row_offset += len(chunk)

            if row_offset % 500000 == 0:
                logger.info(f"Indexed {row_offset} rows...")

        return row_offset

    def get_optimized_batch_ranges(
        self,
        max_sources_per_batch: int,
        min_sources_per_batch: int = 500,
        max_fits_sets_per_batch: int = 50,
    ) -> List[List[int]]:
        """
        Generate optimized batch row ranges grouped by FITS sets.

        Groups sources by FITS file sets to maximize I/O efficiency
        using a greedy algorithm. The goal is to minimize FITS file loading
        by keeping sources using the same FITS files together.

        **Batching Strategy - Atomic Tile Sets:**
        - A tile (FITS set) is NEVER split across batches UNLESS it individually
          exceeds max_sources_per_batch.
        - Multiple tiles CAN be combined into the same batch if their combined
          total is <= max_sources_per_batch.
        - Only tiles that exceed max_sources_per_batch are split into multiple
          consecutive batches.

        This ensures that output files/folders contain complete tile sets,
        making downstream processing and organization more predictable.

        **Algorithm:**
        1. Sort FITS sets by size (largest first)
        2. For sets > max_sources_per_batch: Split into max-sized chunks (unavoidable)
        3. For sets <= max_sources_per_batch: Keep atomic, combine with others if room

        Args:
            max_sources_per_batch: Maximum sources per batch (hard limit for memory)
            min_sources_per_batch: Minimum sources before flushing (efficiency threshold)
            max_fits_sets_per_batch: Maximum FITS sets per batch (limits I/O complexity)

        Returns:
            List of row index lists, each representing a batch
        """
        # Step 1: Calculate source count for each FITS set
        weights = {fits_set: len(rows) for fits_set, rows in self.fits_set_to_row_indices.items()}

        # Step 2: Sort FITS sets by size (largest first for greedy allocation)
        sorted_fits_sets = sorted(weights.keys(), key=lambda s: weights[s], reverse=True)

        # Step 3: Separate into "must split" (oversized) and "keep atomic" sets
        # Oversized sets: Exceed max_sources_per_batch, MUST be split
        # Atomic sets: Can fit in a single batch, should NOT be split
        oversized_sets = [s for s in sorted_fits_sets if weights[s] > max_sources_per_batch]
        atomic_sets = [s for s in sorted_fits_sets if weights[s] <= max_sources_per_batch]

        batches: List[List[int]] = []
        assigned: Set[int] = set()

        # Step 4: Process oversized FITS sets first
        # These MUST be split because they exceed max_sources_per_batch.
        # Each oversized set is split into consecutive batches of max_sources_per_batch.
        for fits_set in oversized_sets:
            available = [
                idx for idx in self.fits_set_to_row_indices[fits_set] if idx not in assigned
            ]
            if not available:
                continue

            # Split into max_sources_per_batch sized chunks using index-based iteration
            # (avoid O(n²) list slicing)
            num_batches_created = 0
            for i in range(0, len(available), max_sources_per_batch):
                batch_rows = available[i : i + max_sources_per_batch]
                assigned.update(batch_rows)
                batches.append(batch_rows)
                num_batches_created += 1

            logger.debug(
                f"Split oversized FITS set ({weights[fits_set]} sources) into "
                f"{num_batches_created} batches"
            )

        # Step 5: Combine atomic FITS sets into batches
        # Each atomic set is kept whole - we only flush when adding the NEXT set
        # would exceed max_sources_per_batch or max_fits_sets_per_batch.
        current_batch: List[int] = []
        current_fits_sets_count = 0

        for fits_set in atomic_sets:
            set_rows = [
                idx for idx in self.fits_set_to_row_indices[fits_set] if idx not in assigned
            ]
            if not set_rows:
                continue

            set_size = len(set_rows)

            # Check if adding this set would exceed limits
            # If so, flush current batch FIRST (before adding this set)
            would_exceed_sources = len(current_batch) + set_size > max_sources_per_batch
            would_exceed_fits_sets = current_fits_sets_count + 1 > max_fits_sets_per_batch

            if current_batch and (would_exceed_sources or would_exceed_fits_sets):
                # Flush current batch before adding this set
                assigned.update(current_batch)
                batches.append(current_batch)
                current_batch = []
                current_fits_sets_count = 0

            # Add this atomic set to current batch (guaranteed to fit now)
            current_batch.extend(set_rows)
            current_fits_sets_count += 1

            # Optional: Flush if we've reached a good batch size (efficiency)
            # This creates more batches but ensures reasonable parallelism
            if len(current_batch) >= min_sources_per_batch:
                assigned.update(current_batch)
                batches.append(current_batch)
                current_batch = []
                current_fits_sets_count = 0

        # Step 6: Flush any remaining sources as final batch
        if current_batch:
            assigned.update(current_batch)
            batches.append(current_batch)

        logger.info(
            f"Created {len(batches)} optimized batches from {self.row_count} sources "
            f"({len(self.fits_set_to_row_indices)} FITS sets, "
            f"{len(oversized_sets)} oversized sets split)"
        )

        return batches

    def get_fits_set_statistics(self) -> Dict:
        """Return statistics about FITS set distribution."""
        sizes = [len(rows) for rows in self.fits_set_to_row_indices.values()]
        return {
            "total_sources": self.row_count,
            "unique_fits_sets": len(self.fits_set_to_row_indices),
            "avg_sources_per_set": sum(sizes) / len(sizes) if sizes else 0,
            "max_sources_per_set": max(sizes) if sizes else 0,
            "min_sources_per_set": min(sizes) if sizes else 0,
        }


class CatalogueBatchReader:
    """
    Reads specific row ranges from catalogues efficiently.

    For parquet: Uses pyarrow's take() for O(1) random access
    For CSV: Uses chunked reading with filtering (slower, recommends parquet)
    """

    def __init__(self, path: str):
        """
        Initialize batch reader.

        Args:
            path: Path to catalogue file
        """
        self.path = path
        self.suffix = Path(path).suffix.lower()

        if self.suffix == ".parquet":
            # Pre-load parquet table for efficient take() operations
            self._parquet_table = pq.read_table(path)
            logger.debug(f"Loaded parquet table with {self._parquet_table.num_rows} rows")
        elif self.suffix == ".csv":
            self._parquet_table = None
            logger.debug(f"CSV reader initialized for {path}")
        else:
            raise ValueError(f"Unsupported catalogue format: {self.suffix}")

    def read_rows(self, row_indices: List[int]) -> pd.DataFrame:
        """
        Read specific rows from catalogue.

        Args:
            row_indices: List of row indices to read (0-based)

        Returns:
            DataFrame containing only the requested rows
        """
        if not row_indices:
            return pd.DataFrame()

        if self.suffix == ".parquet":
            return self._read_rows_parquet(row_indices)
        else:
            return self._read_rows_csv(row_indices)

    def _read_rows_parquet(self, row_indices: List[int]) -> pd.DataFrame:
        """Read specific rows from parquet using pyarrow take()."""
        selected = self._parquet_table.take(row_indices)
        return selected.to_pandas()

    def _read_rows_csv(self, row_indices: List[int]) -> pd.DataFrame:
        """
        Read specific rows from CSV by streaming through chunks.

        Note: This is slower than parquet due to sequential scanning.
        For large catalogues, converting to parquet format is recommended.

        The row_indices are treated as positional indices (0, 1, 2...) representing
        the row's position in the CSV file, NOT the original DataFrame index.
        This matches the indices created by _build_index_from_csv.
        """
        result_rows = []

        # Stream through CSV in chunks, checking which chunks contain our target rows
        for chunk_idx, chunk in enumerate(
            pd.read_csv(self.path, chunksize=DEFAULT_CATALOGUE_CHUNK_SIZE)
        ):
            # Calculate the row range this chunk covers (positional indices)
            chunk_start_row = chunk_idx * DEFAULT_CATALOGUE_CHUNK_SIZE
            chunk_end_row = chunk_start_row + len(chunk)

            # Find which of our target row_indices fall within this chunk
            chunk_indices = [i for i in row_indices if chunk_start_row <= i < chunk_end_row]

            if chunk_indices:
                # Convert to local indices within this chunk
                local_indices = [i - chunk_start_row for i in chunk_indices]
                result_rows.append(chunk.iloc[local_indices])

            # Early exit if we've found all rows
            if len(result_rows) and sum(len(r) for r in result_rows) >= len(row_indices):
                break

        if result_rows:
            return pd.concat(result_rows, ignore_index=True)
        return pd.DataFrame()

    def close(self):
        """Release resources."""
        if self._parquet_table is not None:
            self._parquet_table = None


def estimate_catalogue_size(path: str) -> int:
    """
    Estimate number of rows in catalogue without loading it fully.

    For parquet files, returns exact row count from metadata.
    For CSV files, estimates by sampling first 10 rows to calculate average line size.

    Args:
        path: Path to catalogue file

    Returns:
        Row count (exact for parquet, estimated for CSV)

    Raises:
        ValueError: If file format is unsupported or CSV cannot be sampled
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".parquet":
        parquet_file = pq.ParquetFile(path)
        return parquet_file.metadata.num_rows
    elif suffix == ".csv":
        file_size = path_obj.stat().st_size
        # Read first few lines to estimate row size
        # Note: Row sizes vary significantly based on fits_file_paths length
        # (e.g., ~558 bytes for Euclid catalogues with 4 FITS paths per source)
        with open(path, "r") as f:
            _header = f.readline()  # Skip header
            sample_lines = [f.readline() for _ in range(10)]
            # Filter out empty lines
            sample_lines = [line for line in sample_lines if line.strip()]

        if not sample_lines:
            raise ValueError(
                f"Cannot estimate catalogue size: CSV file {path} has no data rows. "
                "Either the file is empty or could not be read."
            )

        avg_line_size = sum(len(line) for line in sample_lines) / len(sample_lines)
        estimated_rows = int(file_size / avg_line_size)
        return estimated_rows
    else:
        raise ValueError(f"Unsupported catalogue format: {suffix}")
