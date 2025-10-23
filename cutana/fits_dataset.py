#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
FITS Dataset management for Cutana - handles process-level FITS file caching.

This module provides the FITSDataset class that manages FITS file loading and caching
at the process level to avoid reloading the same files across sub-batches.
"""

import os
from typing import Dict, List, Any, Optional, Tuple, Set
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger
from dotmap import DotMap

from .fits_reader import load_fits_file
from .performance_profiler import PerformanceProfiler, ContextProfiler
from .catalogue_preprocessor import parse_fits_file_paths, extract_fits_sets


def load_fits_sets(
    fits_sets: List[tuple],
    fits_extensions: List[str],
    config: DotMap = None,
    profiler: Optional[PerformanceProfiler] = None,
) -> Dict[str, Tuple[fits.HDUList, Dict[str, WCS]]]:
    """
    Load FITS files for given FITS file sets.

    Args:
        fits_sets: List of FITS file set tuples
        fits_extensions: List of FITS extensions to load
        config: Configuration DotMap (unused, kept for compatibility)
        profiler: Optional performance profiler

    Returns:
        Dictionary mapping fits_path -> (hdul, wcs_dict)
    """
    loaded_fits_data = {}

    with ContextProfiler(profiler, "FitsLoading"):
        for fits_set in fits_sets:
            for fits_path in fits_set:
                if fits_path not in loaded_fits_data:
                    try:
                        hdul, wcs_dict = load_fits_file(fits_path, fits_extensions, is_preview=True)
                        loaded_fits_data[fits_path] = (hdul, wcs_dict)
                    except Exception as e:
                        logger.error(f"Failed to load FITS file {fits_path}: {e}")
                        continue

    return loaded_fits_data


def prepare_fits_sets_and_sources(
    source_batch: List[Dict[str, Any]],
) -> Dict[tuple, List[Dict[str, Any]]]:
    """
    Parse FITS paths and group sources by their FITS file sets.

    Now uses the extract_fits_sets function from catalogue_preprocessor for consistency.

    Args:
        source_batch: List of source dictionaries

    Returns:
        Dictionary mapping FITS file sets (as tuples) to lists of sources
    """
    fits_set_to_sources = {}

    for source_data in source_batch:
        source_id = source_data["SourceID"]
        try:
            # Parse FITS file paths using the standardized function
            fits_paths = parse_fits_file_paths(source_data["fits_file_paths"])

            if fits_paths:
                # Use extract_fits_sets to create consistent FITS set signatures
                fits_set_dict, _ = extract_fits_sets(fits_paths)

                # Get the FITS set tuple (there should be only one)
                for fits_set in fits_set_dict.keys():
                    # Group sources by their FITS file set
                    if fits_set not in fits_set_to_sources:
                        fits_set_to_sources[fits_set] = []
                    fits_set_to_sources[fits_set].append(source_data)
                    break  # Only process the first (and should be only) fits_set

        except Exception as e:
            logger.error(f"Error parsing FITS paths for source {source_id}: {e}")
            continue

    return fits_set_to_sources


class FITSDataset:
    """
    Manages process-level FITS file caching to avoid reloading same files across sub-batches.

    This class handles:
    - Process-level FITS caching
    - Loading only missing FITS files
    - Smart memory management to free unused files
    - Cleanup on completion
    """

    def __init__(
        self,
        config: DotMap,
        profiler: Optional[PerformanceProfiler] = None,
        job_tracker: Optional[Any] = None,
        process_name: Optional[str] = None,
    ):
        self.config = config
        self.profiler = profiler
        self.job_tracker = job_tracker
        self.process_name = process_name
        self.fits_cache = {}  # fits_path -> (hdul, wcs_dict)
        self.fits_set_to_sources = {}  # Will be set during initialization
        self.total_sources = 0  # Track total sources for loading strategy

    def initialize_from_sources(self, source_batch: List[Dict[str, Any]]) -> None:
        """
        Initialize the dataset by preparing FITS sets for all sources.

        Args:
            source_batch: List of all source dictionaries for the process
        """
        logger.info(f"Initializing FITSDataset for {len(source_batch)} sources")
        self.total_sources = len(source_batch)
        self.fits_set_to_sources = prepare_fits_sets_and_sources(source_batch)
        logger.info(f"Found {len(self.fits_set_to_sources)} unique FITS sets")

    def prepare_sub_batch(
        self, sub_batch: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[fits.HDUList, Dict[str, WCS]]]:
        """
        Prepare FITS data for a sub-batch, loading only missing files.

        Args:
            sub_batch: List of source dictionaries for this sub-batch

        Returns:
            Dictionary of FITS data needed for this sub-batch
        """
        # Determine which FITS sets are needed for this sub-batch
        needed_fits_sets = self._get_fits_sets_for_sub_batch(sub_batch)

        # Load missing FITS files
        self._load_missing_fits_files(needed_fits_sets)

        # Return relevant cached data
        return self._get_fits_data_for_sets(needed_fits_sets)

    def free_unused_after_sub_batch(
        self,
        current_sub_batch: List[Dict[str, Any]],
        remaining_sub_batches: List[List[Dict[str, Any]]],
    ) -> None:
        """
        Free FITS files that won't be needed in remaining sub-batches.

        Args:
            current_sub_batch: Current sub-batch that was just processed
            remaining_sub_batches: List of remaining sub-batches
        """
        if not remaining_sub_batches:
            return

        current_fits_sets = self._get_fits_sets_for_sub_batch(current_sub_batch)
        future_fits_sets = self._get_fits_sets_for_sub_batches(remaining_sub_batches)

        # Find files that can be freed
        files_to_free = []
        for fits_set in current_fits_sets:
            if fits_set not in future_fits_sets:
                for fits_path in fits_set:
                    if fits_path in self.fits_cache:
                        files_to_free.append(fits_path)

        # Free the files
        for fits_path in files_to_free:
            self._free_fits_file(fits_path)

    def cleanup(self) -> None:
        """
        Clean up all remaining FITS files in the cache.
        """
        if not self.fits_cache:
            return

        logger.debug(f"Cleaning up {len(self.fits_cache)} remaining FITS files")

        for fits_path in list(self.fits_cache.keys()):
            self._free_fits_file(fits_path)

        self.fits_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about the current cache state.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_files": len(self.fits_cache),
            "total_fits_sets": len(self.fits_set_to_sources),
        }

    def _get_fits_sets_for_sub_batch(self, sub_batch: List[Dict[str, Any]]) -> List[tuple]:
        """Get FITS sets needed for a specific sub-batch."""
        sub_batch_source_ids = {source["SourceID"] for source in sub_batch}
        needed_fits_sets = []

        for fits_set, sources_for_set in self.fits_set_to_sources.items():
            if any(source["SourceID"] in sub_batch_source_ids for source in sources_for_set):
                needed_fits_sets.append(fits_set)

        return needed_fits_sets

    def _get_fits_sets_for_sub_batches(self, sub_batches: List[List[Dict[str, Any]]]) -> Set[tuple]:
        """Get all FITS sets needed for multiple sub-batches."""
        all_source_ids = set()
        for sub_batch in sub_batches:
            all_source_ids.update(source["SourceID"] for source in sub_batch)

        needed_fits_sets = set()
        for fits_set, sources_for_set in self.fits_set_to_sources.items():
            if any(source["SourceID"] in all_source_ids for source in sources_for_set):
                needed_fits_sets.add(fits_set)

        return needed_fits_sets

    def _load_missing_fits_files(self, fits_sets: List[tuple]) -> None:
        """Load FITS files that are not yet in the cache."""
        files_to_load = []
        for fits_set in fits_sets:
            for fits_path in fits_set:
                if fits_path not in self.fits_cache:
                    files_to_load.append(fits_path)

        if not files_to_load:
            return

        logger.info(f"Loading {len(files_to_load)} new FITS files into cache")

        # Report loading stage if tracker available
        if self.job_tracker and self.process_name:
            from .cutout_process import _report_stage

            _report_stage(
                self.process_name, f"Loading {len(files_to_load)} FITS files", self.job_tracker
            )

        with ContextProfiler(self.profiler, "FitsLoading"):
            for idx, fits_path in enumerate(files_to_load):
                try:
                    # Report progress for each file if many files
                    if (
                        len(files_to_load) > 5
                        and self.job_tracker
                        and self.process_name
                        and idx % 5 == 0
                    ):
                        from .cutout_process import _report_stage

                        _report_stage(
                            self.process_name,
                            f"Loading FITS file {idx+1}/{len(files_to_load)}",
                            self.job_tracker,
                        )

                    # Determine loading strategy based on total sources
                    hdul, wcs_dict = load_fits_file(
                        fits_path,
                        self.config.fits_extensions,
                        n_sources=self.total_sources,
                        is_preview=False,
                    )
                    self.fits_cache[fits_path] = (hdul, wcs_dict)
                    logger.debug(f"Loaded: {os.path.basename(fits_path)}")
                except Exception as e:
                    logger.error(f"Failed to load FITS file {fits_path}: {e}")

    def _get_fits_data_for_sets(
        self, fits_sets: List[tuple]
    ) -> Dict[str, Tuple[fits.HDUList, Dict[str, WCS]]]:
        """Extract cached FITS data for specific FITS sets."""
        result = {}
        for fits_set in fits_sets:
            for fits_path in fits_set:
                if fits_path in self.fits_cache:
                    result[fits_path] = self.fits_cache[fits_path]
        return result

    def _free_fits_file(self, fits_path: str) -> None:
        """Free a specific FITS file from cache."""
        if fits_path not in self.fits_cache:
            return

        try:
            hdul, _ = self.fits_cache[fits_path]
            hdul.close()
        except Exception as e:
            logger.warning(f"Error closing FITS file {fits_path}: {e}")
        finally:
            # Always remove from cache even if close fails
            del self.fits_cache[fits_path]
            logger.debug(f"Freed: {os.path.basename(fits_path)}")
