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
from typing import Any, Dict, List, Optional, Set, Tuple

from astropy.io import fits
from astropy.wcs import WCS
from dotmap import DotMap
from loguru import logger

from .catalogue_preprocessor import SELECTABLE_BAND_NAMES, extract_filter_name, extract_fits_sets
from .fits_paths import parse_fits_file_paths
from .fits_reader import load_fits_file
from .performance_profiler import ContextProfiler, PerformanceProfiler
from .profiling_types import Stage


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

    with ContextProfiler(profiler, Stage.FITS_LOADING):
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


#: Selections already reported as "not band selection", so the notice is emitted once per
#: process instead of once per call. `_load_missing_fits_files` calls this per sub-batch, so
#: a long run produced thousands of copies of a one-off configuration message -- in the same
#: stream operators are asked to search for "Skipping FITS set".
_BAND_SELECTION_OFF_REPORTED: Set[frozenset] = set()


def _log_band_selection_off(band_names: Set[str]) -> None:
    """Say once that the selection named no band, and is therefore narrowing nothing.

    Said at all because the silent version of this is a typo: ``selected_extensions =
    ["NIRH"]`` loads every file in every set, and for a one-file set nothing downstream
    notices. ``UNKNOWN`` is the expected, non-typo case and reads as such.
    """
    key = frozenset(band_names)
    if key in _BAND_SELECTION_OFF_REPORTED:
        return
    _BAND_SELECTION_OFF_REPORTED.add(key)
    if band_names == {"UNKNOWN"}:
        logger.info(
            "Band selection off: the catalogue's filenames carry no band this recogniser "
            f"knows ({sorted(SELECTABLE_BAND_NAMES)}), so every file in each set is loaded."
        )
    else:
        logger.info(
            f"Band selection off: selected_extensions {sorted(band_names)} names none of "
            f"the bands {sorted(SELECTABLE_BAND_NAMES)}, so every file in each set is "
            "loaded. Check the spelling if you meant to select bands."
        )


def get_selected_band_names(config: DotMap) -> Optional[Set[str]]:
    """Band names named by ``selected_extensions``, or None to load every file.

    Shared by the orchestrator's dataset loading and by ``create_cutouts_direct``:
    both have to narrow a FITS set the same way, and when they disagree the tensor
    ends up with more channels than ``channel_weights`` has entries, which
    ``combine_channels`` then applies positionally.

    A selection naming no band ``extract_filter_name`` can produce is not band
    selection, so narrowing is off. In practice that is the UI on non-Euclid data:
    it fills ``selected_extensions`` from ``analyse_source_catalogue``, whose ``name``
    is ``extract_filter_name``'s own output, so a file the recogniser cannot classify
    arrives as ``UNKNOWN`` — a label with no band behind it to narrow on. A Python
    caller can put anything there, HDU names included, and it is read the same way.

    Decided from the selection rather than from the match result, which would conflate
    "this names no band" with "this names a band the set does not have"; the second
    case would then load whatever the set happens to contain.

    Args:
        config: Cutana configuration DotMap.

    Returns:
        Set of band names (e.g. ``{"VIS"}``), or None when no filtering applies.
    """
    selected = config.selected_extensions
    if not selected:
        return None
    band_names = set()
    for ext in selected:
        if isinstance(ext, dict) and "name" in ext:
            band_names.add(ext["name"])
        elif isinstance(ext, str):
            band_names.add(ext)
    # "PRIMARY" as a band name means "use all files" (no band filtering)
    if not band_names or band_names == {"PRIMARY"}:
        return None
    if not band_names & SELECTABLE_BAND_NAMES:
        _log_band_selection_off(band_names)
        return None
    return band_names


def select_fits_set_bands(fits_set: tuple, band_names: Optional[Set[str]]) -> tuple:
    """Narrow a FITS set to the requested bands, preserving the catalogue order.

    Order is preserved because ``channel_weights`` is applied positionally against
    the loaded channels; reordering here would silently pair weights with the
    wrong bands.

    Whether the selection is band selection at all is decided once, by
    ``get_selected_band_names``; ``band_names`` is None when it is not.

    Args:
        fits_set: FITS file paths forming one set (one tile's channels).
        band_names: Bands to keep, from ``get_selected_band_names``, or None to keep
            the set unchanged.

    Returns:
        The filtered set, in catalogue order.

    Raises:
        ValueError: If the selection names bands but matches no file in the set.
            The configuration asks for data this set does not contain, and loading
            the set whole would hand ``combine_channels`` the wrong bands under the
            requested names. What a miss costs is the caller's decision: the direct
            path lets it out, while ``FITSDataset._load_missing_fits_files`` skips
            that set, having already refused a selection that misses every set.
    """
    if not band_names:
        return fits_set

    filter_names = [extract_filter_name(path) for path in fits_set]
    selected = tuple(path for path, name in zip(fits_set, filter_names) if name in band_names)
    if not selected:
        raise ValueError(
            f"selected_extensions names the band(s) {sorted(band_names)}, which match none "
            f"of the bands {filter_names} in the FITS set "
            f"[{', '.join(os.path.basename(path) for path in fits_set)}]. Use band names "
            f"from the catalogue's fits_file_paths, or 'PRIMARY' to disable band selection."
        )
    return selected


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
        self._require_a_satisfiable_band_selection()

    def _require_a_satisfiable_band_selection(self) -> None:
        """Refuse a band selection that no FITS set this process holds can satisfy.

        A single set the selection misses is heterogeneity, and
        ``_load_missing_fits_files`` skips it so the tiles around it survive. A
        selection that misses every set is a misspelled or inapplicable
        ``selected_extensions``: every set would be skipped and the process would
        finish having extracted nothing, which since #425 is a result the parent
        believes.

        Checked here because it is the only place that sees all the sets at once,
        and because failing before the first sub-batch means failing before
        anything has been written.

        Raises:
            ValueError: If band selection is on and matches no set.
        """
        band_names = get_selected_band_names(self.config)
        if not band_names or not self.fits_set_to_sources:
            return

        fits_sets = list(self.fits_set_to_sources)
        for fits_set in fits_sets:
            if any(extract_filter_name(path) in band_names for path in fits_set):
                return

        present = sorted({extract_filter_name(path) for fits_set in fits_sets for path in fits_set})
        raise ValueError(
            f"selected_extensions names the band(s) {sorted(band_names)}, which none of the "
            f"{len(fits_sets)} FITS set(s) this process holds carries -- the bands present are "
            f"{present}. Every set would be skipped, so this run would extract nothing. Check "
            f"selected_extensions against the band names in the catalogue's fits_file_paths, "
            f"or use 'PRIMARY' to disable band selection."
        )

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
        """Load FITS files that are not yet in the cache.

        When selected_extensions specifies specific bands (e.g. ["VIS"]),
        only FITS files matching those bands are loaded, skipping unneeded ones.

        A set the selection matches nothing in costs that set, not the batch: it
        loads no files, and ``_process_source_sub_batch`` skips it with an error
        the same way it always has for a set that could not be read. This is the
        behaviour the worker path had before band selection could raise at all,
        when it filtered per file and a set with no wanted band simply
        contributed none, and heterogeneous catalogues depend on it: one tile
        that lacks the selected bands must not cost the tiles around it. Letting
        the raise out would drop every source the worker holds, including
        sub-batches already written to zarr, whose incremental-write receipt is
        only issued on the success path.

        A selection that matches *nothing the worker holds* is a different thing,
        and ``initialize_from_sources`` refuses it before any sub-batch runs. It
        has to be decided there: this method is called once per sub-batch, and a
        sub-batch is one FITS set, so nothing here can tell a lone heterogeneous
        tile from a selection that fits no tile at all.
        """
        band_names = get_selected_band_names(self.config)
        skipped = 0
        files_to_load = []
        for fits_set in fits_sets:
            # Same narrowing as the direct path, so the two cannot drift apart; only
            # what a miss costs differs, for the reason in the docstring above.
            try:
                wanted = select_fits_set_bands(fits_set, band_names)
            except ValueError as e:
                logger.error(f"Skipping FITS set, no requested band present: {e}")
                continue
            skipped += len(fits_set) - len(wanted)
            for fits_path in wanted:
                if fits_path in self.fits_cache:
                    continue
                files_to_load.append(fits_path)

        if not files_to_load:
            return

        if skipped > 0:
            logger.info(
                f"Band filter active ({band_names}): loading {len(files_to_load)} files, "
                f"skipped {skipped} files for unneeded bands"
            )
        logger.info(f"Loading {len(files_to_load)} new FITS files into cache")

        # Report loading stage if tracker available
        if self.job_tracker and self.process_name:
            from .cutout_process import _report_stage  # noqa: PLC0415, I001  # lazy: avoid circular import (cutout_process imports fits_dataset)

            _report_stage(
                self.process_name, f"Loading {len(files_to_load)} FITS files", self.job_tracker
            )

        with ContextProfiler(self.profiler, Stage.FITS_LOADING):
            for idx, fits_path in enumerate(files_to_load):
                try:
                    # Report progress for each file if many files
                    if (
                        len(files_to_load) > 5
                        and self.job_tracker
                        and self.process_name
                        and idx % 5 == 0
                    ):
                        from .cutout_process import _report_stage  # noqa: PLC0415, I001  # lazy: avoid circular import (cutout_process imports fits_dataset)

                        _report_stage(
                            self.process_name,
                            f"Loading FITS file {idx + 1}/{len(files_to_load)}",
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
        """Extract cached FITS data for specific FITS sets.

        Only returns files that are in the cache (files filtered out by band
        selection during loading will not be present).
        """
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
