#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Fast in-process cutout generation for small batches.

Provides create_cutouts_direct() for rapid cutout creation without subprocess
overhead. Reuses the same vectorized processing pipeline as the full orchestrator
but runs entirely in-process with mmap FITS loading.

Typical use cases:
- Preview generation (< 200 cutouts)
- Quick-look analysis (< 500 cutouts)
- Interactive exploration in notebooks

Requests usually carry few cutouts spread across many tiles, so the cost is
dominated by opening/reading tiles (I/O bound). Distinct tiles are processed on a
thread pool: FITS I/O releases the GIL, while a process pool's per-task pickling
of cutout arrays/WCS plus interpreter startup measured far slower for this shape.

For large catalogues (> 1000 sources), use StreamingOrchestrator instead.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from dotmap import DotMap
from loguru import logger

from .cutout_process_utils import _process_sources_batch_vectorized_with_fits_set
from .fits_dataset import (
    get_selected_band_names,
    load_fits_sets,
    prepare_fits_sets_and_sources,
    select_fits_set_bands,
)
from .system_monitor import SystemMonitor

# Conservative ceiling on auto-selected threads. The work is I/O bound (opening and
# reading tiles), so a handful of concurrent reads already hides the latency; beyond
# that, extra threads mostly oversubscribe a shared/networked filesystem (and add GIL
# contention) for diminishing returns. An explicit max_workers is NOT capped by this.
_MAX_AUTO_WORKERS = 8

# Per-FITS-set progress line, shared by the serial and threaded paths so the
# wording stays in sync.  Worded as a completion count ("3/8 done"), not a set
# index — in the threaded path the number is how many sets have finished, in
# completion order, not the identity of the set that just finished.
_FITS_SET_PROGRESS = "Processed {}/{} FITS sets"


def _resolve_worker_count(max_workers: Optional[int], n_sets: int) -> int:
    """Resolve the number of worker threads for ``n_sets`` FITS sets.

    Args:
        max_workers: Caller cap. None auto-selects min(n_sets, effective CPUs, cap)
            using the k8s/cgroup-aware CPU count; an explicit value is honoured but
            never exceeds the number of sets (extra threads would idle).
        n_sets: Number of distinct FITS file sets to process.

    Returns:
        Thread count >= 1; always 1 for a single set (no executor overhead).

    Raises:
        ValueError: If ``max_workers`` is provided and is < 1.
    """
    if n_sets <= 1:
        return 1
    if max_workers is None:
        # Probe the cgroup/k8s CPU limit only on the auto path that needs it — the
        # single-tile and explicit-count paths above must not pay for it.
        available_cpus = SystemMonitor().get_effective_cpu_count()
        return min(n_sets, max(1, available_cpus), _MAX_AUTO_WORKERS)
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    return min(max_workers, n_sets)


def _process_one_fits_set(
    fits_set: tuple,
    sources_for_set: List[Dict[str, Any]],
    fits_extensions: List[str],
    config: DotMap,
    band_names: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    """Load one FITS set, extract its cutouts, and release the files.

    Each set is loaded and closed in isolation so that peak memory scales with the
    number of *concurrently* processed tiles, not the total tile count. This is
    also the unit of work distributed across threads.

    Args:
        fits_set: Tuple of FITS file paths forming one set (one tile's channels).
        sources_for_set: Sources whose cutouts come from this FITS set.
        fits_extensions: FITS extensions to load.
        config: Cutana configuration DotMap (read-only; safe to share across threads).
        band_names: Bands to narrow the set to, or None to load it whole. Resolved
            once by the caller rather than per set inside every thread, and required
            rather than defaulted so omitting it cannot restore the unnarrowed load.

    Returns:
        List of batch result dicts for this set.

    Raises:
        ValueError: If ``band_names`` names bands that match no file in this set
            (raised by ``select_fits_set_bands``).
        RuntimeError: If none of the set's FITS files could be loaded. A required
            input being entirely unreadable is a broken invariant, not a recoverable
            state — fail hard rather than silently dropping the set's sources.
    """
    # Narrow the set to the configured bands before loading, the same way
    # FITSDataset does for the worker path. Without this the direct path loads every
    # file in the set, so a subset configuration (e.g. selected_extensions=["NIR-H",
    # "NIR-Y", "NIR-J"] against a 4-file Euclid set) builds a 4-channel tensor for
    # 3 weights, and combine_channels pairs them positionally — silently taking
    # VIS, NIR-H, NIR-Y instead of the requested bands. Kept in a separate local so
    # the RuntimeError below still names what the catalogue listed.
    wanted_files = select_fits_set_bands(fits_set, band_names)

    # load_fits_sets only ever returns paths from this set, so its result is already
    # exactly the data for the set — no further filtering needed.
    loaded_fits_data = load_fits_sets([wanted_files], fits_extensions)

    if not loaded_fits_data:
        raise RuntimeError(f"No FITS data could be loaded for set: {fits_set}")

    try:
        return _process_sources_batch_vectorized_with_fits_set(
            sources_for_set, loaded_fits_data, config, profiler=None
        )
    finally:
        # Always close FITS files. A close failure is best-effort cleanup (we may
        # be in an error path), but surface the root cause rather than silently
        # discarding it.
        for fits_path, (hdul, _) in loaded_fits_data.items():
            try:
                hdul.close()
            except Exception as close_error:
                logger.warning(f"Failed to close FITS file {fits_path}: {close_error}")


def create_cutouts_direct(
    catalogue_df: pd.DataFrame,
    config: DotMap,
    max_workers: Optional[int] = None,
    *,
    log_set_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Create cutouts directly in-process without subprocess overhead.

    This is the fast path for small batches (< ~1000 sources). It runs the same
    vectorized processing pipeline as the full orchestrator but avoids subprocess
    spawning, JSON/TOML serialization, shared memory IPC, and job tracking.

    FITS files are loaded with memory mapping for fast access. Each distinct FITS
    set (tile) is loaded and closed in isolation, so peak memory scales with the
    number of concurrently processed tiles rather than the whole request.

    Args:
        catalogue_df: DataFrame with columns: SourceID, RA, Dec,
            diameter_pixel (or diameter_arcsec), fits_file_paths.
        config: Cutana configuration DotMap (from get_default_config()).
        max_workers: Number of worker threads for processing tiles concurrently.
            None (default) auto-selects min(n_tiles, effective_cpus, 8) using the
            k8s/cgroup-aware CPU count. Pass 1 to force serial processing; a
            single-tile request always runs serially regardless.
        log_set_progress: When True (default) and the request spans more than one
            FITS set, log a per-set completion heartbeat as tiles finish. Callers
            that invoke this many times in a tight loop (e.g. once per catalogue)
            should pass False — the heartbeat is then pure noise and the caller's
            own per-batch logging is the right place for progress.

    Returns:
        List of batch result dicts (one per FITS set, in catalogue grouping order),
        each containing:
            - "cutouts": ndarray of shape (N_sources, H, W, N_channels)
            - "metadata": list of per-source metadata dicts
            - "wcs": list of WCS dicts per source
            - "channel_names": list of channel name strings

    Raises:
        ValueError: If catalogue_df is empty, max_workers < 1, or ``selected_extensions``
            names bands that match no file in one of the catalogue's FITS sets. The last
            is the one a misconfigured run actually hits: the orchestrator and streaming
            backends log it and skip that set, but here it reaches the caller.
        KeyError: If required columns are missing.
        RuntimeError: If no valid cutouts were generated.
    """
    if len(catalogue_df) == 0:
        raise ValueError("Empty catalogue provided")

    required_columns = {"SourceID", "RA", "Dec", "fits_file_paths"}
    missing = required_columns - set(catalogue_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    has_size = "diameter_pixel" in catalogue_df.columns or "diameter_arcsec" in catalogue_df.columns
    if not has_size:
        raise KeyError("Need either 'diameter_pixel' or 'diameter_arcsec' column")

    n_sources = len(catalogue_df)
    logger.info(f"Creating {n_sources} cutouts directly (in-process)")

    # Convert DataFrame to list of dicts
    source_batch = catalogue_df.to_dict("records")

    # Group sources by their FITS file sets
    fits_set_to_sources = prepare_fits_sets_and_sources(source_batch)
    n_sets = len(fits_set_to_sources)
    logger.info(f"Grouped into {n_sets} unique FITS file sets")

    # Determine FITS extensions to load
    fits_extensions = config.fits_extensions
    # Resolved once here, not per set inside each thread.
    band_names = get_selected_band_names(config)

    n_workers = _resolve_worker_count(max_workers, n_sets)

    # Process each FITS set through the vectorized pipeline. Results are collected
    # in FITS-set submission order so the output is independent of thread timing.
    # Log per-set progress as each tile finishes (a single tile can take a while
    # and was otherwise silent between the "Grouped into N sets" and "Generated N
    # cutouts" lines), unless the caller opts out or there's only one set to do.
    log_progress = log_set_progress and n_sets > 1
    all_results = []
    if n_workers == 1:
        for done, (fits_set, sources_for_set) in enumerate(fits_set_to_sources.items(), start=1):
            all_results.extend(
                _process_one_fits_set(
                    fits_set, sources_for_set, fits_extensions, config, band_names
                )
            )
            if log_progress:
                logger.info(_FITS_SET_PROGRESS.format(done, n_sets))
    else:
        logger.info(f"Processing {n_sets} FITS sets across {n_workers} threads")
        # Collect by submission index so output stays in catalogue grouping
        # order, but report progress via as_completed so the count reflects
        # real completions rather than blocking on the first (possibly slow) set.
        results_by_index: Dict[int, List[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_index = {
                executor.submit(
                    _process_one_fits_set,
                    fits_set,
                    sources_for_set,
                    fits_extensions,
                    config,
                    band_names,
                ): index
                for index, (fits_set, sources_for_set) in enumerate(fits_set_to_sources.items())
            }
            for done, future in enumerate(as_completed(future_to_index), start=1):
                results_by_index[future_to_index[future]] = future.result()
                if log_progress:
                    logger.info(_FITS_SET_PROGRESS.format(done, n_sets))
        for index in range(n_sets):
            all_results.extend(results_by_index[index])

    if not all_results:
        raise RuntimeError("No valid cutouts were generated")

    total_cutouts = sum(len(r["metadata"]) for r in all_results if "metadata" in r)
    logger.info(f"Generated {total_cutouts} cutouts directly")

    return all_results
