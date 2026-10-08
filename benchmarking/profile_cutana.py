#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Single-command Cutana bottleneck profiler.

Runs the real Cutana streaming pipeline across a matrix of scenarios and produces an
interpretable, trustworthy bottleneck analysis: which stage limits throughput
(read -> extract -> resize -> normalise -> combine -> transfer) and *whether that
stage is CPU-bound or stalled off-CPU (mostly blocked on I/O)*.

The CPU-vs-stall split is **lazy-safe**: it never forces an eager FITS read. Each stage
records wall time, CPU time (``time.process_time``) and disk bytes
(``/proc/self/io`` ``read_bytes``) inside the worker via ``PerformanceProfiler``; the
orchestrator hands these back through ``get_worker_info()`` (issue #354), so
``stall_time = wall - cpu`` decomposes a stage's wall into real CPU work vs off-CPU
time (dominated by blocked-on-read here) without changing any read behaviour.

``read_bytes`` is block-device accounting: it is meaningful only on block-backed
storage. On NFS (the Datalabs store) the counter is still *readable* but always reads
0, so the disk-byte / MB-per-s figures collapse to 0 there -- indistinguishable from a
genuinely cache-warm run, not a clean ``n/a``. A true ``n/a`` is reported only when the
counter is entirely unavailable (non-Linux / no ``/proc/self/io``). The wall/cpu/stall
split, in contrast, works on any filesystem -- treat it as the primary signal on NFS.

Source data is taken from ready-made Cutana catalogues (parquet/CSV with columns
SourceID, RA, Dec, diameter_pixel, fits_file_paths). Every FITS path is checked for
existence on the running machine, so the profiler skips tiles whose data is absent.

Usage:
    python benchmarking/profile_cutana.py --quick
    python benchmarking/profile_cutana.py --full
    python benchmarking/profile_cutana.py --scenarios vis1_dense_cold,visnir3_dense_warm
    python benchmarking/profile_cutana.py --list
    python benchmarking/profile_cutana.py --deep-profile py-spy --scenarios visnir3_dense_warm

Common options: --catalogue PATH --out DIR --repeats N --target-cutouts N --workers N
                --resolution PX
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from loguru import logger

from cutana import StreamingOrchestrator, get_default_config
from cutana.catalogue_preprocessor import extract_filter_name, parse_fits_file_paths
from cutana.performance_profiler import _read_process_io_bytes
from cutana.profiling_types import COMPUTE_STAGES, WorkerInfo

# Plotting is optional (matplotlib). create_profile_plots is None on a headless box or
# before the plot module exists, in which case the profiler still runs and writes JSON.
try:
    from benchmarking.profile_plots import create_profile_plots
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent))
        from profile_plots import create_profile_plots
    except ImportError:
        create_profile_plots = None

# Per-source pipeline stages emitted by PerformanceProfiler, in execution order.
# Sourced from the single canonical definition (cutana.profiling_types.COMPUTE_STAGES)
# so adding a stage in the pipeline automatically flows through to the profiler.
STAGE_ORDER = list(COMPUTE_STAGES)

# Band sets per scenario family, in canonical order. visnir3 = 4 input bands combined
# to 3 output channels. fits_file_paths, selected_extensions and channel_weights are
# all emitted in this order because combine_channels binds weights positionally.
BAND_SETS = {
    "vis1": ["VIS"],
    "visnir3": ["VIS", "NIR-H", "NIR-J", "NIR-Y"],
}

# Default source catalogues: ready-made Q1 search catalogues pointing at the local-XFS
# Euclid repository. Override with --catalogue (a parquet/CSV file or a directory).
DEFAULT_CATALOGUE = Path("/media/user/AnomalyMatch/tests/test_data/q1")

# Worker subprocess basenames to clear before a run (kill by PID, never `pkill -f`).
STRAY_PROCESS_MARKERS = ("cutout_process", "prediction_process")

# Rough seconds-per-thousand-cutouts used only to print an up-front ETA.
ETA_SECONDS_PER_KCUTOUT = 6.0

# How many sources to place on each tile in the sparse (many-tiles) regime.
SPARSE_SOURCES_PER_TILE = 200

_TILE_ID_RE = re.compile(r"TILE(\d+)")


@dataclass
class Scenario:
    """One profiling scenario: a point in bands x res x workers x density x cache."""

    bands: str  # "vis1" | "visnir3"
    resolution: int  # output cutout size in pixels
    workers: int  # max worker subprocesses
    density: str  # "dense" | "sparse"
    cache: str  # "cold" | "warm"

    @property
    def name(self) -> str:
        """Stable scenario identifier used for selection, filenames and table rows."""
        return f"{self.bands}_{self.density}_{self.cache}_r{self.resolution}_w{self.workers}"


@dataclass
class TileData:
    """A tile: its band FITS paths (canonical order, existence-checked) and source rows."""

    tile_id: str
    band_paths: List[str]
    sources: List[Dict[str, Any]]


# --------------------------------------------------------------------------------------
# System info, stray-process hygiene, cache-state probe
# --------------------------------------------------------------------------------------
def _cgroup_cpu_quota() -> Optional[float]:
    """Return the cgroup CPU quota in cores, or None if unconstrained/unavailable.

    Container CPU limits, not ``nproc``, bound parallel scaling, so this is recorded
    with every run. Reads cgroup v2 (``cpu.max``) then v1 quota/period.
    """
    try:
        v2 = Path("/sys/fs/cgroup/cpu.max")
        if v2.exists():
            quota, period = v2.read_text().split()
            if quota == "max":
                return None
            return int(quota) / int(period)
        quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_path.exists() and period_path.exists():
            quota = int(quota_path.read_text())
            period = int(period_path.read_text())
            if quota <= 0:
                return None
            return quota / period
    except (OSError, ValueError):
        return None
    return None


def gather_system_info(catalogue: Path) -> Dict[str, Any]:
    """Collect reproducibility context recorded with every run."""
    return {
        "hostname": os.uname().nodename,
        "nproc": os.cpu_count(),
        "cgroup_cpu_quota_cores": _cgroup_cpu_quota(),
        "total_memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "proc_self_io_available": Path("/proc/self/io").exists(),
        "catalogue": str(catalogue),
        "python": sys.version.split()[0],
    }


def kill_stray_workers() -> List[int]:
    """Terminate leftover Cutana worker subprocesses by PID before measuring.

    A stray reader from an earlier aborted run can steal disk bandwidth and corrupt
    timings. We resolve PIDs via process iteration and signal them individually --
    never ``pkill -f <pattern>``, which also matches and kills this profiler's own
    shell. Our own PID and parent are always excluded.

    Returns:
        The list of PIDs that were sent SIGKILL.
    """
    killed: List[int] = []
    own = {os.getpid(), os.getppid()}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid in own:
                continue
            cmdline = " ".join(proc.info["cmdline"] or [])
            if any(marker in cmdline for marker in STRAY_PROCESS_MARKERS):
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    if killed:
        logger.warning(f"Killed {len(killed)} stray worker process(es): {killed}")
    return killed


def measure_disk_read_rate(paths: List[str], sample_mb: int = 64) -> float:
    """Coarse cache-state probe: MB/s of disk bytes fetched while reading a sample.

    Reads the first band file in 8 MB chunks until at least ``sample_mb`` has been
    consumed (so a ``sample_mb < 8`` still reads one 8 MB chunk), and divides the disk
    bytes observed via ``/proc/self/io`` by the elapsed time. This is a *cache-state
    indicator*, not a true bandwidth benchmark: only the bytes not already in the page
    cache count as disk bytes, so a fully cold sample reports roughly block-device
    bandwidth while a warm sample fetches ~0 disk bytes and so reports ~0 MB/s; a
    partially cached sample lands in between. On NFS ``read_bytes`` is always 0 (see
    :func:`cutana.performance_profiler._read_process_io_bytes`), so this also returns
    ~0 MB/s there regardless of real cache state -- it cannot probe cache on NFS. When
    the counter is entirely unavailable it falls back to bytes-read / time. Not part of
    the measured pipeline.

    Returns:
        MB/s computed from actual disk bytes fetched, or 0.0 if unmeasurable.
    """
    if not paths:
        return 0.0
    start_bytes = _read_process_io_bytes()
    start = time.perf_counter()
    read = 0
    try:
        with open(paths[0], "rb") as handle:
            while read < sample_mb * 1024 * 1024:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                read += len(chunk)
    except OSError:
        return 0.0
    elapsed = time.perf_counter() - start
    end_bytes = _read_process_io_bytes()
    if start_bytes is not None and end_bytes is not None:
        disk_bytes = max(0, end_bytes - start_bytes)
    else:
        disk_bytes = read
    if elapsed <= 0:
        return 0.0
    return (disk_bytes / 1024 / 1024) / elapsed


# --------------------------------------------------------------------------------------
# Catalogue loading (existence-checked) + tile selection
# --------------------------------------------------------------------------------------
def channel_weights_for(bands: str) -> Dict[str, List[float]]:
    """Channel-combination weights for a band set, keyed in canonical band order.

    vis1 is a single passthrough channel. visnir3 maps 4 input bands to 3 output
    channels: VIS, NIR-H, and the mean of NIR-J/NIR-Y. The exact weights do not
    change which pipeline stage dominates, only the channel-mix arithmetic.
    """
    if bands == "vis1":
        return {"VIS": [1.0]}
    if bands == "visnir3":
        return {
            "VIS": [1.0, 0.0, 0.0],
            "NIR-H": [0.0, 1.0, 0.0],
            "NIR-J": [0.0, 0.0, 0.5],
            "NIR-Y": [0.0, 0.0, 0.5],
        }
    raise ValueError(f"Unknown band set: {bands}")


def _catalogue_files(catalogue: Path) -> List[Path]:
    """Resolve a catalogue path (file or directory) to a list of parquet/CSV files."""
    if catalogue.is_dir():
        files = sorted(catalogue.glob("*.parquet")) + sorted(catalogue.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No parquet/CSV catalogues found in {catalogue}")
        return files
    if catalogue.is_file():
        return [catalogue]
    raise FileNotFoundError(f"Catalogue path not found: {catalogue}")


def _tile_id_from_path(path: str) -> str:
    """Extract the numeric TILE id from a Euclid mosaic filename (or the basename)."""
    match = _TILE_ID_RE.search(path)
    return match.group(1) if match else Path(path).stem


def load_tiles(catalogue: Path, bands: str) -> List[TileData]:
    """Load source tiles for a band set, checking every FITS path exists on this machine.

    Sources are grouped by their ``fits_file_paths`` (one group per tile), so the
    on-disk existence check runs once per tile rather than per source. For each tile
    the band FITS paths are reordered into the canonical band order; a tile is skipped
    (with a warning) if any required band is absent from the record or missing on disk.

    Returns:
        Discovered tiles in deterministic (tile_id) order.
    """
    canonical = BAND_SETS[bands]
    columns = ["SourceID", "RA", "Dec", "diameter_pixel", "fits_file_paths"]
    frames = []
    for file in _catalogue_files(catalogue):
        if file.suffix == ".parquet":
            frames.append(pd.read_parquet(file, columns=columns))
        else:
            frames.append(pd.read_csv(file, usecols=columns))
    catalogue_df = pd.concat(frames, ignore_index=True)

    tiles: List[TileData] = []
    skipped_missing = 0
    for fits_str, group in catalogue_df.groupby("fits_file_paths", sort=False):
        raw_paths = parse_fits_file_paths(fits_str, normalize=False)
        band_to_path = {extract_filter_name(p): p for p in raw_paths}

        ordered_paths = []
        usable = True
        for band in canonical:
            path = band_to_path.get(band)
            if path is None or not os.path.exists(path):
                usable = False
                break
            ordered_paths.append(path)
        if not usable:
            skipped_missing += 1
            continue

        tile_id = _tile_id_from_path(ordered_paths[0])
        sources = group[["SourceID", "RA", "Dec", "diameter_pixel"]].to_dict("records")
        tiles.append(TileData(tile_id=tile_id, band_paths=ordered_paths, sources=sources))

    tiles.sort(key=lambda t: t.tile_id)
    logger.info(
        f"Loaded {len(tiles)} tiles with full {bands} coverage from {catalogue} "
        f"({skipped_missing} tiles skipped for missing/absent paths)"
    )
    return tiles


def select_tiles(
    tiles: List[TileData],
    density: str,
    target_cutouts: int,
    cache: str,
    used_tiles: set,
) -> List[TileData]:
    """Choose tiles for a scenario.

    ``target_cutouts`` is a floor, not a hard cap: whole tiles are accumulated until
    their combined source count first reaches it (``write_catalogue`` later trims the
    written rows to exactly ``target_cutouts``). dense packs onto the fewest tiles
    (largest first); sparse spreads across many tiles. cold scenarios prefer tiles not
    yet read this run so the page cache is genuinely cold; warm scenarios are warmed
    separately. Selected tiles are added to ``used_tiles``.

    Returns:
        The chosen tiles (a subset, possibly reordered).
    """
    if cache == "cold":
        # Prefer never-read tiles, then fall back to the rest if too few remain.
        fresh = [t for t in tiles if t.tile_id not in used_tiles]
        rest = [t for t in tiles if t.tile_id in used_tiles]
        candidates = fresh + rest
    else:
        candidates = list(tiles)

    if density == "dense":
        candidates.sort(key=lambda t: len(t.sources), reverse=True)
    selected: List[TileData] = []
    accumulated = 0
    for tile in candidates:
        contribution = (
            len(tile.sources)
            if density == "dense"
            else min(SPARSE_SOURCES_PER_TILE, len(tile.sources))
        )
        selected.append(tile)
        accumulated += contribution
        if accumulated >= target_cutouts:
            break

    for tile in selected:
        used_tiles.add(tile.tile_id)
    return selected


def write_catalogue(
    selected: List[TileData], density: str, target_cutouts: int, out_csv: Path
) -> Tuple[int, int]:
    """Write a scenario catalogue CSV from selected tiles.

    Returns:
        (n_sources, n_tiles) actually written.

    Raises:
        RuntimeError: If no sources could be written.
    """
    rows: List[Dict[str, Any]] = []
    tiles_used = 0
    for tile in selected:
        # load_tiles only ever yields tiles that have sources, so per_tile is
        # always non-empty here.
        per_tile = tile.sources if density == "dense" else tile.sources[:SPARSE_SOURCES_PER_TILE]
        band_paths_str = str(tile.band_paths)
        for source in per_tile:
            rows.append(
                {
                    "SourceID": source["SourceID"],
                    "RA": source["RA"],
                    "Dec": source["Dec"],
                    "diameter_pixel": source["diameter_pixel"],
                    "fits_file_paths": band_paths_str,
                }
            )
        tiles_used += 1
        if len(rows) >= target_cutouts:
            break

    if not rows:
        raise RuntimeError("No sources selected for catalogue; check catalogue / band coverage")

    rows = rows[:target_cutouts]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(f"Wrote catalogue: {len(rows)} sources across {tiles_used} tile(s) -> {out_csv}")
    return len(rows), tiles_used


def warm_cache(selected: List[TileData]) -> None:
    """Prime the page cache by reading every band file of the selected tiles once.

    Raises:
        OSError: If any band file cannot be read. A warm scenario whose files did
            not actually load would silently measure a cold cache, so we fail hard
            rather than mislabel the result.
    """
    for tile in selected:
        for path in tile.band_paths:
            with open(path, "rb") as handle:
                while handle.read(32 * 1024 * 1024):
                    pass


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def _effective_cores() -> int:
    """Usable core count: the cgroup CPU quota if set, else the node core count.

    The cgroup quota is what actually bounds parallel work in a container, so thread
    budgeting must use it rather than ``os.cpu_count()`` (which sees the whole node).
    """
    quota = _cgroup_cpu_quota()
    if quota is not None and quota >= 1:
        return int(quota)
    return os.cpu_count() or 1


def auto_process_threads(workers: int) -> int:
    """Threads per worker that keep total threads near the core budget (no oversubscribe).

    With W workers on C effective cores, ~C/W threads each keeps the machine busy
    without the workers fighting over cores.
    """
    return max(1, _effective_cores() // max(1, workers))


def build_config(
    scenario: Scenario,
    csv_path: Path,
    output_dir: Path,
    process_threads: int,
    normalisation: str,
):
    """Assemble a StreamingOrchestrator config for a scenario.

    The pipeline does representative work: per-cutout resize, ``normalisation``
    (default asinh, the production method), and -- for visnir3 -- a real 4-band ->
    3-channel combination via ``channel_weights``. This is what we profile to find
    the bottleneck and to measure future improvements against, not a match for the
    AnomalyMatch prediction config.

    process_threads caps each worker's BLAS/OpenMP thread pool. It is set explicitly
    because Cutana's default (effective_cores // 4) reads the *node* core count, not
    the cgroup quota, so on a high-core node with a small quota each worker would grab
    far too many threads and the workers oversubscribe the CPU.
    """
    config = get_default_config()
    config.output_format = "fits"
    config.target_resolution = scenario.resolution
    config.normalisation_method = normalisation
    config.selected_extensions = list(BAND_SETS[scenario.bands])
    config.channel_weights = channel_weights_for(scenario.bands)
    config.fits_extensions = ["PRIMARY"]
    config.apply_flux_conversion = True
    config.max_workers = scenario.workers
    config.process_threads = process_threads
    config.skip_memory_calibration_wait = True
    config.console_log_level = "WARNING"
    config.max_workflow_time_seconds = 1800
    config.source_catalogue = str(csv_path)
    config.output_dir = str(output_dir)
    return config


# --------------------------------------------------------------------------------------
# Per-scenario runner
# --------------------------------------------------------------------------------------
def _drain_batches(orchestrator: StreamingOrchestrator, n_batches: int) -> Tuple[int, List[float]]:
    """Drain ``n_batches`` batches, returning total cutouts and per-batch wall times.

    ``next_batch()`` always returns a dict containing ``cutouts`` in streaming mode
    (it raises rather than returning ``None``), so the result is read directly.
    """
    total = 0
    batch_times: List[float] = []
    for _ in range(n_batches):
        start = time.perf_counter()
        result = orchestrator.next_batch()
        batch_times.append(time.perf_counter() - start)
        total += len(result["cutouts"])
    return total, batch_times


def aggregate_worker_info(worker_info: Dict[str, WorkerInfo]) -> Dict[str, Any]:
    """Aggregate per-worker detail into per-stage CPU/stall/read_bytes + parallel efficiency.

    The parallel-efficiency figure is an *estimate*: the true definition needs the
    serial runtime, which we don't measure, so we approximate it with ``busy_total``
    (the sum of per-worker lifetimes). That sum is inflated by straggler workers, so
    the estimate is biased and only meant as a rough "are the workers actually
    overlapping" indicator -- not an accurate metric (#360 review).

    Returns:
        Dict with ``stages`` (per stage: wall, cpu, stall_time, read_bytes summed over
        workers), ``parallel_efficiency_estimate`` and ``n_workers``.
    """
    stages: Dict[str, Dict[str, Any]] = {
        stage: {
            "wall": 0.0,
            "cpu": 0.0,
            "stall_time": 0.0,
            "read_bytes": 0,
            "read_bytes_known": True,
        }
        for stage in STAGE_ORDER
    }
    busy_total = 0.0
    span_start: Optional[float] = None
    span_end: Optional[float] = None

    for info in worker_info.values():
        if info.start_time is not None and info.end_time is not None:
            busy_total += info.end_time - info.start_time
            span_start = info.start_time if span_start is None else min(span_start, info.start_time)
            span_end = info.end_time if span_end is None else max(span_end, info.end_time)

        perf_steps = (info.performance or {}).get("steps", {})
        for stage in STAGE_ORDER:
            step = perf_steps.get(stage)
            if not step:
                continue
            stages[stage]["wall"] += step["total_time"]
            stages[stage]["cpu"] += step["cpu_time"]
            stages[stage]["stall_time"] += step["stall_time"]
            read_bytes = step["read_bytes"]
            if read_bytes is None:
                stages[stage]["read_bytes_known"] = False
            else:
                stages[stage]["read_bytes"] += read_bytes

    n_workers = len(worker_info)
    if span_start is not None and span_end is not None and span_end > span_start and n_workers:
        # busy_total / (n_workers * span) -- a straggler-biased estimate, see docstring.
        parallel_efficiency_estimate = busy_total / (n_workers * (span_end - span_start))
    else:
        parallel_efficiency_estimate = 0.0

    # Fraction of each worker's lifetime spent in the profiled compute stages. The
    # remainder (1 - compute_fraction) is un-profiled time: cutout streaming/ACK
    # round-trips to the parent, plus spawn/idle. A low value points at the
    # transfer path rather than compute as the throughput limiter.
    total_stage_wall = sum(s["wall"] for s in stages.values())
    compute_fraction = total_stage_wall / busy_total if busy_total > 0 else 0.0

    return {
        "stages": stages,
        "parallel_efficiency_estimate": parallel_efficiency_estimate,
        "compute_fraction": compute_fraction,
        "n_workers": n_workers,
    }


def _summarise_stages(stages: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the dominant stage (by wall), classify CPU- vs stall-bound, and total disk MB/s."""
    dominant = None
    dominant_wall = 0.0
    total_stall = 0.0
    total_read_bytes = 0
    read_bytes_known = True
    for stage, vals in stages.items():
        if vals["wall"] > dominant_wall:
            dominant_wall = vals["wall"]
            dominant = stage
        total_stall += vals["stall_time"]
        if vals["read_bytes_known"]:
            total_read_bytes += vals["read_bytes"]
        else:
            read_bytes_known = False

    nature = "n/a"
    cpu_frac = None
    if dominant:
        dom = stages[dominant]
        cpu_frac = dom["cpu"] / dom["wall"] if dom["wall"] > 0 else 0.0
        nature = "cpu-bound" if dom["cpu"] >= dom["stall_time"] else "stall-bound"

    disk_mb_s = None
    if read_bytes_known and total_stall > 0:
        disk_mb_s = (total_read_bytes / 1024 / 1024) / total_stall

    return {
        "dominant_stage": dominant,
        "dominant_nature": nature,
        "dominant_cpu_fraction": cpu_frac,
        "total_read_mb": (total_read_bytes / 1024 / 1024) if read_bytes_known else None,
        "disk_mb_per_s": disk_mb_s,
    }


def run_scenario(
    scenario: Scenario,
    tiles: List[TileData],
    args: argparse.Namespace,
    used_tiles: set,
) -> Dict[str, Any]:
    """Run one scenario end-to-end with warmup, steady-state and repeats.

    Returns:
        A result dict with throughput (median/p95 over repeats), startup fraction,
        per-stage CPU/stall/read_bytes aggregation, parallel efficiency, the
        self-reported cache read rate, and the scenario/catalogue knobs.
    """
    logger.info(f"=== Scenario: {scenario.name} ===")
    kill_stray_workers()

    scenario_dir = Path(args.out) / scenario.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scenario_dir / "catalogue.csv"

    selected = select_tiles(
        tiles, scenario.density, args.target_cutouts, scenario.cache, used_tiles
    )
    n_sources, n_tiles = write_catalogue(selected, scenario.density, args.target_cutouts, csv_path)
    band_paths = sorted({p for t in selected for p in t.band_paths})

    if scenario.cache == "warm":
        warm_cache(selected)
    cache_read_rate = measure_disk_read_rate(band_paths)

    # Threads per worker: explicit --process-threads, else an oversubscription-safe auto.
    process_threads = args.process_threads or auto_process_threads(scenario.workers)

    repeats: List[Dict[str, Any]] = []
    for repeat_idx in range(args.repeats):
        kill_stray_workers()
        output_dir = scenario_dir / f"repeat_{repeat_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        config = build_config(scenario, csv_path, output_dir, process_threads, args.normalisation)

        # Batch size must be small enough to produce at least as many internal
        # batches as workers (else only a subset of workers ever spawns). Auto =
        # ~2x workers' worth of batches so all workers run and pipeline. An
        # explicit --batch-size overrides this.
        batch_size = args.batch_size or max(250, args.target_cutouts // (scenario.workers * 2))

        orchestrator = StreamingOrchestrator(config)
        try:
            # init_streaming's own max_workers/min_workers govern parallelism (it does
            # not read config.max_workers), so the scenario's worker count must be
            # passed here or every scenario would silently run with the default 4.
            # Pre-spawn all workers (min == max) so the steady-state measurement is not
            # contaminated by lazy spawn; the spawn cost still shows in startup_fraction.
            orchestrator.init_streaming(
                batch_size=batch_size,
                write_to_disk=False,
                max_workers=scenario.workers,
                min_workers=scenario.workers,
            )
            n_batches = orchestrator.get_batch_count()
            if n_batches == 0:
                logger.warning(f"{scenario.name}: no batches produced")
                continue

            # Workers run concurrently and batches pipeline, so per-batch next_batch()
            # times do NOT sum to wall time (a later batch is often already done when
            # polled). The honest aggregate throughput is total cutouts over the full
            # drain wall time; with enough cutouts, worker spawn is amortised. The
            # first-batch latency (which absorbs spawn) is reported separately as a
            # startup fraction so startup-dominated (too-small) runs are flagged.
            wall_start = time.perf_counter()
            total_cutouts, batch_times = _drain_batches(orchestrator, n_batches)
            wall_total = time.perf_counter() - wall_start

            worker_info = orchestrator.get_worker_info()
        finally:
            orchestrator.cleanup()

        # Every catalogue source must yield a cutout; a shortfall means the run is
        # not measuring what it claims to, so fail hard rather than report it.
        if total_cutouts < n_sources:
            raise RuntimeError(
                f"{scenario.name}: produced only {total_cutouts}/{n_sources} cutouts"
            )
        if not batch_times:
            raise RuntimeError(f"{scenario.name}: no batches were drained")

        # Draining at least one batch always takes measurable wall time, so divide
        # directly: a non-positive wall_total is a broken clock, not a state to paper
        # over with a 0.0 fallback.
        throughput = total_cutouts / wall_total
        first_batch_latency = batch_times[0]
        aggregation = aggregate_worker_info(worker_info)
        repeats.append(
            {
                "throughput": throughput,
                "total_cutouts": total_cutouts,
                "wall_total": wall_total,
                "first_batch_latency": first_batch_latency,
                "startup_fraction": first_batch_latency / wall_total,
                "aggregation": aggregation,
                "worker_spans": _compact_worker_spans(worker_info),
            }
        )
        logger.info(f"{scenario.name} repeat {repeat_idx}: {throughput:.1f} img/s")

    return _summarise_repeats(
        scenario, repeats, n_sources, n_tiles, cache_read_rate, process_threads, args.normalisation
    )


def _compact_worker_spans(worker_info: Dict[str, WorkerInfo]) -> List[Dict[str, Any]]:
    """Extract compact per-worker (start, end, n_sources) spans for the Gantt plot.

    Called after a full drain, when every worker has completed, so a missing
    ``end_time`` signals a worker that never finished -- fail hard rather than
    silently drop it from the Gantt.
    """
    spans = []
    for pid, info in worker_info.items():
        if info.end_time is None:
            raise RuntimeError(f"Worker {pid} has no end_time after drain (did not complete)")
        spans.append(
            {
                "pid": pid,
                "start": info.start_time,
                "end": info.end_time,
                "n_sources": info.n_sources,
            }
        )
    return spans


def _summarise_repeats(
    scenario: Scenario,
    repeats: List[Dict[str, Any]],
    n_sources: int,
    n_tiles: int,
    cache_read_rate: float,
    process_threads: int,
    normalisation: str,
) -> Dict[str, Any]:
    """Reduce repeats to median/p95 throughput and a representative stage breakdown."""
    base = {
        "scenario": asdict(scenario),
        "name": scenario.name,
        "n_sources": n_sources,
        "n_tiles": n_tiles,
        "process_threads": process_threads,
        "normalisation": normalisation,
        "cache_read_mb_per_s": cache_read_rate,
    }
    if not repeats:
        return {**base, "ok": False}

    throughputs = sorted(r["throughput"] for r in repeats)
    median_throughput = median(throughputs)
    # Representative repeat = the one whose throughput is closest to the median.
    representative = min(repeats, key=lambda r: abs(r["throughput"] - median_throughput))
    stage_summary = _summarise_stages(representative["aggregation"]["stages"])

    return {
        **base,
        "ok": True,
        "throughput_median": median_throughput,
        "throughput_p95": float(np.percentile(throughputs, 95)),
        "throughput_min": throughputs[0],
        "throughput_max": throughputs[-1],
        "startup_fraction": representative["startup_fraction"],
        "parallel_efficiency_estimate": representative["aggregation"][
            "parallel_efficiency_estimate"
        ],
        "compute_fraction": representative["aggregation"]["compute_fraction"],
        "n_workers_observed": representative["aggregation"]["n_workers"],
        "stages": representative["aggregation"]["stages"],
        "stage_summary": stage_summary,
        "worker_spans": representative.get("worker_spans", []),
        "repeats": [{k: v for k, v in r.items() if k != "aggregation"} for r in repeats],
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def print_ranked_table(results: List[Dict[str, Any]]) -> None:
    """Print a ranked bottleneck table to stdout (sorted by throughput, descending)."""
    ok_results = [r for r in results if r.get("ok")]
    ok_results.sort(key=lambda r: r["throughput_median"], reverse=True)

    header = (
        f"{'scenario':<36} {'img/s':>8} {'p95':>8} {'thr':>4} {'dominant':>16} "
        f"{'nature':>10} {'cpu%':>6} {'MB/s':>8} {'start%':>7} {'par-eff~':>8} {'compute%':>9}"
    )
    print("\n" + "=" * len(header))
    print("RANKED BOTTLENECK TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in ok_results:
        summ = r["stage_summary"]
        cpu_pct = (
            f"{summ['dominant_cpu_fraction'] * 100:.0f}"
            if summ["dominant_cpu_fraction"] is not None
            else "n/a"
        )
        mb_s = f"{summ['disk_mb_per_s']:.0f}" if summ["disk_mb_per_s"] is not None else "n/a"
        print(
            f"{r['name']:<36} {r['throughput_median']:>8.1f} {r['throughput_p95']:>8.1f} "
            f"{r.get('process_threads', 0):>4} {str(summ['dominant_stage']):>16} "
            f"{summ['dominant_nature']:>10} {cpu_pct:>6} "
            f"{mb_s:>8} {r['startup_fraction'] * 100:>6.0f}% {r['parallel_efficiency_estimate']:>8.2f} "
            f"{r.get('compute_fraction', 0) * 100:>8.0f}%"
        )
    failed = [r for r in results if not r.get("ok")]
    if failed:
        print("\nFailed scenarios:", ", ".join(r["name"] for r in failed))
    print("=" * len(header) + "\n")


def write_json(results: List[Dict[str, Any]], system_info: Dict[str, Any], path: Path) -> None:
    """Write the full machine-readable result set."""
    path.write_text(
        json.dumps({"system_info": system_info, "results": results}, indent=2, default=str)
    )
    logger.info(f"Wrote results JSON: {path}")


# --------------------------------------------------------------------------------------
# Scenario matrix + ETA
# --------------------------------------------------------------------------------------
def quick_scenarios(args: argparse.Namespace) -> List[Scenario]:
    """Four headline scenarios: {vis1, visnir3} x {cold, warm}, dense, at chosen res/workers."""
    return [
        Scenario(
            bands=b, resolution=args.resolution, workers=args.workers, density="dense", cache=c
        )
        for b in ("vis1", "visnir3")
        for c in ("cold", "warm")
    ]


def full_scenarios(args: argparse.Namespace) -> List[Scenario]:
    """The full matrix: bands x {64,180,256} x {1,4,8} x {dense,sparse} x {cold,warm}."""
    return [
        Scenario(bands, resolution, workers, density, cache)
        for bands in ("vis1", "visnir3")
        for resolution in (64, 180, 256)
        for workers in (1, 4, 8)
        for density in ("dense", "sparse")
        for cache in ("cold", "warm")
    ]


def parse_scenarios(spec: str, args: argparse.Namespace) -> List[Scenario]:
    """Parse a comma-separated scenario spec like ``vis1_dense_cold,visnir3_sparse_warm``."""
    scenarios = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        bands, density, cache = _parse_scenario_token(token)
        scenarios.append(Scenario(bands, args.resolution, args.workers, density, cache))
    return scenarios


def _parse_scenario_token(token: str) -> Tuple[str, str, str]:
    """Split and validate a ``<bands>_<density>_<cache>`` scenario token."""
    parts = token.split("_")
    if len(parts) != 3:
        raise ValueError(f"Bad scenario '{token}'; expected <bands>_<density>_<cache>")
    bands, density, cache = parts
    if bands not in BAND_SETS:
        raise ValueError(f"Bad bands '{bands}' in '{token}'; expected one of {list(BAND_SETS)}")
    if density not in ("dense", "sparse"):
        raise ValueError(f"Bad density '{density}' in '{token}'; expected dense|sparse")
    if cache not in ("cold", "warm"):
        raise ValueError(f"Bad cache '{cache}' in '{token}'; expected cold|warm")
    return bands, density, cache


def estimate_eta_seconds(scenarios: List[Scenario], args: argparse.Namespace) -> float:
    """Rough total ETA: per scenario ~ repeats x target_cutouts x per-kcutout constant."""
    per_scenario = args.repeats * (args.target_cutouts / 1000.0) * ETA_SECONDS_PER_KCUTOUT
    return len(scenarios) * (per_scenario + 5.0)


# --------------------------------------------------------------------------------------
# Deep profile (py-spy / scalene)
# --------------------------------------------------------------------------------------
def run_deep_profile(
    tool: str, scenario: Scenario, tiles: List[TileData], args: argparse.Namespace
) -> None:
    """Run a single scenario under py-spy or scalene for line/native/IO attribution.

    py-spy records the whole worker process tree (``--subprocesses``) into a
    flamegraph; scalene splits Python vs native vs system/IO time as the independent
    check on the stall-time attribution. Degrades with a clear message if the tool is
    missing.
    """
    if tool not in ("py-spy", "scalene"):
        raise ValueError(f"Unknown deep-profile tool '{tool}'; expected py-spy|scalene")
    if shutil.which(tool) is None:
        logger.error(
            f"Deep-profile tool '{tool}' not found on PATH. "
            f"Install via the [benchmark] extra (pip install cutana[benchmark]) and retry."
        )
        return

    scenario_dir = Path(args.out) / f"deep_{scenario.name}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scenario_dir / "catalogue.csv"
    selected = select_tiles(tiles, scenario.density, args.target_cutouts, scenario.cache, set())
    write_catalogue(selected, scenario.density, args.target_cutouts, csv_path)
    if scenario.cache == "warm":
        warm_cache(selected)

    process_threads = args.process_threads or auto_process_threads(scenario.workers)
    runner = scenario_dir / "_run_once.py"
    runner.write_text(
        _single_run_script(
            scenario, csv_path, scenario_dir / "deep_output", process_threads, args.normalisation
        )
    )

    if tool == "py-spy":
        out_svg = scenario_dir / "flamegraph.svg"
        cmd = [
            "py-spy",
            "record",
            "--subprocesses",
            "--format",
            "flamegraph",
            "--output",
            str(out_svg),
            "--",
            sys.executable,
            str(runner),
        ]
        logger.info(f"py-spy recording whole worker tree -> {out_svg}")
    else:
        cmd = [
            "scalene",
            "--cli",
            "--json",
            "--outfile",
            str(scenario_dir / "scalene.json"),
            str(runner),
        ]
        logger.info(f"scalene profiling -> {scenario_dir / 'scalene.json'}")

    kill_stray_workers()
    subprocess.run(cmd, check=False)
    logger.info(f"Deep profile complete: {scenario_dir}")


def _single_run_script(
    scenario: Scenario, csv_path: Path, output_dir: Path, process_threads: int, normalisation: str
) -> str:
    """Generate a tiny standalone script that streams one scenario once (for deep profilers)."""
    return (
        "from pathlib import Path\n"
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from cutana import StreamingOrchestrator\n"
        "from profile_cutana import build_config, Scenario\n"
        f"scenario = Scenario({scenario.bands!r}, {scenario.resolution}, {scenario.workers}, "
        f"{scenario.density!r}, {scenario.cache!r})\n"
        f"config = build_config(scenario, Path({str(csv_path)!r}), Path({str(output_dir)!r}), "
        f"{process_threads}, {normalisation!r})\n"
        "orch = StreamingOrchestrator(config)\n"
        "orch.init_streaming(batch_size=2500, write_to_disk=False)\n"
        "for _ in range(orch.get_batch_count()):\n"
        "    orch.next_batch()\n"
        "orch.cleanup()\n"
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface."""
    parser = argparse.ArgumentParser(description="Single-command Cutana bottleneck profiler")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="4 headline scenarios (~minutes)")
    mode.add_argument("--full", action="store_true", help="full scenario matrix (prints ETA)")
    mode.add_argument("--scenarios", type=str, help="comma-separated <bands>_<density>_<cache>")
    parser.add_argument(
        "--deep-profile", choices=["py-spy", "scalene"], help="deep-profile one scenario"
    )
    parser.add_argument("--list", action="store_true", help="list selected scenarios and exit")
    parser.add_argument("--dry-run", action="store_true", help="print plan + ETA, do not run")
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--out", type=Path, default=Path("benchmarking/results/profile"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-cutouts", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--process-threads",
        type=int,
        default=0,
        help="BLAS/OpenMP threads per worker; 0 = auto (effective_cores // workers)",
    )
    parser.add_argument(
        "--normalisation",
        type=str,
        default="asinh",
        choices=["asinh", "linear", "log", "zscale", "none"],
        help="per-cutout normalisation method (default asinh, the production method)",
    )
    parser.add_argument("--resolution", type=int, default=180)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="internal batch size; 0 = auto (~2x workers' worth of batches)",
    )
    return parser


def select_scenarios(args: argparse.Namespace) -> List[Scenario]:
    """Resolve the scenario list from the chosen mode (defaults to --quick)."""
    if args.full:
        return full_scenarios(args)
    if args.scenarios:
        return parse_scenarios(args.scenarios, args)
    return quick_scenarios(args)


def main(argv: Optional[List[str]] = None) -> int:
    """Profiler entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    args.out = Path(args.out)

    scenarios = select_scenarios(args)
    eta = estimate_eta_seconds(scenarios, args)

    print(f"\nCutana profiler: {len(scenarios)} scenario(s), catalogue={args.catalogue}")
    for scenario in scenarios:
        print(f"  - {scenario.name}")
    print(
        f"Estimated total runtime: ~{eta / 60:.1f} min "
        f"(repeats={args.repeats}, target_cutouts={args.target_cutouts})\n"
    )

    if args.list or args.dry_run:
        return 0

    if not args.catalogue.exists():
        logger.error(f"Catalogue path not found: {args.catalogue}")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    system_info = gather_system_info(args.catalogue)
    logger.info(f"System info: {system_info}")

    # Tile discovery is per band set; cache to avoid re-reading large catalogues.
    tiles_by_bands: Dict[str, List[TileData]] = {}

    def tiles_for(bands: str) -> List[TileData]:
        if bands not in tiles_by_bands:
            tiles_by_bands[bands] = load_tiles(args.catalogue, bands)
        return tiles_by_bands[bands]

    if args.deep_profile:
        scenario = scenarios[0]
        run_deep_profile(args.deep_profile, scenario, tiles_for(scenario.bands), args)
        return 0

    # Track tiles already read this run so cold scenarios can prefer fresh ones.
    used_tiles: set = set()
    results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        tiles = tiles_for(scenario.bands)
        if not tiles:
            logger.error(f"No tiles with {scenario.bands} coverage; skipping {scenario.name}")
            results.append({"name": scenario.name, "scenario": asdict(scenario), "ok": False})
            continue
        try:
            results.append(run_scenario(scenario, tiles, args, used_tiles))
        except Exception as exc:
            logger.exception(f"Scenario {scenario.name} failed: {exc}")
            results.append({"name": scenario.name, "scenario": asdict(scenario), "ok": False})

    print_ranked_table(results)
    write_json(results, system_info, args.out / "results.json")

    # create_profile_plots is None when matplotlib / the plot module is unavailable
    # (e.g. a headless box) so the core measurement still completes.
    if create_profile_plots is not None:
        try:
            create_profile_plots(results, args.out)
        except Exception as exc:
            logger.warning(f"Plot generation failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
