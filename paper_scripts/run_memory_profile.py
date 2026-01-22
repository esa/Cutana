#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Memory profiling benchmark for Cutana paper.

Creates plots showing memory consumption over time for the 1 tile case:
- Astropy baseline with 4 threads (best baseline performance)
- Cutana with 1 worker
- Cutana with 4 workers

Uses memory_profiler for accurate memory tracking including child processes.

HPC Benchmarking Practices:
- Warm-up runs before measurements
- 0.5s sampling interval for accuracy
- Includes child processes
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import toml
from memory_profiler import memory_usage

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from cutana.get_default_config import get_default_config  # noqa: E402
from cutana.logging_config import setup_logging  # noqa: E402
from cutana.orchestrator import Orchestrator  # noqa: E402
from paper_scripts.astropy_baseline import process_catalogue_astropy  # noqa: E402
from paper_scripts.plots import create_memory_plot  # noqa: E402


def profile_astropy_memory(
    catalogue_df: pd.DataFrame, output_dir: Path, config: dict, threads: int = 1
) -> Tuple[List[float], List[float]]:
    """
    Profile memory usage of Astropy baseline with full pipeline.

    NOTE: Thread isolation is not perfect in memory profiling mode because numpy
    modules remain imported between runs. For accurate thread-limited performance
    benchmarks, use run_framework_comparison.py which uses subprocess isolation.
    Memory usage is less sensitive to thread counts anyway.

    Args:
        catalogue_df: Source catalogue DataFrame
        output_dir: Output directory for temporary files
        config: Configuration dictionary from benchmark_config.toml
        threads: Number of threads to use (1 or 4)

    Returns:
        Tuple of (memory_history, timestamps)
    """
    logger.info(f"Profiling Astropy baseline memory usage ({threads} threads, full pipeline)")

    # Load memory profiling config
    memory_config = config["memory_profile"]
    sampling_interval = memory_config["sampling_interval"]
    include_children = memory_config["include_children"]

    # Load astropy baseline config
    baseline_config = config["astropy_baseline"]

    result = None

    def wrapper():
        nonlocal result
        result = process_catalogue_astropy(
            catalogue_df,
            fits_extension="PRIMARY",
            target_resolution=baseline_config["target_resolution"],
            apply_flux_conversion=baseline_config["apply_flux_conversion"],
            interpolation=baseline_config["interpolation"],
            output_dir=output_dir / f"astropy_{threads}t_temp",
            zeropoint_keyword=baseline_config["zeropoint_keyword"],
            process_threads=threads,
        )
        return result

    # Monitor memory usage
    mem_usage = memory_usage(
        (wrapper, ()),
        interval=sampling_interval,
        timeout=None,
        include_children=include_children,
        max_usage=False,
        retval=False,
    )

    timestamps = [i * sampling_interval for i in range(len(mem_usage))]

    logger.info(
        f"Astropy ({threads}t) memory profile: peak={max(mem_usage):.1f}MB, avg={np.mean(mem_usage):.1f}MB"
    )

    return mem_usage, timestamps


def profile_cutana_memory(
    catalogue_df: pd.DataFrame, max_workers: int, output_dir: str, benchmark_config: dict
) -> Tuple[List[float], List[float]]:
    """
    Profile memory usage of Cutana.

    Args:
        catalogue_df: Source catalogue DataFrame
        max_workers: Number of worker processes
        output_dir: Output directory for results
        benchmark_config: Benchmark configuration dictionary from benchmark_config.toml

    Returns:
        Tuple of (memory_history, timestamps)
    """
    logger.info(f"Profiling Cutana memory usage with {max_workers} workers")

    # Load memory profiling config
    memory_config = benchmark_config["memory_profile"]
    sampling_interval = memory_config["sampling_interval"]
    include_children = memory_config["include_children"]

    # Load benchmark overrides for cutana
    cutana_overrides = benchmark_config["cutana"]

    result = None

    def wrapper():
        nonlocal result

        # Get default Cutana config
        cutana_config = get_default_config()

        # Override with benchmark-specific values
        cutana_config.max_workers = max_workers
        cutana_config.N_batch_cutout_process = cutana_overrides["N_batch_cutout_process"]
        cutana_config.output_format = cutana_overrides["output_format"]
        cutana_config.output_dir = output_dir
        cutana_config.target_resolution = cutana_overrides["target_resolution"]
        cutana_config.data_type = cutana_overrides["data_type"]
        cutana_config.normalisation_method = cutana_overrides["normalisation_method"]
        cutana_config.interpolation = cutana_overrides["interpolation"]
        cutana_config.apply_flux_conversion = cutana_overrides["apply_flux_conversion"]
        cutana_config.loadbalancer.max_sources_per_process = cutana_overrides[
            "max_sources_per_process"
        ]
        cutana_config.loadbalancer.skip_memory_calibration_wait = cutana_overrides[
            "skip_memory_calibration_wait"
        ]

        # Set log levels to INFO to maintain console output
        cutana_config.log_level = "INFO"
        cutana_config.console_log_level = "INFO"

        # Simple channel weights for single channel
        cutana_config.channel_weights = {"PRIMARY": [1.0]}
        cutana_config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]
        cutana_config.source_catalogue = "benchmark_catalogue"

        # Run benchmark
        orchestrator = Orchestrator(cutana_config)
        result = orchestrator.start_processing(catalogue_df)
        return result

    # Monitor memory usage including child processes
    mem_usage = memory_usage(
        (wrapper, ()),
        interval=sampling_interval,
        timeout=None,
        include_children=include_children,
        max_usage=False,
        retval=False,
    )

    timestamps = [i * sampling_interval for i in range(len(mem_usage))]

    logger.info(
        f"Cutana ({max_workers}w) memory profile: peak={max(mem_usage):.1f}MB, avg={np.mean(mem_usage):.1f}MB"
    )

    return mem_usage, timestamps


def save_memory_stats(
    astropy_4t_data: Tuple[List[float], List[float]],
    cutana_1w_data: Tuple[List[float], List[float]],
    cutana_4w_data: Tuple[List[float], List[float]],
    output_path: Path,
    catalogue_description: str = "1 Tile, 4 FITS, 50k Sources",
):
    """Save memory statistics to JSON and raw traces to CSV."""
    astropy_4t_mem, astropy_4t_time = astropy_4t_data
    cutana_1w_mem, cutana_1w_time = cutana_1w_data
    cutana_4w_mem, cutana_4w_time = cutana_4w_data

    stats = {
        "catalogue_description": catalogue_description,
        "astropy_4_threads": {
            "peak_memory_mb": max(astropy_4t_mem),
            "avg_memory_mb": np.mean(astropy_4t_mem),
            "peak_memory_gb": max(astropy_4t_mem) / 1024,
            "avg_memory_gb": np.mean(astropy_4t_mem) / 1024,
            "duration_seconds": max(astropy_4t_time),
        },
        "cutana_1_worker": {
            "peak_memory_mb": max(cutana_1w_mem),
            "avg_memory_mb": np.mean(cutana_1w_mem),
            "peak_memory_gb": max(cutana_1w_mem) / 1024,
            "avg_memory_gb": np.mean(cutana_1w_mem) / 1024,
            "duration_seconds": max(cutana_1w_time),
        },
        "cutana_4_workers": {
            "peak_memory_mb": max(cutana_4w_mem),
            "avg_memory_mb": np.mean(cutana_4w_mem),
            "peak_memory_gb": max(cutana_4w_mem) / 1024,
            "avg_memory_gb": np.mean(cutana_4w_mem) / 1024,
            "duration_seconds": max(cutana_4w_time),
        },
    }

    # Save statistics JSON
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved memory statistics to: {output_path}")

    # Save raw memory traces to CSV for plot recreation
    csv_path = output_path.parent / output_path.name.replace("_stats_", "_traces_").replace(
        ".json", ".csv"
    )

    # Pad shorter arrays with NaN to make them the same length
    max_len = max(len(astropy_4t_time), len(cutana_1w_time), len(cutana_4w_time))

    def pad_array(arr, target_len):
        """Pad array with NaN to target length."""
        if len(arr) < target_len:
            return list(arr) + [np.nan] * (target_len - len(arr))
        return arr

    traces_df = pd.DataFrame(
        {
            "astropy_4t_time_sec": pad_array(astropy_4t_time, max_len),
            "astropy_4t_memory_mb": pad_array(astropy_4t_mem, max_len),
            "cutana_1w_time_sec": pad_array(cutana_1w_time, max_len),
            "cutana_1w_memory_mb": pad_array(cutana_1w_mem, max_len),
            "cutana_4w_time_sec": pad_array(cutana_4w_time, max_len),
            "cutana_4w_memory_mb": pad_array(cutana_4w_mem, max_len),
        }
    )

    traces_df.to_csv(csv_path, index=False)
    logger.info(f"Saved raw memory traces to: {csv_path}")

    return stats


def main():
    """Main memory profiling execution."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Memory profiling: Track memory usage over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--size",
        choices=["small", "big"],
        default="small",
        help="Catalogue size to use (default: small)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode (same as normal for memory profiling)"
    )

    args = parser.parse_args()

    setup_logging(log_level="INFO", console_level="INFO")

    logger.info("Starting memory profiling benchmarks")
    logger.info(f"Catalogue size: {args.size}")

    # Load configuration
    config_path = Path(__file__).parent / "benchmark_config.toml"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    config = toml.load(config_path)
    logger.info("Loaded configuration from benchmark_config.toml")

    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    output_dir = results_dir / "memory_profile"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load appropriate catalogue
    if args.test:
        # Test mode: use 12k test catalogue
        catalogue_path = script_dir / "catalogues" / "test" / "12k-1tile-4channel.csv"
    else:
        # Full mode: use size-specific catalogues
        catalogues_dir = script_dir / "catalogues" / args.size
        if args.size == "small":
            catalogue_path = catalogues_dir / "50k-1tile-4channel.csv"
        else:  # big
            catalogue_path = catalogues_dir / "200k-8tile-1channel.csv"

    if not catalogue_path.exists():
        logger.error(f"Catalogue not found: {catalogue_path}")
        sys.exit(1)

    catalogue_df = pd.read_csv(catalogue_path)
    logger.info(f"Loaded catalogue with {len(catalogue_df)} sources")

    # Determine catalogue description for plot titles
    if args.test:
        catalogue_description = "12k sources, 1 tile, 4 FITS"
    else:
        if args.size == "small":
            catalogue_description = "50k sources, 1 tile, 4 FITS"
        else:  # big
            catalogue_description = "200k sources, 8 tiles, 1 FITS"

    logger.info(f"Catalogue: {catalogue_description}")

    try:
        # Profile Astropy baseline with 4 threads (best baseline)
        logger.info("\nProfiling Astropy baseline (4 threads)...")
        astropy_4t_data = profile_astropy_memory(catalogue_df, output_dir, config, threads=4)

        # Profile Cutana 1 worker
        logger.info("\nProfiling Cutana with 1 worker...")
        cutana_1w_output = output_dir / "cutana_1worker"
        cutana_1w_data = profile_cutana_memory(catalogue_df, 1, str(cutana_1w_output), config)

        # Profile Cutana 4 workers
        logger.info("\nProfiling Cutana with 4 workers...")
        cutana_4w_output = output_dir / "cutana_4workers"
        cutana_4w_data = profile_cutana_memory(catalogue_df, 4, str(cutana_4w_output), config)

        # Create plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        figures_dir = script_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_path = figures_dir / f"memory_profile_{timestamp}.png"
        create_memory_plot(
            astropy_4t_data, cutana_1w_data, cutana_4w_data, plot_path, catalogue_description
        )

        # Save statistics
        stats_path = results_dir / f"memory_profile_stats_{timestamp}.json"
        stats = save_memory_stats(
            astropy_4t_data, cutana_1w_data, cutana_4w_data, stats_path, catalogue_description
        )

        # Print summary
        logger.info("\nMemory Profiling Summary:")
        logger.info(
            f"Astropy 4t: peak={stats['astropy_4_threads']['peak_memory_gb']:.2f}GB, avg={stats['astropy_4_threads']['avg_memory_gb']:.2f}GB"
        )
        logger.info(
            f"Cutana 1w: peak={stats['cutana_1_worker']['peak_memory_gb']:.2f}GB, avg={stats['cutana_1_worker']['avg_memory_gb']:.2f}GB"
        )
        logger.info(
            f"Cutana 4w: peak={stats['cutana_4_workers']['peak_memory_gb']:.2f}GB, avg={stats['cutana_4_workers']['avg_memory_gb']:.2f}GB"
        )

        logger.info("\nMemory profiling completed successfully!")

    except Exception as e:
        logger.error(f"Memory profiling failed: {e}")
        logger.error("Exception details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
