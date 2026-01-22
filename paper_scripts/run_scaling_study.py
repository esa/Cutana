#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Thread scaling study for Cutana paper.

Investigates Cutana scaling from 1 to 6 threads for the 4 tiles case
(4 tiles, 1 FITS per tile, ~12.5k sources per tile = 50k total sources).

Measures:
- Runtime vs number of workers
- Throughput (sources/second) vs number of workers
- Speedup factor relative to single-threaded
- Parallel efficiency

HPC Benchmarking Practices:
- Warm-up runs before measurements
- Cache warming for realistic performance
- Multiple worker counts tested
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import toml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from cutana.get_default_config import get_default_config  # noqa: E402
from cutana.logging_config import setup_logging  # noqa: E402
from cutana.orchestrator import Orchestrator  # noqa: E402
from paper_scripts.plots import create_scaling_plots  # noqa: E402


def warmup_fits_cache(catalogue_df: pd.DataFrame, warmup_size: int = 100):
    """
    Warm up filesystem cache by reading FITS headers.

    Ensures fair benchmarking by pre-loading FITS metadata into cache.
    Files are properly closed after reading to avoid memory buildup.

    Args:
        catalogue_df: Source catalogue DataFrame
        warmup_size: Number of sources to use for warmup
    """
    import ast

    from astropy.io import fits

    logger.info(f"Warming up FITS cache with {warmup_size} sources...")

    warmup_df = catalogue_df.head(warmup_size)
    fits_files_seen = set()
    total_sources = len(warmup_df)

    for idx, source in warmup_df.iterrows():
        # Progress indicator every 10 sources
        if (idx + 1) % 10 == 0 or (idx + 1) == total_sources:
            logger.info(f"  Cache warmup progress: {idx + 1}/{total_sources} sources")

        fits_paths_str = source["fits_file_paths"]
        if isinstance(fits_paths_str, str):
            fits_paths = ast.literal_eval(fits_paths_str)
        else:
            fits_paths = fits_paths_str

        # Handle both single and multiple FITS files
        if isinstance(fits_paths, list):
            paths_to_warm = fits_paths
        else:
            paths_to_warm = [fits_paths]

        for fits_path in paths_to_warm:
            if fits_path not in fits_files_seen:
                try:
                    # Open, read header, and immediately close
                    hdul = fits.open(fits_path, memmap=True, lazy_load_hdus=True)
                    _ = hdul[0].header  # Read header to warm cache
                    hdul.close()  # Explicitly close to free memory
                    fits_files_seen.add(fits_path)
                except Exception as e:
                    logger.warning(f"Cache warmup failed for {fits_path}: {e}")

    logger.info(f"Cache warmed: {len(fits_files_seen)} unique FITS files loaded")


def run_cutana_scaling_test(
    catalogue_df: pd.DataFrame, num_workers: int, output_dir: str, cutana_overrides: dict
) -> Dict[str, any]:
    """
    Run Cutana with specified number of workers.

    Args:
        catalogue_df: Source catalogue DataFrame
        num_workers: Number of worker processes
        output_dir: Output directory for results
        cutana_overrides: Benchmark overrides from benchmark_config.toml

    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running Cutana scaling test with {num_workers} workers")

    # Get default Cutana config
    config = get_default_config()

    # Override with benchmark-specific values
    config.max_workers = num_workers
    config.N_batch_cutout_process = cutana_overrides["N_batch_cutout_process"]
    config.output_format = cutana_overrides["output_format"]
    config.output_dir = output_dir
    config.target_resolution = cutana_overrides["target_resolution"]
    config.data_type = cutana_overrides["data_type"]
    config.normalisation_method = cutana_overrides["normalisation_method"]
    config.interpolation = cutana_overrides["interpolation"]
    config.apply_flux_conversion = cutana_overrides["apply_flux_conversion"]
    config.loadbalancer.max_sources_per_process = cutana_overrides["max_sources_per_process"]
    config.loadbalancer.skip_memory_calibration_wait = cutana_overrides[
        "skip_memory_calibration_wait"
    ]
    config.process_threads = cutana_overrides["process_threads"]

    # Set log levels to INFO to maintain console output
    # (Orchestrator will call setup_logging() again with these values)
    config.log_level = "INFO"
    config.console_log_level = "INFO"

    # Single channel for 4 tiles case
    config.channel_weights = {"PRIMARY": [1.0]}
    config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]
    config.source_catalogue = "scaling_benchmark"

    # Run benchmark
    start_time = time.time()
    orchestrator = Orchestrator(config)
    results = orchestrator.start_processing(catalogue_df)
    end_time = time.time()

    total_time = end_time - start_time
    sources_per_second = len(catalogue_df) / total_time if total_time > 0 else 0

    benchmark_results = {
        "num_workers": num_workers,
        "total_sources": len(catalogue_df),
        "total_time_seconds": total_time,
        "sources_per_second": sources_per_second,
        "workflow_status": (
            results.get("status", "unknown") if isinstance(results, dict) else "unknown"
        ),
    }

    logger.info(f"Cutana ({num_workers} workers):")
    logger.info(f"  Total time: {total_time:.2f} seconds")
    logger.info(f"  Sources per second: {sources_per_second:.2f}")

    return benchmark_results


def run_scaling_study(
    catalogue_df: pd.DataFrame,
    worker_counts: List[int],
    output_dir: Path,
    cutana_overrides: dict,
    warmup: bool = True,
) -> List[Dict[str, any]]:
    """
    Run scaling study across different worker counts.

    Args:
        catalogue_df: Source catalogue DataFrame
        worker_counts: List of worker counts to test
        output_dir: Output directory for results
        cutana_overrides: Benchmark overrides from benchmark_config.toml
        warmup: If True, warm up cache before first run

    Returns:
        List of benchmark results
    """
    all_results = []

    # Warm up cache once before all tests
    if warmup:
        logger.info("Performing one-time cache warmup before scaling tests")
        warmup_fits_cache(catalogue_df, warmup_size=min(100, len(catalogue_df)))

    for num_workers in worker_counts:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing with {num_workers} workers")
        logger.info(f"{'='*80}\n")

        # Create worker-specific output directory
        worker_output = output_dir / f"workers_{num_workers}"
        worker_output.mkdir(parents=True, exist_ok=True)

        try:
            result = run_cutana_scaling_test(
                catalogue_df, num_workers, str(worker_output), cutana_overrides
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Scaling test failed for {num_workers} workers: {e}")
            logger.error("Exception details:", exc_info=True)

    return all_results


def calculate_scaling_metrics(results: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Calculate scaling metrics from results.

    Args:
        results: List of benchmark results

    Returns:
        Dictionary with scaling metrics
    """
    # Sort by number of workers
    results_sorted = sorted(results, key=lambda x: x["num_workers"])

    # Get baseline (single worker) performance
    baseline = next((r for r in results_sorted if r["num_workers"] == 1), None)

    if not baseline:
        logger.warning("No single-worker baseline found")
        baseline_time = results_sorted[0]["total_time_seconds"]
    else:
        baseline_time = baseline["total_time_seconds"]

    # Calculate metrics for each worker count
    metrics = {
        "worker_counts": [],
        "runtimes": [],
        "throughputs": [],
        "speedups": [],
        "efficiencies": [],
    }

    for result in results_sorted:
        num_workers = result["num_workers"]
        runtime = result["total_time_seconds"]
        throughput = result["sources_per_second"]

        # Speedup = baseline_time / current_time
        speedup = baseline_time / runtime if runtime > 0 else 0

        # Efficiency = speedup / num_workers (ideal is 1.0)
        efficiency = speedup / num_workers if num_workers > 0 else 0

        metrics["worker_counts"].append(num_workers)
        metrics["runtimes"].append(runtime)
        metrics["throughputs"].append(throughput)
        metrics["speedups"].append(speedup)
        metrics["efficiencies"].append(efficiency)

    return metrics


def save_scaling_results(
    results: List[Dict[str, any]], metrics: Dict[str, any], output_dir: Path, timestamp: str
):
    """Save scaling results and metrics to JSON and CSV."""

    # Save raw results
    results_path = output_dir / f"scaling_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved scaling results to: {results_path}")

    # Save metrics
    metrics_path = output_dir / f"scaling_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved scaling metrics to: {metrics_path}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(
        {
            "Workers": metrics["worker_counts"],
            "Runtime (s)": metrics["runtimes"],
            "Throughput (sources/s)": metrics["throughputs"],
            "Speedup": metrics["speedups"],
            "Efficiency (%)": [e * 100 for e in metrics["efficiencies"]],
        }
    )

    summary_path = output_dir / f"scaling_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved scaling summary to: {summary_path}")

    return summary_df


def main():
    """Main scaling study execution."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Thread scaling study: Test Cutana performance across worker counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--size",
        choices=["small", "big"],
        default="small",
        help="Catalogue size to use (default: small)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: use 100k-1tile-4channel catalogue"
    )

    args = parser.parse_args()

    setup_logging(log_level="INFO", console_level="INFO")

    logger.info("Starting thread scaling study")
    logger.info(f"Mode: {'TEST' if args.test else 'FULL'}")
    logger.info(f"Catalogue size: {args.size}")

    # Load benchmark configuration
    config_path = Path(__file__).parent / "benchmark_config.toml"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    benchmark_config = toml.load(config_path)
    logger.info("Loaded configuration from benchmark_config.toml")

    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    output_dir = results_dir / "scaling_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load appropriate catalogue
    if args.test:
        # Test mode: use 12k test catalogue
        catalogue_path = script_dir / "catalogues" / "test" / "12k-1tile-4channel.csv"
    else:
        # Full mode: use size-specific catalogues
        catalogues_dir = script_dir / "catalogues" / args.size
        if args.size == "small":
            catalogue_path = catalogues_dir / "50k-4tiles-1channel.csv"
        else:  # big
            catalogue_path = catalogues_dir / "100k-4tiles-1channel.csv"

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
            catalogue_description = "50k sources, 4 tiles, 1 FITS"
        else:  # big
            catalogue_description = "100k sources, 4 tiles, 1 FITS"

    logger.info(f"Catalogue: {catalogue_description}")

    # Extract config sections
    scaling_config = benchmark_config["scaling_study"]
    worker_counts = scaling_config["worker_counts"]
    cutana_overrides = benchmark_config["cutana"]
    logger.info(f"Testing worker counts: {worker_counts}")

    try:
        # Run scaling study
        results = run_scaling_study(catalogue_df, worker_counts, output_dir, cutana_overrides)

        # Calculate scaling metrics
        metrics = calculate_scaling_metrics(results)

        # Add catalogue description to metrics
        metrics["catalogue_description"] = catalogue_description

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_df = save_scaling_results(results, metrics, results_dir, timestamp)

        # Create plots
        figures_dir = script_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        create_scaling_plots(metrics, figures_dir, timestamp, catalogue_description)

        # Print summary
        logger.info("\nScaling Study Summary:")
        logger.info("\n" + summary_df.to_string(index=False))

        logger.info("\nThread scaling study completed successfully!")

    except Exception as e:
        logger.error(f"Scaling study failed: {e}")
        logger.error("Exception details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
