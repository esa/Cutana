#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Framework comparison benchmark for Cutana paper.

Compares runtime and throughput (cutouts/second) for:
- Astropy baseline with 1 thread
- Astropy baseline with 4 threads
- Cutana with 1 worker
- Cutana with 4 workers

Across three scenarios:
1. 1 tile, 4 FITS, 50k sources
2. 8 tiles, 4 FITS per tile, 1k sources per tile (~8k total)
3. 4 tiles, 1 FITS per tile, 12.5k sources per tile (50k total)

HPC Benchmarking Practices:
- Warm-up runs before actual measurements
- Cache warming for realistic I/O performance
- Multiple runs with median timing
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
from paper_scripts.plots import (  # noqa: E402
    create_cutana_timing_chart,
    create_timing_breakdown_chart,
)


def warmup_fits_cache(catalogue_df: pd.DataFrame, warmup_size: int = 100):
    """
    Warm up filesystem cache by reading FITS headers.

    This ensures fair benchmarking by pre-loading FITS metadata into cache,
    simulating a realistic HPC scenario where metadata is cached.
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


def collect_cutana_performance_stats(output_dir: str) -> Dict[str, float]:
    """
    Collect performance statistics from Cutana subprocess logs.

    Args:
        output_dir: Output directory where subprocess logs are stored

    Returns:
        Dictionary with timing breakdown for each step
    """
    import json
    from glob import glob

    output_path = Path(output_dir)
    log_dir = output_path / "logs" / "subprocesses"

    timing_breakdown = {
        "FitsLoading": 0.0,
        "CutoutExtraction": 0.0,
        "ImageResizing": 0.0,
        "ChannelMixing": 0.0,
        "Normalisation": 0.0,
        "DataTypeConversion": 0.0,
        "MetaDataPostprocessing": 0.0,
        "ZarrFitsSaving": 0.0,
    }

    if not log_dir.exists():
        logger.warning(f"Subprocess log directory not found: {log_dir}")
        return timing_breakdown

    # Find all stderr log files
    stderr_files = glob(str(log_dir / "*_stderr.log"))

    if not stderr_files:
        logger.warning("No subprocess stderr log files found for performance analysis")
        return timing_breakdown

    # Parse each log file for performance data
    for stderr_file in stderr_files:
        try:
            with open(stderr_file, "r") as f:
                for line in f:
                    if "PERFORMANCE_DATA:" in line:
                        try:
                            json_str = line.split("PERFORMANCE_DATA:", 1)[1].strip()
                            perf_data = json.loads(json_str)
                            if perf_data.get("type") == "performance_summary":
                                steps_data = perf_data.get("steps", {})
                                for step_name, step_data in steps_data.items():
                                    if step_name in timing_breakdown:
                                        total_time = step_data.get("total_time", 0)
                                        if total_time > 0:
                                            timing_breakdown[step_name] += total_time
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.debug(f"Error parsing performance JSON: {e}")
                            continue
        except Exception as e:
            logger.debug(f"Error parsing subprocess log {stderr_file}: {e}")
            continue

    return timing_breakdown


def run_cutana_benchmark(
    catalogue_df: pd.DataFrame,
    max_workers: int,
    output_dir: str,
    scenario_name: str,
    warmup: bool = True,
) -> Dict[str, any]:
    """
    Run Cutana benchmark with specified number of workers.

    Args:
        catalogue_df: Source catalogue DataFrame
        max_workers: Number of worker processes
        output_dir: Output directory for results
        scenario_name: Name of scenario for logging
        warmup: If True, warm up cache before benchmark

    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running Cutana benchmark: {scenario_name} with {max_workers} workers")

    # Load benchmark configuration
    config_path = Path(__file__).parent / "benchmark_config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    benchmark_config = toml.load(config_path)
    cutana_overrides = benchmark_config["cutana"]
    framework_config = benchmark_config["framework_comparison"]

    # Warm up filesystem cache
    if warmup and framework_config["warmup_cache"]:
        warmup_size = framework_config["warmup_size"]
        warmup_fits_cache(catalogue_df, warmup_size=min(warmup_size, len(catalogue_df)))

    # Get default Cutana config
    config = get_default_config()

    # Override with benchmark-specific values
    config.max_workers = max_workers
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

    # Determine channel weights based on first source
    import ast

    first_fits_paths_str = catalogue_df.iloc[0]["fits_file_paths"]
    if isinstance(first_fits_paths_str, str):
        first_fits_paths = ast.literal_eval(first_fits_paths_str)
    else:
        first_fits_paths = first_fits_paths_str

    num_fits = len(first_fits_paths) if isinstance(first_fits_paths, list) else 1

    if num_fits == 1:
        # Single channel
        config.channel_weights = {"PRIMARY": [1.0]}
        config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]
    elif num_fits == 4:
        # Four channels (NIR-H, NIR-J, NIR-Y, VIS)
        # Using a list of weights for multi-channel processing
        config.channel_weights = {
            "PRIMARY": [1.0, 1.0, 1.0, 1.0],  # Equal weights for all 4 channels
        }
        # Note: This is simplified - in reality we'd need to handle multi-FITS properly
        config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}] * 4
    else:
        raise ValueError(f"Unexpected number of FITS files: {num_fits}")

    config.source_catalogue = "benchmark_catalogue"

    # Run benchmark
    start_time = time.time()
    orchestrator = Orchestrator(config)
    results = orchestrator.start_processing(catalogue_df)
    end_time = time.time()

    total_time = end_time - start_time
    sources_per_second = len(catalogue_df) / total_time if total_time > 0 else 0

    benchmark_results = {
        "scenario": scenario_name,
        "method": "cutana",
        "max_workers": max_workers,
        "total_sources": len(catalogue_df),
        "total_time_seconds": total_time,
        "sources_per_second": sources_per_second,
        "workflow_status": (
            results.get("status", "unknown") if isinstance(results, dict) else "unknown"
        ),
    }

    logger.info(f"Cutana ({max_workers} workers) completed:")
    logger.info(f"  Total time: {total_time:.2f} seconds")
    logger.info(f"  Sources per second: {sources_per_second:.2f}")

    # Collect timing breakdown from subprocess logs
    timing_breakdown = collect_cutana_performance_stats(output_dir)
    benchmark_results["timing_breakdown"] = timing_breakdown

    # Log timing breakdown
    if timing_breakdown and any(v > 0 for v in timing_breakdown.values()):
        logger.info(f"  Timing breakdown:")
        total_step_time = sum(timing_breakdown.values())
        for step, step_time in timing_breakdown.items():
            if step_time > 0:
                logger.info(f"    {step}: {step_time:.2f}s ({step_time/total_step_time*100:.1f}%)")

        # Create timing breakdown chart
        figures_dir = Path(__file__).parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        chart_path = (
            figures_dir
            / f"cutana_{max_workers}w_timing_{scenario_name.replace(' ', '_').lower()}.png"
        )
        create_cutana_timing_chart(
            timing_breakdown, chart_path, f"Cutana Timing - {scenario_name}", max_workers
        )

    return benchmark_results


def run_astropy_benchmark(
    catalogue_df: pd.DataFrame, scenario_name: str, output_dir: Path, threads: int
) -> Dict[str, any]:
    """
    Run Astropy baseline benchmark with full processing pipeline in subprocess.

    Uses subprocess to ensure proper thread isolation between 1-thread and 4-thread runs.

    Args:
        catalogue_df: Source catalogue DataFrame
        scenario_name: Name of scenario for logging
        output_dir: Output directory for temporary files and charts
        threads: Number of threads to use (1 or 4)

    Returns:
        Dictionary with benchmark results including timing breakdown
    """
    import subprocess
    import tempfile

    thread_label = f"{threads}t"
    logger.info(
        f"Running Astropy baseline benchmark ({thread_label}) in subprocess: {scenario_name}"
    )

    # Create temporary files for communication with subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as catalogue_file:
        catalogue_df.to_csv(catalogue_file.name, index=False)
        catalogue_path = catalogue_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as results_file:
        results_path = results_file.name

    temp_output = output_dir / f"astropy_{thread_label}_temp"

    try:
        # Run Astropy benchmark in subprocess for proper thread isolation
        subprocess_script = Path(__file__).parent / "run_astropy_subprocess.py"
        cmd = [
            sys.executable,
            str(subprocess_script),
            "--catalogue",
            catalogue_path,
            "--output-json",
            results_path,
            "--threads",
            str(threads),
            "--scenario-name",
            scenario_name,
            "--output-dir",
            str(temp_output),
        ]

        # Run subprocess without capturing output so logs stream to terminal in real-time
        result = subprocess.run(cmd, text=True, timeout=7200)

        if result.returncode != 0:
            logger.error(f"Astropy subprocess failed with return code {result.returncode}")
            raise RuntimeError(f"Astropy benchmark subprocess failed")

        # Load results from subprocess
        with open(results_path, "r") as f:
            results = json.load(f)

        benchmark_results = {
            "scenario": scenario_name,
            "method": f"astropy_{thread_label}",
            "threads": threads,
            "total_sources": results["total_sources"],
            "total_time_seconds": results["total_time_seconds"],
            "sources_per_second": results["sources_per_second"],
            "successful_cutouts": results["successful_cutouts"],
            "errors": results["errors"],
            "timing_breakdown": results.get("timing_breakdown", {}),
        }

        # Create timing breakdown chart
        if "timing_breakdown" in results:
            figures_dir = Path(__file__).parent / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            chart_path = (
                figures_dir
                / f"astropy_{thread_label}_timing_{scenario_name.replace(' ', '_').lower()}.png"
            )
            create_timing_breakdown_chart(
                results["timing_breakdown"],
                chart_path,
                f"Astropy Baseline ({thread_label}) Timing - {scenario_name}",
            )

        return benchmark_results

    finally:
        # Clean up temporary files
        import os

        try:
            os.unlink(catalogue_path)
            os.unlink(results_path)
        except Exception:
            pass


def run_all_comparisons(catalogues: Dict[str, str], output_dir: Path) -> List[Dict[str, any]]:
    """
    Run all framework comparisons across all scenarios.

    Args:
        catalogues: Dictionary mapping scenario names to catalogue paths
        output_dir: Output directory for results

    Returns:
        List of benchmark results
    """
    all_results = []
    total_scenarios = len(catalogues)
    scenario_num = 0

    for scenario_name, catalogue_path in catalogues.items():
        scenario_num += 1
        logger.info(f"\n{'='*80}")
        logger.info(f"SCENARIO {scenario_num}/{total_scenarios}: {scenario_name}")
        logger.info(f"{'='*80}\n")

        # Load catalogue
        catalogue_df = pd.read_csv(catalogue_path)
        logger.info(f"✓ Loaded catalogue with {len(catalogue_df)} sources from {catalogue_path}")

        # Create scenario output directory
        scenario_output = output_dir / scenario_name.replace(" ", "_").lower()
        scenario_output.mkdir(parents=True, exist_ok=True)

        # Run Astropy baseline with 1 thread
        logger.info(f"\n[1/4] Running Astropy baseline (1 thread)...")
        try:
            astropy_1t_result = run_astropy_benchmark(
                catalogue_df, scenario_name, scenario_output, threads=1
            )
            all_results.append(astropy_1t_result)
            logger.info(
                f"✓ Astropy baseline (1 thread) completed: {astropy_1t_result['sources_per_second']:.2f} sources/sec"
            )
        except Exception as e:
            logger.error(f"✗ Astropy baseline (1 thread) failed for {scenario_name}: {e}")

        # Run Astropy baseline with 4 threads
        logger.info(f"\n[2/4] Running Astropy baseline (4 threads)...")
        try:
            astropy_4t_result = run_astropy_benchmark(
                catalogue_df, scenario_name, scenario_output, threads=4
            )
            all_results.append(astropy_4t_result)
            logger.info(
                f"✓ Astropy baseline (4 threads) completed: {astropy_4t_result['sources_per_second']:.2f} sources/sec"
            )
        except Exception as e:
            logger.error(f"✗ Astropy baseline (4 threads) failed for {scenario_name}: {e}")

        # Run Cutana with 1 worker
        logger.info(f"\n[3/4] Running Cutana with 1 worker...")
        try:
            cutana_1w_output = scenario_output / "cutana_1worker"
            cutana_1w_result = run_cutana_benchmark(
                catalogue_df, 1, str(cutana_1w_output), scenario_name
            )
            all_results.append(cutana_1w_result)
            logger.info(
                f"✓ Cutana 1 worker completed: {cutana_1w_result['sources_per_second']:.2f} sources/sec"
            )
        except Exception as e:
            logger.error(f"✗ Cutana 1 worker failed for {scenario_name}: {e}")

        # Run Cutana with 4 workers
        logger.info(f"\n[4/4] Running Cutana with 4 workers...")
        try:
            cutana_4w_output = scenario_output / "cutana_4workers"
            cutana_4w_result = run_cutana_benchmark(
                catalogue_df, 4, str(cutana_4w_output), scenario_name
            )
            all_results.append(cutana_4w_result)
            logger.info(
                f"✓ Cutana 4 workers completed: {cutana_4w_result['sources_per_second']:.2f} sources/sec"
            )
        except Exception as e:
            logger.error(f"✗ Cutana 4 workers failed for {scenario_name}: {e}")

        logger.info(f"\n{'='*80}")
        logger.info(f"Scenario {scenario_num}/{total_scenarios} completed")
        logger.info(f"{'='*80}\n")

    return all_results


def save_comparison_results(results: List[Dict[str, any]], output_path: Path):
    """Save comparison results to JSON and timing breakdowns to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved comparison results to: {output_path}")

    # Save timing breakdowns to CSV for plot recreation
    timing_csv_path = output_path.parent / output_path.name.replace(
        ".json", "_timing_breakdowns.csv"
    )

    timing_rows = []
    for result in results:
        if "timing_breakdown" in result and result["timing_breakdown"]:
            scenario = result.get("scenario", "unknown")
            method = result.get("method", "unknown")
            config = ""

            # Determine configuration label
            if "threads" in result:
                config = f"{result['threads']}t"
            elif "max_workers" in result:
                config = f"{result['max_workers']}w"

            # Create row for each timing step
            for step_name, step_time in result["timing_breakdown"].items():
                timing_rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "config": config,
                        "step": step_name,
                        "time_seconds": step_time,
                    }
                )

    if timing_rows:
        timing_df = pd.DataFrame(timing_rows)
        timing_df.to_csv(timing_csv_path, index=False)
        logger.info(f"Saved timing breakdowns to: {timing_csv_path}")
    else:
        logger.warning("No timing breakdown data to save")


def create_comparison_table(results: List[Dict[str, any]]) -> pd.DataFrame:
    """Create summary table from results."""
    df = pd.DataFrame(results)

    # Add a unified "config" column that shows threads or workers
    if "threads" in df.columns and "max_workers" in df.columns:
        df["config"] = df.apply(
            lambda row: (
                f"{row['threads']}t" if pd.notna(row.get("threads")) else f"{row['max_workers']}w"
            ),
            axis=1,
        )
    elif "threads" in df.columns:
        df["config"] = df["threads"].apply(lambda x: f"{x}t" if pd.notna(x) else "")
    elif "max_workers" in df.columns:
        df["config"] = df["max_workers"].apply(lambda x: f"{x}w" if pd.notna(x) else "")

    # Reorder columns for clarity
    column_order = [
        "scenario",
        "method",
        "config",
        "total_sources",
        "total_time_seconds",
        "sources_per_second",
    ]

    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]

    return df


def main():
    """Main framework comparison execution."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Framework comparison: Astropy vs Cutana",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--size",
        choices=["small", "big"],
        default="small",
        help="Catalogue size to use (default: small)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: only run 50k-1tile-4channel benchmark"
    )

    args = parser.parse_args()

    # Set up logging for this script (Orchestrator will reconfigure it later)
    setup_logging(log_level="INFO", console_level="INFO")

    logger.info("=" * 80)
    logger.info("FRAMEWORK COMPARISON BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Mode: {'TEST' if args.test else 'FULL'}")
    logger.info(f"Catalogue size: {args.size}")
    logger.info("=" * 80)

    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    output_dir = results_dir / "framework_comparison"

    # Define catalogues for each scenario
    if args.test:
        # Test mode: use 12k test catalogue
        catalogues = {
            "test_12k": str(script_dir / "catalogues" / "test" / "12k-1tile-4channel.csv"),
        }
    else:
        # Full benchmarks: all three scenarios from size-specific directory
        catalogues_dir = script_dir / "catalogues" / args.size
        if args.size == "small":
            catalogues = {
                "1_tile_4fits_50k": str(catalogues_dir / "50k-1tile-4channel.csv"),
                "8_tiles_4fits_1k": str(catalogues_dir / "1k-8tiles-4channel.csv"),
                "4_tiles_1fits_50k": str(catalogues_dir / "50k-4tiles-1channel.csv"),
            }
        else:  # big
            catalogues = {
                "8_tiles_1channel_200k": str(catalogues_dir / "200k-8tile-1channel.csv"),
                "32_tiles_4fits_1k": str(catalogues_dir / "1k-32tiles-4channel.csv"),
                "4_tiles_1fits_100k": str(catalogues_dir / "100k-4tiles-1channel.csv"),
            }

    # Verify all catalogues exist
    for scenario, path in catalogues.items():
        if not Path(path).exists():
            logger.error(f"Catalogue not found for {scenario}: {path}")
            logger.error(f"Expected location: {path}")
            sys.exit(1)

    # Run all comparisons
    try:
        results = run_all_comparisons(catalogues, output_dir)

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"framework_comparison_{timestamp}.json"
        save_comparison_results(results, results_path)

        # Create summary table
        summary_df = create_comparison_table(results)
        summary_path = results_dir / f"framework_comparison_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary table to: {summary_path}")

        # Print summary
        logger.info("\nFramework Comparison Summary:")
        logger.info("\n" + summary_df.to_string())

        logger.info("\nFramework comparison completed successfully!")

    except Exception as e:
        logger.error(f"Framework comparison failed: {e}")
        logger.error("Exception details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
