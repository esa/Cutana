#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Recreate all paper plots from saved benchmark data.

This script reads CSV and JSON files from the results/ directory and
regenerates all plots in the figures/ directory.

Usage:
    python recreate_plots.py                    # Use latest results
    python recreate_plots.py --timestamp YYYYMMDD_HHMMSS  # Use specific timestamp

Reads from:
- results/memory_profile_traces_*.csv
- results/framework_comparison_*_timing_breakdowns.csv
- results/scaling_summary_*.csv
- results/scaling_metrics_*.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from cutana.logging_config import setup_logging  # noqa: E402
from paper_scripts.plots import (  # noqa: E402
    create_cutana_timing_chart,
    create_memory_plot,
    create_scaling_plots,
    create_timing_breakdown_chart,
)


def find_latest_file(results_dir: Path, pattern: str) -> Optional[Path]:
    """
    Find the most recent file matching pattern.

    Args:
        results_dir: Results directory
        pattern: Glob pattern to match

    Returns:
        Path to most recent file or None if not found
    """
    matching_files = list(results_dir.glob(pattern))

    if not matching_files:
        return None

    # Sort by modification time (most recent first)
    matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return matching_files[0]


def find_file_by_timestamp(results_dir: Path, pattern: str, timestamp: str) -> Optional[Path]:
    """
    Find file matching pattern with specific timestamp.

    Args:
        results_dir: Results directory
        pattern: Glob pattern with {timestamp} placeholder
        timestamp: Timestamp string YYYYMMDD_HHMMSS

    Returns:
        Path to file or None if not found
    """
    file_pattern = pattern.replace("{timestamp}", timestamp)
    matching_files = list(results_dir.glob(file_pattern))

    if matching_files:
        return matching_files[0]
    return None


def load_memory_traces(csv_path: Path) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Load memory traces from CSV.

    Args:
        csv_path: Path to memory traces CSV

    Returns:
        Tuple of (astropy_4t_data, cutana_1w_data, cutana_4w_data)
        where each data is (memory_history, timestamps)
    """
    logger.info(f"Loading memory traces from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Extract data for each method (remove NaN values)
    astropy_4t_data = (
        df["astropy_4t_memory_mb"].dropna().tolist(),
        df["astropy_4t_time_sec"].dropna().tolist(),
    )
    cutana_1w_data = (
        df["cutana_1w_memory_mb"].dropna().tolist(),
        df["cutana_1w_time_sec"].dropna().tolist(),
    )
    cutana_4w_data = (
        df["cutana_4w_memory_mb"].dropna().tolist(),
        df["cutana_4w_time_sec"].dropna().tolist(),
    )

    logger.info(
        f"✓ Loaded memory traces: Astropy 4t={len(astropy_4t_data[0])} points, "
        f"Cutana 1w={len(cutana_1w_data[0])} points, Cutana 4w={len(cutana_4w_data[0])} points"
    )

    return astropy_4t_data, cutana_1w_data, cutana_4w_data


def load_timing_breakdowns(csv_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load timing breakdowns from CSV.

    Args:
        csv_path: Path to timing breakdowns CSV

    Returns:
        Nested dictionary: {scenario: {method_config: {step: time}}}
    """
    logger.info(f"Loading timing breakdowns from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Group by scenario, method, and config
    breakdowns = {}
    for scenario in df["scenario"].unique():
        breakdowns[scenario] = {}
        scenario_df = df[df["scenario"] == scenario]

        for _, row in scenario_df.iterrows():
            method = row["method"]
            config = row["config"]
            step = row["step"]
            time_val = row["time_seconds"]

            # Create key like "astropy_1t" or "cutana_4w"
            key = f"{method}_{config}" if config else method

            if key not in breakdowns[scenario]:
                breakdowns[scenario][key] = {}

            breakdowns[scenario][key][step] = time_val

    logger.info(f"✓ Loaded timing breakdowns for {len(breakdowns)} scenarios")

    return breakdowns


def load_scaling_metrics(json_path: Path) -> Dict[str, any]:
    """
    Load scaling metrics from JSON.

    Args:
        json_path: Path to scaling metrics JSON

    Returns:
        Scaling metrics dictionary
    """
    logger.info(f"Loading scaling metrics from: {json_path}")

    with open(json_path, "r") as f:
        metrics = json.load(f)

    logger.info(f"✓ Loaded scaling metrics for {len(metrics['worker_counts'])} worker counts")

    return metrics


def recreate_memory_plot(
    results_dir: Path,
    figures_dir: Path,
    timestamp: Optional[str] = None,
    catalogue_desc_override: Optional[str] = None,
):
    """Recreate memory consumption plot."""
    logger.info("\n" + "=" * 80)
    logger.info("Recreating Memory Consumption Plot")
    logger.info("=" * 80)

    # Find memory traces CSV
    if timestamp:
        csv_path = find_file_by_timestamp(
            results_dir, f"memory_profile_traces_{timestamp}.csv", timestamp
        )
        stats_path = find_file_by_timestamp(
            results_dir, f"memory_profile_stats_{timestamp}.json", timestamp
        )
    else:
        csv_path = find_latest_file(results_dir, "memory_profile_traces_*.csv")
        stats_path = find_latest_file(results_dir, "memory_profile_stats_*.json")

    if not csv_path:
        logger.error("Memory traces CSV not found")
        return False

    # Extract timestamp from filename for output naming
    csv_filename = csv_path.stem  # e.g., "memory_profile_traces_20241023_121336"
    output_timestamp = csv_filename.replace("memory_profile_traces_", "")

    try:
        # Load data
        astropy_4t_data, cutana_1w_data, cutana_4w_data = load_memory_traces(csv_path)

        # Determine catalogue description
        if catalogue_desc_override:
            catalogue_description = catalogue_desc_override
            logger.info(f"Using user-provided catalogue description: {catalogue_description}")
        else:
            # Load catalogue description from stats JSON
            catalogue_description = None
            if stats_path and stats_path.exists():
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                    catalogue_description = stats.get("catalogue_description")
                    if catalogue_description:
                        logger.info(
                            f"Loaded catalogue description from stats: {catalogue_description}"
                        )

            if not catalogue_description:
                logger.warning(
                    "No catalogue description found in data files. Using default fallback."
                )
                logger.warning("For correct titles, use: --catalogue-desc 'YOUR DESCRIPTION'")
                catalogue_description = "1 Tile, 4 FITS, 50k Sources"

        # Create output directory
        recreated_dir = figures_dir / "recreated"
        recreated_dir.mkdir(parents=True, exist_ok=True)

        # Create plot with same filename as original
        output_path = recreated_dir / f"memory_profile_{output_timestamp}.png"
        create_memory_plot(
            astropy_4t_data, cutana_1w_data, cutana_4w_data, output_path, catalogue_description
        )

        logger.info(f"✓ Memory plot recreated: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to recreate memory plot: {e}")
        logger.error("Exception details:", exc_info=True)
        return False


def recreate_timing_breakdown_plots(
    results_dir: Path, figures_dir: Path, timestamp: Optional[str] = None
):
    """Recreate timing breakdown plots for all scenarios."""
    logger.info("\n" + "=" * 80)
    logger.info("Recreating Timing Breakdown Plots")
    logger.info("=" * 80)

    # Find timing breakdowns CSV
    if timestamp:
        csv_path = find_file_by_timestamp(
            results_dir, f"framework_comparison_{timestamp}_timing_breakdowns.csv", timestamp
        )
    else:
        csv_path = find_latest_file(results_dir, "framework_comparison_*_timing_breakdowns.csv")

    if not csv_path:
        logger.error("Timing breakdowns CSV not found")
        return False

    try:
        # Load data
        breakdowns = load_timing_breakdowns(csv_path)

        # Create output directory
        recreated_dir = figures_dir / "recreated"
        recreated_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        total_count = 0

        # Create plots for each scenario and method
        for scenario, methods in breakdowns.items():
            for method_config, timing_data in methods.items():
                total_count += 1

                # Determine output filename (same as original, without _recreated)
                scenario_slug = scenario.replace(" ", "_").lower()
                output_path = recreated_dir / f"{method_config}_timing_{scenario_slug}.png"

                # Determine title and type
                if method_config.startswith("astropy"):
                    title = f"Astropy Baseline ({method_config.split('_')[1]}) Timing - {scenario}"
                    create_timing_breakdown_chart(timing_data, output_path, title)
                elif method_config.startswith("cutana"):
                    # Extract worker count
                    workers_str = method_config.split("_")[1]
                    max_workers = int(workers_str.replace("w", ""))
                    title = f"Cutana Timing - {scenario}"
                    create_cutana_timing_chart(timing_data, output_path, title, max_workers)
                else:
                    logger.warning(f"Unknown method type: {method_config}")
                    continue

                logger.info(f"✓ Created: {output_path}")
                success_count += 1

        logger.info(f"✓ Timing breakdown plots recreated: {success_count}/{total_count}")
        return success_count == total_count

    except Exception as e:
        logger.error(f"Failed to recreate timing breakdown plots: {e}")
        logger.error("Exception details:", exc_info=True)
        return False


def recreate_scaling_plots(
    results_dir: Path,
    figures_dir: Path,
    timestamp: Optional[str] = None,
    catalogue_desc_override: Optional[str] = None,
):
    """Recreate scaling study plots."""
    logger.info("\n" + "=" * 80)
    logger.info("Recreating Scaling Study Plots")
    logger.info("=" * 80)

    # Find scaling metrics JSON
    if timestamp:
        json_path = find_file_by_timestamp(
            results_dir, f"scaling_metrics_{timestamp}.json", timestamp
        )
    else:
        json_path = find_latest_file(results_dir, "scaling_metrics_*.json")

    if not json_path:
        logger.error("Scaling metrics JSON not found")
        return False

    # Extract timestamp from filename for output naming
    json_filename = json_path.stem  # e.g., "scaling_metrics_20241023_121336"
    output_timestamp = json_filename.replace("scaling_metrics_", "")

    try:
        # Load data
        metrics = load_scaling_metrics(json_path)

        # Determine catalogue description
        if catalogue_desc_override:
            catalogue_description = catalogue_desc_override
            logger.info(f"Using user-provided catalogue description: {catalogue_description}")
        else:
            catalogue_description = metrics.get("catalogue_description")
            if catalogue_description:
                logger.info(f"Loaded catalogue description from metrics: {catalogue_description}")
            else:
                logger.warning(
                    "No catalogue description found in data files. Using default fallback."
                )
                logger.warning("For correct titles, use: --catalogue-desc 'YOUR DESCRIPTION'")
                catalogue_description = "4 Tiles, 1 FITS, 50k Sources"

        # Create output directory
        recreated_dir = figures_dir / "recreated"
        recreated_dir.mkdir(parents=True, exist_ok=True)

        # Create plots with same filename as original
        create_scaling_plots(metrics, recreated_dir, output_timestamp, catalogue_description)

        logger.info(f"✓ Scaling plots recreated")
        return True

    except Exception as e:
        logger.error(f"Failed to recreate scaling plots: {e}")
        logger.error("Exception details:", exc_info=True)
        return False


def main():
    """Main plot recreation execution."""
    parser = argparse.ArgumentParser(
        description="Recreate all paper plots from saved benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recreate_plots.py                      # Use latest results, auto-detect size
  python recreate_plots.py --size big           # Override to big size descriptions
  python recreate_plots.py --size small         # Override to small size descriptions
  python recreate_plots.py --timestamp 20241023_143000 --size big
        """,
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Specific timestamp to use (YYYYMMDD_HHMMSS). If not specified, uses latest.",
    )
    parser.add_argument(
        "--size",
        choices=["small", "big"],
        help="Catalogue size (small or big). Overrides all plot titles with correct descriptions.",
    )

    args = parser.parse_args()

    setup_logging(log_level="INFO", console_level="INFO")

    logger.info("=" * 80)
    logger.info("RECREATE ALL PAPER PLOTS FROM SAVED DATA")
    logger.info("=" * 80)
    if args.timestamp:
        logger.info(f"Using timestamp: {args.timestamp}")
    else:
        logger.info("Using latest results")
    if args.size:
        logger.info(f"Size override: {args.size}")
    logger.info("=" * 80)

    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    figures_dir = script_dir / "figures"

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        logger.error("Please run benchmarks first using create_results.py")
        sys.exit(1)

    # Ensure figures directory exists
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Determine catalogue descriptions based on size
    memory_desc = None
    scaling_desc = None

    if args.size:
        if args.size == "small":
            memory_desc = "50k sources, 1 tile, 4 FITS"
            scaling_desc = "50k sources, 4 tiles, 1 FITS"
        else:  # big
            memory_desc = "200k sources, 8 tiles, 1 FITS"
            scaling_desc = "100k sources, 4 tiles, 1 FITS"

        logger.info(f"Memory plot description: {memory_desc}")
        logger.info(f"Scaling plot description: {scaling_desc}")

    # Track success
    all_successful = True

    # 1. Recreate memory plot
    if not recreate_memory_plot(results_dir, figures_dir, args.timestamp, memory_desc):
        logger.warning("Memory plot recreation failed")
        all_successful = False

    # 2. Recreate timing breakdown plots
    if not recreate_timing_breakdown_plots(results_dir, figures_dir, args.timestamp):
        logger.warning("Timing breakdown plots recreation failed")
        all_successful = False

    # 3. Recreate scaling plots
    if not recreate_scaling_plots(results_dir, figures_dir, args.timestamp, scaling_desc):
        logger.warning("Scaling plots recreation failed")
        all_successful = False

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PLOT RECREATION SUMMARY")
    logger.info("=" * 80)

    recreated_dir = figures_dir / "recreated"
    logger.info(f"Output directory: {recreated_dir}")

    if all_successful:
        logger.info("\n✓ All plots recreated successfully!")
        logger.info(f"\nRecreated plots are in: {recreated_dir}/")
        logger.info("  - memory_profile_TIMESTAMP.png")
        logger.info("  - *_timing_*.png (multiple timing breakdown charts)")
        logger.info("  - scaling_study_TIMESTAMP.png")
    else:
        logger.warning("\n⚠ Some plots failed to recreate. Please review the logs above.")
        logger.info(f"Partial results available in: {recreated_dir}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
