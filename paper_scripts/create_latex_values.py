#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Generate LaTeX macros from benchmark results.

Reads results from:
- run_framework_comparison.py
- run_memory_profile.py
- run_scaling_study.py

Generates LaTeX \newcommand definitions for use in paper.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from cutana.logging_config import setup_logging  # noqa: E402


def find_latest_result_file(results_dir: Path, pattern: str) -> Path:
    """
    Find the most recent result file matching pattern.

    Args:
        results_dir: Results directory
        pattern: Glob pattern to match

    Returns:
        Path to most recent file
    """
    matching_files = list(results_dir.glob(pattern))

    if not matching_files:
        raise FileNotFoundError(f"No files matching {pattern} in {results_dir}")

    # Sort by modification time (most recent first)
    matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return matching_files[0]


def extract_framework_comparison_values(results_path: Path) -> Dict[str, Any]:
    """
    Extract values from framework comparison results.

    Args:
        results_path: Path to framework comparison JSON

    Returns:
        Dictionary of extracted values
    """
    logger.info(f"Reading framework comparison results from: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    # Extract values for 1 tile scenario (most relevant for comparison)
    astropy_1t_result = next(
        (
            r
            for r in results
            if r["scenario"] == "1_tile_4fits_100k" and r["method"] == "astropy_1t"
        ),
        None,
    )
    astropy_4t_result = next(
        (
            r
            for r in results
            if r["scenario"] == "1_tile_4fits_100k" and r["method"] == "astropy_4t"
        ),
        None,
    )
    cutana_1w_result = next(
        (
            r
            for r in results
            if r["scenario"] == "1_tile_4fits_100k"
            and r["method"] == "cutana"
            and r["max_workers"] == 1
        ),
        None,
    )
    cutana_4w_result = next(
        (
            r
            for r in results
            if r["scenario"] == "1_tile_4fits_100k"
            and r["method"] == "cutana"
            and r["max_workers"] == 4
        ),
        None,
    )

    if not all([astropy_1t_result, astropy_4t_result, cutana_1w_result, cutana_4w_result]):
        logger.warning("Could not find all required results in framework comparison")
        return {}

    # Calculate speedup factors (using Astropy 1-thread as baseline)
    baseline_time = astropy_1t_result["total_time_seconds"]
    speedup_astropy_4t = (
        baseline_time / astropy_4t_result["total_time_seconds"]
        if astropy_4t_result["total_time_seconds"] > 0
        else 0
    )
    speedup_cutana_1w = (
        baseline_time / cutana_1w_result["total_time_seconds"]
        if cutana_1w_result["total_time_seconds"] > 0
        else 0
    )
    speedup_cutana_4w = (
        baseline_time / cutana_4w_result["total_time_seconds"]
        if cutana_4w_result["total_time_seconds"] > 0
        else 0
    )
    scaling_factor = (
        cutana_4w_result["sources_per_second"] / cutana_1w_result["sources_per_second"]
        if cutana_1w_result["sources_per_second"] > 0
        else 0
    )

    values = {
        "astropyOneThreadTime": f"{astropy_1t_result['total_time_seconds']:.1f}",
        "astropyOneThreadRate": f"{astropy_1t_result['sources_per_second']:.1f}",
        "astropyFourThreadTime": f"{astropy_4t_result['total_time_seconds']:.1f}",
        "astropyFourThreadRate": f"{astropy_4t_result['sources_per_second']:.1f}",
        "cutanaSingleTime": f"{cutana_1w_result['total_time_seconds']:.1f}",
        "cutanaSingleRate": f"{cutana_1w_result['sources_per_second']:.1f}",
        "cutanaFourTime": f"{cutana_4w_result['total_time_seconds']:.1f}",
        "cutanaFourRate": f"{cutana_4w_result['sources_per_second']:.1f}",
        "speedupAstropyFourThread": f"{speedup_astropy_4t:.2f}",
        "speedupCutanaSingle": f"{speedup_cutana_1w:.2f}",
        "speedupCutanaFour": f"{speedup_cutana_4w:.2f}",
        "scalingFactor": f"{scaling_factor:.2f}",
    }

    logger.info("Extracted framework comparison values:")
    for key, value in values.items():
        logger.info(f"  {key}: {value}")

    return values


def extract_memory_profile_values(stats_path: Path) -> Dict[str, Any]:
    """
    Extract values from memory profiling results.

    Args:
        stats_path: Path to memory profile stats JSON

    Returns:
        Dictionary of extracted values
    """
    logger.info(f"Reading memory profile stats from: {stats_path}")

    with open(stats_path, "r") as f:
        stats = json.load(f)

    values = {
        "memoryAstropyFourThreads": f"{stats['astropy_4_threads']['peak_memory_gb']:.2f}",
        "memoryUsageSingle": f"{stats['cutana_1_worker']['peak_memory_gb']:.2f}",
        "memoryUsageFour": f"{stats['cutana_4_workers']['peak_memory_gb']:.2f}",
    }

    logger.info("Extracted memory profile values:")
    for key, value in values.items():
        logger.info(f"  {key}: {value}")

    return values


def generate_latex_macros(values: Dict[str, str], output_path: Path):
    """
    Generate LaTeX macro definitions.

    Args:
        values: Dictionary of macro names and values
        output_path: Path to save LaTeX file
    """
    logger.info(f"Generating LaTeX macros to: {output_path}")

    latex_content = [
        "% Performance benchmark variables - generated from paper_scripts/create_latex_values.py",
        "% Generated automatically - DO NOT EDIT MANUALLY",
        "",
    ]

    # Define all macros in order requested
    macro_definitions = {
        "astropyOneThreadTime": "Astropy 1 thread time (seconds)",
        "astropyOneThreadRate": "Astropy 1 thread cutouts per second",
        "astropyFourThreadTime": "Astropy 4 threads time (seconds)",
        "astropyFourThreadRate": "Astropy 4 threads cutouts per second",
        "cutanaSingleTime": "Cutana 1 worker time (seconds)",
        "cutanaSingleRate": "Cutana 1 worker cutouts/sec",
        "cutanaFourTime": "Cutana 4 workers time (seconds)",
        "cutanaFourRate": "Cutana 4 workers cutouts/sec",
        "speedupAstropyFourThread": "Astropy 4t vs 1t speedup factor",
        "speedupCutanaSingle": "Cutana 1w vs Astropy 1t speedup factor",
        "speedupCutanaFour": "Cutana 4w vs Astropy 1t speedup factor",
        "scalingFactor": "Cutana 4w vs 1w scaling factor",
        "memoryAstropyFourThreads": "Memory usage Astropy 4 threads (GB)",
        "memoryUsageSingle": "Memory usage Cutana 1 worker (GB)",
        "memoryUsageFour": "Memory usage Cutana 4 workers (GB)",
    }

    for macro_name, description in macro_definitions.items():
        value = values.get(macro_name, "TBD")
        latex_content.append(f"\\newcommand{{\\{macro_name}}}{{{value}}}       % {description}")

    latex_content.append("")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(latex_content))

    logger.info(f"LaTeX macros saved to: {output_path}")


def create_summary_table(values: Dict[str, str], output_path: Path):
    """
    Create human-readable summary table.

    Args:
        values: Dictionary of values
        output_path: Path to save summary
    """
    logger.info(f"Creating summary table: {output_path}")

    summary_lines = [
        "=" * 80,
        "BENCHMARK RESULTS SUMMARY FOR PAPER",
        "=" * 80,
        "",
        "FRAMEWORK COMPARISON (1 Tile, 4 FITS, 50k Sources):",
        "-" * 80,
        f"Astropy 1 thread:       {values.get('astropyOneThreadTime', 'TBD')}s ({values.get('astropyOneThreadRate', 'TBD')} cutouts/s)",
        f"Astropy 4 threads:      {values.get('astropyFourThreadTime', 'TBD')}s ({values.get('astropyFourThreadRate', 'TBD')} cutouts/s)",
        f"Cutana 1 worker:        {values.get('cutanaSingleTime', 'TBD')}s ({values.get('cutanaSingleRate', 'TBD')} cutouts/s)",
        f"Cutana 4 workers:       {values.get('cutanaFourTime', 'TBD')}s ({values.get('cutanaFourRate', 'TBD')} cutouts/s)",
        "",
        "SPEEDUP FACTORS (vs Astropy 1 thread baseline):",
        "-" * 80,
        f"Astropy 4t vs 1t:       {values.get('speedupAstropyFourThread', 'TBD')}x",
        f"Cutana 1w vs Astropy 1t: {values.get('speedupCutanaSingle', 'TBD')}x",
        f"Cutana 4w vs Astropy 1t: {values.get('speedupCutanaFour', 'TBD')}x",
        f"Cutana 4w vs 1w:        {values.get('scalingFactor', 'TBD')}x",
        "",
        "MEMORY USAGE (PEAK):",
        "-" * 80,
        f"Astropy 4 threads:      {values.get('memoryAstropyFourThreads', 'TBD')} GB",
        f"Cutana 1 worker:        {values.get('memoryUsageSingle', 'TBD')} GB",
        f"Cutana 4 workers:       {values.get('memoryUsageFour', 'TBD')} GB",
        "",
        "=" * 80,
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(summary_lines))

    logger.info(f"Summary table saved to: {output_path}")

    # Also print to console
    logger.info("\n" + "\n".join(summary_lines))


def format_scenario_name(scenario: str, total_sources: int) -> str:
    """
    Format scenario name for display.

    Args:
        scenario: Raw scenario name (e.g., "8_tiles_1channel_200k")
        total_sources: Actual total source count from data

    Returns:
        Formatted scenario name (e.g., "8 Tiles - 1 Channel - 200,000")
    """
    # Parse the scenario name
    parts = scenario.split("_")

    # Extract components based on pattern
    if "tiles" in scenario:
        tiles_idx = parts.index("tiles") - 1
        n_tiles = parts[tiles_idx]

        if "channel" in scenario:
            # e.g., 8_tiles_1channel_200k
            channel_idx = [i for i, p in enumerate(parts) if "channel" in p][0]
            n_channels = parts[channel_idx].replace("channel", "")

            # Use actual total_sources count
            return f"{n_tiles} Tiles - {n_channels} Channel - {total_sources:,}"

        elif "fits" in scenario:
            # e.g., 32_tiles_4fits_1k or 4_tiles_1fits_100k
            fits_idx = [i for i, p in enumerate(parts) if "fits" in p][0]
            n_fits = parts[fits_idx].replace("fits", "")

            # Use actual total_sources count
            return f"{n_tiles} Tiles - {n_fits} FITS - {total_sources:,}"

    # Fallback: just capitalize and replace underscores
    return scenario.replace("_", " ").title()


def format_tool_name(method: str, config: str) -> str:
    """
    Format tool name with threads/workers.

    Args:
        method: Method name (e.g., "astropy_1t", "cutana")
        config: Config string (e.g., "1.0t", "4.0w")

    Returns:
        Formatted tool name (e.g., "Astropy (1 Thread)", "Cutana (4 Workers)")
    """
    # Extract thread/worker count from config
    if "t" in config:
        count = int(float(config.replace("t", "")))
        thread_label = "Thread" if count == 1 else "Threads"
    elif "w" in config:
        count = int(float(config.replace("w", "")))
        thread_label = "Worker" if count == 1 else "Workers"
    else:
        thread_label = "Unknown"
        count = 0

    # Get tool name
    if "astropy" in method:
        tool = "Astropy"
    elif "cutana" in method:
        tool = "Cutana"
    else:
        tool = method.capitalize()

    return f"{tool} ({count} {thread_label})"


def create_framework_comparison_table(csv_path: Path, output_path: Path):
    """
    Create LaTeX table from framework comparison CSV.

    Args:
        csv_path: Path to framework comparison summary CSV
        output_path: Path to save LaTeX table
    """
    logger.info(f"Creating framework comparison LaTeX table from: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Start LaTeX table
    latex_lines = [
        "% Framework comparison table - generated from paper_scripts/create_latex_values.py",
        "% Generated automatically - DO NOT EDIT MANUALLY",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Framework Comparison: Astropy vs Cutana Performance}",
        "\\label{tab:framework_comparison}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "\\textbf{Scenario} & \\textbf{Tool (Threads/Workers)} & \\textbf{Runtime (s)} & \\textbf{Sources/sec} \\\\",
        "\\midrule",
    ]

    # Group by scenario
    current_scenario = None
    for _, row in df.iterrows():
        scenario = row["scenario"]
        method = row["method"]
        config = row["config"]
        runtime = row["total_time_seconds"]
        throughput = row["sources_per_second"]
        total_sources = row["total_sources"]

        # Format values - use actual total_sources count instead of parsing from name
        scenario_formatted = format_scenario_name(scenario, total_sources)
        tool_formatted = format_tool_name(method, config)
        runtime_formatted = f"{runtime:.1f}"
        throughput_formatted = f"{throughput:.1f}"

        # Add scenario divider if new scenario
        if current_scenario != scenario:
            if current_scenario is not None:
                latex_lines.append("\\midrule")
            current_scenario = scenario

        # Add row
        latex_lines.append(
            f"{scenario_formatted} & {tool_formatted} & {runtime_formatted} & {throughput_formatted} \\\\"
        )

    # End table
    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    logger.info(f"LaTeX table saved to: {output_path}")


def main():
    """Main LaTeX values generation."""
    setup_logging(log_level="INFO", console_level="INFO")

    logger.info("Generating LaTeX values from benchmark results")

    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        logger.error("Please run benchmarks first using create_results.py")
        sys.exit(1)

    all_values = {}

    try:
        # Extract framework comparison values
        try:
            framework_results = find_latest_result_file(results_dir, "framework_comparison_*.json")
            framework_values = extract_framework_comparison_values(framework_results)
            all_values.update(framework_values)
        except Exception as e:
            logger.warning(f"Could not extract framework comparison values: {e}")

        # Extract memory profile values
        try:
            memory_stats = find_latest_result_file(results_dir, "memory_profile_stats_*.json")
            memory_values = extract_memory_profile_values(memory_stats)
            all_values.update(memory_values)
        except Exception as e:
            logger.warning(f"Could not extract memory profile values: {e}")

        # Generate LaTeX macros
        latex_dir = script_dir / "latex"
        latex_dir.mkdir(parents=True, exist_ok=True)
        latex_output = latex_dir / "latex_values.tex"
        generate_latex_macros(all_values, latex_output)

        # Create summary table
        summary_output = latex_dir / "benchmark_summary.txt"
        create_summary_table(all_values, summary_output)

        # Create framework comparison LaTeX table
        try:
            framework_csv = find_latest_result_file(
                results_dir, "framework_comparison_summary_*.csv"
            )
            table_output = latex_dir / "framework_comparison_table.tex"
            create_framework_comparison_table(framework_csv, table_output)
            logger.info(f"Framework comparison table: {table_output}")
        except Exception as e:
            logger.warning(f"Could not create framework comparison table: {e}")

        logger.info("\n✓ LaTeX values generation completed successfully!")
        logger.info(f"\nLaTeX file: {latex_output}")
        logger.info(f"Summary: {summary_output}")

    except Exception as e:
        logger.error(f"LaTeX values generation failed: {e}")
        logger.error("Exception details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
