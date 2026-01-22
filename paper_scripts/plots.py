#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Plotting functions for Cutana paper benchmarks.

This module contains all plotting and visualization functions used by the
benchmark scripts for creating timing breakdown charts and other visualizations.
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import toml
from loguru import logger


def load_plot_config() -> Dict:
    """Load plot configuration from benchmark_config.toml."""
    config_path = Path(__file__).parent / "benchmark_config.toml"
    if config_path.exists():
        config = toml.load(config_path)
        return config.get("plots", {})
    else:
        # Return defaults if config file not found
        return {
            "dpi": 300,
            "figure_width": 12,
            "figure_height": 6,
            "colors": [
                "#ff9999",
                "#66b3ff",
                "#99ff99",
                "#ffcc99",
                "#ff99cc",
                "#99ffff",
                "#ffff99",
                "#cc99ff",
            ],
        }


def create_timing_breakdown_chart(
    timing_data: Dict[str, float], output_path: Path, title: str = "Processing Step Timing"
):
    """
    Create bar chart showing timing breakdown by processing step.

    Args:
        timing_data: Dictionary mapping step names to time in seconds
        output_path: Path to save the chart
        title: Chart title
    """
    plot_config = load_plot_config()

    # Format step names for readability
    step_name_map = {
        "fits_loading": "FITS Loading",
        "cutout_extraction": "Cutout Extraction",
        "resizing": "Image Resizing",
        "flux_conversion": "Flux Conversion",
        "normalization": "Normalization",
        "fits_writing": "FITS Writing",
    }

    steps = []
    times = []
    for step, time_val in timing_data.items():
        formatted_name = step_name_map.get(step, step)
        steps.append(formatted_name)
        times.append(time_val)

    # Calculate percentages
    total_time = sum(times)
    percentages = [(t / total_time * 100) if total_time > 0 else 0 for t in times]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(plot_config["figure_width"], plot_config["figure_height"]))
    colors = plot_config["colors"]
    bars = ax.bar(steps, times, color=colors[: len(steps)], alpha=0.8)

    ax.set_xlabel("Processing Step", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, time_val, pct in zip(bars, times, percentages):
        label_text = f"{time_val:.2f}s\n({pct:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.01,
            label_text,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Increase y-axis limit to prevent label overlap with frame
    ax.set_ylim(0, max(times) * 1.2)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_config["dpi"], bbox_inches="tight")
    logger.info(f"Saved timing breakdown chart to: {output_path}")
    plt.close()


def create_cutana_timing_chart(
    timing_data: Dict[str, float], output_path: Path, title: str, max_workers: int
):
    """
    Create bar chart showing Cutana timing breakdown.

    Args:
        timing_data: Dictionary mapping step names to time in seconds
        output_path: Path to save the chart
        title: Chart title
        max_workers: Number of workers used
    """
    plot_config = load_plot_config()

    # Format step names for readability
    step_name_map = {
        "FitsLoading": "FITS Loading",
        "CutoutExtraction": "Cutout Extraction",
        "ImageResizing": "Image Resizing",
        "ChannelMixing": "Channel Mixing",
        "Normalisation": "Normalization",
        "DataTypeConversion": "Data Type Conversion",
        "MetaDataPostprocessing": "Metadata Processing",
        "ZarrFitsSaving": "Output Writing",
    }

    steps = []
    times = []
    for step, time_val in timing_data.items():
        if time_val > 0:  # Only include steps with non-zero time
            formatted_name = step_name_map.get(step, step)
            steps.append(formatted_name)
            times.append(time_val)

    if not times:
        logger.warning("No timing data available for Cutana chart")
        return

    # Calculate percentages
    total_time = sum(times)
    percentages = [(t / total_time * 100) if total_time > 0 else 0 for t in times]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(plot_config["figure_width"], plot_config["figure_height"]))
    colors = plot_config["colors"]
    bars = ax.bar(steps, times, color=colors[: len(steps)], alpha=0.8)

    ax.set_xlabel("Processing Step", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(f"{title} ({max_workers} workers)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, time_val, pct in zip(bars, times, percentages):
        label_text = f"{time_val:.2f}s\n({pct:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.01,
            label_text,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Increase y-axis limit to prevent label overlap with frame
    ax.set_ylim(0, max(times) * 1.2)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_config["dpi"], bbox_inches="tight")
    logger.info(f"Saved Cutana timing breakdown chart to: {output_path}")
    plt.close()


def create_comparison_chart(
    astropy_timing: Dict[str, float],
    cutana_timing: Dict[str, float],
    output_path: Path,
    scenario_name: str,
):
    """
    Create side-by-side comparison chart for Astropy vs Cutana timing.

    Args:
        astropy_timing: Astropy timing breakdown
        cutana_timing: Cutana timing breakdown
        output_path: Path to save the chart
        scenario_name: Name of the scenario
    """
    plot_config = load_plot_config()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(plot_config["figure_width"] * 1.5, plot_config["figure_height"])
    )

    # Astropy chart (left)
    astropy_steps = []
    astropy_times = []
    step_name_map_astropy = {
        "fits_loading": "FITS Loading",
        "cutout_extraction": "Cutout Extraction",
        "resizing": "Resizing",
        "flux_conversion": "Flux Conversion",
        "normalization": "Normalization",
        "fits_writing": "FITS Writing",
    }

    for step, time_val in astropy_timing.items():
        astropy_steps.append(step_name_map_astropy.get(step, step))
        astropy_times.append(time_val)

    colors = plot_config["colors"]
    bars1 = ax1.bar(astropy_steps, astropy_times, color=colors[: len(astropy_steps)], alpha=0.8)
    ax1.set_title(f"Astropy Baseline - {scenario_name}", fontsize=13)
    ax1.set_ylabel("Time (seconds)", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", rotation=45, labelsize=9)

    # Increase y-axis limit to prevent label overlap with frame
    if astropy_times:
        ax1.set_ylim(0, max(astropy_times) * 1.15)

    # Add total time label
    total_astropy = sum(astropy_times)
    ax1.text(
        0.5,
        0.98,
        f"Total: {total_astropy:.2f}s",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Cutana chart (right)
    cutana_steps = []
    cutana_times = []
    step_name_map_cutana = {
        "FitsLoading": "FITS Loading",
        "CutoutExtraction": "Cutout Extraction",
        "ImageResizing": "Resizing",
        "ChannelMixing": "Channel Mixing",
        "Normalisation": "Normalization",
        "DataTypeConversion": "Data Type Conv.",
        "MetaDataPostprocessing": "Metadata Proc.",
        "ZarrFitsSaving": "Output Writing",
    }

    for step, time_val in cutana_timing.items():
        if time_val > 0:
            cutana_steps.append(step_name_map_cutana.get(step, step))
            cutana_times.append(time_val)

    bars2 = ax2.bar(cutana_steps, cutana_times, color=colors[: len(cutana_steps)], alpha=0.8)
    ax2.set_title(f"Cutana - {scenario_name}", fontsize=13)
    ax2.set_ylabel("Time (seconds)", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=45, labelsize=9)

    # Increase y-axis limit to prevent label overlap with frame
    if cutana_times:
        ax2.set_ylim(0, max(cutana_times) * 1.15)

    # Add total time label
    total_cutana = sum(cutana_times)
    ax2.text(
        0.5,
        0.98,
        f"Total: {total_cutana:.2f}s",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_config["dpi"], bbox_inches="tight")
    logger.info(f"Saved comparison chart to: {output_path}")
    plt.close()


def create_memory_plot(
    astropy_4t_data: Tuple,
    cutana_1w_data: Tuple,
    cutana_4w_data: Tuple,
    output_path: Path,
    catalogue_description: str = "1 Tile, 4 FITS, 50k Sources",
):
    """
    Create memory consumption comparison plot.

    Args:
        astropy_4t_data: Tuple of (memory_history, timestamps) for Astropy 4 threads
        cutana_1w_data: Tuple of (memory_history, timestamps) for Cutana 1 worker
        cutana_4w_data: Tuple of (memory_history, timestamps) for Cutana 4 workers
        output_path: Path to save plot
        catalogue_description: Description of catalogue (e.g. "1 Tile, 4 FITS, 50k Sources")
    """
    plot_config = load_plot_config()

    logger.info("Creating memory consumption plot")

    fig, ax = plt.subplots(figsize=(plot_config["figure_width"], plot_config["figure_height"]))

    # Unpack data
    astropy_4t_mem, astropy_4t_time = astropy_4t_data
    cutana_1w_mem, cutana_1w_time = cutana_1w_data
    cutana_4w_mem, cutana_4w_time = cutana_4w_data

    # Convert to minutes
    astropy_4t_time_min = [t / 60 for t in astropy_4t_time]
    cutana_1w_time_min = [t / 60 for t in cutana_1w_time]
    cutana_4w_time_min = [t / 60 for t in cutana_4w_time]

    # Convert to GB
    astropy_4t_mem_gb = [m / 1024 for m in astropy_4t_mem]
    cutana_1w_mem_gb = [m / 1024 for m in cutana_1w_mem]
    cutana_4w_mem_gb = [m / 1024 for m in cutana_4w_mem]

    # Plot memory traces
    ax.plot(
        astropy_4t_time_min,
        astropy_4t_mem_gb,
        label="Astropy (4 threads)",
        linewidth=2,
        color="#1f77b4",
        alpha=0.8,
    )
    ax.plot(
        cutana_1w_time_min,
        cutana_1w_mem_gb,
        label="Cutana (1 worker)",
        linewidth=2,
        color="#ff7f0e",
        alpha=0.8,
    )
    ax.plot(
        cutana_4w_time_min,
        cutana_4w_mem_gb,
        label="Cutana (4 workers)",
        linewidth=2,
        color="#2ca02c",
        alpha=0.8,
    )

    # Add peak markers
    astropy_4t_peak = max(astropy_4t_mem_gb)
    cutana_1w_peak = max(cutana_1w_mem_gb)
    cutana_4w_peak = max(cutana_4w_mem_gb)

    ax.axhline(
        y=astropy_4t_peak,
        color="#1f77b4",
        linestyle="--",
        alpha=0.5,
        label=f"Astropy 4t peak: {astropy_4t_peak:.2f} GB",
    )
    ax.axhline(
        y=cutana_1w_peak,
        color="#ff7f0e",
        linestyle="--",
        alpha=0.5,
        label=f"Cutana 1w peak: {cutana_1w_peak:.2f} GB",
    )
    ax.axhline(
        y=cutana_4w_peak,
        color="#2ca02c",
        linestyle="--",
        alpha=0.5,
        label=f"Cutana 4w peak: {cutana_4w_peak:.2f} GB",
    )

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Memory Usage (GB)", fontsize=12)
    ax.set_title(f"Memory Consumption Comparison ({catalogue_description})", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_config["dpi"], bbox_inches="tight")
    logger.info(f"Saved memory plot to: {output_path}")
    plt.close()


def create_scaling_plots(
    metrics: Dict[str, any],
    output_dir: Path,
    timestamp: str,
    catalogue_description: str = "4 Tiles, 1 FITS, 50k Sources",
):
    """
    Create scaling analysis plots.

    Args:
        metrics: Scaling metrics dictionary with keys:
                 - worker_counts: List of worker counts
                 - runtimes: List of runtimes
                 - throughputs: List of throughputs
                 - speedups: List of speedup factors
                 - efficiencies: List of parallel efficiencies
        output_dir: Output directory for plots
        timestamp: Timestamp string for filenames
        catalogue_description: Description of catalogue (e.g. "4 Tiles, 1 FITS, 50k Sources")
    """
    plot_config = load_plot_config()

    worker_counts = metrics["worker_counts"]
    runtimes = metrics["runtimes"]
    throughputs = metrics["throughputs"]
    speedups = metrics["speedups"]
    efficiencies = metrics["efficiencies"]

    # Create 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Runtime vs Workers
    ax1.plot(worker_counts, runtimes, marker="o", linewidth=2, markersize=8, color="#1f77b4")
    ax1.set_xlabel("Number of Workers", fontsize=12)
    ax1.set_ylabel("Runtime (seconds)", fontsize=12)
    ax1.set_title("Runtime vs Number of Workers", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(worker_counts)

    # 2. Throughput vs Workers
    ax2.plot(worker_counts, throughputs, marker="s", linewidth=2, markersize=8, color="#ff7f0e")
    ax2.set_xlabel("Number of Workers", fontsize=12)
    ax2.set_ylabel("Throughput (sources/second)", fontsize=12)
    ax2.set_title("Throughput vs Number of Workers", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(worker_counts)

    # 3. Speedup vs Workers (with ideal linear scaling reference)
    ax3.plot(
        worker_counts,
        speedups,
        marker="^",
        linewidth=2,
        markersize=8,
        color="#2ca02c",
        label="Actual",
    )
    ax3.plot(
        worker_counts,
        worker_counts,
        linestyle="--",
        linewidth=2,
        color="#d62728",
        label="Ideal Linear",
        alpha=0.7,
    )
    ax3.set_xlabel("Number of Workers", fontsize=12)
    ax3.set_ylabel("Speedup Factor", fontsize=12)
    ax3.set_title("Speedup vs Number of Workers", fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(worker_counts)
    ax3.legend(fontsize=10)

    # 4. Parallel Efficiency
    ax4.plot(
        worker_counts,
        [e * 100 for e in efficiencies],
        marker="D",
        linewidth=2,
        markersize=8,
        color="#9467bd",
    )
    ax4.axhline(
        y=100, linestyle="--", linewidth=2, color="#d62728", alpha=0.7, label="Ideal (100%)"
    )
    ax4.set_xlabel("Number of Workers", fontsize=12)
    ax4.set_ylabel("Parallel Efficiency (%)", fontsize=12)
    ax4.set_title("Parallel Efficiency vs Number of Workers", fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(worker_counts)
    ax4.legend(fontsize=10)

    plt.suptitle(f"Cutana Thread Scaling Analysis ({catalogue_description})", fontsize=15, y=1.00)
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"scaling_study_{timestamp}.png"
    plt.savefig(plot_path, dpi=plot_config["dpi"], bbox_inches="tight")
    logger.info(f"Saved scaling plots to: {plot_path}")
    plt.close()
