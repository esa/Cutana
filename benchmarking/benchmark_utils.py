#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Shared utilities for Cutana benchmark scripts.

This module contains common functionality used by both benchmark_q1_tiles.py
and benchmark_q1_datalabs.py to reduce code duplication.
"""

import json
import socket
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from loguru import logger
from memory_profiler import memory_usage

from cutana.orchestrator import Orchestrator


def monitor_memory_usage(func, *args, **kwargs):
    """Monitor memory usage of a function using memory_profiler."""
    result = None

    def wrapper():
        nonlocal result
        result = func(*args, **kwargs)
        return result

    # Monitor memory usage including child processes
    mem_usage = memory_usage((wrapper, ()), interval=0.5, timeout=None, include_children=True)

    return {
        "result": result,
        "memory_stats": {
            "peak_memory_mb": max(mem_usage) if mem_usage else 0,
            "avg_memory_mb": sum(mem_usage) / len(mem_usage) if mem_usage else 0,
            "memory_history": mem_usage,
            "timestamps": [i * 0.5 for i in range(len(mem_usage))],
        },
    }


def calculate_cutout_size_from_segmentation_area(
    seg_area: float, multiplier: float = 5.0, min_cutout_size: int = 32, max_cutout_size: int = None
) -> int:
    """
    Calculate cutout size in pixels from segmentation area.

    Assumes circular segmentation area and calculates diameter, then applies
    multiplier for context around the source.

    Args:
        seg_area: Segmentation area in pixels (assumed circular)
        multiplier: Multiplier for context around source
        min_cutout_size: Minimum cutout size in pixels
        max_cutout_size: Maximum cutout size in pixels (None = no limit)

    Returns:
        Cutout size in pixels (constrained to min/max limits)
    """
    # Calculate diameter assuming circular segmentation area: diameter = 2 * sqrt(area / π)
    # Then multiply by context factor
    diameter = 2 * np.sqrt(seg_area / np.pi)
    cutout_size = int(diameter * multiplier)

    # Ensure even number for centering
    if cutout_size % 2 != 0:
        cutout_size += 1

    # Apply constraints (only minimum, remove maximum to allow diameter_pixel to reflect segmentation_area)
    cutout_size = max(min_cutout_size, cutout_size)
    if max_cutout_size is not None:
        cutout_size = min(cutout_size, max_cutout_size)

    return cutout_size


def run_benchmark_core_with_orchestrator(catalogue: pd.DataFrame, config) -> Dict[str, Any]:
    """Core benchmark function that runs the orchestrator."""
    logger.info(f"Loaded catalogue with {len(catalogue)} sources")

    # Log cutout size statistics
    if "diameter_pixel" in catalogue.columns:
        sizes = catalogue["diameter_pixel"]
        logger.info(
            f"Cutout size statistics: min={sizes.min()}, max={sizes.max()}, "
            f"mean={sizes.mean():.1f}, median={sizes.median():.1f}"
        )

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator and run processing
    orchestrator = Orchestrator(config)
    results = orchestrator.start_processing(catalogue)

    return {"results": results, "config": config, "total_sources": len(catalogue)}


def run_benchmark_with_monitoring(
    catalogue: pd.DataFrame, config, benchmark_name: str = "benchmark"
) -> Dict[str, Any]:
    """
    Run the benchmark with memory and performance monitoring.

    Args:
        catalogue: Source catalogue DataFrame
        config: Configuration dictionary
        benchmark_name: Name of the benchmark for logging

    Returns:
        Benchmark results dictionary
    """
    import psutil

    hostname = socket.gethostname()
    logger.info(f"Starting {benchmark_name} with {config['max_workers']} workers on {hostname}")

    # Record start time and system state
    start_time = time.time()
    start_cpu_count = psutil.cpu_count()
    start_memory = psutil.virtual_memory()

    # Run benchmark with memory monitoring
    monitoring_result = monitor_memory_usage(
        run_benchmark_core_with_orchestrator, catalogue, config
    )

    end_time = time.time()
    total_time = end_time - start_time

    core_result = monitoring_result["result"]
    memory_stats = monitoring_result["memory_stats"]

    total_sources = core_result["total_sources"]
    orchestrator_results = core_result["results"]
    used_config = core_result["config"]

    sources_per_second = total_sources / total_time if total_time > 0 else 0

    # Handle orchestrator results - they are workflow status, not individual cutout results
    workflow_status = (
        orchestrator_results.get("status", "unknown")
        if isinstance(orchestrator_results, dict)
        else "unknown"
    )
    completed_batches = (
        orchestrator_results.get("completed_batches", 0)
        if isinstance(orchestrator_results, dict)
        else 0
    )

    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": hostname,
        "benchmark_name": benchmark_name,
        "total_sources": total_sources,
        "max_workers": config["max_workers"],
        "total_time_seconds": total_time,
        "sources_per_second": sources_per_second,
        "memory_stats": memory_stats,
        "system_info": {
            "cpu_count": start_cpu_count,
            "total_memory_gb": start_memory.total / (1024**3),
            "available_memory_gb": start_memory.available / (1024**3),
        },
        "config": used_config,
        "results_summary": {
            "workflow_status": workflow_status,
            "completed_batches": completed_batches,
            "expected_batches": config["max_workers"],
        },
    }

    logger.info("Benchmark Results:")
    logger.info(f"  Total time: {total_time:.2f} seconds")
    logger.info(f"  Sources per second: {sources_per_second:.2f}")
    logger.info(f"  Peak memory usage: {memory_stats['peak_memory_mb']:.1f} MB")
    logger.info(f"  Average memory usage: {memory_stats['avg_memory_mb']:.1f} MB")

    return benchmark_results


def collect_performance_statistics(output_dir: Path) -> Dict[str, Any]:
    """
    Collect performance statistics from subprocess logs.

    Args:
        output_dir: Directory where subprocess logs are stored

    Returns:
        Dictionary containing aggregated performance statistics
    """
    try:
        log_dir = output_dir / "logs" / "subprocesses"
        if not log_dir.exists():
            logger.warning(f"Subprocess log directory not found: {log_dir}")
            return {}

        # Find all stderr log files (where performance profiler outputs)
        stderr_files = glob(str(log_dir / "*_stderr.log"))

        if not stderr_files:
            logger.warning("No subprocess stderr log files found for performance analysis")
            return {}

        # Check for LoadBalancer event log file
        event_log_path = output_dir / "logs" / "loadbalancer_events.jsonl"

        # Aggregate performance statistics with granular profiling
        aggregate_stats = {
            "total_processes": len(stderr_files),
            "steps": {
                "FitsLoading": {"times": [], "count": 0},
                "CutoutExtraction": {"times": [], "count": 0},
                "ImageResizing": {"times": [], "count": 0},
                "ChannelMixing": {"times": [], "count": 0},
                "Normalisation": {"times": [], "count": 0},
                "DataTypeConversion": {"times": [], "count": 0},
                "MetaDataPostprocessing": {"times": [], "count": 0},
                "ZarrFitsSaving": {"times": [], "count": 0},
            },
            "total_sources_processed": 0,
            "total_runtime": 0.0,
            "loadbalancer_events": [],  # Track LoadBalancer events
        }

        # Read LoadBalancer events from dedicated event log file
        if event_log_path.exists():
            logger.info(f"Reading LoadBalancer events from: {event_log_path}")
            try:
                with open(event_log_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            # Convert LoadBalancer events to format expected by charts
                            converted_event = {
                                "type": f"{event.get('category', 'unknown')}_{event.get('event_type', 'unknown')}",
                                "timestamp": event.get("timestamp"),
                                "data": event.get("data", {}),
                                "category": event.get("category", "unknown"),
                                "event_type": event.get("event_type", "unknown"),
                            }

                            # Convert spawn decision events to expected format
                            if (
                                event.get("category") == "Spawn"
                                and event.get("event_type") == "decision"
                            ):
                                converted_event["type"] = "spawn_decision"
                                converted_event["allowed"] = event.get("data", {}).get(
                                    "can_spawn", False
                                )
                                converted_event["message"] = event.get("data", {}).get("reason", "")

                            # Convert memory status events to expected format
                            elif (
                                event.get("category") == "Memory"
                                and event.get("event_type") == "status"
                            ):
                                converted_event["type"] = "memory_status"
                                data = event.get("data", {})
                                converted_event["main_process_mb"] = data.get(
                                    "main_process_memory_mb", 0
                                )
                                converted_event["worker_allocation_mb"] = data.get(
                                    "worker_allocation_mb", 0
                                )
                                converted_event["worker_peak_mb"] = data.get("worker_peak_mb", 0)
                                converted_event["message"] = (
                                    f"Memory Status - Main: {converted_event['main_process_mb']:.1f}MB, Worker Alloc: {converted_event['worker_allocation_mb']:.1f}MB, Worker Peak: {converted_event['worker_peak_mb']:.1f}MB"
                                )

                            aggregate_stats["loadbalancer_events"].append(converted_event)

                        except json.JSONDecodeError as e:
                            logger.debug(f"Failed to parse event JSON on line {line_num}: {e}")
                            continue

                logger.info(
                    f"Loaded {len(aggregate_stats['loadbalancer_events'])} LoadBalancer events"
                )
            except Exception as e:
                logger.warning(f"Failed to read LoadBalancer event log: {e}")
        else:
            logger.info(f"No LoadBalancer event log found at: {event_log_path}")

        # Parse each log file for performance data
        for stderr_file in stderr_files:
            try:
                with open(stderr_file, "r") as f:
                    for line in f:
                        # Look for structured performance data
                        if "PERFORMANCE_DATA:" in line:
                            try:
                                # Extract JSON from the line
                                json_str = line.split("PERFORMANCE_DATA:", 1)[1].strip()
                                perf_data = json.loads(json_str)
                                if perf_data.get("type") == "performance_summary":
                                    # Aggregate performance data
                                    steps_data = perf_data.get("steps", {})
                                    for step_name, step_data in steps_data.items():
                                        if step_name in aggregate_stats["steps"]:
                                            total_time = step_data.get("total_time", 0)
                                            count = step_data.get("count", 0)
                                            if total_time > 0 and count > 0:
                                                aggregate_stats["steps"][step_name]["times"].append(
                                                    total_time
                                                )
                                                aggregate_stats["steps"][step_name][
                                                    "count"
                                                ] += count

                                    aggregate_stats["total_sources_processed"] += perf_data.get(
                                        "total_sources", 0
                                    )
                                    aggregate_stats["total_runtime"] += perf_data.get(
                                        "total_runtime", 0
                                    )
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.debug(f"Error parsing performance JSON: {e}")
                                continue

            except Exception as e:
                logger.debug(f"Error parsing subprocess log {stderr_file}: {e}")
                continue

        return aggregate_stats

    except Exception as e:
        logger.error(f"Failed to collect performance statistics: {e}")
        return {}


def create_performance_charts(
    results: Dict[str, Any], perf_stats: Dict[str, Any], output_dir: Path, chart_suffix: str = ""
) -> None:
    """
    Create performance analysis charts.

    Args:
        results: Benchmark results
        perf_stats: Performance statistics
        output_dir: Directory to save charts
        chart_suffix: Suffix to add to chart filename
    """
    try:
        timestamp = int(time.time())

        # Create a figure with multiple subplots (now 6 plots for LoadBalancer info)
        fig = plt.figure(figsize=(20, 16))

        # Create grid: 3 rows, 2 columns
        ax1 = plt.subplot(3, 2, 1)
        ax2 = plt.subplot(3, 2, 2)
        ax3 = plt.subplot(3, 2, 3)
        ax4 = plt.subplot(3, 2, 4)
        ax5 = plt.subplot(3, 2, 5)  # LoadBalancer memory tracking
        ax6 = plt.subplot(3, 2, 6)  # LoadBalancer spawn decisions

        # 1. Memory Usage Over Time
        memory_stats = results.get("memory_stats", {})
        if memory_stats.get("memory_history") and memory_stats.get("timestamps"):
            start_time = memory_stats["timestamps"][0]
            relative_times = [(t - start_time) / 60 for t in memory_stats["timestamps"]]

            ax1.plot(relative_times, memory_stats["memory_history"], linewidth=2, color="blue")
            ax1.set_title("Memory Usage During Benchmark")
            ax1.set_xlabel("Time (minutes)")
            ax1.set_ylabel("Memory Usage (MB)")
            ax1.grid(True, alpha=0.3)

        # 2. Processing Steps Time Distribution (Pie Chart)
        step_names = []
        step_times = []
        colors = [
            "#ff9999",
            "#66b3ff",
            "#99ff99",
            "#ffcc99",
            "#ff99cc",
            "#99ffff",
            "#ffff99",
            "#cc99ff",
        ]

        if perf_stats.get("steps"):
            for step, data in perf_stats["steps"].items():
                if data.get("times"):
                    # Format step names for better readability
                    formatted_name = (
                        step.replace("FitsLoading", "FITS Loading")
                        .replace("CutoutExtraction", "Cutout Extraction")
                        .replace("ImageResizing", "Image Resizing")
                        .replace("ChannelMixing", "Channel Mixing")
                        .replace("Normalisation", "Normalization")
                        .replace("DataTypeConversion", "Data Type Conversion")
                        .replace("MetaDataPostprocessing", "Metadata Processing")
                        .replace("ZarrFitsSaving", "Output Writing")
                    )
                    step_names.append(formatted_name)
                    step_times.append(sum(data["times"]))

        if step_times:
            ax2.pie(
                step_times,
                labels=step_names,
                autopct="%1.1f%%",
                colors=colors[: len(step_times)],
                startangle=90,
            )
            ax2.set_title("Time Distribution by Processing Step")
        else:
            # Create a placeholder chart if no performance data
            ax2.text(
                0.5,
                0.5,
                "No performance data\navailable",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Time Distribution by Processing Step")

        # 3. Performance Metrics Bar Chart with Error Bars
        if perf_stats.get("steps"):
            steps = []
            avg_times = []
            std_devs = []

            for step, data in perf_stats["steps"].items():
                if data.get("times"):
                    # Format step names for consistency
                    formatted_name = (
                        step.replace("FitsLoading", "FITS Loading")
                        .replace("CutoutExtraction", "Cutout Extraction")
                        .replace("ImageResizing", "Image Resizing")
                        .replace("ChannelMixing", "Channel Mixing")
                        .replace("Normalisation", "Normalization")
                        .replace("DataTypeConversion", "Data Type Conversion")
                        .replace("MetaDataPostprocessing", "Metadata Processing")
                        .replace("ZarrFitsSaving", "Output Writing")
                    )

                    steps.append(formatted_name)
                    times = data["times"]
                    avg_time = sum(times) / len(times) if times else 0
                    avg_times.append(avg_time)

                    # Calculate standard deviation for error bars
                    if len(times) > 1:
                        mean = avg_time
                        variance = sum((x - mean) ** 2 for x in times) / len(times)
                        std_dev = variance**0.5
                    else:
                        std_dev = 0
                    std_devs.append(std_dev)

            if steps:
                bars = ax3.bar(
                    steps,
                    avg_times,
                    yerr=std_devs,
                    capsize=5,
                    color=colors[: len(steps)],
                    alpha=0.7,
                    error_kw={"elinewidth": 2, "capthick": 2},
                )
                ax3.set_title("Average Time per Processing Step (with Error Bars)")
                ax3.set_ylabel("Time (seconds)")
                ax3.set_xticklabels(steps, rotation=45, ha="right")

                # Add value labels on bars
                for bar, value, std_dev in zip(bars, avg_times, std_devs):
                    label_text = f"{value:.2f}s"
                    if std_dev > 0:
                        label_text += f"±{std_dev:.2f}"
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + std_dev + 0.01,
                        label_text,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        # 4. Performance Efficiency Metrics
        system_info = results.get("system_info", {})
        max_workers = results.get("max_workers", 1)
        total_time = results.get("total_time_seconds", 1)
        total_sources = results.get("total_sources", 0)

        # Calculate meaningful performance metrics
        sources_per_sec = total_sources / total_time if total_time > 0 else 0
        peak_memory_gb = memory_stats.get("peak_memory_mb", 0) / 1024
        avg_memory_gb = memory_stats.get("avg_memory_mb", 0) / 1024
        memory_efficiency = (avg_memory_gb / peak_memory_gb * 100) if peak_memory_gb > 0 else 0
        cpu_cores = system_info.get("cpu_count", 1)
        worker_efficiency = (sources_per_sec / max_workers) if max_workers > 0 else 0
        memory_per_source = (avg_memory_gb * 1024 / total_sources) if total_sources > 0 else 0

        # Create a more meaningful performance metrics chart
        efficiency_metrics = {
            "Memory Efficiency (%)": memory_efficiency,
            "Sources/Worker/sec": worker_efficiency,
            "Memory/Source (MB)": memory_per_source,
            "CPU Utilization (%)": (max_workers / cpu_cores * 100) if cpu_cores > 0 else 0,
        }

        bars = ax4.bar(
            range(len(efficiency_metrics)),
            list(efficiency_metrics.values()),
            color=["#2E8B57", "#4682B4", "#CD853F", "#9932CC"],
        )

        # Create informative title with key metrics
        title = f"Performance Metrics ({max_workers} workers, {sources_per_sec:.1f} sources/sec)"
        ax4.set_title(title)
        ax4.set_xticks(range(len(efficiency_metrics)))
        ax4.set_xticklabels(list(efficiency_metrics.keys()), rotation=45, ha="right")

        # Add value labels on bars
        for bar, key, value in zip(bars, efficiency_metrics.keys(), efficiency_metrics.values()):
            if "%" in key:
                label = f"{value:.1f}%"
            elif "MB" in key:
                label = f"{value:.2f}"
            else:
                label = f"{value:.2f}"
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(efficiency_metrics.values()) * 0.01,
                label,
                ha="center",
                va="bottom",
            )

        # 5. LoadBalancer Memory Tracking
        loadbalancer_events = perf_stats.get("loadbalancer_events", [])
        memory_events = [e for e in loadbalancer_events if e.get("type") == "memory_status"]

        if memory_events:
            # Extract memory tracking data
            main_process_mb = []
            worker_alloc_mb = []
            worker_peak_mb = []
            event_times = []

            for i, event in enumerate(memory_events):
                event_times.append(i)  # Use index as proxy for time
                main_process_mb.append(event.get("main_process_mb", 0))
                worker_alloc_mb.append(event.get("worker_allocation_mb", 0))
                worker_peak_mb.append(event.get("worker_peak_mb", 0))

            # Plot memory tracking
            ax5.plot(
                event_times,
                main_process_mb,
                label="Main Process",
                linewidth=2,
                color="blue",
                marker="o",
            )
            ax5.plot(
                event_times,
                worker_alloc_mb,
                label="Worker Allocation",
                linewidth=2,
                color="green",
                marker="s",
            )
            ax5.plot(
                event_times,
                worker_peak_mb,
                label="Worker Peak",
                linewidth=2,
                color="red",
                marker="^",
            )

            ax5.set_title("LoadBalancer Memory Tracking")
            ax5.set_xlabel("Time (event index)")
            ax5.set_ylabel("Memory (MB)")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(
                0.5,
                0.5,
                "No LoadBalancer memory\nevents recorded",
                ha="center",
                va="center",
                transform=ax5.transAxes,
            )
            ax5.set_title("LoadBalancer Memory Tracking")

        # 6. LoadBalancer Spawn Decisions
        spawn_events = [e for e in loadbalancer_events if e.get("type") == "spawn_decision"]

        if spawn_events:
            allowed_count = sum(1 for e in spawn_events if e.get("allowed"))
            denied_count = len(spawn_events) - allowed_count

            # Create pie chart of spawn decisions
            sizes = [allowed_count, denied_count]
            labels = [f"Allowed ({allowed_count})", f"Denied ({denied_count})"]
            colors = ["#2E8B57", "#DC143C"]

            wedges, texts, autotexts = ax6.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )
            ax6.set_title("LoadBalancer Spawn Decisions")

            # Add decision reasons as text below
            denied_reasons = {}
            for event in spawn_events:
                if not event.get("allowed"):
                    msg = event.get("message", "")
                    # Extract reason from message
                    if "CPU limit" in msg:
                        reason = "CPU limit"
                    elif "memory" in msg.lower():
                        reason = "Memory constraint"
                    elif "CPU usage" in msg:
                        reason = "High CPU usage"
                    else:
                        reason = "Other"
                    denied_reasons[reason] = denied_reasons.get(reason, 0) + 1

            if denied_reasons:
                reason_text = "Denial reasons:\n" + "\n".join(
                    [f"• {r}: {c}" for r, c in denied_reasons.items()]
                )
                ax6.text(
                    0.5,
                    -0.2,
                    reason_text,
                    ha="center",
                    va="top",
                    transform=ax6.transAxes,
                    fontsize=10,
                )
        else:
            ax6.text(
                0.5,
                0.5,
                "No spawn decisions\nrecorded",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )
            ax6.set_title("LoadBalancer Spawn Decisions")

        plt.tight_layout()

        # Save the comprehensive performance chart
        chart_filename = (
            f"performance_analysis_{chart_suffix}_{timestamp}.png"
            if chart_suffix
            else f"performance_analysis_{timestamp}.png"
        )
        chart_path = output_dir / chart_filename
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved performance analysis charts to: {chart_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Failed to create performance charts: {e}")


def save_benchmark_results(
    results: Dict[str, Any], output_dir: Path, filename_suffix: str = ""
) -> None:
    """
    Save benchmark results to JSON and create comprehensive analysis plots.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save results
        filename_suffix: Suffix to add to filenames
    """
    # Save JSON results
    timestamp = int(time.time())
    json_filename = (
        f"benchmark_results_{filename_suffix}_{timestamp}.json"
        if filename_suffix
        else f"benchmark_results_{timestamp}.json"
    )
    json_path = output_dir / json_filename

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved benchmark results to: {json_path}")

    # Collect performance statistics from subprocess logs
    output_config_dir = Path(results["config"]["output_dir"])

    # Handle relative paths - make them absolute from current working directory
    if not output_config_dir.is_absolute():
        output_config_dir = Path.cwd() / output_config_dir

    logger.info(f"Looking for performance data in: {output_config_dir}")
    perf_stats = collect_performance_statistics(output_config_dir)

    # Create comprehensive performance charts
    create_performance_charts(results, perf_stats, output_dir, filename_suffix)


def read_optimized_catalog(
    catalog_path: str,
    min_seg_area: float = 10.0,
    max_sources: int = None,
    cutout_size_multiplier: float = 3.0,
    min_cutout_size: int = 32,
    max_cutout_size: int = None,
) -> pd.DataFrame:
    """
    Read catalog with memory optimization and pre-filtering.

    Applies performance optimizations from benchmark_q1_tiles.py:
    - Memory-mapped FITS loading with lazy HDU loading
    - Pre-filtering by segmentation area
    - Sort by segmentation area (largest first)
    - Early termination when max_sources reached

    Args:
        catalog_path: Path to FITS catalog file
        min_seg_area: Minimum segmentation area in pixels to include
        max_sources: Maximum number of sources to return (None = all)
        cutout_size_multiplier: Multiplier for segmentation area to cutout size
        min_cutout_size: Minimum cutout size in pixels
        max_cutout_size: Maximum cutout size in pixels (None = no limit)

    Returns:
        DataFrame with optimized source list
    """
    logger.info(f"Reading catalog with optimizations: {catalog_path}")

    if not Path(catalog_path).exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    # Use memory-mapped FITS loading for large catalogs - optimized for NFS
    with fits.open(catalog_path, memmap=True, lazy_load_hdus=True) as hdul:
        catalog_data = hdul[1].data

        # Get total number of sources in this catalog
        total_sources_in_catalog = len(catalog_data)
        logger.info(
            f"Catalog {Path(catalog_path).name} contains {total_sources_in_catalog} sources"
        )

        # Get required columns (these are memory-mapped, so efficient)
        object_ids = catalog_data["OBJECT_ID"]
        ras = catalog_data["RIGHT_ASCENSION"]
        decs = catalog_data["DECLINATION"]
        seg_areas = catalog_data["SEGMENTATION_AREA"]

        # Pre-filter by segmentation area to avoid processing tiny sources
        logger.info(f"Pre-filtering sources by segmentation area >= {min_seg_area} pixels")

        sources = []
        for i in range(len(object_ids)):
            # Skip tiny sources early to save processing time
            if seg_areas[i] < min_seg_area:
                continue

            # Calculate cutout size from segmentation area
            cutout_diameter_pixels = calculate_cutout_size_from_segmentation_area(
                seg_areas[i], cutout_size_multiplier, min_cutout_size, max_cutout_size
            )

            source = {
                "OBJECT_ID": int(object_ids[i]),
                "RA": float(ras[i]),
                "Dec": float(decs[i]),
                "diameter_pixel": cutout_diameter_pixels,
                "segmentation_area": float(seg_areas[i]),
            }
            sources.append(source)

        logger.info(
            f"Found {len(sources)} sources after pre-filtering (from {total_sources_in_catalog} total)"
        )

    # Sort by segmentation area (largest first)
    sources.sort(key=lambda x: x["segmentation_area"], reverse=True)

    # Apply max_sources limit if specified
    if max_sources and len(sources) > max_sources:
        sources = sources[:max_sources]
        logger.info(f"Limited to {max_sources} largest sources by segmentation area")

    # Log segmentation area range
    if sources:
        logger.info(
            f"Segmentation area range: {sources[-1]['segmentation_area']:.1f} - {sources[0]['segmentation_area']:.1f} pixels"
        )

    return pd.DataFrame(sources)


def create_fits_path_cache(
    data_dir: Path, catalog_tiles: Dict[str, str], extensions: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    Create FITS file path cache to avoid repeated glob operations.

    Args:
        data_dir: Directory containing FITS files
        catalog_tiles: Dictionary mapping catalog filenames to tile IDs
        extensions: List of extensions to cache (e.g., ["VIS", "NIR-H"])

    Returns:
        Dictionary mapping tile_id -> extension -> file_path
    """
    logger.info("Pre-building FITS file path cache for all tiles...")
    fits_path_cache = {}

    for cat_file, tile_id in catalog_tiles.items():
        fits_path_cache[tile_id] = {}
        for extension in extensions:
            pattern = f"EUC_MER_BGSUB-MOSAIC-{extension}_TILE{tile_id}-*.fits"
            matching_files = list(data_dir.glob(pattern))
            if matching_files:
                fits_path_cache[tile_id][extension] = str(matching_files[0])

    logger.info(f"Cached FITS paths for {len(fits_path_cache)} tiles")
    return fits_path_cache
