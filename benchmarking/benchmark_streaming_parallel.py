#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Benchmark for StreamingOrchestrator: serial baseline and parallel comparison.

Produces cutouts from the NIR-H/J/Y benchmark catalogue using the streaming
orchestrator in in-memory mode, measuring next_batch() latency and throughput.

Usage:
    python benchmarking/benchmark_streaming_parallel.py [--max-sources N] [--batch-size N]
                                                        [--max-workers N [N ...]] [--min-workers N]

Charts saved to benchmarking/results/streaming_parallel/
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from cutana.get_default_config import get_default_config
from cutana.logging_config import setup_logging
from cutana.performance_profiler import ContextProfiler, PerformanceProfiler
from cutana.streaming_orchestrator import StreamingOrchestrator

# === Configuration ===
CATALOGUE_PATH = str(
    Path(__file__).parent.parent / "data" / "benchmark_input_10k_cutouts_nirh_nirj_niry.csv"
)
USE_EXTENSIONS = ["NIR-H", "NIR-J", "NIR-Y"]
CHANNEL_MATRIX = [
    [1.0, 0.0, 0.0],  # Channel 1: NIR-H
    [0.0, 1.0, 0.0],  # Channel 2: NIR-J
    [0.0, 0.0, 1.0],  # Channel 3: NIR-Y
]
TARGET_RESOLUTION = 128
DATA_TYPE = "float32"
OUTPUT_FORMAT = "zarr"
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_SOURCES = 10000

# Distinct colors for workers in Gantt chart
WORKER_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#E68A8E"]


def get_benchmark_config(output_dir: str):
    """Build config for streaming benchmark."""
    config = get_default_config()

    config.source_catalogue = CATALOGUE_PATH
    config.max_workers = 4
    config.N_batch_cutout_process = 1000
    config.output_format = OUTPUT_FORMAT
    config.output_dir = output_dir
    config.fits_extensions = ["PRIMARY"]
    config.target_resolution = TARGET_RESOLUTION
    config.data_type = DATA_TYPE
    config.normalisation_method = "linear"
    config.interpolation = "bilinear"
    config.log_level = "INFO"
    config.write_to_disk = False

    # Multi-channel config
    config.channel_weights = {}
    for i, ext in enumerate(USE_EXTENSIONS):
        weights = [CHANNEL_MATRIX[j][i] for j in range(len(CHANNEL_MATRIX))]
        config.channel_weights[ext] = weights

    config.apply_flux_conversion = True
    config.selected_extensions = [{"name": ext, "ext": "PRIMARY"} for ext in USE_EXTENSIONS]

    # Skip memory calibration wait for faster benchmarking
    config.loadbalancer.skip_memory_calibration_wait = True

    return config


def run_benchmark(config, batch_size: int, max_sources: int, max_workers: int, min_workers: int):
    """Run streaming benchmark with given number of workers."""
    label = f"max_workers={max_workers},min_workers={min_workers}"
    logger.info(f"=== Benchmark ({label}): batch_size={batch_size}, max_sources={max_sources} ===")

    profiler = PerformanceProfiler(f"streaming_{label}")
    orchestrator = StreamingOrchestrator(config)

    with ContextProfiler(profiler, "init_streaming"):
        orchestrator.init_streaming(
            batch_size=batch_size,
            write_to_disk=False,
            max_workers=max_workers,
            min_workers=min_workers,
        )

    total_batches = orchestrator.get_batch_count()
    logger.info(f"Total batches: {total_batches}")

    batch_times = []  # (batch_idx, wall_start, wall_end, n_cutouts)
    total_cutouts = 0
    wall_start = time.perf_counter()

    for i in range(total_batches):
        if total_cutouts >= max_sources:
            break

        t0 = time.perf_counter()
        with ContextProfiler(profiler, "next_batch") as cp:
            result = orchestrator.next_batch()
        t1 = time.perf_counter()

        n_cutouts = len(result["cutouts"]) if result["cutouts"] is not None else 0
        total_cutouts += n_cutouts
        batch_times.append((i, t0 - wall_start, t1 - wall_start, n_cutouts))

        logger.info(
            f"[{label}] Batch {i + 1}/{total_batches}: {n_cutouts} cutouts, "
            f"latency={cp.duration:.2f}s, total={total_cutouts}"
        )

    wall_end = time.perf_counter()
    total_wall_time = wall_end - wall_start

    # Get worker events before cleanup
    worker_events = orchestrator.get_worker_events()

    orchestrator.cleanup()

    profiler._total_sources = total_cutouts
    profiler.log_performance_summary()

    throughput = total_cutouts / total_wall_time if total_wall_time > 0 else 0
    logger.info(
        f"\n=== Results ({label}) ===\n"
        f"  Total cutouts: {total_cutouts}\n"
        f"  Wall time: {total_wall_time:.2f}s\n"
        f"  Throughput: {throughput:.1f} cutouts/s\n"
        f"  Batches completed: {len(batch_times)}\n"
        f"  Avg batch latency: {np.mean([bt[2] - bt[1] for bt in batch_times]):.2f}s"
    )

    return {
        "max_workers": max_workers,
        "min_workers": min_workers,
        "batch_times": batch_times,
        "worker_events": worker_events,
        "total_cutouts": total_cutouts,
        "total_wall_time": total_wall_time,
        "throughput": throughput,
        "profiler_stats": profiler.get_statistics(),
    }


def create_comparison_charts(all_results: list, output_dir: Path):
    """Generate comparison charts across different worker counts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_configs = len(all_results)
    fig, axes = plt.subplots(
        n_configs + 1, 1, figsize=(16, 5 * (n_configs + 1)), constrained_layout=True
    )
    if n_configs + 1 == 1:
        axes = [axes]

    fig.suptitle("Streaming Orchestrator: Serial vs Parallel", fontsize=14)

    # Per-config: Gantt chart of worker events
    for idx, results in enumerate(all_results):
        ax = axes[idx]
        worker_events = results["worker_events"]
        max_workers = results["max_workers"]
        throughput = results["throughput"]
        wall_time = results["total_wall_time"]

        if worker_events:
            # Group events by worker
            worker_spans = {}
            for worker_id, event, ts in worker_events:
                if worker_id not in worker_spans:
                    worker_spans[worker_id] = {}
                worker_spans[worker_id][event] = ts

            # Determine time offset (first event)
            t0 = min(ts for _, _, ts in worker_events)

            # Assign visual row per unique worker
            worker_ids = sorted(worker_spans.keys())
            worker_row = {wid: i for i, wid in enumerate(worker_ids)}

            for wid, events in worker_spans.items():
                if "start" in events and "end" in events:
                    start = events["start"] - t0
                    duration = events["end"] - events["start"]
                    row = worker_row[wid]
                    color = WORKER_COLORS[row % len(WORKER_COLORS)]
                    ax.barh(
                        row,
                        duration,
                        left=start,
                        height=0.6,
                        color=color,
                        edgecolor="white",
                        linewidth=0.5,
                        alpha=0.85,
                    )

            ax.set_yticks(range(len(worker_ids)))
            ax.set_yticklabels([f"Worker {i}" for i in range(len(worker_ids))])
        else:
            # No worker events (shouldn't happen), show batch times
            batch_times = results["batch_times"]
            for bt in batch_times:
                _, start, end, _ = bt
                ax.barh(
                    0,
                    end - start,
                    left=start,
                    height=0.4,
                    color="#C44E52",
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.8,
                )
            ax.set_yticks([0])
            ax.set_yticklabels(["Worker 0"])

        ax.set_xlabel("Time (seconds)")
        ax.set_title(
            f"max_workers={max_workers}, min_workers={results['min_workers']} — "
            f"{results['total_cutouts']} cutouts in {wall_time:.1f}s "
            f"({throughput:.0f} cutouts/s)"
        )
        ax.grid(axis="x", alpha=0.3)

    # Final subplot: throughput comparison
    ax = axes[-1]
    worker_counts = [r["max_workers"] for r in all_results]
    throughputs = [r["throughput"] for r in all_results]
    wall_times = [r["total_wall_time"] for r in all_results]

    bar_width = 0.35
    x = np.arange(len(worker_counts))

    bars1 = ax.bar(x - bar_width / 2, throughputs, bar_width, color="#4C72B0", label="Throughput")
    ax2 = ax.twinx()
    bars2 = ax2.bar(
        x + bar_width / 2, wall_times, bar_width, color="#C44E52", alpha=0.7, label="Wall time"
    )

    ax.set_xlabel("Max workers")
    ax.set_ylabel("Throughput (cutouts/s)", color="#4C72B0")
    ax2.set_ylabel("Wall time (s)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in worker_counts])
    ax.set_title("Throughput and wall time comparison")

    # Add value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#C44E52",
        )

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    chart_path = output_dir / "streaming_parallel_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    logger.info(f"Comparison charts saved to {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="StreamingOrchestrator parallel benchmark")
    parser.add_argument(
        "--max-sources",
        type=int,
        default=DEFAULT_MAX_SOURCES,
        help=f"Max sources to process (default: {DEFAULT_MAX_SOURCES})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"User-facing batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Worker counts to benchmark (default: 1 2 4)",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Workers pre-spawned at init time (default: 1)",
    )
    args = parser.parse_args()

    setup_logging(log_level="INFO", console_level="INFO")

    results_dir = Path(__file__).parent / "results" / "streaming_parallel"
    output_dir = str(results_dir / "output")

    config = get_benchmark_config(output_dir)

    logger.info(f"Catalogue: {CATALOGUE_PATH}")
    logger.info(f"Max sources: {args.max_sources}, Batch size: {args.batch_size}")
    logger.info(f"Worker configs: {args.max_workers}, Min workers: {args.min_workers}")
    logger.info(
        f"Resolution: {TARGET_RESOLUTION}, Dtype: {DATA_TYPE}, Channels: {len(USE_EXTENSIONS)}"
    )

    all_results = []
    for max_workers in args.max_workers:
        results = run_benchmark(
            config, args.batch_size, args.max_sources, max_workers, args.min_workers
        )
        all_results.append(results)

    create_comparison_charts(all_results, results_dir)

    # Print summary table
    logger.info("\n=== Summary ===")
    logger.info(
        f"{'Max W':<8} {'Min W':<8} {'Cutouts':<10} {'Wall time':<12} {'Throughput':<15} {'Speedup':<10}"
    )
    baseline_time = all_results[0]["total_wall_time"]
    for r in all_results:
        speedup = baseline_time / r["total_wall_time"] if r["total_wall_time"] > 0 else 0
        logger.info(
            f"{r['max_workers']:<8} {r['min_workers']:<8} {r['total_cutouts']:<10} "
            f"{r['total_wall_time']:<12.2f} {r['throughput']:<15.1f} {speedup:<10.2f}x"
        )


if __name__ == "__main__":
    main()
