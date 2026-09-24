#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Interpretable plots for the Cutana bottleneck profiler.

Renders, from a profiler result set, the headline views needed to read the bottleneck
*and its nature* at a glance:

- Per-scenario stacked stage bars, each stage split CPU vs stall (primary plot).
- Throughput bars, cold vs warm, grouped by band.
- Throughput-vs-workers scaling curves (when multiple worker counts are present).
- Disk MB/s vs throughput scatter (which subsystem is the ceiling).
- Per-worker Gantt charts (overlap / stragglers) from compact worker spans.

matplotlib is an optional benchmark dependency; the Agg backend keeps it headless.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
from loguru import logger  # noqa: E402

from cutana.profiling_types import COMPUTE_STAGES, Stage  # noqa: E402

# Stage order comes from the single canonical definition so plots and the profiler
# never drift apart. One distinct colour per stage keeps the dominant stage readable.
STAGE_ORDER = list(COMPUTE_STAGES)
STAGE_COLORS = {
    Stage.FITS_LOADING: "#4C72B0",
    Stage.CUTOUT_EXTRACTION: "#DD8452",
    Stage.IMAGE_RESIZING: "#55A868",
    Stage.CHANNEL_MIXING: "#C44E52",
    Stage.NORMALISATION: "#8172B3",
    Stage.METADATA_POSTPROCESSING: "#937860",
}
WORKER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]


def create_profile_plots(results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate all profiler plots into ``out_dir/plots`` (best-effort per plot)."""
    ok_results = [r for r in results if r.get("ok")]
    if not ok_results:
        logger.warning("No successful scenarios; skipping plots")
        return

    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_stage_breakdown(ok_results, plots_dir / "stage_breakdown.png")
    _plot_throughput(ok_results, plots_dir / "throughput.png")
    _plot_scaling(ok_results, plots_dir / "scaling.png")
    _plot_disk_vs_throughput(ok_results, plots_dir / "disk_vs_throughput.png")
    _plot_gantts(ok_results, plots_dir)
    logger.info(f"Wrote profiler plots to {plots_dir}")


def _plot_stage_breakdown(results: List[Dict[str, Any]], path: Path) -> None:
    """Stacked horizontal bars: per scenario, each stage coloured distinctly.

    Each stage segment is drawn in its own colour (so the dominant stage is obvious);
    within a segment the CPU part is solid and the stall part is hatched, so both the
    bottleneck stage and whether it is CPU- or stall-bound read at a glance.
    """
    fig, ax = plt.subplots(figsize=(14, max(3, 0.7 * len(results) + 2)))
    y_labels = []
    for row, result in enumerate(results):
        stages = result["stages"]
        left = 0.0
        for stage in STAGE_ORDER:
            vals = stages.get(stage, {})
            cpu = vals.get("cpu", 0.0)
            stall = vals.get("stall_time", 0.0)
            color = STAGE_COLORS[stage]
            if cpu > 0:
                ax.barh(row, cpu, left=left, color=color, edgecolor="white", linewidth=0.3)
                left += cpu
            if stall > 0:
                ax.barh(
                    row,
                    stall,
                    left=left,
                    color=color,
                    edgecolor="white",
                    linewidth=0.3,
                    hatch="////",
                    alpha=0.55,
                )
                left += stall
        y_labels.append(f"{result['name']}\n({result['throughput_median']:.0f} img/s)")

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Aggregate worker time (s) — solid = CPU, hatched = stall")
    ax.set_title("Per-stage breakdown (colour = stage, lazy-safe CPU vs stall)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[s], label=s) for s in STAGE_ORDER]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, facecolor="grey", hatch="////", alpha=0.55, label="stall")
    )
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_throughput(results: List[Dict[str, Any]], path: Path) -> None:
    """Grouped throughput bars: cold vs warm per (band, density, resolution, workers)."""
    grouped: Dict[str, Dict[str, float]] = {}
    for result in results:
        scenario = result["scenario"]
        key = f"{scenario['bands']}_{scenario['density']}_r{scenario['resolution']}_w{scenario['workers']}"
        grouped.setdefault(key, {})[scenario["cache"]] = result["throughput_median"]

    keys = sorted(grouped)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(keys) + 2), 5))
    width = 0.38
    cold = [grouped[k].get("cold", 0.0) for k in keys]
    warm = [grouped[k].get("warm", 0.0) for k in keys]
    positions = range(len(keys))
    ax.bar([p - width / 2 for p in positions], cold, width, label="cold", color="#4C72B0")
    ax.bar([p + width / 2 for p in positions], warm, width, label="warm", color="#DD8452")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title("Throughput: cold vs warm cache")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_scaling(results: List[Dict[str, Any]], path: Path) -> None:
    """Throughput vs worker count, one line per (band, density, cache, resolution)."""
    lines: Dict[str, List] = {}
    for result in results:
        scenario = result["scenario"]
        key = f"{scenario['bands']}_{scenario['density']}_{scenario['cache']}_r{scenario['resolution']}"
        lines.setdefault(key, []).append((scenario["workers"], result["throughput_median"]))

    # Only meaningful when at least one configuration spans multiple worker counts.
    if not any(len({w for w, _ in pts}) > 1 for pts in lines.values()):
        logger.info("Single worker count across scenarios; skipping scaling plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, points in sorted(lines.items()):
        points.sort()
        xs = [w for w, _ in points]
        ys = [t for _, t in points]
        ax.plot(xs, ys, marker="o", label=key)
    ax.set_xlabel("Workers")
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title("Parallel scaling")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_disk_vs_throughput(results: List[Dict[str, Any]], path: Path) -> None:
    """Scatter of measured disk MB/s against throughput, to see which is the ceiling."""
    xs, ys, labels = [], [], []
    for result in results:
        disk = result["stage_summary"].get("disk_mb_per_s")
        if disk is None:
            continue
        xs.append(disk)
        ys.append(result["throughput_median"])
        labels.append(result["name"])
    if not xs:
        logger.info("No disk MB/s measurements; skipping disk-vs-throughput plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, color="#55A868")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Pipeline disk read rate (MB/s, from read_bytes / stall)")
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title("Disk bandwidth vs throughput")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_gantts(results: List[Dict[str, Any]], plots_dir: Path) -> None:
    """Per-scenario worker Gantt charts (one PNG per scenario that reports worker spans)."""
    for result in results:
        spans = result.get("worker_spans") or []
        if not spans:
            continue
        t0 = min(s["start"] for s in spans)
        fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(spans) + 1)))
        for row, span in enumerate(sorted(spans, key=lambda s: s["start"])):
            color = WORKER_COLORS[row % len(WORKER_COLORS)]
            ax.barh(
                row, span["end"] - span["start"], left=span["start"] - t0, color=color, alpha=0.85
            )
        ax.set_yticks(range(len(spans)))
        ax.set_yticklabels([f"w{i}" for i in range(len(spans))], fontsize=7)
        ax.set_xlabel("Seconds since first worker start")
        ax.set_title(
            f"Worker timeline: {result['name']} (par-eff~ {result['parallel_efficiency_estimate']:.2f})"
        )
        fig.tight_layout()
        fig.savefig(plots_dir / f"gantt_{result['name']}.png", dpi=120)
        plt.close(fig)
