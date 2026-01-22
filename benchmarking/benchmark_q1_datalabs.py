#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Benchmark script for Cutana Q1 datalabs performance evaluation.

This script uses the Q1_data_layout.csv to select specific tiles and create cutouts
for benchmarking on datalabs infrastructure. Supports configurable output paths
and FITS extensions.

Usage:
    python benchmarking/benchmark_q1_datalabs.py --tiles 102018666,102018668 \\
        --output-dir /path/to/output
"""

import argparse
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from astropy.io import fits
from loguru import logger

from cutana.get_default_config import get_default_config
from cutana.logging_config import setup_logging

try:
    # Try relative import first (when used as module)
    from .benchmark_utils import (
        calculate_cutout_size_from_segmentation_area,
        run_benchmark_with_monitoring,
        save_benchmark_results,
    )
except ImportError:
    # Fall back to absolute import (when run directly)
    sys.path.append(str(Path(__file__).parent))
    from benchmark_utils import (
        calculate_cutout_size_from_segmentation_area,
        run_benchmark_with_monitoring,
        save_benchmark_results,
    )

# Default configuration parameters
DEFAULT_MAX_WORKERS = 4
DEFAULT_CUTOUT_SIZE_MULTIPLIER = 5.0
DEFAULT_MIN_CUTOUT_SIZE = 32
DEFAULT_MAX_CUTOUT_SIZE = 512
DEFAULT_OUTPUT_FORMAT = "zarr"
DEFAULT_TARGET_RESOLUTION = 150
DEFAULT_OUTPUT_DIR = "/media/team_workspaces/AnomalyMatch/Q1_experiments/"
DEFAULT_N_TILES = 4
DEFAULT_SOURCES_PER_TILE = 5000
N_BATCH_CUTOUT_PROCESS = 1000


# monitor_memory_usage function moved to benchmark_utils.py


# calculate_cutout_size_from_segmentation_area function moved to benchmark_utils.py


def load_data_layout(layout_csv_path: Path) -> pd.DataFrame:
    """Load Q1_data_layout.csv and return as DataFrame."""
    try:
        df = pd.read_csv(layout_csv_path)
        logger.info(f"Loaded data layout with {len(df)} tiles from {layout_csv_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data layout from {layout_csv_path}: {e}")
        raise


def select_tiles(
    data_layout: pd.DataFrame, tile_ids: List[str], n_tiles: int = None
) -> pd.DataFrame:
    """Select specific tiles from data layout."""
    if tile_ids:
        # Filter to specific tiles
        selected = data_layout[data_layout["tileID"].isin(tile_ids)].copy()

        missing_tiles = set(tile_ids) - set(selected["tileID"].tolist())
        if missing_tiles:
            logger.warning(f"Requested tiles not found in layout: {missing_tiles}")

        logger.info(f"Selected {len(selected)} specific tiles")
        return selected

    elif n_tiles:
        # Random selection of N tiles
        if n_tiles >= len(data_layout):
            logger.info(f"Requested {n_tiles} tiles, using all {len(data_layout)} available")
            return data_layout

        selected = data_layout.sample(n=n_tiles, random_state=42).copy()
        logger.info(f"Randomly selected {len(selected)} tiles from {len(data_layout)} available")
        return selected

    else:
        logger.info("No specific selection criteria, using all available tiles")
        return data_layout


def create_benchmark_input_from_tiles(
    selected_tiles: pd.DataFrame,
    sources_per_tile: int = 1000,
    fits_channels: List[str] = ["NIR_H_path", "NIR_J_path", "NIR_Y_path"],
) -> pd.DataFrame:
    """
    Create benchmark input DataFrame from selected tiles.

    Args:
        selected_tiles: DataFrame with tile information
        sources_per_tile: Maximum sources to extract per tile
        fits_channels: List of FITS channels to include (default: all NIR channels)

    Returns:
        DataFrame with source information for benchmarking
    """
    all_sources = []

    for _, tile_row in selected_tiles.iterrows():
        tile_id = tile_row["tileID"]
        catalog_path = tile_row["catalogue_path"]

        # Select image paths based on requested channels
        image_paths = []
        for channel in fits_channels:
            if tile_row[channel] and Path(tile_row[channel]).exists():
                image_paths.append(tile_row[channel])

        if not image_paths:
            logger.warning(
                f"No valid image files found for tile {tile_id} in channels {fits_channels}"
            )
            continue

        logger.info(
            f"Processing tile {tile_id}: catalog={Path(catalog_path).name}, "
            f"channels={[Path(p).name for p in image_paths]}"
        )

        if not Path(catalog_path).exists():
            logger.error(f"Catalog file not found: {catalog_path}")
            continue

        # Verify all image files exist
        missing_files = [p for p in image_paths if not Path(p).exists()]
        if missing_files:
            logger.error(f"Image files not found for tile {tile_id}: {missing_files}")
            continue

        try:
            # Use optimized catalog reading with memory mapping and pre-filtering
            with fits.open(catalog_path, memmap=True, lazy_load_hdus=True) as hdul:
                catalog_data = hdul[1].data

                # Get required columns (these are memory-mapped, so efficient)
                object_ids = catalog_data["OBJECT_ID"]
                ras = catalog_data["RIGHT_ASCENSION"]
                decs = catalog_data["DECLINATION"]
                seg_areas = catalog_data["SEGMENTATION_AREA"]

                # Pre-filter by segmentation area to avoid processing tiny sources
                min_seg_area = 10.0  # Skip very small sources (< 10 pixels)
                logger.info(f"Pre-filtering sources by segmentation area >= {min_seg_area} pixels")

                # Collect valid sources with pre-filtering
                valid_sources = []
                for i in range(len(object_ids)):
                    # Skip tiny sources early to save processing time
                    if seg_areas[i] < min_seg_area:
                        continue
                    valid_sources.append((i, seg_areas[i]))

                # Sort by segmentation area (largest first) and take top sources
                valid_sources.sort(key=lambda x: x[1], reverse=True)

                # Limit sources per tile, taking the largest ones
                n_sources = min(len(valid_sources), sources_per_tile)
                selected_sources = valid_sources[:n_sources]

                logger.info(
                    f"Selected {len(selected_sources)} largest sources (by segmentation area) "
                    f"from {len(object_ids)} available in tile {tile_id} (after pre-filtering from {len(valid_sources)})"
                )
                if len(selected_sources) > 0:
                    min_area = selected_sources[-1][1]
                    max_area = selected_sources[0][1]
                    logger.info(f"Segmentation area range: {min_area:.1f} - {max_area:.1f} pixels")

                # Add sources from this tile
                for i, _ in selected_sources:
                    # Calculate cutout size from segmentation area
                    # (no max limit to preserve size variation)
                    cutout_diameter_pixels = calculate_cutout_size_from_segmentation_area(
                        seg_areas[i],
                        DEFAULT_CUTOUT_SIZE_MULTIPLIER,
                        DEFAULT_MIN_CUTOUT_SIZE,
                        max_cutout_size=None,
                    )

                    source = {
                        "SourceID": f"TILE_{tile_id}_{object_ids[i]}",
                        "RA": float(ras[i]),
                        "Dec": float(decs[i]),
                        "diameter_pixel": cutout_diameter_pixels,
                        # Only use diameter_pixel (mutual exclusivity)
                        "segmentation_area": float(
                            seg_areas[i]
                        ),  # Store original area for reference
                        "fits_file_paths": str(
                            image_paths
                        ),  # Include all selected channels as list
                    }
                    all_sources.append(source)

        except Exception as e:
            logger.error(f"Failed to process tile {tile_id}: {e}")
            continue

    logger.info(
        f"Created benchmark input with {len(all_sources)} sources from {len(selected_tiles)} tiles"
    )
    return pd.DataFrame(all_sources)


# run_benchmark_core function moved to benchmark_utils.py (as run_benchmark_core_with_orchestrator)


def run_benchmark(catalogue: pd.DataFrame, config) -> Dict[str, Any]:
    """
    Run the benchmark with memory and performance monitoring.

    Args:
        catalogue: Source catalogue DataFrame
        config: Configuration DotMap

    Returns:
        Benchmark results dictionary
    """
    return run_benchmark_with_monitoring(catalogue, config, "Q1_datalabs_benchmark")


# collect_performance_statistics function moved to benchmark_utils.py


# create_performance_charts function moved to benchmark_utils.py


# save_benchmark_results function moved to benchmark_utils.py


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark Cutana on datalabs Q1 data")
    parser.add_argument(
        "--layout-csv",
        type=str,
        default="Q1_data_layout.csv",
        help="Path to Q1_data_layout.csv (default: Q1_data_layout.csv)",
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default="",
        help="Comma-separated list of tile IDs (default: random selection or all tiles)",
    )
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=DEFAULT_N_TILES,
        help="Number of random tiles to select (overridden by --tiles)",
    )
    parser.add_argument(
        "--sources-per-tile",
        type=int,
        default=DEFAULT_SOURCES_PER_TILE,
        help="Maximum sources per tile (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for benchmark results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--fits-channels",
        type=str,
        default="NIR_H_path,NIR_J_path,NIR_Y_path",
        help="Comma-separated list of FITS channels to include (default: all NIR channels)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum worker processes (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default=DEFAULT_OUTPUT_FORMAT,
        choices=["zarr", "fits"],
        help=f"Output format (default: {DEFAULT_OUTPUT_FORMAT})",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO", console_level="INFO")
    logger.info("Starting Cutana Q1 datalabs benchmark")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Arguments: {args}")

    # Create timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Parse tile selection
        tile_ids = []
        if args.tiles:
            tile_ids = [tid.strip() for tid in args.tiles.split(",") if tid.strip()]

        # Parse FITS channels
        fits_channels = [ch.strip() for ch in args.fits_channels.split(",") if ch.strip()]
        logger.info(f"Using FITS channels: {fits_channels}")

        # Load data layout
        layout_csv_path = Path(args.layout_csv)
        if not layout_csv_path.exists():
            logger.error(f"Layout CSV not found: {layout_csv_path}")
            logger.error("Please run: python scripts/create_q1_data_layout.py first")
            sys.exit(1)

        data_layout = load_data_layout(layout_csv_path)

        # Select tiles
        selected_tiles = select_tiles(data_layout, tile_ids, args.n_tiles)
        if selected_tiles.empty:
            logger.error("No tiles selected for benchmarking")
            sys.exit(1)

        # Create benchmark input
        logger.info("Creating benchmark input from selected tiles...")
        catalogue = create_benchmark_input_from_tiles(
            selected_tiles,
            sources_per_tile=args.sources_per_tile,
            fits_channels=fits_channels,
        )

        if catalogue.empty:
            logger.error("No sources found for benchmarking")
            sys.exit(1)

        # Calculate batch size
        total_sources = len(catalogue)

        # Configuration with timestamped output directory
        timestamped_output_dir = Path(args.output_dir) / timestamp
        config = get_default_config()

        # Override with benchmark-specific values
        config.max_workers = args.max_workers
        config.N_batch_cutout_process = N_BATCH_CUTOUT_PROCESS
        config.output_format = args.output_format
        config.output_dir = str(timestamped_output_dir)
        config.fits_extensions = ["PRIMARY"]  # Use PRIMARY extension for multi-channel FITS files
        config.target_resolution = DEFAULT_TARGET_RESOLUTION
        config.data_type = "float32"  # Use data_type for output format
        config.normalisation_method = "linear"
        config.interpolation = "bilinear"
        config.log_level = "INFO"
        config.source_catalogue = "benchmark_generated_catalogue"  # Placeholder for validation

        # Configure channel_weights based on the fits_channels selected
        if len(fits_channels) > 1:
            # Multi-channel: create channel_weights dict with proper names
            # Extract base names from fits_channels (e.g., "NIR_H_path" -> "NIR_H")
            channel_names = []
            for channel in fits_channels:
                if channel.endswith("_path"):
                    channel_names.append(channel[:-5])  # Remove '_path' suffix
                else:
                    channel_names.append(channel)

            # Create identity channel weights (each input channel maps to one output channel)
            config.channel_weights = {
                channel_name: [1.0 if i == j else 0.0 for j in range(len(channel_names))]
                for i, channel_name in enumerate(channel_names)
            }

            # Set selected extensions based on channel names
            config.selected_extensions = [
                {"name": name, "ext": "PRIMARY"} for name in channel_names
            ]
        else:
            # Single-channel: simple passthrough
            config.channel_weights = {"PRIMARY": [1.0]}
            config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]

        logger.info(f"Configuration: {config}")

        # Run benchmark
        results = run_benchmark(catalogue, config)

        # Save results with timestamp
        results_dir = timestamped_output_dir / "benchmark_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_benchmark_results(results, results_dir, "datalabs")

        logger.info(f"Datalabs benchmark completed successfully! Results saved to: {results_dir}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        # Show stack trace
        logger.error("Exception details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
