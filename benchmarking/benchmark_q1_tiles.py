#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Benchmark script for Cutana Q1 tiles performance evaluation.

This script dynamically discovers Euclid tiles and corresponding catalogues
in the specified data folder, creates cutouts using 4 workers, and monitors
memory consumption and runtime performance.

Usage:
    python benchmarking/benchmark_q1_tiles.py [--data-folder PATH]

    --data-folder PATH: Path to folder containing FITS files and catalogues
                       (defaults to script_dir/../data)

The script will automatically discover:
- Catalog files: EUC_MER_FINAL-CAT_TILE*-*_*.fits
- FITS files: EUC_MER_BGSUB-MOSAIC-{extension}_TILE{tile_id}-*.fits
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, Any
from datetime import datetime

from cutana.logging_config import setup_logging
from cutana.get_default_config import get_default_config

try:
    # Try relative import first (when used as module)
    from .benchmark_utils import (
        run_benchmark_with_monitoring,
        save_benchmark_results,
        read_optimized_catalog,
        create_fits_path_cache,
    )
except ImportError:
    # Fall back to absolute import (when run directly)
    sys.path.append(str(Path(__file__).parent))
    from benchmark_utils import (
        run_benchmark_with_monitoring,
        save_benchmark_results,
        read_optimized_catalog,
        create_fits_path_cache,
    )

# CONFIGURATION PARAMETERS
TARGET_TOTAL_SOURCES = 50000  # Total sources to process
MAX_WORKERS = 15  # Number of worker processes
CUTOUT_SIZE_MULTIPLIER = 3.0  # Multiplier for segmentation area to cutout size (increased from 1.0)
MIN_CUTOUT_SIZE = 32  # Minimum cutout size in pixels
MAX_CUTOUT_SIZE = 512  # Maximum cutout size in pixels
OUTPUT_FORMAT = "zarr"  # Output format: "zarr" or "fits"
TARGET_RESOLUTION = 128  # Target resolution for cutouts
N_BATCH_CUTOUT_PROCESS = 2500  # Process sources in batches of 1000

# EXTENSION CONFIGURATION
# Select which instrument extensions to use.
# Options:  ['VIS'], ['NIR-H', 'NIR-J', 'NIR-Y'], or ['VIS', 'NIR-H', 'NIR-J', 'NIR-Y']
USE_EXTENSIONS = ["VIS"]
# USE_EXTENSIONS = ["NIR-H", "NIR-J", "NIR-Y"]
# CHANNEL COMBINATION MATRIX
# Define how to combine multiple channels (only used if USE_EXTENSIONS has multiple entries)
# Format: List of lists where each inner list represents weights for one output channel
CHANNEL_MATRIX = [
    [1.0, 0.0, 0.0],  # Channel 1: 100% NIR-H
    [0.0, 1.0, 0.0],  # Channel 2: 100% NIR-J
    [0.0, 0.0, 1.0],  # Channel 3: 100% NIR-Y
]


def get_benchmark_config():
    """Get default config and customize for benchmark."""
    config = get_default_config()

    # Override with benchmark-specific values
    config.max_workers = MAX_WORKERS
    config.N_batch_cutout_process = N_BATCH_CUTOUT_PROCESS
    config.output_format = OUTPUT_FORMAT
    config.output_dir = "benchmarking/output"
    config.fits_extensions = ["PRIMARY"]
    config.target_resolution = TARGET_RESOLUTION
    config.data_type = "uint8"  # Use data_type for output format
    config.normalisation_method = "linear"
    config.interpolation = "bilinear"
    config.log_level = "INFO"

    # Add required channel_weights for batch processing based on CHANNEL_MATRIX
    if len(USE_EXTENSIONS) > 1:
        # Multi-channel: create channel_weights dict using CHANNEL_MATRIX
        config.channel_weights = {}
        for i, ext in enumerate(USE_EXTENSIONS):
            # Extract weights for this extension from CHANNEL_MATRIX
            # CHANNEL_MATRIX[output_channel][input_extension]
            weights = [CHANNEL_MATRIX[j][i] for j in range(len(CHANNEL_MATRIX))]
            config.channel_weights[ext] = weights
    else:
        # Single-channel: simple passthrough
        config.channel_weights = {USE_EXTENSIONS[0]: [1.0]}

    config.apply_flux_conversion = True  # Flux conversion for eg Euclid, default True
    config.source_catalogue = "benchmark_generated_catalogue"  # Placeholder for validation

    # Set selected extensions based on USE_EXTENSIONS
    config.selected_extensions = [{"name": ext, "ext": "PRIMARY"} for ext in USE_EXTENSIONS]

    # Configure LoadBalancer event logging for better benchmark analysis
    config.loadbalancer.event_log_file = f"{config.output_dir}/logs/loadbalancer_events.jsonl"

    return config


# monitor_memory_usage function moved to benchmark_utils.py


# calculate_cutout_size_from_segmentation_area function moved to benchmark_utils.py


def discover_euclid_data(data_dir: Path) -> Dict[str, str]:
    """
    Dynamically discover Euclid catalog and FITS files in the data directory.

    Args:
        data_dir: Path to the data directory

    Returns:
        Dictionary mapping catalog filenames to their tile IDs
    """
    catalog_tiles = {}

    # Find all catalog files matching Euclid naming pattern
    catalog_pattern = "EUC_MER_FINAL-CAT_TILE*-*_*.fits"
    catalog_files = list(data_dir.glob(catalog_pattern))

    for cat_file in catalog_files:
        # Extract tile ID from filename
        # Pattern: EUC_MER_FINAL-CAT_TILE{tile_id}-{checksum}_{timestamp}Z_{version}.fits
        filename = cat_file.name
        if "TILE" in filename:
            # Extract tile ID between "TILE" and next "-"
            tile_start = filename.find("TILE") + 4
            tile_end = filename.find("-", tile_start)
            if tile_end > tile_start:
                tile_id = filename[tile_start:tile_end]
                catalog_tiles[filename] = tile_id
                logger.info(f"Discovered catalog for tile {tile_id}: {filename}")

    logger.info(f"Found {len(catalog_tiles)} catalog files in {data_dir}")
    return catalog_tiles


def create_benchmark_input_csv(data_dir: Path = None) -> str:
    """
    Create benchmark input CSV from Euclid catalog files.
    Supports multi-channel processing based on USE_EXTENSIONS configuration.

    Args:
        data_dir: Path to data directory (defaults to script_dir/../data)

    Returns:
        Path to created CSV file
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    # Dynamically discover catalog files and tile IDs
    catalog_tiles = discover_euclid_data(data_dir)

    all_sources = []
    sources_per_worker = TARGET_TOTAL_SOURCES // MAX_WORKERS

    logger.info(
        f"Creating benchmark input CSV with {TARGET_TOTAL_SOURCES} total sources ({sources_per_worker} per worker)"
    )

    # Pre-build FITS file path cache to avoid repeated glob operations
    fits_path_cache = create_fits_path_cache(data_dir, catalog_tiles, USE_EXTENSIONS)

    # First pass: collect all available sources using unified optimized function
    all_available_sources = []

    for cat_file, tile_id in catalog_tiles.items():
        cat_path = data_dir / cat_file

        if not cat_path.exists():
            logger.error(f"Catalog file not found: {cat_path}")
            continue

        # Check if required extensions are available for this tile using cache
        available_extensions = []
        if tile_id in fits_path_cache:
            for extension in USE_EXTENSIONS:
                if extension in fits_path_cache[tile_id]:
                    available_extensions.append(extension)
                    logger.debug(
                        f"Found {extension} file for tile {tile_id}: {Path(fits_path_cache[tile_id][extension]).name}"
                    )

        if len(available_extensions) < len(USE_EXTENSIONS):
            missing = set(USE_EXTENSIONS) - set(available_extensions)
            logger.warning(f"Tile {tile_id} missing extensions {missing}, skipping")
            continue

        logger.info(f"Reading catalog: {cat_file}")

        # Build FITS file paths once for this tile (all sources share same FITS files)
        fits_paths = []
        for extension in USE_EXTENSIONS:
            if tile_id in fits_path_cache and extension in fits_path_cache[tile_id]:
                fits_paths.append(fits_path_cache[tile_id][extension])
            else:
                logger.warning(f"No {extension} FITS file found for tile {tile_id} in cache")

        if not fits_paths:
            logger.warning(
                f"No FITS files found for tile {tile_id} with extensions {USE_EXTENSIONS}"
            )
            continue

        fits_paths_str = str(fits_paths)  # Convert once for all sources in this tile

        # Use unified optimized catalog reading
        tile_sources_df = read_optimized_catalog(
            str(cat_path),
            min_seg_area=10.0,
            max_sources=None,  # Don't limit per tile, we'll select globally
            cutout_size_multiplier=CUTOUT_SIZE_MULTIPLIER,
            min_cutout_size=MIN_CUTOUT_SIZE,
            max_cutout_size=None,
        )

        # Convert DataFrame to list format and add tile-specific information
        sources_added = 0
        for _, row in tile_sources_df.iterrows():
            source = {
                "SourceID": f"TILE_{tile_id}_{row['OBJECT_ID']}",
                "RA": row["RA"],
                "Dec": row["Dec"],
                "diameter_pixel": row["diameter_pixel"],
                "segmentation_area": row["segmentation_area"],
                "fits_file_paths": fits_paths_str,  # Multi-channel support - reuse string
            }
            all_available_sources.append(source)
            sources_added += 1

        logger.info(f"Added {sources_added} sources from catalog {cat_file}")

        # Early termination optimization: if we have enough large sources, stop processing more catalogs
        if (
            len(all_available_sources) >= TARGET_TOTAL_SOURCES * 2
        ):  # Collect 2x target to ensure quality selection
            logger.info(
                f"Collected {len(all_available_sources)} sources (>= 2x target), stopping early for efficiency"
            )
            break

    # Second pass: sort by segmentation area and select largest sources
    total_available = len(all_available_sources)
    logger.info(f"Found {total_available} total available sources across all catalogs")

    # Sort by segmentation area (largest first)
    all_available_sources.sort(key=lambda x: x["segmentation_area"], reverse=True)

    if total_available >= TARGET_TOTAL_SOURCES:
        # Take the largest sources by segmentation area
        all_sources = all_available_sources[:TARGET_TOTAL_SOURCES]
        logger.info(
            f"Selected {len(all_sources)} largest sources by segmentation area for benchmarking"
        )
        logger.info(
            f"Segmentation area range:"
            f"{all_sources[-1]['segmentation_area']:.1f} - {all_sources[0]['segmentation_area']:.1f} pixels"
        )
    else:
        # Use all available sources
        all_sources = all_available_sources
        logger.warning(f"Only {total_available} sources available, using all of them")

    # Create DataFrame and save to CSV with extension info in filename
    df = pd.DataFrame(all_sources)
    extension_suffix = "_".join(USE_EXTENSIONS).replace("-", "")
    output_path = data_dir / f"benchmark_input_10k_cutouts_{extension_suffix.lower()}.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Created benchmark input CSV with {len(all_sources)} sources at: {output_path}")
    return str(output_path)


# run_benchmark_core function moved to benchmark_utils.py


def run_benchmark(csv_path: str, config=None) -> Dict[str, Any]:
    """
    Run the benchmark with memory and performance monitoring.

    Args:
        csv_path: Path to input CSV file
        config: Optional orchestrator config (defaults to get_benchmark_config())

    Returns:
        Benchmark results dictionary
    """
    # Load catalogue
    catalogue = pd.read_csv(csv_path)

    # Use provided config or default
    if config is None:
        config = get_benchmark_config()

    extension_suffix = "_".join(USE_EXTENSIONS).replace("-", "")
    benchmark_name = f"Q1_tiles_benchmark_{extension_suffix.lower()}"

    return run_benchmark_with_monitoring(catalogue, config, benchmark_name)


# collect_performance_statistics function moved to benchmark_utils.py


# create_performance_charts function moved to benchmark_utils.py


# save_benchmark_results function moved to benchmark_utils.py


def main(data_folder: str = None):
    """
    Main benchmark execution.

    Args:
        data_folder: Path to data folder containing FITS files and catalogues
                    (defaults to script_dir/../data)
    """
    setup_logging(log_level="INFO", console_level="INFO")

    logger.info("Starting Cutana Q1 tiles benchmark")

    # Create timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up data directory
    if data_folder:
        data_dir = Path(data_folder)
        if not data_dir.exists():
            logger.error(f"Data folder not found: {data_dir}")
            sys.exit(1)
        logger.info(f"Using data folder: {data_dir}")
    else:
        data_dir = Path(__file__).parent.parent / "data"
        logger.info(f"Using default data folder: {data_dir}")

    try:
        # Create benchmark input CSV
        csv_path = create_benchmark_input_csv(data_dir)

        # Get config and update with timestamped output directory
        config = get_benchmark_config()
        config.output_dir = f"benchmarking/output/{timestamp}"

        # Update LoadBalancer event log path to use timestamped output directory
        config.loadbalancer.event_log_file = f"{config.output_dir}/logs/loadbalancer_events.jsonl"

        # Run benchmark with updated config
        results = run_benchmark(csv_path, config)

        # Save results with timestamp
        output_dir = Path(f"benchmarking/results/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_benchmark_results(results, output_dir, "tiles")

        logger.info(f"Benchmark completed successfully! Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        # Show stack trace
        logger.error("Exception details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Cutana Q1 tiles performance")
    parser.add_argument(
        "--data-folder",
        type=str,
        help="Path to data folder containing FITS files and catalogues (defaults to script_dir/../data)",
    )

    args = parser.parse_args()
    main(args.data_folder)
