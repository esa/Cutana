#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Cutana release testing suite.

Single script to run before any release to heavily exercise Cutana features for
robustness, performance, and correctness.

Key capabilities:
1. Ground-Truth Pre-Validation: Verifies that ground-truth archives exist for all configurations
   before starting any generation runs.
2. Generation Runs: Executes cutout generation across backends (direct, disk,
   disk-streaming, mem-streaming), formats (zarr, fits), normalisations, and channel combinations.
3. Correctness Verification: Compares generated cutouts against ground-truth cutouts for a fixed set
   of predefined sources, verifying pixel-wise correctness. For FITS output it additionally checks
   the astrometry written into each cutout against the parent tile header (see ``wcs_check.py``);
   a pixel diff cannot see a shifted WCS.
4. Ground-Truth Generation: Saves reference cutouts for any configuration using --generate-ground-truth.
5. Matrix Testing & GitHub Report: Runs a pre-release matrix across configurations (--matrix) and
   outputs a GitHub-ready markdown summary report.

Usage::

    # Single test run with automatic ground-truth verification
    python tests/release/release_test.py \
        --catalogue-size small \
        --normalisation  asinh \
        --gen-type       disk-streaming \
        --ch-in-out      1vis1 \
        --output-folder  /tmp/cutana_release_test

    # Generate ground-truth cutouts for a configuration
    python tests/release/release_test.py \
        --catalogue-size small \
        --normalisation  asinh \
        --gen-type       direct \
        --ch-in-out      1vis1 \
        --output-folder  /tmp/cutana_gt_gen \
        --generate-ground-truth

    # Run the full pre-release test matrix
    python tests/release/release_test.py \
        --matrix \
        --output-folder /tmp/cutana_release_matrix

Exit code: 0 if all runs and verifications pass, 1 if any fails.
"""

import argparse
import datetime
import json
import os
import platform
import shutil
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from astropy.io import fits as astropy_fits
from dotmap import DotMap
from loguru import logger
from wcs_check import verify_cutout_wcs

import cutana
from cutana import (
    Orchestrator,
    StreamingOrchestrator,
    create_cutouts_direct,
    get_default_config,
)
from cutana.cutout_writer_zarr import create_zarr_from_memory
from cutana.logging_config import setup_logging

# ── Catalogues ────────────────────────────────────────────────────────────────
_CATALOGUE_ROOT = Path(__file__).parent / "catalogues"

# Throughput is only meaningful next to the last release measured the same way, on the same
# machine: the absolute numbers move with disk, cache state and core count, the delta does not.
# Each matrix run writes its own baseline file, and promoting it is a copy over this path.
_BASELINE_ROOT = Path(__file__).parent / "baselines"
DEFAULT_BASELINE_PATH: Path = _BASELINE_ROOT / "previous_release.json"
BASELINE_FILENAME = "throughput_baseline.json"
CATALOGUE_FILES: dict[str, Path] = {
    "big": _CATALOGUE_ROOT / "q1_big_catalogue.parquet",
    "medium": _CATALOGUE_ROOT / "q1_medium_catalogue.parquet",
    "small": _CATALOGUE_ROOT / "q1_small_catalogue.parquet",
}

RANDOMIZATION_SEED = 42
PREDEFINED_SAMPLE_SIZE = 50
N_SIZE_STRATA = 6

# ── Ground Truth Root ─────────────────────────────────────────────────────────
_DEFAULT_GT_ROOT = Path(__file__).parent / "ground_truth"

# ── Channel configurations ────────────────────────────────────────────────────
# channel_weights semantics (from image_processor.py::combine_channels):
#   key   = input channel name  (column of the combination matrix)
#   value = list[float] of length n_output_channels  (row weights for that input)
#
# User's (4,3) default matrix [[1,0,0,0],[0,1,0.5,0],[0,0,0.5,1]]
# (rows = output channels, columns = inputs VIS/NIR-H/NIR-Y/NIR-J)
# → transposed to column-per-input format used by Cutana:
#   VIS: [1,0,0], NIR-H: [0,1,0], NIR-Y: [0,0.5,0.5], NIR-J: [0,0,1]

CH_IN_OUT_CONFIGS: dict[str, dict] = {
    "1vis1": {
        "label": "1 VIS → 1",
        "channel_weights": {"VIS": [1.0]},
        "selected_extensions": [{"name": "VIS", "ext": "PRIMARY"}],
    },
    "3nisp3": {
        "label": "3 NISP → 3",
        "channel_weights": {
            "NIR-H": [1.0, 0.0, 0.0],
            "NIR-Y": [0.0, 1.0, 0.0],
            "NIR-J": [0.0, 0.0, 1.0],
        },
        "selected_extensions": [
            {"name": "NIR-H", "ext": "PRIMARY"},
            {"name": "NIR-Y", "ext": "PRIMARY"},
            {"name": "NIR-J", "ext": "PRIMARY"},
        ],
    },
    "4visnisp3": {
        "label": "4 VIS+NISP → 3",
        "channel_weights": {
            "VIS": [1.0, 0.0, 0.0],
            "NIR-H": [0.0, 1.0, 0.0],
            "NIR-Y": [0.0, 0.5, 0.5],
            "NIR-J": [0.0, 0.0, 1.0],
        },
        "selected_extensions": [
            {"name": "VIS", "ext": "PRIMARY"},
            {"name": "NIR-H", "ext": "PRIMARY"},
            {"name": "NIR-Y", "ext": "PRIMARY"},
            {"name": "NIR-J", "ext": "PRIMARY"},
        ],
    },
}

# ── Pre-Release Test Matrix ──────────────────────────────────────────────────
# Rows 1-12 are a strength-2 covering array over
#   gen_type(4) x ch_in_out(3) x normalisation(3) x data_type(2) x target_resolution(3):
# all 89 level pairs occur at least once in 12 runs (full factorial would be 216).
# output_format is spread over the six disk-writing rows so that zarr and fits each
# see all three channel configs, both dtypes and both disk backends.
RELEASE_TEST_MATRIX: list[dict] = [
    dict(
        catalogue_size="small",
        gen_type="disk",
        ch_in_out="4visnisp3",
        normalisation="none",
        output_format="fits",
        data_type="float32",
        target_resolution=64,
    ),
    dict(
        catalogue_size="small",
        gen_type="disk-streaming",
        ch_in_out="4visnisp3",
        normalisation="asinh",
        output_format="zarr",
        data_type="uint8",
        target_resolution=150,
    ),
    dict(
        catalogue_size="small",
        gen_type="disk-streaming",
        ch_in_out="3nisp3",
        normalisation="log",
        output_format="fits",
        data_type="float32",
        target_resolution=224,
    ),
    dict(
        catalogue_size="small",
        gen_type="mem-streaming",
        ch_in_out="1vis1",
        normalisation="log",
        output_format="zarr",
        data_type="uint8",
        target_resolution=64,
    ),
    dict(
        catalogue_size="small",
        gen_type="direct",
        ch_in_out="1vis1",
        normalisation="none",
        output_format="zarr",
        data_type="float32",
        target_resolution=150,
    ),
    dict(
        catalogue_size="small",
        gen_type="mem-streaming",
        ch_in_out="3nisp3",
        normalisation="none",
        output_format="zarr",
        data_type="uint8",
        target_resolution=224,
    ),
    dict(
        catalogue_size="small",
        gen_type="direct",
        ch_in_out="3nisp3",
        normalisation="asinh",
        output_format="zarr",
        data_type="uint8",
        target_resolution=64,
    ),
    dict(
        catalogue_size="small",
        gen_type="disk",
        ch_in_out="1vis1",
        normalisation="asinh",
        output_format="fits",
        data_type="uint8",
        target_resolution=224,
    ),
    dict(
        catalogue_size="small",
        gen_type="disk",
        ch_in_out="3nisp3",
        normalisation="log",
        output_format="zarr",
        data_type="float32",
        target_resolution=150,
    ),
    dict(
        catalogue_size="small",
        gen_type="direct",
        ch_in_out="4visnisp3",
        normalisation="log",
        output_format="zarr",
        data_type="uint8",
        target_resolution=224,
    ),
    dict(
        catalogue_size="small",
        gen_type="mem-streaming",
        ch_in_out="4visnisp3",
        normalisation="asinh",
        output_format="zarr",
        data_type="float32",
        target_resolution=150,
    ),
    dict(
        catalogue_size="small",
        gen_type="disk-streaming",
        ch_in_out="1vis1",
        normalisation="none",
        output_format="zarr",
        data_type="float32",
        target_resolution=64,
    ),
]

# ── Temporary files created during the run ────────────────────────────────────
_TMP_FILES: list[Path] = []


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth paths & validation
# ─────────────────────────────────────────────────────────────────────────────


def get_ground_truth_path(
    catalogue_size: str,
    ch_in_out: str,
    normalisation: str,
    data_type: str,
    target_resolution: int,
    base_dir: Path | None = None,
) -> Path:
    """Derive standard path to the ground-truth Zarr archive for a given configuration."""
    root = base_dir if base_dir is not None else _DEFAULT_GT_ROOT
    config_name = f"{ch_in_out}_{normalisation}_{data_type}_{target_resolution}px"
    return root / catalogue_size / config_name / "images.zarr"


def check_ground_truth_exists(gt_path: Path) -> bool:
    """Check if a ground-truth Zarr archive exists and is valid (not a Git LFS pointer)."""
    if not gt_path.exists():
        return False
    meta_path = gt_path.parent / f"{gt_path.stem}_metadata.parquet"
    if not meta_path.exists():
        return False

    # Check that metadata parquet file is not an unpulled Git LFS pointer
    try:
        with open(meta_path, "rb") as f:
            header = f.read(100)
        if b"version https://git-lfs.github.com/spec/" in header:
            logger.error(
                f"Ground-truth file {meta_path.name} at {meta_path} is a Git LFS pointer. "
                "Please run 'git lfs pull' to download the real ground-truth files."
            )
            return False
        if not header.startswith(b"PAR1"):
            logger.error(
                f"Ground-truth file {meta_path.name} is not a valid Parquet file (missing 'PAR1' header)."
            )
            return False
    except Exception as e:
        logger.error(f"Failed to inspect ground-truth metadata file {meta_path}: {e}")
        return False

    return True


def validate_ground_truths_exist(
    configs: list[dict],
    ground_truth_override: str | None = None,
) -> None:
    """Verify that ground-truth archives exist for all configurations before running.

    Raises:
        FileNotFoundError: If one or more configurations lack a ground-truth archive.
    """
    missing: list[tuple[dict, Path]] = []
    for cfg in configs:
        gt_path = (
            Path(ground_truth_override)
            if ground_truth_override is not None
            else get_ground_truth_path(
                catalogue_size=cfg["catalogue_size"],
                ch_in_out=cfg["ch_in_out"],
                normalisation=cfg["normalisation"],
                data_type=cfg["data_type"],
                target_resolution=cfg["target_resolution"],
            )
        )
        if not check_ground_truth_exists(gt_path):
            missing.append((cfg, gt_path))

    if missing:
        lines = ["Ground-truth archives missing for the following configuration(s):"]
        for cfg, path in missing:
            cfg_str = (
                f"catalogue={cfg['catalogue_size']}, channels={cfg['ch_in_out']}, "
                f"norm={cfg['normalisation']}, dtype={cfg['data_type']}, res={cfg['target_resolution']}px"
            )
            lines.append(f"  • [{cfg_str}]\n    Expected path: {path}")
        lines.append(
            "\nPlease generate ground-truth archives before running tests using --generate-ground-truth, "
            "or pass --skip-verification to bypass correctness verification."
        )
        raise FileNotFoundError("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cutana release testing suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--matrix",
        action="store_true",
        default=False,
        help="Run the pre-release test matrix covering multiple backends and configurations",
    )

    # Configuration arguments (required for single-test mode, optional with --matrix)
    parser.add_argument(
        "--catalogue-size",
        choices=["big", "medium", "small"],
        default="small",
        help="Catalogue to use: big (~2.2M), medium (~600k), small (10k sources)",
    )
    parser.add_argument(
        "--normalisation",
        choices=["log", "asinh", "none"],
        default="asinh",
        help="Normalisation method to apply",
    )
    parser.add_argument(
        "--gen-type",
        choices=["direct", "disk", "disk-streaming", "mem-streaming"],
        default="direct",
        help="Generation backend to use",
    )
    parser.add_argument(
        "--ch-in-out",
        choices=["1vis1", "3nisp3", "4visnisp3"],
        default="1vis1",
        help="Channel configuration: 1vis1, 3nisp3, or 4visnisp3",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Root output directory for this test run",
    )

    # Optional configuration overrides
    parser.add_argument(
        "--randomize",
        action="store_true",
        default=False,
        help="Randomize catalogue with fixed seed=42 (writes temp parquet)",
    )
    parser.add_argument(
        "--combination-matrix",
        default=None,
        metavar="JSON",
        help=(
            "Override default channel combination matrix as a JSON string. "
            "Rows = output channels, columns = input channels. "
            'Example: "[[1,0,0,0],[0,1,0.5,0],[0,0,0.5,1]]"'
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["zarr", "fits"],
        default="zarr",
        help="Output format for disk-based gen-types (default: zarr)",
    )
    parser.add_argument(
        "--target-resolution",
        type=int,
        default=150,
        metavar="INT",
        help="Target cutout resolution in pixels (default: 150)",
    )
    parser.add_argument(
        "--data-type",
        choices=["uint8", "float32"],
        default="uint8",
        help="Data type for the output cutouts (default: uint8)",
    )

    # Ground-truth & Verification options
    parser.add_argument(
        "--ground-truth",
        default=None,
        metavar="PATH",
        help="Path to ground-truth Zarr archive or directory (default: auto-resolved from config)",
    )
    parser.add_argument(
        "--generate-ground-truth",
        action="store_true",
        default=False,
        help="Save cutouts for predefined sources as ground truth for this configuration",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        default=False,
        help="Skip ground-truth comparison (run for execution only)",
    )
    parser.add_argument(
        "--skip-wcs-check",
        action="store_true",
        default=False,
        help="Skip the cutout-WCS-vs-parent-tile check on FITS output",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Pixel difference tolerance (default: 0.0 for uint8, 1e-4 for float32)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Pin max_workers instead of using the default for this version. Use it when "
            "comparing two releases whose defaults differ, so the delta measures the code "
            "rather than the default."
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="PATH",
        help=(
            "Throughput baseline to compare the matrix against "
            "(default: tests/release/baselines/previous_release.json if it exists). "
            "Each matrix run writes its own baseline into the output folder; promote it "
            "by copying that file over the default path at release time."
        ),
    )
    parser.add_argument(
        "--report-file",
        default=None,
        metavar="PATH",
        help="Path to write GitHub release markdown report (default: <output-folder>/release_test_report.md)",
    )

    args = parser.parse_args()
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Channel configuration helpers
# ─────────────────────────────────────────────────────────────────────────────


def apply_combination_matrix_override(json_str: str, ch_cfg: dict) -> dict:
    """Convert JSON matrix (rows=outputs, cols=inputs) to channel_weights dict.

    Example:
        json_str = "[[1,0,0,0],[0,1,0.5,0],[0,0,0.5,1]]"
        input channels (cols): VIS, NIR-H, NIR-Y, NIR-J
        → {"VIS": [1,0,0], "NIR-H": [0,1,0], "NIR-Y": [0,0.5,0.5], "NIR-J": [0,0,1]}
    """
    matrix = json.loads(json_str)  # list[list[float]], shape [n_out][n_in]
    input_channels = list(ch_cfg["channel_weights"].keys())
    n_out = len(matrix)
    return {
        ch: [float(matrix[out_idx][in_idx]) for out_idx in range(n_out)]
        for in_idx, ch in enumerate(input_channels)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Catalogue loading
# ─────────────────────────────────────────────────────────────────────────────


def verify_catalogue_file(file_path: Path) -> None:
    """Verify that the catalogue file exists, is not a Git LFS pointer, and is a valid Parquet file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Catalogue file not found: {file_path}")

    # Check that it's not a Git LFS pointer file (read first 100 bytes)
    try:
        with open(file_path, "rb") as f:
            header = f.read(100)
    except Exception as e:
        raise RuntimeError(f"Failed to read file {file_path}: {e}")

    if b"version https://git-lfs.github.com/spec/" in header:
        raise RuntimeError(
            f"Catalogue file {file_path.name} is a Git LFS pointer, not the actual file. "
            "Please run 'git lfs pull' to download the real catalogue files."
        )

    if not header.startswith(b"PAR1"):
        raise RuntimeError(
            f"Catalogue file {file_path.name} is not a valid Parquet file (missing 'PAR1' header). "
            "Please run 'git lfs pull' to download the real files."
        )


def load_catalogue(
    catalogue_size: str,
    randomize: bool,
    output_folder: Path,
) -> tuple[pd.DataFrame, Path]:
    """Load the requested parquet catalogue and optionally randomize it.

    Returns:
        (df, catalogue_path): df is the in-memory DataFrame; catalogue_path is
        the file path to pass to Orchestrator / StreamingOrchestrator.
    """
    parquet_path = CATALOGUE_FILES[catalogue_size]
    verify_catalogue_file(parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} sources from {parquet_path.name}")

    if randomize:
        df = df.sample(frac=1, random_state=RANDOMIZATION_SEED).reset_index(drop=True)
        logger.info(f"Randomized catalogue (seed={RANDOMIZATION_SEED})")
        tmp_path = output_folder / "tmp_randomized_catalogue.parquet"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp_path, index=False)
        _TMP_FILES.append(tmp_path)
        logger.info(f"Wrote randomized catalogue → {tmp_path}")
        return df, tmp_path

    return df, parquet_path


# ─────────────────────────────────────────────────────────────────────────────
# Config building
# ─────────────────────────────────────────────────────────────────────────────


def build_config(
    catalogue_path: Path,
    output_dir: Path,
    output_format: str,
    target_resolution: int,
    normalisation: str,
    ch_cfg: dict,
    data_type: str,
    max_workers: int | None = None,
):
    """Build a Cutana DotMap config from parameters and channel configuration.

    Args:
        catalogue_path: Catalogue to read.
        output_dir: Where the run writes its cutouts.
        output_format: ``zarr`` or ``fits``.
        target_resolution: Cutout size in pixels.
        normalisation: Normalisation method.
        ch_cfg: One entry of ``CH_IN_OUT_CONFIGS``.
        data_type: Output dtype.
        max_workers: Pin the worker count instead of taking the default. Only for
            comparing two releases whose defaults differ; see ``--max-workers``.

    Returns:
        The config.
    """
    config = get_default_config()
    config.name = "release_test"
    config.source_catalogue = str(catalogue_path)
    config.output_dir = str(output_dir)
    config.output_format = output_format
    config.target_resolution = target_resolution
    config.normalisation_method = normalisation
    config.channel_weights = ch_cfg["channel_weights"]
    config.selected_extensions = ch_cfg["selected_extensions"]
    config.fits_extensions = ["PRIMARY"]
    config.data_type = data_type
    config.skip_catalogue_validation = False
    config.log_level = "INFO"
    config.console_log_level = "WARNING"
    # max_workers is otherwise left alone: get_default_config() derives it from the
    # CPU count and the LoadBalancer adjusts it at runtime, which is the behaviour a
    # release report should be measuring. Pinning it exists for one case -- comparing
    # against a release whose default differed -- where leaving it alone measures the
    # change in the default rather than the change in the code.
    if max_workers is not None:
        config.max_workers = max_workers
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Generation backends
# ─────────────────────────────────────────────────────────────────────────────


def run_direct(df: pd.DataFrame, config) -> list[dict]:
    """Run create_cutouts_direct() on the full catalogue DataFrame.

    ``create_cutouts_direct`` takes its worker count as an argument and does not read
    ``config.max_workers``, so the pin has to be passed through or ``--max-workers``
    would silently do nothing here while the record still claimed the pinned value.
    """
    logger.info(f"Running direct on {len(df):,} sources")
    return create_cutouts_direct(df, config, max_workers=config.max_workers)


def run_disk(catalogue_path: Path, config) -> None:
    """Run Orchestrator.start_processing() — output written to disk."""
    logger.info(f"Running disk mode for catalogue: {catalogue_path.name}")
    orch = Orchestrator(config)
    result = orch.start_processing(str(catalogue_path))
    if result["status"] == "failed":
        raise RuntimeError(f"Orchestrator failed: {result['error']}")


def run_disk_streaming(catalogue_path: Path, config) -> None:
    """Run StreamingOrchestrator with write_to_disk=True — output written to disk."""
    logger.info(f"Running disk-streaming for catalogue: {catalogue_path.name}")
    orch = StreamingOrchestrator(config)
    orch.init_streaming(
        batch_size=config.N_batch_cutout_process,
        write_to_disk=True,
        max_workers=config.max_workers,
        min_workers=1,
    )
    n_batches = orch.get_batch_count()
    logger.info(f"disk-streaming: {n_batches} batches")
    for i in range(n_batches):
        orch.next_batch()
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            logger.info(f"  batch {i + 1}/{n_batches} done")
    orch.cleanup()


def run_mem_streaming(catalogue_path: Path, config) -> list[dict]:
    """Run StreamingOrchestrator with write_to_disk=False — results collected in memory."""
    logger.info(f"Running mem-streaming for catalogue: {catalogue_path.name}")
    orch = StreamingOrchestrator(config)
    orch.init_streaming(
        batch_size=config.N_batch_cutout_process,
        write_to_disk=False,
        max_workers=config.max_workers,
        min_workers=1,
    )
    n_batches = orch.get_batch_count()
    logger.info(f"mem-streaming: {n_batches} batches")
    all_results = []
    for i in range(n_batches):
        batch = orch.next_batch()
        all_results.append(batch)
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            logger.info(f"  batch {i + 1}/{n_batches} done")
    orch.cleanup()
    return all_results


def run_generation(
    df: pd.DataFrame,
    config,
    gen_type: str,
    catalogue_path: Path,
) -> list[dict] | None:
    """Dispatch to the appropriate generation backend.

    Returns:
        list[dict] for direct and mem-streaming (in-memory results).
        None for disk and disk-streaming (output is on disk).
    """
    if gen_type == "direct":
        return run_direct(df, config)
    elif gen_type == "disk":
        run_disk(catalogue_path, config)
        return None
    elif gen_type == "disk-streaming":
        run_disk_streaming(catalogue_path, config)
        return None
    elif gen_type == "mem-streaming":
        return run_mem_streaming(catalogue_path, config)
    else:
        raise ValueError(f"Unknown gen_type: {gen_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Sampling & Predefined Sources
# ─────────────────────────────────────────────────────────────────────────────


def select_sample(df: pd.DataFrame, n_sources: int = PREDEFINED_SAMPLE_SIZE) -> pd.DataFrame:
    """Select a deterministic, size-stratified sample of sources for ground truth.

    A uniform draw from the Q1 catalogues is ~77% sources of <=16px diameter, so the
    reference set would consist almost entirely of tiny postage stamps upsampled 10-25x
    and would never exercise downsampling or large-cutout resampling. Stratifying over
    log-spaced diameter bins keeps the draw deterministic while covering the whole size
    range, and the largest source is always pinned into the sample.
    """
    n = min(n_sources, len(df))
    size_col = "diameter_pixel" if "diameter_pixel" in df.columns else "diameter_arcsec"
    sizes = df[size_col].astype(float)

    edges = np.unique(np.geomspace(sizes.min(), sizes.max(), N_SIZE_STRATA + 1))
    strata = pd.cut(sizes, edges, include_lowest=True, labels=False)

    per_stratum = max(1, n // int(strata.nunique()))
    picks = [
        grp.sample(n=min(per_stratum, len(grp)), random_state=RANDOMIZATION_SEED)
        for _, grp in df.groupby(strata, observed=True)
    ]
    # Pin the largest source so the extreme of the distribution is always verified.
    picks.append(df.loc[[sizes.idxmax()]])
    sample = pd.concat(picks).drop_duplicates(subset="SourceID")

    if len(sample) < n:  # top up from the remainder to reach the requested size
        rest = df.drop(index=sample.index)
        sample = pd.concat(
            [
                sample,
                rest.sample(n=min(n - len(sample), len(rest)), random_state=RANDOMIZATION_SEED),
            ]
        )
    return sample.iloc[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Cutout collection helpers
# ─────────────────────────────────────────────────────────────────────────────


def _collect_cutouts_by_id(results: list[dict]) -> dict[str, np.ndarray]:
    """Flatten batch result dicts into {str(source_id): cutout (H, W, C)}.

    Handles both create_cutouts_direct format (ndarray tensor, N,H,W,C)
    and mem-streaming format (list of per-source arrays).
    """
    collected: dict[str, np.ndarray] = {}
    for batch in results:
        cutouts = batch["cutouts"]
        metadata = batch["metadata"]
        if isinstance(cutouts, np.ndarray):
            # create_cutouts_direct: shape (N, H, W, C)
            for i, meta in enumerate(metadata):
                sid = str(meta["source_id"])
                collected[sid] = cutouts[i]
        else:
            # mem-streaming: list of per-source arrays, each (H, W, C)
            for arr, meta in zip(cutouts, metadata):
                sid = str(meta["source_id"])
                collected[sid] = np.asarray(arr)
    return collected


def read_generated_cutouts_by_id(
    output_dir: Path,
    sample_source_ids: set[str],
) -> dict[str, np.ndarray]:
    """Scan all *_metadata.parquet + *.zarr pairs produced by Cutana.

    Returns:
        {str(source_id): cutout array (H, W, C)} for all requested IDs found.
    """
    result: dict[str, np.ndarray] = {}
    for meta_path in sorted(output_dir.rglob("*_metadata.parquet")):
        meta_df = pd.read_parquet(meta_path)
        if "source_id" not in meta_df.columns:
            continue
        mask = meta_df["source_id"].astype(str).isin(sample_source_ids)
        if not mask.any():
            continue
        matching_indices = meta_df.index[mask].tolist()
        matching_ids = meta_df.loc[mask, "source_id"].astype(str).tolist()
        # Derive zarr path: "images_metadata.parquet" → "images.zarr"
        zarr_name = meta_path.name.replace("_metadata.parquet", ".zarr")
        zarr_path = meta_path.parent / zarr_name
        if not zarr_path.exists():
            logger.warning(f"Zarr store not found for metadata: {meta_path}")
            continue
        root = zarr.open(str(zarr_path), mode="r")
        images = root["images"]  # NHWC: (N, H, W, C)
        for idx, sid in zip(matching_indices, matching_ids):
            result[sid] = np.array(images[idx])  # (H, W, C)
    return result


def read_generated_fits_by_id(
    output_dir: Path,
    sample_source_ids: set[str],
) -> dict[str, np.ndarray]:
    """Read back FITS cutout files for sampled source IDs.

    Matching is done by reading the 'SOURCE' metadata card from the primary HDU header.
    """
    result: dict[str, np.ndarray] = {}
    str_sample_ids = {str(sid) for sid in sample_source_ids}

    for fits_path in output_dir.rglob("*.fits"):
        try:
            with astropy_fits.open(fits_path, memmap=False) as hdul:
                if len(hdul) == 0:
                    continue
                primary_header = hdul[0].header
                if "SOURCE" not in primary_header:
                    continue
                source_id = str(primary_header["SOURCE"])
                if source_id not in str_sample_ids:
                    continue

                channels = [
                    hdu.data for hdu in hdul if hdu.data is not None and len(hdu.data.shape) == 2
                ]
                if not channels:
                    continue
                # Stack channels along last axis: (H, W, C)
                result[source_id] = np.stack(channels, axis=-1)
                if len(result) == len(str_sample_ids):
                    break
        except Exception as e:
            logger.warning(f"Failed to read/parse FITS file {fits_path}: {e}")
            continue
    return result


def collect_test_cutouts(
    source_ids: set[str],
    config,
    gen_type: str,
    gen_results: list[dict] | None,
) -> dict[str, np.ndarray]:
    """Unified helper to retrieve cutouts for given source IDs from memory or disk output."""
    if gen_type in ("direct", "mem-streaming"):
        if gen_results is None:
            return {}
        all_cutouts = _collect_cutouts_by_id(gen_results)
        return {sid: arr for sid, arr in all_cutouts.items() if sid in source_ids}
    elif gen_type in ("disk", "disk-streaming"):
        output_dir = Path(config.output_dir)
        if config.output_format == "zarr":
            return read_generated_cutouts_by_id(output_dir, source_ids)
        else:
            return read_generated_fits_by_id(output_dir, source_ids)
    else:
        raise ValueError(f"Unexpected gen_type: {gen_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Loading, Saving & Verification
# ─────────────────────────────────────────────────────────────────────────────


def load_ground_truth(ground_truth_path: Path) -> dict[str, np.ndarray]:
    """Load ground-truth cutouts from a Zarr archive directory.

    Returns:
        dict mapping str(source_id) -> np.ndarray (H, W, C)
    """
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground-truth path not found: {ground_truth_path}")

    # Search for all *_metadata.parquet files
    search_dir = (
        ground_truth_path
        if ground_truth_path.is_dir() and not ground_truth_path.name.endswith(".zarr")
        else ground_truth_path.parent
    )
    meta_files = list(search_dir.glob("*_metadata.parquet"))
    if not meta_files:
        meta_files = list(search_dir.rglob("*_metadata.parquet"))

    gt_cutouts: dict[str, np.ndarray] = {}
    for meta_path in sorted(meta_files):
        # Validate that meta_path is not an unpulled Git LFS pointer
        with open(meta_path, "rb") as f:
            header = f.read(100)
        if b"version https://git-lfs.github.com/spec/" in header:
            raise RuntimeError(
                f"Ground-truth file {meta_path.name} is a Git LFS pointer. "
                "Please run 'git lfs pull' to download the real ground-truth files."
            )
        if not header.startswith(b"PAR1"):
            raise RuntimeError(
                f"Ground-truth file {meta_path.name} is not a valid Parquet file (missing 'PAR1' header)."
            )

        meta_df = pd.read_parquet(meta_path)
        if "source_id" not in meta_df.columns:
            continue
        zarr_name = meta_path.name.replace("_metadata.parquet", ".zarr")
        zarr_path = meta_path.parent / zarr_name
        if not zarr_path.exists():
            continue
        root = zarr.open(str(zarr_path), mode="r")
        images = root["images"]  # NHWC: (N, H, W, C)
        for idx, sid in enumerate(meta_df["source_id"].astype(str)):
            gt_cutouts[sid] = np.array(images[idx])

    if not gt_cutouts:
        raise ValueError(f"No valid ground truth cutouts found at: {ground_truth_path}")

    return gt_cutouts


def _write_cutouts_zarr_archive(
    sample_df: pd.DataFrame,
    cutouts_dict: dict[str, np.ndarray],
    output_zarr_path: Path,
    config,
) -> None:
    """Write cutouts and metadata to a Zarr archive using create_zarr_from_memory."""
    meta_lookup = {str(row["SourceID"]): row for _, row in sample_df.iterrows()}

    cutouts_nchw: list[np.ndarray] = []
    metadata: list[dict] = []
    for sid, cutout in cutouts_dict.items():
        if sid not in meta_lookup:
            continue
        cutouts_nchw.append(np.array(cutout).transpose(2, 0, 1))  # HWC → CHW
        src = meta_lookup[sid]

        meta_entry = {
            "source_id": sid,
            "RA": float(src["RA"]),
            "Dec": float(src["Dec"]),
            "fits_file_paths": str(src["fits_file_paths"]),
        }
        if "diameter_pixel" in src:
            meta_entry["diameter_pixel"] = int(src["diameter_pixel"])
        elif "diameter_arcsec" in src:
            meta_entry["diameter_arcsec"] = float(src["diameter_arcsec"])
        else:
            raise KeyError(
                "Source metadata must contain either 'diameter_pixel' or 'diameter_arcsec'"
            )

        metadata.append(meta_entry)

    if not cutouts_nchw:
        logger.warning(f"No cutouts matched metadata to save to {output_zarr_path}")
        return

    images = np.stack(cutouts_nchw, axis=0)  # (N, C, H, W)
    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    create_zarr_from_memory(images, metadata, str(output_zarr_path), config, append=False)


def save_ground_truth(
    sample_df: pd.DataFrame,
    config,
    gen_type: str,
    gen_results: list[dict] | None,
    output_path: Path,
) -> None:
    """Save cutouts from current run for predefined sources as ground-truth."""
    sample_source_ids = set(sample_df["SourceID"].astype(str).tolist())
    cutouts_dict = collect_test_cutouts(sample_source_ids, config, gen_type, gen_results)

    if not cutouts_dict:
        raise RuntimeError("No cutouts extracted to save as ground truth")

    zarr_path = output_path if output_path.name.endswith(".zarr") else output_path / "images.zarr"
    _write_cutouts_zarr_archive(sample_df, cutouts_dict, zarr_path, config)
    logger.info(f"Saved {len(cutouts_dict)} ground-truth cutouts → {zarr_path.parent}")


def generate_ground_truth_for_sample(
    sample_df: pd.DataFrame,
    config,
    output_folder: Path,
    gt_path: Path,
) -> float:
    """Generate and save ground-truth cutouts for predefined sample using in-memory StreamingOrchestrator."""
    tmp_sample_path = output_folder / "tmp_gt_sample.parquet"
    tmp_sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(tmp_sample_path, index=False)

    gt_config = DotMap(config.toDict(), _dynamic=False)
    gt_config.source_catalogue = str(tmp_sample_path)
    gt_config.N_batch_cutout_process = max(10, len(sample_df))

    t0 = time.time()
    mem_results = run_mem_streaming(tmp_sample_path, gt_config)
    elapsed = time.time() - t0

    save_ground_truth(sample_df, gt_config, "mem-streaming", mem_results, gt_path)

    if tmp_sample_path.exists():
        try:
            tmp_sample_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove temp file {tmp_sample_path}: {e}")

    return elapsed


def verify_ground_truth(
    ground_truth_path: Path,
    config,
    gen_type: str,
    gen_results: list[dict] | None,
    tolerance: float = 1e-5,
) -> dict:
    """Compare generated cutouts against ground-truth cutouts for predefined sources.

    Returns:
        dict with n_checked, n_passed, n_missing, max_diff, mean_diff, status.
    """
    logger.info(f"Loading ground truth from {ground_truth_path} …")
    gt_cutouts = load_ground_truth(ground_truth_path)

    logger.info(f"Collecting test cutouts for {len(gt_cutouts)} ground-truth sources …")
    test_cutouts = collect_test_cutouts(set(gt_cutouts.keys()), config, gen_type, gen_results)

    diffs: list[float] = []
    n_passed = 0
    missing: list[str] = []

    for sid, gt in gt_cutouts.items():
        if sid not in test_cutouts:
            logger.warning(f"Source {sid} missing from generated output")
            missing.append(sid)
            continue
        test_arr = test_cutouts[sid]
        if test_arr.shape != gt.shape:
            logger.warning(
                f"Source {sid} shape mismatch: test {test_arr.shape} vs ground truth {gt.shape}"
            )
            diffs.append(float("inf"))
            continue

        diff = np.abs(gt.astype(np.float64) - test_arr.astype(np.float64))
        max_diff = float(diff.max())
        diffs.append(max_diff)
        if max_diff <= tolerance:
            n_passed += 1

    status = "PASS" if n_passed == len(gt_cutouts) and len(missing) == 0 else "FAIL"

    return {
        "n_checked": len(gt_cutouts),
        "n_passed": n_passed,
        "n_missing": len(missing),
        "max_diff": max(diffs) if diffs else 0.0,
        "mean_diff": float(np.mean(diffs)) if diffs else 0.0,
        "status": status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary & Markdown Report Generation
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(
    args: argparse.Namespace,
    n_total: int,
    elapsed: float,
    check_stats: dict | None,
    wcs_stats: dict | None,
) -> None:
    """Print a formatted summary of a single release test run."""
    ch_label = CH_IN_OUT_CONFIGS[args.ch_in_out]["label"]
    throughput = n_total / elapsed if elapsed > 0 else 0.0
    output_folder = Path(args.output_folder)

    w = 58
    print("\n" + "═" * w)
    print("  Cutana Release Test Summary")
    print("─" * w)
    print(f"  Catalogue:     {args.catalogue_size} ({n_total:,} sources)")
    print(f"  Gen-type:      {args.gen_type}")
    print(f"  Ch-in-out:     {ch_label}")
    print(f"  Normalisation: {args.normalisation}")
    print(f"  Output format: {args.output_format}")
    print(f"  Resolution:    {args.target_resolution}px")
    print(f"  Data type:     {args.data_type}")
    print(f"  Randomized:    {'yes (seed=42)' if args.randomize else 'no'}")
    print("─" * w)
    print(f"  Total cutouts: {n_total:,}")
    print(f"  Total time:    {elapsed:.1f} s")
    print(f"  Throughput:    {throughput:.1f} cutout/s")
    if check_stats is not None:
        print("─" * w)
        n_checked = check_stats["n_checked"]
        n_passed = check_stats["n_passed"]
        status = "✓ PASS" if check_stats["status"] == "PASS" else "✗ FAIL"
        print(f"  Ground-Truth:  {n_checked} sources verified")
        print(f"  Result:        {status}  ({n_passed}/{n_checked} passed)")
        if check_stats["n_missing"] > 0:
            print(f"  Missing:       {check_stats['n_missing']} sources")
        print(f"  Max |Δpixel|:  {check_stats['max_diff']:.5f}")
        print(f"  Mean|Δpixel|:  {check_stats['mean_diff']:.5f}")
    if wcs_stats is not None:
        print("─" * w)
        status = "✓ PASS" if wcs_stats["status"] == "PASS" else "✗ FAIL"
        print(
            f"  WCS check:     {wcs_stats['n_hdus']} cutout HDUs from "
            f"{wcs_stats['n_sources']} sources vs parent tiles"
        )
        print(f"  Result:        {status}  ({wcs_stats['n_failures']} issues)")
        if wcs_stats["n_missing"] > 0:
            print(f"  Missing:       {wcs_stats['n_missing']} sources")
        if wcs_stats["n_resize_unchecked"] > 0:
            print(f"  No size card:  {wcs_stats['n_resize_unchecked']} HDUs, resize not checked")
        print(f"  Affine spread: {wcs_stats['max_affine_spread']:.5f} parent px")
        print(f"  Origin resid.: {wcs_stats['max_origin_residual']:.5f} parent px")
        print(f"  Source offset: {wcs_stats['max_source_offset']:.3f} parent px")
    print("─" * w)
    print(f"  Cutana output:  {output_folder / 'cutana_output'}")
    print("═" * w + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Throughput baseline
# ─────────────────────────────────────────────────────────────────────────────

_RUN_KEY_FIELDS = (
    "catalogue_size",
    "gen_type",
    "ch_in_out",
    "normalisation",
    "output_format",
    "data_type",
    "target_resolution",
)


def run_key(rec: dict) -> str:
    """Identify a matrix row by its configuration rather than its position.

    The matrix is a covering array, so inserting or reordering a row shifts every
    index after it. Pairing a run with its counterpart in an older release has to
    survive that, which means keying on what the run *is*.

    Args:
        rec: A run record, or any mapping carrying the seven configuration fields.

    Returns:
        A stable key for this configuration.
    """
    return "|".join(str(rec[field]) for field in _RUN_KEY_FIELDS)


def write_throughput_baseline(run_records: list[dict], path: Path) -> None:
    """Save this run's throughputs so the next release can be compared against it.

    Only runs that passed are recorded. A failed run either produced nothing or
    produced the wrong pixels, and in both cases its rate measures something other
    than the work the next release will be doing.

    Args:
        run_records: The matrix run records.
        path: Where to write the baseline JSON.
    """
    runs = {
        run_key(rec): {
            "throughput": rec["throughput"],
            "elapsed": rec["elapsed"],
            # Recorded so a later release can tell whether it is comparing like with
            # like: the worker default is itself a thing releases change, and on a
            # latency-bound filesystem it moves throughput more than the code does.
            "max_workers": rec["max_workers"],
        }
        for rec in run_records
        if rec["status"] == "PASS" and rec["throughput"] is not None
    }
    payload = {
        "cutana_version": cutana.__version__,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "machine": f"{platform.system()} ({platform.machine()})",
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Wrote throughput baseline ({len(runs)} runs) → {path}")


def load_throughput_baseline(path: Path) -> dict | None:
    """Read a previous release's throughputs, or report that there are none.

    A missing baseline is the normal state for the first release that has one, so
    it is not an error; the report simply leaves the comparison columns out.

    Args:
        path: The baseline JSON to read.

    Returns:
        The parsed baseline, or None if the file does not exist.

    Raises:
        ValueError: If the file exists but is not a baseline this script wrote.
    """
    if not path.exists():
        logger.info(f"No throughput baseline at {path}; the report will omit the delta columns")
        return None

    baseline = json.loads(path.read_text(encoding="utf-8"))
    # Every top-level field generate_markdown_report reads, so a hand-edited baseline
    # fails here rather than KeyError-ing after the whole matrix has run.
    missing = {"cutana_version", "created", "machine", "runs"} - set(baseline)
    if missing:
        raise ValueError(f"{path} is not a throughput baseline: missing {sorted(missing)}")
    for key, entry in baseline["runs"].items():
        entry_missing = {"throughput", "max_workers"} - set(entry)
        if entry_missing:
            raise ValueError(
                f"{path} run '{key}' is missing {sorted(entry_missing)}; it predates the "
                f"worker-count check and cannot be compared safely. Re-measure the baseline."
            )
    logger.info(
        f"Comparing throughput against cutana {baseline['cutana_version']} "
        f"({len(baseline['runs'])} runs) from {path}"
    )
    return baseline


def _comparable_baseline_entry(rec: dict, baseline: dict) -> dict | None:
    """Return the baseline entry this run may be compared against, or None.

    A run that crashed has no throughput, and a run that failed verification produced
    something other than the work the baseline measured, so neither is a comparison --
    the report must not draw a delta for it nor warn about its worker count. One
    predicate for both so they cannot drift apart.

    Args:
        rec: A matrix run record.
        baseline: A loaded baseline.

    Returns:
        The matching baseline entry, or None when the pair cannot be compared.
    """
    if rec["throughput"] is None or rec["status"] != "PASS":
        return None
    return baseline["runs"].get(run_key(rec))


def format_throughput_delta(rec: dict, baseline: dict) -> tuple[str, str]:
    """Render one run's throughput change against the baseline.

    Args:
        rec: The run record.
        baseline: A loaded baseline.

    Returns:
        The absolute change in cutouts/second and the relative change, each already
        formatted, and each ``-`` where the pair cannot be compared.
    """
    previous = _comparable_baseline_entry(rec, baseline)
    if previous is None:
        return "-", "-"

    absolute = rec["throughput"] - previous["throughput"]
    relative = absolute / previous["throughput"] * 100.0
    return f"{absolute:+.1f}", f"{relative:+.1f}%"


def worker_count_mismatches(run_records: list[dict], baseline: dict) -> list[str]:
    """Find compared pairs that did not run with the same worker count.

    Cutout extraction on a network filesystem is bound by read latency rather than
    CPU, so oversubscribing the machine hides that latency and raises throughput.
    A release that changes the ``max_workers`` default therefore moves these numbers
    on its own, and a delta measured across such a change says nothing about the
    code. The report cannot decide which comparison the reader wanted, but it can
    refuse to present a confounded one silently.

    Args:
        run_records: The matrix run records.
        baseline: A loaded baseline.

    Returns:
        One description per mismatched configuration, in table order.
    """
    mismatches = []
    for index, rec in enumerate(run_records, 1):
        previous = _comparable_baseline_entry(rec, baseline)
        if previous is None or previous["max_workers"] == rec["max_workers"]:
            continue
        mismatches.append(
            f"run {index} ({rec['gen_type']} | {rec['ch_in_out']}): "
            f"{previous['max_workers']} workers then, {rec['max_workers']} now"
        )
    return mismatches


def generate_markdown_report(
    run_records: list[dict],
    report_path: Path,
    baseline: dict | None = None,
) -> None:
    """Generate a GitHub-ready markdown report for release notes / PR comments.

    Args:
        run_records: The matrix run records.
        report_path: Where to write the markdown.
        baseline: A previous release's throughputs. When given, the results table
            gains the change against it; when not, those columns are left out
            rather than filled with blanks.
    """
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_runs = len(run_records)
    passed_runs = sum(1 for r in run_records if r["status"] == "PASS")
    failed_runs = total_runs - passed_runs

    overall_badge = "🟢 **PASSED**" if failed_runs == 0 else "🔴 **FAILED**"

    lines = [
        "# Cutana Pre-Release Test Report",
        "",
        f"**Date:** {now_str}  ",
        f"**Environment:** {platform.system()} ({platform.machine()}) | Python {platform.python_version()} | {os.cpu_count()} CPU cores  ",
        f"**Status:** {overall_badge} ({passed_runs}/{total_runs} runs passed)  ",
        "",
        "## Test Matrix Results",
        "",
        r"| # | Catalogue | Backend | Channels | Norm | Format | Dtype | Res | Cutouts | Time (s) | Throughput (c/s) |"
        + (r" Δ c/s | Δ % |" if baseline else "")
        + r" Verification | Max \|Δ\| | Mean \|Δ\| | WCS |",
        "|---|---|---|---|---|---|---|---|---|---|---|"
        + ("---|---|" if baseline else "")
        + "---|---|---|---|",
    ]

    for i, rec in enumerate(run_records, 1):
        status_icon = "✓ PASS" if rec["status"] == "PASS" else "✗ FAIL"
        if rec["status"] == "SKIPPED":
            status_icon = "⚪ N/A"

        max_diff_str = f"{rec['max_diff']:.5f}" if rec["max_diff"] is not None else "-"
        mean_diff_str = f"{rec['mean_diff']:.5f}" if rec["mean_diff"] is not None else "-"
        time_str = f"{rec['elapsed']:.1f}" if rec["elapsed"] is not None else "-"
        thru_str = f"{rec['throughput']:.1f}" if rec["throughput"] is not None else "-"
        wcs_stats = rec["wcs_stats"]
        if wcs_stats is None:
            wcs_str = "-"
        elif wcs_stats["status"] == "PASS":
            wcs_str = f"✓ {wcs_stats['n_hdus']} HDUs"
        else:
            wcs_str = f"✗ {wcs_stats['n_failures']} issues"

        if baseline:
            delta_abs, delta_rel = format_throughput_delta(rec, baseline)
            delta_cells = f" {delta_abs} | {delta_rel} |"
        else:
            delta_cells = ""

        row = (
            f"| {i} | {rec['catalogue_size']} | `{rec['gen_type']}` | {rec['ch_in_out']} | "
            f"{rec['normalisation']} | {rec['output_format']} | `{rec['data_type']}` | "
            f"{rec['target_resolution']}px | {rec['n_total']:,} | {time_str} | "
            f"{thru_str} |{delta_cells} {status_icon} | {max_diff_str} | {mean_diff_str} | {wcs_str} |"
        )
        lines.append(row)

    if baseline:
        lines.extend(
            [
                "",
                f"Δ columns compare throughput against cutana "
                f"{baseline['cutana_version']}, measured on {baseline['machine']} "
                f"at {baseline['created']}. Positive is faster than that release. "
                f"A `-` means the configuration has no counterpart there — it is new, "
                f"or it could not run.",
            ]
        )
        mismatches = worker_count_mismatches(run_records, baseline)
        if mismatches:
            lines.extend(
                [
                    "",
                    "> [!WARNING]",
                    "> **The worker count differs between the two runs, so these deltas "
                    "measure the default as well as the code.** Extraction here is bound by "
                    "read latency rather than CPU, so a higher worker count raises throughput "
                    "on its own. Re-measure both sides with `--max-workers` pinned to compare "
                    "the code alone.",
                    ">",
                ]
            )
            lines.extend(f"> - {mismatch}" for mismatch in mismatches)

    # The three WCS diagnostics reach the terminal through print_summary(), which only the
    # single-run branch calls -- so in --matrix mode, the exact mode the release checklist
    # prescribes, the report showed a tick and an HDU count and nothing else. max_source_offset
    # in particular is the direct signature of issue #240: a distribution collapsing to zero
    # means every source was forced to the cutout centre.
    wcs_rows = [(i, rec) for i, rec in enumerate(run_records, 1) if rec["wcs_stats"] is not None]
    if wcs_rows:
        lines.extend(
            [
                "",
                "### WCS Check Diagnostics",
                "",
                "Worst residual over the sampled cutouts of each run, in parent-tile pixels. "
                "A source offset that collapses to zero means the cutout WCS put every source "
                "at the centre (issue #240); an affine spread that grows with the frame means "
                "the projection was re-tangented (issue #390). *Resize unchecked* counts HDUs "
                "whose header records no catalogue size, so the resize cross-check could not "
                "run on them.",
                "",
                "| # | Sources | HDUs | Missing | Resize unchecked | Max affine spread | "
                "Max origin residual | Max source offset |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for i, rec in wcs_rows:
            wcs_stats = rec["wcs_stats"]
            lines.append(
                f"| {i} | {wcs_stats['n_sources']} | {wcs_stats['n_hdus']} | "
                f"{wcs_stats['n_missing']} | {wcs_stats['n_resize_unchecked']} | "
                f"{wcs_stats['max_affine_spread']:.5f} | "
                f"{wcs_stats['max_origin_residual']:.5f} | "
                f"{wcs_stats['max_source_offset']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Summary",
            f"- **Total Configurations Tested:** {total_runs}",
            f"- **Passed:** {passed_runs}",
            f"- **Failed:** {failed_runs}",
            "",
            "---",
            "*Report generated automatically by `tests/release/release_test.py`*",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote GitHub release markdown report → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────


def cleanup_temp_files() -> None:
    """Remove temporary files created during the run (e.g. randomized catalogue)."""
    for path in _TMP_FILES:
        if path.exists():
            try:
                path.unlink()
                logger.debug(f"Removed temp file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {path}: {e}")
    _TMP_FILES.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Single Test Execution
# ─────────────────────────────────────────────────────────────────────────────


def run_single_test(
    catalogue_size: str,
    gen_type: str,
    ch_in_out: str,
    normalisation: str,
    output_format: str,
    data_type: str,
    target_resolution: int,
    randomize: bool,
    combination_matrix: str | None,
    output_folder: Path,
    ground_truth_arg: str | None,
    generate_ground_truth_flag: bool,
    skip_verification_flag: bool,
    skip_wcs_check_flag: bool,
    tolerance_arg: float | None,
    max_workers: int | None = None,
) -> dict:
    """Execute a single release test configuration and return record for reporting."""
    # ── Channel config ────────────────────────────────────────────────────────
    ch_cfg = deepcopy(CH_IN_OUT_CONFIGS[ch_in_out])
    if combination_matrix is not None:
        ch_cfg["channel_weights"] = apply_combination_matrix_override(combination_matrix, ch_cfg)
        logger.info("Applied custom combination matrix override")

    # ── Catalogue ─────────────────────────────────────────────────────────────
    df, catalogue_path = load_catalogue(catalogue_size, randomize, output_folder)
    n_total = len(df)

    # ── Config ────────────────────────────────────────────────────────────────
    run_output_dir = output_folder / "cutana_output"
    config = build_config(
        catalogue_path=catalogue_path,
        output_dir=run_output_dir,
        output_format=output_format,
        target_resolution=target_resolution,
        normalisation=normalisation,
        ch_cfg=ch_cfg,
        data_type=data_type,
        max_workers=max_workers,
    )

    gt_path = (
        Path(ground_truth_arg)
        if ground_truth_arg is not None
        else get_ground_truth_path(
            catalogue_size=catalogue_size,
            ch_in_out=ch_in_out,
            normalisation=normalisation,
            data_type=data_type,
            target_resolution=target_resolution,
        )
    )

    # ── Fast Ground Truth Generation Path ─────────────────────────────────────
    if generate_ground_truth_flag:
        sample_df = select_sample(df, n_sources=PREDEFINED_SAMPLE_SIZE)
        logger.info(
            f"Generating ground truth for {len(sample_df)} stratified sources using StreamingOrchestrator …"
        )
        elapsed = generate_ground_truth_for_sample(sample_df, config, output_folder, gt_path)
        throughput = len(sample_df) / elapsed if elapsed > 0 else 0.0
        logger.info(f"GT generation complete in {elapsed:.2f}s ({throughput:.1f} cutout/s)")

        return {
            "catalogue_size": catalogue_size,
            "gen_type": "mem-streaming (GT)",
            "ch_in_out": ch_in_out,
            "normalisation": normalisation,
            "output_format": output_format,
            "data_type": data_type,
            "target_resolution": target_resolution,
            "n_total": len(sample_df),
            "elapsed": elapsed,
            "throughput": throughput,
            "check_stats": None,
            "wcs_stats": None,
            "status": "PASS",
            "max_diff": None,
            "mean_diff": None,
        }

    # ── Main generation run ───────────────────────────────────────────────────
    logger.info(f"Starting generation: {gen_type} on {n_total:,} sources")
    t0 = time.time()
    gen_results = run_generation(df, config, gen_type, catalogue_path)
    elapsed = time.time() - t0
    throughput = n_total / elapsed if elapsed > 0 else 0.0
    logger.info(f"Generation complete in {elapsed:.1f}s ({throughput:.1f} cutout/s)")

    # ── Ground Truth Verification ─────────────────────────────────────────────
    # Resolve tolerance: default 0.0 for uint8, 1e-4 for float32
    if tolerance_arg is not None:
        tolerance = tolerance_arg
    else:
        tolerance = 0.0 if data_type == "uint8" else 1e-4

    check_stats = None
    if not skip_verification_flag:
        logger.info(f"Running ground-truth verification against {gt_path} …")
        check_stats = verify_ground_truth(
            ground_truth_path=gt_path,
            config=config,
            gen_type=gen_type,
            gen_results=gen_results,
            tolerance=tolerance,
        )

    # ── Cutout WCS Verification ───────────────────────────────────────────────
    # Only FITS output carries a per-cutout WCS, and only the disk backends write it.
    wcs_stats = None
    if (
        not skip_wcs_check_flag
        and output_format == "fits"
        and gen_type in ("disk", "disk-streaming")
    ):
        logger.info("Verifying cutout WCS against the parent tile headers …")
        # Its own stratified draw, not the ground-truth set: under --randomize the row
        # order changes and the two samples diverge. Any sources in the output will do,
        # and stratifying by size keeps the range of resize factors covered.
        wcs_sample = select_sample(df, PREDEFINED_SAMPLE_SIZE)
        wcs_stats = verify_cutout_wcs(
            output_dir=Path(config.output_dir),
            catalogue_df=df,
            selected_extensions=ch_cfg["selected_extensions"],
            source_ids=set(wcs_sample["SourceID"].astype(str)),
            padding_factor=config.padding_factor,
        )
        for failure in wcs_stats["failures"]:
            logger.error(f"WCS check: {failure}")

    run_status = "PASS"
    if check_stats is not None and check_stats["status"] != "PASS":
        run_status = "FAIL"
    if wcs_stats is not None and wcs_stats["status"] != "PASS":
        run_status = "FAIL"

    return {
        "catalogue_size": catalogue_size,
        "gen_type": gen_type,
        "ch_in_out": ch_in_out,
        "normalisation": normalisation,
        "output_format": output_format,
        "data_type": data_type,
        "target_resolution": target_resolution,
        # Read after the run, not before: the LoadBalancer rewrites config.max_workers
        # while it works (7 on the disk backends here, from the N-1 rule), and the count
        # the run actually used is the one worth comparing against another release.
        "max_workers": config.max_workers,
        "n_total": n_total,
        "elapsed": elapsed,
        "throughput": throughput,
        "check_stats": check_stats,
        "wcs_stats": wcs_stats,
        "status": run_status,
        "max_diff": check_stats["max_diff"] if check_stats is not None else None,
        "mean_diff": check_stats["mean_diff"] if check_stats is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_level="INFO",
        console_level="WARNING",
        log_dir=str(output_folder),
    )

    report_path = (
        Path(args.report_file)
        if args.report_file is not None
        else output_folder / "release_test_report.md"
    )

    baseline_path = Path(args.baseline) if args.baseline is not None else DEFAULT_BASELINE_PATH
    baseline = load_throughput_baseline(baseline_path)

    # ── Pre-run Validation: Verify Ground Truth Archives Exist ────────────────
    if not args.generate_ground_truth and not args.skip_verification:
        if args.matrix:
            configs_to_validate = RELEASE_TEST_MATRIX
        else:
            configs_to_validate = [
                {
                    "catalogue_size": args.catalogue_size,
                    "ch_in_out": args.ch_in_out,
                    "normalisation": args.normalisation,
                    "data_type": args.data_type,
                    "target_resolution": args.target_resolution,
                }
            ]
        try:
            logger.info("Pre-validating ground-truth archives availability …")
            validate_ground_truths_exist(configs_to_validate, args.ground_truth)
            logger.info("All required ground-truth archives exist.")
        except FileNotFoundError as e:
            logger.error(f"Ground-truth pre-validation failed:\n{e}")
            return 1

    run_records: list[dict] = []
    overall_exit_code = 0

    if args.matrix:
        logger.info(f"Starting Cutana Release Test Matrix ({len(RELEASE_TEST_MATRIX)} runs) …")
        for i, item in enumerate(RELEASE_TEST_MATRIX, 1):
            sub_folder = output_folder / f"run_{i:02d}_{item['gen_type']}_{item['ch_in_out']}"
            sub_folder.mkdir(parents=True, exist_ok=True)
            run_log_file = sub_folder / "cutana.log"
            run_log_handler_id = logger.add(
                str(run_log_file),
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                colorize=False,
            )
            logger.info(
                f"\n--- Matrix Run {i}/{len(RELEASE_TEST_MATRIX)}: "
                f"{item['gen_type']} | {item['ch_in_out']} | {item['normalisation']} | "
                f"{item['output_format']} | {item['data_type']} ---"
            )
            try:
                rec = run_single_test(
                    catalogue_size=item["catalogue_size"],
                    gen_type=item["gen_type"],
                    ch_in_out=item["ch_in_out"],
                    normalisation=item["normalisation"],
                    output_format=item["output_format"],
                    data_type=item["data_type"],
                    target_resolution=item["target_resolution"],
                    randomize=args.randomize,
                    combination_matrix=args.combination_matrix,
                    output_folder=sub_folder,
                    ground_truth_arg=args.ground_truth,
                    generate_ground_truth_flag=args.generate_ground_truth,
                    skip_verification_flag=args.skip_verification,
                    skip_wcs_check_flag=args.skip_wcs_check,
                    tolerance_arg=args.tolerance,
                    max_workers=args.max_workers,
                )
                run_records.append(rec)
                if rec["status"] == "FAIL":
                    overall_exit_code = 1
                    logger.warning(
                        f"Run {i} failed verification; diagnostic logs retained in {run_log_file}"
                    )
                else:
                    # Clean up passed run's cutana_output to conserve disk space
                    run_output_dir = sub_folder / "cutana_output"
                    if run_output_dir.exists():
                        shutil.rmtree(run_output_dir, ignore_errors=True)
                        logger.debug(f"Cleaned up output directory: {run_output_dir}")
                    # Remove the per-run log file for passed runs
                    if run_log_handler_id is not None:
                        logger.remove(run_log_handler_id)
                        run_log_handler_id = None
                    if run_log_file.exists():
                        run_log_file.unlink(missing_ok=True)
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Matrix run {i} crashed: {e}\n{tb_str}")
                logger.warning(f"Crash logs for run {i} retained in {run_log_file}")
                run_records.append(
                    {
                        "catalogue_size": item["catalogue_size"],
                        "gen_type": item["gen_type"],
                        "ch_in_out": item["ch_in_out"],
                        "normalisation": item["normalisation"],
                        "output_format": item["output_format"],
                        "data_type": item["data_type"],
                        "target_resolution": item["target_resolution"],
                        "max_workers": args.max_workers,
                        "n_total": 0,
                        "elapsed": None,
                        "throughput": None,
                        "check_stats": None,
                        "wcs_stats": None,
                        "status": "FAIL",
                        "max_diff": None,
                        "mean_diff": None,
                    }
                )
                overall_exit_code = 1
            finally:
                if run_log_handler_id is not None:
                    try:
                        logger.remove(run_log_handler_id)
                    except ValueError:
                        pass
    else:
        single_log_file = output_folder / "cutana.log"
        single_log_handler_id = logger.add(
            str(single_log_file),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            colorize=False,
        )
        try:
            rec = run_single_test(
                catalogue_size=args.catalogue_size,
                gen_type=args.gen_type,
                ch_in_out=args.ch_in_out,
                normalisation=args.normalisation,
                output_format=args.output_format,
                data_type=args.data_type,
                target_resolution=args.target_resolution,
                randomize=args.randomize,
                combination_matrix=args.combination_matrix,
                output_folder=output_folder,
                ground_truth_arg=args.ground_truth,
                generate_ground_truth_flag=args.generate_ground_truth,
                skip_verification_flag=args.skip_verification,
                skip_wcs_check_flag=args.skip_wcs_check,
                tolerance_arg=args.tolerance,
                max_workers=args.max_workers,
            )
            run_records.append(rec)
            print_summary(
                args, rec["n_total"], rec["elapsed"], rec["check_stats"], rec["wcs_stats"]
            )
            if rec["status"] == "FAIL":
                overall_exit_code = 1
                logger.warning(f"Test failed verification; logs retained in {single_log_file}")
            else:
                if single_log_handler_id is not None:
                    logger.remove(single_log_handler_id)
                    single_log_handler_id = None
                if single_log_file.exists():
                    single_log_file.unlink(missing_ok=True)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Release test failed: {e}\n{tb_str}")
            logger.warning(f"Crash logs retained in {single_log_file}")
            cleanup_temp_files()
            return 1
        finally:
            if single_log_handler_id is not None:
                try:
                    logger.remove(single_log_handler_id)
                except ValueError:
                    pass

    # ── Write Markdown Report ─────────────────────────────────────────────────
    generate_markdown_report(run_records, report_path, baseline)

    # Written for every run, not only the release one: the comparison is only fair
    # between two runs of the same machine, so whoever measures the next release wants
    # this run's numbers to hand rather than a figure copied out of an old report.
    # A ground-truth pass runs a 50-source sample, which is not a throughput
    # measurement; its records carry no max_workers either.
    if args.matrix and not args.generate_ground_truth:
        write_throughput_baseline(run_records, output_folder / BASELINE_FILENAME)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cleanup_temp_files()

    return overall_exit_code


if __name__ == "__main__":
    sys.exit(main())
