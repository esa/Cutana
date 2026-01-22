#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Naive Astropy Cutout2D baseline implementation for benchmarking.

This provides a simple reference implementation using astropy.nddata.Cutout2D
with memory-mapped FITS loading, similar to cutana/fits_reader.py approach.

Now includes realistic processing steps:
- Cutout extraction
- Resizing to target resolution
- Flux conversion (AB magnitude)
- Normalization (0-1 range)
- Writing to individual FITS files
- Detailed timing for each step

IMPORTANT: Thread control via environment variables must be set BEFORE importing
numpy, scipy, skimage, etc. All scientific computing imports are done inside
process_catalogue_astropy() after setting environment variables.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict

import toml
from loguru import logger

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def load_baseline_config() -> Dict:
    """Load Astropy baseline configuration from benchmark_config.toml."""
    config_path = Path(__file__).parent / "benchmark_config.toml"
    if config_path.exists():
        config = toml.load(config_path)
        return config.get("astropy_baseline", {})
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")


def process_catalogue_astropy(
    catalogue_df,
    fits_extension: str = "PRIMARY",
    target_resolution: int = 256,
    apply_flux_conversion: bool = False,
    interpolation: str = "bilinear",
    output_dir: Path = None,
    zeropoint_keyword: str = "ABMAGLIM",
    process_threads: int = None,
) -> Dict[str, any]:
    """
    Process entire catalogue using naive Astropy approach with full pipeline.

    This is a simple sequential implementation without:
    - Parallel processing
    - FITS set optimization
    - Memory management optimization

    Now includes realistic processing steps:
    - Cutout extraction
    - Resizing to target resolution
    - Flux conversion (optional)
    - Normalization (0-1 range)
    - Writing to individual FITS files
    - Detailed timing for each step

    Args:
        catalogue_df: Source catalogue DataFrame
        fits_extension: FITS extension to process
        target_resolution: Target size for resizing (in pixels)
        apply_flux_conversion: Whether to apply flux conversion
        interpolation: Interpolation method for resizing
        output_dir: Directory to write FITS files (temporary, will be cleaned up)
        zeropoint_keyword: FITS header keyword for AB zeropoint
        process_threads: Number of threads to use (1, 4, etc.)

    Returns:
        Dictionary with results and timing breakdown
    """
    # =========================================================================
    # STEP 1: Set thread limits BEFORE any numpy/scipy imports
    # NOTE: When called via run_astropy_subprocess.py, CPU affinity is already
    # set at process level (more reliable on Windows). These env vars provide
    # additional thread control and backwards compatibility for direct testing.
    # =========================================================================
    if process_threads is not None:
        thread_env_vars = {
            "OMP_NUM_THREADS": str(process_threads),
            "MKL_NUM_THREADS": str(process_threads),
            "OPENBLAS_NUM_THREADS": str(process_threads),
            "NUMBA_NUM_THREADS": str(process_threads),
            "VECLIB_MAXIMUM_THREADS": str(process_threads),
            "NUMEXPR_NUM_THREADS": str(process_threads),
        }

        for var, value in thread_env_vars.items():
            os.environ[var] = value

        logger.info(
            f"Set thread limit to {process_threads} via environment variables "
            f"(BEFORE numpy/scipy imports). CPU affinity set at subprocess level."
        )

    # =========================================================================
    # STEP 2: NOW import numpy, scipy, astropy, skimage (AFTER setting env vars)
    # =========================================================================
    import ast
    import shutil

    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from skimage import transform

    from cutana.constants import JANSKY_AB_ZEROPONT

    logger.info(
        f"Processing {len(catalogue_df)} sources with Astropy baseline "
        f"(full pipeline, {process_threads} threads)"
    )

    # Create output directory for temporary FITS files
    if output_dir is None:
        output_dir = Path("./astropy_baseline_output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    cutouts = []
    errors = []

    # Timing breakdown for each step
    timing = {
        "fits_loading": 0.0,
        "cutout_extraction": 0.0,
        "resizing": 0.0,
        "flux_conversion": 0.0,
        "normalization": 0.0,
        "fits_writing": 0.0,
    }

    # Cache for loaded FITS files to avoid reloading
    fits_cache = {}

    for idx, source in catalogue_df.iterrows():
        try:
            source_id = source["SourceID"]
            ra = source["RA"]
            dec = source["Dec"]
            diameter_pixel = source["diameter_pixel"]

            # Parse fits_file_paths (stored as string representation of list)
            fits_paths_str = source["fits_file_paths"]
            if isinstance(fits_paths_str, str):
                fits_paths = ast.literal_eval(fits_paths_str)
            else:
                fits_paths = fits_paths_str

            # For multi-channel, just take first FITS file for baseline
            # (simplification - real Cutana handles all channels)
            fits_path = fits_paths[0] if isinstance(fits_paths, list) else fits_paths

            # Step 1: Load FITS file (use cache if available)
            t0 = time.time()
            if fits_path not in fits_cache:
                try:
                    # Load FITS with memory mapping
                    hdul = fits.open(fits_path, memmap=True, lazy_load_hdus=True)
                    if fits_extension == "PRIMARY":
                        header = hdul[0].header
                    else:
                        header = hdul[fits_extension].header
                    wcs = WCS(header)
                    fits_cache[fits_path] = (hdul, wcs, header)
                    timing["fits_loading"] += time.time() - t0
                except Exception as e:
                    logger.error(f"Failed to load FITS file {fits_path}: {e}")
                    errors.append({"source_id": source_id, "error": str(e)})
                    continue
            else:
                hdul, wcs, header = fits_cache[fits_path]
                timing["fits_loading"] += time.time() - t0

            # Step 2: Extract cutout using Cutout2D
            t0 = time.time()
            if fits_extension == "PRIMARY":
                data = hdul[0].data
            else:
                data = hdul[fits_extension].data

            position = SkyCoord(ra * u.degree, dec * u.degree, frame="icrs")
            cutout = Cutout2D(
                data,
                position,
                size=(diameter_pixel, diameter_pixel),
                wcs=wcs,
                mode="partial",
                fill_value=0.0,
            )
            cutout_data = cutout.data
            timing["cutout_extraction"] += time.time() - t0

            # Step 3: Resize cutout
            t0 = time.time()
            if cutout_data.shape[:2] != (target_resolution, target_resolution):
                # Map interpolation methods
                if interpolation == "nearest":
                    order = 0
                elif interpolation == "bilinear":
                    order = 1
                elif interpolation == "biquadratic":
                    order = 2
                elif interpolation == "bicubic":
                    order = 3
                else:
                    order = 1

                resized_cutout = transform.resize(
                    cutout_data,
                    (target_resolution, target_resolution),
                    order=order,
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(cutout_data.dtype)
            else:
                resized_cutout = cutout_data.copy()
            timing["resizing"] += time.time() - t0

            # Step 4: Flux conversion (if enabled)
            t0 = time.time()
            if apply_flux_conversion:
                zeropoint = header.get(zeropoint_keyword, None)
                if zeropoint is not None:
                    flux_converted = resized_cutout * 10 ** (-0.4 * zeropoint) * JANSKY_AB_ZEROPONT
                else:
                    flux_converted = resized_cutout
            else:
                flux_converted = resized_cutout
            timing["flux_conversion"] += time.time() - t0

            # Step 5: Normalize to 0-1 range
            t0 = time.time()
            img_min, img_max = flux_converted.min(), flux_converted.max()
            if img_max > img_min:
                normalized_cutout = (flux_converted - img_min) / (img_max - img_min)
            else:
                normalized_cutout = np.zeros_like(flux_converted)
            timing["normalization"] += time.time() - t0

            # Step 6: Write to FITS file
            t0 = time.time()
            output_path = output_dir / f"{source_id}_cutout.fits"
            hdu = fits.PrimaryHDU(data=normalized_cutout)
            # Copy basic WCS information
            for key in [
                "CRVAL1",
                "CRVAL2",
                "CRPIX1",
                "CRPIX2",
                "CD1_1",
                "CD1_2",
                "CD2_1",
                "CD2_2",
                "CTYPE1",
                "CTYPE2",
            ]:
                if key in header:
                    hdu.header[key] = header[key]
            hdul_out = fits.HDUList([hdu])
            hdul_out.writeto(output_path, overwrite=True)
            hdul_out.close()
            timing["fits_writing"] += time.time() - t0

            cutouts.append(
                {
                    "source_id": source_id,
                    "output_path": str(output_path),
                    "shape": normalized_cutout.shape,
                }
            )

            # Progress logging every 1000 sources
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(catalogue_df):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                progress_pct = (idx + 1) / len(catalogue_df) * 100
                logger.info(
                    f"Progress: {idx + 1}/{len(catalogue_df)} sources "
                    f"({progress_pct:.1f}%) - {rate:.1f} sources/sec"
                )

        except Exception as e:
            logger.error(f"Error processing source {source_id}: {e}")
            errors.append({"source_id": source_id, "error": str(e)})

    # Close all cached FITS files
    for hdul, _, _ in fits_cache.values():
        hdul.close()

    end_time = time.time()
    total_time = end_time - start_time
    sources_per_second = len(cutouts) / total_time if total_time > 0 else 0

    # Clean up temporary FITS files
    logger.info(f"Cleaning up temporary FITS files in {output_dir}")
    try:
        shutil.rmtree(output_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory {output_dir}: {e}")

    results = {
        "total_sources": len(catalogue_df),
        "successful_cutouts": len(cutouts),
        "errors": len(errors),
        "total_time_seconds": total_time,
        "sources_per_second": sources_per_second,
        "method": "astropy_baseline",
        "fits_extension": fits_extension,
        "timing_breakdown": timing,
    }

    logger.info(f"Astropy baseline completed:")
    logger.info(f"  Total time: {total_time:.2f} seconds")
    logger.info(f"  Sources per second: {sources_per_second:.2f}")
    logger.info(f"  Successful: {len(cutouts)}, Errors: {len(errors)}")
    logger.info(f"  Timing breakdown:")
    for step, step_time in timing.items():
        logger.info(f"    {step}: {step_time:.2f}s ({step_time/total_time*100:.1f}%)")

    return results


def main():
    """Test the Astropy baseline implementation."""
    import pandas as pd

    from cutana.logging_config import setup_logging
    from paper_scripts.plots import create_timing_breakdown_chart

    setup_logging(log_level="INFO", console_level="INFO")

    # Load config
    config = load_baseline_config()

    # Test with small sample
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"

    reference_csv = data_dir / "benchmark_input_10k_cutouts_nirh_nirj_niry_vis.csv"

    if not reference_csv.exists():
        logger.error(f"Reference catalogue not found: {reference_csv}")
        sys.exit(1)

    # Load catalogue and take small sample for testing
    catalogue_df = pd.read_csv(reference_csv)
    sample_df = catalogue_df.head(100)  # Test with 100 sources

    logger.info(f"Testing Astropy baseline with {len(sample_df)} sources")
    logger.info(f"Configuration: {config}")

    results = process_catalogue_astropy(
        sample_df,
        fits_extension="PRIMARY",
        target_resolution=config["target_resolution"],
        apply_flux_conversion=config["apply_flux_conversion"],
        interpolation=config["interpolation"],
        zeropoint_keyword=config["zeropoint_keyword"],
        process_threads=1,  # Test with 1 thread
    )

    logger.info("Test completed successfully!")
    logger.info(f"Results: {results}")

    # Create timing breakdown chart
    if "timing_breakdown" in results:
        output_dir = Path("./test_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output_dir / "astropy_baseline_timing.png"
        create_timing_breakdown_chart(
            results["timing_breakdown"], chart_path, "Astropy Baseline Timing Breakdown"
        )


if __name__ == "__main__":
    main()
