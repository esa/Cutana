#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Consolidated Euclid-compliant Mock Test Data Generator for Cutana

This script generates mock astronomical test data following ESA Euclid mission formats
as documented in docs/euclid_fits.md. It replaces all previous mock data generation scripts.

Features:
- Euclid-compliant FITS file formats (catalogs and mosaics)
- Realistic astronomical data with proper WCS headers
- Multiple tile datasets for comprehensive testing
- Cutana-compatible CSV catalogues with absolute file paths
- Configurable data sizes and parameters

Usage:
    python generate_test_data.py                    # Generate all data sizes
    python generate_test_data.py --size small       # Generate only small dataset
    python generate_test_data.py --tiles 1          # Generate single tile
    python generate_test_data.py --help             # Show help
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from loguru import logger


def setup_logging(verbose: bool = False):
    """Configure logging for the script."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def generate_euclid_filename(data_type: str, tile_id: str, instrument: str = "VIS") -> str:
    """Generate Euclid-compliant FITS filename."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"

    if data_type == "catalog":
        checksum = "CC66F6"  # Mock catalog checksum
        return f"EUC_MER_FINAL-CAT_TILE{tile_id}-{checksum}_{timestamp}_00.00.fits"
    elif data_type == "mosaic":
        checksum = "ACBD03"  # Mock mosaic checksum
        return f"EUC_MER_BGSUB-MOSAIC-{instrument}_TILE{tile_id}-{checksum}_{timestamp}_00.00.fits"
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def create_euclid_wcs(ra_center: float, dec_center: float, size: Tuple[int, int]) -> WCS:
    """Create Euclid-compliant WCS object."""
    wcs = WCS(naxis=2)

    # Euclid VIS pixel scale: ~0.1 arcsec/pixel
    pixel_scale = 0.1 / 3600.0  # Convert to degrees

    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_center, dec_center]
    wcs.wcs.crpix = [size[1] / 2, size[0] / 2]  # Center of image
    wcs.wcs.cd = [[-pixel_scale, 0], [0, pixel_scale]]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.equinox = 2000.0

    return wcs


def create_realistic_astronomical_image(
    size: Tuple[int, int],
    num_sources: int = None,
    source_positions: List[Tuple[float, float]] = None,
) -> np.ndarray:
    """Create realistic background-subtracted astronomical image data with optional white pixel sources."""
    if num_sources is None:
        num_sources = int(size[0] * size[1] / 100000)  # ~1 source per 100k pixels

    logger.debug(f"Creating {size[0]}x{size[1]} image with ~{num_sources} sources")

    # Background-subtracted noise (mean ~0) - start with black background
    data = np.random.normal(0, 0.005, size).astype(np.float32)  # Very low noise for testing

    # If source positions are provided, place bright white pixels at those locations
    if source_positions:
        logger.info(f"Placing white pixels at {len(source_positions)} specified source positions")
        for ra_pixel, dec_pixel in source_positions:
            # Convert to integer pixel coordinates
            x_pixel = int(round(ra_pixel))
            y_pixel = int(round(dec_pixel))

            # Check bounds
            if 0 <= x_pixel < size[1] and 0 <= y_pixel < size[0]:
                # Place a bright white cross pattern for easy identification
                data[y_pixel, x_pixel] = 10000.0  # Very bright center pixel

                # Add surrounding bright pixels to make it easily detectable
                for dy in [-2, -1, 0, 1, 2]:
                    for dx in [-2, -1, 0, 1, 2]:
                        ny, nx = y_pixel + dy, x_pixel + dx
                        if 0 <= nx < size[1] and 0 <= ny < size[0]:
                            # Create a Gaussian-like pattern
                            distance = np.sqrt(dx**2 + dy**2)
                            if distance <= 10.0:
                                brightness = 10000.0 * np.exp(-(distance**2) / 2.0)
                                data[ny, nx] = max(data[ny, nx], brightness)
    else:
        # Add realistic astronomical sources at random positions
        for _ in range(num_sources):
            x = np.random.randint(100, size[1] - 100)
            y = np.random.randint(100, size[0] - 100)

            # PSF parameters (Euclid-like)
            sigma = np.random.uniform(2.0, 4.0)  # ~0.2-0.4 arcsec FWHM
            amplitude = np.random.lognormal(5.0, 1.5)  # Log-normal flux distribution

            # Create PSF (Gaussian approximation)
            extent = int(5 * sigma)
            y_start, y_end = max(0, y - extent), min(size[0], y + extent)
            x_start, x_end = max(0, x - extent), min(size[1], x + extent)

            y_grid, x_grid = np.ogrid[y_start:y_end, x_start:x_end]
            gaussian = amplitude * np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma**2))

            data[y_start:y_end, x_start:x_end] += gaussian

    return data


def create_euclid_mosaic_fits(
    output_path: Path,
    tile_id: str,
    ra_center: float,
    dec_center: float,
    size: Tuple[int, int] = (2000, 2000),
    instrument: str = "VIS",
    source_positions: List[Tuple[float, float]] = None,
) -> str:
    """Create Euclid-compliant mosaic FITS file."""
    filename = generate_euclid_filename("mosaic", tile_id, instrument)
    filepath = output_path / filename

    logger.info(f"Creating Euclid mosaic: {filename}")

    # Create realistic image data with optional source positions
    data = create_realistic_astronomical_image(size, source_positions=source_positions)

    # Create primary HDU with data
    primary_hdu = fits.PrimaryHDU(data)
    header = primary_hdu.header

    # Add Euclid-compliant WCS
    wcs = create_euclid_wcs(ra_center, dec_center, size)
    header.update(wcs.to_header())

    # Add Euclid-specific headers
    header["TELESCOP"] = "EUCLID"
    header["INSTRUME"] = instrument
    header["TILEID"] = tile_id
    header["DATE-OBS"] = datetime.utcnow().isoformat() + "Z"
    header["BUNIT"] = "electron/s"
    header["DATATYPE"] = "BGSUB-MOSAIC"
    header["ORIGIN"] = "Cutana Mock Data Generator"
    header["COMMENT"] = "Mock Euclid-format data for testing"

    # Add data quality information
    header["EXPTIME"] = 565.0  # Typical Euclid VIS exposure time
    header["GAIN"] = 3.1  # Typical VIS gain
    header["READNOIS"] = 4.2  # Read noise in electrons

    # Add MAGZERO header for flux conversion compatibility
    # Use instrument-specific AB magnitude zeropoints matching flux_conversion.py defaults
    if instrument == "VIS":
        header["MAGZERO"] = 24.6  # Typical zeropoint
    elif instrument == "NIR_H":
        header["MAGZERO"] = 29.9  # Typical zeropoint
    elif instrument == "NIR_J":
        header["MAGZERO"] = 30.0  # Typical zeropoint
    elif instrument == "NIR_Y":
        header["MAGZERO"] = 29.8  # Typical zeropoint
    else:
        header["MAGZERO"] = 24.6  # Default zeropoint

    # Write FITS file
    primary_hdu.writeto(filepath, overwrite=True)

    return str(filepath)


def create_euclid_catalog_fits(
    output_path: Path,
    tile_id: str,
    ra_center: float,
    dec_center: float,
    mosaic_size: Tuple[int, int] = (2000, 2000),
    num_sources: int = 465,
) -> str:
    """Create Euclid-compliant catalog FITS file."""
    filename = generate_euclid_filename("catalog", tile_id)
    filepath = output_path / filename

    logger.info(f"Creating Euclid catalog: {filename} ({num_sources} sources)")

    # Generate realistic source catalog
    # Calculate actual field of view based on image size and pixel scale
    pixel_scale_deg = 0.1 / 3600.0  # 0.1 arcsec/pixel in degrees
    field_of_view_deg = mosaic_size[0] * pixel_scale_deg  # Use image size for FOV

    # Generate source positions (uniform distribution within actual image bounds)
    # Use 90% of FOV to ensure sources are well within image bounds
    fov_margin = field_of_view_deg * 0.45  # 45% on each side = 90% total
    ra_values = ra_center + np.random.uniform(-fov_margin, fov_margin, num_sources)
    dec_values = dec_center + np.random.uniform(-fov_margin, fov_margin, num_sources)

    # PSF-fitted positions (with small astrometric uncertainties)
    astrometric_error = 0.1 / 3600  # 0.1 arcsec uncertainty
    ra_psf = ra_values + np.random.normal(0, astrometric_error, num_sources)
    dec_psf = dec_values + np.random.normal(0, astrometric_error, num_sources)

    # Generate object IDs (Euclid-like format)
    object_ids = (np.arange(1, num_sources + 1) + int(tile_id) * 1000000).astype(np.int64)

    # Generate realistic flux measurements
    # Log-normal distribution for fluxes (typical for astronomical sources)
    base_flux = np.random.lognormal(3.5, 1.2, num_sources).astype(np.float32)

    # Different aperture sizes with expected scaling
    flux_1fwhm = base_flux
    flux_2fwhm = base_flux * np.random.uniform(1.4, 1.8, num_sources)
    flux_3fwhm = flux_2fwhm * np.random.uniform(1.2, 1.5, num_sources)

    # Create astropy table with Euclid-compliant columns
    table = Table()

    # Core astrometric columns
    table["OBJECT_ID"] = object_ids
    table["RIGHT_ASCENSION"] = ra_values
    table["DECLINATION"] = dec_values
    table["RIGHT_ASCENSION_PSF_FITTING"] = ra_psf
    table["DECLINATION_PSF_FITTING"] = dec_psf

    # Detection and segmentation
    table["SEGMENTATION_MAP_ID"] = np.arange(1, num_sources + 1, dtype=np.int64)
    table["VIS_DET"] = np.ones(num_sources, dtype=np.int32)

    # Photometric measurements (key apertures)
    table["FLUX_VIS_1FWHM_APER"] = flux_1fwhm
    table["FLUX_VIS_2FWHM_APER"] = flux_2fwhm
    table["FLUX_VIS_3FWHM_APER"] = flux_3fwhm

    # Add additional aperture sizes (Euclid has many)
    for aperture in [4, 5, 6, 8, 10, 12, 15, 20]:
        scaling = 1.0 + 0.1 * aperture  # Larger apertures capture more flux
        table[f"FLUX_VIS_{aperture}PIX_APER"] = flux_1fwhm * np.random.uniform(
            scaling * 0.9, scaling * 1.1, num_sources
        )

    # Add flux uncertainties (typically 5-15% for good detections)
    for col in table.colnames:
        if "FLUX_VIS" in col and "ERR" not in col:
            error_col = col.replace("FLUX_VIS", "FLUX_ERR_VIS")
            table[error_col] = table[col] * np.random.uniform(0.05, 0.15, num_sources)

    # Shape measurements (basic ellipticity)
    table["ELLIPTICITY_1"] = np.random.normal(0, 0.1, num_sources).astype(np.float32)
    table["ELLIPTICITY_2"] = np.random.normal(0, 0.1, num_sources).astype(np.float32)
    table["SIZE_RADIUS"] = np.random.uniform(2.0, 8.0, num_sources).astype(np.float32)

    # Create FITS file structure
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["EXTEND"] = True
    primary_hdu.header["ORIGIN"] = "Cutana Mock Data Generator"
    primary_hdu.header["COMMENT"] = "Mock Euclid catalog for testing"

    # Create binary table HDU
    table_hdu = fits.BinTableHDU(table, name="EUC_MER__FINAL_CATALOG")

    # Add catalog-specific headers
    table_hdu.header["TILEID"] = tile_id
    table_hdu.header["DATE"] = datetime.utcnow().isoformat() + "Z"
    table_hdu.header["CATALOG"] = "FINAL"
    table_hdu.header["INSTRUME"] = "MER"
    table_hdu.header["TELESCOP"] = "EUCLID"
    table_hdu.header["NSOURCES"] = num_sources

    # Write FITS file
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(filepath, overwrite=True)

    return str(filepath)


def create_cutana_input_catalog(
    output_path: Path,
    catalog_fits: str,
    mosaic_fits: str,
    size_name: str,
    num_sources: int,
    additional_channels: List[str] = None,
) -> str:
    """Create Cutana-compatible input CSV catalog with multi-channel support."""
    catalog_csv = output_path / f"euclid_cutana_catalogue_{size_name}.csv"

    logger.info(f"Creating Cutana catalog: {catalog_csv.name} ({num_sources} sources)")

    # Read sources from Euclid catalog FITS
    with fits.open(catalog_fits) as hdul:
        cat_table = Table.read(hdul["EUC_MER__FINAL_CATALOG"])

    # Select random subset of sources
    if len(cat_table) > num_sources:
        indices = np.random.choice(len(cat_table), num_sources, replace=False)
        selected_sources = cat_table[indices]
    else:
        selected_sources = cat_table

    # Convert to Cutana format
    cutana_sources = []
    for row in selected_sources:
        # Use PSF-fitted positions (more accurate)
        ra = float(row["RIGHT_ASCENSION_PSF_FITTING"])
        dec = float(row["DECLINATION_PSF_FITTING"])

        # Generate realistic cutout sizes based on source brightness
        base_flux = float(row["FLUX_VIS_2FWHM_APER"])

        # Brighter sources get larger cutouts
        if base_flux > 1000:
            diameter_arcsec = np.random.uniform(20.0, 30.0)
        elif base_flux > 100:
            diameter_arcsec = np.random.uniform(15.0, 25.0)
        else:
            diameter_arcsec = np.random.uniform(10.0, 20.0)

        # Convert to pixels (Euclid VIS: 0.1 arcsec/pixel)
        diameter_pixel = int(diameter_arcsec / 0.1)

        # Ensure even sizes for better processing
        if diameter_pixel % 2 == 1:
            diameter_pixel += 1

        # Build FITS file paths list (single-channel by default, multi-channel if additional_channels provided)
        fits_paths = [Path(mosaic_fits).resolve().as_posix()]

        if additional_channels:
            # For multi-channel test data, include additional channel files
            fits_paths.extend(
                [
                    Path(mosaic_fits).resolve().as_posix().replace(".fits", f"_{channel}.fits")
                    for channel in additional_channels
                ]
            )

        cutana_sources.append(
            {
                "SourceID": f"MockSource_{int(row['OBJECT_ID'])}",  # Use MockSource_ for test compatibility
                "RA": ra,
                "Dec": dec,
                "diameter_pixel": diameter_pixel,  # Only use diameter_pixel (mutual exclusivity with diameter_arcsec)
                "fits_file_paths": str(fits_paths),  # List of FITS files for this source
            }
        )

    # If we need more sources for large catalog, duplicate some with slight variations
    if num_sources > len(cutana_sources):
        logger.info(f"Padding catalog from {len(cutana_sources)} to {num_sources} sources")
        original_count = len(cutana_sources)
        while len(cutana_sources) < num_sources:
            # Pick a random existing source to duplicate
            base_source = cutana_sources[np.random.randint(0, original_count)].copy()

            # Add small random variations to position (within ~1 arcsec)
            ra_variation = np.random.uniform(-0.0003, 0.0003)  # ~1 arcsec at dec=2
            dec_variation = np.random.uniform(-0.0003, 0.0003)  # ~1 arcsec

            base_source["SourceID"] = f"MockSource_Pad_{len(cutana_sources):05d}"
            base_source["RA"] = float(base_source["RA"]) + ra_variation
            base_source["Dec"] = float(base_source["Dec"]) + dec_variation

            cutana_sources.append(base_source)

    # Save as CSV
    df = pd.DataFrame(cutana_sources)
    df.to_csv(catalog_csv, index=False)

    return str(catalog_csv)


def generate_tile_data(
    output_path: Path,
    tile_id: str,
    ra_center: float,
    dec_center: float,
    mosaic_size: Tuple[int, int],
    catalog_sources: int,
    multi_channel: bool = False,
) -> Dict[str, str]:
    """Generate complete data for one Euclid tile with optional multi-channel support."""
    logger.info(f"Generating tile {tile_id} data (RA={ra_center:.3f}, Dec={dec_center:.3f})")

    # First create catalog to get source positions
    catalog_file = create_euclid_catalog_fits(
        output_path, tile_id, ra_center, dec_center, mosaic_size, catalog_sources
    )

    # Extract source positions and convert to pixel coordinates
    source_pixel_positions = []
    try:
        with fits.open(catalog_file) as hdul:
            cat_table = Table.read(hdul["EUC_MER__FINAL_CATALOG"])

            # Create WCS for coordinate conversion
            wcs = create_euclid_wcs(ra_center, dec_center, mosaic_size)

            for row in cat_table:
                ra = float(row["RIGHT_ASCENSION"])
                dec = float(row["DECLINATION"])

                # Convert RA/Dec to pixel coordinates
                from astropy import units as u
                from astropy.coordinates import SkyCoord

                coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
                pixel_x, pixel_y = wcs.world_to_pixel(coord)
                source_pixel_positions.append((float(pixel_x), float(pixel_y)))

        logger.info(f"Converted {len(source_pixel_positions)} source positions to pixels")
    except Exception as e:
        logger.warning(f"Could not extract source positions for white pixel placement: {e}")
        source_pixel_positions = None

    # Create primary mosaic FITS with white pixels at source positions
    mosaic_file = create_euclid_mosaic_fits(
        output_path,
        tile_id,
        ra_center,
        dec_center,
        mosaic_size,
        source_positions=source_pixel_positions,
    )

    # Create additional channel files for multi-channel testing
    additional_channel_files = []
    if multi_channel:
        channels = ["NIR_H", "NIR_J", "NIR_Y"]  # Euclid NIR channels
        for channel in channels:
            channel_file = create_euclid_mosaic_fits(
                output_path,
                tile_id,
                ra_center,
                dec_center,
                mosaic_size,
                instrument=channel,
                source_positions=source_pixel_positions,
            )
            additional_channel_files.append(channel_file)

        logger.info(
            f"Created {len(additional_channel_files)} additional channel files for multi-channel testing"
        )

    return {
        "tile_id": tile_id,
        "mosaic_file": mosaic_file,
        "catalog_file": catalog_file,
        "additional_channels": additional_channel_files,
        "ra_center": ra_center,
        "dec_center": dec_center,
    }


def generate_all_test_data(output_path: Path, args) -> Dict:
    """Generate complete Euclid-compliant test data suite."""
    logger.info("🌟 Generating Euclid-compliant test data for Cutana")

    # Configuration for different tile sizes
    if args.fast:
        mosaic_size = (1000, 1000)  # Smaller for faster generation
        logger.info("Using fast mode: smaller image sizes")
    else:
        mosaic_size = (2000, 2000)  # Realistic but manageable size

    # Tile configuration (multiple tiles for comprehensive testing)
    tile_configs = [
        {"id": "102018211", "ra": 150.12345, "dec": 2.34567, "sources": 465},
        {"id": "102018212", "ra": 150.45678, "dec": 2.65432, "sources": 523},
        {"id": "102018213", "ra": 149.87654, "dec": 2.12345, "sources": 398},
    ]

    # Limit tiles if requested
    if args.tiles:
        tile_configs = tile_configs[: args.tiles]

    # Generate tile data
    tiles_data = []
    for config in tile_configs:
        tile_data = generate_tile_data(
            output_path,
            config["id"],
            config["ra"],
            config["dec"],
            mosaic_size,
            config["sources"],
            multi_channel=getattr(args, "multi_channel", False),
        )
        tiles_data.append(tile_data)

    # Create Cutana input catalogs (different sizes)
    catalog_configs = [
        {"name": "small", "sources": 25},  # Updated to match test expectations
        {"name": "medium", "sources": 100},
        {"name": "large", "sources": 500},
        {"name": "xlarge", "sources": 2500},  # Large-scale testing
    ]

    # Add multi-channel catalog variant
    if getattr(args, "multi_channel", False):
        catalog_configs.append({"name": "multi_channel", "sources": 50})

    # Filter by requested size
    if args.size:
        catalog_configs = [c for c in catalog_configs if c["name"] == args.size]

    cutana_catalogs = []
    primary_tile = tiles_data[0]  # Use first tile for Cutana catalogs

    for config in catalog_configs:
        # For multi-channel catalog, include additional channels
        if config["name"] == "multi_channel" and primary_tile.get("additional_channels"):
            additional_channels = ["NIR_H", "NIR_J", "NIR_Y"]
        else:
            additional_channels = None

        catalog_file = create_cutana_input_catalog(
            output_path,
            primary_tile["catalog_file"],
            primary_tile["mosaic_file"],
            config["name"],
            config["sources"],
            additional_channels=additional_channels,
        )
        cutana_catalogs.append(catalog_file)

    # Create comprehensive metadata
    metadata = {
        "format": "euclid_compliant",
        "generated": datetime.utcnow().isoformat() + "Z",
        "generator": "Cutana Mock Data Generator v2.0",
        "tiles": tiles_data,
        "cutana_catalogs": cutana_catalogs,
        "mosaic_size": mosaic_size,
        "pixel_scale_arcsec": 0.1,
        "coordinate_system": "ICRS",
        "description": "Mock Euclid-format astronomical data for Cutana testing",
        "files_created": [],
    }

    # Collect all created files
    for tile in tiles_data:
        metadata["files_created"].extend([tile["mosaic_file"], tile["catalog_file"]])
    metadata["files_created"].extend(cutana_catalogs)

    # Save metadata
    metadata_file = output_path / "euclid_test_data_info.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.success("✅ Test data generation complete!")
    logger.info(f"📊 Tiles created: {len(tiles_data)}")
    logger.info(f"📊 Cutana catalogs: {len(cutana_catalogs)}")
    logger.info(f"📊 Total files: {len(metadata['files_created'])}")
    logger.info(f"📄 Metadata saved: {metadata_file}")

    return metadata


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate Euclid-compliant mock test data for Cutana",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Generate all data (3 tiles, all sizes)
  %(prog)s --size small         # Generate only small Cutana catalogs
  %(prog)s --tiles 1            # Generate single tile only
  %(prog)s --fast               # Use smaller images for faster generation
  %(prog)s --verbose            # Enable debug logging
        """,
    )

    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "xlarge", "multi_channel"],
        help="Generate only specific Cutana catalog size",
    )
    parser.add_argument(
        "--tiles", type=int, choices=[1, 2, 3], help="Number of tiles to generate (default: 3)"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Use smaller images for faster generation"
    )
    parser.add_argument(
        "--multi-channel", action="store_true", help="Generate multi-channel FITS files for testing"
    )
    parser.add_argument("--output", type=Path, help="Output directory (default: script directory)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)

    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = Path(__file__).parent

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Output directory: {output_path}")

    try:
        # Generate data
        start_time = time.time()
        metadata = generate_all_test_data(output_path, args)
        generation_time = time.time() - start_time

        # Summary
        total_size = sum(
            Path(f).stat().st_size for f in metadata["files_created"] if Path(f).exists()
        )

        logger.success(f"🎉 Generation completed in {generation_time:.1f}s")
        logger.info(f"📦 Total data size: {total_size / (1024*1024):.1f} MB")
        logger.info(f"📂 Files available in: {output_path}")

        return 0

    except KeyboardInterrupt:
        logger.warning("⚡ Generation interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"💥 Generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
