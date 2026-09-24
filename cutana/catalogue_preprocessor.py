#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Catalogue preprocessing and validation functions for Cutana.

Provides functionality to validate, preprocess, and analyze source catalogues,
including comprehensive data validation, FITS file checking, and metadata extraction.
"""

import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from loguru import logger

from .catalogue_sample import read_catalogue_sample
from .catalogue_validation import CatalogueValidationError
from .fits_paths import parse_fits_file_paths
from .source_footprint import check_sources_in_products
from .validation_sampling import sample_for_validation

# Catalogue size at or above which the duplicate-SourceID check is skipped: the check
# is O(n) over the frame, and Cutana must stay usable on billion-source catalogues.
# StreamingOrchestrator sizes its internal batches to stay below this.
DUPLICATE_CHECK_THRESHOLD = 100_000


__all__ = ["CatalogueValidationError"]  # re-exported: the historical import site


def extract_fits_sets(
    fits_files: List[str], filters: List[str] = None
) -> Tuple[Dict[tuple, List[str]], Dict[str, float]]:
    """
    Extract FITS sets from a list of FITS files and determine resolution ratios.

    Args:
        fits_files: List of FITS file paths
        filters: Optional list of filter names for resolution checking

    Returns:
        Tuple of (fits_set_dict, resolution_ratios) where:
        - fits_set_dict: Dict mapping fits_set tuples to list of fits files
        - resolution_ratios: Dict mapping each FITS path to its pixel scale ratio
          against the first path's scale. Keyed by path, not by filter: two tiles
          the recogniser cannot classify share the label ``UNKNOWN``, so a
          filter-keyed dict silently kept whichever the row happened to list last
          and the check's outcome depended on file order.
    """
    fits_set_dict = {}
    resolution_ratios = {}

    # Normalize paths and create FITS set signature
    normalized_paths = [os.path.normpath(path) for path in fits_files]
    fits_set = tuple(normalized_paths)
    fits_set_dict[fits_set] = normalized_paths

    # Calculate resolution ratios if filters provided
    if filters and len(normalized_paths) > 1:
        pixel_scales = {}

        for fits_path in normalized_paths:
            try:
                # Get pixel scale from WCS
                with fits.open(fits_path) as hdul:
                    # Try PRIMARY extension first, then first extension with WCS.
                    # A WCS() failure on one HDU is expected (not every HDU has
                    # celestial coords), so we continue searching — but log at
                    # debug level so the root cause is not silently discarded.
                    wcs_obj = None
                    for hdu in hdul:
                        try:
                            if hasattr(hdu, "header") and hdu.header:
                                test_wcs = WCS(hdu.header, naxis=2)
                                if test_wcs.has_celestial:
                                    wcs_obj = test_wcs
                                    break
                        except Exception as wcs_error:
                            logger.debug(
                                f"Skipping HDU in {fits_path} (no usable WCS): {wcs_error}"
                            )
                            continue

                    if wcs_obj:
                        pixel_scale_matrix = wcs_obj.pixel_scale_matrix
                        pixel_scale_deg = abs(pixel_scale_matrix[0, 0])  # degrees per pixel
                        # sanity check of the wcs direction
                        assert pixel_scale_deg == np.max(np.abs(pixel_scale_matrix)), (
                            f"unexpected pixel scale matrix. Expected pixel scale {pixel_scale_deg} from [0,0] of {pixel_scale_matrix}"
                        )
                        pixel_scale_arcsec = pixel_scale_deg * 3600.0
                        # Keyed by path: an unrecognised product has a pixel scale worth
                        # comparing like any other, but every one of them is labelled
                        # UNKNOWN, so keying by filter compared only the last of them.
                        pixel_scales[fits_path] = pixel_scale_arcsec

            except Exception as e:
                logger.warning(f"Could not determine resolution for {fits_path}: {e}")
                continue

        # Calculate resolution ratios relative to first filter
        if len(pixel_scales) > 1:
            reference_scale = list(pixel_scales.values())[0]
            for scaled_path, scale in pixel_scales.items():
                resolution_ratios[scaled_path] = scale / reference_scale

    return fits_set_dict, resolution_ratios


#: Euclid filename -> band patterns. Order matters: more specific patterns first.
_FILTER_PATTERNS = [
    (r"VIS", "VIS"),
    (r"NIR-?Y", "NIR-Y"),
    (r"NIR-?H", "NIR-H"),
    (r"NIR-?J", "NIR-J"),
    # More specific patterns for NIR variations
    (r"NIR_H", "NIR-H"),
    (r"NIR_Y", "NIR-Y"),
    (r"NIR_J", "NIR-J"),
    # Single letter patterns - match at word boundaries, underscores, or start of word
    (r"(?:^|[^A-Z])Y(?:[^A-Z]|$)", "Y"),
    (r"(?:^|[^A-Z])J(?:[^A-Z]|$)", "J"),
    (r"(?:^|[^A-Z])H(?:[^A-Z]|$)", "H"),
]

#: Every band `extract_filter_name` can name, so a caller can ask whether a string is a
#: band at all. Derived from the pattern table rather than written out again: the two
#: drifting apart is what would make `selected_extensions` look like an HDU-name list and
#: silently skip band selection. `"UNKNOWN"` is deliberately absent -- it is the answer for
#: a filename that names no band, never something a user selects.
SELECTABLE_BAND_NAMES = frozenset(band for _, band in _FILTER_PATTERNS)


def extract_filter_name(filename: str) -> str:
    """
    Extract filter name from FITS filename. This is a Euclid specific function, based on the namings of the extensions.

    Args:
        filename: FITS file path or name

    Returns:
        Filter name (e.g., 'VIS', 'NIR-Y', 'NIR-H'), or 'UNKNOWN' when no Euclid pattern
        matches.

        'UNKNOWN' is deliberately a constant. `channel_weights` is one dictionary for the
        whole run, so a channel label has to mean the same thing in every row; anything
        derived from the filename of an unrecognised tile is a *tile* identity and
        changes row to row, which makes the catalogue's rows disagree about their own
        channels. A band token is the only per-file thing that is stable across a survey,
        and recognising one is exactly what this function does or fails to do.

        One 'UNKNOWN' channel in a row is workable: it is a single channel, so
        `validate_channel_order_consistency` pairs it without consulting its name. Two
        collapse onto the same label and are refused, because they are genuinely
        indistinguishable to weights and to the WCS lookup.
    """
    filename_upper = Path(filename).name.upper()

    for pattern, filter_name in _FILTER_PATTERNS:
        if re.search(pattern, filename_upper):
            return filter_name

    return "UNKNOWN"


def analyze_fits_file(fits_path: str) -> Dict[str, Any]:
    """
    Analyze a FITS file and return extension information.

    Args:
        fits_path: Path to FITS file

    Returns:
        Dictionary with extension information
    """
    try:
        if not Path(fits_path).exists():
            logger.warning(f"FITS file not found: {fits_path}")
            return {
                "path": fits_path,
                "exists": False,
                "filter": extract_filter_name(fits_path),
                "extensions": [],
                "num_extensions": 0,
                "error": "File not found",
            }

        with fits.open(fits_path) as hdul:
            extensions = []
            for i, hdu in enumerate(hdul):
                ext_info = {
                    "index": i,
                    "name": hdu.name if hasattr(hdu, "name") else f"HDU{i}",
                    "type": type(hdu).__name__,
                    "has_data": hdu.header["NAXIS"] > 0,
                }
                extensions.append(ext_info)

            return {
                "path": fits_path,
                "exists": True,
                "filter": extract_filter_name(fits_path),
                "extensions": extensions,
                "num_extensions": len(extensions),
                "error": None,
            }

    except Exception as e:
        logger.error(f"Error Analysing FITS file {fits_path}: {e}")
        return {
            "path": fits_path,
            "exists": False,
            "filter": extract_filter_name(fits_path),
            "extensions": [],
            "num_extensions": 0,
            "error": str(e),
        }


def validate_catalogue_columns(catalogue_df: pd.DataFrame) -> List[str]:
    """
    Validate that required columns exist and have correct types.

    Args:
        catalogue_df: DataFrame to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    required_columns = ["SourceID", "RA", "Dec", "fits_file_paths"]

    # Check for required columns
    missing_columns = []
    for col in required_columns:
        if col not in catalogue_df.columns:
            missing_columns.append(col)

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return errors  # Can't continue validation without basic columns

    # Check for size column (either diameter_pixel or diameter_arcsec)
    if (
        "diameter_pixel" not in catalogue_df.columns
        and "diameter_arcsec" not in catalogue_df.columns
    ):
        errors.append("Must have either 'diameter_pixel' or 'diameter_arcsec' column")

    # Validate data types
    try:
        # SourceID should be convertible to string (allow any type that can be converted)
        # Try converting to string to test if it's possible
        pd.Series(catalogue_df["SourceID"]).astype(str)
    except Exception as e:
        errors.append(f"SourceID column values cannot be converted to strings: {e}")

    try:
        # RA and Dec should be numeric
        pd.to_numeric(catalogue_df["RA"], errors="raise")
        pd.to_numeric(catalogue_df["Dec"], errors="raise")
    except Exception as e:
        errors.append(f"RA and Dec columns must be numeric: {e}")

    try:
        # Size columns should be numeric if they exist
        if "diameter_pixel" in catalogue_df.columns:
            pd.to_numeric(catalogue_df["diameter_pixel"], errors="raise")
        if "diameter_arcsec" in catalogue_df.columns:
            pd.to_numeric(catalogue_df["diameter_arcsec"], errors="raise")
    except Exception as e:
        errors.append(f"Size columns must be numeric: {e}")

    return errors


def validate_coordinate_ranges(catalogue_df: pd.DataFrame) -> List[str]:
    """
    Validate RA, Dec, and size values are in expected ranges.

    Args:
        catalogue_df: DataFrame to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    check_df = sample_for_validation(catalogue_df, 10000, "coordinate ranges")

    try:
        # Validate RA range (0-360 degrees)
        ra_values = pd.to_numeric(check_df["RA"], errors="coerce")
        if ra_values.isnull().any():
            errors.append("Some RA values are not valid numbers")
        elif (ra_values < 0).any() or (ra_values > 360).any():
            errors.append("RA values must be between 0 and 360 degrees")

        # Validate Dec range (-90 to +90 degrees)
        dec_values = pd.to_numeric(check_df["Dec"], errors="coerce")
        if dec_values.isnull().any():
            errors.append("Some Dec values are not valid numbers")
        elif (dec_values < -90).any() or (dec_values > 90).any():
            errors.append("Dec values must be between -90 and +90 degrees")

        # Validate size ranges
        if "diameter_pixel" in check_df.columns:
            diameter_pixel = pd.to_numeric(check_df["diameter_pixel"], errors="coerce")
            if diameter_pixel.isnull().any():
                errors.append("Some diameter_pixel values are not valid numbers")
            elif (diameter_pixel <= 0).any() or (diameter_pixel > 10000).any():
                errors.append("diameter_pixel values must be between 1 and 10000 pixels")

        if "diameter_arcsec" in check_df.columns:
            diameter_arcsec = pd.to_numeric(check_df["diameter_arcsec"], errors="coerce")
            if diameter_arcsec.isnull().any():
                errors.append("Some diameter_arcsec values are not valid numbers")
            elif (diameter_arcsec <= 0).any() or (diameter_arcsec > 3600).any():
                errors.append("diameter_arcsec values must be between 0 and 3600 arcseconds")

    except Exception as e:
        errors.append(f"Error validating coordinate ranges: {e}")

    return errors


def validate_resolution_ratios(catalogue_df: pd.DataFrame) -> List[str]:
    """
    Validate that if diameter_pixel is used with multiple filters, resolution ratios are acceptable.

    Args:
        catalogue_df: DataFrame to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Only check if diameter_pixel is used
    if "diameter_pixel" not in catalogue_df.columns:
        return errors

    # Check first few sources for multi-filter scenarios
    sample_size = min(10, len(catalogue_df))

    for idx in range(sample_size):
        row = catalogue_df.iloc[idx]

        try:
            # Parse FITS file paths
            fits_paths = parse_fits_file_paths(row["fits_file_paths"])

            if len(fits_paths) > 1:
                # Multiple filters - check resolution ratios
                try:
                    filters = [extract_filter_name(path) for path in fits_paths]
                    _, resolution_ratios = extract_fits_sets(fits_paths, filters)

                    # Check if any resolution ratio differs by more than 0.1%
                    for scaled_path, ratio in resolution_ratios.items():
                        deviation = abs(ratio - 1.0)
                        if deviation > 0.0001:  # 0.01% = 0.0001
                            errors.append(
                                f"Resolution ratio difference of {deviation * 100:.2f}% detected between filters."
                                f"When using multiple filters with different resolutions, you must specify 'diameter_arcsec'"
                                f"instead of 'diameter_pixel' to avoid ambiguity about which filter's pixel scale to"
                                f"reference. Found resolution ratio {ratio:.4f} for "
                                f"{os.path.basename(scaled_path)}."
                            )
                            return errors  # Return immediately after first error

                except Exception as e:
                    logger.warning(
                        f"Could not check resolution ratios for source {row.get('SourceID', 'unknown')}: {e}"
                    )
                    continue

        except Exception as e:
            logger.warning(
                f"Error processing source {row.get('SourceID', 'unknown')} for resolution validation: {e}"
            )
            continue

    return errors


def check_fits_files_exist(catalogue_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Check if FITS files exist. Smart checking based on number of unique files.

    Args:
        catalogue_df: DataFrame with fits_file_paths column

    Returns:
        Tuple of (errors, warnings) lists
    """
    errors = []
    warnings = []

    # Collect all unique FITS files
    unique_fits_files = set()
    parse_errors = []

    for _, row in catalogue_df.iterrows():
        try:
            fits_paths = parse_fits_file_paths(row["fits_file_paths"])
            for fits_path in fits_paths:
                if fits_path:  # Skip empty strings
                    unique_fits_files.add(fits_path)
        except Exception as e:
            parse_errors.append(f"Source {row['SourceID']}: {e}")
        if len(unique_fits_files) > 100:
            # fix to save time not going thorugh the entire cat
            logger.info("More than 100 unique FITS files found, stopping further parsing")
            break

    if parse_errors:
        errors.extend(parse_errors[:5])  # Show first 5 parse errors
        if len(parse_errors) > 5:
            errors.append(f"... and {len(parse_errors) - 5} more parsing errors")

    unique_fits_files = list(unique_fits_files)
    logger.info(f"Found {len(unique_fits_files)} unique FITS files to check")

    if len(unique_fits_files) == 0:
        errors.append("No valid FITS file paths found in catalogue")
        return errors, warnings

    # Smart checking strategy
    if len(unique_fits_files) < 100:
        # Check all files if less than 100
        files_to_check = unique_fits_files
        logger.info(f"Checking all {len(files_to_check)} FITS files")
    else:
        # Randomly check 50 files if 100 or more
        files_to_check = random.sample(unique_fits_files, 50)
        logger.info(f"Randomly checking 50 out of {len(unique_fits_files)} FITS files")
        warnings.append(f"Only checked 50 out of {len(unique_fits_files)} FITS files randomly")

    # Check file existence
    missing_files = []
    for fits_path in files_to_check:
        if not Path(fits_path).exists():
            missing_files.append(fits_path)

    if missing_files:
        errors.append(
            f"Missing FITS files ({len(missing_files)} checked): {', '.join(missing_files[:3])}"
        )
        if len(missing_files) > 3:
            errors.append(f"... and {len(missing_files) - 3} more missing files")

    return errors, warnings


def preprocess_catalogue(catalogue_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess catalogue by resetting index and any other required operations.
    Ensures SourceID column is converted to string type.

    Args:
        catalogue_df: Input DataFrame

    Returns:
        Preprocessed DataFrame with reset index and string SourceID

    Raises:
        CatalogueValidationError: If rows share both SourceID and position, leaving
            no way to key them apart during cutout extraction.
    """
    logger.info(f"Preprocessing catalogue with {len(catalogue_df)} sources")

    # Reset index to ensure contiguous indices
    processed_df = catalogue_df.reset_index(drop=True)

    # Log if index was non-contiguous
    if not catalogue_df.index.equals(pd.RangeIndex(len(catalogue_df))):
        logger.info("Reset non-contiguous catalogue index")

    # Check for duplicate SourceIDs in small catalogues. For large catalogues the
    # check is skipped: unique SourceIDs are the caller's responsibility (#283), and
    # Cutana must stay usable on billion-source catalogues, which rules out any
    # whole-catalogue scan or accumulation of seen IDs.
    #
    # Streaming callers pass one internal batch at a time, so the check effectively
    # always runs for them; only a single-shot load of a >=100k catalogue skips it.
    # StreamingOrchestrator keeps its internal batches under DUPLICATE_CHECK_THRESHOLD
    # so that stays true at survey scale.
    if "SourceID" in processed_df.columns:
        processed_df["SourceID"] = processed_df["SourceID"].astype(str)

        if len(processed_df) < DUPLICATE_CHECK_THRESHOLD:
            duplicated_ids = processed_df["SourceID"].duplicated()
            if duplicated_ids.any():
                logger.warning(
                    f"Duplicate SourceIDs detected ({int(duplicated_ids.sum())} duplicates). "
                    "Reformatting all SourceIDs as SourceID_RA_Dec to prevent silent data loss."
                )
                # Keep the originals: the reformatted IDs appear nowhere in the
                # caller's file, so quoting them in an error would send the user
                # looking for rows that do not exist.
                original_ids = processed_df["SourceID"]
                processed_df["SourceID"] = (
                    processed_df["SourceID"]
                    + "_"
                    + processed_df["RA"].map("{:.10f}".format)
                    + "_"
                    + processed_df["Dec"].map("{:.10f}".format)
                )

                # Rows identical in ID *and* position survive the reformat unchanged —
                # typically exact duplicate catalogue rows. There is no attribute left
                # to tell them apart, so keying by SourceID would silently keep one
                # cutout per group and leave the caller expecting more batches than can
                # ever be produced. Refuse the input instead of guessing.
                still_colliding = processed_df["SourceID"].duplicated()
                if still_colliding.any():
                    n_exact = int(still_colliding.sum())
                    examples = original_ids[still_colliding].head(3).tolist()
                    raise CatalogueValidationError(
                        f"{n_exact} catalogue rows are exact duplicates (identical SourceID, "
                        f"RA and Dec) and cannot be distinguished; e.g. {examples}. Cutout "
                        "extraction is keyed by SourceID, so these rows would collapse into "
                        "one cutout per group and the run would produce fewer cutouts than "
                        "the catalogue has sources. Deduplicate the catalogue first, e.g. "
                        "df.drop_duplicates(subset=['SourceID', 'RA', 'Dec'])."
                    )
        else:
            logger.info(
                f"Large catalogue ({len(processed_df):,} sources) detected — "
                "skipping duplicate SourceID check."
            )

    return processed_df


def load_catalogue(catalogue_path: str) -> pd.DataFrame:
    """
    Load catalogue from file without validation.

    Args:
        catalogue_path: Path to catalogue file (CSV, FITS, or parquet)

    Returns:
        DataFrame

    Raises:
        ValueError: If file format is unsupported
        NotImplementedError: If file format is not yet implemented (parquet)
    """
    catalogue_file = Path(catalogue_path)

    if catalogue_file.suffix.lower() == ".csv":
        catalogue_df = pd.read_csv(catalogue_file)
    elif catalogue_file.suffix.lower() in [".fits", ".fit"]:
        table = Table.read(catalogue_file)
        catalogue_df = table.to_pandas()
    elif catalogue_file.suffix.lower() == ".parquet":
        catalogue_df = pd.read_parquet(catalogue_file)
    else:
        raise ValueError(f"Unsupported catalogue format: {catalogue_file.suffix}")

    logger.info(
        f"Loaded catalogue with {len(catalogue_df)} sources and columns: {list(catalogue_df.columns)}"
    )
    return catalogue_df


def stream_catalogue_chunks(
    path: str,
    batch_size: int = 100000,
    columns: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream catalogue in chunks for memory-efficient processing.

    Works with both CSV and Parquet formats. For parquet, uses pyarrow's
    iter_batches for true streaming. For CSV, uses pandas chunksize.

    Args:
        path: Path to catalogue file (CSV or Parquet)
        batch_size: Number of rows per chunk
        columns: Optional list of columns to load (None = all columns)

    Yields:
        DataFrame chunks with '_row_idx' column added for tracking

    Raises:
        ValueError: If file format is unsupported
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    row_offset = 0

    if suffix == ".parquet":
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            df = batch.to_pandas()
            df["_row_idx"] = range(row_offset, row_offset + len(df))
            row_offset += len(df)
            yield df

    elif suffix == ".csv":
        read_kwargs = {"chunksize": batch_size}
        if columns:
            read_kwargs["usecols"] = columns

        for chunk in pd.read_csv(path, **read_kwargs):
            chunk["_row_idx"] = range(row_offset, row_offset + len(chunk))
            row_offset += len(chunk)
            yield chunk

    else:
        raise ValueError(f"Unsupported catalogue format for streaming: {suffix}")


def validate_catalogue_sample(
    path: str,
    sample_size: int = 10000,
    skip_fits_check: bool = False,
) -> List[str]:
    """
    Validate a sample from the catalogue without loading it fully.

    Streams through the catalogue and validates column types, coordinate ranges,
    and optionally FITS file existence on a sample.

    Args:
        path: Path to catalogue file
        sample_size: Number of rows to sample for validation
        skip_fits_check: Skip FITS file existence checking

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Load first chunk to validate columns
    first_chunk = None
    for chunk in stream_catalogue_chunks(path, batch_size=min(sample_size, 10000)):
        first_chunk = chunk
        break

    if first_chunk is None or first_chunk.empty:
        return ["Could not read catalogue or catalogue is empty"]

    # Validate columns
    column_errors = validate_catalogue_columns(first_chunk)
    if column_errors:
        return column_errors

    # Sample rows for coordinate validation
    # Collect sample across multiple chunks if needed
    sample_rows = []
    target_sample = sample_size

    for chunk in stream_catalogue_chunks(path, batch_size=100000):
        # Random sample from this chunk
        chunk_sample_size = min(len(chunk), target_sample - len(sample_rows))
        if chunk_sample_size > 0:
            if len(chunk) <= chunk_sample_size:
                sample_rows.append(chunk)
            else:
                sample_rows.append(chunk.sample(n=chunk_sample_size, random_state=42))

        if len(sample_rows) > 0 and sum(len(df) for df in sample_rows) >= target_sample:
            break

    if sample_rows:
        sample_df = pd.concat(sample_rows, ignore_index=True)

        # Validate coordinate ranges on sample
        range_errors = validate_coordinate_ranges(sample_df)
        errors.extend(range_errors)

        # Validate resolution ratios on sample
        resolution_errors = validate_resolution_ratios(sample_df)
        errors.extend(resolution_errors)

        # Check FITS files exist on sample
        if not skip_fits_check:
            fits_errors, _ = check_fits_files_exist(sample_df)
            errors.extend(fits_errors)

            # And that the sources are actually inside them. Behind the same flag, because it is
            # the same cost -- opening FITS files -- and the check means nothing without them.
            position_errors, position_warnings = check_sources_in_products(sample_df)
            errors.extend(position_errors)
            for warning in position_warnings:
                # Warnings do not fail a run, so they have to be logged or they are lost -- and a
                # cutout that will be silently half-trimmed is exactly what someone wants told.
                logger.warning(warning)

    return errors


def load_and_validate_catalogue(catalogue_path: str, skip_fits_check: bool = False) -> pd.DataFrame:
    """
    Load catalogue from file and perform comprehensive validation.

    Args:
        catalogue_path: Path to catalogue file (CSV or FITS)
        skip_fits_check: Skip FITS file existence checking (for testing)

    Returns:
        Validated and preprocessed DataFrame

    Raises:
        CatalogueValidationError: If validation fails
    """
    logger.info(f"Loading and validating catalogue: {catalogue_path}")

    # Load the catalogue
    catalogue_df = load_catalogue(catalogue_path)

    # Validate columns and types
    column_errors = validate_catalogue_columns(catalogue_df)
    if column_errors:
        raise CatalogueValidationError(f"Column validation failed: {'; '.join(column_errors)}")

    # Validate coordinate ranges
    range_errors = validate_coordinate_ranges(catalogue_df)
    if range_errors:
        raise CatalogueValidationError(f"Coordinate validation failed: {'; '.join(range_errors)}")

    # Validate resolution ratios for diameter_pixel usage
    resolution_errors = validate_resolution_ratios(catalogue_df)
    if resolution_errors:
        raise CatalogueValidationError(
            f"Resolution validation failed: {'; '.join(resolution_errors)}"
        )

    # Check FITS files exist (unless skipped)
    if not skip_fits_check:
        fits_errors, fits_warnings = check_fits_files_exist(catalogue_df)
        if fits_errors:
            raise CatalogueValidationError(f"FITS file validation failed: {'; '.join(fits_errors)}")

        # Log warnings but don't fail
        for warning in fits_warnings:
            logger.warning(warning)

        # And that the sources are inside the files they name. Same gate, same reason: it
        # opens FITS files. Without it a catalogue whose sources sit outside their products
        # is accepted and yields a full-size, all-trim cutout. `analyse_source_catalogue`
        # runs the same check itself for the UI, which does not come through here.
        position_errors, position_warnings = check_sources_in_products(catalogue_df)
        if position_errors:
            raise CatalogueValidationError(
                f"Source position validation failed: {'; '.join(position_errors)}"
            )
        for warning in position_warnings:
            logger.warning(warning)

    # Preprocess catalogue
    processed_df = preprocess_catalogue(catalogue_df)

    logger.info(
        f"Successfully validated and preprocessed catalogue with {len(processed_df)} sources"
    )
    return processed_df


def analyse_source_catalogue(catalogue_path: str) -> Dict[str, Any]:
    """
    Analyze a source catalogue and return comprehensive metadata.
    This function combines validation and analysis functionality.

    Args:
        catalogue_path: Path to catalogue file (CSV or FITS)

    Returns:
        Dictionary containing analysis results

    Raises:
        CatalogueValidationError: If validation fails
    """
    logger.info(f"Starting analysis of catalogue: {catalogue_path}")

    catalogue_df, num_sources, count_estimated, sampling_scope = read_catalogue_sample(
        catalogue_path
    )
    if catalogue_df.empty:
        raise CatalogueValidationError("Catalogue is empty")
    errors = validate_catalogue_columns(catalogue_df)
    if not errors:
        errors.extend(validate_coordinate_ranges(catalogue_df))
        errors.extend(validate_resolution_ratios(catalogue_df))
    if errors:
        raise CatalogueValidationError("; ".join(errors))

    # Cache headers only for this bounded sample; never accumulate whole-catalogue state.
    fits_by_path = {}
    expected_filters = None
    extensions_by_filter = {}
    total_fits_entries = 0
    for _, row in catalogue_df.iterrows():
        # The sample's index is positional within the sample, not a catalogue row number
        # (see `read_catalogue_sample`), so name the source instead: it is unique and the
        # user can find it in their own file.
        source = row["SourceID"]
        paths = parse_fits_file_paths(row["fits_file_paths"])
        filters = [extract_filter_name(path) for path in paths]
        # Two tiles in a row sharing a label cannot work: weights and WCS are both looked
        # up by name, so the bands would be indistinguishable — two Euclid tiles of the
        # same band, or two files this recogniser cannot classify, which both come back
        # as UNKNOWN.
        if not filters or len(set(filters)) != len(filters):
            raise CatalogueValidationError(
                f"Source {source}: FITS paths do not map to distinct channels: {filters}. "
                "Rename the files so each band is identifiable, or use the Python API "
                "with explicit channel labels."
            )
        if expected_filters is None:
            expected_filters = filters
        elif set(filters) != set(expected_filters):
            # The *set*, not the sequence: weights resolve by name now, and the WCS check
            # reads each row's own path order, so a row listing its bands in a different
            # order is processed correctly. A row carrying different bands is not — the
            # tensor would have a different width from the one `channel_weights` describes.
            missing = sorted(set(expected_filters) - set(filters))
            extra = sorted(set(filters) - set(expected_filters))
            raise CatalogueValidationError(
                f"Source {source}: bands {sorted(filters)} differ from the sampled "
                f"catalogue's "
                f"{sorted(expected_filters)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
            )
        total_fits_entries += len(paths)
        for path, filter_name in zip(paths, filters):
            if path not in fits_by_path:
                info = analyze_fits_file(path)
                if not info["exists"] or info["error"]:
                    raise CatalogueValidationError(f"Cannot analyze {path}: {info['error']}")
                fits_by_path[path] = info
            info = fits_by_path[path]
            layout = [
                (ext["index"], ext["name"], ext["type"], ext["has_data"])
                for ext in info["extensions"]
            ]
            if filter_name in extensions_by_filter and extensions_by_filter[filter_name] != layout:
                raise CatalogueValidationError(
                    f"Inconsistent HDU order/layout for filter {filter_name}: {path}"
                )
            extensions_by_filter[filter_name] = layout

    position_errors, position_warnings = check_sources_in_products(catalogue_df)
    if position_errors:
        raise CatalogueValidationError("; ".join(position_errors))
    for warning in position_warnings:
        logger.warning(warning)

    sample_size = len(catalogue_df)
    unique_fits_files = list(fits_by_path)
    fits_analysis_results = list(fits_by_path.values())
    extensions_display = [
        {"name": name, "ext": ", ".join(dict.fromkeys(ext[2] for ext in layout))}
        for name, layout in extensions_by_filter.items()
    ]
    avg_fits_per_source = total_fits_entries / sample_size
    logger.info(
        f"Channel discovery checked {sample_size} rows from {sampling_scope}; "
        "unsampled rows have not been validated"
    )

    result = {
        "num_sources": num_sources,
        "num_sources_estimated": count_estimated,
        "sampling_scope": sampling_scope,
        "fits_files": unique_fits_files,
        "num_unique_fits_files": len(unique_fits_files),
        "avg_fits_per_source": avg_fits_per_source,
        "extensions": extensions_display,
        "fits_analysis": fits_analysis_results,
        "extensions_by_filter": dict(
            extensions_by_filter
        ),  # Convert sets to lists for JSON serialization
        "catalogue_columns": list(catalogue_df.columns),
        "sample_analysis_size": sample_size,
    }

    logger.info(
        f"Catalogue analysis complete: {num_sources} sources, {len(unique_fits_files)} FITS files, {len(extensions_display)}"
        f"filter types"
    )
    return result
