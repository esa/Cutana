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

import ast
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from loguru import logger


class CatalogueValidationError(Exception):
    """Exception raised when catalogue validation fails."""

    pass


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
        - resolution_ratios: Dict mapping filter names to pixel scale ratios
    """
    import os

    from astropy.io import fits
    from astropy.wcs import WCS

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
                filter_name = extract_filter_name(fits_path)
                if filter_name == "UNKNOWN":
                    continue

                # Get pixel scale from WCS
                with fits.open(fits_path) as hdul:
                    # Try PRIMARY extension first, then first extension with WCS
                    wcs_obj = None
                    for hdu in hdul:
                        try:
                            if hasattr(hdu, "header") and hdu.header:
                                test_wcs = WCS(hdu.header, naxis=2)
                                if test_wcs.has_celestial:
                                    wcs_obj = test_wcs
                                    break
                        except Exception:
                            continue

                    if wcs_obj:
                        pixel_scale_matrix = wcs_obj.pixel_scale_matrix
                        pixel_scale_deg = abs(pixel_scale_matrix[0, 0])  # degrees per pixel
                        # sanity check of the wcs direction
                        assert pixel_scale_deg == np.max(
                            np.abs(pixel_scale_matrix)
                        ), f"unexpected pixel scale matrix. Expected pixel scale {pixel_scale_deg} from [0,0] of {pixel_scale_matrix}"
                        pixel_scale_arcsec = pixel_scale_deg * 3600.0
                        pixel_scales[filter_name] = pixel_scale_arcsec

            except Exception as e:
                logger.warning(f"Could not determine resolution for {fits_path}: {e}")
                continue

        # Calculate resolution ratios relative to first filter
        if len(pixel_scales) > 1:
            reference_scale = list(pixel_scales.values())[0]
            for filter_name, scale in pixel_scales.items():
                resolution_ratios[filter_name] = scale / reference_scale

    return fits_set_dict, resolution_ratios


def extract_filter_name(filename: str) -> str:
    """
    Extract filter name from FITS filename. This is a Euclid specific function, based on the namings of the extensions.

    Args:
        filename: FITS file path or name

    Returns:
        Filter name (e.g., 'VIS', 'NIR-Y', 'NIR-H') or 'UNKNOWN'
    """
    filename_upper = Path(filename).name.upper()

    # Common Euclid filter patterns - order matters! More specific patterns first
    filter_patterns = [
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

    for pattern, filter_name in filter_patterns:
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
                    "has_data": hdu.data is not None,
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


def parse_fits_file_paths(fits_paths_str: str, normalize: bool = True) -> List[str]:
    """
    Parse the fits_file_paths column which may be in string representation of list.

    Args:
        fits_paths_str: String representation of FITS file paths
        normalize: Whether to normalize paths using os.path.normpath (default: True)

    Returns:
        List of FITS file paths (normalized if normalize=True)

    Raises:
        ValueError: If the input is malformed (e.g., unbalanced brackets or invalid syntax)
    """
    fits_paths = []

    # Handle different formats
    if isinstance(fits_paths_str, str):
        # Remove any extra whitespace
        fits_paths_str = fits_paths_str.strip()

        # Check for malformed list syntax (unbalanced brackets)
        starts_with_bracket = fits_paths_str.startswith("[")
        ends_with_bracket = fits_paths_str.endswith("]")
        if starts_with_bracket != ends_with_bracket:
            raise ValueError(f"Malformed FITS paths string (unbalanced brackets): {fits_paths_str}")

        # Try to evaluate as Python literal (list)
        if starts_with_bracket and ends_with_bracket:
            fits_paths = ast.literal_eval(fits_paths_str)
        # If it's a single path without brackets
        elif fits_paths_str:
            fits_paths = [fits_paths_str]

    # If it's already a list
    elif isinstance(fits_paths_str, list):
        fits_paths = fits_paths_str

    # Normalize paths if requested
    if normalize and fits_paths:
        fits_paths = [os.path.normpath(path) for path in fits_paths]

    return fits_paths


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

    # For large catalogues (>10K), only check 10K subset
    if len(catalogue_df) > 10001:
        logger.info(
            f"Large catalogue ({len(catalogue_df)} sources), spot-checking 10000 random sources"
        )
        check_df = catalogue_df.sample(n=10000, random_state=42)
    else:
        check_df = catalogue_df

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
                    for filter_name, ratio in resolution_ratios.items():
                        deviation = abs(ratio - 1.0)
                        if deviation > 0.0001:  # 0.01% = 0.0001
                            errors.append(
                                f"Resolution ratio difference of {deviation*100:.2f}% detected between filters."
                                f"When using multiple filters with different resolutions, you must specify 'diameter_arcsec'"
                                f"instead of 'diameter_pixel' to avoid ambiguity about which filter's pixel scale to"
                                f"reference. Found resolution ratio {ratio:.4f} for filter {filter_name}."
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

    for idx, row in catalogue_df.iterrows():
        try:
            fits_paths = parse_fits_file_paths(row["fits_file_paths"])
            for fits_path in fits_paths:
                if fits_path:  # Skip empty strings
                    unique_fits_files.add(fits_path)
        except Exception as e:
            parse_errors.append(f"Row {idx}: {e}")
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


def preprocess_catalogue(catalogue_df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Preprocess catalogue by resetting index and any other required operations.
    Ensures SourceID column is converted to string type.

    Args:
        catalogue_df: Input DataFrame
        config: Optional configuration DotMap for validation

    Returns:
        Preprocessed DataFrame with reset index and string SourceID
    """
    logger.info(f"Preprocessing catalogue with {len(catalogue_df)} sources")

    # Reset index to ensure contiguous indices
    processed_df = catalogue_df.reset_index(drop=True)

    # Log if index was non-contiguous
    if not catalogue_df.index.equals(pd.RangeIndex(len(catalogue_df))):
        logger.info("Reset non-contiguous catalogue index")

    # Ensure SourceID is string type
    if "SourceID" in processed_df.columns:
        try:
            processed_df["SourceID"] = processed_df["SourceID"].astype(str)
            logger.debug("Converted SourceID column to string type")
        except Exception as e:
            logger.warning(f"Could not convert SourceID to string: {e}")

    # Validate extension order if config is provided
    if config is not None:
        validate_extension_order_matches_fits_order(processed_df, config)

    return processed_df


def validate_extension_order_matches_fits_order(catalogue_df: pd.DataFrame, config) -> None:
    """
    Validate that the order of extensions in config matches the order in FITS files.

    TODO: This function needs to be implemented to ensure that:
    1. The order of extensions in config.selected_extensions matches the actual
       order of extensions found in the FITS files referenced by the catalogue
    2. Channel weights are applied in the correct order corresponding to the
       actual FITS file structure
    3. Multi-channel processing uses consistent extension ordering across all sources

    This is critical for ensuring that channel combination weights are applied
    to the correct input channels, preventing silent data corruption where
    e.g., NIR-H weights might be applied to NIR-J data due to ordering mismatches.

    Args:
        catalogue_df: Preprocessed catalogue DataFrame
        config: Configuration DotMap containing selected_extensions and channel_weights

    Raises:
        CatalogueValidationError: If extension ordering is inconsistent
    """
    # TODO: Implement extension order validation
    # See issue.md for detailed implementation requirements
    pass


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
        from astropy.table import Table

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
    import pyarrow.parquet as pq

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

    # Load and validate catalogue
    catalogue_df = load_and_validate_catalogue(catalogue_path)
    num_sources = len(catalogue_df)
    logger.info(f"Found {num_sources} sources in catalogue")

    # Analyze FITS files per source (sample from first few rows)
    sample_size = min(5, len(catalogue_df))
    unique_fits_files = {}  # initialise as dict to use as an "ordered" set

    logger.info(f"Analysing FITS files from first {sample_size} sources...")

    for idx in range(sample_size):
        row = catalogue_df.iloc[idx]

        # Get FITS file paths for this source
        fits_paths_raw = row.get("fits_file_paths", [])
        fits_paths = parse_fits_file_paths(fits_paths_raw)

        for fits_path in fits_paths:
            unique_fits_files[fits_path] = None  # like an ordered set

    # Convert to list for consistent ordering
    unique_fits_files = list(unique_fits_files.keys())

    # Analyze each unique FITS file for extensions
    logger.info(f"Analysing {len(unique_fits_files)} unique FITS files...")

    fits_analysis_results = []
    extensions_by_filter = {}

    for fits_path in unique_fits_files:
        fits_info = analyze_fits_file(fits_path)
        fits_analysis_results.append(fits_info)

        if fits_info["exists"] and fits_info["extensions"]:
            filter_name = fits_info["filter"]
            ext_types = [ext["type"] for ext in fits_info["extensions"]]

            if filter_name not in extensions_by_filter:
                extensions_by_filter[filter_name] = set()

            extensions_by_filter[filter_name].update(ext_types)

    # Create extensions summary for UI display
    extensions_display = []
    for filter_name, ext_types in extensions_by_filter.items():
        ext_list = list(ext_types)
        extensions_display.append({"name": filter_name, "ext": ", ".join(ext_list)})

    # No sorting

    # Calculate average FITS files per source
    total_fits_entries = 0
    valid_sources = 0

    # Test checking a random sample of 100 sources
    sample_size_2 = min(100, len(catalogue_df))
    sample_indices = random.sample(range(len(catalogue_df)), sample_size_2)
    for idx in sample_indices:
        row = catalogue_df.iloc[idx]
        fits_paths_raw = row.get("fits_file_paths", [])
        fits_paths = parse_fits_file_paths(fits_paths_raw)

        if fits_paths:
            total_fits_entries += len(fits_paths)
            valid_sources += 1
    if sample_size_2 < len(catalogue_df):
        logger.info(f"Sampled {sample_size_2} sources for average FITS per source calculation")
        valid_sources = len(catalogue_df)
        total_fits_entries = int(total_fits_entries * (len(catalogue_df) / sample_size_2))
    avg_fits_per_source = total_fits_entries / valid_sources if valid_sources > 0 else 0

    result = {
        "num_sources": num_sources,
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
        "validated": True,
        "preprocessed": True,
    }

    logger.info(
        f"Catalogue analysis complete: {num_sources} sources, {len(unique_fits_files)} FITS files, {len(extensions_display)}"
        f"filter types"
    )
    return result
