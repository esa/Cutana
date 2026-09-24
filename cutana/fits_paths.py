#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Reading the `fits_file_paths` column.

Its own module because three modules need it and one of them --
`catalogue_preprocessor` -- also needs to call code that parses paths, which made the dependency
circular the moment the source-footprint check arrived. Parsing a column is lower-level than
preprocessing a catalogue, so it belongs below both.
"""

import ast
import os
from typing import List


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
