#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Check that a catalogue's sources actually fall inside the products they name.

The failure this exists to catch is silent. A source outside its FITS file's footprint still
produces a cutout -- an array of the requested size, entirely edge-trim -- so a run completes,
writes N files, and reports nothing wrong. Nothing else in the validation path notices: the
columns are present, the coordinates are legal sky positions, and the file is readable. Only the
pairing of the two is wrong.

Two outcomes, and the difference decides what a user does next:

* the centre is outside the image -- an **error**, since that cutout is entirely trim and the row
  is a mistake
* the centre is inside but the requested box runs past an edge -- a **warning**, since a partial
  cutout is often exactly what was wanted at a survey boundary

`check_sources_in_products` is exported from the package. It is the one validation helper that is,
because it is the one an external tool has a reason to call on its own: a catalogue builder wants
to check its own output before handing it over, and reaching into a submodule for that would be a
dependency with no compatibility promise.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger

from .cutout_extraction import arcsec_to_pixels
from .fits_paths import parse_fits_file_paths
from .validation_sampling import sample_for_validation

#: How many rows to check when the catalogue is larger than this.
#:
#: A systematic mistake -- the wrong tile column, a coordinate frame mix-up -- shows up in the
#: first hundred rows; a single bad row in a million is what the run's own per-source reporting is
#: for.
DEFAULT_SAMPLE_SIZE = 1000

#: How many distinct FITS files to open before giving up and saying so.
#:
#: Each one is a header read, and on the NFS volumes this runs against that is the dominant cost.
#: A multi-band row names four to six tiles, so a 1000-row sample can reach several thousand files
#: without a cap. `check_fits_files_exist` caps at 100 for the same reason.
MAX_FILES_TO_OPEN = 100

#: How many problems to name individually before summarising the rest.
#:
#: A validation report is read by a person. Twenty examples is enough to see the pattern; the
#: count is what says how big it is.
MAX_REPORTED = 20


def read_image_footprint(fits_path: str) -> Optional[Tuple[WCS, Tuple[int, int]]]:
    """Read the celestial WCS and pixel dimensions of a FITS file.

    Args:
        fits_path: File to read.

    Returns:
        The WCS and the (height, width) of the HDU it came from, or None when no HDU has a
        celestial WCS. That is not an error here -- `check_fits_files_exist` owns whether a file is
        usable at all, and this check has nothing to say about a file it cannot place on the sky.

    Raises:
        KeyError: If a celestial HDU has no NAXIS1/NAXIS2. Having found the WCS, an unreadable
            shape is a broken invariant rather than a reason to keep looking: continuing would
            return None and leave the file silently unchecked.
    """
    with fits.open(fits_path, memmap=True) as hdul:
        for hdu in hdul:
            try:
                wcs_obj = WCS(hdu.header, naxis=2)
            except Exception as error:
                # Expected: not every HDU carries celestial coordinates. Logged rather than
                # dropped, because a file where *no* HDU has one is worth being able to explain.
                logger.debug(f"Skipping HDU in {fits_path} (no usable WCS): {error}")
                continue
            if not wcs_obj.has_celestial:
                continue
            return wcs_obj, (int(hdu.header["NAXIS2"]), int(hdu.header["NAXIS1"]))

    logger.warning(f"No HDU in {fits_path} has a celestial WCS; its sources cannot be checked")
    return None


def source_offset_from_image(
    wcs_obj: WCS, shape: Tuple[int, int], ra: float, dec: float
) -> Tuple[float, float, float]:
    """Locate a sky position in an image, and say how far outside it is if it is.

    `world_to_pixel` returns 0-based pixel *centres*, so an image of width W covers -0.5 to
    W - 0.5. Treating it as 0 to W - 1 makes the outer half-pixel of every edge read as outside,
    which turns legitimate sources into errors that fail the whole run.

    Args:
        wcs_obj: The image's celestial WCS.
        shape: The image's (height, width) in pixels.
        ra: Right ascension in degrees.
        dec: Declination in degrees.

    Returns:
        The x and y pixel coordinates, and how many pixels outside the image the position falls --
        zero when it is inside. The distance is what makes a report actionable: two pixels past an
        edge is a rounding argument, twenty thousand is the wrong tile.
    """
    x, y = wcs_obj.world_to_pixel(SkyCoord(ra=ra, dec=dec, unit="deg"))
    x, y = float(x), float(y)
    height, width = shape

    outside_x = max(0.0, -0.5 - x, x - (width - 0.5))
    outside_y = max(0.0, -0.5 - y, y - (height - 0.5))
    return x, y, float(np.hypot(outside_x, outside_y))


def cutout_fits_inside(x: float, y: float, shape: Tuple[int, int], diameter: int) -> bool:
    """Whether a box of `diameter` pixels centred on (x, y) lies wholly inside the image.

    The same -0.5 convention as `source_offset_from_image`, so a source that is inside cannot have
    a cutout the two functions disagree about.

    Args:
        x: Column of the centre, 0-based.
        y: Row of the centre, 0-based.
        shape: The image's (height, width) in pixels.
        diameter: Cutout side length in pixels.

    Returns:
        True when nothing would be trimmed.
    """
    height, width = shape
    half = diameter / 2
    return (
        x - half >= -0.5
        and y - half >= -0.5
        and x + half <= width - 0.5
        and y + half <= height - 0.5
    )


def _cutout_diameter_pixels(row: pd.Series, wcs_obj: WCS) -> Optional[int]:
    """The cutout size this row asks for, in pixels, or None if it does not say."""
    if "diameter_pixel" in row.index and pd.notna(row["diameter_pixel"]):
        return int(row["diameter_pixel"])
    if "diameter_arcsec" in row.index and pd.notna(row["diameter_arcsec"]):
        return arcsec_to_pixels(float(row["diameter_arcsec"]), wcs_obj)
    return None


def _summarise(problems: List[str], subjects: set, headline: str) -> List[str]:
    """Name the first few problems and count the sources behind them.

    Args:
        problems: One line per problem, in the order they were found.
        subjects: The distinct `SourceID`s involved. Counted rather than `len(problems)`, because
            one source outside all four of its bands is one bad source and four bad pairings, and
            the headline says "source".
        headline: A format string taking the source count.

    Returns:
        The headline, the first `MAX_REPORTED` problems, and a count of the rest.
    """
    if not problems:
        return []
    reported = problems[:MAX_REPORTED]
    if len(problems) > len(reported):
        reported.append(f"... and {len(problems) - len(reported)} more")
    return [headline.format(len(subjects))] + reported


def check_sources_in_products(
    catalogue_df: pd.DataFrame, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> Tuple[List[str], List[str]]:
    """Check that each source falls inside every FITS file its row names.

    Returns the same `(errors, warnings)` shape as `check_fits_files_exist`, so it slots into the
    same place in a validation report.

    Args:
        catalogue_df: Catalogue with `SourceID`, `RA`, `Dec` and `fits_file_paths`, and one of
            `diameter_pixel` or `diameter_arcsec`.
        sample_size: Rows to check when the catalogue is larger than this.

    Returns:
        Errors for sources whose position falls outside a file they name, warnings for cutouts
        that fit only partly inside one. A file with no readable celestial WCS produces neither --
        this check has nothing to say about a file it cannot place on the sky, and
        `check_fits_files_exist` owns whether the file is usable at all.

        A malformed `fits_file_paths` cell is an error rather than an exception, because this runs
        inside a validation pass whose job is to collect problems rather than stop at the first.
    """
    outside: List[str] = []
    trimmed: List[str] = []
    outside_sources: set = set()
    trimmed_sources: set = set()

    # One header read per distinct file rather than per row: a catalogue is usually many sources
    # across few tiles, and the same file is named over and over.
    footprints: Dict[str, Optional[Tuple[WCS, Tuple[int, int]]]] = {}
    unopened: set = set()

    sample = sample_for_validation(catalogue_df, sample_size, "source positions")
    for _, row in sample.iterrows():
        source_id = row["SourceID"]
        ra, dec = float(row["RA"]), float(row["Dec"])

        try:
            fits_paths = parse_fits_file_paths(row["fits_file_paths"])
        except ValueError as error:
            # The same treatment `check_fits_files_exist` gives it. Raising here would abort a
            # validation pass that exists to report every problem at once -- and that check stops
            # parsing after 100 unique files, so it may never reach this row at all.
            outside.append(f"{source_id}: {error}")
            outside_sources.add(source_id)
            continue

        for fits_path in fits_paths:
            if not fits_path or not os.path.exists(fits_path):
                # `check_fits_files_exist` reports this, and reporting it twice in one report
                # makes one problem look like two.
                continue

            if fits_path not in footprints:
                if len(footprints) >= MAX_FILES_TO_OPEN:
                    unopened.add(fits_path)
                    continue
                footprints[fits_path] = read_image_footprint(fits_path)

            footprint = footprints[fits_path]
            if footprint is None:
                continue

            wcs_obj, shape = footprint
            x, y, distance = source_offset_from_image(wcs_obj, shape, ra, dec)
            name = os.path.basename(fits_path)

            if distance > 0:
                outside_sources.add(source_id)
                outside.append(
                    f"{source_id}: RA {ra:.5f} Dec {dec:.5f} is {distance:.1f} pixels outside "
                    f"{name} ({shape[1]}x{shape[0]})"
                )
                continue

            diameter = _cutout_diameter_pixels(row, wcs_obj)
            if diameter is None:
                continue
            if not cutout_fits_inside(x, y, shape, diameter):
                trimmed_sources.add(source_id)
                trimmed.append(
                    f"{source_id}: the {diameter} pixel cutout runs past the edge of {name} "
                    "and will be trimmed"
                )

    if unopened:
        # Said out loud: a clean report over a capped set is not a clean report over the whole
        # catalogue, and silence here would read as one.
        logger.warning(
            f"Checked {len(footprints)} FITS files and stopped; {len(unopened)} more were named "
            "by the sampled rows and were not opened"
        )

    return (
        _summarise(
            outside,
            outside_sources,
            "{} source(s) fall outside a FITS file their row names, so their cutouts would be "
            "entirely edge-trim:",
        ),
        _summarise(
            trimmed,
            trimmed_sources,
            "{} source(s) have a cutout that extends past the edge of its FITS file and will be "
            "trimmed:",
        ),
    )
