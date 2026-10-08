#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Tests for the source-in-product footprint check."""

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from cutana import source_footprint
from cutana.catalogue_preprocessor import validate_catalogue_sample
from cutana.source_footprint import (
    MAX_REPORTED,
    check_sources_in_products,
    cutout_fits_inside,
    read_image_footprint,
    source_offset_from_image,
)

#: The centre of the synthetic image every test here uses.
CENTRE_RA, CENTRE_DEC = 150.0, 2.0

#: Its size in pixels and its scale. Deliberately small and square, so a pixel offset can be
#: worked out on paper.
SIZE = 200
PIXEL_SCALE_DEG = 0.001


def write_image(path, size=SIZE, celestial=True):
    """Write a FITS file with a celestial WCS centred on CENTRE_RA, CENTRE_DEC."""
    header = fits.Header()
    if celestial:
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRVAL1"] = CENTRE_RA
        header["CRVAL2"] = CENTRE_DEC
        # 0-based pixel (size/2 - 1) is the reference, so the image spans -0.5 .. size-0.5.
        header["CRPIX1"] = size / 2
        header["CRPIX2"] = size / 2
        header["CDELT1"] = -PIXEL_SCALE_DEG
        header["CDELT2"] = PIXEL_SCALE_DEG
    fits.PrimaryHDU(data=np.zeros((size, size), dtype=np.float32), header=header).writeto(path)
    return str(path)


def sky_at_pixel(wcs_obj, x, y):
    """The sky position of a pixel, so a test can place a source exactly where it means to."""
    position = wcs_obj.pixel_to_world(x, y)
    return float(position.ra.deg), float(position.dec.deg)


def catalogue(rows):
    return pd.DataFrame(rows)


def row(path, ra=CENTRE_RA, dec=CENTRE_DEC, source_id="S1", **extra):
    return {
        "SourceID": source_id,
        "RA": ra,
        "Dec": dec,
        "diameter_pixel": 20,
        "fits_file_paths": str([path]),
        **extra,
    }


class TestReadImageFootprint:
    def test_reads_the_wcs_and_the_shape(self, tmp_path):
        found = read_image_footprint(write_image(tmp_path / "image.fits"))

        assert found is not None
        wcs_obj, shape = found
        assert isinstance(wcs_obj, WCS)
        assert shape == (SIZE, SIZE)

    def test_a_file_with_no_celestial_wcs_is_not_an_error(self, tmp_path, caplog):
        """This check has nothing to say about a file it cannot place on the sky.

        Whether the file is usable at all belongs to `check_fits_files_exist`; reporting it here
        as well would make one problem look like two. Silence would be wrong too, though -- the
        file goes entirely unchecked, so it is logged.
        """
        assert read_image_footprint(write_image(tmp_path / "flat.fits", celestial=False)) is None


class TestSourceOffset:
    @pytest.fixture
    def image(self, tmp_path):
        return read_image_footprint(write_image(tmp_path / "image.fits"))

    @pytest.mark.parametrize(
        "x, y, inside",
        [
            # The image spans -0.5 .. SIZE-0.5, because `world_to_pixel` returns pixel centres.
            (0.0, 0.0, True),
            (SIZE - 1.0, SIZE - 1.0, True),
            # Inside the outer half of the last pixel. Treating the image as 0 .. SIZE-1 makes
            # these errors that fail the whole run, reported as "0 pixels outside".
            (-0.4, 0.0, True),
            (SIZE - 0.6, SIZE - 0.6, True),
            # Genuinely past the edge.
            (-0.6, 0.0, False),
            (SIZE - 0.4, 0.0, False),
        ],
    )
    def test_the_edge_is_half_a_pixel_beyond_the_last_centre(self, image, x, y, inside):
        wcs_obj, shape = image

        _, _, distance = source_offset_from_image(wcs_obj, shape, *sky_at_pixel(wcs_obj, x, y))

        assert (distance == 0.0) is inside

    def test_the_distance_says_how_wrong_it_is(self, image):
        """Two pixels past an edge is a rounding argument; twenty thousand is the wrong tile.

        A boolean inside/outside cannot tell those apart, and they call for different actions.
        """
        wcs_obj, shape = image

        _, _, distance = source_offset_from_image(
            wcs_obj, shape, *sky_at_pixel(wcs_obj, SIZE + 99.5, 0.0)
        )

        assert distance == pytest.approx(100.0, abs=0.5)


class TestCutoutFitsInside:
    @pytest.mark.parametrize(
        "x, diameter, fits_inside",
        [
            (100.0, 20, True),
            # Exactly flush with the left edge: the box spans -0.5 .. 19.5.
            (9.5, 20, True),
            (9.4, 20, False),
            (SIZE - 10.5, 20, True),
            (SIZE - 10.4, 20, False),
        ],
    )
    def test_it_uses_the_same_edge_as_the_position_check(self, x, diameter, fits_inside):
        # Two conventions would let a source be inside while its zero-size cutout was not.
        assert cutout_fits_inside(x, 100.0, (SIZE, SIZE), diameter) is fits_inside


class TestCheckSourcesInProducts:
    def test_a_source_inside_its_product_reports_nothing(self, tmp_path):
        path = write_image(tmp_path / "image.fits")

        errors, warnings = check_sources_in_products(catalogue([row(path)]))

        assert errors == []
        assert warnings == []

    def test_a_source_outside_its_product_is_an_error(self, tmp_path):
        """The silent failure this check exists for: the cutout is produced, and is entirely trim.

        Nothing else in the validation path notices -- the columns are there, the coordinates are
        a legal sky position, and the file is readable. Only the pairing is wrong.
        """
        path = write_image(tmp_path / "image.fits")
        wcs_obj, _ = read_image_footprint(path)
        ra, dec = sky_at_pixel(wcs_obj, SIZE + 400, 0.0)

        errors, warnings = check_sources_in_products(catalogue([row(path, ra=ra, dec=dec)]))

        assert len(errors) == 2
        assert errors[0].startswith("1 source(s) fall outside")
        assert "entirely edge-trim" in errors[0]
        # The source and how far out it is, so the report is actionable without reopening the file.
        assert "S1" in errors[1] and "image.fits" in errors[1]
        assert warnings == []

    def test_a_cutout_running_past_an_edge_is_a_warning_not_an_error(self, tmp_path):
        """A partial cutout is often exactly what was wanted at a survey boundary.

        Failing a run for it would make the check unusable on real data; saying nothing would let
        a half-empty cutout pass for a whole one.
        """
        path = write_image(tmp_path / "image.fits")
        wcs_obj, _ = read_image_footprint(path)
        ra, dec = sky_at_pixel(wcs_obj, 4.0, 100.0)

        errors, warnings = check_sources_in_products(
            catalogue([row(path, ra=ra, dec=dec, diameter_pixel=20)])
        )

        assert errors == []
        assert len(warnings) == 2
        assert warnings[0].startswith("1 source(s) have a cutout that extends past")

    def test_a_size_in_arcseconds_is_converted_before_it_is_checked(self, tmp_path):
        # `diameter_pixel` and `diameter_arcsec` are alternatives, and the second reaches a
        # different branch: nothing was exercising the conversion.
        path = write_image(tmp_path / "image.fits")
        wcs_obj, _ = read_image_footprint(path)
        ra, dec = sky_at_pixel(wcs_obj, 4.0, 100.0)
        sized = row(path, ra=ra, dec=dec)
        del sized["diameter_pixel"]
        # 0.001 deg/pixel is 3.6 arcsec/pixel, so 72 arcsec is 20 pixels.
        sized["diameter_arcsec"] = 72.0

        errors, warnings = check_sources_in_products(catalogue([sized]))

        assert errors == []
        assert len(warnings) == 2

    def test_a_row_with_no_size_is_only_checked_for_its_position(self, tmp_path):
        # A catalogue may carry neither column yet -- that is `validate_catalogue_columns`'s
        # complaint, not this one's.
        path = write_image(tmp_path / "image.fits")
        sized = row(path)
        del sized["diameter_pixel"]

        assert check_sources_in_products(catalogue([sized])) == ([], [])

    def test_a_missing_file_is_left_to_the_check_that_owns_it(self, tmp_path):
        # Reporting it here too would make one problem look like two in the same report.
        assert check_sources_in_products(catalogue([row(str(tmp_path / "absent.fits"))])) == (
            [],
            [],
        )

    def test_a_malformed_paths_cell_is_reported_rather_than_raised(self, tmp_path):
        """This runs inside a pass whose job is to collect problems, not stop at the first.

        `check_fits_files_exist` reports it the same way -- and stops parsing after 100 unique
        files, so it may never reach the bad row at all.
        """
        bad = row(tmp_path / "image.fits")
        bad["fits_file_paths"] = "[unbalanced"

        errors, _ = check_sources_in_products(catalogue([bad]))

        assert len(errors) == 2
        assert "Malformed FITS paths string" in errors[1]

    def test_one_source_outside_four_bands_is_one_source(self, tmp_path):
        # The headline says "source". Counting pairings would report a four-band row as four.
        paths = [write_image(tmp_path / f"band{index}.fits") for index in range(4)]
        wcs_obj, _ = read_image_footprint(paths[0])
        ra, dec = sky_at_pixel(wcs_obj, SIZE + 400, 0.0)
        multi = row(paths[0], ra=ra, dec=dec)
        multi["fits_file_paths"] = str(paths)

        errors, _ = check_sources_in_products(catalogue([multi]))

        assert errors[0].startswith("1 source(s) fall outside")
        # One line per pairing underneath, because each names a different file.
        assert len(errors) == 5

    def test_many_bad_rows_are_counted_rather_than_all_listed(self, tmp_path):
        """A validation report is read by a person, and a thousand identical lines is not read."""
        path = write_image(tmp_path / "image.fits")
        wcs_obj, _ = read_image_footprint(path)
        ra, dec = sky_at_pixel(wcs_obj, SIZE + 400, 0.0)
        rows = [
            row(path, ra=ra, dec=dec, source_id=f"S{index}") for index in range(MAX_REPORTED + 10)
        ]

        errors, _ = check_sources_in_products(catalogue(rows))

        assert errors[0].startswith(f"{MAX_REPORTED + 10} source(s) fall outside")
        assert errors[-1] == "... and 10 more"
        assert len(errors) == MAX_REPORTED + 2

    def test_a_large_catalogue_is_sampled_deterministically(self, tmp_path):
        # A report that named different rows on each run would read as a catalogue that keeps
        # changing -- and validation runs twice in a normal session.
        path = write_image(tmp_path / "image.fits")
        wcs_obj, _ = read_image_footprint(path)
        ra, dec = sky_at_pixel(wcs_obj, SIZE + 400, 0.0)
        rows = [row(path, ra=ra, dec=dec, source_id=f"S{index}") for index in range(50)]

        first = check_sources_in_products(catalogue(rows), sample_size=10)
        again = check_sources_in_products(catalogue(rows), sample_size=10)

        assert first[0][0].startswith("10 source(s) fall outside")
        assert first == again

    def test_one_header_read_per_file_however_many_rows_name_it(self, tmp_path, monkeypatch):
        """A catalogue is usually many sources across few tiles, and this opens FITS files."""
        path = write_image(tmp_path / "image.fits")
        reads = []
        real = source_footprint.read_image_footprint

        def counted(fits_path):
            reads.append(fits_path)
            return real(fits_path)

        monkeypatch.setattr(source_footprint, "read_image_footprint", counted)
        check_sources_in_products(catalogue([row(path, source_id=f"S{i}") for i in range(25)]))

        assert reads == [path]

    def test_the_number_of_files_opened_is_capped(self, tmp_path, monkeypatch):
        """Each file is a header read, and on NFS that is the dominant cost.

        A multi-band row names four to six tiles, so a 1000-row sample reaches thousands of files
        without a cap. `check_fits_files_exist` caps for the same reason.
        """
        monkeypatch.setattr(source_footprint, "MAX_FILES_TO_OPEN", 3)
        paths = [write_image(tmp_path / f"image{index}.fits") for index in range(10)]
        reads = []
        real = source_footprint.read_image_footprint
        monkeypatch.setattr(
            source_footprint,
            "read_image_footprint",
            lambda path: (reads.append(path), real(path))[1],
        )

        check_sources_in_products(catalogue([row(path) for path in paths]))

        assert len(reads) == 3


def test_the_streaming_validator_actually_runs_this_check(tmp_path, monkeypatch):
    """One line of wiring, and nothing else would notice it being removed."""
    called = []
    monkeypatch.setattr(
        "cutana.catalogue_preprocessor.check_sources_in_products",
        lambda df: (called.append(len(df)), ([], []))[1],
    )

    path = write_image(tmp_path / "image.fits")
    catalogue_path = tmp_path / "sources.csv"
    catalogue([row(path)]).to_csv(catalogue_path, index=False)

    validate_catalogue_sample(str(catalogue_path))

    assert called == [1]
