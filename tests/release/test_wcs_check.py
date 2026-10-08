#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the release-test WCS regression check.

The check itself runs on real Euclid tiles during a release run, where a silently
no-op check would be worse than no check at all. These tests pin it on a synthetic
tile: a header from the real ``create_wcs_header`` must pass, and headers carrying the
two historical WCS bugs (#240 sub-pixel CRPIX, #390 re-tangented projection) must fail.
"""

import json

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS, Sip
from wcs_check import (
    _channel_of_hdu,
    _check_one_cutout_file,
    _parent_paths_by_channel,
    _recorded_size,
    check_cutout_wcs_against_parent,
    verify_cutout_wcs,
)

from cutana import cutout_writer_fits
from cutana.cutout_writer_fits import create_wcs_header
from cutana.get_default_config import get_default_config
from cutana.orchestrator import Orchestrator

TILE_DEC = -51.5
TILE_SIZE = 19200
VIS_SCALE_DEG = 2.7777778e-05  # 0.1 arcsec/px


def _tile_wcs() -> WCS:
    """Euclid-like undistorted TAN mosaic: 19200 px at 0.1 arcsec/px."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [57.9990741, TILE_DEC]
    wcs.wcs.crpix = [TILE_SIZE / 2.0, TILE_SIZE / 2.0]
    wcs.wcs.cd = [[-VIS_SCALE_DEG, 0.0], [0.0, VIS_SCALE_DEG]]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs


def _cutout_header(tile, source_x, source_y, extraction_size, final_size):
    """Build a cutout header through the real writer, as the pipeline does."""
    cutout_writer_fits._wcs_header_cache.clear()
    ra, dec = tile.wcs_pix2world(source_x, source_y, 0)
    origin_x = int(source_x - extraction_size // 2)
    origin_y = int(source_y - extraction_size // 2)
    header = create_wcs_header(
        (final_size, final_size),
        original_wcs=tile,
        ra_source=float(ra),
        dec_source=float(dec),
        extraction_origin_x=origin_x,
        extraction_origin_y=origin_y,
        extraction_size=extraction_size,
    )
    return header, float(ra), float(dec)


def _write_cutout_file(
    path, tiles, channel_names, source_pixel, size, source_id="s0", sized_by="pixel"
):
    """Write a cutout file shaped like the writer's output: metadata primary + image HDUs.

    ``source_pixel`` locates the source in the *first* tile; the others are cut at the
    same sky position, which is a different parent pixel when the grids differ.

    ``sized_by`` picks which catalogue column sized the source, and the primary header
    follows the writer: the column that did size it carries the value, the other is left
    UNDEFINED (``metadata.get`` returned ``None``). ``"neither"`` writes both undefined,
    the only case that leaves the resize cross-check unexercised.
    """
    tiles = tiles if isinstance(tiles, list) else [tiles]
    ra, dec = tiles[0].wcs_pix2world(source_pixel[0], source_pixel[1], 0)

    primary = fits.PrimaryHDU()
    primary.header["SOURCE"] = source_id
    primary.header["RA"] = float(ra)
    primary.header["DEC"] = float(dec)
    primary.header["SIZEPIX"] = size if sized_by == "pixel" else None
    primary.header["SIZEARC"] = size * VIS_SCALE_DEG * 3600.0 if sized_by == "arcsec" else None

    hdus = [primary]
    for tile, name in zip(tiles, channel_names):
        source_x, source_y = tile.wcs_world2pix(float(ra), float(dec), 0)
        header, _, _ = _cutout_header(tile, float(source_x), float(source_y), size, size)
        image = fits.ImageHDU(data=np.zeros((size, size), dtype=np.float32))
        image.header.update(header)
        image.header["CHANNEL"] = name
        hdus.append(image)
    fits.HDUList(hdus).writeto(path, overwrite=True)


@pytest.mark.parametrize(
    "extraction_size,final_size",
    [
        (1800, 1800),  # no resize
        (300, 150),  # downsample
        (16, 224),  # heavy upsample: half a parent pixel is 7 cutout pixels
    ],
)
def test_pipeline_header_passes(extraction_size, final_size):
    """A header from the production WCS builder agrees with the parent tile."""
    tile = _tile_wcs()
    header, ra, dec = _cutout_header(tile, 17800.3, 17800.7, extraction_size, final_size)

    result = check_cutout_wcs_against_parent(header, (final_size, final_size), tile, ra, dec, None)

    assert result["ok"], result["reasons"]
    assert result["resize"] == pytest.approx(final_size / extraction_size, rel=1e-9)
    # The source sits where the integer extraction window put it, within a pixel of
    # the centre on each axis but not forced onto it.
    assert 0.0 < result["source_offset_px"] < 2.0


def test_half_pixel_crpix_shift_is_caught():
    """A half-pixel CRPIX error survives a pixel diff but not this check (issue #240)."""
    tile = _tile_wcs()
    final_size = 1800
    header, ra, dec = _cutout_header(tile, 17800.3, 17800.7, final_size, final_size)
    header["CRPIX1"] = header["CRPIX1"] + 0.5

    result = check_cutout_wcs_against_parent(header, (final_size, final_size), tile, ra, dec, None)

    assert not result["ok"]
    assert result["origin_residual"] == pytest.approx(0.5, abs=1e-6)
    assert any("pixel-aligned" in reason for reason in result["reasons"])


def test_retangented_projection_is_caught():
    """CRVAL at the source with CRPIX at the centre rotates the frame (issue #390)."""
    tile = _tile_wcs()
    final_size = 1800
    header, ra, dec = _cutout_header(tile, 17800.3, 17800.7, final_size, final_size)
    # The pre-#390 construction: re-tangent at the source, centre the source.
    header["CRVAL1"] = ra
    header["CRVAL2"] = dec
    header["CRPIX1"] = final_size / 2.0 + 0.5
    header["CRPIX2"] = final_size / 2.0 + 0.5

    result = check_cutout_wcs_against_parent(header, (final_size, final_size), tile, ra, dec, None)

    assert not result["ok"]
    assert any("translation+scale" in reason for reason in result["reasons"])


def test_resize_without_cd_rescaling_is_caught():
    """CD left unscaled while CRPIX assumes the resize: the implied origin stops being integral."""
    tile = _tile_wcs()
    extraction_size, final_size = 300, 150
    header, ra, dec = _cutout_header(tile, 17800.3, 17800.7, extraction_size, final_size)
    # Undo the writer's pixel-scale rescaling, leaving CRPIX correct for the resize.
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2"):
        if key in header:
            header[key] = header[key] * (final_size / extraction_size)

    result = check_cutout_wcs_against_parent(header, (final_size, final_size), tile, ra, dec, None)

    assert not result["ok"]
    assert any("pixel-aligned" in reason for reason in result["reasons"])


def test_self_consistent_wrong_resize_is_caught():
    """A wrong resize used for both CRPIX and CD passes every internal property.

    ``create_wcs_header`` builds CRPIX and the CD rescaling from one ``resize`` value,
    so dropping the padding factor from ``extraction_size`` leaves a header that is
    affine, pixel-aligned and entirely wrong about its sky footprint. Only the
    catalogue size catches it.
    """
    tile = _tile_wcs()
    final_size = 150
    header, ra, dec = _cutout_header(tile, 17800.3, 17800.7, 300, final_size)

    consistent = check_cutout_wcs_against_parent(
        header, (final_size, final_size), tile, ra, dec, expected_extraction_size=300
    )
    assert consistent["ok"], consistent["reasons"]

    result = check_cutout_wcs_against_parent(
        header, (final_size, final_size), tile, ra, dec, expected_extraction_size=600
    )
    assert not result["ok"]
    assert any("wrong sky footprint" in reason for reason in result["reasons"])
    # The blind spot this closes: nothing else about the header looks wrong.
    assert result["affine_spread"] < 1e-6
    assert result["origin_residual"] < 1e-6


def test_a_correct_run_passes_end_to_end(tmp_path):
    """The PASS case for the entry point ``release_test.py`` actually calls.

    Every other ``verify_cutout_wcs`` test here asserts FAIL, so a change that made the check
    always fail would ship green in the file whose job is to stop it silently becoming a no-op.
    This also exercises the catalogue -> ``_parent_paths_by_channel`` -> extensions wiring, which
    the ``_check_one_cutout_file`` tests only reach from below.
    """
    tile = _tile_wcs()
    tile_path = tmp_path / "VIS_tile.fits"
    fits.PrimaryHDU(header=tile.to_header()).writeto(tile_path)
    catalogue = pd.DataFrame({"SourceID": ["s0"], "fits_file_paths": [str([str(tile_path)])]})
    _write_cutout_file(tmp_path / "s0.fits", tile, ["channel_1"], (17800.3, 17800.7), 128)

    stats = verify_cutout_wcs(tmp_path, catalogue, [{"name": "VIS", "ext": "PRIMARY"}], {"s0"}, 1.0)

    assert stats["status"] == "PASS", stats["failures"]
    assert stats["n_sources"] == 1
    assert stats["n_hdus"] == 1
    assert stats["n_failed"] == 0
    assert stats["n_missing"] == 0
    # The diagnostics the release report quotes: a real measurement, not a default.
    assert stats["max_affine_spread"] < 1e-6
    assert stats["max_origin_residual"] < 1e-6
    assert stats["max_source_offset"] > 0.0


def test_a_distorted_parent_is_named_as_such(tmp_path):
    """A SIP parent must not be reported as a re-tangented CRVAL on every source.

    ``create_wcs_header`` copies the parent header and shifts CRPIX without propagating SIP --
    it warns about this and points at issue #238. The residual is real; the diagnosis was not.
    """
    tile = _tile_wcs()
    tile.sip = Sip(
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        tile.wcs.crpix,
    )
    # A mild quadratic term, well under a pixel across the frame.
    tile.sip.a[2, 0] = 1e-9
    tile.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    final_size = 128
    header, ra, dec = _cutout_header(tile, 17800.3, 17800.7, final_size, final_size)

    result = check_cutout_wcs_against_parent(header, (final_size, final_size), tile, ra, dec, None)

    assert not result["ok"]
    assert "issue #238" in result["reasons"][0]
    assert "SIP distortion" in result["reasons"][0]


def test_missing_sampled_sources_fail(tmp_path):
    """A sample that mostly produced no output must not report a green check."""
    tile = _tile_wcs()
    tile_path = tmp_path / "VIS_tile.fits"
    fits.PrimaryHDU(header=tile.to_header()).writeto(tile_path)
    catalogue = pd.DataFrame(
        {"SourceID": ["s0", "s1"], "fits_file_paths": [str([str(tile_path)])] * 2}
    )
    _write_cutout_file(tmp_path / "s0.fits", tile, ["channel_1"], (17800.3, 17800.7), 128)

    stats = verify_cutout_wcs(
        tmp_path, catalogue, [{"name": "VIS", "ext": "PRIMARY"}], {"s0", "s1"}, 1.0
    )

    assert stats["status"] == "FAIL"
    assert stats["n_missing"] == 1
    assert stats["n_hdus"] == 1
    assert any("no cutout under" in failure for failure in stats["failures"])


def test_hdu_to_tile_mapping_is_positional(tmp_path):
    """HDU n is checked against input channel n, since the writer names HDUs generically.

    Pinned with a VIS-like and a NISP-like tile, whose pixel scales differ, so a swapped
    mapping cannot pass. Note the limit this exposes: the check allows any whole-pixel
    translation, so a swap between two tiles on the *same* grid — two NISP bands, say —
    is invisible to it and stays the pixel comparison's job.
    """
    tile_vis = _tile_wcs()
    tile_nisp = _tile_wcs()
    tile_nisp.wcs.cd = [[-3 * VIS_SCALE_DEG, 0.0], [0.0, 3 * VIS_SCALE_DEG]]

    paths = {}
    for name, wcs in (("VIS", tile_vis), ("NIR-H", tile_nisp)):
        path = tmp_path / f"{name}_tile.fits"
        fits.PrimaryHDU(header=wcs.to_header()).writeto(path)
        paths[name] = str(path)

    cutout_path = tmp_path / "cutout.fits"
    _write_cutout_file(
        cutout_path, [tile_vis, tile_nisp], ["channel_1", "channel_2"], (17800.3, 17800.7), 128
    )
    extensions = {"VIS": "PRIMARY", "NIR-H": "PRIMARY"}

    results, failures = _check_one_cutout_file(cutout_path, paths, extensions, 1.0)
    assert not failures, failures
    assert len(results) == 2

    swapped = {"VIS": paths["NIR-H"], "NIR-H": paths["VIS"]}
    _, swapped_failures = _check_one_cutout_file(cutout_path, swapped, extensions, 1.0)
    assert swapped_failures


def test_tensor_order_follows_the_catalogue_not_selected_extensions(tmp_path):
    """The HDU-to-tile mapping must key off the row's tile order, not the config's.

    ``combine_channels`` resolves ``channel_weights`` onto the tensor by name, so
    ``selected_extensions`` order says nothing about which band is which HDU. Deriving
    the positional mapping from it made this check attribute a cutout to the wrong
    parent tile whenever the two orders disagreed — a false verdict on exactly the
    check meant to catch band swaps.
    """
    tile_vis = _tile_wcs()
    tile_nisp = _tile_wcs()
    tile_nisp.wcs.cd = [[-3 * VIS_SCALE_DEG, 0.0], [0.0, 3 * VIS_SCALE_DEG]]

    paths = {}
    for name, wcs in (("VIS", tile_vis), ("NIR-H", tile_nisp)):
        path = tmp_path / f"{name}_tile.fits"
        fits.PrimaryHDU(header=wcs.to_header()).writeto(path)
        paths[name] = str(path)

    # The catalogue lists NIR-H first, so the tensor does too.
    catalogue = pd.DataFrame(
        {"SourceID": ["s0"], "fits_file_paths": [str([paths["NIR-H"], paths["VIS"]])]}
    )
    _write_cutout_file(
        tmp_path / "s0.fits",
        [tile_nisp, tile_vis],
        ["channel_1", "channel_2"],
        (5900.3, 5900.7),
        128,
    )

    # ...while the config names them in the opposite order.
    stats = verify_cutout_wcs(
        tmp_path,
        catalogue,
        [{"name": "VIS", "ext": "PRIMARY"}, {"name": "NIR-H", "ext": "PRIMARY"}],
        {"s0"},
        1.0,
    )

    assert stats["status"] == "PASS", stats["failures"]
    assert stats["n_hdus"] == 2


def test_unselected_bands_take_no_tensor_column(tmp_path):
    """A row may name tiles the run never loads; those take no HDU and no tensor column.

    Band filtering drops any tile outside ``selected_extensions`` before loading, so a
    three-tile row run as two bands yields two HDUs. Counting the dropped tile as a
    tensor column shifts every later HDU onto the wrong parent — the shape of the
    ``3nisp3`` release run, whose catalogue leads with the unselected VIS tile.
    """
    tile_vis = _tile_wcs()
    tile_nisp = _tile_wcs()
    tile_nisp.wcs.cd = [[-3 * VIS_SCALE_DEG, 0.0], [0.0, 3 * VIS_SCALE_DEG]]

    paths = {}
    for name, wcs in (("VIS", tile_vis), ("NIR-H", tile_nisp), ("NIR-Y", tile_nisp)):
        path = tmp_path / f"{name}_tile.fits"
        fits.PrimaryHDU(header=wcs.to_header()).writeto(path)
        paths[name] = str(path)

    # The catalogue leads with VIS, which the run below does not select.
    catalogue = pd.DataFrame(
        {
            "SourceID": ["s0"],
            "fits_file_paths": [str([paths["VIS"], paths["NIR-H"], paths["NIR-Y"]])],
        }
    )
    _write_cutout_file(
        tmp_path / "s0.fits",
        [tile_nisp, tile_nisp],
        ["channel_1", "channel_2"],
        (5900.3, 5900.7),
        128,
    )

    stats = verify_cutout_wcs(
        tmp_path,
        catalogue,
        [{"name": "NIR-H", "ext": "PRIMARY"}, {"name": "NIR-Y", "ext": "PRIMARY"}],
        {"s0"},
        1.0,
    )

    assert stats["status"] == "PASS", stats["failures"]
    # Both HDUs checked: a dropped tile must not cost the run a column of coverage.
    assert stats["n_hdus"] == 2


def test_ambiguous_tile_names_raise():
    """Two tiles that resolve to the same channel must not silently collapse onto one key."""
    with pytest.raises(ValueError, match="both resolve to channel"):
        _parent_paths_by_channel(
            str(["/tiles/EUC_MER_VIS_TILE1.fits", "/tiles/EUC_MER_VIS_TILE2.fits"])
        )


def test_unrecognised_tiles_collide_because_they_share_one_label():
    """Two tiles the Euclid rules cannot classify are both UNKNOWN, so neither is nameable.

    The label has to be a constant -- `channel_weights` is one dictionary for the whole
    run -- so a row carrying two of them cannot be mapped back to parent tiles by name,
    and this check is what reports that rather than guessing.
    """
    with pytest.raises(ValueError, match="both resolve to channel"):
        _parent_paths_by_channel(str(["/tiles/mystery_a.fits", "/tiles/mystery_b.fits"]))


def test_extra_image_hdu_raises():
    """More image HDUs than input channels makes the positional mapping meaningless."""
    hdu = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32))
    hdu.header["CHANNEL"] = "channel_3"
    with pytest.raises(ValueError, match="no counterpart"):
        _channel_of_hdu(hdu, 2, ["VIS", "NIR-H"])


def test_empty_output_directory_fails(tmp_path):
    """Nothing checked is reported as a failure, never as a silent pass."""
    catalogue = pd.DataFrame(
        {"SourceID": ["s0"], "fits_file_paths": [str(["/nonexistent/VIS_tile.fits"])]}
    )

    stats = verify_cutout_wcs(tmp_path, catalogue, [{"name": "VIS", "ext": "PRIMARY"}], {"s0"}, 1.0)

    assert stats["status"] == "FAIL"
    assert stats["n_hdus"] == 0
    assert any("no cutout WCS could be checked" in failure for failure in stats["failures"])


def test_probe_residuals_are_exact_for_a_correct_header():
    """The residual really is the integer extraction origin, at every probe point."""
    tile = _tile_wcs()
    source_x, source_y = 12345.6, 9876.4
    extraction_size = final_size = 256
    header, ra, dec = _cutout_header(tile, source_x, source_y, extraction_size, final_size)

    result = check_cutout_wcs_against_parent(header, (final_size, final_size), tile, ra, dec, None)

    assert result["ok"], result["reasons"]
    assert result["affine_spread"] < 1e-6
    assert result["origin_residual"] < 1e-6
    assert np.isclose(result["resize"], 1.0)


def _write_arcsec_cutout(path, tile, extraction_size, final_size, diameter_arcsec):
    """A cutout whose catalogue row sized it in arcsec, so SIZEPIX is UNDEFINED."""
    source_x, source_y = 17800.3, 17800.7
    header, ra, dec = _cutout_header(tile, source_x, source_y, extraction_size, final_size)

    primary = fits.PrimaryHDU()
    primary.header["SOURCE"] = "s0"
    primary.header["RA"] = ra
    primary.header["DEC"] = dec
    primary.header["SIZEPIX"] = None
    primary.header["SIZEARC"] = diameter_arcsec

    image = fits.ImageHDU(data=np.zeros((final_size, final_size), dtype=np.float32))
    image.header.update(header)
    image.header["CHANNEL"] = "channel_1"
    fits.HDUList([primary, image]).writeto(path, overwrite=True)


def test_arcsec_sized_source_is_resize_checked(tmp_path):
    """The resize cross-check must not go inert on a diameter_arcsec catalogue.

    Such a catalogue leaves ``SIZEPIX`` undefined for every source, which used to skip the
    cross-check for the whole run. The size is recovered from ``SIZEARC`` through the parent
    tile's pixel scale, so the self-consistent wrong resize of #390's class is still caught.
    """
    tile = _tile_wcs()
    tile_path = tmp_path / "VIS_tile.fits"
    fits.PrimaryHDU(header=tile.to_header()).writeto(tile_path)
    paths = {"VIS": str(tile_path)}
    extensions = {"VIS": "PRIMARY"}
    arcsec_per_px = VIS_SCALE_DEG * 3600.0

    # 30" at 0.1"/px is the 300 px window the header was actually built for.
    good = tmp_path / "good.fits"
    _write_arcsec_cutout(good, tile, 300, 150, 300 * arcsec_per_px)
    results, failures = _check_one_cutout_file(good, paths, extensions, 1.0)
    assert not failures, failures
    assert results[0]["resize_checked"]

    # Same header, catalogue says the window should have been twice as wide: the WCS is
    # affine and pixel-aligned and still describes the wrong sky footprint.
    bad = tmp_path / "bad.fits"
    _write_arcsec_cutout(bad, tile, 300, 150, 600 * arcsec_per_px)
    results, failures = _check_one_cutout_file(bad, paths, extensions, 1.0)
    assert any("wrong sky footprint" in failure for failure in failures), failures
    assert results[0]["affine_spread"] < 1e-6
    assert results[0]["origin_residual"] < 1e-6


def test_source_with_no_size_card_is_counted_not_failed(tmp_path):
    """A source the catalogue sized in neither column loses only the resize sub-check."""
    tile = _tile_wcs()
    tile_path = tmp_path / "VIS_tile.fits"
    fits.PrimaryHDU(header=tile.to_header()).writeto(tile_path)
    catalogue = pd.DataFrame({"SourceID": ["s0"], "fits_file_paths": [str([str(tile_path)])]})
    _write_cutout_file(
        tmp_path / "s0.fits", tile, ["channel_1"], (17800.3, 17800.7), 128, sized_by="neither"
    )

    stats = verify_cutout_wcs(tmp_path, catalogue, [{"name": "VIS", "ext": "PRIMARY"}], {"s0"}, 1.0)

    assert stats["status"] == "PASS", stats["failures"]
    assert stats["n_resize_unchecked"] == 1
    assert stats["n_hdus"] == 1


def test_a_broken_file_is_recorded_not_raised(tmp_path):
    """One unmappable cutout fails the check without aborting a multi-hour release run."""
    tile = _tile_wcs()
    tile_path = tmp_path / "VIS_tile.fits"
    fits.PrimaryHDU(header=tile.to_header()).writeto(tile_path)
    catalogue = pd.DataFrame(
        {
            "SourceID": ["s0", "s1"],
            "fits_file_paths": [
                str(["/tiles/mystery_a.fits", "/tiles/mystery_b.fits"]),
                str([str(tile_path)]),
            ],
        }
    )
    _write_cutout_file(tmp_path / "s0.fits", tile, ["channel_1"], (17800.3, 17800.7), 128, "s0")
    _write_cutout_file(tmp_path / "s1.fits", tile, ["channel_1"], (17800.3, 17800.7), 128, "s1")

    stats = verify_cutout_wcs(
        tmp_path, catalogue, [{"name": "VIS", "ext": "PRIMARY"}], {"s0", "s1"}, 1.0
    )

    assert stats["status"] == "FAIL"
    assert any("both resolve to channel" in failure for failure in stats["failures"])
    # The healthy source was still checked: the bad row cost its own file, not the run.
    assert stats["n_hdus"] == 1
    assert stats["n_missing"] == 0


def test_arcsec_window_is_sized_once_not_per_channel(tmp_path):
    """A multi-resolution set is sized from one tile, the way the pipeline sizes it.

    ``all_source_offsets`` keeps the first tile file's extraction window and every channel's
    WCS is built from it, so converting the arcsec diameter with each channel's own pixel
    scale would report a wrong footprint for every non-first channel of a run that behaved
    exactly as the pipeline defines it. Here the NISP tile is 3x coarser, so a per-channel
    conversion would expect a 100 px window against the 300 px one actually used.
    """
    tile_vis = _tile_wcs()
    tile_nisp = _tile_wcs()
    tile_nisp.wcs.cd = [[-3 * VIS_SCALE_DEG, 0.0], [0.0, 3 * VIS_SCALE_DEG]]

    paths = {}
    for name, wcs in (("VIS", tile_vis), ("NIR-H", tile_nisp)):
        path = tmp_path / f"{name}_tile.fits"
        fits.PrimaryHDU(header=wcs.to_header()).writeto(path)
        paths[name] = str(path)

    extraction_size, final_size = 300, 150
    ra, dec = tile_vis.wcs_pix2world(17800.3, 17800.7, 0)
    primary = fits.PrimaryHDU()
    primary.header["SOURCE"] = "s0"
    primary.header["RA"] = float(ra)
    primary.header["DEC"] = float(dec)
    primary.header["SIZEPIX"] = None
    # 30" is the 300 px window at the VIS scale that sized the source.
    primary.header["SIZEARC"] = extraction_size * VIS_SCALE_DEG * 3600.0

    hdus = [primary]
    for tile, name in ((tile_vis, "channel_1"), (tile_nisp, "channel_2")):
        source_x, source_y = tile.wcs_world2pix(float(ra), float(dec), 0)
        header, _, _ = _cutout_header(
            tile, float(source_x), float(source_y), extraction_size, final_size
        )
        image = fits.ImageHDU(data=np.zeros((final_size, final_size), dtype=np.float32))
        image.header.update(header)
        image.header["CHANNEL"] = name
        hdus.append(image)
    cutout_path = tmp_path / "cutout.fits"
    fits.HDUList(hdus).writeto(cutout_path)

    results, failures = _check_one_cutout_file(
        cutout_path, paths, {"VIS": "PRIMARY", "NIR-H": "PRIMARY"}, 1.0
    )

    assert not failures, failures
    assert [result["resize_checked"] for result in results] == [True, True]


def _run_orchestrator_fits(tmp_path, catalogue_row, padding_factor, target_resolution):
    """Produce real FITS cutouts from a synthetic VIS tile through the real pipeline.

    Mirrors the release matrix's ``disk`` backend: the same Orchestrator, output format and
    single-VIS channel configuration, on a tile small enough to run in CI.
    """
    tile = _tile_wcs()
    # A tile the source fits inside, at the Euclid VIS scale the WCS above encodes.
    tile.wcs.crpix = [256.0, 256.0]
    tile_path = tmp_path / "VIS_synth_tile.fits"
    header = tile.to_header()
    header["EXTNAME"] = "PRIMARY"
    fits.PrimaryHDU(
        data=np.random.default_rng(0).normal(size=(512, 512)).astype(np.float32), header=header
    ).writeto(tile_path, overwrite=True)

    ra, dec = tile.wcs_pix2world(251.3, 262.7, 0)
    catalogue = pd.DataFrame(
        [
            {
                "SourceID": "arcsec_0",
                "RA": float(ra),
                "Dec": float(dec),
                "fits_file_paths": json.dumps([str(tile_path)]),
                **catalogue_row,
            }
        ]
    )
    catalogue_path = tmp_path / "catalogue.csv"
    catalogue.to_csv(catalogue_path, index=False)

    output_dir = tmp_path / "cutana_output"
    config = get_default_config()
    config.source_catalogue = str(catalogue_path)
    config.output_dir = str(output_dir)
    config.output_format = "fits"
    config.data_type = "float32"
    config.normalisation_method = "none"
    config.apply_flux_conversion = False
    config.max_workers = 1
    config.padding_factor = padding_factor
    config.target_resolution = target_resolution
    config.channel_weights = {"VIS": [1.0]}
    config.fits_extensions = ["PRIMARY"]
    config.selected_extensions = [{"name": "VIS", "ext": "PRIMARY"}]
    config.available_extensions = [{"name": "VIS", "ext": "PRIMARY"}]

    Orchestrator(config).run()
    return output_dir, catalogue


def test_arcsec_catalogue_end_to_end_through_the_pipeline(tmp_path):
    """The SIZEARC branch against cutouts the real pipeline produced, not hand-built headers.

    Every other arcsec test here writes the primary header itself, so nothing pinned that the
    writer emits the cards the check reads, or that the reconstruction lands on the same window
    the extraction path actually used. The Q1 release catalogues are ``diameter_pixel`` only, so
    this is the only place the arcsec path is exercised end to end.
    """
    # 12.8" at the tile's 0.1 arcsec/px is a 128 px diameter; padding widens the window to 192.
    output_dir, catalogue = _run_orchestrator_fits(
        tmp_path, {"diameter_arcsec": 12.8}, padding_factor=1.5, target_resolution=64
    )

    written = list(output_dir.rglob("*.fits"))
    assert written, "the pipeline wrote no FITS cutout"
    primary = fits.getheader(written[0], 0)
    # The premise of the branch: an arcsec catalogue records SIZEARC and no usable SIZEPIX.
    assert primary["SIZEARC"] == pytest.approx(12.8)
    assert _recorded_size(primary, "SIZEPIX") is None

    stats = verify_cutout_wcs(
        output_dir, catalogue, [{"name": "VIS", "ext": "PRIMARY"}], {"arcsec_0"}, 1.5
    )

    assert stats["status"] == "PASS", stats["failures"]
    # The point of the test: the cross-check ran rather than skipping, and it agreed with the
    # window the extraction path chose -- int(round(12.8 / 0.1) * 1.5) = 192 parent pixels.
    assert stats["n_resize_unchecked"] == 0
    assert stats["n_hdus"] == 1

    # And it is live on this path: the same output against a wrong padding factor must fail,
    # or the assertion above would hold for a check that never compared anything.
    wrong = verify_cutout_wcs(
        output_dir, catalogue, [{"name": "VIS", "ext": "PRIMARY"}], {"arcsec_0"}, 3.0
    )
    assert wrong["status"] == "FAIL"
    assert any("wrong sky footprint" in failure for failure in wrong["failures"])
