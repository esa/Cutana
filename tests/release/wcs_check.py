#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Astrometric regression check for FITS cutouts produced by a release test run.

The ground-truth comparison in ``release_test.py`` is pixel-wise only, so a cutout
whose pixels are perfect but whose WCS is shifted by half a pixel passes it. That is
exactly the failure class of issue #240 (CRPIX assumed the source sits at the cutout
centre) and issue #390 (the projection was re-tangented at the source, rotating the
cutout frame). Both survive a pixel diff and both corrupt every downstream position
measured from a cutout.

The oracle here is the parent tile header, read back from the catalogue's
``fits_file_paths``, not the formula the writer used. A correct cutout WCS must be the
parent WCS composed with an *integer* pixel translation (the extraction origin) and a
uniform scale (the resize). Three properties follow, and all three are checked:

**Affine constancy** — map several cutout pixels to the sky with the cutout WCS and back
to parent pixels with the parent WCS. The residual against the assumed
``parent = origin + (cut + 0.5) / resize - 0.5`` relation must be the *same* at every
probe. A CRVAL re-tangented at the source makes it drift across the frame, by the
meridian convergence between the tile centre and the source.

**Integer origin** — that constant residual must be a whole number of parent pixels,
because the extraction window starts on a pixel boundary. A half-pixel CRPIX error shows
up here and nowhere else, as does a CD matrix rescaled inconsistently with CRPIX.

**Expected resize** — ``create_wcs_header`` derives CRPIX *and* the CD rescaling from one
``resize`` value, so a wrong one stays self-consistent and passes the first two
properties while every cutout claims the wrong sky footprint. The resize is therefore
also checked against ``final_size / int(catalogue_size * padding_factor)``, where the
catalogue size reaches the header by the metadata path rather than the WCS branch:
``SIZEPIX`` directly, or ``SIZEARC`` converted through the parent tile's own pixel
scale, the same way ``arcsec_to_pixels`` sizes an arcsec-catalogue window. Cutouts that
record neither size skip this sub-check, and the run reports how many did.

Together they pin the cutout WCS to the parent WCS up to a whole-pixel translation, which
the pixel-wise ground-truth comparison already covers. That residual freedom is also the
check's limit: two tiles on the same grid — two NISP bands, say — are interchangeable here,
so a channel swap between them is the pixel comparison's job, not this one's.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits as astropy_fits
from astropy.wcs import WCS
from loguru import logger

from cutana.catalogue_preprocessor import extract_filter_name, parse_fits_file_paths
from cutana.cutout_extraction import get_pixel_scale_arcsec_per_pixel

# Residual tolerances in *parent tile* pixels. The round trip through two TAN
# projections is accurate to ~1e-9 px, so these only need to sit far below the
# smallest error worth catching (half a pixel).
AFFINE_TOLERANCE_PX = 0.02
ORIGIN_TOLERANCE_PX = 0.02
# The resize is a ratio of two exactly-representable sizes, so it should match to
# floating-point noise, not to a physically motivated tolerance.
RESIZE_RELATIVE_TOLERANCE = 1e-9

# Parent headers are on NFS; cache them so a tile shared by many sources is read once.
_PARENT_WCS_CACHE: dict[tuple[str, str], WCS] = {}


def _get_parent_wcs(fits_path: str, extension: str) -> WCS:
    """Read (and cache) the WCS of a parent tile extension. Header only, no pixel data."""
    key = (fits_path, extension)
    if key not in _PARENT_WCS_CACHE:
        header = astropy_fits.getheader(fits_path, extension)
        _PARENT_WCS_CACHE[key] = WCS(header)
    return _PARENT_WCS_CACHE[key]


def _resize_factor(cutout_wcs: WCS, parent_wcs: WCS) -> float:
    """Recover final_size / extraction_size from the two pixel scale matrices.

    This is what the header *claims*; on its own it is self-confirming, because the
    writer builds CRPIX and the CD rescaling from the same value. The caller compares
    it against the size recorded by the metadata path (see ``expected_extraction_size``).
    """
    det_parent = abs(np.linalg.det(parent_wcs.pixel_scale_matrix))
    det_cutout = abs(np.linalg.det(cutout_wcs.pixel_scale_matrix))
    if det_cutout == 0.0:
        raise ValueError("Cutout WCS has a singular pixel scale matrix")
    return math.sqrt(det_parent / det_cutout)


def check_cutout_wcs_against_parent(
    cutout_header: astropy_fits.Header,
    cutout_shape: tuple,
    parent_wcs: WCS,
    source_ra: float,
    source_dec: float,
    expected_extraction_size: int | None,
) -> dict:
    """Check one cutout's WCS against the WCS of the tile it was cut from.

    Args:
        cutout_header: Header of the cutout image HDU (carries the cutout WCS).
        cutout_shape: Cutout array shape (height, width).
        parent_wcs: WCS of the parent tile extension this channel was cut from. A parent
            carrying SIP distortion is reported as such rather than measured, since the writer
            does not propagate SIP (issue #238).
        source_ra: Source RA in degrees, as written to the cutout's primary header.
        source_dec: Source Dec in degrees.
        expected_extraction_size: Pre-resize window size in parent pixels, derived from
            the catalogue diameter and the run's padding factor (see
            ``_expected_extraction_size``). ``None`` skips the resize cross-check, for
            sources whose header records no catalogue size at all.

    Returns:
        dict with ``ok``, ``reasons`` (list of failure descriptions), ``resize_checked``
        (whether the resize cross-check ran), and the measured ``affine_spread``,
        ``origin_residual``, ``resize`` and ``source_offset_px`` diagnostics.
    """
    cutout_wcs = WCS(cutout_header)
    height, width = cutout_shape[0], cutout_shape[1]
    resize = _resize_factor(cutout_wcs, parent_wcs)

    # Corners plus centre: enough to expose any rotation or scale drift, since the
    # relation being tested is affine.
    probe_x = np.array([0.0, width - 1.0, 0.0, width - 1.0, (width - 1) / 2.0])
    probe_y = np.array([0.0, 0.0, height - 1.0, height - 1.0, (height - 1) / 2.0])

    ra, dec = cutout_wcs.pixel_to_world_values(probe_x, probe_y)
    parent_x, parent_y = parent_wcs.world_to_pixel_values(ra, dec)

    # Residual = the extraction origin implied by each probe. Constant and integral
    # for a correct cutout WCS.
    residual_x = parent_x - ((probe_x + 0.5) / resize - 0.5)
    residual_y = parent_y - ((probe_y + 0.5) / resize - 0.5)
    affine_spread = float(max(np.ptp(residual_x), np.ptp(residual_y)))

    origin_x = float(np.mean(residual_x))
    origin_y = float(np.mean(residual_y))
    origin_residual = float(max(abs(origin_x - round(origin_x)), abs(origin_y - round(origin_y))))

    source_x, source_y = cutout_wcs.world_to_pixel_values(source_ra, source_dec)
    source_x = float(source_x)
    source_y = float(source_y)
    source_inside = -0.5 <= source_x <= width - 0.5 and -0.5 <= source_y <= height - 0.5
    source_offset_px = float(
        math.hypot(source_x - (width - 1) / 2.0, source_y - (height - 1) / 2.0) / resize
    )

    reasons: list[str] = []
    # Named before the residual reasons, because it explains them. ``create_wcs_header`` copies
    # the parent header and shifts CRPIX without propagating SIP (it warns about exactly this and
    # points at issue #238), so a distorted parent makes every source in the run fail with a
    # residual that is real but whose cause is the dropped distortion -- not a re-tangented CRVAL
    # or a sub-pixel CRPIX error. Without this the check blames the wrong bug on every row.
    if parent_wcs.sip is not None:
        reasons.append(
            "parent tile WCS carries SIP distortion, which the cutout WCS does not propagate "
            "(issue #238), so the cutout is astrometrically wrong by the dropped distortion; "
            "any residuals below follow from that rather than from a re-tangented CRVAL or a "
            "sub-pixel CRPIX error"
        )
    if not math.isfinite(affine_spread) or affine_spread > AFFINE_TOLERANCE_PX:
        reasons.append(
            f"cutout WCS is not a pure translation+scale of the parent WCS "
            f"(residual spreads by {affine_spread:.4f} px across the frame, "
            f"tolerance {AFFINE_TOLERANCE_PX}); re-tangented CRVAL or mis-rescaled CD"
        )
    if not math.isfinite(origin_residual) or origin_residual > ORIGIN_TOLERANCE_PX:
        reasons.append(
            f"extraction origin is not pixel-aligned "
            f"(({origin_x:.4f}, {origin_y:.4f}) is {origin_residual:.4f} px off an "
            f"integer, tolerance {ORIGIN_TOLERANCE_PX}); sub-pixel CRPIX error, cf. issue #240"
        )
    if expected_extraction_size is not None:
        expected_resize = height / expected_extraction_size
        if not math.isclose(resize, expected_resize, rel_tol=RESIZE_RELATIVE_TOLERANCE):
            reasons.append(
                f"cutout WCS claims a resize of {resize:.6f} but the catalogue size and "
                f"padding factor imply {expected_resize:.6f} "
                f"({height}/{expected_extraction_size}); the cutout covers the wrong "
                f"sky footprint even though its WCS is internally consistent"
            )
    if not source_inside:
        reasons.append(
            f"source maps to ({source_x:.2f}, {source_y:.2f}), outside the {width}x{height} cutout"
        )

    return {
        "ok": not reasons,
        "reasons": reasons,
        "affine_spread": affine_spread,
        "origin_residual": origin_residual,
        "resize": resize,
        "source_offset_px": source_offset_px,
        "resize_checked": expected_extraction_size is not None,
    }


def _recorded_size(header: astropy_fits.Header, key: str) -> float | None:
    """Read one of the writer's size cards as a positive number, or ``None``.

    ``SIZEPIX`` and ``SIZEARC`` are both always written, but the column that did not
    size the source leaves 0 or a FITS UNDEFINED card behind (the catalogue value was
    ``None``). Both mean "this column did not size the source", not "the size is zero",
    so they must not reach the arithmetic below.
    """
    value = header[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _expected_extraction_size(
    diameter_pixel: float | None,
    diameter_arcsec: float | None,
    sizing_wcs: WCS | None,
    padding_factor: float,
) -> int | None:
    """Pre-resize window size in parent pixels, from the catalogue size in the header.

    Mirrors the sizing the extraction path applies: ``diameter_pixel`` wins outright,
    otherwise the arcsec diameter is converted through a tile pixel scale
    (``arcsec_to_pixels``, clamped to one pixel), and the padding factor scales the
    result with the same truncation as ``extract_cutouts_batch_vectorized``.
    Reconstructing the arcsec case matters because a ``diameter_arcsec`` catalogue
    leaves ``SIZEPIX`` at 0, which left the resize cross-check inert for the whole run.

    Args:
        diameter_pixel: ``SIZEPIX`` if the catalogue sized this source in pixels, else None.
        diameter_arcsec: ``SIZEARC`` if the catalogue sized it in arcsec, else None.
        sizing_wcs: WCS of the tile that sized the source (see ``_sizing_wcs``), or None
            when it could not be resolved. Only the arcsec branch needs it.
        padding_factor: ``config.padding_factor`` for the run.

    Returns:
        The window size in parent pixels, or ``None`` when the header records no usable
        catalogue size — the only case that leaves the resize cross-check unexercised.
    """
    if diameter_pixel is not None:
        size_px = int(diameter_pixel)
    elif diameter_arcsec is not None and sizing_wcs is not None:
        size_px = max(1, int(round(diameter_arcsec / get_pixel_scale_arcsec_per_pixel(sizing_wcs))))
    else:
        return None
    size = int(size_px * padding_factor)
    # A window that truncates to nothing cannot have produced this cutout; report it as
    # unrecorded rather than dividing by zero.
    return size if size > 0 else None


def _parent_paths_by_channel(fits_file_paths) -> dict[str, str]:
    """Map channel name -> parent tile path for one catalogue row.

    Raises:
        ValueError: If two tiles resolve to the same channel name. Collapsing them onto
            one key would compare a cutout against the wrong tile. Two tiles of the same
            Euclid band in one row do this, and so do two tiles ``extract_filter_name``
            cannot classify: both are labelled ``UNKNOWN``, because the label is a
            ``channel_weights`` key and has to mean the same thing in every row. Such a
            row genuinely cannot be mapped back to its parents by name, and saying so is
            the point — the alternative is guessing.
    """
    paths: dict[str, str] = {}
    for path in parse_fits_file_paths(fits_file_paths):
        channel = extract_filter_name(path)
        if channel in paths:
            raise ValueError(
                f"Tiles {paths[channel]!r} and {path!r} both resolve to channel "
                f"{channel!r}; cannot tell which one a cutout channel came from"
            )
        paths[channel] = path
    return paths


def _extensions_by_channel(selected_extensions: list) -> dict[str, str]:
    """Map channel name -> the tile extension Cutana read for it."""
    return {ext["name"]: ext["ext"] for ext in selected_extensions}


def _sizing_wcs(tensor_order: list, parent_paths: dict, extensions: dict) -> WCS | None:
    """WCS of the tile the pipeline sized the source from, for the arcsec conversion.

    The pipeline keeps the first tile file's extraction window for every channel, so the
    first column of this row's tensor is the closest observable stand-in. Which file a
    worker reached first is not recoverable from the output; on a set whose channels
    share a grid — every Euclid MER mosaic, and so every release run — the distinction
    does not arise, and on a multi-resolution set the pipeline's own single-window
    behaviour is the thing worth questioning.

    Takes the row's tensor order rather than the run's ``selected_extensions`` order:
    those differ whenever the configuration lists the bands in a different order from
    the catalogue, and it is the row that decides which tile is sized from.

    Returns None when the row's tensor is empty, or when the catalogue names no tile for
    that channel, which skips the resize cross-check; the per-HDU "cannot tell which
    tile" failure reports the cause.
    """
    if not tensor_order:
        return None
    channel = tensor_order[0]
    if channel not in parent_paths:
        return None
    return _get_parent_wcs(parent_paths[channel], extensions[channel])


def _channel_of_hdu(hdu, index: int, tensor_order: list) -> str:
    """Name the input channel whose WCS a cutout image HDU carries.

    Unless the run is extraction-only, the writer names output HDUs ``channel_1``,
    ``channel_2``, … and attaches the WCS of the *n*-th input channel to the *n*-th
    HDU, so position is the only link back to a parent tile. The tensor column order
    is the row's ``fits_file_paths`` order narrowed to the bands the run selected:
    band filtering never loads an unselected tile, so it takes no tensor column.
    ``combine_channels`` resolves ``channel_weights`` onto that order by name, so the
    weight dictionary's own order says nothing about the tensor and must not be used
    here.

    Raises:
        ValueError: If the file holds more image HDUs than the row has selected tiles,
            which would make the positional mapping meaningless.
    """
    name = hdu.header["CHANNEL"]
    if name in tensor_order:
        return name
    if index >= len(tensor_order):
        raise ValueError(
            f"Cutout image HDU {index + 1} ({name!r}) has no counterpart in the row's "
            f"channel order {tensor_order}; the HDU-to-tile mapping is positional"
        )
    return tensor_order[index]


def _check_one_cutout_file(
    fits_path: Path,
    parent_paths: dict[str, str],
    extensions: dict[str, str],
    padding_factor: float,
) -> tuple[list[dict], list[str]]:
    """Check every image HDU of one cutout file. Returns (results, per-HDU failure lines)."""
    results: list[dict] = []
    failures: list[str] = []
    # The row's `fits_file_paths` order, minus the bands the run did not select: band
    # filtering never loads those tiles, so they occupy no tensor column. Keeping them
    # here would shift every HDU after the first gap onto the wrong parent tile.
    tensor_order = [channel for channel in parent_paths if channel in extensions]

    with astropy_fits.open(fits_path, memmap=False) as hdul:
        source_id = str(hdul[0].header["SOURCE"])
        source_ra = float(hdul[0].header["RA"])
        source_dec = float(hdul[0].header["DEC"])
        # Whichever of the two the catalogue used; the other is 0 or UNDEFINED.
        diameter_pixel = _recorded_size(hdul[0].header, "SIZEPIX")
        diameter_arcsec = _recorded_size(hdul[0].header, "SIZEARC")
        # One window per source, not one per channel: the extraction origin and size the
        # writer builds every channel's WCS from come from the first tile file the worker
        # processed (``all_source_offsets`` in ``cutout_process_utils``), and the arcsec
        # conversion uses that file's pixel scale. Converting per channel instead would
        # report a wrong footprint for every non-first channel of a multi-resolution set.
        expected_extraction_size = _expected_extraction_size(
            diameter_pixel,
            diameter_arcsec,
            _sizing_wcs(tensor_order, parent_paths, extensions),
            padding_factor,
        )
        image_hdus = [hdu for hdu in hdul[1:] if hdu.data is not None and hdu.data.ndim == 2]

        for index, hdu in enumerate(image_hdus):
            channel = _channel_of_hdu(hdu, index, tensor_order)
            # `tensor_order` is drawn from both mappings, so this cannot fire as the code
            # stands. It is a backstop: nothing else in this file contains an exception,
            # so a future change that broke the invariant would otherwise abort the whole
            # release run on one file instead of reporting the source that tripped it.
            if channel not in parent_paths or channel not in extensions:
                failures.append(
                    f"{source_id}/HDU {index + 1}: cannot tell which tile this channel "
                    f"was cut from (resolved to {channel!r}; catalogue lists "
                    f"{sorted(parent_paths)}, run selected {sorted(extensions)})"
                )
                continue
            if "CRPIX1" not in hdu.header:
                failures.append(f"{source_id}/{channel}: no WCS written to the cutout")
                continue

            parent_wcs = _get_parent_wcs(parent_paths[channel], extensions[channel])
            result = check_cutout_wcs_against_parent(
                hdu.header,
                hdu.data.shape,
                parent_wcs,
                source_ra,
                source_dec,
                expected_extraction_size,
            )
            results.append(result)
            for reason in result["reasons"]:
                failures.append(f"{source_id}/{channel}: {reason}")

    return results, failures


def verify_cutout_wcs(
    output_dir: Path,
    catalogue_df: pd.DataFrame,
    selected_extensions: list,
    source_ids: set,
    padding_factor: float,
) -> dict:
    """Verify the WCS of the generated FITS cutouts for a set of sampled sources.

    Args:
        output_dir: Cutana output directory to scan for ``*.fits`` cutouts.
        catalogue_df: The catalogue used for the run, for the parent tile paths.
        selected_extensions: ``config.selected_extensions`` for the run, naming the bands
            it loaded and which tile extension each was read from. Its order is
            irrelevant — the tensor follows each row's ``fits_file_paths`` order,
            narrowed to these bands.
        source_ids: Source IDs to check. Every one of them must be found in the output.
        padding_factor: ``config.padding_factor``, for the expected extraction size.

    Returns:
        dict with ``n_sources``, ``n_missing``, ``n_hdus``, ``n_failed``,
        ``n_resize_unchecked``, ``n_failures``, ``max_affine_spread``,
        ``max_origin_residual``, ``max_source_offset``, ``failures`` and ``status``.
    """
    wanted = {str(sid) for sid in source_ids}
    catalogue_paths = {
        str(row["SourceID"]): row["fits_file_paths"]
        for _, row in catalogue_df[catalogue_df["SourceID"].astype(str).isin(wanted)].iterrows()
    }
    extensions = _extensions_by_channel(selected_extensions)

    all_results: list[dict] = []
    failures: list[str] = []
    checked_ids: set[str] = set()

    for fits_path in output_dir.rglob("*.fits"):
        primary_header = astropy_fits.getheader(fits_path, 0)
        if "SOURCE" not in primary_header:
            continue  # not a Cutana cutout
        source_id = str(primary_header["SOURCE"])
        if source_id not in wanted or source_id in checked_ids:
            continue
        checked_ids.add(source_id)
        try:
            results, file_failures = _check_one_cutout_file(
                fits_path,
                _parent_paths_by_channel(catalogue_paths[source_id]),
                extensions,
                padding_factor,
            )
        except (KeyError, ValueError, OSError) as exc:
            # A missing header card, an ambiguous tile name, an unreadable parent tile:
            # all are real failures of this cutout, but a release run has already spent
            # hours producing the output, so one bad file must not abort the check. Record
            # it -- the run still fails -- and carry on through the rest of the sample.
            logger.warning(f"WCS check: {source_id}: {exc}")
            failures.append(f"{source_id}: {exc}")
            continue
        all_results.extend(results)
        failures.extend(file_failures)
        if checked_ids == wanted:
            break

    # A check that silently covered nothing, or only a fraction of the sample, is
    # indistinguishable from a pass in the release report — so it is a failure.
    missing = wanted - checked_ids
    if missing:
        logger.warning(f"WCS check: {len(missing)} sampled sources produced no cutout file")
        failures.append(
            f"{len(missing)}/{len(wanted)} sampled sources have no cutout under "
            f"{output_dir} (e.g. {sorted(missing)[:3]})"
        )
    if not all_results:
        failures.append(f"no cutout WCS could be checked under {output_dir}")

    n_failed = sum(1 for result in all_results if not result["ok"])
    # Not a failure -- a source the catalogue sized in neither column is still worth
    # checking for the affine and origin properties -- but it must be visible, because a
    # run where it is the whole sample has no resize coverage at all.
    n_resize_unchecked = sum(1 for result in all_results if not result["resize_checked"])
    if n_resize_unchecked:
        logger.warning(
            f"WCS check: {n_resize_unchecked}/{len(all_results)} cutout HDUs record no "
            "catalogue size, so their resize was not cross-checked"
        )

    return {
        "n_sources": len(checked_ids),
        "n_missing": len(missing),
        "n_hdus": len(all_results),
        "n_failed": n_failed,
        "n_resize_unchecked": n_resize_unchecked,
        "n_failures": len(failures),
        "max_affine_spread": max((r["affine_spread"] for r in all_results), default=0.0),
        "max_origin_residual": max((r["origin_residual"] for r in all_results), default=0.0),
        "max_source_offset": max((r["source_offset_px"] for r in all_results), default=0.0),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
