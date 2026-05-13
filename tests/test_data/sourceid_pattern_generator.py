#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Generate FITS tiles whose pixels visually encode each source's ``SourceID``.

The SourceID mapping test needs a way to catch misalignment that is invisible to
a streaming-vs-direct cross-check: if both paths happened to swap the same two
entries, a symmetric comparison would pass. We instead embed the identifier into
the pixels themselves — each source region in the tile is stamped with a bitmap
rendering of its SourceID (the digits "1", "2", ... shaped out of bright
pixels on a dark background). The test then asserts that the pattern recovered
from each cutout matches the ``metadata["source_id"]`` it was paired with.

The generator writes plain FITS tiles (one or more) plus catalogue rows ready
to be concatenated into a DataFrame / CSV. An optional ``save_previews`` flag
dumps matching PNGs so a human can eyeball the encoded data — useful when
debugging orientation or scale choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

# 3x5 bitmap font for digits 0-9. Each glyph is arranged top-row-first so it
# reads naturally when displayed with ``origin="upper"`` (numpy default).
DIGIT_FONT = {
    "0": np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8),
    "1": np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=np.uint8),
    "2": np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=np.uint8),
    "3": np.array([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=np.uint8),
    "4": np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8),
    "5": np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=np.uint8),
    "6": np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8),
    "7": np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8),
    "8": np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8),
    "9": np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=np.uint8),
}

GLYPH_H = 5
GLYPH_W = 3

# Default patch side length. Must fit the longest rendered digit string with
# the chosen scale. Exposed so tests can pick a ``diameter_pixel`` that matches.
PATCH_SIZE = 48
GLYPH_SCALE = 6  # digits end up 15 px wide, 30 px tall at this scale

BRIGHT = 1.0
BG = 0.0


def _digits_of(source_id: str) -> List[str]:
    # Render only the numeric characters. Non-digit prefixes/suffixes are fine
    # (the SourceID may be "SOURCE_DUP_150.1..._2.0..." after dedup); we pick
    # the last run of digits so the image shows the "interesting" number.
    digits = [c for c in source_id if c in DIGIT_FONT]
    if not digits:
        raise ValueError(f"source_id {source_id!r} has no digit characters to render as a pattern")
    return digits


def render_id_patch(
    source_id: str,
    patch_size: int = PATCH_SIZE,
    glyph_scale: int = GLYPH_SCALE,
) -> np.ndarray:
    """Render ``source_id`` as a centred row of bright digits on a dark patch."""
    digits = _digits_of(source_id)

    glyph_h = GLYPH_H * glyph_scale
    glyph_w = GLYPH_W * glyph_scale
    gap = glyph_scale  # one glyph-pixel of gap between digits
    total_w = len(digits) * glyph_w + (len(digits) - 1) * gap

    if total_w > patch_size or glyph_h > patch_size:
        raise ValueError(
            "Rendered text does not fit in the requested patch: "
            f"text={total_w}x{glyph_h}, patch={patch_size}x{patch_size}"
        )

    patch = np.full((patch_size, patch_size), BG, dtype=np.float32)
    y0 = (patch_size - glyph_h) // 2
    x0 = (patch_size - total_w) // 2

    for i, d in enumerate(digits):
        glyph = DIGIT_FONT[d]
        scaled = np.kron(glyph, np.ones((glyph_scale, glyph_scale), dtype=np.uint8))
        x = x0 + i * (glyph_w + gap)
        patch[y0 : y0 + glyph_h, x : x + glyph_w] = np.where(scaled == 1, BRIGHT, BG)

    return patch


def pattern_mask(
    source_id: str,
    patch_size: int = PATCH_SIZE,
    glyph_scale: int = GLYPH_SCALE,
) -> np.ndarray:
    """Boolean mask of lit pixels, suitable for comparison with a cutout pattern."""
    return render_id_patch(source_id, patch_size, glyph_scale) > (BG + BRIGHT) / 2


@dataclass(frozen=True)
class EncodedSource:
    """One source to stamp into a tile; centre is expressed in tile pixel coords."""

    source_id: str
    tile_index: int  # 0-based index into the tile_centres sequence
    centre_x: int  # column in the tile
    centre_y: int  # row in the tile


def _make_wcs(ra_center: float, dec_center: float, tile_shape: Tuple[int, int]) -> WCS:
    pixel_scale = 0.1 / 3600.0  # Euclid VIS-like 0.1 arcsec/pixel
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_center, dec_center]
    wcs.wcs.crpix = [tile_shape[1] / 2, tile_shape[0] / 2]
    wcs.wcs.cd = [[-pixel_scale, 0], [0, pixel_scale]]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.equinox = 2000.0
    return wcs


def build_encoded_tiles(
    out_dir: Path,
    sources: Sequence[EncodedSource],
    tile_centres: Sequence[Tuple[float, float]],
    tile_shape: Tuple[int, int] = (400, 400),
    patch_size: int = PATCH_SIZE,
    save_previews: bool = False,
) -> Tuple[List[Path], List[dict]]:
    """Create FITS tiles with encoded source patches and matching catalogue rows.

    Args:
        out_dir: Directory where FITS (and optional PNG) files are written.
        sources: Sources to stamp, each keyed to one of ``tile_centres``.
        tile_centres: (RA, Dec) centre per tile, in degrees.
        tile_shape: (height, width) of every tile, in pixels.
        patch_size: Side length of each stamped patch in pixels.
        save_previews: If True, also write a PNG next to each FITS tile.

    Returns:
        Tuple of (tile_paths, catalogue_rows). Each catalogue row is a dict
        with SourceID / RA / Dec / diameter_pixel / fits_file_paths ready to
        be turned into a DataFrame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_pixels = [np.full(tile_shape, BG, dtype=np.float32) for _ in tile_centres]
    rows: List[dict] = []

    # Mirror the bounds arithmetic cutana uses in
    # cutout_extraction.extract_cutouts_vectorized_from_extension so the
    # stamp we put in the tile covers exactly the pixels cutana will extract
    # (avoids 1-pixel shifts from WCS float round-trip at non-CRPIX pixels).
    half_left = patch_size // 2
    half_right = patch_size - half_left

    for src in sources:
        if not 0 <= src.tile_index < len(tile_centres):
            raise ValueError(f"EncodedSource {src!r} references non-existent tile_index")
        tile = tile_pixels[src.tile_index]
        ra_c, dec_c = tile_centres[src.tile_index]
        wcs = _make_wcs(ra_c, dec_c, tile.shape)

        # Go tile-pixel -> (RA, Dec) -> tile-pixel via the same SkyCoord path
        # cutana uses. The second hop may land a few 1e-10 off the original
        # integer pixel, and int(...) truncation then picks a different row.
        # By stamping at cutana's bounds we stay in lock-step.
        ra, dec = wcs.wcs_pix2world([[src.centre_x, src.centre_y]], 0)[0]
        sky = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
        px_actual, py_actual = wcs.world_to_pixel(sky)
        px_actual = float(np.asarray(px_actual))
        py_actual = float(np.asarray(py_actual))

        x_min = int(px_actual - half_left)
        x_max = int(px_actual + half_right)
        y_min = int(py_actual - half_left)
        y_max = int(py_actual + half_right)

        if not (0 <= y_min and y_max <= tile.shape[0]):
            raise ValueError(f"{src!r}: patch falls outside tile vertically")
        if not (0 <= x_min and x_max <= tile.shape[1]):
            raise ValueError(f"{src!r}: patch falls outside tile horizontally")
        if (y_max - y_min, x_max - x_min) != (patch_size, patch_size):
            raise ValueError(
                f"{src!r}: computed stamp bounds "
                f"{(y_max - y_min, x_max - x_min)} != {(patch_size, patch_size)}"
            )

        patch = render_id_patch(src.source_id, patch_size)
        tile[y_min:y_max, x_min:x_max] = patch

        rows.append(
            {
                "SourceID": src.source_id,
                "RA": float(ra),
                "Dec": float(dec),
                "diameter_pixel": patch_size,
                "_tile_index": src.tile_index,
            }
        )

    tile_paths: List[Path] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
    for idx, (tile_data, (ra_c, dec_c)) in enumerate(zip(tile_pixels, tile_centres)):
        # The cutana streaming path filters FITS files by filter-name patterns
        # derived from the filename (``extract_filter_name``). Include "VIS" in
        # the name so ``selected_extensions=["VIS"]`` picks up our synthetic
        # tiles the same way it picks up real Euclid mosaics.
        filename = f"ENC_MER_BGSUB-MOSAIC-VIS_TILE{idx:03d}_{timestamp}.fits"
        path = out_dir / filename
        hdu = fits.PrimaryHDU(tile_data)
        wcs = _make_wcs(ra_c, dec_c, tile_data.shape)
        hdu.header.update(wcs.to_header())
        hdu.header["TELESCOP"] = "EUCLID"
        hdu.header["INSTRUME"] = "VIS"
        hdu.header["BUNIT"] = "electron/s"
        hdu.header["MAGZERO"] = 24.6
        hdu.writeto(path, overwrite=True)
        tile_paths.append(path)
        if save_previews:
            save_preview_png(path.with_suffix(".png"), tile_data)

    # Rewrite each row's fits_file_paths to the resolved tile path for that source.
    for row in rows:
        tile_idx = row.pop("_tile_index")
        row["fits_file_paths"] = str([tile_paths[tile_idx].resolve().as_posix()])

    return tile_paths, rows


def write_companion_band_tile(source_tile_path: Path, band: str, out_dir: Path) -> Path:
    """Write a sibling FITS file with the same pixels and WCS as ``source_tile_path``
    but a filename and INSTRUME header that make ``extract_filter_name`` report
    ``band``. Used to put the same encoded patch at the same sky position in
    multiple "filter" files for multi-band tests.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with fits.open(source_tile_path) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
    header["INSTRUME"] = band
    out_path = out_dir / f"ENC_MER_BGSUB-MOSAIC-{band}_TILE000_companion.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(out_path, overwrite=True)
    return out_path


def save_preview_png(path: Path, pixels: np.ndarray) -> None:
    """Write a grayscale PNG of ``pixels`` for visual inspection."""
    # Lazy import — matplotlib is only needed for the optional PNG preview
    # hook, and keeping it out of the module import keeps test collection fast
    # for the many callers that never save previews.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pixels, cmap="gray", origin="upper", vmin=BG, vmax=BRIGHT)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=80)
    plt.close(fig)
