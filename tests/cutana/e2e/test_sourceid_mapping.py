#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end SourceID<->cutout mapping test with metadata encoded in pixels.

The concern this test guards against is silent misalignment between the
cutouts emitted by cutana and the per-source metadata emitted next to them.
A naive streaming-vs-direct cross-check is not enough: if both paths share a
symmetric swap bug, their outputs would still agree.

Instead, we stamp each source location in a synthetic tile with a *bitmap
rendering of its SourceID* — "1" gets a pixel drawing of the digit "1",
"42" gets "42", and so on. Any misalignment between cutout pixels and the
metadata emitted with them shows up as the cutout depicting a different
number than ``metadata["source_id"]`` claims.

The same encoded tiles + catalogue are used for both paths independently,
per the review on PR #320. Multiple tiles are used in the catalogue so that
the mapping is exercised across FITS-set boundaries too.

Closes #311.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from cutana import StreamingOrchestrator, get_default_config
from cutana.catalogue_preprocessor import preprocess_catalogue
from cutana.direct_cutout import create_cutouts_direct
from tests.test_data.sourceid_pattern_generator import (
    PATCH_SIZE,
    EncodedSource,
    build_encoded_tiles,
    pattern_mask,
    write_companion_band_tile,
)

# IDs chosen to give distinct patterns across tiles and to include both 1- and
# 2-digit numbers so a 1-pixel horizontal drift would be caught.
_ENCODED_SOURCES: Tuple[EncodedSource, ...] = (
    EncodedSource("1", tile_index=0, centre_x=80, centre_y=80),
    EncodedSource("2", tile_index=0, centre_x=200, centre_y=80),
    EncodedSource("3", tile_index=0, centre_x=320, centre_y=200),
    EncodedSource("42", tile_index=1, centre_x=100, centre_y=200),
    EncodedSource("7", tile_index=1, centre_x=260, centre_y=120),
    EncodedSource("89", tile_index=1, centre_x=260, centre_y=320),
)
# Enforced invariant: every encoded SourceID must render to a unique digit
# pattern. If two sources shared the same ID, a swap between them would still
# pass the pixel-pattern assertion and the test would silently lose its teeth.
assert len({s.source_id for s in _ENCODED_SOURCES}) == len(_ENCODED_SOURCES), (
    "Encoded SourceIDs must be unique so a swap between two sources cannot "
    "produce identical rendered patterns."
)


def _assert_per_tile_digit_uniqueness(sources: Sequence[EncodedSource]) -> None:
    """Per-tile invariant: no two SourceIDs in the same tile share a digit.

    Without this, a partial-pattern swap could leave the shared digit pixels
    intact and slip past the pixel assertion — e.g. ``"23"`` mis-paired with
    a cutout from a standalone ``"3"`` source would still light up the ``"3"``
    pixels. Forcing disjoint per-tile digit sets means any cross-source pixel
    leak shows up as at least one mismatched digit.
    """
    digits_per_tile: dict[int, set[str]] = {}
    for src in sources:
        seen = digits_per_tile.setdefault(src.tile_index, set())
        new_digits = set(src.source_id)
        overlap = seen & new_digits
        if overlap:
            raise AssertionError(
                f"Tile {src.tile_index} has SourceID {src.source_id!r} sharing "
                f"digit(s) {sorted(overlap)} with another source in the same tile. "
                "Pick IDs with disjoint digit sets per tile so a partial-pattern "
                "swap cannot pass the pixel assertion."
            )
        seen |= new_digits


_assert_per_tile_digit_uniqueness(_ENCODED_SOURCES)
_TILE_CENTRES: Tuple[Tuple[float, float], ...] = ((150.0, 2.0), (150.3, 2.2))
_TILE_SHAPE: Tuple[int, int] = (400, 400)

# Bands available to the multi-band parametrization. Order matters: the first
# entry is the band the encoded VIS tiles are already saved as, so additional
# bands are byte-identical companion copies of those tiles. Filter names line
# up with what ``extract_filter_name`` reports for these filenames.
_MULTI_BAND_BANDS: Tuple[str, ...] = ("VIS", "NIR-H", "NIR-J", "NIR-Y")


@pytest.fixture
def encoded_catalogue(request, tmp_path) -> dict:
    """Build encoded VIS tiles + catalogue, optionally with companion bands.

    Parametrize indirectly with ``request.param = n_bands`` (1..4) to add
    ``n_bands - 1`` byte-identical sibling FITS files per VIS tile (same WCS,
    only INSTRUME and filename swapped so ``extract_filter_name`` reports the
    intended band). Each catalogue row's ``fits_file_paths`` then lists all
    ``n_bands`` files for its tile, exercising FITS-set assembly. Without
    indirect parametrization the fixture defaults to ``n_bands=1``, which is
    what the direct-path test uses.
    """
    n_bands: int = getattr(request, "param", 1)
    if not 1 <= n_bands <= len(_MULTI_BAND_BANDS):
        raise ValueError(f"n_bands must be in 1..{len(_MULTI_BAND_BANDS)}, got {n_bands}")
    bands = list(_MULTI_BAND_BANDS[:n_bands])

    tile_paths, rows = build_encoded_tiles(
        tmp_path / "tiles",
        _ENCODED_SOURCES,
        _TILE_CENTRES,
        tile_shape=_TILE_SHAPE,
    )

    if n_bands > 1:
        # For each VIS tile, write n_bands - 1 companion FITS files (byte-
        # identical pixels + WCS, INSTRUME / filename swapped per band) so
        # cutana sees them as distinct bands of the same sky region.
        band_tiles_per_tile_idx: List[List[Path]] = []
        for t_idx, vis_tile in enumerate(tile_paths):
            per_tile: List[Path] = [Path(vis_tile)]
            for band in bands[1:]:
                companion = write_companion_band_tile(
                    Path(vis_tile), band, tmp_path / "multi_band_tiles" / f"tile{t_idx:03d}"
                )
                per_tile.append(companion)
            band_tiles_per_tile_idx.append(per_tile)

        sid_to_tile_idx = {src.source_id: src.tile_index for src in _ENCODED_SOURCES}
        for row in rows:
            t_idx = sid_to_tile_idx[row["SourceID"]]
            row["fits_file_paths"] = str(
                [Path(p).resolve().as_posix() for p in band_tiles_per_tile_idx[t_idx]]
            )

    df = pd.DataFrame(rows)
    cat_path = tmp_path / f"encoded_catalogue_{n_bands}.csv"
    df.to_csv(cat_path, index=False)
    return {
        "catalogue_path": cat_path,
        "catalogue_df": df,
        "patch_size": PATCH_SIZE,
        "tile_paths": tile_paths,
        "n_bands": n_bands,
        "bands": bands,
    }


def _make_config(output_dir: str) -> "object":
    """Config that preserves pixel values so the stamped pattern round-trips unchanged."""
    cfg = get_default_config()
    # target resolution matches the stamped patch size -> no resize distortion
    cfg.target_resolution = PATCH_SIZE
    # "none" keeps the 0/1 pattern intact; any monotone stretch would also
    # survive the threshold comparison we use below, but "none" is the tightest
    # check and needs the least reasoning.
    cfg.normalisation_method = "none"
    cfg.interpolation = "nearest"
    cfg.flux_conserved_resizing = False
    cfg.do_only_cutout_extraction = False
    cfg.apply_flux_conversion = False
    cfg.fits_extensions = ["PRIMARY"]
    cfg.selected_extensions = ["VIS"]
    cfg.channel_weights = {"VIS": [1.0]}
    cfg.data_type = "float32"
    cfg.padding_factor = 1.0
    cfg.output_format = "zarr"
    cfg.output_dir = output_dir
    cfg.log_level = "WARNING"
    cfg.console_log_level = "WARNING"
    cfg.skip_catalogue_validation = True
    cfg.skip_memory_calibration_wait = True
    cfg.max_workers = 1
    cfg.max_workflow_time_seconds = 600
    cfg.process_id = "test_sourceid_encoded"
    return cfg


def _assert_cutout_encodes_source_id(
    cutout: np.ndarray,
    source_id: str,
    patch_size: int,
    bands: Sequence[str] | None = None,
) -> None:
    """Assert every channel in ``cutout`` spells out the digits of ``source_id``.

    Accepts either a 2D cutout ``(H, W)`` or a multi-channel cutout
    ``(H, W, C)`` so the same oracle covers single-band streaming/direct paths
    and multi-band FITS-set assembly. When ``bands`` is provided, failure
    messages identify which band's channel is mis-encoded — useful when one
    source has several sibling-band cutouts.
    """
    arr = np.asarray(cutout)
    if arr.ndim == 2:
        channels: List[Tuple[np.ndarray, str]] = [(arr, "")]
    elif arr.ndim == 3:
        if bands is not None and len(bands) != arr.shape[2]:
            raise AssertionError(
                f"bands={bands!r} (len {len(bands)}) does not match cutout channel "
                f"axis (len {arr.shape[2]})"
            )
        channels = [
            (
                arr[:, :, ch],
                f"Channel {ch}" + (f" ({bands[ch]})" if bands is not None else ""),
            )
            for ch in range(arr.shape[2])
        ]
    else:
        raise AssertionError(f"Unexpected cutout shape: {arr.shape}")

    mask_expected = pattern_mask(source_id, patch_size)
    for img, ctx in channels:
        prefix = f"{ctx}: " if ctx else ""
        assert img.shape == (patch_size, patch_size), (
            f"{prefix}Cutout channel shape {img.shape} != expected {(patch_size, patch_size)}"
        )
        # Threshold at the midpoint between bg (0) and bright (1). Any
        # normalisation that preserves ordering would still pass; with
        # normalisation="none" we get exact 0/1 back, so the threshold is
        # cosmetic but robust.
        mask_cutout = img > 0.5

        if not np.array_equal(mask_cutout, mask_expected):
            n_wrong = int(np.count_nonzero(mask_cutout != mask_expected))
            matching_ids = [
                src.source_id
                for src in _ENCODED_SOURCES
                if np.array_equal(mask_cutout, pattern_mask(src.source_id, patch_size))
            ]
            raise AssertionError(
                f"{prefix}Cutout paired with metadata source_id={source_id!r} does "
                f"NOT contain the digits rendered for {source_id!r}: {n_wrong} pixels "
                f"differ from the expected pattern. The cutout pixels actually "
                f"match source_id(s) = {matching_ids!r}. This means cutana emitted "
                "a cutout/metadata pair where the two entries come from different "
                "sources — a SourceID<->cutout swap."
            )


def _direct_cutouts_by_id(cfg, catalogue_df: pd.DataFrame) -> dict:
    """Run the in-process direct path, return {source_id: cutout_array}."""
    prepared = preprocess_catalogue(catalogue_df.copy())
    results = create_cutouts_direct(prepared, cfg)
    out: dict = {}
    for result in results:
        cutouts = result["cutouts"]
        for i, meta in enumerate(result["metadata"]):
            sid = str(meta["source_id"])
            assert sid not in out, f"Duplicate source_id {sid!r} in direct output"
            out[sid] = np.asarray(cutouts[i])
    return out


def _streaming_cutouts_by_id(cfg) -> Tuple[dict, dict]:
    """Run the StreamingOrchestrator path, return ({sid: cutout}, {sid: metadata})."""
    orch = StreamingOrchestrator(cfg)
    cutouts: dict = {}
    metadata: dict = {}
    try:
        orch.init_streaming(batch_size=3, write_to_disk=False)
        for _ in range(orch.get_batch_count()):
            batch = orch.next_batch()
            assert len(batch["cutouts"]) == len(batch["metadata"]), (
                "Streaming produced a batch where len(cutouts) != len(metadata); "
                "this is the exact shape-level misalignment we are trying to detect."
            )
            for cutout, meta in zip(batch["cutouts"], batch["metadata"]):
                sid = str(meta["source_id"])
                assert sid not in cutouts, f"Duplicate source_id {sid!r} emitted by streaming"
                cutouts[sid] = np.asarray(cutout)
                metadata[sid] = meta
    finally:
        orch.cleanup()
    return cutouts, metadata


@pytest.mark.parametrize("encoded_catalogue", [1, 2, 3, 4], indirect=True)
def test_direct_path_source_id_matches_pixel_content(encoded_catalogue, tmp_path):
    """Direct path: every cutout encodes its ``metadata.source_id``, for
    ``n_bands ∈ {1, 2, 3, 4}`` files per source.

    Mirrors the streaming parametrization: ``n_bands=1`` covers the single-file
    fast path, ``n_bands>1`` exercises FITS-set assembly with sibling bands so
    a per-channel swap inside the direct path surfaces as a pattern mismatch.
    """
    n_bands: int = encoded_catalogue["n_bands"]
    bands: List[str] = encoded_catalogue["bands"]
    df = encoded_catalogue["catalogue_df"]
    patch = encoded_catalogue["patch_size"]
    cfg = _make_config(str(tmp_path / f"out_direct_{n_bands}"))
    cfg.selected_extensions = bands
    # Identity per-band weights -> output channel ``i`` is exactly band ``i``,
    # so each channel can be checked independently against the expected pattern.
    cfg.channel_weights = {
        b: [1.0 if i == j else 0.0 for j in range(n_bands)] for i, b in enumerate(bands)
    }

    cutouts_by_id = _direct_cutouts_by_id(cfg, df)

    expected_ids = {src.source_id for src in _ENCODED_SOURCES}
    assert set(cutouts_by_id) == expected_ids, (
        f"Direct path ({n_bands} bands) emitted SourceIDs differ from input: "
        f"missing={expected_ids - set(cutouts_by_id)}, "
        f"extra={set(cutouts_by_id) - expected_ids}"
    )

    for sid, cutout in cutouts_by_id.items():
        arr = np.asarray(cutout)
        if arr.ndim == 2:
            # Promote single-band (H, W) -> (H, W, 1) so the multi-channel
            # helper sees a consistent layout regardless of n_bands.
            arr = arr[:, :, None]
        assert arr.shape == (patch, patch, n_bands), (
            f"SourceID {sid!r}: expected ({patch},{patch},{n_bands}) cutout, got shape {arr.shape}"
        )
        _assert_cutout_encodes_source_id(arr, sid, patch, bands=bands)


@pytest.mark.parametrize("encoded_catalogue", [1, 2, 3, 4], indirect=True)
def test_streaming_path_dedup_preserves_pixel_mapping(encoded_catalogue, tmp_path):
    """Duplicate SourceIDs get reformatted to SourceID_RA_Dec, and each reformatted
    ID still lands on the cutout whose pixels were stamped at its (RA, Dec), at
    every band count ``n_bands ∈ {1, 2, 3, 4}``.

    The reformatted ID encodes the (RA, Dec) of the source; we use that to look
    up the *original* SourceID that was stamped at that position and verify the
    cutout pixels match its rendering. This catches swaps inside the dedup path,
    including any FITS-set-assembly swap that only manifests when the source has
    sibling-band siblings.
    """
    n_bands: int = encoded_catalogue["n_bands"]
    bands: List[str] = encoded_catalogue["bands"]
    df: pd.DataFrame = encoded_catalogue["catalogue_df"].copy()
    patch = encoded_catalogue["patch_size"]

    # Pick two distinct rows and collapse their SourceIDs, forcing dedup.
    original_map = {
        (round(float(row["RA"]), 10), round(float(row["Dec"]), 10)): str(row["SourceID"])
        for _, row in df.iterrows()
    }
    dup_rows = df.iloc[:2].copy()
    dup_rows["SourceID"] = "DUP"
    other_rows = df.iloc[2:].copy()
    combined_df = pd.concat([dup_rows, other_rows], ignore_index=True)
    assert combined_df["SourceID"].duplicated().any()

    dup_cat_path = tmp_path / f"dup_catalogue_{n_bands}.csv"
    combined_df.to_csv(dup_cat_path, index=False)

    cfg = _make_config(str(tmp_path / f"out_dedup_{n_bands}"))
    cfg.source_catalogue = str(dup_cat_path)
    cfg.selected_extensions = bands
    cfg.channel_weights = {
        b: [1.0 if i == j else 0.0 for j in range(n_bands)] for i, b in enumerate(bands)
    }

    cutouts_by_id, metadata_by_id = _streaming_cutouts_by_id(cfg)

    # Exactly 2 sources should have had their ID reformatted as DUP_ra_dec.
    dedup_ids = [sid for sid in cutouts_by_id if sid.startswith("DUP_")]
    assert len(dedup_ids) == 2, f"Expected 2 reformatted duplicate IDs, got {dedup_ids!r}"

    # No silent data loss: every input row is still represented, and no extras.
    assert len(cutouts_by_id) == len(combined_df)

    for sid in dedup_ids:
        # Reformatted ID is "DUP_<ra>_<dec>". Pull the coordinates back out,
        # find the original SourceID stamped at that position, and assert the
        # cutout pixels match *that* digit rendering.
        _, ra_str, dec_str = sid.rsplit("_", 2)
        embedded_ra = float(ra_str)
        embedded_dec = float(dec_str)
        meta = metadata_by_id[sid]
        assert meta["ra"] == pytest.approx(embedded_ra, abs=1e-9)
        assert meta["dec"] == pytest.approx(embedded_dec, abs=1e-9)

        origin_key = (round(embedded_ra, 10), round(embedded_dec, 10))
        assert origin_key in original_map, (
            f"Reformatted ID {sid!r} embeds coordinates that don't match any "
            f"source in the catalogue (known: {list(original_map)!r})"
        )
        original_sid = original_map[origin_key]
        # Promote single-band (H, W) -> (H, W, 1) so the multi-channel helper
        # sees a consistent layout regardless of n_bands.
        arr = np.asarray(cutouts_by_id[sid])
        if arr.ndim == 2:
            arr = arr[:, :, None]
        assert arr.shape == (patch, patch, n_bands), (
            f"Reformatted ID {sid!r}: expected ({patch},{patch},{n_bands}) cutout, "
            f"got shape {arr.shape}"
        )
        _assert_cutout_encodes_source_id(arr, original_sid, patch, bands=bands)


@pytest.mark.parametrize("encoded_catalogue", [1, 2, 3, 4], indirect=True)
def test_streaming_path_source_id_and_metadata_match(encoded_catalogue, tmp_path):
    """Streaming path: every cutout encodes its ``metadata.source_id`` and carries
    the matching RA/Dec, for ``n_bands ∈ {1, 2, 3, 4}`` files per source.

    n_bands=1 covers the single-file fast path (subsumes the previous single-band
    streaming test); n_bands>1 exercises FITS-set assembly with sibling bands.
    All ``_ENCODED_SOURCES`` run together, so a per-tile, per-source or per-channel
    swap surfaces as a pattern mismatch on at least one of them. The metadata
    RA/Dec check against the input catalogue catches the orthogonal failure mode
    where the right SourceID rides on the wrong astrometry.
    """
    n_bands: int = encoded_catalogue["n_bands"]
    bands: List[str] = encoded_catalogue["bands"]
    cat_path: Path = encoded_catalogue["catalogue_path"]
    df: pd.DataFrame = encoded_catalogue["catalogue_df"]
    patch: int = encoded_catalogue["patch_size"]

    cfg = _make_config(str(tmp_path / f"out_streaming_{n_bands}"))
    cfg.source_catalogue = str(cat_path)
    cfg.selected_extensions = bands
    # Identity per-band weights -> output channel ``i`` is exactly band ``i``,
    # so each channel can be checked independently against the expected pattern.
    cfg.channel_weights = {
        b: [1.0 if i == j else 0.0 for j in range(n_bands)] for i, b in enumerate(bands)
    }

    cutouts_by_id, metadata_by_id = _streaming_cutouts_by_id(cfg)

    expected_ids = {src.source_id for src in _ENCODED_SOURCES}
    assert set(cutouts_by_id) == expected_ids, (
        f"Streaming ({n_bands} bands) emitted SourceIDs differ from input: "
        f"missing={expected_ids - set(cutouts_by_id)}, "
        f"extra={set(cutouts_by_id) - expected_ids}"
    )

    input_by_id = {str(row["SourceID"]): row for _, row in df.iterrows()}
    for sid, meta in metadata_by_id.items():
        expected_row = input_by_id[sid]
        assert meta["ra"] == pytest.approx(float(expected_row["RA"])), (
            f"RA mismatch for {sid}: metadata={meta['ra']} input={expected_row['RA']}"
        )
        assert meta["dec"] == pytest.approx(float(expected_row["Dec"])), (
            f"Dec mismatch for {sid}: metadata={meta['dec']} input={expected_row['Dec']}"
        )

    for sid, cutout in cutouts_by_id.items():
        arr = np.asarray(cutout)
        if arr.ndim == 2:
            # Single-band cutouts may come back as (H, W); promote so the helper
            # sees a consistent (H, W, n_bands) layout for both branches.
            arr = arr[:, :, None]
        assert arr.shape == (patch, patch, n_bands), (
            f"SourceID {sid!r}: expected ({patch},{patch},{n_bands}) cutout, got shape {arr.shape}"
        )
        _assert_cutout_encodes_source_id(arr, sid, patch, bands=bands)
