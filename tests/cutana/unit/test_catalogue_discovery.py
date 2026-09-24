#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Discovery reads bounded samples and rejects inconsistent filter/HDU layouts."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from astropy.io import fits
from astropy.table import Table

from cutana.catalogue_preprocessor import (
    CatalogueValidationError,
    analyse_source_catalogue,
    analyze_fits_file,
)
from cutana.catalogue_sample import read_catalogue_sample
from cutana.validate_config import validate_channel_order_consistency


def make_catalogue(tmp_path, filters=("NIR-H", "VIS"), rows=5):
    paths = []
    for name in filters:
        path = tmp_path / f"tile_{name}.fits"
        fits.PrimaryHDU(np.ones((4, 4))).writeto(path, overwrite=True)
        paths.append(str(path))
    return pd.DataFrame(
        {
            "SourceID": [str(i) for i in range(rows)],
            "RA": [1.0] * rows,
            "Dec": [2.0] * rows,
            "diameter_arcsec": [1.0] * rows,
            "fits_file_paths": [str(paths)] * rows,
        }
    )


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_catalogue_order_and_header_cache(tmp_path, suffix):
    frame = make_catalogue(tmp_path)
    path = tmp_path / f"catalogue.{suffix}"
    if suffix == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path)
    with patch(
        "cutana.catalogue_preprocessor.analyze_fits_file", wraps=analyze_fits_file
    ) as analyze:
        result = analyse_source_catalogue(str(path))
    assert [ext["name"] for ext in result["extensions"]] == ["NIR-H", "VIS"]
    assert analyze.call_count == 2
    assert result["num_sources"] == 5
    assert not result["num_sources_estimated"]
    assert result["sample_analysis_size"] == 5


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_a_row_carrying_different_bands_fails(tmp_path, change):
    """A different *set* of bands is a different tensor width, which cannot be processed.

    Order is not part of this: weights resolve by name and the WCS check reads each row's
    own `fits_file_paths`, so a reordered row is fine -- see the test below.
    """
    frame = make_catalogue(tmp_path)
    paths = [str(tmp_path / "tile_NIR-H.fits"), str(tmp_path / "tile_VIS.fits")]
    if change == "missing":
        paths.pop()
    else:
        paths[0] = str(tmp_path / "unidentified.fits")
    frame.loc[4, "fits_file_paths"] = str(paths)
    path = tmp_path / "catalogue.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(CatalogueValidationError, match="differ from the sampled"):
        analyse_source_catalogue(str(path))


def test_rows_may_list_their_bands_in_any_order(tmp_path):
    """Requiring one order was the constraint this change set out to remove."""
    frame = make_catalogue(tmp_path)
    reversed_paths = [str(tmp_path / "tile_VIS.fits"), str(tmp_path / "tile_NIR-H.fits")]
    frame.loc[4, "fits_file_paths"] = str(reversed_paths)
    path = tmp_path / "catalogue.csv"
    frame.to_csv(path, index=False)

    result = analyse_source_catalogue(str(path))

    assert sorted(ext["name"] for ext in result["extensions"]) == ["NIR-H", "VIS"]


def test_unidentified_filter_is_usable_end_to_end(tmp_path):
    """One unrecognised tile per row names a channel UNKNOWN, and that channel processes.

    Only Euclid MER names resolve to a filter, so hard-failing every unrecognised name
    locked other surveys out of the UI entirely. The label has to be a constant: it is
    one `channel_weights` key for the whole run, and anything read off the filename of
    an unrecognised tile identifies the *tile*, so it differs from row to row and the
    catalogue's rows stop agreeing about their own channels.

    Asserts the processing half too, which is what a discovery-only assertion missed:
    the UI builds `channel_weights = {"UNKNOWN": [...]}` from this, and a single
    unnamed channel pairs positionally rather than by name.
    """
    frame = make_catalogue(tmp_path, filters=("mystery",))
    path = tmp_path / "catalogue.csv"
    frame.to_csv(path, index=False)

    result = analyse_source_catalogue(str(path))
    assert [ext["name"] for ext in result["extensions"]] == ["UNKNOWN"]

    weights = {ext["name"]: [1.0] for ext in result["extensions"]}
    assert validate_channel_order_consistency(["tile_mystery"], weights) == ["UNKNOWN"]


def test_colliding_filters_fail(tmp_path):
    """Two tiles sharing a label cannot be told apart by name-resolved weights or WCS."""
    # Both stems carry the VIS token, so both resolve to the same channel.
    frame = make_catalogue(tmp_path, filters=("VIS_a", "VIS_b"))
    path = tmp_path / "catalogue.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(CatalogueValidationError, match="distinct channels"):
        analyse_source_catalogue(str(path))


def test_hdu_layout_mismatch_fails(tmp_path):
    frame = make_catalogue(tmp_path, filters=("VIS",))
    path2 = tmp_path / "other_VIS.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((4, 4)), name="SCI")]).writeto(path2)
    frame.loc[4, "fits_file_paths"] = str([str(path2)])
    path = tmp_path / "catalogue.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(CatalogueValidationError, match="HDU order/layout"):
        analyse_source_catalogue(str(path))


def test_csv_reads_only_bounded_prefix(tmp_path):
    path = tmp_path / "catalogue.csv"
    pd.DataFrame({"id": range(20000)}).to_csv(path, index=False)
    with patch("cutana.catalogue_sample.pd.read_csv", wraps=pd.read_csv) as read:
        sample, count, estimated, scope = read_catalogue_sample(path)
    assert read.call_args.kwargs == {"nrows": 10000}
    assert len(sample) == 100
    assert sample.id.max() < 10000
    assert count >= 10000 and estimated
    assert "CSV" in scope


def test_parquet_reads_only_bounded_row_group_windows(tmp_path):
    path = tmp_path / "catalogue.parquet"
    pd.DataFrame({"id": range(50000)}).to_parquet(path, row_group_size=5000)
    sample, count, estimated, scope = read_catalogue_sample(path)
    assert len(sample) == 100 and count == 50000 and not estimated
    assert (sample.id % 5000 < 2500).all()
    assert len(set(sample.id // 5000)) <= 4
    pd.testing.assert_frame_equal(sample, read_catalogue_sample(path)[0])


def test_empty_catalogue_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    make_catalogue(tmp_path, rows=0).to_csv(path, index=False)
    with pytest.raises(CatalogueValidationError, match="empty"):
        analyse_source_catalogue(str(path))


def test_fits_table_sample_preserves_text_paths(tmp_path):
    frame = make_catalogue(tmp_path)
    path = tmp_path / "catalogue.fits"
    Table.from_pandas(frame).write(path)
    sample, count, estimated, _ = read_catalogue_sample(path)
    assert count == len(frame) and not estimated
    assert isinstance(sample.iloc[0]["fits_file_paths"], str)
    assert analyse_source_catalogue(str(path))["num_sources"] == count


def test_billion_row_metadata_does_not_expand_the_read_budget():
    catalogue = MagicMock()
    catalogue.metadata.num_rows = 1_000_000_000
    catalogue.num_row_groups = 100_000
    batch = pa.record_batch({"id": list(range(100))})
    catalogue.iter_batches.side_effect = lambda **kwargs: iter([batch])
    with patch("cutana.catalogue_sample.pq.ParquetFile") as parquet:
        parquet.return_value.__enter__.return_value = catalogue
        sample, count, estimated, _ = read_catalogue_sample("billion.parquet")
    assert count == 1_000_000_000 and not estimated and len(sample) == 100
    assert catalogue.iter_batches.call_count == 4
    for call in catalogue.iter_batches.call_args_list:
        assert call.kwargs["batch_size"] == 2500
        assert len(call.kwargs["row_groups"]) == 1
    catalogue.read.assert_not_called()
    catalogue.read_row_group.assert_not_called()
