#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for catalogue_streamer module.

Tests the streaming infrastructure for memory-efficient catalogue loading:
- CatalogueIndex building and FITS set optimization
- CatalogueBatchReader for row-specific reading
- estimate_catalogue_size for size estimation
"""

import pandas as pd
import pytest

from cutana.catalogue_streamer import (
    CatalogueBatchReader,
    CatalogueIndex,
    estimate_catalogue_size,
)


class TestCatalogueIndex:
    """Tests for CatalogueIndex class."""

    @pytest.fixture
    def sample_csv_catalogue(self, tmp_path):
        """Create a sample CSV catalogue for testing."""
        csv_path = tmp_path / "test_catalogue.csv"
        data = {
            "SourceID": ["src1", "src2", "src3", "src4", "src5"],
            "RA": [10.0, 10.1, 10.2, 10.3, 10.4],
            "Dec": [20.0, 20.1, 20.2, 20.3, 20.4],
            "diameter_pixel": [64, 64, 64, 64, 64],
            "fits_file_paths": [
                "['/path/a.fits']",
                "['/path/a.fits']",
                "['/path/b.fits']",
                "['/path/b.fits']",
                "['/path/c.fits']",
            ],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_parquet_catalogue(self, tmp_path):
        """Create a sample Parquet catalogue for testing."""
        parquet_path = tmp_path / "test_catalogue.parquet"
        data = {
            "SourceID": ["src1", "src2", "src3", "src4", "src5", "src6"],
            "RA": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "Dec": [20.0, 20.1, 20.2, 20.3, 20.4, 20.5],
            "diameter_pixel": [64, 64, 64, 64, 64, 64],
            "fits_file_paths": [
                "['/path/a.fits']",
                "['/path/a.fits']",
                "['/path/b.fits']",
                "['/path/b.fits']",
                "['/path/b.fits']",
                "['/path/c.fits']",
            ],
        }
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path, index=False)
        return parquet_path

    def test_build_from_csv(self, sample_csv_catalogue):
        """Test building index from CSV file."""
        index = CatalogueIndex.build_from_path(str(sample_csv_catalogue))

        assert index.row_count == 5
        assert len(index.fits_set_to_row_indices) == 3  # 3 unique FITS sets

    def test_build_from_parquet(self, sample_parquet_catalogue):
        """Test building index from Parquet file."""
        index = CatalogueIndex.build_from_path(str(sample_parquet_catalogue))

        assert index.row_count == 6
        assert len(index.fits_set_to_row_indices) == 3  # 3 unique FITS sets

    def test_fits_set_grouping(self, sample_parquet_catalogue):
        """Test that sources are correctly grouped by FITS set."""
        index = CatalogueIndex.build_from_path(str(sample_parquet_catalogue))

        # Find the FITS set with the most sources (should be /path/b.fits with 3)
        max_set_size = max(len(rows) for rows in index.fits_set_to_row_indices.values())
        assert max_set_size == 3  # /path/b.fits has 3 sources

    def test_get_optimized_batch_ranges(self, sample_parquet_catalogue):
        """Test optimized batch creation."""
        index = CatalogueIndex.build_from_path(str(sample_parquet_catalogue))

        # Get batches with small batch size to force multiple batches
        batches = index.get_optimized_batch_ranges(
            max_sources_per_batch=2,
            min_sources_per_batch=1,
            max_fits_sets_per_batch=10,
        )

        # All sources should be assigned
        total_sources = sum(len(batch) for batch in batches)
        assert total_sources == 6

    def test_get_fits_set_statistics(self, sample_parquet_catalogue):
        """Test FITS set statistics calculation."""
        index = CatalogueIndex.build_from_path(str(sample_parquet_catalogue))
        stats = index.get_fits_set_statistics()

        assert stats["total_sources"] == 6
        assert stats["unique_fits_sets"] == 3
        assert stats["max_sources_per_set"] == 3  # /path/b.fits
        assert stats["min_sources_per_set"] == 1  # /path/c.fits

    def test_unsupported_format_raises(self, tmp_path):
        """Test that unsupported formats raise ValueError."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("not a catalogue")

        with pytest.raises(ValueError, match="Unsupported catalogue format"):
            CatalogueIndex.build_from_path(str(txt_path))

    def test_empty_csv_catalogue(self, tmp_path):
        """Test building index from empty CSV catalogue (header only)."""
        csv_path = tmp_path / "empty_catalogue.csv"
        # Create CSV with header but no data rows
        csv_path.write_text("SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")

        index = CatalogueIndex.build_from_path(str(csv_path))

        assert index.row_count == 0
        assert len(index.fits_set_to_row_indices) == 0
        # get_optimized_batch_ranges should return empty list for empty catalogue
        batches = index.get_optimized_batch_ranges(max_sources_per_batch=100)
        assert batches == []

    def test_empty_parquet_catalogue(self, tmp_path):
        """Test building index from empty Parquet catalogue."""
        parquet_path = tmp_path / "empty_catalogue.parquet"
        # Create empty DataFrame with correct schema
        df = pd.DataFrame(
            {
                "SourceID": pd.Series([], dtype=str),
                "RA": pd.Series([], dtype=float),
                "Dec": pd.Series([], dtype=float),
                "diameter_pixel": pd.Series([], dtype=int),
                "fits_file_paths": pd.Series([], dtype=str),
            }
        )
        df.to_parquet(parquet_path, index=False)

        index = CatalogueIndex.build_from_path(str(parquet_path))

        assert index.row_count == 0
        assert len(index.fits_set_to_row_indices) == 0
        # get_optimized_batch_ranges should return empty list for empty catalogue
        batches = index.get_optimized_batch_ranges(max_sources_per_batch=100)
        assert batches == []


class TestCatalogueBatchReader:
    """Tests for CatalogueBatchReader class."""

    @pytest.fixture
    def sample_parquet_catalogue(self, tmp_path):
        """Create a sample Parquet catalogue for testing."""
        parquet_path = tmp_path / "test_catalogue.parquet"
        data = {
            "SourceID": [f"src{i}" for i in range(10)],
            "RA": [10.0 + i * 0.1 for i in range(10)],
            "Dec": [20.0 + i * 0.1 for i in range(10)],
            "diameter_pixel": [64] * 10,
            "fits_file_paths": [["/path/a.fits"]] * 10,
        }
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path, index=False)
        return parquet_path

    @pytest.fixture
    def sample_csv_catalogue(self, tmp_path):
        """Create a sample CSV catalogue for testing."""
        csv_path = tmp_path / "test_catalogue.csv"
        data = {
            "SourceID": [f"src{i}" for i in range(10)],
            "RA": [10.0 + i * 0.1 for i in range(10)],
            "Dec": [20.0 + i * 0.1 for i in range(10)],
            "diameter_pixel": [64] * 10,
            "fits_file_paths": ["['/path/a.fits']"] * 10,
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_read_rows_parquet(self, sample_parquet_catalogue):
        """Test reading specific rows from Parquet."""
        reader = CatalogueBatchReader(str(sample_parquet_catalogue))

        # Read rows 2, 5, 7
        result = reader.read_rows([2, 5, 7])

        assert len(result) == 3
        assert "src2" in result["SourceID"].values
        assert "src5" in result["SourceID"].values
        assert "src7" in result["SourceID"].values

        reader.close()

    def test_read_rows_csv(self, sample_csv_catalogue):
        """Test reading specific rows from CSV."""
        reader = CatalogueBatchReader(str(sample_csv_catalogue))

        # Read rows 0, 3, 9
        result = reader.read_rows([0, 3, 9])

        assert len(result) == 3
        assert "src0" in result["SourceID"].values
        assert "src3" in result["SourceID"].values
        assert "src9" in result["SourceID"].values

        reader.close()

    def test_read_empty_rows(self, sample_parquet_catalogue):
        """Test reading empty row list returns empty DataFrame."""
        reader = CatalogueBatchReader(str(sample_parquet_catalogue))

        result = reader.read_rows([])
        assert len(result) == 0

        reader.close()

    def test_unsupported_format_raises(self, tmp_path):
        """Test that unsupported formats raise ValueError."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("not a catalogue")

        with pytest.raises(ValueError, match="Unsupported catalogue format"):
            CatalogueBatchReader(str(txt_path))


class TestEstimateCatalogueSize:
    """Tests for estimate_catalogue_size function."""

    def test_estimate_parquet_size(self, tmp_path):
        """Test estimating Parquet catalogue size."""
        parquet_path = tmp_path / "test.parquet"
        data = {"SourceID": [f"src{i}" for i in range(100)]}
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path, index=False)

        estimated = estimate_catalogue_size(str(parquet_path))
        assert estimated == 100

    def test_estimate_csv_size(self, tmp_path):
        """Test estimating CSV catalogue size."""
        csv_path = tmp_path / "test.csv"
        data = {"SourceID": [f"src{i}" for i in range(100)]}
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        estimated = estimate_catalogue_size(str(csv_path))
        # CSV estimation is approximate, should be within reasonable range
        assert 50 < estimated < 200

    def test_unsupported_format_raises(self, tmp_path):
        """Test that unsupported formats raise ValueError."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("not a catalogue")

        with pytest.raises(ValueError, match="Unsupported catalogue format"):
            estimate_catalogue_size(str(txt_path))


class TestIntegration:
    """Integration tests for the streaming infrastructure."""

    def test_full_streaming_workflow(self, tmp_path):
        """Test complete streaming workflow: index -> batch ranges -> read."""
        # Create a larger catalogue
        parquet_path = tmp_path / "large_catalogue.parquet"
        num_sources = 1000

        # Create data with varying FITS sets
        fits_sets = [
            "['/path/a.fits']",
            "['/path/a.fits']",
            "['/path/b.fits']",
            "['/path/b.fits']",
            "['/path/c.fits']",
        ]

        data = {
            "SourceID": [f"src{i}" for i in range(num_sources)],
            "RA": [10.0 + i * 0.001 for i in range(num_sources)],
            "Dec": [20.0 + i * 0.001 for i in range(num_sources)],
            "diameter_pixel": [64] * num_sources,
            "fits_file_paths": [fits_sets[i % len(fits_sets)] for i in range(num_sources)],
        }
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path, index=False)

        # Build index
        index = CatalogueIndex.build_from_path(str(parquet_path))
        assert index.row_count == num_sources

        # Get batch ranges
        batch_ranges = index.get_optimized_batch_ranges(
            max_sources_per_batch=100,
            min_sources_per_batch=50,
        )
        assert len(batch_ranges) > 0

        # Verify all sources are covered
        all_indices = set()
        for batch in batch_ranges:
            all_indices.update(batch)
        assert len(all_indices) == num_sources

        # Read a few batches
        reader = CatalogueBatchReader(str(parquet_path))

        for batch_indices in batch_ranges[:3]:
            batch_df = reader.read_rows(batch_indices)
            assert len(batch_df) == len(batch_indices)

        reader.close()

    def test_true_streaming_batches_read_independently(self, tmp_path):
        """
        Test that batches are read independently (true streaming).

        This verifies that we don't load all batches into memory at once -
        each batch is read on-demand from the batch_reader.
        """
        # Create catalogue with multiple FITS sets to ensure multiple batches
        parquet_path = tmp_path / "streaming_test.parquet"
        num_sources = 500

        # 5 different FITS sets, 100 sources each
        fits_sets = [f"['/path/tile_{i}.fits']" for i in range(5)]

        data = {
            "SourceID": [f"src{i}" for i in range(num_sources)],
            "RA": [10.0 + i * 0.001 for i in range(num_sources)],
            "Dec": [20.0 + i * 0.001 for i in range(num_sources)],
            "diameter_pixel": [64] * num_sources,
            "fits_file_paths": [fits_sets[i // 100] for i in range(num_sources)],
        }
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path, index=False)

        # Build index (lightweight - just row indices per FITS set)
        index = CatalogueIndex.build_from_path(str(parquet_path))
        assert index.row_count == num_sources
        assert len(index.fits_set_to_row_indices) == 5  # 5 unique FITS sets

        # Get batch ranges with small batch size to force multiple batches
        batch_ranges = index.get_optimized_batch_ranges(
            max_sources_per_batch=100,
            min_sources_per_batch=50,
        )
        assert len(batch_ranges) >= 5  # At least one batch per FITS set

        # Create reader
        reader = CatalogueBatchReader(str(parquet_path))

        # Simulate true streaming: read batches one at a time
        # Track that each batch is independent and complete
        all_source_ids = set()
        for i, batch_indices in enumerate(batch_ranges):
            # Read this batch on-demand
            batch_df = reader.read_rows(batch_indices)

            # Verify batch size matches
            assert len(batch_df) == len(batch_indices)

            # Verify no duplicate sources across batches
            batch_ids = set(batch_df["SourceID"].tolist())
            assert (
                len(batch_ids.intersection(all_source_ids)) == 0
            ), "Duplicate sources across batches!"
            all_source_ids.update(batch_ids)

        # Verify all sources were covered
        assert len(all_source_ids) == num_sources

        reader.close()
