#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for FITSDataset class.

Tests the process-level FITS file caching functionality to ensure efficient
loading and memory management across sub-batches.
"""

from unittest.mock import Mock, patch

import pytest
from astropy.io import fits
from astropy.wcs import WCS
from dotmap import DotMap
from loguru import logger

from cutana import fits_dataset
from cutana.catalogue_preprocessor import SELECTABLE_BAND_NAMES, extract_filter_name
from cutana.fits_dataset import (
    FITSDataset,
    get_selected_band_names,
    prepare_fits_sets_and_sources,
    select_fits_set_bands,
)
from cutana.performance_profiler import PerformanceProfiler


class TestFITSDataset:
    """Test suite for FITSDataset functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = DotMap()
        config.fits_extensions = ["PRIMARY", "SCI"]
        return config

    @pytest.fixture
    def mock_profiler(self):
        """Create a mock profiler."""
        return Mock(spec=PerformanceProfiler)

    @pytest.fixture
    def sample_sources(self):
        """Create sample source data for testing."""
        return [
            {
                "SourceID": "source_001",
                "RA": 45.0,
                "Dec": 12.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['/path/to/file1.fits', '/path/to/file2.fits']",
            },
            {
                "SourceID": "source_002",
                "RA": 46.0,
                "Dec": 13.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['/path/to/file1.fits', '/path/to/file2.fits']",  # Same files
            },
            {
                "SourceID": "source_003",
                "RA": 47.0,
                "Dec": 14.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['/path/to/file3.fits']",  # Different file
            },
        ]

    @pytest.fixture
    def mock_hdul_and_wcs(self):
        """Create mock FITS HDU list and WCS."""
        hdul = Mock(spec=fits.HDUList)
        hdul.close = Mock()
        wcs_dict = {"PRIMARY": Mock(spec=WCS)}
        return hdul, wcs_dict

    def test_initialization(self, mock_config, mock_profiler):
        """Test FITSDataset initialization."""
        dataset = FITSDataset(mock_config, mock_profiler)

        assert dataset.config is mock_config
        assert dataset.profiler is mock_profiler
        assert dataset.fits_cache == {}
        assert dataset.fits_set_to_sources == {}

    @patch("cutana.fits_dataset.prepare_fits_sets_and_sources")
    def test_initialize_from_sources(self, mock_prepare, mock_config, sample_sources):
        """Test initialization from source batch."""
        # Mock the prepare function to return expected FITS sets
        mock_fits_sets = {
            ("/path/to/file1.fits", "/path/to/file2.fits"): sample_sources[:2],
            ("/path/to/file3.fits",): [sample_sources[2]],
        }
        mock_prepare.return_value = mock_fits_sets

        dataset = FITSDataset(mock_config)
        dataset.initialize_from_sources(sample_sources)

        mock_prepare.assert_called_once_with(sample_sources)
        assert dataset.fits_set_to_sources == mock_fits_sets

    @patch("cutana.fits_dataset.load_fits_file")
    def test_prepare_sub_batch_loads_missing_files(
        self, mock_load_fits, mock_config, sample_sources, mock_hdul_and_wcs
    ):
        """Test that prepare_sub_batch loads only missing FITS files."""
        # Setup
        dataset = FITSDataset(mock_config)
        dataset.total_sources = len(sample_sources)  # Initialize total_sources
        dataset.fits_set_to_sources = {
            ("/path/to/file1.fits", "/path/to/file2.fits"): sample_sources[:2]
        }

        # Mock FITS loading
        mock_load_fits.return_value = mock_hdul_and_wcs

        # Test with empty cache - should load both files
        sub_batch = sample_sources[:1]
        result = dataset.prepare_sub_batch(sub_batch)

        # Verify files were loaded
        assert mock_load_fits.call_count == 2
        assert "/path/to/file1.fits" in dataset.fits_cache
        assert "/path/to/file2.fits" in dataset.fits_cache
        assert "/path/to/file1.fits" in result
        assert "/path/to/file2.fits" in result

    @patch("cutana.fits_dataset.load_fits_file")
    def test_prepare_sub_batch_reuses_cached_files(
        self, mock_load_fits, mock_config, sample_sources, mock_hdul_and_wcs
    ):
        """Test that prepare_sub_batch reuses already cached files."""
        # Setup
        dataset = FITSDataset(mock_config)
        dataset.total_sources = len(sample_sources)  # Initialize total_sources
        dataset.fits_set_to_sources = {
            ("/path/to/file1.fits", "/path/to/file2.fits"): sample_sources[:2]
        }

        # Pre-populate cache
        dataset.fits_cache["/path/to/file1.fits"] = mock_hdul_and_wcs
        mock_load_fits.return_value = mock_hdul_and_wcs

        # Test with partial cache - should only load missing file
        sub_batch = sample_sources[:1]
        result = dataset.prepare_sub_batch(sub_batch)

        # Verify only one file was loaded (the missing one)
        assert mock_load_fits.call_count == 1
        mock_load_fits.assert_called_with(
            "/path/to/file2.fits",
            mock_config.fits_extensions,
            n_sources=len(sample_sources),
            is_preview=False,
        )
        assert "/path/to/file1.fits" in result
        assert "/path/to/file2.fits" in result

    def test_get_fits_sets_for_sub_batch(self, mock_config, sample_sources):
        """Test identification of FITS sets needed for a sub-batch."""
        dataset = FITSDataset(mock_config)
        dataset.fits_set_to_sources = {
            ("/path/to/file1.fits", "/path/to/file2.fits"): sample_sources[:2],
            ("/path/to/file3.fits",): [sample_sources[2]],
        }

        # Test sub-batch that needs first FITS set
        sub_batch = sample_sources[:1]  # source_001
        needed_sets = dataset._get_fits_sets_for_sub_batch(sub_batch)

        assert len(needed_sets) == 1
        assert ("/path/to/file1.fits", "/path/to/file2.fits") in needed_sets

        # Test sub-batch that needs second FITS set
        sub_batch = [sample_sources[2]]  # source_003
        needed_sets = dataset._get_fits_sets_for_sub_batch(sub_batch)

        assert len(needed_sets) == 1
        assert ("/path/to/file3.fits",) in needed_sets

    def test_free_unused_after_sub_batch(self, mock_config, sample_sources, mock_hdul_and_wcs):
        """Test freeing of FITS files not needed in remaining sub-batches."""
        dataset = FITSDataset(mock_config)
        dataset.fits_set_to_sources = {
            ("/path/to/file1.fits", "/path/to/file2.fits"): sample_sources[:2],
            ("/path/to/file3.fits",): [sample_sources[2]],
        }

        # Pre-populate cache with all files
        dataset.fits_cache["/path/to/file1.fits"] = mock_hdul_and_wcs
        dataset.fits_cache["/path/to/file2.fits"] = mock_hdul_and_wcs
        dataset.fits_cache["/path/to/file3.fits"] = mock_hdul_and_wcs

        # Current sub-batch used file3, remaining sub-batches need files 1&2
        current_sub_batch = [sample_sources[2]]  # Uses file3
        remaining_sub_batches = [sample_sources[:2]]  # Use files 1&2

        dataset.free_unused_after_sub_batch(current_sub_batch, remaining_sub_batches)

        # file3 should be freed, files 1&2 should remain
        assert "/path/to/file3.fits" not in dataset.fits_cache
        assert "/path/to/file1.fits" in dataset.fits_cache
        assert "/path/to/file2.fits" in dataset.fits_cache

        # Verify close was called on freed file
        mock_hdul_and_wcs[0].close.assert_called()

    def test_free_unused_no_remaining_batches(self, mock_config, sample_sources, mock_hdul_and_wcs):
        """Test that no files are freed when no remaining batches."""
        dataset = FITSDataset(mock_config)
        dataset.fits_set_to_sources = {("/path/to/file1.fits",): sample_sources}
        dataset.fits_cache["/path/to/file1.fits"] = mock_hdul_and_wcs

        # No remaining sub-batches
        dataset.free_unused_after_sub_batch(sample_sources, [])

        # File should remain in cache
        assert "/path/to/file1.fits" in dataset.fits_cache
        mock_hdul_and_wcs[0].close.assert_not_called()

    def test_cleanup(self, mock_config, mock_hdul_and_wcs):
        """Test cleanup of all cached files."""
        dataset = FITSDataset(mock_config)

        # Mock multiple files in cache
        hdul1, wcs1 = Mock(spec=fits.HDUList), Mock()
        hdul1.close = Mock()
        hdul2, wcs2 = Mock(spec=fits.HDUList), Mock()
        hdul2.close = Mock()

        dataset.fits_cache["/path/to/file1.fits"] = (hdul1, wcs1)
        dataset.fits_cache["/path/to/file2.fits"] = (hdul2, wcs2)

        dataset.cleanup()

        # Verify all files were closed and cache cleared
        hdul1.close.assert_called_once()
        hdul2.close.assert_called_once()
        assert len(dataset.fits_cache) == 0

    def test_cleanup_empty_cache(self, mock_config):
        """Test cleanup with empty cache does nothing."""
        dataset = FITSDataset(mock_config)

        # Should not raise any errors
        dataset.cleanup()
        assert len(dataset.fits_cache) == 0

    @patch("cutana.fits_dataset.load_fits_file")
    def test_load_missing_fits_files_handles_errors(self, mock_load_fits, mock_config):
        """Test error handling during FITS file loading."""
        dataset = FITSDataset(mock_config)

        # Mock load_fits_file to raise exception
        mock_load_fits.side_effect = RuntimeError("Failed to load FITS file")

        # Should not raise exception but log error
        fits_sets = [("/path/to/bad_file.fits",)]
        dataset._load_missing_fits_files(fits_sets)

        # File should not be in cache
        assert "/path/to/bad_file.fits" not in dataset.fits_cache

    def test_free_fits_file_handles_errors(self, mock_config, mock_hdul_and_wcs):
        """Test error handling during FITS file freeing."""
        dataset = FITSDataset(mock_config)

        # Create mock that raises exception on close
        bad_hdul = Mock(spec=fits.HDUList)
        bad_hdul.close.side_effect = RuntimeError("Close failed")
        dataset.fits_cache["/path/to/bad_file.fits"] = (bad_hdul, {})

        # Should not raise exception but log warning
        dataset._free_fits_file("/path/to/bad_file.fits")

        # File should still be removed from cache despite close error
        assert "/path/to/bad_file.fits" not in dataset.fits_cache

    def test_integration_full_workflow(self, mock_config, sample_sources):
        """Test complete workflow integration."""
        with patch("cutana.fits_dataset.load_fits_file") as mock_load_fits:
            # Mock FITS loading - create separate mocks for each file
            def create_mock_fits(path):
                hdul_mock = Mock(spec=fits.HDUList)
                hdul_mock.close = Mock()
                wcs_mock = {"PRIMARY": Mock(spec=WCS)}
                return (hdul_mock, wcs_mock)

            mock_load_fits.side_effect = lambda path, exts, n_sources=0, is_preview=False: (
                create_mock_fits(path)
            )

            dataset = FITSDataset(mock_config)

            # Initialize with all sources
            dataset.initialize_from_sources(sample_sources)

            # Verify FITS sets were identified
            assert len(dataset.fits_set_to_sources) > 0

            # Process first sub-batch (sources 001, 002 - same FITS files)
            sub_batch_1 = sample_sources[:2]
            fits_data_1 = dataset.prepare_sub_batch(sub_batch_1)

            initial_cache_size = len(dataset.fits_cache)
            assert initial_cache_size >= 2  # Should have at least file1.fits and file2.fits
            assert len(fits_data_1) >= 2

            # Process second sub-batch (source 003 - different FITS file)
            sub_batch_2 = [sample_sources[2]]
            fits_data_2 = dataset.prepare_sub_batch(sub_batch_2)

            # Should have loaded additional file for source 003
            total_cache_size = len(dataset.fits_cache)
            assert total_cache_size >= initial_cache_size

            # Free files not needed after first batch
            dataset.free_unused_after_sub_batch(sub_batch_1, [sub_batch_2])

            # Final cleanup after second batch
            dataset.free_unused_after_sub_batch(sub_batch_2, [])
            dataset.cleanup()

            # All files should be freed
            assert len(dataset.fits_cache) == 0


class TestPrepareFitsSetsAndSources:
    """Test suite for the prepare_fits_sets_and_sources function."""

    @patch("cutana.fits_dataset.parse_fits_file_paths")
    @patch("cutana.fits_dataset.extract_fits_sets")
    def test_grouping_sources_by_fits_sets(self, mock_extract, mock_parse):
        """Test grouping sources by their FITS file sets."""
        # Sample sources
        sources = [
            {"SourceID": "001", "fits_file_paths": "['/a.fits', '/b.fits']"},
            {"SourceID": "002", "fits_file_paths": "['/a.fits', '/b.fits']"},  # Same set
            {"SourceID": "003", "fits_file_paths": "['/c.fits']"},  # Different set
        ]

        # Mock parsing and extraction
        mock_parse.side_effect = [["/a.fits", "/b.fits"], ["/a.fits", "/b.fits"], ["/c.fits"]]
        mock_extract.side_effect = [
            ({("/a.fits", "/b.fits"): None}, None),
            ({("/a.fits", "/b.fits"): None}, None),
            ({("/c.fits",): None}, None),
        ]

        result = prepare_fits_sets_and_sources(sources)

        # Should group sources by their FITS sets
        assert len(result) == 2
        assert ("/a.fits", "/b.fits") in result
        assert ("/c.fits",) in result
        assert len(result[("/a.fits", "/b.fits")]) == 2  # sources 001, 002
        assert len(result[("/c.fits",)]) == 1  # source 003

    @patch("cutana.fits_dataset.parse_fits_file_paths")
    def test_handles_parse_errors(self, mock_parse):
        """Test handling of FITS path parsing errors."""
        sources = [
            {"SourceID": "001", "fits_file_paths": "invalid_format"},
            {"SourceID": "002", "fits_file_paths": "['/valid.fits']"},
        ]

        # First call raises exception, second succeeds
        mock_parse.side_effect = [ValueError("Invalid format"), ["/valid.fits"]]

        # Mock extract_fits_sets for the valid call
        with patch("cutana.fits_dataset.extract_fits_sets") as mock_extract:
            mock_extract.return_value = ({("/valid.fits",): None}, None)

            result = prepare_fits_sets_and_sources(sources)

        # Should only include valid source
        assert len(result) == 1
        assert ("/valid.fits",) in result
        assert len(result[("/valid.fits",)]) == 1

    def test_empty_source_list(self):
        """Test with empty source list."""
        result = prepare_fits_sets_and_sources([])
        assert result == {}

    @patch("cutana.fits_dataset.parse_fits_file_paths")
    def test_sources_with_no_fits_files(self, mock_parse):
        """Test sources that have no FITS files."""
        sources = [{"SourceID": "001", "fits_file_paths": "[]"}]
        mock_parse.return_value = []

        result = prepare_fits_sets_and_sources(sources)

        # Should return empty result for sources with no FITS files
        assert result == {}


class TestBandSelection:
    """Band narrowing shared by FITSDataset and create_cutouts_direct (#420, #426)."""

    # A Euclid set is ordered VIS, NIR-H, NIR-Y, NIR-J. channel_weights is applied
    # positionally, so which files survive AND their order both matter.
    EUCLID_SET = (
        "/d/EUC_MER_BGSUB-MOSAIC-VIS_TILE1-A_2024.fits",
        "/d/EUC_MER_BGSUB-MOSAIC-NIR-H_TILE1-B_2024.fits",
        "/d/EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE1-C_2024.fits",
        "/d/EUC_MER_BGSUB-MOSAIC-NIR-J_TILE1-D_2024.fits",
    )

    @pytest.mark.parametrize(
        "selected, expected",
        [
            (None, None),
            ([], None),
            ([{"name": "PRIMARY", "ext": "PRIMARY"}], None),
            ([{"name": "VIS", "ext": "PRIMARY"}], {"VIS"}),
            (["VIS", "NIR-H"], {"VIS", "NIR-H"}),
        ],
    )
    def test_get_selected_band_names(self, selected, expected):
        """PRIMARY-only and empty selections mean 'no filtering', not 'no bands'."""
        assert get_selected_band_names(DotMap({"selected_extensions": selected})) == expected

    def test_select_bands_keeps_catalogue_order(self):
        """3nisp3 must yield NIR-H, NIR-Y, NIR-J — not the first three files.

        This is the #420 failure: without narrowing, positional weighting paired
        the three NISP weights with VIS, NIR-H, NIR-Y.
        """
        selected = select_fits_set_bands(self.EUCLID_SET, {"NIR-H", "NIR-Y", "NIR-J"})

        assert [extract_filter_name(p) for p in selected] == ["NIR-H", "NIR-Y", "NIR-J"]

    def test_select_bands_single_band(self):
        """1vis1 narrows to exactly the VIS file."""
        assert select_fits_set_bands(self.EUCLID_SET, {"VIS"}) == (self.EUCLID_SET[0],)

    def test_select_bands_without_filter_is_identity(self):
        """No selection leaves the set untouched, so 4visnisp3 is unaffected."""
        assert select_fits_set_bands(self.EUCLID_SET, None) is self.EUCLID_SET

    @pytest.mark.parametrize("selection", [["SCI", "IMAGE"], ["NIR-Q"]])
    def test_hdu_names_and_unknown_names_switch_band_selection_off(self, selection):
        """A selection naming no band ``extract_filter_name`` can produce is not band selection.

        ``SCI``/``IMAGE`` come from the UI's ``analyze_fits_file()`` path; ``NIR-Q`` is
        simply not a band this codebase knows. Both must leave every set alone rather
        than read as "a band that happens to be absent" and raise.
        """
        config = DotMap({"selected_extensions": selection, "fits_extensions": ["PRIMARY"]})

        assert get_selected_band_names(config) is None

    @pytest.fixture(autouse=True)
    def _forget_reported_selections(self):
        """The notice is deduplicated per process, so tests must not inherit each other's.

        Without this the assertions depend on execution order: whichever test reaches a
        given selection first sees the line and the rest see silence.
        """
        fits_dataset._BAND_SELECTION_OFF_REPORTED.clear()
        yield
        fits_dataset._BAND_SELECTION_OFF_REPORTED.clear()

    @staticmethod
    def _captured_info(action):
        """Run `action` with the cutana logger captured, and return the INFO lines."""
        messages = []
        logger.enable("cutana")
        sink_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
        try:
            action()
        finally:
            logger.remove(sink_id)
            logger.disable("cutana")
        return messages

    def test_a_typo_that_narrows_nothing_says_so(self):
        """A misspelt band silently loads every file in every set, and for a one-file set
        nothing downstream notices, so the log line is the only signal the user gets."""
        config = DotMap({"selected_extensions": ["NIRH"], "fits_extensions": ["PRIMARY"]})

        messages = self._captured_info(lambda: get_selected_band_names(config))

        assert any("Band selection off" in m and "NIRH" in m for m in messages)
        assert any("Check the spelling" in m for m in messages)

    def test_the_unknown_case_is_not_described_as_a_typo(self):
        """`UNKNOWN` is what the UI produces for non-Euclid filenames, and is expected.

        It reaches here through `analyse_source_catalogue`, whose extension `name` is
        `extract_filter_name`'s own output -- so this is the common path on non-Euclid
        data, not a mistake, and the notice must not read like one.
        """
        config = DotMap(
            {"selected_extensions": [{"name": "UNKNOWN", "ext": "PrimaryHDU"}]},
        )

        messages = self._captured_info(lambda: get_selected_band_names(config))

        assert any("carry no band this recogniser knows" in m for m in messages)
        assert not any("Check the spelling" in m for m in messages)

    def test_the_notice_is_said_once_per_process_not_once_per_sub_batch(self):
        """`_load_missing_fits_files` runs per sub-batch, per worker.

        A one-off configuration message repeated per sub-batch buries the `Skipping FITS
        set` errors operators are told to search for in the same stream.
        """
        config = DotMap({"selected_extensions": ["NIRH"], "fits_extensions": ["PRIMARY"]})
        dataset = FITSDataset(config)

        def five_sub_batches():
            with patch("cutana.fits_dataset.load_fits_file", return_value=(Mock(), {})):
                for index in range(5):
                    dataset._load_missing_fits_files([(f"/d/tile_{index}.fits",)])

        messages = self._captured_info(five_sub_batches)

        assert len([m for m in messages if "Band selection off" in m]) == 1

    def test_a_euclid_set_under_an_hdu_name_selection_is_left_alone(self):
        """Deciding from the match result instead of the selection would raise here."""
        config = DotMap({"selected_extensions": ["SCI"], "fits_extensions": ["PRIMARY"]})

        band_names = get_selected_band_names(config)

        assert select_fits_set_bands(self.EUCLID_SET, band_names) is self.EUCLID_SET

    def test_select_bands_narrows_a_single_file_set(self):
        """A one-file set is narrowed like any other: it has a band, and it can be wrong."""
        vis_only = (self.EUCLID_SET[0],)

        assert select_fits_set_bands(vis_only, {"VIS"}) == vis_only

    def test_select_bands_raises_when_a_named_band_is_absent(self):
        """A selection that names bands and matches nothing is a broken configuration.

        Loading the set whole instead would hand ``combine_channels`` the wrong
        bands under the requested names.
        """
        vis_and_h = self.EUCLID_SET[:2]

        with pytest.raises(ValueError, match="match none of the bands"):
            select_fits_set_bands(vis_and_h, {"NIR-J"})

    def test_select_bands_raises_for_a_single_file_set_of_the_wrong_band(self):
        """A one-file set is the case nothing downstream reliably catches.

        ``combine_channels`` has to be given ``channel_names`` to catch a band loaded
        under the wrong label, so a row listing only NIR-H under
        ``selected_extensions=["VIS"]`` must not get as far as being weighted as VIS.
        """
        nir_h_only = (self.EUCLID_SET[1],)

        with pytest.raises(ValueError, match="match none of the bands"):
            select_fits_set_bands(nir_h_only, {"VIS"})

    VIS_ONLY_SET = ("/d/EUC_MER_BGSUB-MOSAIC-VIS_TILE2-A_2024.fits",)

    @staticmethod
    def _sources(*fits_sets):
        """One source per FITS set, in the shape ``initialize_from_sources`` parses."""
        return [
            {"SourceID": f"source_{index}", "fits_file_paths": str(list(fits_set))}
            for index, fits_set in enumerate(fits_sets)
        ]

    def _dataset_over(self, selected, *fits_sets):
        """Drive the real entry point: a worker calls this before any sub-batch."""
        config = DotMap({"selected_extensions": selected, "fits_extensions": ["PRIMARY"]})
        dataset = FITSDataset(config)
        dataset.initialize_from_sources(self._sources(*fits_sets))
        return dataset

    def test_a_set_with_no_requested_band_costs_that_set_and_not_the_batch(self):
        """One heterogeneous tile must not cost the sets around it (#425).

        Driven through ``initialize_from_sources`` and one ``prepare_sub_batch``
        per FITS set, because that is what a worker does: sub-batches are built
        one per set, so a call holding two sets never happens in production and a
        test that passes two proves nothing about the skip.
        """
        dataset = self._dataset_over(
            ["NIR-H", "NIR-Y", "NIR-J"], self.EUCLID_SET[1:], self.VIS_ONLY_SET
        )

        with patch("cutana.fits_dataset.load_fits_file", return_value=(Mock(), {})) as load:
            for sub_batch in ([{"SourceID": "source_0"}], [{"SourceID": "source_1"}]):
                dataset.prepare_sub_batch(sub_batch)

        assert [call.args[0] for call in load.call_args_list] == list(self.EUCLID_SET[1:])

    def test_a_selection_matching_nothing_anywhere_fails_instead_of_extracting_nothing(self):
        """A selection no set can satisfy is a configuration error, not heterogeneity.

        Skipping every set would load nothing and finish, which since #425 is a
        result the parent believes. It has to fail before the first sub-batch, or
        it fails after cutouts are already in the zarr. The message names what was
        asked for and what is there, because that is the whole diagnosis.
        """
        with patch("cutana.fits_dataset.load_fits_file", return_value=(Mock(), {})) as load:
            with pytest.raises(ValueError, match="none of the 1 FITS set") as excinfo:
                self._dataset_over(["NIR-J"], self.EUCLID_SET[:2])

        assert "['NIR-J']" in str(excinfo.value)
        assert "['NIR-H', 'VIS']" in str(excinfo.value)
        load.assert_not_called()

    def test_the_whole_batch_check_does_not_fire_on_a_partial_miss(self):
        """One set short of the selected bands is heterogeneity, and must survive init.

        The boundary between the two behaviours: the worker filtered per file
        before band selection could raise, so a set carrying none of the wanted
        bands contributed nothing and the rest of the batch went on.
        """
        dataset = self._dataset_over(["NIR-J"], self.EUCLID_SET, self.VIS_ONLY_SET)

        with patch("cutana.fits_dataset.load_fits_file", return_value=(Mock(), {})) as load:
            for sub_batch in ([{"SourceID": "source_0"}], [{"SourceID": "source_1"}]):
                dataset.prepare_sub_batch(sub_batch)

        assert [call.args[0] for call in load.call_args_list] == [self.EUCLID_SET[3]]

    @pytest.mark.parametrize(
        "path",
        [
            *EUCLID_SET,
            "/d/EUC_MER_BGSUB-MOSAIC-NIR_J_TILE1-E_2024.fits",
            "/d/survey_H_band.fits",
            "/d/survey_Y_band.fits",
            "/d/survey_tile_0_sci.fits",
            "/d/random.fits",
        ],
    )
    def test_selectable_band_names_cannot_drift_from_the_extractor(self, path):
        """Everything the extractor returns is a selectable band, or exactly UNKNOWN.

        That is the invariant "names no known band, so it is not band selection" rests
        on. A Euclid-only fixture would keep passing if the fallback started returning
        something else -- a basename stem, say -- while the set stopped being exhaustive
        over the extractor's range and the rule quietly lost its meaning.
        """
        assert "UNKNOWN" not in SELECTABLE_BAND_NAMES
        name = extract_filter_name(path)
        assert name in SELECTABLE_BAND_NAMES or name == "UNKNOWN"
