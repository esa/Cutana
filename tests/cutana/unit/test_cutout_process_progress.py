#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for cutout_process progress reporting.

Tests the progress reporting mechanism to ensure:
1. Progress is properly reported to JobTracker
2. All sub-batches are processed and included in output
3. No fallback values that hide issues
"""

import tempfile
from unittest.mock import Mock, patch
import numpy as np

from cutana.cutout_process import create_cutouts_batch, _report_stage
from cutana.job_tracker import JobTracker
from cutana.cutout_writer_zarr import prepare_cutouts_for_zarr
from dotmap import DotMap


class TestCutoutProcessProgress:
    """Test progress reporting in cutout_process."""

    def test_progress_reporting_with_sub_batches(self):
        """Test that progress is correctly reported for each sub-batch."""
        # Create test data with 3000 sources (will create 3 sub-batches of 1000)
        source_batch = [
            {
                "SourceID": f"source_{i:04d}",
                "RA": 45.0 + i * 0.001,
                "Dec": 12.0 + i * 0.001,
                "diameter_pixel": 16,
                "fits_file_paths": ["test.fits"],
            }
            for i in range(3000)
        ]

        # Create config
        config = DotMap()
        config.process_id = "test_process_001"
        config.N_batch_cutout_process = 1000  # Sub-batch size
        config.fits_extensions = ["PRIMARY"]
        config.target_resolution = 16
        config.data_type = "float32"
        config.interpolation = "bilinear"
        config.channel_weights = {"PRIMARY": [1.0]}
        config.apply_flux_conversion = False
        config.job_tracker_session_id = "test_session"
        config.output_format = "zarr"  # Use zarr format for incremental writing
        config.output_dir = "/tmp/test_cutouts"  # Required for zarr path generation

        # Mock JobTracker
        mock_job_tracker = Mock(spec=JobTracker)
        mock_job_tracker.update_process_stage.return_value = True
        mock_job_tracker.report_process_progress.return_value = True

        # Mock zarr writing functions for incremental writing
        with patch("cutana.cutout_process.create_process_zarr_archive_initial") as mock_zarr_create:
            with patch("cutana.cutout_process.append_to_zarr_archive") as mock_zarr_append:
                mock_zarr_create.return_value = True
                mock_zarr_append.return_value = True

                # Mock FITS loading and processing
                with patch("cutana.cutout_process.FITSDataset") as mock_fits_dataset:
                    with patch("cutana.cutout_process._process_source_sub_batch") as mock_process:
                        # Mock FITS dataset
                        mock_fits_instance = Mock()
                        mock_fits_dataset.return_value = mock_fits_instance
                        mock_fits_instance.initialize_from_sources.return_value = None
                        mock_fits_instance.prepare_sub_batch.return_value = {}
                        mock_fits_instance.free_unused_after_sub_batch.return_value = None
                        mock_fits_instance.cleanup.return_value = None

                        # Mock processing to return batch results
                        def create_mock_result(sub_batch):
                            n_sources = len(sub_batch)
                            return [
                                {
                                    "cutouts": np.zeros((n_sources, 16, 16, 1), dtype=np.float32),
                                    "metadata": [{"source_id": s["SourceID"]} for s in sub_batch],
                                }
                            ]

                        mock_process.side_effect = lambda sb, *args, **kwargs: create_mock_result(
                            sb
                        )

                        # Run the function
                        results = create_cutouts_batch(source_batch, config, mock_job_tracker)

                        # Verify progress was reported 5 times (initial + once per sub-batch + final report)
                        assert mock_job_tracker.report_process_progress.call_count == 5

                        # Check the progress values reported
                        calls = mock_job_tracker.report_process_progress.call_args_list
                        assert calls[0][0] == ("test_process_001", 0, 3000)  # Initial progress
                        assert calls[1][0] == ("test_process_001", 1000, 3000)  # First sub-batch
                        assert calls[2][0] == ("test_process_001", 2000, 3000)  # Second sub-batch
                        assert calls[3][0] == ("test_process_001", 3000, 3000)  # Third sub-batch
                        assert calls[4][0] == (
                            "test_process_001",
                            3000,
                            3000,
                        )  # Final completion report

                        # With incremental zarr writing, we get one result with "written_incrementally" marker
                        assert len(results) == 1
                        assert results[0]["metadata"][0]["source_id"] == "written_incrementally"

                        # Verify zarr functions were called correctly
                        assert mock_zarr_create.call_count == 1  # Initial zarr creation
                        assert (
                            mock_zarr_append.call_count == 2
                        )  # Two append operations (for sub-batches 2 and 3)

    def test_zarr_writer_handles_single_batch_correctly(self):
        """Test that zarr writer properly processes single batch data."""
        # Create mock batch data (representing one sub-batch)
        n_sources = 1000
        batch_data = {
            "cutouts": np.ones((n_sources, 16, 16, 1), dtype=np.float32),
            "metadata": [{"source_id": f"source{i:04d}"} for i in range(n_sources)],
        }

        # Process with zarr writer
        images_array, metadata = prepare_cutouts_for_zarr(batch_data)

        # Verify the batch is processed correctly
        assert images_array.shape[0] == 1000  # Number of sources
        assert images_array.shape == (1000, 1, 16, 16)  # NCHW format
        assert len(metadata) == 1000

        # Verify the data is correctly formatted
        assert np.all(images_array == 1)  # All values should be 1

        # Verify metadata is correctly formatted
        assert metadata[0]["source_id"] == "source0000"
        assert metadata[999]["source_id"] == "source0999"

    def test_no_fallback_values_in_progress_reporting(self):
        """Test that there are no fallback values masking real issues."""
        config = DotMap()
        config.process_id = "test_process"
        config.job_tracker_session_id = "test_session"

        # Create a JobTracker with mocked writer
        with patch("cutana.job_tracker.ProcessStatusWriter") as mock_writer:
            mock_writer_instance = Mock()
            mock_writer.return_value = mock_writer_instance

            # Simulate missing progress file - report_process_progress should fail
            mock_writer_instance.report_process_progress.return_value = False
            mock_writer_instance.update_process_progress.return_value = False

            job_tracker = JobTracker(progress_dir=tempfile.gettempdir(), session_id="test_session")

            # This should fail without creating fallback values
            result = job_tracker.report_process_progress("test_process", 100)

            # Without total_sources provided and no file, it should fail
            assert result is False

    def test_stage_reporting_without_conditionals(self):
        """Test that stage reporting doesn't have conditionals."""
        mock_job_tracker = Mock(spec=JobTracker)
        mock_job_tracker.update_process_stage.return_value = False

        # This should not raise or have conditionals
        _report_stage("test_process", "TestStage", mock_job_tracker)

        # Verify it was called
        mock_job_tracker.update_process_stage.assert_called_once_with("test_process", "TestStage")
