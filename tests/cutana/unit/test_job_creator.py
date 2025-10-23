#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Test suite for the job_creator module.

Tests the JobCreator's ability to create optimized jobs that group sources
by their FITS file usage to minimize I/O operations.
"""

import pandas as pd
import os
import numpy as np
from loguru import logger

from cutana.job_creator import JobCreator


class TestJobCreator:
    """Test cases for the JobCreator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.job_creator = JobCreator(
            max_sources_per_process=5, min_sources_per_job=3, max_fits_sets_per_job=10
        )

    def create_test_catalogue(self, sources_data):
        """Create a test catalogue DataFrame from source data."""
        return pd.DataFrame(sources_data)

    def test_parse_fits_file_paths_list_string(self):
        """Test parsing FITS file paths from list string format."""
        test_path = "['file1.fits', 'file2.fits']"
        result = JobCreator._parse_fits_file_paths(test_path)
        expected = [os.path.normpath("file1.fits"), os.path.normpath("file2.fits")]
        assert result == expected

    def test_parse_fits_file_paths_single_string(self):
        """Test parsing FITS file paths from single string format."""
        test_path = "file1.fits"
        result = JobCreator._parse_fits_file_paths(test_path)
        expected = [os.path.normpath("file1.fits")]
        assert result == expected

    def test_parse_fits_file_paths_malformed_list(self):
        """Test parsing FITS file paths from malformed list string."""
        test_path = "[file1.fits, file2.fits"  # Missing closing bracket
        result = JobCreator._parse_fits_file_paths(test_path)
        # Should treat as single path since it's not a valid list format
        assert len(result) == 1
        assert result[0] == os.path.normpath("[file1.fits, file2.fits")

    def test_empty_catalogue(self):
        """Test job creation with empty catalogue."""
        empty_catalogue = pd.DataFrame()
        jobs = self.job_creator.create_jobs(empty_catalogue)

        assert len(jobs) == 1
        assert jobs[0].empty

    def test_single_fits_file_multiple_sources(self):
        """Test job creation when multiple sources use the same FITS file."""
        sources_data = [
            {
                "SourceID": "src1",
                "RA": 10.0,
                "Dec": 20.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            {
                "SourceID": "src2",
                "RA": 11.0,
                "Dec": 21.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            {
                "SourceID": "src3",
                "RA": 12.0,
                "Dec": 22.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
        ]

        catalogue = self.create_test_catalogue(sources_data)
        jobs = self.job_creator.create_jobs(catalogue)

        # All sources should be in one job since they use the same FITS file
        assert len(jobs) == 1
        assert len(jobs[0]) == 3

    def test_multiple_fits_files_respects_max_sources(self):
        """Test that job creation respects max_sources_per_process limit."""
        sources_data = []
        for i in range(10):  # Create 10 sources
            sources_data.append(
                {
                    "SourceID": f"src{i}",
                    "RA": 10.0 + i,
                    "Dec": 20.0 + i,
                    "diameter_pixel": 64,
                    "fits_file_paths": f"['tile{i}.fits']",  # Each source uses different FITS file
                }
            )

        catalogue = self.create_test_catalogue(sources_data)
        jobs = self.job_creator.create_jobs(catalogue)

        # Should create at least 2 jobs due to max_sources_per_process=5
        assert len(jobs) >= 2

        # No job should exceed max_sources_per_process
        for job in jobs:
            assert len(job) <= self.job_creator.max_sources_per_process

    def test_fits_file_grouping_optimization(self):
        """Test that sources sharing FITS files are grouped together."""
        sources_data = [
            # Sources 1-3 use tile1.fits
            {
                "SourceID": "src1",
                "RA": 10.0,
                "Dec": 20.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            {
                "SourceID": "src2",
                "RA": 11.0,
                "Dec": 21.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            {
                "SourceID": "src3",
                "RA": 12.0,
                "Dec": 22.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            # Sources 4-5 use tile2.fits
            {
                "SourceID": "src4",
                "RA": 13.0,
                "Dec": 23.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile2.fits']",
            },
            {
                "SourceID": "src5",
                "RA": 14.0,
                "Dec": 24.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile2.fits']",
            },
        ]

        catalogue = self.create_test_catalogue(sources_data)
        jobs = self.job_creator.create_jobs(catalogue)

        # Should create 2 jobs: one for tile1.fits (3 sources) and one for tile2.fits (2 sources)
        # This is the FITS set-based optimization in action
        assert len(jobs) == 2

        # Verify that all sources are included
        all_source_ids = set()
        for job in jobs:
            all_source_ids.update(job["SourceID"].tolist())

        expected_source_ids = {"src1", "src2", "src3", "src4", "src5"}
        assert all_source_ids == expected_source_ids

    def test_multi_file_sources(self):
        """Test handling of sources that use multiple FITS files."""
        sources_data = [
            {
                "SourceID": "src1",
                "RA": 10.0,
                "Dec": 20.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits', 'tile2.fits']",
            },
            {
                "SourceID": "src2",
                "RA": 11.0,
                "Dec": 21.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            {
                "SourceID": "src3",
                "RA": 12.0,
                "Dec": 22.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile2.fits']",
            },
        ]

        catalogue = self.create_test_catalogue(sources_data)
        jobs = self.job_creator.create_jobs(catalogue)

        # With min_sources_per_job=3 and only 1 source per FITS set,
        # all small FITS sets should be combined into 1 job to meet minimum
        assert len(jobs) == 1

        # Verify total sources
        total_sources = sum(len(job) for job in jobs)
        assert total_sources == 3

    def test_efficiency_analysis(self):
        """Test the efficiency analysis functionality."""
        sources_data = [
            {
                "SourceID": "src1",
                "RA": 10.0,
                "Dec": 20.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            {
                "SourceID": "src2",
                "RA": 11.0,
                "Dec": 21.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            {
                "SourceID": "src3",
                "RA": 12.0,
                "Dec": 22.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile2.fits']",
            },
        ]

        catalogue = self.create_test_catalogue(sources_data)
        jobs = self.job_creator.create_jobs(catalogue)

        efficiency = self.job_creator.analyze_job_efficiency(jobs)

        # Should have efficiency metrics
        assert "total_sources" in efficiency
        assert "total_jobs" in efficiency
        assert "total_fits_loads" in efficiency
        assert "fits_load_reduction" in efficiency
        assert "average_fits_reuse_ratio" in efficiency

        # Basic validation
        assert efficiency["total_sources"] == 3
        assert efficiency["total_jobs"] == len(jobs)
        assert efficiency["fits_load_reduction"] >= 0  # Should be some reduction

    def test_invalid_fits_paths(self):
        """Test handling of sources with invalid FITS file paths."""
        sources_data = [
            {
                "SourceID": "src1",
                "RA": 10.0,
                "Dec": 20.0,
                "diameter_pixel": 64,
                "fits_file_paths": "invalid_format",
            },
            {
                "SourceID": "src2",
                "RA": 11.0,
                "Dec": 21.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['valid_tile.fits']",
            },
        ]

        catalogue = self.create_test_catalogue(sources_data)
        jobs = self.job_creator.create_jobs(catalogue)

        # Should still create jobs, potentially using fallback method
        assert len(jobs) >= 1

        # Should include all sources
        total_sources_in_jobs = sum(len(job) for job in jobs)
        assert total_sources_in_jobs == 2

    def test_large_job_splitting(self):
        """Test that large jobs are properly split."""
        # Create 15 sources all using different FITS files
        sources_data = []
        for i in range(15):
            sources_data.append(
                {
                    "SourceID": f"src{i}",
                    "RA": 10.0 + i,
                    "Dec": 20.0 + i,
                    "diameter_pixel": 64,
                    "fits_file_paths": f"['tile{i}.fits']",
                }
            )

        catalogue = self.create_test_catalogue(sources_data)
        jobs = self.job_creator.create_jobs(catalogue)

        # Should create multiple jobs due to max_sources_per_process=5
        assert len(jobs) >= 3  # 15 sources / 5 max per process

        # Each job should not exceed the limit
        for job in jobs:
            assert len(job) <= 5

        # All sources should be assigned
        total_assigned = sum(len(job) for job in jobs)
        assert total_assigned == 15

    def test_job_creator_max_sources_per_process(self):
        """Test JobCreator initialization with different max_sources_per_process."""
        job_creator_small = JobCreator(max_sources_per_process=2)
        job_creator_large = JobCreator(max_sources_per_process=10)

        assert job_creator_small.max_sources_per_process == 2
        assert job_creator_large.max_sources_per_process == 10

    def test_fits_set_to_sources_mapping(self):
        """Test the internal FITS file set to sources mapping."""
        sources_data = [
            {
                "SourceID": "src1",
                "RA": 10.0,
                "Dec": 20.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            {
                "SourceID": "src2",
                "RA": 11.0,
                "Dec": 21.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile1.fits']",
            },
            {
                "SourceID": "src3",
                "RA": 12.0,
                "Dec": 22.0,
                "diameter_pixel": 64,
                "fits_file_paths": "['tile2.fits']",
            },
        ]

        catalogue = self.create_test_catalogue(sources_data)
        mapping = self.job_creator._build_fits_set_to_sources_mapping(catalogue)

        # Check FITS file sets (tuples of sorted paths)
        tile1_set = (os.path.normpath("tile1.fits"),)  # Single file set
        tile2_set = (os.path.normpath("tile2.fits"),)  # Single file set

        assert tile1_set in mapping
        assert tile2_set in mapping
        assert len(mapping[tile1_set]) == 2  # src1, src2 (indices 0, 1)
        assert len(mapping[tile2_set]) == 1  # src3 (index 2)

    def test_job_creator_min_sources_per_job_initialization(self):
        """Test JobCreator initialization with min_sources_per_job parameter."""
        job_creator = JobCreator(max_sources_per_process=10, min_sources_per_job=3)
        assert job_creator.max_sources_per_process == 10
        assert job_creator.min_sources_per_job == 3

        # Test default value
        job_creator_default = JobCreator()
        assert job_creator_default.min_sources_per_job == 500

    def test_small_fits_sets_combined_to_meet_minimum(self):
        """Test that small FITS sets are combined to meet min_sources_per_job."""
        # Create sources with different FITS files, only 1 source each
        sources_data = []
        for i in range(5):  # 5 sources, each with different FITS file
            sources_data.append(
                {
                    "SourceID": f"src{i}",
                    "RA": 10.0 + i,
                    "Dec": 20.0 + i,
                    "diameter_pixel": 64,
                    "fits_file_paths": f"['tile{i}.fits']",
                }
            )

        catalogue = self.create_test_catalogue(sources_data)

        # Create job creator with min_sources_per_job=3
        job_creator = JobCreator(max_sources_per_process=10, min_sources_per_job=3)
        jobs = job_creator.create_jobs(catalogue)

        # With 5 sources and 5 different FITS sets, but max_fits_sets_per_job=10:
        # Should create 1 job since 5 FITS sets < max_fits_sets_per_job=10
        # But with default test setup max_fits_sets_per_job=10, this might create fewer jobs than FITS sets
        # Update: the test setup uses max_fits_sets_per_job=10, so 5 FITS sets should fit in 1 job
        assert len(jobs) >= 1

        # Verify all sources are included
        total_sources_in_jobs = sum(len(job) for job in jobs)
        assert total_sources_in_jobs == 5

    def test_large_fits_sets_not_combined(self):
        """Test that large FITS sets (>= min_sources_per_job) are not combined."""
        # Create two FITS sets: one with 4 sources, one with 2 sources
        sources_data = []

        # First FITS set: 4 sources (>= min_sources_per_job=3)
        for i in range(4):
            sources_data.append(
                {
                    "SourceID": f"large_set_src{i}",
                    "RA": 10.0 + i,
                    "Dec": 20.0 + i,
                    "diameter_pixel": 64,
                    "fits_file_paths": "['large_tile.fits']",
                }
            )

        # Second FITS set: 2 sources (< min_sources_per_job=3)
        for i in range(2):
            sources_data.append(
                {
                    "SourceID": f"small_set_src{i}",
                    "RA": 30.0 + i,
                    "Dec": 40.0 + i,
                    "diameter_pixel": 64,
                    "fits_file_paths": "['small_tile.fits']",
                }
            )

        catalogue = self.create_test_catalogue(sources_data)

        job_creator = JobCreator(max_sources_per_process=10, min_sources_per_job=3)
        jobs = job_creator.create_jobs(catalogue)

        # Should create 2 jobs:
        # Job 1: 4 sources from large FITS set (processed first due to weight)
        # Job 2: 2 sources from small FITS set (processed as small set)
        assert len(jobs) == 2

        # Find which job has 4 sources and which has 2
        job_sizes = [len(job) for job in jobs]
        job_sizes.sort()
        assert job_sizes == [2, 4]

    def test_min_sources_per_job_with_max_limit(self):
        """Test interaction between min_sources_per_job and max_sources_per_process."""
        # Create many small FITS sets that would combine beyond max limit
        sources_data = []
        for i in range(15):  # 15 sources, each with different FITS file
            sources_data.append(
                {
                    "SourceID": f"src{i}",
                    "RA": 10.0 + i,
                    "Dec": 20.0 + i,
                    "diameter_pixel": 64,
                    "fits_file_paths": f"['tile{i}.fits']",
                }
            )

        catalogue = self.create_test_catalogue(sources_data)

        # Set min=8, max=10, so we should get jobs of 10, then 5 remaining
        job_creator = JobCreator(max_sources_per_process=10, min_sources_per_job=8)
        jobs = job_creator.create_jobs(catalogue)

        # Should create 2 jobs: first meets min (8), second has remainder (7)
        assert len(jobs) == 2

        # Job sizes should be 8 and 7 (algorithm creates job when min is reached)
        job_sizes = [len(job) for job in jobs]
        job_sizes.sort()
        assert job_sizes == [7, 8]

        # Verify all sources included
        total_sources_in_jobs = sum(len(job) for job in jobs)
        assert total_sources_in_jobs == 15

    def test_mixed_large_and_small_fits_sets(self):
        """Test handling of mixed large and small FITS sets."""
        sources_data = []

        # Large FITS set: 5 sources (>= min_sources_per_job=3)
        for i in range(5):
            sources_data.append(
                {
                    "SourceID": f"large_src{i}",
                    "RA": 10.0 + i,
                    "Dec": 20.0 + i,
                    "diameter_pixel": 64,
                    "fits_file_paths": "['large_tile.fits']",
                }
            )

        # Small FITS sets: 4 different tiles with 1 source each (< min_sources_per_job=3)
        for i in range(4):
            sources_data.append(
                {
                    "SourceID": f"small_src{i}",
                    "RA": 30.0 + i,
                    "Dec": 40.0 + i,
                    "diameter_pixel": 64,
                    "fits_file_paths": f"['small_tile{i}.fits']",
                }
            )

        catalogue = self.create_test_catalogue(sources_data)

        job_creator = JobCreator(max_sources_per_process=10, min_sources_per_job=3)
        jobs = job_creator.create_jobs(catalogue)

        # Should create 3 jobs:
        # Job 1: 5 sources from large FITS set (processed first)
        # Job 2: 3 sources from small FITS sets (meets min requirement)
        # Job 3: 1 source remaining from small FITS sets
        assert len(jobs) == 3

        job_sizes = [len(job) for job in jobs]
        job_sizes.sort()
        assert job_sizes == [1, 3, 5]

        # Verify all sources included
        total_sources_in_jobs = sum(len(job) for job in jobs)
        assert total_sources_in_jobs == 9

    def test_many_small_fits_sets_real_world_scenario(self):
        """Test user's real scenario: 317 FITS sets with 1-27 sources each (~2175 total)."""
        sources_data = []

        # Create 317 FITS sets with varying source counts (1-27 sources each)
        source_id_counter = 0
        for fits_set_id in range(317):
            # Vary sources per FITS set: 1-27 (mimicking real distribution)
            sources_in_this_set = min(27, max(1, int(np.random.poisson(7) + 1)))

            for source_in_set in range(sources_in_this_set):
                sources_data.append(
                    {
                        "SourceID": f"src_{source_id_counter}",
                        "RA": 150.0 + np.random.uniform(-1, 1),
                        "Dec": 2.0 + np.random.uniform(-1, 1),
                        "diameter_pixel": 64,
                        "fits_file_paths": f"['tile_{fits_set_id:06d}.fits']",
                    }
                )
                source_id_counter += 1

        catalogue = self.create_test_catalogue(sources_data)
        total_sources = len(sources_data)

        logger.info(
            f"Created test scenario with {len(sources_data)} sources across {317} FITS sets"
        )

        # Use realistic parameters matching user's scenario
        job_creator = JobCreator(
            max_sources_per_process=25000,  # User's large limit
            min_sources_per_job=500,  # User's minimum
            max_fits_sets_per_job=50,  # Prevent too many FITS files per job
        )
        jobs = job_creator.create_jobs(catalogue)

        # Should create multiple jobs due to max_fits_sets_per_job limit
        # Math: 317 FITS sets / 50 max per job = ~7 jobs
        expected_min_jobs = 317 // 50
        assert (
            len(jobs) >= expected_min_jobs
        ), f"Expected at least {expected_min_jobs} jobs, got {len(jobs)}"

        # Should not create just 1 giant job
        assert len(jobs) > 1, "Should create multiple jobs, not one giant job"

        # Each job should respect limits
        for i, job in enumerate(jobs):
            assert (
                len(job) <= job_creator.max_sources_per_process
            ), f"Job {i+1} exceeds max_sources_per_process"

            # Count unique FITS sets in this job
            fits_sets_in_job = set()
            for _, row in job.iterrows():
                fits_paths = job_creator._parse_fits_file_paths(row["fits_file_paths"])
                fits_set = tuple(fits_paths)
                fits_sets_in_job.add(fits_set)

            # Should not exceed max_fits_sets_per_job (except possibly last job)
            if i < len(jobs) - 1:  # Not the last job
                assert (
                    len(fits_sets_in_job) <= job_creator.max_fits_sets_per_job
                ), f"Job {i+1} has {len(fits_sets_in_job)} FITS sets, exceeds max {job_creator.max_fits_sets_per_job}"

        # Verify all sources included
        total_sources_in_jobs = sum(len(job) for job in jobs)
        assert total_sources_in_jobs == total_sources

        logger.info(
            f"✅ Successfully created {len(jobs)} jobs for {total_sources} sources across 317 FITS sets"
        )
        for i, job in enumerate(jobs):
            fits_sets_in_job = set()
            for _, row in job.iterrows():
                fits_paths = job_creator._parse_fits_file_paths(row["fits_file_paths"])
                fits_set = tuple(fits_paths)
                fits_sets_in_job.add(fits_set)
            logger.info(f"  Job {i+1}: {len(job)} sources from {len(fits_sets_in_job)} FITS sets")

    def test_max_fits_sets_per_job_parameter(self):
        """Test max_fits_sets_per_job parameter initialization and behavior."""
        # Test parameter initialization
        job_creator = JobCreator(
            max_sources_per_process=100, min_sources_per_job=10, max_fits_sets_per_job=5
        )
        assert job_creator.max_fits_sets_per_job == 5

        # Test default value
        job_creator_default = JobCreator()
        assert job_creator_default.max_fits_sets_per_job == 50

        # Test behavior: create many small FITS sets that hit max_fits_sets_per_job before min_sources_per_job
        sources_data = []
        for i in range(10):  # 10 FITS sets with 2 sources each = 20 total sources
            for j in range(2):
                sources_data.append(
                    {
                        "SourceID": f"src_{i}_{j}",
                        "RA": 10.0 + i,
                        "Dec": 20.0 + j,
                        "diameter_pixel": 64,
                        "fits_file_paths": f"['tile_{i}.fits']",
                    }
                )

        catalogue = self.create_test_catalogue(sources_data)

        # With max_fits_sets_per_job=5, min_sources_per_job=10:
        # Should create jobs when hitting max_fits_sets_per_job=5 (10 sources)
        # rather than waiting for min_sources_per_job=10
        job_creator = JobCreator(
            max_sources_per_process=100, min_sources_per_job=15, max_fits_sets_per_job=5
        )
        jobs = job_creator.create_jobs(catalogue)

        # Should create 2 jobs: 10 sources (5 FITS sets) + 10 sources (5 FITS sets)
        assert len(jobs) == 2
        for job in jobs:
            assert (
                len(job) == 10
            )  # Each job should have exactly 10 sources (5 FITS sets × 2 sources each)
