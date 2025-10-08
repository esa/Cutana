#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
#!/usr/bin/env python
"""
Job creator module for Cutana - optimizes job splitting based on FITS file usage.

This module handles:
- Grouping sources by their FITS file paths to minimize file loading
- Creating balanced jobs that respect max_sources_per_process limits
- Optimizing I/O operations by maximizing FITS file reuse within jobs
"""

import ast
import os
from typing import Dict, List, Any, Set
from collections import defaultdict
import pandas as pd
from loguru import logger


class JobCreator:
    """
    Creates optimized processing jobs by grouping sources that share FITS files.

    This reduces the number of times FITS files need to be loaded by ensuring
    sources that use the same FITS files are processed together.
    """

    def __init__(
        self,
        max_sources_per_process: int = 1000,
        min_sources_per_job: int = 500,
        max_fits_sets_per_job: int = 50,
    ):
        """
        Initialize the job creator.

        Args:
            max_sources_per_process: Maximum number of sources per job
            min_sources_per_job: Minimum number of sources per job (combines FITS sets if needed)
            max_fits_sets_per_job: Maximum number of FITS sets per job (prevents very long jobs)
        """
        self.max_sources_per_process = max_sources_per_process
        self.min_sources_per_job = min_sources_per_job
        self.max_fits_sets_per_job = max_fits_sets_per_job

    @staticmethod
    def _parse_fits_file_paths(fits_paths_str: str) -> List[str]:
        """
        Parse FITS file paths from string representation.

        Args:
            fits_paths_str: String containing FITS file paths (list format or single path)

        Returns:
            List of normalized FITS file paths
        """
        try:
            if isinstance(fits_paths_str, str):
                # Handle different string formats
                if fits_paths_str.startswith("[") and fits_paths_str.endswith("]"):
                    # String representation of list like "['path1', 'path2']"
                    try:
                        fits_paths = ast.literal_eval(fits_paths_str)
                    except (ValueError, SyntaxError):
                        logger.warning("Failed to parse fits_file_paths with ast.literal_eval")
                        # Fallback: try to extract paths manually
                        fits_paths = [
                            path.strip().strip("'\"")
                            for path in fits_paths_str.strip("[]").split(",")
                        ]
                else:
                    # Single path string
                    fits_paths = [fits_paths_str]
            else:
                fits_paths = fits_paths_str

            # Normalize paths to handle Windows path separators properly
            normalized_paths = [os.path.normpath(path) for path in fits_paths]
            return normalized_paths

        except Exception as e:
            logger.error(f"Error parsing FITS paths '{fits_paths_str}': {e}")
            return []

    def _build_fits_set_to_sources_mapping(
        self, catalogue_data: pd.DataFrame
    ) -> Dict[tuple, List[int]]:
        """
        Build mapping from FITS file sets to source indices that use them.

        Groups sources by their complete set of FITS files for optimal multi-channel processing.

        Args:
            catalogue_data: Source catalogue DataFrame

        Returns:
            Dictionary mapping FITS file set signatures to lists of source indices
        """
        fits_set_to_sources = defaultdict(list)

        for idx, row in catalogue_data.iterrows():
            source_id = row["SourceID"]
            try:
                fits_paths = self._parse_fits_file_paths(row["fits_file_paths"])

                # Create a signature for this FITS file set (sorted tuple for consistency)
                fits_set_signature = tuple(fits_paths)

                # Group sources by their FITS file set signature
                fits_set_to_sources[fits_set_signature].append(idx)

            except Exception as e:
                logger.error(f"Error processing source {source_id}: {e}")
                continue

        return dict(fits_set_to_sources)

    def _calculate_fits_set_weights(
        self, fits_set_to_sources: Dict[tuple, List[int]]
    ) -> Dict[tuple, float]:
        """
        Calculate weights for FITS file sets based on how many sources use them.

        Args:
            fits_set_to_sources: Mapping from FITS file sets to source indices

        Returns:
            Dictionary of FITS file set weights (higher = more sources)
        """
        weights = {}
        for fits_set, source_indices in fits_set_to_sources.items():
            weights[fits_set] = len(source_indices)
        return weights

    def _greedy_job_creation(
        self, catalogue_data: pd.DataFrame, fits_set_to_sources: Dict[tuple, List[int]]
    ) -> List[List[int]]:
        """
        Create jobs using a greedy algorithm that maximizes FITS file set reuse.

        Groups sources by their complete FITS file sets for optimal multi-channel processing.
        Combines small FITS sets to meet min_sources_per_job requirement.

        Args:
            catalogue_data: Source catalogue DataFrame
            fits_set_to_sources: Mapping from FITS file sets to source indices

        Returns:
            List of jobs, where each job is a list of source indices
        """
        # Track which sources have been assigned to jobs
        assigned_sources: Set[int] = set()
        jobs: List[List[int]] = []

        # Calculate FITS set weights (prioritize sets with more sources)
        fits_set_weights = self._calculate_fits_set_weights(fits_set_to_sources)

        # Sort FITS sets by weight (descending) to process high-value sets first
        sorted_fits_sets = sorted(
            fits_set_weights.keys(), key=lambda s: fits_set_weights[s], reverse=True
        )

        # Separate large and small FITS sets
        large_fits_sets = []
        small_fits_sets = []

        for fits_set in sorted_fits_sets:
            source_count = fits_set_weights[fits_set]
            if source_count >= self.min_sources_per_job:
                large_fits_sets.append(fits_set)
            else:
                small_fits_sets.append(fits_set)

        # First, process large FITS sets (>= min_sources_per_job) as before
        for fits_set in large_fits_sets:
            available_sources = [
                idx for idx in fits_set_to_sources[fits_set] if idx not in assigned_sources
            ]

            if not available_sources:
                continue

            # Group available sources into jobs of max_sources_per_process
            while available_sources:
                job_sources = available_sources[: self.max_sources_per_process]
                available_sources = available_sources[self.max_sources_per_process :]

                assigned_sources.update(job_sources)
                jobs.append(job_sources)

                fits_set_description = f"{len(fits_set)} FITS files"
                if len(fits_set) <= 3:
                    fits_set_description = ", ".join(os.path.basename(f) for f in fits_set)

                logger.debug(
                    f"Created job with {len(job_sources)} sources using large FITS set: {fits_set_description}"
                )

        # Then, combine small FITS sets to meet min_sources_per_job
        current_job_sources = []
        current_job_fits_sets_count = 0

        for fits_set in small_fits_sets:
            available_sources = [
                idx for idx in fits_set_to_sources[fits_set] if idx not in assigned_sources
            ]

            if not available_sources:
                continue

            # Add sources from this FITS set to current job
            current_job_sources.extend(available_sources)
            current_job_fits_sets_count += 1

            # Check if we should create a job
            should_create_job = False
            reason = ""

            if len(current_job_sources) >= self.max_sources_per_process:
                should_create_job = True
                reason = "hit max sources limit"
            elif current_job_fits_sets_count >= self.max_fits_sets_per_job:
                should_create_job = True
                reason = "hit max FITS sets limit"
            elif len(current_job_sources) >= self.min_sources_per_job:
                should_create_job = True
                reason = "met minimum sources"

            if should_create_job:
                # Take exactly max_sources_per_process sources if we exceeded
                if len(current_job_sources) > self.max_sources_per_process:
                    job_sources = current_job_sources[: self.max_sources_per_process]
                    current_job_sources = current_job_sources[self.max_sources_per_process :]
                    # Since we're splitting, the created job has all FITS sets so far
                    fits_sets_in_created_job = current_job_fits_sets_count
                    current_job_fits_sets_count = 1 if current_job_sources else 0
                else:
                    # Take all current sources
                    job_sources = current_job_sources
                    fits_sets_in_created_job = current_job_fits_sets_count
                    current_job_sources = []
                    current_job_fits_sets_count = 0

                assigned_sources.update(job_sources)
                jobs.append(job_sources)

                logger.debug(
                    f"Created combined job with {len(job_sources)} sources from {fits_sets_in_created_job} FITS sets ({reason})"
                )

        # Handle any remaining sources from small FITS sets
        if current_job_sources:
            assigned_sources.update(current_job_sources)
            jobs.append(current_job_sources)

            logger.debug(
                f"Created final combined job with {len(current_job_sources)} sources from {current_job_fits_sets_count} FITS sets (remaining sources)"
            )

        # Handle any remaining unassigned sources (shouldn't happen with correct algorithm)
        all_indices = set(range(len(catalogue_data)))
        remaining_sources = list(all_indices - assigned_sources)

        if remaining_sources:
            logger.warning(
                f"Found {len(remaining_sources)} unassigned sources, creating additional jobs"
            )

            # Create additional jobs for remaining sources
            while remaining_sources:
                job_sources = remaining_sources[: self.max_sources_per_process]
                remaining_sources = remaining_sources[self.max_sources_per_process :]
                jobs.append(job_sources)

        return jobs

    def create_jobs(self, catalogue_data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Create optimized processing jobs from catalogue data grouped by FITS file sets.

        Args:
            catalogue_data: Source catalogue DataFrame

        Returns:
            List of DataFrames, each representing a job batch
        """
        if catalogue_data.empty:
            logger.warning("Empty catalogue data provided")
            return [pd.DataFrame()]

        total_sources = len(catalogue_data)
        logger.info(
            f"Creating jobs for {total_sources} sources with max {self.max_sources_per_process} sources per job"
        )

        # Build FITS file set to sources mapping
        fits_set_to_sources = self._build_fits_set_to_sources_mapping(catalogue_data)

        if not fits_set_to_sources:
            logger.error("No valid FITS file set mappings found")
            # Fallback: create simple batches
            return self._create_simple_batches(catalogue_data)

        logger.info(f"Found {len(fits_set_to_sources)} unique FITS file sets across all sources")

        # Log FITS file set usage statistics
        fits_set_usage_stats = {
            fits_set: len(source_list) for fits_set, source_list in fits_set_to_sources.items()
        }

        # Show top 5 most used FITS file sets
        sorted_usage = sorted(fits_set_usage_stats.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top FITS file sets by source count:")
        for fits_set, count in sorted_usage[:5]:
            if len(fits_set) <= 3:
                set_description = ", ".join(os.path.basename(f) for f in fits_set)
            else:
                set_description = f"{len(fits_set)} FITS files"
            logger.info(f"  [{set_description}]: {count} sources")

        # Create jobs using greedy algorithm
        job_indices = self._greedy_job_creation(catalogue_data, fits_set_to_sources)

        # Convert job indices to DataFrames
        job_dataframes = []
        for i, indices in enumerate(job_indices):
            job_df = catalogue_data.iloc[indices].copy()
            job_dataframes.append(job_df)

            # Log job statistics - determine the FITS set for this job
            job_fits_sets = set()
            for _, row in job_df.iterrows():
                fits_paths = self._parse_fits_file_paths(row["fits_file_paths"])
                fits_set = tuple(fits_paths)
                job_fits_sets.add(fits_set)

            if len(job_fits_sets) == 1:
                fits_set = list(job_fits_sets)[0]
                total_fits_files = len(fits_set)
                logger.info(
                    f"Job {i+1}: {len(job_df)} sources using {total_fits_files} FITS files (single set)"
                )
            else:
                total_fits_files = sum(len(fits_set) for fits_set in job_fits_sets)
                logger.info(
                    f"Job {i+1}: {len(job_df)} sources using {len(job_fits_sets)} different FITS"
                    f"sets ({total_fits_files} total files)"
                )

        logger.info(f"Created {len(job_dataframes)} optimized jobs for {total_sources} sources")

        # Validate that all sources are included
        total_assigned = sum(len(job_df) for job_df in job_dataframes)
        if total_assigned != total_sources:
            logger.error(
                f"Job creation error: assigned {total_assigned} sources but had {total_sources} input sources"
            )

        return job_dataframes

    def _create_simple_batches(self, catalogue_data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Fallback method to create simple batches when FITS optimization fails.

        Args:
            catalogue_data: Source catalogue DataFrame

        Returns:
            List of DataFrames with simple batch splitting
        """
        logger.warning("Falling back to simple batch creation")

        batches = []
        total_sources = len(catalogue_data)

        for start_idx in range(0, total_sources, self.max_sources_per_process):
            end_idx = min(start_idx + self.max_sources_per_process, total_sources)
            batch = catalogue_data.iloc[start_idx:end_idx].copy()
            batches.append(batch)

        logger.info(f"Created {len(batches)} simple batches")
        return batches

    def analyze_job_efficiency(self, jobs: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze the efficiency of created jobs based on FITS file set optimization.

        Args:
            jobs: List of job DataFrames

        Returns:
            Dictionary with efficiency statistics
        """
        if not jobs:
            return {"error": "No jobs to analyze"}

        total_sources = sum(len(job) for job in jobs)
        total_fits_loads = 0
        fits_set_reuse_stats = []
        unique_fits_sets = set()

        for job in jobs:
            job_fits_sets = set()
            job_total_files = set()

            for _, row in job.iterrows():
                fits_paths = self._parse_fits_file_paths(row["fits_file_paths"])
                fits_set = tuple(fits_paths)
                job_fits_sets.add(fits_set)
                job_total_files.update(fits_paths)
                unique_fits_sets.add(fits_set)

            total_fits_loads += len(job_total_files)

            # Calculate reuse ratio for this job (sources per unique FITS file)
            if job_total_files:
                file_reuse_ratio = len(job) / len(job_total_files)
                fits_set_reuse_stats.append(file_reuse_ratio)

        # Calculate what naive processing would have cost (each source loads its own files)
        naive_total_fits_loads = 0
        for job in jobs:
            for _, row in job.iterrows():
                fits_paths = self._parse_fits_file_paths(row["fits_file_paths"])
                naive_total_fits_loads += len(fits_paths)

        efficiency = {
            "total_sources": total_sources,
            "total_jobs": len(jobs),
            "unique_fits_sets": len(unique_fits_sets),
            "total_fits_loads": total_fits_loads,
            "naive_fits_loads": naive_total_fits_loads,
            "fits_load_reduction": (
                (naive_total_fits_loads - total_fits_loads) / naive_total_fits_loads * 100
                if naive_total_fits_loads > 0
                else 0
            ),
            "average_sources_per_job": total_sources / len(jobs),
            "average_fits_reuse_ratio": (
                sum(fits_set_reuse_stats) / len(fits_set_reuse_stats) if fits_set_reuse_stats else 0
            ),
            "max_fits_reuse_ratio": max(fits_set_reuse_stats) if fits_set_reuse_stats else 0,
        }

        return efficiency
