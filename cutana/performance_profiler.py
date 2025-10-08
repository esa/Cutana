#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
#!/usr/bin/env python
"""
Performance profiler for Cutana cutout processing.

This module provides a lightweight profiler to track runtime of different
processing steps within the cutout creation pipeline. Each cutout process
instance should have one profiler to monitor performance.
"""

import time
import os
from typing import Dict, List, Any, Optional
from loguru import logger


class PerformanceProfiler:
    """
    Lightweight performance profiler for tracking runtime of processing steps.

    Each cutout process should have one instance of this profiler to track:
    - FITS file loading time
    - Cutout extraction time
    - Image processing time
    - Output writing time
    """

    def __init__(self, process_id: Optional[str] = None):
        """
        Initialize performance profiler.

        Args:
            process_id: Optional process identifier, defaults to PID
        """
        self.process_id = process_id or f"cutout_process_{os.getpid()}"
        self.runtime_per_element: Dict[str, List[float]] = {
            "fits_loading": [],
            "cutout_extraction": [],
            "image_processing": [],
            "output_writing": [],
        }
        self._start_times: Dict[str, float] = {}
        self._total_sources = 0
        self._profile_start_time = time.time()

        logger.debug(f"Performance profiler initialized for {self.process_id}")

    def start_timing(self, step: str) -> None:
        """
        Start timing a processing step.

        Args:
            step: Name of the processing step to time
        """
        self._start_times[step] = time.time()
        logger.debug(f"{self.process_id}: Started timing {step}")

    def end_timing(self, step: str) -> float:
        """
        End timing a processing step and record the duration.

        Args:
            step: Name of the processing step to stop timing

        Returns:
            Duration in seconds

        Raises:
            ValueError: If start_timing was not called for this step
        """
        if step not in self._start_times:
            raise ValueError(f"start_timing() was not called for step '{step}'")

        duration = time.time() - self._start_times[step]

        # Initialize step list if it doesn't exist
        if step not in self.runtime_per_element:
            self.runtime_per_element[step] = []

        self.runtime_per_element[step].append(duration)
        del self._start_times[step]

        logger.debug(f"{self.process_id}: {step} took {duration:.4f} seconds")
        return duration

    def record_source_processed(self) -> None:
        """Record that a source has been processed."""
        self._total_sources += 1

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for all timed steps.

        Returns:
            Dictionary containing timing statistics
        """
        stats = {
            "process_id": self.process_id,
            "total_sources": self._total_sources,
            "total_runtime": time.time() - self._profile_start_time,
            "steps": {},
        }

        for step, times in self.runtime_per_element.items():
            if times and len(times) > 0:  # Only include steps that were actually timed
                stats["steps"][step] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "mean_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "time_per_source": sum(times) / max(self._total_sources, 1),
                }

        return stats

    def log_performance_summary(self) -> None:
        """Log a summary of performance statistics."""
        stats = self.get_statistics()
        total_time = stats["total_runtime"]
        total_sources = stats["total_sources"]

        logger.info(f"Performance Summary for {self.process_id}:")
        logger.info(f"  Total runtime: {total_time:.2f} seconds")
        logger.info(f"  Sources processed: {total_sources}")
        if total_time > 0:
            logger.info(f"  Sources per second: {total_sources/total_time:.2f}")
        else:
            logger.info("  Sources per second: N/A (zero runtime)")

        for step, step_stats in stats["steps"].items():
            logger.info(f"  {step}:")
            if total_time > 0:
                logger.info(
                    f"Total time: {step_stats['total_time']:.2f}s ({step_stats['total_time']/total_time*100:.1f}% of total)"
                )
            else:
                logger.info(f"    Total time: {step_stats['total_time']:.2f}s (N/A% of total)")
            logger.info(f"    Mean time per operation: {step_stats['mean_time']*1000:.1f}ms")
            logger.info(f"    Operations: {step_stats['count']}")
            if total_sources > 0:
                logger.info(f"    Time per source: {step_stats['time_per_source']*1000:.1f}ms")

        # Write structured performance data to stderr for benchmark parsing
        import sys
        import json

        structured_stats = {
            "type": "performance_summary",
            "process_id": self.process_id,
            "total_runtime": total_time,
            "total_sources": total_sources,
            "sources_per_second": total_sources / total_time if total_time > 0 else 0,
            "steps": stats["steps"],
            "timestamp": time.time(),
        }
        print(f"PERFORMANCE_DATA: {json.dumps(structured_stats)}", file=sys.stderr, flush=True)

    def get_bottlenecks(self, threshold_percent: float = 20.0) -> List[str]:
        """
        Identify performance bottlenecks.

        Args:
            threshold_percent: Minimum percentage of total time to be considered a bottleneck

        Returns:
            List of step names that are bottlenecks
        """
        stats = self.get_statistics()
        total_time = stats["total_runtime"]
        bottlenecks = []

        for step, step_stats in stats["steps"].items():
            if total_time > 0:
                step_percentage = (step_stats["total_time"] / total_time) * 100
                if step_percentage >= threshold_percent:
                    bottlenecks.append(f"{step} ({step_percentage:.1f}% of total time)")
            else:
                # If total_time is 0, skip percentage calculation
                continue

        return bottlenecks


class ContextProfiler:
    """Context manager for easy timing of code blocks."""

    def __init__(self, profiler: PerformanceProfiler, step: str):
        """
        Initialize context profiler.

        Args:
            profiler: PerformanceProfiler instance
            step: Name of the step to time
        """
        self.profiler = profiler
        self.step = step
        self.duration = None

    def __enter__(self):
        """Start timing when entering context."""
        if self.profiler:
            self.profiler.start_timing(self.step)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing when exiting context."""
        if self.profiler:
            self.duration = self.profiler.end_timing(self.step)


def time_step(profiler: PerformanceProfiler, step: str):
    """
    Decorator for timing function calls.

    Args:
        profiler: PerformanceProfiler instance
        step: Name of the step to time
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with ContextProfiler(profiler, step):
                return func(*args, **kwargs)

        return wrapper

    return decorator
