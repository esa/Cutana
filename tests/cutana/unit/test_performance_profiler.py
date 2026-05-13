#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the performance_profiler module.

Tests cover:
- Timing of code blocks using ContextProfiler
- Statistics generation
- Bottleneck identification
"""

import re

import pytest

from cutana.performance_profiler import ContextProfiler, PerformanceProfiler


class TestPerformanceProfiler:
    """Test suite for the PerformanceProfiler functions"""

    def test_multiple_steps_and_sources(self):
        """Test that multiple steps and sources are correctly recorded in statistics"""
        profiler = PerformanceProfiler()
        num_sources = 10

        # Simulate processing of 3 steps for num_sources
        for i in range(num_sources):
            with ContextProfiler(profiler=profiler, step="step_1"):
                pass
            with ContextProfiler(profiler=profiler, step="step_2"):
                pass
            with ContextProfiler(profiler=profiler, step="step_3"):
                pass
            profiler.record_source_processed()

        statistics = profiler.get_statistics()

        assert statistics["total_sources"] == num_sources
        assert statistics["total_runtime"] > 0

        for step in ["step_1", "step_2", "step_3"]:
            step_stats = statistics["steps"][step]
            assert step_stats["count"] == num_sources
            assert step_stats["total_time"] > 0
            assert step_stats["mean_time"] > 0
            assert step_stats["std_dev"] > 0
            assert step_stats["min_time"] > 0
            assert step_stats["perc_25"] > 0
            assert step_stats["median_time"] > 0
            assert step_stats["perc_75"] > 0
            assert step_stats["perc_95"] > 0
            assert step_stats["max_time"] > 0
            assert step_stats["time_per_source"] > 0

        # Check that steps for which no timing was recorded are not included in statistics
        with pytest.raises(ValueError, match=re.escape("start_timing() was not called for step")):
            profiler.end_timing("step_4")

    def test_identify_bottlenecks(self):
        """Test that bottlnecks are correctly identified"""

        times = iter([0.0, 0.0, 0.2, 0.2, 1.0, 1.0])

        def mock_time():
            return next(times)

        profiler = PerformanceProfiler(timing_function=mock_time)

        with ContextProfiler(profiler=profiler, step="step_1"):
            pass
        with ContextProfiler(profiler=profiler, step="step_2"):
            pass

        bottleneck = profiler.get_bottlenecks(threshold_percent=30.0)[0]
        assert "step_2" in bottleneck
