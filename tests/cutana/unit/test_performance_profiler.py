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

from cutana import performance_profiler as perf_mod
from cutana.performance_profiler import (
    ContextProfiler,
    PerformanceProfiler,
    _read_process_io_bytes,
)


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

        assert statistics["total_sources_processed"] == num_sources
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


class TestCpuStallSplit:
    """Test the lazy-safe CPU-vs-stall attribution added to the profiler.

    Wall and CPU clocks are mocked so the split is deterministic; the previous
    sleep/busy-loop tests were timing-dependent and could flip on a busy machine
    (a sleeping step may be scheduled out, a busy step preempted).
    """

    def _profiler_with_clocks(self, monkeypatch, wall_values, cpu_values):
        """Build a profiler whose wall + CPU clocks return fixed sequences.

        wall is consumed at __init__, start_timing, end_timing and get_statistics;
        CPU (``time.process_time``) at start_timing and end_timing.
        """
        wall = iter(wall_values)
        cpu = iter(cpu_values)
        monkeypatch.setattr(perf_mod.time, "process_time", lambda: next(cpu))
        return PerformanceProfiler(timing_function=lambda: next(wall))

    def test_stall_attributed_when_no_cpu(self, monkeypatch):
        """A step that burns no CPU has all its wall time counted as stall."""
        # wall: init, start, end, total_runtime ; cpu: start, end
        profiler = self._profiler_with_clocks(monkeypatch, [0.0, 0.0, 1.0, 1.0], [0.0, 0.0])
        with ContextProfiler(profiler=profiler, step="stall_step"):
            pass

        step_stats = profiler.get_statistics()["steps"]["stall_step"]
        assert step_stats["cpu_time"] == pytest.approx(0.0)
        assert step_stats["stall_time"] == pytest.approx(1.0)

    def test_cpu_attributed_when_fully_busy(self, monkeypatch):
        """A step whose CPU time equals its wall time has zero stall."""
        profiler = self._profiler_with_clocks(monkeypatch, [0.0, 0.0, 1.0, 1.0], [0.0, 1.0])
        with ContextProfiler(profiler=profiler, step="busy_step"):
            pass

        step_stats = profiler.get_statistics()["steps"]["busy_step"]
        assert step_stats["cpu_time"] == pytest.approx(1.0)
        assert step_stats["stall_time"] == pytest.approx(0.0)

    def test_stall_never_negative_when_cpu_exceeds_wall(self, monkeypatch):
        """CPU > wall (multithreaded work / clock skew) must clamp stall at 0, not go negative."""
        profiler = self._profiler_with_clocks(monkeypatch, [0.0, 0.0, 1.0, 1.0], [0.0, 3.0])
        with ContextProfiler(profiler=profiler, step="multithread_step"):
            pass

        assert profiler.get_statistics()["steps"]["multithread_step"]["stall_time"] == 0.0

    def test_read_bytes_is_none_or_nonnegative_int(self):
        """read_bytes is either None (counter unavailable) or a non-negative int."""
        profiler = PerformanceProfiler()
        with ContextProfiler(profiler=profiler, step="any_step"):
            pass

        read_bytes = profiler.get_statistics()["steps"]["any_step"]["read_bytes"]
        assert read_bytes is None or (isinstance(read_bytes, int) and read_bytes >= 0)

    def test_read_process_io_bytes_probe(self):
        """The /proc/self/io probe returns None or a non-negative int, never raises."""
        value = _read_process_io_bytes()
        assert value is None or (isinstance(value, int) and value >= 0)
