#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for the SystemMonitor module.

Tests cover:
- System resource monitoring (CPU, memory, disk)
- Resource constraint checking
- Memory usage calculations
- Kubernetes environment detection
- Resource limit calculations
- Resource history trimming
- Cgroup v1/v2 file parsing for K8s limits
"""

import builtins
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cutana.system_monitor import SystemMonitor


class TestSystemMonitor:
    """Test suite for SystemMonitor class."""

    @pytest.fixture
    def system_monitor(self):
        """Create SystemMonitor instance for testing."""
        return SystemMonitor()

    def test_system_monitor_initialization(self, system_monitor):
        """Test SystemMonitor initializes correctly."""
        assert system_monitor is not None
        assert hasattr(system_monitor, "resource_history")
        assert hasattr(system_monitor, "_lock")

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_get_system_resources(self, mock_disk, mock_memory, mock_cpu, system_monitor):
        """Test system resource monitoring."""
        # Setup mocks
        mock_cpu.return_value = 75.5
        mock_memory.return_value.total = 16 * 1024**3  # 16GB
        mock_memory.return_value.available = 10 * 1024**3  # 10GB
        mock_memory.return_value.percent = 37.5
        mock_disk.return_value.free = 500 * 1024**3  # 500GB
        mock_disk.return_value.total = 1000 * 1024**3  # 1TB

        resources = system_monitor.get_system_resources()

        assert resources["cpu_percent"] == 75.5
        assert resources["memory_total"] == 16 * 1024**3
        assert resources["memory_available"] == 10 * 1024**3
        assert resources["memory_percent"] == 37.5
        assert resources["disk_free"] == 500 * 1024**3
        assert "timestamp" in resources
        assert "resource_source" in resources

    @patch("psutil.cpu_count")
    def test_get_cpu_count(self, mock_cpu_count, system_monitor):
        """Test CPU count detection."""
        mock_cpu_count.return_value = 8
        cpu_count = system_monitor.get_cpu_count()
        assert cpu_count == 8

    @patch("socket.gethostname", return_value="datalab-node-001")
    def test_datalabs_environment_detection(self, mock_hostname, system_monitor):
        """Test datalabs environment detection."""
        is_datalabs = system_monitor._is_datalabs_environment()
        assert is_datalabs is True

    @patch("socket.gethostname", return_value="regular-server")
    def test_non_datalabs_environment_detection(self, mock_hostname, system_monitor):
        """Test non-datalabs environment detection."""
        is_datalabs = system_monitor._is_datalabs_environment()
        assert is_datalabs is False

    def test_kubernetes_pod_limits_detection(self, system_monitor):
        """Test Kubernetes pod limits detection."""
        cpu_limit, memory_limit = system_monitor._get_kubernetes_pod_limits()
        # Should return None values if not in k8s or no limits set
        assert cpu_limit is None or isinstance(cpu_limit, int)
        assert memory_limit is None or isinstance(memory_limit, int)

    @patch("psutil.virtual_memory")
    def test_check_memory_constraints_sufficient(self, mock_memory, system_monitor):
        """Test memory constraint checking with sufficient memory."""
        # Mock complete memory object with all required attributes
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3  # 16GB total
        mock_mem.available = 8 * 1024**3  # 8GB available
        mock_mem.percent = 50.0  # 50% used
        mock_memory.return_value = mock_mem

        # Request 4GB - should be allowed
        can_proceed = system_monitor.check_memory_constraints(4 * 1024**3)
        assert can_proceed is True

    @patch("psutil.virtual_memory")
    def test_check_memory_constraints_insufficient(self, mock_memory, system_monitor):
        """Test memory constraint checking with insufficient memory."""
        mock_memory.return_value.available = 2 * 1024**3  # 2GB available

        # Request 4GB - should be denied
        can_proceed = system_monitor.check_memory_constraints(4 * 1024**3)
        assert can_proceed is False

    def test_estimate_memory_usage(self, system_monitor):
        """Test memory usage estimation for processes."""
        tile_size = 2 * 1024**3  # 2GB tile
        num_workers = 4

        # Should use 2.5x multiplier
        estimated = system_monitor.estimate_memory_usage(tile_size, num_workers)
        expected = tile_size * num_workers * 2.5
        assert estimated == expected

    def test_get_conservative_cpu_limit(self, system_monitor):
        """Test conservative CPU limit calculation."""
        cpu_limit = system_monitor.get_conservative_cpu_limit(max_workers=8)
        assert isinstance(cpu_limit, int)
        assert cpu_limit > 0

    @patch("psutil.Process")
    def test_get_current_process_memory_mb(self, mock_process, system_monitor):
        """Test current process memory usage detection."""
        mock_process.return_value.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB

        memory_mb = system_monitor.get_current_process_memory_mb()
        assert memory_mb == 512.0

    @patch("psutil.virtual_memory")
    def test_memory_pressure_detection(self, mock_memory, system_monitor):
        """Test memory pressure detection."""
        # High memory usage scenario - Mock complete memory object
        mock_mem = MagicMock()
        mock_mem.total = 10 * 1024**3  # 10GB total
        mock_mem.available = 1 * 1024**3  # 1GB available
        mock_mem.percent = 90.0  # 90% used
        mock_memory.return_value = mock_mem

        resources = system_monitor.get_system_resources()
        assert resources["memory_percent"] == 90.0

        # Should detect high memory pressure
        can_proceed = system_monitor.check_memory_constraints(2 * 1024**3)  # Request 2GB
        assert can_proceed is False

    def test_resource_history_tracking(self, system_monitor):
        """Test resource usage history tracking."""
        with patch.object(system_monitor, "get_system_resources") as mock_resources:
            mock_resources.return_value = {
                "cpu_percent": 50.0,
                "memory_percent": 60.0,
                "timestamp": 1234567890,
            }

            # Record multiple snapshots
            system_monitor.record_resource_snapshot()
            system_monitor.record_resource_snapshot()
            system_monitor.record_resource_snapshot()

            history = system_monitor.get_resource_history()
            assert len(history) == 3
            assert all("cpu_percent" in snapshot for snapshot in history)
            assert all("memory_percent" in snapshot for snapshot in history)


# ────────────────────────────────────────────────────────────────────
# Resource history trimming (line 364-365)
# ────────────────────────────────────────────────────────────────────


class TestResourceHistoryTrimming:
    """Tests for history trimming when >1000 entries."""

    def test_history_trimmed_at_threshold(self):
        monitor = SystemMonitor()
        dummy = {"cpu_percent": 1.0, "memory_percent": 2.0, "timestamp": 0}

        with patch.object(monitor, "get_system_resources", return_value=dummy):
            # Record 1001 snapshots to exceed the 1000-entry threshold
            for _ in range(1001):
                monitor.record_resource_snapshot()

        history = monitor.get_resource_history()
        # After exceeding 1000, history should be trimmed to the last 500
        assert len(history) == 500

    def test_history_not_trimmed_below_threshold(self):
        monitor = SystemMonitor()
        dummy = {"cpu_percent": 1.0, "memory_percent": 2.0, "timestamp": 0}

        with patch.object(monitor, "get_system_resources", return_value=dummy):
            for _ in range(999):
                monitor.record_resource_snapshot()

        history = monitor.get_resource_history()
        assert len(history) == 999


# ────────────────────────────────────────────────────────────────────
# Cgroup v1 memory limit parsing (lines 100-113)
# ────────────────────────────────────────────────────────────────────


def _is_cgroup_path(s):
    """Check if a path string refers to a cgroup file, regardless of OS separators."""
    normalized = s.replace("\\", "/")
    return normalized.startswith("/sys/fs/cgroup/") or normalized.lstrip("/").startswith(
        "sys/fs/cgroup/"
    )


def _cgroup_to_relative(s):
    """Convert a cgroup path to a relative path for tmp_path lookup."""
    return s.replace("\\", "/").lstrip("/")


def _make_cgroup_patcher(tmp_path):
    """Create a context manager that redirects cgroup paths to tmp_path.

    Patches Path.exists() and builtins.open() so hardcoded /sys/fs/cgroup/*
    paths read from files under tmp_path instead.
    """
    real_open = builtins.open
    real_exists = Path.exists

    def fake_exists(self):
        s = str(self)
        if _is_cgroup_path(s):
            return (tmp_path / _cgroup_to_relative(s)).exists()
        return real_exists(self)

    def fake_open(file, *args, **kwargs):
        s = str(file)
        if _is_cgroup_path(s):
            return real_open(str(tmp_path / _cgroup_to_relative(s)), *args, **kwargs)
        return real_open(file, *args, **kwargs)

    @contextmanager
    def patcher():
        with patch.object(Path, "exists", fake_exists), patch("builtins.open", fake_open):
            yield

    return patcher()


class TestCgroupMemoryParsing:
    """Tests for _get_kubernetes_pod_limits memory cgroup file parsing."""

    def test_cgroup_v1_memory_limit(self, tmp_path):
        """Real method reads cgroup v1 memory.limit_in_bytes."""
        limit_bytes = 8 * 1024**3  # 8GB
        cgroup_dir = tmp_path / "sys" / "fs" / "cgroup" / "memory"
        cgroup_dir.mkdir(parents=True)
        (cgroup_dir / "memory.limit_in_bytes").write_text(str(limit_bytes))

        monitor = SystemMonitor()
        monitor._k8s_limits_cache = None
        monitor._k8s_limits_logged = False

        with _make_cgroup_patcher(tmp_path):
            mem_limit, _cpu_limit = monitor._get_kubernetes_pod_limits()

        assert mem_limit == limit_bytes

    def test_cgroup_v2_memory_max_unlimited(self, tmp_path):
        """Real method skips 'max' value in cgroup v2 memory.max."""
        cgroup_dir = tmp_path / "sys" / "fs" / "cgroup"
        cgroup_dir.mkdir(parents=True)
        (cgroup_dir / "memory.max").write_text("max\n")

        monitor = SystemMonitor()
        monitor._k8s_limits_cache = None
        monitor._k8s_limits_logged = False

        with _make_cgroup_patcher(tmp_path):
            mem_limit, _cpu_limit = monitor._get_kubernetes_pod_limits()

        assert mem_limit is None

    def test_cgroup_v2_memory_numeric(self, tmp_path):
        """Real method reads numeric cgroup v2 memory.max."""
        limit_bytes = 4 * 1024**3
        cgroup_dir = tmp_path / "sys" / "fs" / "cgroup"
        cgroup_dir.mkdir(parents=True)
        (cgroup_dir / "memory.max").write_text(f"{limit_bytes}\n")

        monitor = SystemMonitor()
        monitor._k8s_limits_cache = None
        monitor._k8s_limits_logged = False

        with _make_cgroup_patcher(tmp_path):
            mem_limit, _cpu_limit = monitor._get_kubernetes_pod_limits()

        assert mem_limit == limit_bytes


class TestCgroupCpuParsing:
    """Tests for CPU cgroup file parsing logic."""

    def test_cgroup_v1_cpu_quota_period(self, tmp_path):
        """Real method reads cgroup v1 cpu quota and period files."""
        cgroup_dir = tmp_path / "sys" / "fs" / "cgroup" / "cpu"
        cgroup_dir.mkdir(parents=True)
        (cgroup_dir / "cpu.cfs_quota_us").write_text("200000\n")
        (cgroup_dir / "cpu.cfs_period_us").write_text("100000\n")

        monitor = SystemMonitor()
        monitor._k8s_limits_cache = None
        monitor._k8s_limits_logged = False

        with _make_cgroup_patcher(tmp_path):
            _mem_limit, cpu_limit = monitor._get_kubernetes_pod_limits()

        assert cpu_limit == 2000  # 2 cores in millicores

    def test_cgroup_v2_cpu_max(self, tmp_path):
        """Real method reads cgroup v2 cpu.max with 'quota period' format."""
        cgroup_dir = tmp_path / "sys" / "fs" / "cgroup"
        cgroup_dir.mkdir(parents=True)
        (cgroup_dir / "cpu.max").write_text("400000 100000\n")

        monitor = SystemMonitor()
        monitor._k8s_limits_cache = None
        monitor._k8s_limits_logged = False

        with _make_cgroup_patcher(tmp_path):
            _mem_limit, cpu_limit = monitor._get_kubernetes_pod_limits()

        assert cpu_limit == 4000  # 4 cores in millicores

    def test_cgroup_v2_cpu_max_unlimited(self, tmp_path):
        """Real method handles 'max period' in cgroup v2 cpu.max gracefully."""
        cgroup_dir = tmp_path / "sys" / "fs" / "cgroup"
        cgroup_dir.mkdir(parents=True)
        (cgroup_dir / "cpu.max").write_text("max 100000\n")

        monitor = SystemMonitor()
        monitor._k8s_limits_cache = None
        monitor._k8s_limits_logged = False

        with _make_cgroup_patcher(tmp_path):
            _mem_limit, cpu_limit = monitor._get_kubernetes_pod_limits()

        # "max" can't be parsed as int, exception is caught, cpu_limit stays None
        assert cpu_limit is None

    def test_k8s_limits_cache(self):
        """_get_kubernetes_pod_limits returns cached results on second call."""
        monitor = SystemMonitor()
        monitor._k8s_limits_cache = (1024, 2000)
        mem, cpu = monitor._get_kubernetes_pod_limits()
        assert mem == 1024
        assert cpu == 2000
        assert cpu == 2000
