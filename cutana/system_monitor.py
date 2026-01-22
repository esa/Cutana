#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
System resource monitoring module for Cutana.

This module handles:
- System resource monitoring (CPU, memory, disk usage)
- Kubernetes pod limit detection
- Resource constraint checking
- Cross-platform resource detection
"""

import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil
from loguru import logger


class SystemMonitor:
    """
    Monitors system resources and provides Kubernetes-aware resource detection.

    Handles both bare-metal and containerized environments with appropriate
    resource limit detection and monitoring.
    """

    def __init__(self):
        """Initialize the system monitor."""
        # Cache for Kubernetes limits to avoid repeated logging
        self._k8s_limits_cache: Optional[Tuple[Optional[int], Optional[int]]] = None
        self._k8s_limits_logged = False

        # Resource history
        self.resource_history = []
        self._lock = threading.Lock()

        # Initialize cpu_percent cache to avoid first-call issues
        self._cpu_initialized = False
        self._initialize_cpu_monitoring()

        logger.info("SystemMonitor initialized")

    def _is_datalabs_environment(self) -> bool:
        """
        Check if we're running on a datalabs node.

        Returns:
            True if hostname starts with 'datalab'
        """
        try:
            hostname = socket.gethostname().lower()
            is_datalabs = hostname.startswith("datalab")
            if is_datalabs:
                logger.trace(f"Detected datalabs environment: {hostname}")
            return is_datalabs
        except Exception as e:
            logger.error(f"Failed to get hostname: {e}")
            return False

    def _initialize_cpu_monitoring(self):
        """Initialize CPU monitoring to prime the psutil cache."""
        try:
            # Prime the CPU percentage cache with a short interval
            psutil.cpu_percent(interval=0.1)
            self._cpu_initialized = True
            logger.debug("CPU monitoring initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize CPU monitoring: {e}")
            self._cpu_initialized = False

    def _get_kubernetes_pod_limits(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get Kubernetes pod resource limits from cgroup files.

        Returns:
            Tuple of (memory_limit_bytes, cpu_limit_millicores) or (None, None) if not available
        """
        # Return cached limits if available
        if self._k8s_limits_cache is not None:
            return self._k8s_limits_cache

        memory_limit = None
        cpu_limit = None

        try:
            # Try to read memory limit from cgroup
            memory_cgroup_paths = [
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
                "/sys/fs/cgroup/memory.max",  # cgroup v2
            ]

            for path in memory_cgroup_paths:
                try:
                    if Path(path).exists():
                        with open(path, "r") as f:
                            limit_str = f.read().strip()
                            if limit_str != "max" and limit_str.isdigit():
                                memory_limit = int(limit_str)
                                if not self._k8s_limits_logged:
                                    logger.info(
                                        f"Found Kubernetes memory limit: {memory_limit / (1024**3):.1f}GB from {path}"
                                    )
                                break
                except Exception as e:
                    logger.debug(f"Failed to read memory limit from {path}: {e}")

        except Exception as e:
            logger.debug(f"Failed to get Kubernetes memory limits: {e}")

        try:
            # Try to read CPU limit from cgroup
            cpu_cgroup_paths = [
                (
                    "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                    "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
                ),  # cgroup v1
                ("/sys/fs/cgroup/cpu.max",),  # cgroup v2
            ]

            for paths in cpu_cgroup_paths:
                try:
                    if len(paths) == 2:
                        # cgroup v1: quota/period
                        quota_path, period_path = paths
                        if Path(quota_path).exists() and Path(period_path).exists():
                            with open(quota_path, "r") as f:
                                quota = int(f.read().strip())
                            with open(period_path, "r") as f:
                                period = int(f.read().strip())

                            if quota > 0 and period > 0:
                                # Convert to millicores (quota is in microseconds)
                                cpu_limit = int((quota / period) * 1000)
                                if not self._k8s_limits_logged:
                                    logger.info(
                                        f"Found Kubernetes CPU limit: {cpu_limit} millicores from {quota_path}"
                                    )
                                break
                    else:
                        # cgroup v2: max format is "quota period"
                        max_path = paths[0]
                        if Path(max_path).exists():
                            with open(max_path, "r") as f:
                                content = f.read().strip()
                                if content != "max" and " " in content:
                                    quota, period = content.split()
                                    quota, period = int(quota), int(period)
                                    if quota > 0 and period > 0:
                                        cpu_limit = int((quota / period) * 1000)
                                        if not self._k8s_limits_logged:
                                            logger.info(
                                                f"Found Kubernetes CPU limit: {cpu_limit} millicores from {max_path}"
                                            )
                                        break
                except Exception as e:
                    logger.debug(f"Failed to read CPU limit from {paths}: {e}")

        except Exception as e:
            logger.debug(f"Failed to get Kubernetes CPU limits: {e}")

        # Cache the results and mark as logged
        self._k8s_limits_cache = (memory_limit, cpu_limit)
        self._k8s_limits_logged = True

        return memory_limit, cpu_limit

    def get_system_resources(self) -> Dict[str, Any]:
        """
        Get current system resource usage.

        On datalabs, uses Kubernetes pod limits if available, otherwise falls back to system resources.

        Returns:
            Dictionary containing resource information
        """
        try:
            # Use interval=None for non-blocking call (returns cached value)
            cpu_percent = psutil.cpu_percent(interval=None)
            if not self._cpu_initialized and cpu_percent == 0.0:
                logger.warning(
                    "SystemMonitor: CPU monitoring not properly initialized, re-initializing"
                )
                self._initialize_cpu_monitoring()
                cpu_percent = psutil.cpu_percent(interval=None)

            memory = psutil.virtual_memory()
            logger.debug(
                f"SystemMonitor: Raw psutil data - CPU: {cpu_percent}%, Memory: {memory.total/(1024**3):.1f}GB total, {memory.available/(1024**3):.1f}GB available, {memory.percent}% used"
            )

            # Use current working directory for disk usage (cross-platform)
            try:
                disk = psutil.disk_usage(".")
            except (OSError, PermissionError):
                # Fallback for restricted systems
                disk = psutil.disk_usage("/") if hasattr(psutil, "disk_usage") else None

            # Check if we're on datalabs and get Kubernetes limits
            resource_source = "system"
            if self._is_datalabs_environment():
                k8s_memory_limit, _ = self._get_kubernetes_pod_limits()

                if k8s_memory_limit is not None:
                    # Use Kubernetes memory limits instead of system memory
                    memory_total = k8s_memory_limit

                    # For K8s pods, get the actual RSS usage from the current process tree
                    # rather than relying on system-wide memory stats which can be misleading
                    try:
                        current_process = psutil.Process()

                        # Get memory usage of current process and all children
                        memory_used = current_process.memory_info().rss
                        for child in current_process.children(recursive=True):
                            try:
                                memory_used += child.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue

                        # Cap at the K8s memory limit
                        memory_used = min(memory_used, memory_total)
                        memory_available = memory_total - memory_used
                        memory_percent = (
                            (memory_used / memory_total) * 100 if memory_total > 0 else 0
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to get process tree memory usage, falling back to system stats: {e}"
                        )
                        # Fallback: use system memory but constrained by K8s limits
                        system_memory_used = memory.total - memory.available
                        memory_used = min(system_memory_used, memory_total)
                        memory_available = max(0, memory_total - memory_used)
                        memory_percent = (
                            (memory_used / memory_total) * 100 if memory_total > 0 else 0
                        )

                    resource_source = "kubernetes_pod"
                    logger.debug(
                        f"Using Kubernetes memory limit: {memory_total / (1024**3):.1f}GB, used: {memory_used / (1024**3):.1f}GB, available: {memory_available / (1024**3):.1f}GB"
                    )
                else:
                    # Fall back to system memory
                    memory_total = memory.total
                    memory_available = memory.available
                    memory_percent = memory.percent
                    resource_source = "system_fallback"
            else:
                # Use system memory normally
                memory_total = memory.total
                memory_available = memory.available
                memory_percent = memory.percent

            result = {
                "cpu_percent": cpu_percent,
                "memory_total": memory_total,
                "memory_available": memory_available,
                "memory_percent": memory_percent,
                "disk_free": disk.free if disk else 0,
                "disk_total": disk.total if disk else 0,
                "resource_source": resource_source,  # Indicate source of resource info
                "timestamp": time.time(),
            }

            logger.debug(
                f"SystemMonitor: Returning resource data - CPU: {cpu_percent}%, Memory: {memory_total/(1024**3):.1f}GB total, {memory_available/(1024**3):.1f}GB available, {memory_percent}% used, Source: {resource_source}"
            )
            return result

        except Exception as e:
            logger.error(f"SystemMonitor: Error getting system resources: {e}")
            import traceback

            logger.error(f"SystemMonitor: Full traceback: {traceback.format_exc()}")

            fallback_result = {
                "cpu_percent": 0.0,
                "memory_total": 8 * 1024**3,  # 8GB fallback
                "memory_available": 4 * 1024**3,  # 4GB fallback
                "memory_percent": 50.0,
                "disk_free": 0,
                "disk_total": 0,
                "resource_source": "error_fallback",
                "timestamp": time.time(),
            }

            logger.warning(
                f"SystemMonitor: Returning fallback data due to error: {fallback_result}"
            )
            return fallback_result

    def check_memory_constraints(self, memory_required: int) -> bool:
        """
        Check if there's sufficient memory for processing.

        On datalabs, uses Kubernetes pod limits when available.

        Args:
            memory_required: Memory required in bytes

        Returns:
            True if sufficient memory is available
        """
        resources = self.get_system_resources()
        available_memory = resources["memory_available"]
        resource_source = resources.get("resource_source", "system")

        # Use a reasonable safety margin (1GB minimum)
        safety_margin = max(1024**3, available_memory * 0.1)  # 1GB or 10% of available
        safe_limit = available_memory - safety_margin

        can_proceed = memory_required <= safe_limit

        if not can_proceed:
            logger.warning(
                f"Memory constraint ({resource_source}): required {memory_required / (1024**3):.1f}GB, "
                f"available {available_memory / (1024**3):.1f}GB, "
                f"safety margin {safety_margin / (1024**3):.1f}GB"
            )
        else:
            logger.debug(
                f"Memory check passed ({resource_source}): required {memory_required / (1024**3):.1f}GB, "
                f"available {available_memory / (1024**3):.1f}GB"
            )

        return can_proceed

    def estimate_memory_usage(self, tile_size: int, num_workers: int) -> int:
        """
        Estimate memory usage for processing.

        Args:
            tile_size: Size of FITS tile in bytes
            num_workers: Number of worker processes

        Returns:
            Estimated memory usage in bytes
        """
        # Conservative estimate: 2.5x tile size per worker
        estimated_per_worker = tile_size * 2.5
        total_estimated = estimated_per_worker * num_workers

        return int(total_estimated)

    def record_resource_snapshot(self) -> None:
        """
        Record a snapshot of current resource usage.
        """
        resources = self.get_system_resources()

        with self._lock:
            self.resource_history.append(resources)

            # Limit history size to avoid memory issues
            if len(self.resource_history) > 1000:
                self.resource_history = self.resource_history[-500:]

    def get_resource_history(self) -> list:
        """
        Get resource usage history.

        Returns:
            List of resource snapshots
        """
        with self._lock:
            return self.resource_history.copy()

    def get_cpu_count(self) -> int:
        """
        Get the number of CPU cores available.

        Returns:
            Number of CPU cores
        """
        return psutil.cpu_count()

    def get_effective_cpu_count(self) -> int:
        """
        Get the effective number of CPU cores available, respecting Kubernetes limits.

        In datalabs/Kubernetes environments, uses pod CPU limits if available.
        Otherwise falls back to physical CPU count.

        Returns:
            Effective number of CPU cores available
        """
        # Get physical CPU count as fallback
        physical_cpu_count = self.get_cpu_count()

        # Check if we're in a datalabs/Kubernetes environment
        if self._is_datalabs_environment():
            memory_limit, cpu_limit_millicores = self._get_kubernetes_pod_limits()

            if cpu_limit_millicores is not None:
                # Convert millicores to cores and ensure at least 1 core
                k8s_cpu_count = max(1, cpu_limit_millicores // 1000)
                logger.debug(
                    f"Using Kubernetes CPU limit: {k8s_cpu_count} cores "
                    f"({cpu_limit_millicores} millicores) instead of physical {physical_cpu_count}"
                )
                return k8s_cpu_count

        logger.debug(f"Using physical CPU count: {physical_cpu_count}")
        return physical_cpu_count

    def get_conservative_cpu_limit(self, max_workers: int) -> int:
        """
        Get conservative CPU limit (N-1 cores, respecting max_workers).

        Args:
            max_workers: Maximum workers requested

        Returns:
            Conservative CPU limit
        """
        cpu_count = self.get_cpu_count()
        return min(max_workers, max(1, cpu_count - 1))

    def get_current_process_memory_mb(self) -> float:
        """
        Get current memory usage of this process in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB

            # Validate memory reading
            if memory_mb < 0:
                logger.warning(f"Negative memory reading: {memory_mb}MB")
                return 0.0

            logger.trace(
                f"Current process memory: {memory_mb:.1f}MB (RSS: {memory_info.rss} bytes)"
            )
            return memory_mb
        except psutil.NoSuchProcess:
            logger.error("Process no longer exists when trying to get memory usage")
            return 0.0
        except psutil.AccessDenied as e:
            logger.error(f"Access denied when getting memory usage (K8s permission issue?): {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return 0.0

    def report_process_memory_to_tracker(
        self, job_tracker, process_name: str, completed_sources: int, update_type: str = "sample"
    ) -> bool:
        """
        Measure current process memory and report it to the job tracker.

        This centralizes memory measurement and reporting logic to keep
        cutout_process.py clean and focused on processing logic.

        Args:
            job_tracker: JobTracker instance to report to
            process_name: Name/ID of the process being tracked
            completed_sources: Number of sources completed so far
            update_type: Type of update - "peak" (replace samples) or "sample" (add to samples)

        Returns:
            True if reporting was successful, False otherwise
        """
        try:
            # Measure current memory with better error handling
            try:
                current_memory_mb = self.get_current_process_memory_mb()
                if current_memory_mb <= 0:
                    logger.warning(
                        f"Invalid memory measurement: {current_memory_mb}MB for {process_name}"
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to get current process memory for {process_name}: {e}")
                return False

            # Read current progress for existing memory samples with detailed error handling
            current_progress = None
            existing_samples = []

            try:
                current_progress = job_tracker.process_reader.read_progress_file(process_name)
                if current_progress is not None:
                    existing_samples = current_progress.get("memory_footprint_samples", [])
                    logger.trace(
                        f"{process_name}: Read {len(existing_samples)} existing memory samples"
                    )
                else:
                    logger.debug(
                        f"No existing progress file for {process_name}, starting with empty samples"
                    )
            except Exception as e:
                logger.warning(f"Failed to read progress file for {process_name}: {e}")
                # Continue with empty samples - this is not fatal

            if update_type == "peak":
                # Replace existing samples with current peak measurement
                progress_update = {
                    "completed_sources": completed_sources,
                    "memory_footprint_mb": current_memory_mb,
                    "memory_footprint_samples": [current_memory_mb],
                }
                logger.debug(f"{process_name}: Peak memory measurement: {current_memory_mb:.1f}MB")

            else:  # update_type == "sample"
                # Add new sample and update peak
                updated_samples = existing_samples + [current_memory_mb]
                peak_memory = max(updated_samples) if updated_samples else current_memory_mb

                progress_update = {
                    "completed_sources": completed_sources,
                    "memory_footprint_mb": peak_memory,
                    "memory_footprint_samples": updated_samples,
                }
                logger.debug(
                    f"{process_name}: Memory sample: {current_memory_mb:.1f}MB, peak: {peak_memory:.1f}MB ({len(updated_samples)} samples)"
                )

            # Report to job tracker with detailed error handling
            try:
                success = job_tracker.update_process_progress(process_name, progress_update)
                if not success:
                    logger.error(
                        f"JobTracker.update_process_progress returned False for {process_name}"
                    )
                return success
            except Exception as e:
                logger.error(f"Failed to update job tracker progress for {process_name}: {e}")
                return False

        except Exception as e:
            logger.error(f"Unexpected error in memory reporting for {process_name}: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
