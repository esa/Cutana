#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Load balancer module for Cutana - manages dynamic resource allocation and process spawning.

This module handles:
- Main process memory monitoring and worker memory allocation
- Dynamic memory-based process spawning decisions
- Periodic memory usage tracking with peak detection
- Real-time load balancing with detailed logging
"""

import time
from collections import deque
from typing import Any, Dict, List, Tuple

from dotmap import DotMap
from loguru import logger

from .process_status_reader import ProcessStatusReader
from .system_monitor import SystemMonitor


class LoadBalancer:
    """
    Manages dynamic load balancing for cutout processing.

    Monitors main process memory and worker memory usage to make intelligent
    decisions about when to spawn new worker processes.
    """

    def __init__(self, progress_dir: str = None, session_id: str = None):
        """
        Initialize the load balancer.

        Args:
            progress_dir: Directory for progress files
            session_id: Session ID for progress file isolation
        """
        self.system_monitor = SystemMonitor()
        self.process_reader = ProcessStatusReader(progress_dir=progress_dir, session_id=session_id)

        # Configuration (will be set by update_config_with_loadbalancing)
        self.memory_safety_margin = 0.1
        self.memory_poll_interval = 3
        self.memory_peak_window = 30
        self.main_process_memory_reserve_gb = 2.0
        self.initial_workers = 1
        self.log_interval = 30
        self.skip_memory_calibration_wait = False

        # Main process memory tracking
        self.main_process_memory_mb = None
        self.main_memory_samples = deque(maxlen=10)  # Track last 10 samples

        # Worker memory tracking with windowed peak detection
        self.worker_memory_history = deque()  # [(timestamp, memory_mb), ...]
        self.worker_memory_peak_mb = None
        self.worker_memory_allocation_mb = None

        # Process performance metrics
        self.processes_measured = 0
        self.active_worker_count = 0

        # Calibration tracking to prevent repeated UI calibration spinners
        self.calibration_completed = False

        # Resource limits
        self.memory_limit_bytes = None
        self.cpu_limit = None

        # FITS size tracking for better estimation
        self.avg_fits_set_size_mb = None
        self.batch_size = 1000  # Will be updated from config
        self.target_resolution = 256  # Will be updated from config
        self.num_channels = 1  # Will be updated based on selected extensions

        # Simple timing for periodic operations
        self.last_log_time = time.time()
        self.last_decision_time = time.time()

        # System memory tracking for real worker memory estimation
        self.system_memory_history = deque(maxlen=20)  # Track last 20 samples
        self.baseline_memory_mb = None  # System memory before workers started

        # Event logging
        self.event_log_file = None

        logger.info("LoadBalancer initialized with improved memory monitoring")

    def _log_event(self, category: str, event_type: str, data: Dict[str, Any] = None) -> None:
        """
        Log a LoadBalancer event to the event log file if configured.

        Args:
            category: Category of the event (e.g., "LoadBalancer", "Memory", "Spawn")
            event_type: Type of event (e.g., "decision", "memory_update", "worker_spawned")
            data: Optional dictionary with event data
        """
        if not self.event_log_file:
            return

        try:
            import json
            from pathlib import Path

            event = {
                "timestamp": time.time(),
                "category": category,
                "event_type": event_type,
                "data": data or {},
            }

            # Ensure parent directory exists
            Path(self.event_log_file).parent.mkdir(parents=True, exist_ok=True)

            # Append event to log file (JSON lines format)
            with open(self.event_log_file, "a") as f:
                f.write(json.dumps(event) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log LoadBalancer event to {self.event_log_file}: {e}")

    def update_memory_tracking(self) -> None:
        """
        Update memory tracking synchronously. Called by orchestrator when needed.
        """
        try:
            # Monitor main process memory
            main_memory = self.system_monitor.get_current_process_memory_mb()

            self.main_memory_samples.append(main_memory)
            # Use average of recent samples to smooth out spikes
            if self.main_memory_samples:
                self.main_process_memory_mb = sum(self.main_memory_samples) / len(
                    self.main_memory_samples
                )

            # Track system memory usage for real worker estimation
            self._update_system_memory_tracking()

            # Update worker memory allocation based on current state
            self._update_worker_memory_allocation()

            # Clean old worker memory history (keep only last window)
            current_time = time.time()
            cutoff_time = current_time - self.memory_peak_window
            while self.worker_memory_history and self.worker_memory_history[0][0] < cutoff_time:
                self.worker_memory_history.popleft()

            # Calculate peak in window
            if self.worker_memory_history:
                self.worker_memory_peak_mb = max(m for _, m in self.worker_memory_history)

        except Exception as e:
            logger.error(f"Error in memory tracking update: {e}")

    def log_memory_status_if_needed(self) -> None:
        """
        Log memory status if enough time has passed. Called by orchestrator periodically.
        """
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self._log_memory_status()
            self.last_log_time = current_time

    def _update_worker_memory_allocation(self):
        """Update the worker memory allocation based on current system state."""
        try:
            resources = self.system_monitor.get_system_resources()
            memory_available = resources["memory_available"] / (1024 * 1024)  # Convert to MB

            # Always use actual measured main process memory
            if self.main_process_memory_mb is not None:
                # Use actual measured main process memory with a small buffer
                main_reserved = self.main_process_memory_mb * 1.2  # 20% buffer for growth
                logger.debug(f"Using actual main process memory: {main_reserved:.1f}MB")
            else:
                # Get main process memory from system monitor
                main_memory = self.system_monitor.get_current_process_memory_mb()
                self.main_process_memory_mb = main_memory
                main_reserved = main_memory * 1.2  # 20% buffer for growth
                logger.debug(
                    f"Retrieved main process memory from system monitor: {main_reserved:.1f}MB"
                )

            # Calculate worker memory allocation
            worker_available = memory_available - main_reserved
            self.worker_memory_allocation_mb = worker_available * (1 - self.memory_safety_margin)

        except Exception as e:
            logger.error(f"Failed to update worker memory allocation: {e}")

    def _update_system_memory_tracking(self):
        """Track system memory usage to compute real worker memory consumption."""
        try:
            resources = self.system_monitor.get_system_resources()
            memory_total = resources["memory_total"] / (1024 * 1024)  # Convert to MB
            memory_available = resources["memory_available"] / (1024 * 1024)
            memory_used = memory_total - memory_available

            # Store system memory usage
            current_time = time.time()
            self.system_memory_history.append((current_time, memory_used))

            # Set baseline when no workers are active yet
            if self.baseline_memory_mb is None and self.active_worker_count == 0:
                self.baseline_memory_mb = memory_used

            # Calculate worker memory based on real usage
            if (
                self.baseline_memory_mb is not None
                and self.active_worker_count > 0
                and len(self.system_memory_history) > 5
            ):

                # Get recent memory usage
                recent_memory = [mem for _, mem in list(self.system_memory_history)[-5:]]
                current_peak_memory = max(recent_memory)

                # Worker memory = (current_peak - baseline - main_process) / active_workers
                main_memory = self.main_process_memory_mb if self.main_process_memory_mb else 0
                worker_total_memory = max(
                    0, current_peak_memory - self.baseline_memory_mb - main_memory
                )

                if worker_total_memory > 0 and self.active_worker_count > 0:
                    per_worker_memory = worker_total_memory / self.active_worker_count

                    # Update worker peak with real measurement
                    current_time = time.time()
                    self.worker_memory_history.append((current_time, per_worker_memory))

                    # Log the real measurement
                    logger.debug(
                        f"Real worker memory measurement: "
                        f"total_used={current_peak_memory:.1f}MB, "
                        f"baseline={self.baseline_memory_mb:.1f}MB, "
                        f"main={main_memory:.1f}MB, "
                        f"worker_total={worker_total_memory:.1f}MB, "
                        f"per_worker={per_worker_memory:.1f}MB "
                        f"({self.active_worker_count} workers)"
                    )

        except Exception as e:
            logger.error(f"Failed to update system memory tracking: {e}")

    def _log_memory_status(self):
        """Log detailed memory status information."""
        main_mem = self.main_process_memory_mb if self.main_process_memory_mb is not None else 0
        worker_alloc = (
            self.worker_memory_allocation_mb if self.worker_memory_allocation_mb is not None else 0
        )
        worker_peak = self.worker_memory_peak_mb if self.worker_memory_peak_mb is not None else 0

        # Log at INFO level so benchmark parser can capture it
        logger.info(
            f"LoadBalancer Memory Status | "
            f"Main Process: {main_mem:.1f}MB | "
            f"Worker Allocation: {worker_alloc:.1f}MB | "
            f"Worker Peak: {worker_peak:.1f}MB | "
            f"Active Workers: {self.active_worker_count} | "
            f"Processes Measured: {self.processes_measured}"
        )

        # Log memory status event
        self._log_event(
            "Memory",
            "status",
            {
                "main_process_memory_mb": main_mem,
                "worker_allocation_mb": worker_alloc,
                "worker_peak_mb": worker_peak,
                "active_workers": self.active_worker_count,
                "processes_measured": self.processes_measured,
                "baseline_memory_mb": self.baseline_memory_mb,
            },
        )

    def update_config_with_loadbalancing(self, config: DotMap, total_sources: int = None) -> None:
        """
        Update configuration with optimal load balancing settings based on system resources.

        Args:
            config: Configuration DotMap to update
            total_sources: Total number of sources to process (optional)
        """
        # Update internal configuration from config
        self.memory_safety_margin = config.loadbalancer.memory_safety_margin
        self.memory_poll_interval = config.loadbalancer.memory_poll_interval
        self.memory_peak_window = config.loadbalancer.memory_peak_window
        self.main_process_memory_reserve_gb = config.loadbalancer.main_process_memory_reserve_gb
        self.initial_workers = config.loadbalancer.initial_workers
        self.log_interval = config.loadbalancer.log_interval
        self.skip_memory_calibration_wait = config.loadbalancer.skip_memory_calibration_wait

        # Setup event logging if configured
        if config.loadbalancer.event_log_file:
            self.event_log_file = config.loadbalancer.event_log_file
            self._log_event(
                "LoadBalancer", "event_logging_enabled", {"log_file": self.event_log_file}
            )

        # Get batch size and target resolution from config
        self.batch_size = config.N_batch_cutout_process
        self.target_resolution = config.target_resolution

        # Determine number of channels from selected extensions
        self.num_channels = len(config.selected_extensions)
        # Infer from channel weights
        self.num_channels = max(len(weights) for weights in config.channel_weights.values())

        # Get system resources
        resources = self.system_monitor.get_system_resources()
        cpu_count = self.system_monitor.get_cpu_count()

        # Get Kubernetes CPU limit if available
        memory_limit, cpu_limit_millicores = self.system_monitor._get_kubernetes_pod_limits()

        # Use K8s CPU limit if available, otherwise use physical CPU count
        effective_cpu_count = cpu_count
        if cpu_limit_millicores is not None:
            effective_cpu_count = max(
                1, cpu_limit_millicores // 1000
            )  # Convert millicores to cores
            logger.info(
                f"Using Kubernetes CPU limit: {effective_cpu_count} cores ({cpu_limit_millicores} millicores)"
            )

        # Determine CPU limit (N-1 cores from effective count)
        max_workers = int(min(config.max_workers, effective_cpu_count - 1))

        # Determine memory limit with safety margin
        memory_total = resources["memory_total"]
        memory_available = resources["memory_available"]
        resource_source = resources.get("resource_source", "system")

        # Apply safety margin to available memory
        memory_limit_bytes = int(memory_available * (1 - self.memory_safety_margin))
        memory_limit_gb = memory_limit_bytes / (1024**3)

        # Determine max_sources_per_process based on job size
        # Check if user has explicitly set max_sources_per_process
        if (
            hasattr(config.loadbalancer, "max_sources_per_process")
            and config.loadbalancer.max_sources_per_process
        ):
            # User has set a value - respect it
            max_sources_per_process = config.loadbalancer.max_sources_per_process
            logger.info(f"Using user-configured max_sources_per_process: {max_sources_per_process}")
        else:
            # Use automatic logic based on job size
            if total_sources is not None and total_sources < 1e6:
                max_sources_per_process = 12500  # Smaller batches for smaller jobs
            else:
                max_sources_per_process = 1e5  # Larger batches for large jobs

        # Set batch size for cutout process
        n_batch_cutout_process = self.batch_size

        config.loadbalancer.max_workers = max_workers
        config.loadbalancer.max_sources_per_process = int(max_sources_per_process)
        config.loadbalancer.N_batch_cutout_process = int(n_batch_cutout_process)
        config.loadbalancer.memory_limit_gb = memory_limit_gb
        config.loadbalancer.memory_limit_bytes = memory_limit_bytes
        config.loadbalancer.resource_source = resource_source
        config.loadbalancer.cpu_count = cpu_count
        config.loadbalancer.memory_available_gb = memory_available / (1024**3)
        config.loadbalancer.memory_total_gb = memory_total / (1024**3)
        config.loadbalancer.safety_margin = self.memory_safety_margin

        # Store limits for future decisions
        self.memory_limit_bytes = memory_limit_bytes
        self.cpu_limit = max_workers

        logger.info(
            f"LoadBalancer configuration ({resource_source}): "
            f"max_workers={max_workers}/{effective_cpu_count}, "
            f"memory_limit={memory_limit_gb:.1f}GB/{memory_total/(1024**3):.1f}GB, "
            f"max_sources_per_process={max_sources_per_process}, "
            f"N_batch_cutout_process={n_batch_cutout_process}, "
            f"initial_workers={self.initial_workers}"
        )

    def update_memory_statistics(self, process_id: str) -> None:
        """
        Update memory usage statistics from a completed process.

        Args:
            process_id: Process ID to get memory statistics from
        """
        # Read process data from progress file
        process_data = self.process_reader.read_progress_file(process_id)

        if not process_data:
            return

        # Get memory footprint samples
        memory_samples = process_data.get("memory_footprint_samples", [])
        if not memory_samples:
            memory_mb = process_data.get("memory_footprint_mb", 0.0)
            if memory_mb > 0:
                memory_samples = [memory_mb]

        if memory_samples:
            # Find peak memory from this process
            peak_memory_mb = max(memory_samples)

            # Add to worker memory history with timestamp
            current_time = time.time()
            self.worker_memory_history.append((current_time, peak_memory_mb))

            # Update peak in current window
            cutoff_time = current_time - self.memory_peak_window
            recent_samples = [m for t, m in self.worker_memory_history if t >= cutoff_time]
            if recent_samples:
                self.worker_memory_peak_mb = max(recent_samples)

            self.processes_measured += 1

            logger.info(
                f"Updated worker memory from {process_id}: "
                f"peak={peak_memory_mb:.1f}MB, "
                f"window_peak={self.worker_memory_peak_mb:.1f}MB "
                f"(from {self.processes_measured} processes)"
            )

    def update_active_worker_count(self, count: int) -> None:
        """
        Update the number of active worker processes.

        Args:
            count: Current number of active worker processes
        """
        old_count = self.active_worker_count
        self.active_worker_count = count

        # Set baseline when workers start/stop
        if count == 0 and old_count > 0:
            # All workers stopped - we can reset baseline on next measurement
            self.baseline_memory_mb = None
            logger.info("All workers stopped - baseline will be reset")
        elif count > 0 and old_count == 0:
            # Workers starting - baseline will be captured automatically
            logger.info(f"Workers starting ({count}) - baseline will be captured")

        if old_count != count:
            logger.info(f"Active worker count: {old_count} → {count}")

    def _get_remaining_worker_memory(self) -> float:
        """
        Get remaining worker memory allocation using total memory approach.

        Uses: TOTAL_MEMORY - main_actual - safety_margin - (worker_peak * active_workers)

        Returns:
            Remaining memory in MB
        """
        try:
            resources = self.system_monitor.get_system_resources()
            memory_total_mb = resources["memory_total"] / (1024 * 1024)

            # Always use actual measured main process memory
            if self.main_process_memory_mb is not None:
                # Use actual measured main process memory with buffer
                main_reserved_mb = self.main_process_memory_mb * 1.2  # 20% buffer
            else:
                # Get main process memory from system monitor
                main_memory = self.system_monitor.get_current_process_memory_mb()
                self.main_process_memory_mb = main_memory
                main_reserved_mb = main_memory * 1.2  # 20% buffer

            safety_margin_mb = memory_total_mb * self.memory_safety_margin

            # Calculate current worker usage
            if self.worker_memory_peak_mb and self.active_worker_count > 0:
                current_worker_usage_mb = self.worker_memory_peak_mb * self.active_worker_count
            else:
                current_worker_usage_mb = 0

            # Total approach: TOTAL - actual_main - safety_margin - current_usage
            remaining_mb = (
                memory_total_mb - main_reserved_mb - safety_margin_mb - current_worker_usage_mb
            )

            return max(0, remaining_mb)

        except Exception as e:
            logger.error(f"Failed to calculate remaining worker memory: {e}")
            return 0

    def can_spawn_new_process(
        self, active_process_count: int, active_process_ids: List[str] = None
    ) -> Tuple[bool, str]:
        """
        Determine if a new process can be spawned based on current resource usage.

        Args:
            active_process_count: Number of currently active processes
            active_process_ids: List of currently active process IDs (optional)

        Returns:
            Tuple of (can_spawn, reason_message)
        """
        self.active_worker_count = active_process_count

        # Log decision timing
        current_time = time.time()
        time_since_last = current_time - self.last_decision_time
        self.last_decision_time = current_time

        # Check CPU limit
        if self.cpu_limit is None:
            self.cpu_limit = self.system_monitor.get_cpu_count() - 1

        if active_process_count >= self.cpu_limit:
            reason = f"CPU limit reached ({active_process_count}/{self.cpu_limit})"
            logger.info(
                f"LoadBalancer spawn decision (after {time_since_last:.1f}s): NO - {reason}"
            )

            # Log spawn decision event
            self._log_event(
                "Spawn",
                "decision",
                {
                    "can_spawn": False,
                    "reason": reason,
                    "active_processes": active_process_count,
                    "cpu_limit": self.cpu_limit,
                    "time_since_last_decision": time_since_last,
                },
            )

            return False, reason

        # For the first worker, always allow (following initial_workers setting)
        if active_process_count < self.initial_workers:
            reason = f"Initial worker spawn ({active_process_count}/{self.initial_workers})"
            logger.info(f"LoadBalancer spawn decision: YES - {reason}")

            # Log spawn decision event
            self._log_event(
                "Spawn",
                "decision",
                {
                    "can_spawn": True,
                    "reason": reason,
                    "active_processes": active_process_count,
                    "initial_workers": self.initial_workers,
                    "time_since_last_decision": time_since_last,
                },
            )

            return True, reason

        # For subsequent workers, check memory constraints
        if not self.worker_memory_allocation_mb:
            self._update_worker_memory_allocation()

        # For additional workers, require real memory measurements from first worker
        # AND that the first worker has completed at least one source
        # Skip this check if skip_memory_calibration_wait is enabled
        if not self.skip_memory_calibration_wait and (
            not self.calibration_completed or self.worker_memory_peak_mb is None
        ):
            # Check if any active process has completed sources using JobTracker
            # IMPORTANT: Use the same session_id as the ProcessStatusReader to access the same progress files
            import tempfile

            from .job_tracker import JobTracker

            temp_tracker = JobTracker(
                progress_dir=tempfile.gettempdir(), session_id=self.process_reader.session_id
            )
            process_details = temp_tracker.get_process_details()
            logger.debug(f"LoadBalancer: Checking {len(process_details)} processes for calibration")

            any_progress = False
            memory_found = False
            for process_id, details in process_details.items():
                # Only check processes that are currently active to avoid reading from stopped processes
                if active_process_ids and process_id not in active_process_ids:
                    logger.debug(f"LoadBalancer: Skipping inactive process {process_id}")
                    continue

                completed_sources = details.get("completed_sources")
                if completed_sources is not None and completed_sources > 0:
                    any_progress = True

                    # Only mark calibration completed once we have memory readings
                    if not self.calibration_completed:
                        logger.info(
                            f"LoadBalancer: Process {process_id} has {completed_sources} completed sources, checking for memory data"
                        )

                    # Try to read memory data from this process
                    process_data = self.process_reader.read_progress_file(process_id)
                    if process_data:
                        memory_samples = process_data.get("memory_footprint_samples", [])
                        if not memory_samples:
                            memory_mb = process_data.get("memory_footprint_mb", 0.0)
                            if memory_mb > 0:
                                memory_samples = [memory_mb]

                        if memory_samples and not memory_found:
                            peak_memory_mb = max(memory_samples)
                            current_time = time.time()
                            self.worker_memory_history.append((current_time, peak_memory_mb))
                            self.worker_memory_peak_mb = peak_memory_mb
                            self.calibration_completed = True
                            memory_found = True
                            logger.info(
                                f"LoadBalancer: Calibration completed with memory measurement from {process_id}: {peak_memory_mb:.1f}MB"
                            )
                            # Continue checking other processes to gather more memory data
                        elif memory_samples and memory_found:
                            # Add additional memory samples from other processes
                            peak_memory_mb = max(memory_samples)
                            current_time = time.time()
                            self.worker_memory_history.append((current_time, peak_memory_mb))
                            # Update peak if this process used more memory
                            if peak_memory_mb > self.worker_memory_peak_mb:
                                self.worker_memory_peak_mb = peak_memory_mb
                                logger.info(
                                    f"LoadBalancer: Updated peak memory from {process_id}: {peak_memory_mb:.1f}MB"
                                )
                    else:
                        logger.debug(
                            f"LoadBalancer: Could not read progress file for {process_id}, will check other processes"
                        )

            if not any_progress or self.worker_memory_peak_mb is None:
                reason = "Waiting for first worker to complete at least one source with memory measurements"
                logger.info(f"LoadBalancer spawn decision: NO - {reason}")

                # Log spawn decision event
                self._log_event(
                    "Spawn",
                    "decision",
                    {
                        "can_spawn": False,
                        "reason": reason,
                        "active_processes": active_process_count,
                        "processes_measured": self.processes_measured,
                        "worker_memory_peak_mb": self.worker_memory_peak_mb,
                        "calibration_completed": self.calibration_completed,
                    },
                )

                return False, reason

        # Use real measured peak memory for spawning decisions
        memory_per_worker = self.worker_memory_peak_mb
        memory_source = "measured"

        # Check if we have measured memory yet
        if memory_per_worker is None:
            if self.skip_memory_calibration_wait:
                # Use a heuristic estimate when skipping calibration wait
                # Estimate: batch_size * target_resolution^2 * num_channels * 4 bytes (float32) * 2.5 safety factor
                # Plus average FITS set size if available
                cutout_memory_mb = (
                    self.batch_size * self.target_resolution**2 * self.num_channels * 4 * 2.5
                ) / (1024 * 1024)
                fits_memory_mb = (
                    self.avg_fits_set_size_mb if self.avg_fits_set_size_mb else 500
                )  # Default 500MB for FITS
                memory_per_worker = cutout_memory_mb + fits_memory_mb
                memory_source = "estimated (calibration skipped)"
                logger.warning(
                    f"Skip calibration enabled - using estimated worker memory: {memory_per_worker:.1f}MB "
                    f"(cutout: {cutout_memory_mb:.1f}MB + FITS: {fits_memory_mb:.1f}MB)"
                )
            else:
                # This shouldn't happen since we check for calibration above
                # but handle it safely to avoid NoneType comparison errors
                reason = "Worker memory peak not yet measured (unexpected state)"
                logger.info(f"LoadBalancer spawn decision: NO - {reason}")
                return False, reason

        # Calculate memory requirement for one new worker using real measurements
        effective_memory = memory_per_worker

        remaining_memory = self._get_remaining_worker_memory()

        if remaining_memory < effective_memory:
            reason = (
                f"Insufficient worker memory: "
                f"remaining={remaining_memory:.1f}MB < "
                f"required={effective_memory:.1f}MB ({memory_source})"
            )
            logger.info(f"LoadBalancer spawn decision: NO - {reason}")

            # Log spawn decision event
            self._log_event(
                "Spawn",
                "decision",
                {
                    "can_spawn": False,
                    "reason": reason,
                    "active_processes": active_process_count,
                    "remaining_memory_mb": remaining_memory,
                    "required_memory_mb": effective_memory,
                    "memory_source": memory_source,
                },
            )

            return False, reason

        # Check system load
        resources = self.system_monitor.get_system_resources()
        cpu_percent = resources["cpu_percent"]
        if cpu_percent > 90:
            reason = f"CPU usage too high: {cpu_percent:.1f}%"
            logger.info(f"LoadBalancer spawn decision: NO - {reason}")

            # Log spawn decision event
            self._log_event(
                "Spawn",
                "decision",
                {
                    "can_spawn": False,
                    "reason": reason,
                    "active_processes": active_process_count,
                    "cpu_percent": cpu_percent,
                },
            )

            return False, reason

        reason = (
            f"Resources available: "
            f"remaining={remaining_memory:.1f}MB, "
            f"required={effective_memory:.1f}MB ({memory_source})"
        )
        logger.info(f"LoadBalancer spawn decision: YES - {reason}")

        # Log spawn decision event
        self._log_event(
            "Spawn",
            "decision",
            {
                "can_spawn": True,
                "reason": reason,
                "active_processes": active_process_count,
                "remaining_memory_mb": remaining_memory,
                "required_memory_mb": effective_memory,
                "memory_source": memory_source,
                "cpu_percent": cpu_percent,
            },
        )

        return True, reason

    def get_spawn_recommendation(
        self, active_processes: Dict[str, Any], pending_batches: int
    ) -> Dict[str, Any]:
        """
        Get recommendation for process spawning based on current state.

        Args:
            active_processes: Dictionary of currently active processes
            pending_batches: Number of batches still to process

        Returns:
            Dictionary with spawn recommendation and reasoning
        """
        active_count = len(active_processes)

        # No need for more processes if no pending work
        if pending_batches == 0:
            return {
                "spawn_new": False,
                "reason": "No pending batches",
                "active_processes": active_count,
                "memory_stats": self._get_memory_stats(),
            }

        # Check if we can spawn a new process
        active_process_ids = list(active_processes.keys())
        can_spawn, reason = self.can_spawn_new_process(active_count, active_process_ids)

        # Get current resource usage
        resources = self.system_monitor.get_system_resources()

        recommendation = {
            "spawn_new": can_spawn,
            "reason": reason,
            "active_processes": active_count,
            "pending_batches": pending_batches,
            "system_resources": {
                "cpu_percent": resources["cpu_percent"],
                "memory_available_gb": resources["memory_available"] / (1024**3),
                "memory_percent": resources["memory_percent"],
            },
            "memory_stats": self._get_memory_stats(),
        }

        return recommendation

    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return {
            "main_process_mb": self.main_process_memory_mb,
            "worker_allocation_mb": self.worker_memory_allocation_mb,
            "worker_peak_mb": self.worker_memory_peak_mb,
            "remaining_mb": self._get_remaining_worker_memory(),
            "processes_measured": self.processes_measured,
        }

    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get comprehensive resource status information for UI display.

        Returns:
            Dictionary with current resource status
        """
        try:
            resources = self.system_monitor.get_system_resources()
            cpu_count = self.system_monitor.get_cpu_count()

            # Get Kubernetes CPU limit if available for UI display
            memory_limit, cpu_limit_millicores = self.system_monitor._get_kubernetes_pod_limits()
            effective_cpu_count = cpu_count
            if cpu_limit_millicores is not None:
                effective_cpu_count = max(1, cpu_limit_millicores // 1000)
        except Exception as e:
            logger.error(f"LoadBalancer: Failed to get system resources: {e}")
            resources = {
                "cpu_percent": 0.0,
                "memory_total": 8 * 1024**3,
                "memory_available": 4 * 1024**3,
                "memory_percent": 50.0,
                "resource_source": "fallback",
            }
            cpu_count = 4
            effective_cpu_count = 4

        memory_stats = self._get_memory_stats()

        status = {
            "system": {
                "cpu_count": effective_cpu_count,  # Use effective CPU count for UI display
                "cpu_percent": float(resources.get("cpu_percent", 0.0)),
                "memory_total_gb": float(resources.get("memory_total", 0)) / (1024**3),
                "memory_available_gb": float(resources.get("memory_available", 0)) / (1024**3),
                "memory_percent": float(resources.get("memory_percent", 0.0)),
                "resource_source": resources.get("resource_source", "system"),
            },
            "limits": {
                "cpu_limit": self.cpu_limit if self.cpu_limit else effective_cpu_count - 1,
                "memory_limit_gb": (
                    float(self.memory_limit_bytes) / (1024**3)
                    if self.memory_limit_bytes
                    else float(resources.get("memory_available", 0))
                    * (1 - self.memory_safety_margin)
                    / (1024**3)
                ),
                "safety_margin": self.memory_safety_margin,
            },
            "performance": {
                "processes_measured": memory_stats["processes_measured"],
                "calibration_completed": self.calibration_completed,
                "main_process_memory_mb": (
                    float(memory_stats["main_process_mb"])
                    if memory_stats["main_process_mb"]
                    else 0.0
                ),
                "worker_allocation_mb": (
                    float(memory_stats["worker_allocation_mb"])
                    if memory_stats["worker_allocation_mb"]
                    else 0.0
                ),
                "worker_peak_mb": (
                    float(memory_stats["worker_peak_mb"]) if memory_stats["worker_peak_mb"] else 0.0
                ),
                "worker_remaining_mb": (
                    float(memory_stats["remaining_mb"]) if memory_stats["remaining_mb"] else 0.0
                ),
                # Legacy fields for compatibility
                "avg_memory_mb": (
                    float(memory_stats["worker_peak_mb"]) if memory_stats["worker_peak_mb"] else 0.0
                ),
                "peak_memory_mb": (
                    float(memory_stats["worker_peak_mb"]) if memory_stats["worker_peak_mb"] else 0.0
                ),
                "memory_samples": len(self.worker_memory_history),
            },
        }

        return status
