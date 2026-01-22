#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Simplified status panel for the main screen."""

import ipywidgets as widgets
from loguru import logger

from ..styles import (
    BACKGROUND_DARK,
    BORDER_COLOR,
    ESA_BLUE_ACCENT,
    ESA_BLUE_GREY,
    SUCCESS_COLOR,
    TEXT_COLOR_LIGHT,
    scale_px,
)
from ..widgets.progress_bar import CutanaProgressBar


class StatusPanel(widgets.VBox):
    """Panel showing processing status with direct updates from orchestrator."""

    def __init__(self, config):
        self.config = config
        self.is_processing = False

        # UI Components
        self.progress_bar = CutanaProgressBar(value=0, max_value=100, description="Progress")
        self.stats_html = widgets.HTML()
        self.processing_indicator = widgets.HTML()
        self.ready_status = widgets.HTML()

        # Combined title and status header (title on left, status on right)
        self.title = widgets.HTML(
            value=f'<h2 style="color: {ESA_BLUE_ACCENT}; margin: 0; font-size: {scale_px(20)}px;">Processing Status</h2>'
        )

        self.status_indicators = widgets.VBox(
            children=[self.ready_status, self.processing_indicator],
            layout=widgets.Layout(width="auto", margin="0", height=f"auto"),
        )

        self.combined_header = widgets.HBox(
            children=[self.title, self.status_indicators],
            layout=widgets.Layout(
                justify_content="space-between",
                align_items="flex-start",
                margin=f"0 0 {scale_px(2)}px 0",
                height=f"auto",
                overflow="hidden",
            ),
        )

        # Progress container - now includes the combined header
        self.progress_container = widgets.VBox(
            children=[
                self.combined_header,
                self.progress_bar,
                self.stats_html,
            ],
            layout=widgets.Layout(
                margin="0",
                padding="0",
                height=f"{scale_px(120)}px",  # Constrain total height to fit in container
            ),
        )

        # Initialize display
        self._initialize_display()

        # Main container - fixed size, no flex dependencies
        super().__init__(
            children=[self.progress_container],
            layout=widgets.Layout(
                padding=f"{scale_px(8)}px",
                background=BACKGROUND_DARK,
                border_radius=f"{scale_px(10)}px",
                border=f"1px solid {BORDER_COLOR}",
                width="100%",
                height="100%",  # Fill the fixed height from main screen
                margin=f"{scale_px(3)}px 0 0 0",
                overflow="auto",  # Allow scrolling when content exceeds container
            ),
        )
        self.add_class("cutana-panel")

    def _initialize_display(self):
        """Initialize the default display state."""
        num_sources = self.config.num_sources
        workers = self.config.max_workers
        # Ready status - make text more compact and responsive
        self.ready_status.value = f'<p style="color: {TEXT_COLOR_LIGHT}; text-align: left; margin: 1px 0; font-size: {scale_px(12)}px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">Ready • {num_sources:,} sources • {workers} workers</p>'

        # Default stats - use new signature with worker allocation info
        self._update_stats_display(
            0, num_sources, 0.0, 0.0, 0.0, 0.0, 0, workers, 0.0, 0.0, 0.0, "--:--:--"
        )

        # Clear processing indicator
        self.processing_indicator.value = ""

    def start_processing(self):
        """Start processing - UI will be updated directly by orchestrator."""
        logger.info("StatusPanel: Starting processing")

        self.is_processing = True

        # Hide ready status
        self.ready_status.layout.display = "none"

        # Reset progress bar
        self.progress_bar.update(value=0, max_value=100)

        # Initialize stats for new processing
        self._update_stats_display(
            0,
            self.config.num_sources,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            self.config.max_workers,
            0.0,
            0.0,
            0.0,
            "--:--:--",
        )

        # Show initial processing message
        self.processing_indicator.value = self._create_spinner_html(
            f"Starting processing for {self.config.num_sources:,} sources...", TEXT_COLOR_LIGHT
        )

    def stop_processing(self):
        """Stop processing."""
        logger.info("StatusPanel: Stop processing requested")

        self.is_processing = False

        # Show ready status again
        self.ready_status.layout.display = "block"

        # Clear processing indicator
        self.processing_indicator.value = ""

    def receive_status_UI_update(self, progress_report):
        """
        Receive progress update directly from orchestrator.

        Args:
            progress_report: ProgressReport object from orchestrator
        """
        try:
            # Convert ProgressReport to dict if needed
            if hasattr(progress_report, "to_dict"):
                status = progress_report.to_dict()
            else:
                status = progress_report

            # Check if processing is complete
            if not status.get("is_processing", False):
                # Processing has completed
                if (
                    status.get("completed_sources", 0) == status.get("total_sources", 0)
                    and status.get("total_sources", 0) > 0
                ):
                    self._handle_completion()
                else:
                    self._handle_stopped()
            else:
                # Normal progress update
                self._handle_progress_update(status)

        except Exception as e:
            logger.error(f"StatusPanel: Error in receive_status_UI_update: {e}")
            import traceback

            logger.error(f"StatusPanel: Full traceback: {traceback.format_exc()}")

    def _handle_progress_update(self, status):
        """Handle progress update from orchestrator."""
        try:
            logger.debug(
                f"StatusPanel: Received progress update - {status.get('completed_sources', 0)}/{status.get('total_sources', 0)} sources, {status.get('active_processes', 0)}/{status.get('max_workers', 0)} workers, Memory: {status.get('memory_used_gb', 0):.1f}/{status.get('memory_total_gb', 0):.1f}GB, Peak RAM: {status.get('peak_worker_memory_mb', 0):.1f}MB, Avg RAM: {status.get('avg_worker_memory_mb', 0):.1f}MB"
            )

            # Extract metrics - backend should provide complete, valid data
            completed = status.get("completed_sources", 0)
            total = status.get("total_sources", self.config.num_sources)
            progress = status.get("progress_percent", 0)
            throughput = status.get("throughput", 0)

            # Enhanced memory information
            memory_used_gb = status.get("memory_used_gb", 0.0)
            memory_total_gb = status.get("memory_total_gb", 0.0)
            memory_pct = status.get("memory_percent", 0.0)

            # Worker information (use new improved fields)
            active_processes = status.get("active_processes", 0)
            max_workers = status.get("max_workers", 0)
            worker_allocation_mb = status.get("worker_memory_allocation_mb", 0.0)
            worker_peak_mb = status.get("worker_memory_peak_mb", 0.0)
            worker_remaining_mb = status.get("worker_memory_remaining_mb", 0.0)

            # Calculate ETA in HH:MM:SS format
            eta_seconds = status.get("eta_seconds")
            eta = self._format_eta_hhmmss(eta_seconds)

            logger.debug(f"StatusPanel: Progress update: {completed}/{total} ({progress:.1f}%)")

            # Update UI
            self.progress_bar.update(value=progress, max_value=100)
            self._update_stats_display(
                completed,
                total,
                throughput,
                memory_used_gb,
                memory_total_gb,
                memory_pct,
                active_processes,
                max_workers,
                worker_allocation_mb,
                worker_peak_mb,
                worker_remaining_mb,
                eta,
            )

            # Show spinner when waiting for first worker to complete sources (only during initial calibration)
            calibration_completed = status.get("calibration_completed")
            if active_processes > 0 and completed == 0 and not calibration_completed:
                self.processing_indicator.value = self._create_spinner_html(
                    "Waiting for first worker to complete sources for memory calibration...",
                    TEXT_COLOR_LIGHT,
                )
            else:
                # Clear processing indicator during active processing (info is in bottom stats now)
                self.processing_indicator.value = ""

        except Exception as e:
            logger.error(f"StatusPanel: Error handling progress update: {e}")
            import traceback

            logger.error(f"StatusPanel: Full traceback: {traceback.format_exc()}")

    def _handle_completion(self):
        """Handle completion event."""
        logger.info("StatusPanel: Processing completed")

        self.is_processing = False

        # Hide ready status when showing completion message
        self.ready_status.layout.display = "none"

        # Update to 100% completion - but keep the final stats visible instead of zeroing them out
        self.progress_bar.update(value=100, max_value=100)
        # DO NOT reset stats to zero - they should show final completion values
        # The last progress update should have the final correct statistics

        # Show completion message
        self.processing_indicator.value = f'<div style="color: {SUCCESS_COLOR}; margin: 3px 0; font-size: 11px; font-weight: 600;">✓ Processing completed successfully!</div>'

    def _handle_stopped(self):
        """Handle stopped event."""
        logger.info("StatusPanel: Processing stopped")

        self.is_processing = False

        # Show ready status again
        self.ready_status.layout.display = "block"

        # Clear processing indicator
        self.processing_indicator.value = ""

    def _format_eta_hhmmss(self, eta_seconds):
        """Format ETA in HH:MM:SS format."""
        if eta_seconds is None or eta_seconds <= 0:
            return "--:--:--"

        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _create_spinner_html(self, message: str, color: str = None) -> str:
        """Create HTML for spinner with message."""
        if color is None:
            color = TEXT_COLOR_LIGHT

        return f"""<div style="display: flex; align-items: center; margin: 3px 0; font-size: 11px; color: {color}; font-weight: 500;">
            <div style="width: 12px; height: 12px; border: 2px solid {color}; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 8px;"></div>
            {message}
        </div>
        <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>"""

    def _update_stats_display(
        self,
        completed,
        total,
        throughput,
        memory_used_gb,
        memory_total_gb,
        memory_pct,
        active_processes,
        max_workers,
        worker_allocation_mb,
        worker_peak_mb,
        worker_remaining_mb,
        eta,
    ):
        """Update the statistics HTML display with enhanced LoadBalancer information."""
        # Format the enhanced progress display with improved memory info
        # Target: "3,000 / 100,000 | 85.3 sources/sec | 14.5GB / 32GB (45.3%) RAM | 3/4 workers | 2.5GB/4.0GB worker alloc (1.5GB free) | ETA 00:09:15"

        # Build the progress text components
        progress_text = f"{completed:,} / {total:,}"
        throughput_text = f"{throughput:.1f} sources/sec" if throughput > 0 else "--"
        memory_text = (
            f"{memory_used_gb:.1f}GB / {memory_total_gb:.1f}GB ({memory_pct:.1f}%) RAM"
            if memory_total_gb > 0
            else "-- RAM"
        )
        # Show "X/Y workers" only when workers are active, otherwise just "Y workers"
        if active_processes > 0 and max_workers > 0:
            workers_text = f"{active_processes}/{max_workers} workers"
        elif max_workers > 0:
            workers_text = f"{max_workers} workers"
        elif active_processes > 0:
            workers_text = f"{active_processes} workers"
        else:
            workers_text = "-- workers"

        # Worker memory allocation info (show allocation, usage, and remaining)
        worker_memory_text = ""
        if worker_allocation_mb > 0:
            worker_alloc_gb = worker_allocation_mb / 1024
            if worker_peak_mb > 0:
                worker_used_gb = worker_peak_mb / 1024
                worker_free_gb = worker_remaining_mb / 1024
                worker_memory_text = f"{worker_used_gb:.1f}GB/{worker_alloc_gb:.1f}GB worker alloc ({worker_free_gb:.1f}GB free)"
            else:
                worker_memory_text = f"{worker_alloc_gb:.1f}GB worker alloc available"
        else:
            worker_memory_text = "-- worker alloc"

        eta_text = f"ETA {eta}"

        # Combine into single line with separators
        status_line = f"{progress_text} | {throughput_text} | {memory_text} | {workers_text} | {worker_memory_text} | {eta_text}"

        # Create more compact status display for smaller screens
        html = f"""
        <div style="background: {ESA_BLUE_GREY}; padding: {scale_px(4)}px {scale_px(6)}px; border-radius: 4px; border: 1px solid {BORDER_COLOR}; margin: 1px 0 0 0; overflow: hidden;">
            <div style="font-size: {scale_px(14)}px; font-weight: 500; color: {TEXT_COLOR_LIGHT}; text-align: center; font-family: monospace; line-height: 1.1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                {status_line}
            </div>
        </div>
        """
        self.stats_html.value = html
