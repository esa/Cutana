#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Interface to cutana backend module."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotmap import DotMap
from loguru import logger

# Import cutana modules - we can presume they exist
from cutana.catalogue_preprocessor import CatalogueValidationError, analyse_source_catalogue
from cutana.orchestrator import Orchestrator
from cutana.preview_generator import (
    generate_previews,
    load_sources_for_previews,
    regenerate_preview_seed,
)
from cutana.progress_report import ProgressReport
from cutana.validate_config import validate_config


class BackendInterface:
    """Interface for communication with cutana backend."""

    # Class variable to store current orchestrator for progress monitoring
    _current_orchestrator = None

    @staticmethod
    async def check_source_catalogue(filepath: str) -> Dict:
        """
        Analyze source catalogue file and return summary information.
        Called by the start_screem

        Returns:
            dict: Contains 'num_sources', 'fits_files', 'extensions'
        """
        # Simulate some processing time for UI feedback
        await asyncio.sleep(0.1)

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Analysing source catalogue: {filepath}")

        # Run analysis in a thread pool to avoid blocking the event loop
        logger.debug("Starting catalogue analysis in thread pool")
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(None, analyse_source_catalogue, str(filepath))
            logger.info(
                f"Catalogue analysis completed: {result['num_sources']} sources, {result['num_unique_fits_files']} FITS files"
            )
            return result
        except CatalogueValidationError as e:
            logger.error(f"Catalogue validation failed: {e}")
            # Re-raise with clear error type for UI handling
            raise ValueError(f"Catalogue validation failed: {e}") from e
        except Exception as e:
            logger.error(f"Error during catalogue analysis: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    @staticmethod
    async def start_processing(config: DotMap, status_panel=None) -> Dict:
        """
        Start the cutout processing with given configuration.

        Args:
            config: Configuration DotMap
            status_panel: Optional reference to UI status panel for direct updates

        This calls the real cutana orchestrator pipeline with optional direct UI updates.
        """
        # Only accept DotMap configs
        if not isinstance(config, DotMap):
            raise TypeError(f"Config must be DotMap, got {type(config)}")

        cutana_config = config

        logger.info(f"Starting cutout processing with {cutana_config.num_sources} sources")
        if status_panel:
            logger.info("Status panel provided for direct UI updates")

        try:
            # Validate configuration
            validate_config(cutana_config, check_paths=False)

            # Verify catalogue path exists
            catalogue_path = cutana_config.source_catalogue
            if not catalogue_path:
                raise ValueError("No source catalogue specified in config")

            # Create orchestrator with validated config and optional status panel
            logger.info("BackendInterface: Creating orchestrator with status panel reference...")
            orchestrator = Orchestrator(cutana_config, status_panel=status_panel)
            logger.info("BackendInterface: Orchestrator created successfully")

            # Store orchestrator reference for progress monitoring (still needed for stop functionality)
            logger.info("BackendInterface: Storing orchestrator reference")
            BackendInterface._current_orchestrator = orchestrator

            # Run processing in executor using streaming catalogue loading
            logger.info("BackendInterface: Starting orchestrator processing in executor...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, orchestrator.run)

            logger.info("BackendInterface: Processing completed successfully")

            # Clear orchestrator reference since processing is complete
            logger.info("BackendInterface: Clearing orchestrator reference")
            BackendInterface._current_orchestrator = None

            return {"status": "success", "result": result}

        except Exception as e:
            logger.error(f"Error starting processing: {e}")
            # Clear orchestrator reference on error as well
            BackendInterface._current_orchestrator = None
            return {"status": "error", "error": str(e)}

    @staticmethod
    async def load_sources_for_previews(
        catalogue_path: str = None, config: DotMap = None, progress_callback=None
    ) -> Dict:
        """
        Load and cache sources for fast preview generation.

        Delegates to cutana.preview_generator.load_sources_for_previews().
        """
        # Only accept DotMap configs
        if config is not None and not isinstance(config, DotMap):
            raise TypeError(f"Config must be DotMap, got {type(config)}")

        # Get catalogue path from config if not provided
        if not catalogue_path and config:
            catalogue_path = config.source_catalogue

        return await load_sources_for_previews(catalogue_path, config, progress_callback)

    @staticmethod
    async def generate_previews(
        num_samples: int = 10, size: int = 256, config: DotMap = None, progress_callback=None
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Generate preview cutouts using cached sources and FITS data.

        Delegates to cutana.preview_generator.generate_previews().
        """
        # Only accept DotMap configs
        if config is not None and not isinstance(config, DotMap):
            raise TypeError(f"Config must be DotMap, got {type(config)}")

        return await generate_previews(num_samples, size, config, progress_callback)

    @staticmethod
    def regenerate_preview_seed() -> int:
        """
        Generate a new preview seed for fresh sample selection.

        Delegates to cutana.preview_generator.regenerate_preview_seed().
        """
        return regenerate_preview_seed()

    @staticmethod
    async def stop_processing() -> Dict[str, Any]:
        """
        Stop any active processing by terminating the orchestrator and its subprocesses.

        Returns:
            Dictionary containing stop operation results
        """
        try:
            if BackendInterface._current_orchestrator is None:
                return {
                    "status": "no_active_session",
                    "message": "No active processing session to stop",
                }

            logger.info("BackendInterface: Stopping active processing...")

            # Call orchestrator's stop method
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, BackendInterface._current_orchestrator.stop_processing
            )

            # Clear orchestrator reference since it's been stopped
            logger.info("BackendInterface: Clearing orchestrator reference after stop")
            BackendInterface._current_orchestrator = None

            logger.info(f"BackendInterface: Processing stopped successfully: {result}")
            return {"status": "success", "result": result}

        except Exception as e:
            logger.error(f"BackendInterface: Error stopping processing: {e}")
            # Clear orchestrator on error as well
            BackendInterface._current_orchestrator = None
            return {"status": "error", "error": f"Failed to stop processing: {str(e)}"}

    @staticmethod
    async def get_processing_status() -> Dict[str, Any]:
        """
        Get current processing status from the active orchestrator.

        Returns:
            Dictionary containing processing status information optimized for UI display
        """
        try:
            # No orchestrator = not processing
            if BackendInterface._current_orchestrator is None:
                return ProgressReport.empty().to_dict()

            # Get UI-optimized status directly from orchestrator
            logger.debug("BackendInterface: Getting progress status from Orchestrator")
            progress_report = BackendInterface._current_orchestrator.get_progress_for_ui()
            logger.debug(
                f"BackendInterface: Received orchestrator status - {progress_report.completed_sources}/{progress_report.total_sources} sources, {progress_report.active_processes}/{progress_report.max_workers} workers, Memory: {progress_report.memory_used_gb:.1f}/{progress_report.memory_total_gb:.1f}GB"
            )

            # Convert to dictionary for backward compatibility
            status_dict = progress_report.to_dict()

            logger.debug(
                f"BackendInterface: Returning status to UI - {status_dict['completed_sources']}/{status_dict['total_sources']} sources ({status_dict['progress_percent']:.1f}%), {status_dict['active_processes']}/{status_dict['max_workers']} workers, Memory: {status_dict['memory_used_gb']:.1f}/{status_dict['memory_total_gb']:.1f}GB, Processing: {status_dict['is_processing']}"
            )

            return status_dict

        except Exception as e:
            # If there's ANY error, just log it and return safe defaults
            logger.warning(f"BackendInterface: Error getting status from orchestrator: {e}")

            # If we have an orchestrator but can't get status, assume it's still running
            error_report = ProgressReport.empty()
            error_report.is_processing = True  # Safer to assume still processing
            error_dict = error_report.to_dict()
            error_dict["error"] = str(e)  # Include error for debugging
            return error_dict
