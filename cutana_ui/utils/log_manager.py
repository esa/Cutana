#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""UI logging manager for Cutana UI."""

import sys
from pathlib import Path

from loguru import logger

# Console log format used throughout
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


class UILogManager:
    """Manages UI logging handlers for loguru.

    This class tracks handler IDs to allow cleanup without affecting user handlers.
    The console handler ID is stored separately to allow dynamic log level changes.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handler_ids = []
            cls._instance._console_handler_id = None
            cls._instance._current_console_level = "WARNING"
            cls._instance._initialized = False
        return cls._instance

    def _init_console_handler(self, level="WARNING"):
        """Initialize console handler, removing loguru's default handler."""
        if self._initialized:
            return

        # Remove loguru's default handler (ID 0)
        try:
            logger.remove(0)
        except ValueError:
            pass  # Already removed

        # Enable cutana logging
        logger.enable("cutana")

        # Add our console handler
        self._console_handler_id = logger.add(
            sys.stderr,
            level=level,
            format=_CONSOLE_FORMAT,
            colorize=True,
        )
        self._handler_ids.append(self._console_handler_id)
        self._current_console_level = level
        self._initialized = True

    def setup(self, output_dir, session_timestamp=None, console_level=None):
        """Set up loguru logging for UI with output directory and separate UI files.

        This function follows loguru best practices:
        - Enables cutana logging (disabled by default at import)
        - Tracks handler IDs to allow cleanup without affecting user handlers

        Args:
            output_dir: Required output directory for logs. Must be provided.
            session_timestamp: Optional timestamp for log file naming consistency.
            console_level: Log level for console output. If None, keeps current level.
        """
        # Always ensure cutana logging is enabled (it may be disabled by cutana/__init__.py import)
        logger.enable("cutana")

        try:
            # Initialize console if not already done
            if not self._initialized:
                self._init_console_handler(console_level or "WARNING")
            elif console_level:
                # Update console level if specified
                self.set_console_level(console_level)

            # Use output directory for logs
            log_dir = Path(output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamp for UI log files
            if session_timestamp is None:
                from datetime import datetime

                session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # Add UI-specific log file
            file_handler_id = logger.add(
                str(log_dir / f"ui_{session_timestamp}.log"),
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                enqueue=True,
            )
            self._handler_ids.append(file_handler_id)

            # Add UI error log file
            error_handler_id = logger.add(
                str(log_dir / f"ui_errors_{session_timestamp}.log"),
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="5 MB",
                retention="30 days",
                compression="zip",
                enqueue=True,
            )
            self._handler_ids.append(error_handler_id)

            logger.debug(f"UI logging configured with directory: {log_dir}")

        except Exception as e:
            logger.warning(f"Error setting up file logging: {e}")

    def set_console_level(self, level: str):
        """Change the console log level dynamically.

        This allows users to adjust the verbosity of logs shown in the Jupyter notebook
        without affecting file logging.

        Args:
            level: Log level string (DEBUG, INFO, WARNING, ERROR).
        """
        # Always ensure cutana logging is enabled (it may be disabled by cutana/__init__.py import)
        logger.enable("cutana")

        # Initialize if not yet done
        if not self._initialized:
            self._init_console_handler(level)
            return

        # Don't change handler if already at this level (but enable was still done above)
        if self._current_console_level == level:
            return

        try:
            logger.remove(self._console_handler_id)
            self._handler_ids.remove(self._console_handler_id)
        except ValueError:
            pass

        self._console_handler_id = logger.add(
            sys.stderr,
            level=level,
            format=_CONSOLE_FORMAT,
            colorize=True,
        )
        self._handler_ids.append(self._console_handler_id)
        self._current_console_level = level


# Singleton instance for use by UI components
_log_manager = UILogManager()

# Initialize console handler immediately so log level dropdown works from the start
# This removes loguru's default handler and adds our controlled handler
_log_manager._init_console_handler("WARNING")


def setup_ui_logging(output_dir, session_timestamp=None, console_level=None):
    """Set up loguru logging for UI. Wrapper for UILogManager.setup()."""
    _log_manager.setup(output_dir, session_timestamp, console_level)


def set_console_log_level(level: str):
    """Change the console log level dynamically. Wrapper for UILogManager.set_console_level()."""
    _log_manager.set_console_level(level)


def get_console_log_level() -> str:
    """Get the current console log level. Returns capitalized level (e.g., 'Warning')."""
    return _log_manager._current_console_level.capitalize()
