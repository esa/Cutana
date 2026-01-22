#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Logging configuration for Cutana.

Configures loguru with rotation and proper formatting.

This module follows loguru best practices for library logging:
- Cutana is disabled by default at import (in __init__.py)
- setup_logging() enables cutana and adds handlers for application contexts
- Does NOT call logger.remove() without tracking handler IDs
- Preserves user-added handlers
- Users can disable cutana logs with logger.disable("cutana")
- Users can re-enable cutana logs with logger.enable("cutana")
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Module-level storage for handler IDs added by cutana
# Used by setup_logging to remove only its own handlers on re-configuration
_cutana_handler_ids: List[int] = []

# Track if this is the first call to setup_logging in this process
_first_setup_done: bool = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    colorize: bool = True,
    console_level: str = "WARNING",
    session_timestamp: Optional[str] = None,
) -> None:
    """
    Configure logging for Cutana with dual-level logging.

    This function adds cutana's logging handlers WITHOUT removing any user-defined
    handlers. This follows loguru best practices for library logging.

    Args:
        log_level: Logging level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
        log_dir: Directory for log files
        colorize: Whether to enable colorized output
        console_level: Logging level for console/notebook output (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
        session_timestamp: Shared timestamp for consistent naming across processes (optional)
    """
    global _first_setup_done

    # Enable cutana logging (it's disabled by default in __init__.py)
    # This is the application context where we want logs to be active
    logger.enable("cutana")

    # Remove only cutana's previously added handlers (not user's handlers)
    # This allows multiple calls to setup_logging without accumulating handlers
    for handler_id in _cutana_handler_ids:
        try:
            logger.remove(handler_id)
        except ValueError:
            # Handler already removed, that's fine
            pass
    _cutana_handler_ids.clear()

    # Detect if we're in a subprocess context
    # Subprocesses are identified by having a session_timestamp (passed from orchestrator)
    is_subprocess = session_timestamp is not None

    # On first setup in a fresh process (like a subprocess), remove the default
    # stderr handler that loguru adds automatically. This prevents duplicate
    # console output. We detect "first setup" by checking if we've done this before.
    # Note: This only affects the default handler, not any user-added handlers.
    if not _first_setup_done:
        try:
            # Handler ID 0 is the default stderr handler loguru adds at import
            # Only remove it if this is a subprocess context (no user handlers expected)
            if is_subprocess:
                logger.remove(0)
        except ValueError:
            # Default handler already removed or doesn't exist
            pass
        _first_setup_done = True

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Add console handler only for non-subprocess contexts
    # In subprocesses, stderr is captured to files, so we skip the console handler
    # to avoid ANSI escape codes ending up in log files
    if not is_subprocess:
        console_handler_id = logger.add(
            sys.stderr,
            level=console_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
            colorize=colorize,
        )
        _cutana_handler_ids.append(console_handler_id)

    # Create timestamped log filename to avoid collisions between tests/processes
    if session_timestamp is None:
        # Fallback to millisecond timestamp if no session timestamp provided
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    log_filename = f"cutana_{session_timestamp}.log"

    # Add file handler with rotation (captures INFO level and above)
    file_handler_id = logger.add(
        f"{log_dir}/{log_filename}",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} - {message}",
        rotation="10 MB",  # Rotate when file reaches 10MB
        retention="7 days",  # Keep logs for 7 days
        compression="gz",  # Compress old logs
        backtrace=True,
        diagnose=True,
        colorize=False,  # Disable colors for file output
        enqueue=True,  # Use background thread for file operations (Windows-friendly)
    )
    _cutana_handler_ids.append(file_handler_id)

    logger.info(
        "Logging configured successfully - console shows {}, files capture {}",
        console_level,
        log_level,
    )
