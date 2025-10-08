#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Logging configuration for Cutana.

Configures loguru with rotation and proper formatting.
"""

import sys
from pathlib import Path
from loguru import logger
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    colorize: bool = True,
    console_level: str = "WARNING",
    session_timestamp: str = None,
) -> None:
    """
    Configure logging for Cutana with dual-level logging.

    Args:
        log_level: Logging level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
        log_dir: Directory for log files
        colorize: Whether to enable colorized output
        console_level: Logging level for console/notebook output (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
        session_timestamp: Shared timestamp for consistent naming across processes (optional)
    """
    # Remove default handler
    logger.remove()

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Add console handler with higher threshold (WARN/ERROR only for notebooks)
    logger.add(
        sys.stderr,
        level=console_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        colorize=colorize,
    )

    # Create timestamped log filename to avoid collisions between tests/processes
    if session_timestamp is None:
        # Fallback to millisecond timestamp if no session timestamp provided
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds

    log_filename = f"cutana_{session_timestamp}.log"

    # Add file handler with rotation (captures INFO level and above)
    logger.add(
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

    logger.info(
        "Logging configured successfully - console shows {}, files capture {}",
        console_level,
        log_level,
    )


def cleanup_logging():
    """
    Clean up loguru handlers to release file locks.

    This is particularly important on Windows where open file handles
    prevent directory deletion during test cleanup.
    """
    import time

    # Remove all handlers to close file locks
    logger.remove()

    # Give a moment for background threads to finish (enqueue=True)
    # This is needed on Windows to ensure file handles are properly closed
    time.sleep(0.1)  # Increased wait time for Windows
