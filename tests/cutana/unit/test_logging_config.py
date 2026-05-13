#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Tests for logging configuration in Cutana.

Tests that:
1. setup_logging does not interfere with user's logging handlers
2. Log files are created in the expected directories
3. Library logging follows loguru best practices
"""

import io
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger

import cutana.logging_config as logging_config
from cutana import Orchestrator, get_default_config
from cutana.logging_config import setup_logging


def _cleanup_cutana_handlers():
    """Test helper to clean up cutana's logging handlers."""
    for handler_id in logging_config._cutana_handler_ids:
        try:
            logger.remove(handler_id)
        except ValueError:
            pass
    logging_config._cutana_handler_ids.clear()
    time.sleep(0.1)


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Reset the logging module state before and after each test."""
    # Reset before test
    logging_config._cutana_handler_ids.clear()
    logging_config._first_setup_done = False

    yield

    # Cleanup after test
    _cleanup_cutana_handlers()
    logging_config._first_setup_done = False


class TestLoggingNonInterference:
    """Test that cutana logging does not interfere with user's logging setup."""

    def test_user_handler_preserved_after_setup_logging(self):
        """Test that user-added handlers are not removed by setup_logging."""
        # Store original handler count
        # Note: We need to get the handler IDs before and after to check

        # Create a custom string sink to capture user logs
        user_log_output = io.StringIO()

        # User adds their own handler BEFORE importing/calling cutana's setup_logging
        user_handler_id = logger.add(
            user_log_output,
            format="{message}",
            level="DEBUG",
        )

        # Log something to verify user handler works
        logger.info("User message before setup_logging")

        # Now import and call cutana's setup_logging
        with tempfile.TemporaryDirectory() as temp_dir:
            # Call setup_logging - this should NOT remove user's handler
            setup_logging(
                log_level="INFO",
                log_dir=temp_dir,
                console_level="WARNING",
            )

            # Log something after setup_logging
            logger.info("User message after setup_logging")

            # Clean up cutana handlers
            _cleanup_cutana_handlers()

        # Get what was logged to user's handler
        user_log_output.seek(0)
        logged_messages = user_log_output.read()

        # User's handler should have captured BOTH messages
        assert "User message before setup_logging" in logged_messages, (
            "User's handler should have captured message before setup_logging"
        )
        assert "User message after setup_logging" in logged_messages, (
            "User's handler should have captured message after setup_logging. "
            "This indicates setup_logging incorrectly removed user's handler."
        )

        # Cleanup user handler
        logger.remove(user_handler_id)

    def test_multiple_user_handlers_preserved(self):
        """Test that multiple user handlers are all preserved after setup_logging."""
        # Create multiple user handlers with different configurations
        user_output_1 = io.StringIO()
        user_output_2 = io.StringIO()
        user_output_3 = io.StringIO()

        # Add handlers with different levels/formats (simulating different use cases)
        handler_id_1 = logger.add(user_output_1, format="[H1] {message}", level="DEBUG")
        handler_id_2 = logger.add(user_output_2, format="[H2] {message}", level="INFO")
        handler_id_3 = logger.add(user_output_3, format="[H3] {message}", level="WARNING")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup cutana logging
            setup_logging(log_level="INFO", log_dir=temp_dir, console_level="ERROR")

            # Log messages at different levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")

            _cleanup_cutana_handlers()

        # Verify each handler captured appropriate messages
        user_output_1.seek(0)
        output_1 = user_output_1.read()
        assert "[H1] Debug message" in output_1, "Handler 1 (DEBUG) should capture debug"
        assert "[H1] Info message" in output_1, "Handler 1 (DEBUG) should capture info"
        assert "[H1] Warning message" in output_1, "Handler 1 (DEBUG) should capture warning"

        user_output_2.seek(0)
        output_2 = user_output_2.read()
        assert "Debug message" not in output_2, "Handler 2 (INFO) should not capture debug"
        assert "[H2] Info message" in output_2, "Handler 2 (INFO) should capture info"
        assert "[H2] Warning message" in output_2, "Handler 2 (INFO) should capture warning"

        user_output_3.seek(0)
        output_3 = user_output_3.read()
        assert "Debug message" not in output_3, "Handler 3 (WARNING) should not capture debug"
        assert "Info message" not in output_3, "Handler 3 (WARNING) should not capture info"
        assert "[H3] Warning message" in output_3, "Handler 3 (WARNING) should capture warning"

        # Cleanup
        logger.remove(handler_id_1)
        logger.remove(handler_id_2)
        logger.remove(handler_id_3)


class TestLogFileCreation:
    """Test that log files are created in the expected directories."""

    def test_log_file_created_in_output_dir(self):
        """Test that log file is created in the specified log directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            setup_logging(
                log_level="INFO",
                log_dir=str(log_dir),
                console_level="WARNING",
            )

            # Log a message to ensure file is written
            logger.info("Test log message for file creation")

            # Give the enqueued logging a moment to flush
            time.sleep(0.2)

            _cleanup_cutana_handlers()

            # Check that log directory was created
            assert log_dir.exists(), "Log directory should be created"

            # Check that at least one log file exists
            log_files = list(log_dir.glob("cutana_*.log"))
            assert len(log_files) >= 1, "At least one log file should be created"

            # Check that the log file contains our message
            with open(log_files[0], "r") as f:
                content = f.read()
            assert "Test log message for file creation" in content

    def test_session_timestamp_creates_consistent_filename(self):
        """Test that providing a session timestamp creates a consistent filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_timestamp = "20231215_120000_123"

            setup_logging(
                log_level="INFO",
                log_dir=temp_dir,
                session_timestamp=session_timestamp,
            )

            logger.info("Test message with session timestamp")

            time.sleep(0.2)

            _cleanup_cutana_handlers()

            expected_filename = f"cutana_{session_timestamp}.log"
            expected_path = Path(temp_dir) / expected_filename

            assert expected_path.exists(), (
                f"Log file with session timestamp should exist: {expected_filename}"
            )


class TestLibraryLoggingPattern:
    """Test that cutana follows library logging best practices.

    According to loguru docs:
    "To use Loguru from inside a library, remember to never call add()
    but use disable() instead so logging functions become no-op."

    These tests verify that cutana follows this pattern:
    - Importing cutana does NOT add handlers
    - Creating Orchestrator does NOT add handlers
    - Users must explicitly call setup_logging() to get handlers
    """

    def test_import_cutana_does_not_add_handlers(self):
        """Test that simply importing cutana does not add any handlers.

        This is critical for library best practices - importing a library
        should NEVER modify the global logger state.
        """
        # Get current handler count
        initial_handlers = set(logger._core.handlers.keys())

        # Import cutana (force reimport by removing from sys.modules if needed)
        import cutana  # noqa: PLC0415, F401 - verify import doesn't add handlers

        # Get handler count after import
        after_import_handlers = set(logger._core.handlers.keys())

        # No new handlers should have been added
        new_handlers = after_import_handlers - initial_handlers
        assert len(new_handlers) == 0, (
            f"Importing cutana should NOT add handlers. New handlers added: {new_handlers}"
        )

    def test_creating_orchestrator_does_not_add_handlers(self):
        """Test that creating an Orchestrator does not add any handlers.

        Users should be able to use cutana without having their logging
        configuration modified. This test would have caught the bug where
        Orchestrator.__init__ automatically called setup_logging().
        """
        # Get current handler count
        initial_handlers = set(logger._core.handlers.keys())

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a minimal config
            config = get_default_config()
            config.output_dir = temp_dir
            config.source_catalogue = "dummy.csv"  # Won't be accessed

            # Patch components that have side effects to isolate the test
            with patch("cutana.orchestrator.JobTracker"):
                with patch("cutana.orchestrator.LoadBalancer"):
                    # Create orchestrator - this should NOT add handlers
                    orchestrator = Orchestrator(config)

        # Get handler count after creating orchestrator
        after_orchestrator_handlers = set(logger._core.handlers.keys())

        # No new handlers should have been added
        new_handlers = after_orchestrator_handlers - initial_handlers
        assert len(new_handlers) == 0, (
            f"Creating Orchestrator should NOT add handlers. "
            f"New handlers added: {new_handlers}. "
            f"This violates loguru library best practices."
        )

    def test_setup_logging_is_opt_in_only(self):
        """Test that handlers are only added when setup_logging() is explicitly called.

        This verifies that cutana follows the opt-in pattern for logging:
        - By default: no handlers added, logs are silent
        - Explicit call to setup_logging(): handlers are added
        """
        # Get current handler count
        initial_handlers = set(logger._core.handlers.keys())

        with tempfile.TemporaryDirectory() as temp_dir:
            # Explicitly call setup_logging - NOW handlers should be added
            setup_logging(log_level="INFO", log_dir=temp_dir)

            after_setup_handlers = set(logger._core.handlers.keys())

            # Handlers should have been added
            new_handlers = after_setup_handlers - initial_handlers
            assert len(new_handlers) > 0, (
                "setup_logging() should add handlers when explicitly called"
            )

            _cleanup_cutana_handlers()

        # After cleanup, our handlers should be removed
        after_cleanup_handlers = set(logger._core.handlers.keys())
        assert after_cleanup_handlers == initial_handlers, (
            "Cleanup should remove only cutana's handlers"
        )
