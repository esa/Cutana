#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Cutana Playwright Test Runner

This script runs real end-to-end UI tests for the Cutana astronomical cutout pipeline
using Voila web application integration with the real backend.

Test Coverage:
- Real UI Tests: End-to-end tests with Voila-served Cutana UI web app
- Backend Integration: Tests actual cutana.orchestrator and cutana.catalogue_analyzer
- Configuration Matrix: Multi-channel processing setup
- Main Screen: Processing interface transition from start screen
- CSV Analysis: Real catalogue analysis with test data

Usage:
  python run_playwright_tests.py                    # Run all real e2e tests
  python run_playwright_tests.py --headless         # Run without browser UI (faster)
  python run_playwright_tests.py --test <test_name> # Run specific test

Test Suite Details:
- Real E2E Tests: Full Voila web app integration with real backend
- Test Data Integration: Uses actual CSV catalogues and FITS files
- Main Screen Workflow: Complete workflow from CSV selection to processing screen
"""

import os
import subprocess
import sys


def run_playwright_tests(headed=True, specific_test=None):
    """Run real Playwright e2e tests with Voila web app integration."""
    # Build the command
    cmd = ["pytest", "-v", "--tb=short"]

    if headed:
        cmd.append("--headed")

    if specific_test:
        cmd.append(specific_test)
    else:
        # Run all real UI tests from tests/playwright/
        cmd.append("tests/playwright/")

    # Print test info
    if specific_test:
        print(f"Running specific test: {specific_test}")
    else:
        print("Running real Cutana UI e2e tests with Voila web app")
        print("   - Voila server startup and UI loading")
        print("   - Real backend integration tests")
        print("   - End-to-end workflow with test data")
        print("   - CSV analysis and main screen transition")

    print(f"\nCommand: {' '.join(cmd)}")
    print("=" * 60)

    # Run the tests
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Cutana real e2e Playwright tests with Voila")
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode (no browser window)"
    )
    parser.add_argument("--test", help="Specific test to run (e.g., test_reach_main_screen)")

    args = parser.parse_args()

    # Check if we're in the cutana environment
    if "cutana" not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        print("Warning: Make sure you're in the cutana conda environment!")
        print("Run: conda activate cutana")
        print()

    exit_code = run_playwright_tests(headed=not args.headless, specific_test=args.test)

    sys.exit(exit_code)
