#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Deployment validation for Cutana astronomical cutout pipeline.

This module provides comprehensive validation that a Cutana installation is working correctly,
including dependency checks, configuration validation, and a minimal end-to-end test.
"""

import sys
import os
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List
import yaml

try:
    from loguru import logger
except ImportError:
    print("ERROR: loguru not installed. Please install cutana dependencies.")
    sys.exit(1)


class DeploymentValidator:
    """Validates Cutana deployment and installation."""

    def __init__(self, verbose: bool = True):
        """Initialize deployment validator.

        Args:
            verbose: Enable detailed logging output
        """
        self.verbose = verbose
        self.results = {}
        self.temp_dirs = []
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for validation."""
        from cutana.logging_config import setup_logging

        # Use consistent logging configuration from cutana
        # Set console level to DEBUG for verbose mode, INFO otherwise
        console_level = "DEBUG" if self.verbose else "INFO"

        # Create temporary log dir for validation
        log_dir = Path(tempfile.gettempdir()) / "cutana_validation_logs"
        log_dir.mkdir(exist_ok=True)
        self.temp_dirs.append(str(log_dir))

        setup_logging(
            log_level="DEBUG",  # File level
            log_dir=str(log_dir),
            colorize=sys.platform != "win32",  # Disable colors on Windows
            console_level=console_level,
        )

    def validate_conda_environment(self) -> bool:
        """Check if running in the correct conda environment.

        Returns:
            bool: True if in 'cutana' conda environment
        """
        logger.info("[CHECK] Checking conda environment...")

        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        if conda_env == "cutana":
            logger.success(f"[PASS] Running in correct conda environment: {conda_env}")
            return True
        else:
            logger.warning(f"[WARN] Not in 'cutana' environment (current: '{conda_env or 'none'}')")
            logger.info("   Run 'conda activate cutana' before using Cutana")
            return False

    def _get_dependencies_from_config(self) -> List[str]:
        """Extract dependencies from environment.yml and pyproject.toml.

        Returns:
            List of unique dependency names
        """
        deps = set()

        # Try to read from pyproject.toml
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            try:
                import toml

                with open(pyproject_path, "r") as f:
                    pyproject = toml.load(f)

                # Get main dependencies
                if "project" in pyproject and "dependencies" in pyproject["project"]:
                    for dep in pyproject["project"]["dependencies"]:
                        # Parse package name from requirement string (e.g., "numpy>=1.20" -> "numpy")
                        dep_name = (
                            dep.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].strip()
                        )
                        deps.add(dep_name)

                logger.debug(f"  Found {len(deps)} dependencies in pyproject.toml")
            except Exception as e:
                logger.warning(f"  Could not read pyproject.toml: {e}")

        # Try to read from environment.yml
        env_path = Path(__file__).parent.parent / "environment.yml"
        if env_path.exists():
            try:
                with open(env_path, "r") as f:
                    env_config = yaml.safe_load(f)

                if "dependencies" in env_config:
                    for dep in env_config["dependencies"]:
                        if isinstance(dep, str) and not dep.startswith("python"):
                            # Parse conda package name
                            dep_name = (
                                dep.split("=")[0].split(">=")[0].split("<")[0].split(">")[0].strip()
                            )
                            # Skip non-Python packages
                            if dep_name not in ["pip", "black", "flake8"]:
                                deps.add(dep_name)
                        elif isinstance(dep, dict) and "pip" in dep:
                            # Parse pip dependencies
                            for pip_dep in dep["pip"]:
                                dep_name = (
                                    pip_dep.split(">=")[0]
                                    .split("==")[0]
                                    .split("<")[0]
                                    .split(">")[0]
                                    .strip()
                                )
                                # Convert package names with hyphens to underscores for import
                                import_name = dep_name.replace("-", "_")
                                deps.add(import_name)

                logger.debug(f"  Found {len(deps)} total unique dependencies")
            except Exception as e:
                logger.warning(f"  Could not read environment.yml: {e}")

        return list(deps)

    def validate_dependencies(self) -> bool:
        """Check all required dependencies are available.

        Returns:
            bool: True if all dependencies are importable
        """
        logger.info("[CHECK] Checking dependencies...")

        # Get dependencies dynamically
        dynamic_deps = self._get_dependencies_from_config()

        # Critical dependencies that must be checked (with import aliases)
        critical_deps = [
            ("numpy", "np"),
            ("pandas", "pd"),
            ("astropy.io", "fits"),
            ("astropy.wcs", "WCS"),
            ("zarr", "zarr"),
            ("dotmap", "DotMap"),
            ("loguru", "logger"),
            ("PIL", "Image"),
            ("fitsbolt", None),
            ("images_to_zarr", None),
        ]

        # Additional dynamic dependencies to check (simple imports)
        # Create a set of already included packages to avoid duplicates
        included_packages = set()
        for module_name, _ in critical_deps:
            # Get base package name
            base_name = module_name.split(".")[0] if "." in module_name else module_name
            included_packages.add(base_name)

        for dep in dynamic_deps:
            # Skip pytest plugins and packages with special characters
            if dep.startswith("pytest") or "-" in dep:
                continue
            # Skip if already in critical deps
            if dep not in included_packages:
                critical_deps.append((dep, None))
                included_packages.add(dep)

        missing = []
        checked = 0
        for module_name, alias in critical_deps:
            try:
                if "." in module_name:
                    parts = module_name.split(".")
                    exec(f"from {'.'.join(parts[:-1])} import {parts[-1]}")
                else:
                    exec(f"import {module_name}" + (f" as {alias}" if alias else ""))
                logger.debug(f"  [OK] {module_name}")
                checked += 1
            except ImportError as e:
                # Some packages have different import names
                if module_name in [
                    "pytest_cov",
                    "pytest_mock",
                    "pytest_asyncio",
                    "pytest_playwright",
                ]:
                    # These are pytest plugins, not directly importable
                    logger.debug(f"  [SKIP] {module_name} (pytest plugin)")
                elif module_name in ["ipyfilechooser", "memory_profiler"]:
                    # Optional dependencies
                    logger.debug(f"  [SKIP] {module_name} (optional)")
                else:
                    logger.error(f"  [FAIL] {module_name}: {e}")
                    missing.append(module_name)

        logger.info(f"  Checked {checked} dependencies")

        if missing:
            logger.error(f"[FAIL] Missing dependencies: {', '.join(missing)}")
            return False
        else:
            logger.success("[PASS] All dependencies available")
            return True

    def validate_configuration(self) -> bool:
        """Test configuration loading and validation.

        Returns:
            bool: True if configuration system works
        """
        logger.info("[CHECK] Checking configuration system...")

        try:
            from cutana.get_default_config import get_default_config
            from cutana.validate_config import validate_config

            # Load default config
            config = get_default_config()
            logger.debug("  [OK] Default config loaded")

            # Set required fields for validation
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(b"SourceID,RA,Dec,diameter_pixel,fits_file_paths\n")
                tmp.write(b"TEST,150.0,2.0,64,\"['test.fits']\"\n")
                temp_catalogue = tmp.name

            config.source_catalogue = temp_catalogue
            config.output_dir = tempfile.mkdtemp(prefix="cutana_validate_")
            self.temp_dirs.append(config.output_dir)
            config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]

            # Validate config (skip path checks since we're using temp files)
            validate_config(config, check_paths=False)
            logger.debug("  [OK] Config validation passed")

            # Clean up temp catalogue
            os.unlink(temp_catalogue)

            logger.success("[PASS] Configuration system working")
            return True

        except Exception as e:
            logger.error(f"[FAIL] Configuration error: {e}")
            return False

    def run_minimal_e2e_test(self) -> bool:
        """Run a minimal end-to-end test.

        Returns:
            bool: True if e2e test passes
        """
        logger.info("[CHECK] Running minimal end-to-end test...")

        temp_data_dir = None
        temp_output_dir = None

        try:
            # Import required modules
            import numpy as np
            import pandas as pd
            from astropy.io import fits
            from astropy.wcs import WCS
            import zarr
            from cutana import get_default_config, Orchestrator

            # Create temporary directories
            temp_data_dir = Path(tempfile.mkdtemp(prefix="cutana_e2e_data_"))
            temp_output_dir = Path(tempfile.mkdtemp(prefix="cutana_e2e_output_"))
            self.temp_dirs.extend([str(temp_data_dir), str(temp_output_dir)])

            logger.debug(f"  Using temp dirs: {temp_data_dir}, {temp_output_dir}")

            # Create minimal synthetic FITS file
            logger.debug("  Creating test FITS file...")
            image_size = 200
            test_data = np.random.randn(image_size, image_size).astype(np.float32) * 100 + 500

            # Add a bright source at center for validation
            center = image_size // 2
            test_data[center - 5 : center + 5, center - 5 : center + 5] = 1000

            # Create WCS
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [image_size / 2, image_size / 2]
            wcs.wcs.cdelt = [-0.0002777778, 0.0002777778]  # ~1 arcsec/pixel
            wcs.wcs.crval = [150.0, 2.0]
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

            # Save FITS
            fits_path = temp_data_dir / "test.fits"
            hdu = fits.PrimaryHDU(data=test_data, header=wcs.to_header())
            hdu.header["MAGZERO"] = 25.0
            hdu.writeto(fits_path, overwrite=True)
            logger.debug(f"  [OK] Created FITS: {fits_path}")

            # Create minimal catalogue
            catalogue_path = temp_data_dir / "test_catalogue.csv"
            catalogue_data = pd.DataFrame(
                [
                    {
                        "SourceID": "TEST_001",
                        "RA": 150.0,
                        "Dec": 2.0,
                        "diameter_pixel": 32,
                        "fits_file_paths": str([str(fits_path)]),
                    }
                ]
            )
            catalogue_data.to_csv(catalogue_path, index=False)
            logger.debug(f"  [OK] Created catalogue: {catalogue_path}")

            # Configure and run Cutana
            logger.debug("  Running Cutana orchestrator...")
            config = get_default_config()
            config.source_catalogue = str(catalogue_path)
            config.output_dir = str(temp_output_dir)
            config.output_format = "zarr"
            config.data_type = "float32"
            config.target_resolution = 32
            config.normalisation_method = "linear"
            config.max_workers = 1
            config.log_level = "ERROR"  # Reduce noise during test
            config.console_log_level = "ERROR"  # Suppress warnings during test
            config.N_batch_cutout_process = 10
            config.loadbalancer.max_sources_per_process = 10
            config.selected_extensions = [{"name": "PRIMARY", "ext": "PRIMARY"}]
            config.apply_flux_conversion = False

            # Run processing
            orchestrator = Orchestrator(config)
            result = orchestrator.start_processing(catalogue_data)

            if result.get("status") != "completed":
                logger.error(f"  Processing failed: {result}")
                return False

            logger.debug("  [OK] Processing completed")

            # Validate output
            zarr_files = list(temp_output_dir.glob("**/images.zarr"))
            if not zarr_files:
                logger.error("  No zarr files generated")
                return False

            logger.debug(f"  [OK] Found zarr file: {zarr_files[0]}")

            # Check zarr contents
            zarr_store = zarr.open(str(zarr_files[0]), mode="r")
            images = zarr_store["images"]

            # Validate shape and values
            if images.shape != (1, 32, 32, 1):  # (N, H, W, C)
                logger.error(f"  Unexpected shape: {images.shape}")
                return False

            cutout = images[0, :, :, 0]
            if not np.all(np.isfinite(cutout)):
                logger.error("  Cutout contains non-finite values")
                return False

            # Check that center is brighter (where we placed the source)
            center_region = cutout[14:18, 14:18]
            edge_mean = np.mean(cutout[0:4, 0:4])
            center_mean = np.mean(center_region)

            if center_mean <= edge_mean:
                logger.error(
                    f"  Center not brighter: center={center_mean:.2f}, edge={edge_mean:.2f}"
                )
                return False

            logger.debug(
                f"  [OK] Zarr validation passed (center={center_mean:.2f}, edge={edge_mean:.2f})"
            )
            logger.success("[PASS] End-to-end test passed")
            return True

        except Exception as e:
            logger.error(f"[FAIL] End-to-end test failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

        finally:
            # Cleanup is handled in cleanup() method
            pass

    def check_git_access(self) -> bool:
        """Check if git is configured and has read access to the repository.

        Returns:
            bool: True if git access is working
        """
        logger.info("[CHECK] Checking git access...")

        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                logger.warning("  Not in a git repository")
                return False

            # Check remote URL (read-only operation)
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                remote_url = result.stdout.strip()
                logger.debug(f"  Remote URL: {remote_url}")

                # Check if it's the Cutana repository
                if "Cutana" in remote_url or "cutana" in remote_url:
                    logger.debug("  [OK] Cutana repository detected")
                else:
                    logger.warning(f"  Repository may not be Cutana: {remote_url}")

            # Test fetch (read-only, makes no changes)
            result = subprocess.run(
                ["git", "fetch", "--dry-run"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                logger.success("[PASS] Git access working")
                return True
            else:
                logger.warning(f"[WARN] Git fetch test failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("[FAIL] Git command timed out")
            return False
        except FileNotFoundError:
            logger.error("[FAIL] Git not found in PATH")
            return False
        except Exception as e:
            logger.error(f"[FAIL] Git check failed: {e}")
            return False

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"  Cleaned up: {temp_dir}")
                except Exception as e:
                    logger.warning(f"  Could not clean {temp_dir}: {e}")

    def run_all_validations(self) -> Dict:
        """Run all validation checks.

        Returns:
            Dict: Results of all validation checks
        """
        logger.info("=" * 60)
        logger.info("[START] Starting Cutana Deployment Validation")
        logger.info("=" * 60)

        start_time = time.time()

        # Run all checks
        self.results["conda_environment"] = self.validate_conda_environment()
        self.results["dependencies"] = self.validate_dependencies()
        self.results["configuration"] = self.validate_configuration()
        self.results["end_to_end"] = self.run_minimal_e2e_test()
        self.results["git_access"] = self.check_git_access()

        # Cleanup
        self.cleanup()

        # Calculate summary
        elapsed = time.time() - start_time
        total_checks = len(self.results)
        passed_checks = sum(1 for v in self.results.values() if v)

        # Print summary with nice table format - use print to ensure it always shows
        print("\n" + "=" * 70)
        print(" " * 20 + "[SUMMARY] Validation Results")
        print("=" * 70)

        # Create a nice table format
        print("  {:<30} {:<10} {:<20}".format("Check", "Status", "Result"))
        print("  " + "-" * 65)

        for check, passed in self.results.items():
            check_name = check.replace("_", " ").title()
            if passed:
                if sys.platform == "win32":
                    status = "[PASS]"
                    symbol = "[OK]"
                else:
                    status = "\033[92m[PASS]\033[0m"  # Green text
                    symbol = "\033[92m✓\033[0m"
            else:
                if sys.platform == "win32":
                    status = "[FAIL]"
                    symbol = "[X]"
                else:
                    status = "\033[91m[FAIL]\033[0m"  # Red text
                    symbol = "\033[91m✗\033[0m"

            print("  {:<30} {:<10} {:<20}".format(check_name, symbol, status))

        print("  " + "-" * 65)
        print("")

        # Overall result with clear summary
        if passed_checks == total_checks:
            if sys.platform == "win32":
                print(f"  [SUCCESS] All {total_checks}/{total_checks} checks PASSED!")
            else:
                print(
                    f"  \033[92m🎉 SUCCESS: All {total_checks}/{total_checks} checks PASSED!\033[0m"
                )
            print(f"  Completed in {elapsed:.1f} seconds")
            print("")
            if sys.platform == "win32":
                print("  [READY] Cutana is ready to use!")
            else:
                print("  ✅ Cutana is ready to use!")
        else:
            if sys.platform == "win32":
                print(f"  [ISSUES FOUND] {passed_checks}/{total_checks} checks passed")
            else:
                print(
                    f"  \033[93m⚠️  ISSUES FOUND: {passed_checks}/{total_checks} checks passed\033[0m"
                )
            print(f"  Completed in {elapsed:.1f} seconds")
            print("")
            print("  Troubleshooting tips:")
            if not self.results.get("conda_environment", False):
                print("    - Run 'conda activate cutana' and try again")
            if not self.results.get("dependencies", False):
                print("    - Run 'conda env create -f environment.yml' to install dependencies")
            if not self.results.get("git_access", False):
                print("    - Check your git configuration and network access")

        print("=" * 70)

        # Also log a simple summary for log files
        logger.info(
            f"Deployment validation completed: {passed_checks}/{total_checks} checks passed"
        )

        return self.results


def deployment_validation(verbose: bool = True) -> Dict:
    """Run deployment validation for Cutana.

    This function performs a comprehensive check of the Cutana installation,
    including dependency verification, configuration testing, and a minimal
    end-to-end test.

    Args:
        verbose: Enable detailed logging output

    Returns:
        Dict: Results of validation checks with boolean values

    Example:
        >>> import cutana
        >>> results = cutana.deployment_validation()
        >>> if all(results.values()):
        ...     print("Cutana is ready!")
    """
    validator = DeploymentValidator(verbose=verbose)
    return validator.run_all_validations()


if __name__ == "__main__":
    # Allow running as a script for testing
    import argparse

    parser = argparse.ArgumentParser(description="Validate Cutana deployment")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    results = deployment_validation(verbose=not args.quiet)

    # Exit with error code if any check failed
    sys.exit(0 if all(results.values()) else 1)
