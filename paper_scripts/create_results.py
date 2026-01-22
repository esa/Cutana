#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Master script to run all paper benchmarks.

Orchestrates execution of:
1. run_framework_comparison.py - Compare Astropy vs Cutana (1w and 4w)
2. run_memory_profile.py - Memory consumption analysis
3. run_scaling_study.py - Thread scaling analysis (1-8 workers)
4. create_latex_values.py - Generate LaTeX macros from results

Usage:
    python create_results.py --size small   # Use small catalogues (faster)
    python create_results.py --size big     # Use big catalogues (full benchmarks)
    python create_results.py --test         # Test mode: only 100k-1tile-4channel
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from cutana.logging_config import setup_logging  # noqa: E402


def run_script(script_name: str, script_path: Path, extra_args: list = None) -> bool:
    """
    Run a Python script and handle output.

    Args:
        script_name: Name of script for logging
        script_path: Path to script
        extra_args: Additional command-line arguments

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {script_name}")
    logger.info(f"{'='*80}\n")

    start_time = time.time()

    try:
        # Build command with extra arguments
        cmd = [sys.executable, str(script_path)]
        if extra_args:
            cmd.extend(extra_args)

        # Run script using subprocess WITHOUT capturing output
        # This allows real-time progress to be shown in terminal
        result = subprocess.run(
            cmd,
            # Don't capture output - let it stream to terminal
            timeout=7200,  # 2 hour timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✓ {script_name} completed successfully in {elapsed:.1f}s")
            return True
        else:
            logger.error(f"✗ {script_name} failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"✗ {script_name} timed out after 2 hours")
        return False
    except Exception as e:
        logger.error(f"✗ {script_name} failed with exception: {e}")
        logger.error("Exception details:", exc_info=True)
        return False


def check_prerequisites(script_dir: Path, catalogue_size: str, test_mode: bool) -> bool:
    """
    Check that all required files and directories exist.

    Args:
        script_dir: Path to paper_scripts directory
        catalogue_size: 'small' or 'big'
        test_mode: If True, only check for test catalogue

    Returns:
        True if all prerequisites met
    """
    logger.info("Checking prerequisites...")

    if test_mode:
        # Test mode: check for 12k test catalogue
        test_catalogue = script_dir / "catalogues" / "test" / "12k-1tile-4channel.csv"
        if not test_catalogue.exists():
            logger.error(f"Test catalogue not found: {test_catalogue}")
            return False
        logger.info(f"✓ Test catalogue found: {test_catalogue}")
    else:
        # Full mode: check for size-specific catalogues
        catalogues_dir = script_dir / "catalogues" / catalogue_size

        if not catalogues_dir.exists():
            logger.error(f"Catalogues directory not found: {catalogues_dir}")
            logger.error(f"Please create catalogues in {catalogues_dir}/")
            return False

        # Check for all catalogues (size-specific)
        if catalogue_size == "small":
            required_catalogues = [
                "50k-1tile-4channel.csv",
                "1k-8tiles-4channel.csv",
                "50k-4tiles-1channel.csv",
            ]
        else:  # big
            required_catalogues = [
                "200k-8tile-1channel.csv",
                "1k-32tiles-4channel.csv",
                "100k-4tiles-1channel.csv",
            ]

        for catalogue in required_catalogues:
            catalogue_path = catalogues_dir / catalogue
            if not catalogue_path.exists():
                logger.warning(f"Catalogue not found: {catalogue_path}")
                logger.warning(f"Some benchmarks may be skipped")
            else:
                logger.info(f"✓ Catalogue found: {catalogue}")

    # Check that script files exist
    required_scripts = [
        "run_framework_comparison.py",
        "run_memory_profile.py",
        "run_scaling_study.py",
        "create_latex_values.py",
    ]

    for script in required_scripts:
        script_path = script_dir / script
        if not script_path.exists():
            logger.error(f"Required script not found: {script_path}")
            return False

    logger.info("✓ All required scripts found")
    logger.info("✓ Prerequisites check passed")
    return True


def main():
    """Main orchestration execution."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Cutana paper benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_results.py --size small          # Use small catalogues
  python create_results.py --size big            # Use big catalogues (full benchmarks)
  python create_results.py --test                # Test mode (12k test catalogue)
        """,
    )
    parser.add_argument(
        "--size",
        choices=["small", "big"],
        default="small",
        help="Catalogue size to use (default: small)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: use 12k test catalogue for faster iteration"
    )

    args = parser.parse_args()

    setup_logging(log_level="INFO", console_level="INFO")

    logger.info("=" * 80)
    logger.info("CUTANA PAPER BENCHMARKS - MASTER EXECUTION SCRIPT")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This will run all benchmarks for the Cutana paper:")
    logger.info("1. Framework comparison (Astropy vs Cutana)")
    logger.info("2. Memory profiling")
    logger.info("3. Thread scaling study")
    logger.info("4. Generate LaTeX values")
    logger.info("")
    logger.info(f"Mode: {'TEST' if args.test else 'FULL'}")
    logger.info(f"Catalogue size: {args.size}")
    logger.info(f"Using catalogues from: paper_scripts/catalogues/{args.size}/")
    logger.info("")

    script_dir = Path(__file__).parent

    # Check prerequisites
    if not check_prerequisites(script_dir, args.size, args.test):
        logger.error("Prerequisites check failed. Aborting.")
        sys.exit(1)

    # Create results directory
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build script arguments
    script_args = ["--size", args.size]
    if args.test:
        script_args.append("--test")

    # Track overall success
    all_successful = True
    start_time = time.time()

    # 1. Framework comparison
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1/4: Framework Comparison")
    logger.info("=" * 80)

    framework_comparison_script = script_dir / "run_framework_comparison.py"
    if not run_script("Framework Comparison", framework_comparison_script, script_args):
        logger.warning("Framework comparison failed, but continuing with other benchmarks")
        all_successful = False

    # 2. Memory profiling
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2/4: Memory Profiling")
    logger.info("=" * 80)

    memory_profile_script = script_dir / "run_memory_profile.py"
    if not run_script("Memory Profiling", memory_profile_script, script_args):
        logger.warning("Memory profiling failed, but continuing with other benchmarks")
        all_successful = False

    # 3. Scaling study
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3/4: Thread Scaling Study")
    logger.info("=" * 80)

    scaling_study_script = script_dir / "run_scaling_study.py"
    if not run_script("Thread Scaling Study", scaling_study_script, script_args):
        logger.warning("Scaling study failed, but continuing with LaTeX generation")
        all_successful = False

    # 4. Generate LaTeX values
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4/4: Generate LaTeX Values")
    logger.info("=" * 80)

    latex_values_script = script_dir / "create_latex_values.py"
    if not run_script("LaTeX Values Generation", latex_values_script):
        logger.warning("LaTeX values generation failed")
        all_successful = False

    # Summary
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total execution time: {hours}h {minutes}m {seconds}s")
    logger.info(f"Results directory: {results_dir}")

    figures_dir = script_dir / "figures"
    latex_dir = script_dir / "latex"

    if all_successful:
        logger.info("\n✓ All benchmarks completed successfully!")
        logger.info("\nNext steps:")
        logger.info(f"1. Review plots in: {figures_dir}/")
        logger.info(f"2. Copy LaTeX values from: {latex_dir}/latex_values.tex")
        logger.info(f"3. Review raw data in: {results_dir}/")
        logger.info("\nFor paper:")
        logger.info(f"  - Plots: {figures_dir}/*.png")
        logger.info(f"  - LaTeX macros: {latex_dir}/latex_values.tex")
        logger.info(f"  - Summary: {latex_dir}/benchmark_summary.txt")
    else:
        logger.warning("\n⚠ Some benchmarks failed. Please review the logs above.")
        logger.info(f"Partial results available in:")
        logger.info(f"  - {figures_dir}/")
        logger.info(f"  - {latex_dir}/")
        logger.info(f"  - {results_dir}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
