#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Test runner for paper benchmarks using small test catalogue.

This script ACTUALLY tests the real benchmark scripts by importing and calling them
with a small test catalogue to verify:
1. astropy_baseline.py works
2. run_framework_comparison.py works
3. run_memory_profile.py works (creates plots!)
4. run_scaling_study.py works (creates plots!)
5. create_latex_values.py works

Uses the actual production code, not test duplicates.
"""

import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from cutana.logging_config import setup_logging  # noqa: E402


def test_astropy_baseline():
    """Test the actual astropy_baseline.py module."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Astropy Baseline Module")
    logger.info("=" * 80)

    try:
        import pandas as pd

        from paper_scripts.astropy_baseline import process_catalogue_astropy

        script_dir = Path(__file__).parent
        test_catalogue = script_dir / "data" / "test-100.csv"

        if not test_catalogue.exists():
            logger.error(f"Test catalogue not found: {test_catalogue}")
            return {"status": "failed", "error": "Test catalogue missing"}

        catalogue_df = pd.read_csv(test_catalogue)
        logger.info(f"Testing with {len(catalogue_df)} sources")

        results = process_catalogue_astropy(catalogue_df, fits_extension="PRIMARY")

        logger.info(
            f"✓ Astropy baseline: {results['total_time_seconds']:.2f}s, {results['sources_per_second']:.1f} sources/sec"
        )

        return {
            "status": "success",
            "time_seconds": results["total_time_seconds"],
            "rate": results["sources_per_second"],
            "errors": results["errors"],
        }

    except Exception as e:
        logger.error(f"✗ Astropy baseline test failed: {e}")
        logger.error("Exception details:", exc_info=True)
        return {"status": "failed", "error": str(e)}


def test_framework_comparison():
    """Test the actual run_framework_comparison.py functions."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Framework Comparison Functions")
    logger.info("=" * 80)

    try:
        import pandas as pd

        from paper_scripts.run_framework_comparison import (
            create_comparison_table,
            run_astropy_benchmark,
            run_cutana_benchmark,
        )

        script_dir = Path(__file__).parent
        test_catalogue = script_dir / "data" / "test-100.csv"
        output_dir = script_dir / "results" / f"test_framework_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        catalogue_df = pd.read_csv(test_catalogue)
        logger.info(f"Testing with {len(catalogue_df)} sources")

        results = []

        # Test Astropy benchmark
        astropy_result = run_astropy_benchmark(catalogue_df, "test_scenario")
        results.append(astropy_result)
        logger.info(f"✓ Astropy: {astropy_result['sources_per_second']:.1f} sources/sec")

        # Test Cutana 1 worker
        cutana_1w_result = run_cutana_benchmark(
            catalogue_df, 1, str(output_dir / "cutana_1w"), "test_scenario"
        )
        results.append(cutana_1w_result)
        logger.info(f"✓ Cutana 1w: {cutana_1w_result['sources_per_second']:.1f} sources/sec")

        # Test table creation
        table = create_comparison_table(results)
        logger.info(f"✓ Created comparison table with {len(table)} rows")

        return {"status": "success", "results_count": len(results), "output_dir": str(output_dir)}

    except Exception as e:
        logger.error(f"✗ Framework comparison test failed: {e}")
        logger.error("Exception details:", exc_info=True)
        return {"status": "failed", "error": str(e)}


def test_memory_profile():
    """Test the actual run_memory_profile.py functions (creates plots!)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Memory Profile Functions (WITH PLOTS)")
    logger.info("=" * 80)

    try:
        import pandas as pd

        from paper_scripts.run_memory_profile import (
            create_memory_plot,
            profile_astropy_memory,
            profile_cutana_memory,
            save_memory_stats,
        )

        script_dir = Path(__file__).parent
        test_catalogue = script_dir / "data" / "test-100.csv"
        output_dir = script_dir / "results" / f"test_memory_{time.strftime('%Y%m%d_%H%M%S')}"
        results_dir = script_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        catalogue_df = pd.read_csv(test_catalogue)
        logger.info(f"Testing with {len(catalogue_df)} sources")

        # Profile Astropy
        logger.info("Profiling Astropy...")
        astropy_data = profile_astropy_memory(catalogue_df)
        logger.info(f"✓ Astropy memory profile: peak={max(astropy_data[0]):.1f}MB")

        # Profile Cutana 1 worker
        logger.info("Profiling Cutana 1 worker...")
        cutana_1w_data = profile_cutana_memory(catalogue_df, 1, str(output_dir / "cutana_1w"))
        logger.info(f"✓ Cutana 1w memory profile: peak={max(cutana_1w_data[0]):.1f}MB")

        # Profile Cutana 4 workers
        logger.info("Profiling Cutana 4 workers...")
        cutana_4w_data = profile_cutana_memory(catalogue_df, 4, str(output_dir / "cutana_4w"))
        logger.info(f"✓ Cutana 4w memory profile: peak={max(cutana_4w_data[0]):.1f}MB")

        # Create plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        figures_dir = script_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_path = figures_dir / f"test_memory_profile_{timestamp}.png"
        create_memory_plot(astropy_data, cutana_1w_data, cutana_4w_data, plot_path)
        logger.info(f"✓ Created memory plot: {plot_path}")

        # Save stats
        stats_path = results_dir / f"test_memory_stats_{timestamp}.json"
        stats = save_memory_stats(astropy_data, cutana_1w_data, cutana_4w_data, stats_path)
        logger.info(f"✓ Saved memory stats: {stats_path}")

        return {
            "status": "success",
            "plot_created": plot_path.exists(),
            "stats_created": stats_path.exists(),
            "plot_path": str(plot_path),
            "stats_path": str(stats_path),
        }

    except Exception as e:
        logger.error(f"✗ Memory profile test failed: {e}")
        logger.error("Exception details:", exc_info=True)
        return {"status": "failed", "error": str(e)}


def test_scaling_study():
    """Test the actual run_scaling_study.py functions (creates plots!)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Scaling Study Functions (WITH PLOTS)")
    logger.info("=" * 80)

    try:
        import pandas as pd

        from paper_scripts.run_scaling_study import (
            calculate_scaling_metrics,
            create_scaling_plots,
            run_cutana_scaling_test,
            save_scaling_results,
        )

        script_dir = Path(__file__).parent
        test_catalogue = script_dir / "data" / "test-100.csv"
        output_dir = script_dir / "results" / f"test_scaling_{time.strftime('%Y%m%d_%H%M%S')}"
        results_dir = script_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        catalogue_df = pd.read_csv(test_catalogue)
        logger.info(f"Testing with {len(catalogue_df)} sources")

        # Test with 1, 2, 4 workers (smaller set for testing)
        worker_counts = [1, 2, 4]
        results = []

        for num_workers in worker_counts:
            logger.info(f"Testing with {num_workers} workers...")
            result = run_cutana_scaling_test(
                catalogue_df, num_workers, str(output_dir / f"workers_{num_workers}")
            )
            results.append(result)
            logger.info(f"✓ {num_workers} workers: {result['sources_per_second']:.1f} sources/sec")

        # Calculate metrics
        metrics = calculate_scaling_metrics(results)
        logger.info(f"✓ Calculated scaling metrics")

        # Create plots
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        figures_dir = script_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        create_scaling_plots(metrics, figures_dir, timestamp)
        plot_path = figures_dir / f"scaling_study_{timestamp}.png"
        logger.info(f"✓ Created scaling plot: {plot_path}")

        # Save results
        summary_df = save_scaling_results(results, metrics, results_dir, timestamp)
        logger.info(f"✓ Saved scaling results")

        return {
            "status": "success",
            "worker_counts_tested": len(worker_counts),
            "plot_created": plot_path.exists(),
            "plot_path": str(plot_path),
        }

    except Exception as e:
        logger.error(f"✗ Scaling study test failed: {e}")
        logger.error("Exception details:", exc_info=True)
        return {"status": "failed", "error": str(e)}


def test_latex_values():
    """Test the actual create_latex_values.py functions."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: LaTeX Values Generation")
    logger.info("=" * 80)

    try:
        from paper_scripts.create_latex_values import create_summary_table, generate_latex_macros

        script_dir = Path(__file__).parent
        results_dir = script_dir / "results"

        # Create test values
        test_values = {
            "astropyMemMapTime": "100.0",
            "astropyMemMapRate": "1000.0",
            "cutanaSingleTime": "50.0",
            "cutanaSingleRate": "2000.0",
            "cutanaFourTime": "20.0",
            "cutanaFourRate": "5000.0",
            "speedupSingle": "2.00",
            "speedupFour": "5.00",
            "scalingFactor": "2.50",
            "memoryUsageSingle": "10.50",
            "memoryUsageFour": "25.00",
        }

        # Generate LaTeX
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        latex_dir = script_dir / "latex"
        latex_dir.mkdir(parents=True, exist_ok=True)
        latex_path = latex_dir / f"test_latex_values_{timestamp}.tex"
        generate_latex_macros(test_values, latex_path)
        logger.info(f"✓ Created LaTeX file: {latex_path}")

        # Create summary
        summary_path = latex_dir / f"test_summary_{timestamp}.txt"
        create_summary_table(test_values, summary_path)
        logger.info(f"✓ Created summary: {summary_path}")

        return {
            "status": "success",
            "latex_created": latex_path.exists(),
            "summary_created": summary_path.exists(),
            "latex_path": str(latex_path),
        }

    except Exception as e:
        logger.error(f"✗ LaTeX values test failed: {e}")
        logger.error("Exception details:", exc_info=True)
        return {"status": "failed", "error": str(e)}


def main():
    """Run all tests."""
    setup_logging(log_level="INFO", console_level="INFO")

    logger.info("=" * 80)
    logger.info("PAPER BENCHMARKS - INTEGRATION TEST")
    logger.info("Testing actual benchmark scripts with small dataset")
    logger.info("=" * 80)

    results = {}

    # Test 1: Astropy baseline
    results["astropy_baseline"] = test_astropy_baseline()

    # Test 2: Framework comparison
    results["framework_comparison"] = test_framework_comparison()

    # Test 3: Memory profile (creates plots!)
    results["memory_profile"] = test_memory_profile()

    # Test 4: Scaling study (creates plots!)
    results["scaling_study"] = test_scaling_study()

    # Test 5: LaTeX values
    results["latex_values"] = test_latex_values()

    # Save test results
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_results_path = results_dir / f"integration_test_{timestamp}.json"

    with open(test_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nTest results saved to: {test_results_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    all_passed = all(r.get("status") == "success" for r in results.values())

    for test_name, result in results.items():
        status = "✓ PASS" if result.get("status") == "success" else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

        # Show special info
        if "plot_path" in result:
            logger.info(f"  Plot created: {result['plot_path']}")
        if "latex_path" in result:
            logger.info(f"  LaTeX created: {result['latex_path']}")

    if all_passed:
        logger.info("\n✓ All tests passed! All benchmark scripts are working correctly.")
        logger.info("\nPlots created:")
        logger.info("  - Memory profile plot")
        logger.info("  - Scaling study plot")
    else:
        logger.error("\n✗ Some tests failed. Please review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
