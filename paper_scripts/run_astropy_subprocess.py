#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Subprocess wrapper for running Astropy baseline with proper thread isolation.

This script is called as a subprocess to ensure thread limits are set BEFORE
any numpy/scipy/astropy imports from previous runs.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from cutana.logging_config import setup_logging  # noqa: E402
from paper_scripts.astropy_baseline import (  # noqa: E402
    load_baseline_config,
    process_catalogue_astropy,
)


def main():
    parser = argparse.ArgumentParser(description="Run Astropy baseline in subprocess")
    parser.add_argument("--catalogue", required=True, help="Path to catalogue CSV")
    parser.add_argument("--output-json", required=True, help="Path to save results JSON")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads")
    parser.add_argument("--scenario-name", required=True, help="Scenario name for logging")
    parser.add_argument("--output-dir", required=True, help="Temporary output directory")

    args = parser.parse_args()

    # =========================================================================
    # CRITICAL: Set CPU affinity BEFORE any other imports (Windows-compatible)
    # =========================================================================
    import os

    import psutil

    current_process = psutil.Process()

    # Pin process to specific cores based on thread count
    available_cores = list(range(psutil.cpu_count(logical=True)))
    cores_to_use = available_cores[: args.threads]  # Use first N cores

    try:
        current_process.cpu_affinity(cores_to_use)
        print(f"CPU affinity set to cores: {cores_to_use}")
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")
        print("Note: On Windows, you may need to run as Administrator for CPU affinity")

    # Also set environment variables for thread limits (belt and suspenders)
    thread_env_vars = {
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "NUMBA_NUM_THREADS": str(args.threads),
        "VECLIB_MAXIMUM_THREADS": str(args.threads),
        "NUMEXPR_NUM_THREADS": str(args.threads),
    }

    for var, value in thread_env_vars.items():
        os.environ[var] = value

    print(f"Thread limit set to {args.threads} via environment variables and CPU affinity")

    # Setup logging
    setup_logging(log_level="INFO", console_level="INFO")

    # Load config and catalogue
    import pandas as pd

    config = load_baseline_config()
    catalogue_df = pd.read_csv(args.catalogue)

    # Run the benchmark (this is a fresh process, so env vars will work)
    results = process_catalogue_astropy(
        catalogue_df,
        fits_extension="PRIMARY",
        target_resolution=config["target_resolution"],
        apply_flux_conversion=config["apply_flux_conversion"],
        interpolation=config["interpolation"],
        output_dir=Path(args.output_dir),
        zeropoint_keyword=config["zeropoint_keyword"],
        process_threads=args.threads,
    )

    # Save results to JSON
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
