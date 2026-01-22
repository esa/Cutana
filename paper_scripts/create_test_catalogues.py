#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Create test and smaller versions of benchmark catalogues.

Generates:
1. test-100.csv - Tiny test catalogue (100 sources)
2. 1k-8tiles-4channel.csv - Smaller version (8k sources)
3. 100k-4tiles-1channel.csv - Smaller version (100k sources)
"""

import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from cutana.logging_config import setup_logging  # noqa: E402


def create_test_catalogue(source_csv: Path, output_csv: Path, n_sources: int = 100):
    """Create tiny test catalogue."""
    logger.info(f"Creating test catalogue with {n_sources} sources")

    df = pd.read_csv(source_csv)
    test_df = df.head(n_sources)
    test_df.to_csv(output_csv, index=False)

    logger.info(f"Created test catalogue: {output_csv} ({len(test_df)} sources)")


def create_8tiles_catalogue(source_csv: Path, output_csv: Path):
    """Create 8 tiles × 1k sources = 8k total from 32 tiles catalogue."""
    logger.info("Creating 8 tiles catalogue (8k sources)")

    df = pd.read_csv(source_csv)

    # Take first 8000 rows (first 8 tiles × 1k sources each)
    # Assuming tiles are grouped sequentially
    subset_df = df.head(8000)
    subset_df.to_csv(output_csv, index=False)

    logger.info(f"Created 8 tiles catalogue: {output_csv} ({len(subset_df)} sources)")


def create_4tiles_catalogue(source_csv: Path, output_csv: Path):
    """Create 4 tiles × 25k sources = 100k total from 8 tiles catalogue."""
    logger.info("Creating 4 tiles catalogue (100k sources)")

    df = pd.read_csv(source_csv)

    # Take first 100000 rows (first 4 tiles × 25k sources each)
    subset_df = df.head(100000)
    subset_df.to_csv(output_csv, index=False)

    logger.info(f"Created 4 tiles catalogue: {output_csv} ({len(subset_df)} sources)")


def main():
    """Main catalogue creation."""
    setup_logging(log_level="INFO", console_level="INFO")

    logger.info("Creating test and smaller catalogues")

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    # Source catalogues
    cat_100k = data_dir / "100k-1tile-4channel.csv"
    cat_32k = data_dir / "1k-32tiles-4channel.csv"
    cat_200k = data_dir / "200k-8tile-1channel.csv"

    # Check source catalogues exist
    for cat in [cat_100k, cat_32k, cat_200k]:
        if not cat.exists():
            logger.error(f"Source catalogue not found: {cat}")
            sys.exit(1)

    # Create test catalogue (100 sources)
    test_cat = data_dir / "test-100.csv"
    create_test_catalogue(cat_100k, test_cat, n_sources=100)

    # Create 8 tiles catalogue (8k sources)
    cat_8tiles = data_dir / "1k-8tiles-4channel.csv"
    create_8tiles_catalogue(cat_32k, cat_8tiles)

    # Create 4 tiles catalogue (100k sources)
    cat_4tiles = data_dir / "100k-4tiles-1channel.csv"
    create_4tiles_catalogue(cat_200k, cat_4tiles)

    logger.info("\nCatalogue creation completed!")
    logger.info(f"  Test: {test_cat.name}")
    logger.info(f"  8 tiles: {cat_8tiles.name}")
    logger.info(f"  4 tiles: {cat_4tiles.name}")


if __name__ == "__main__":
    main()
