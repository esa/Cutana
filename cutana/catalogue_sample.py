#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Bounded catalogue reads for discovery and previews, never a full-file scan."""

import random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.table import Table

from .catalogue_streamer import estimate_catalogue_size
from .catalogue_validation import CatalogueValidationError


def read_catalogue_sample(path, sample_size=100, seed=42):
    """Read a bounded discovery sample and report the population estimate.

    CSV samples come from the first 10,000 rows. Parquet samples come from
    bounded windows in up to four randomly selected row groups. Neither is a
    uniform whole-catalogue sample. FITS binary tables support mapped row access.

    The returned frame carries a plain positional index in every format, and that is part
    of the contract. The three readers have no common row numbering that can be produced
    within the read budget -- a Parquet row's absolute position needs the row counts of
    every group before it, which is a metadata scan proportional to the catalogue. So the
    index means "n-th row of this sample" and nothing more; callers naming a row to the
    user should quote its ``SourceID``, which is unique and findable in their file.

    Args:
        path: CSV, Parquet or FITS catalogue path.
        sample_size: Maximum returned rows (at most 10,000).
        seed: Reproducible local random seed.

    Returns:
        Tuple of dataframe, population count, count-is-estimated, sampling scope.

    Raises:
        ValueError: For unsupported formats or invalid sample sizes.
        CatalogueValidationError: If a FITS catalogue carries no table extension.
    """
    if not 1 <= sample_size <= 10000:
        raise ValueError("sample_size must be between 1 and 10000")
    suffix = Path(path).suffix.lower()
    rng = random.Random(seed)
    if suffix == ".csv":
        pool = pd.read_csv(path, nrows=10000)
        estimated = len(pool) == 10000
        count = max(len(pool), estimate_catalogue_size(path)) if estimated else len(pool)
        scope = "first 10000 CSV rows"
    elif suffix == ".parquet":
        with pq.ParquetFile(path) as catalogue:
            count = catalogue.metadata.num_rows
            groups = sorted(
                rng.sample(range(catalogue.num_row_groups), min(4, catalogue.num_row_groups))
            )
            frames = []
            for group in groups:
                batches = catalogue.iter_batches(batch_size=2500, row_groups=[group])
                batch = next(batches, None)
                if batch is not None:
                    frames.append(batch.to_pandas())
            pool = pd.concat(frames) if frames else pd.DataFrame()
        estimated = False
        scope = "first 2500 rows of up to 4 random Parquet row groups"
    elif suffix in (".fits", ".fit"):
        with fits.open(path, memmap=True) as hdul:
            # TableHDU as well as BinTableHDU: `load_catalogue` reads both through
            # `Table.read`, so narrowing to binary tables here would reject a catalogue
            # the rest of the pipeline accepts.
            table = next(
                (hdu for hdu in hdul if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))),
                None,
            )
            if table is None:
                raise CatalogueValidationError(
                    f"{path} has no table extension, so it is not a source catalogue. "
                    "Point at the catalogue file rather than an image tile."
                )
            count = table.header["NAXIS2"]
            indices = sorted(rng.sample(range(count), min(sample_size, count)))
            pool = Table(table.data[indices]).to_pandas()
        estimated = False
        scope = "random FITS table rows"
    else:
        raise ValueError(f"Unsupported catalogue format: {suffix}")
    sample = pool.sample(n=min(sample_size, len(pool)), random_state=seed).sort_index()
    return sample.reset_index(drop=True), count, estimated, scope
