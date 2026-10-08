#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Choosing which rows a validation check looks at.

Three checks had grown their own copy of "if the catalogue is large, take a deterministic sample",
each with its own threshold and its own bare `random_state=42`. One copy, one seed.

The seed matters more than it looks: validation runs twice in a normal session -- once when the
catalogue is loaded and once before the run -- and a report that named different rows each time
would read as a catalogue that keeps changing.
"""

from typing import Optional

import pandas as pd
from loguru import logger


def sample_for_validation(
    catalogue_df: pd.DataFrame, sample_size: int, what: Optional[str] = None
) -> pd.DataFrame:
    """The rows to check: all of them, or a deterministic sample of a large catalogue.

    Args:
        catalogue_df: The catalogue.
        sample_size: Most rows to return.
        what: What is being checked, for the log line. A sample that is not mentioned reads as a
            full pass, and a clean report over 0.1% of a catalogue is worth saying out loud.

    Returns:
        The whole catalogue, or `sample_size` rows of it chosen the same way every time.
    """
    if len(catalogue_df) <= sample_size:
        return catalogue_df

    logger.info(
        f"Large catalogue ({len(catalogue_df)} sources), "
        f"checking {sample_size} random rows{f' for {what}' if what else ''}"
    )
    return catalogue_df.sample(n=sample_size, random_state=42)
