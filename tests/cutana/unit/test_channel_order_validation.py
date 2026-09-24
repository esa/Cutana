#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Name resolution must not depend on dictionary or catalogue insertion order."""

import numpy as np
import pytest

from cutana.image_processor import combine_channels
from cutana.validate_config import validate_channel_order_consistency


@pytest.mark.parametrize("names", [["VIS", "NIR-H"], ["NIR-H", "VIS"]])
def test_reordered_weights_preserve_pixels(names):
    values = {"VIS": 2.0, "NIR-H": 7.0}
    tensor = np.array([values[name] for name in names], dtype=np.float32).reshape(1, 1, 1, 2)
    weights = {"NIR-H": [0.0, 3.0, 1.0], "VIS": [4.0, 0.0, 2.0]}
    result = combine_channels(tensor, weights, names)
    np.testing.assert_array_equal(result, np.array([8.0, 21.0, 11.0]).reshape(1, 1, 1, 3))


@pytest.mark.parametrize(
    "names, weights",
    [
        ([], {}),
        (["VIS", "NIR-H"], {"VIS": [1.0]}),
        (["VIS"], {"VIS": [1.0], "NIR-H": [1.0]}),
        (["tile_H2"], {"H": [1.0]}),
        (["tile_VIS", "other_VIS"], {"VIS": [1.0], "NIR-H": [1.0]}),
        (["tile_NIR-H", "tile_VIS"], {"H": [1.0], "NIR-H": [1.0]}),
        (["OTHER"], {"VIS": [1.0]}),
    ],
)
def test_invalid_mapping_fails(names, weights):
    with pytest.raises(ValueError, match="Channel mapping"):
        validate_channel_order_consistency(names, weights)


def test_filename_tokens_and_normalized_filter_separator():
    assert validate_channel_order_consistency(
        ["tile_NIR_H_IMAGE", "tile_VIS_IMAGE"], {"VIS": [1.0], "NIR-H": [2.0]}
    ) == ["NIR-H", "VIS"]


def test_exact_names_take_precedence():
    assert validate_channel_order_consistency(["NIR-H", "H"], {"H": [1.0], "NIR-H": [2.0]}) == [
        "NIR-H",
        "H",
    ]


def test_names_are_required():
    with pytest.raises(ValueError, match="channel_names is required"):
        combine_channels(np.ones((1, 1, 1, 1)), {"VIS": [1.0]})


@pytest.mark.parametrize("name", ["VIS", "PRIMARY", "tile_NIR_H_IMAGE"])
def test_default_primary_key_accepts_any_single_channel(name):
    """One channel against the `get_default_config` key pairs unambiguously.

    The name is deliberately not checked here: with a single weight entry and a single
    channel there is no other pairing to pick, so there is no order to get wrong.
    """
    assert validate_channel_order_consistency([name], {"PRIMARY": [1.0]}) == ["PRIMARY"]


def test_primary_escape_hatch_does_not_extend_to_multiple_channels():
    with pytest.raises(ValueError, match="one weight entry per tensor channel"):
        validate_channel_order_consistency(["VIS", "NIR-H"], {"PRIMARY": [1.0]})


@pytest.mark.parametrize("name", ["VIS", "UNKNOWN", "survey_tile_0_sci"])
def test_unknown_key_accepts_any_single_channel(name):
    """`UNKNOWN` is what the UI offers for a tile the recogniser cannot classify.

    It has to pair positionally for the same reason `PRIMARY` does: the label names no
    band, so there is nothing in the channel's own name to check it against. Discovery
    already refuses a row with two of them, which is the case where a wrong pairing
    could silently mis-weight pixels.
    """
    assert validate_channel_order_consistency([name], {"UNKNOWN": [1.0]}) == ["UNKNOWN"]


def test_a_real_band_key_must_still_match_its_channel():
    """The escape is for keys that name no band, not for single channels in general.

    Widening it to any lone key would drop the check that catches a tile loaded under
    the wrong band label -- the one thing `select_fits_set_bands` relies on downstream.
    """
    with pytest.raises(ValueError, match="missing or ambiguous"):
        validate_channel_order_consistency(["EUC_MER_BGSUB-MOSAIC-NIR-H_T1"], {"VIS": [1.0]})


@pytest.mark.parametrize(
    "name, key",
    [
        ("EUC_MER_BGSUB-MOSAIC-NIR_H_TILE1", "NIR-H"),
        ("EUC_MER_BGSUB-MOSAIC-NIR-H_TILE1", "NIR_H"),
        ("EUC_MER_BGSUB-MOSAIC-NIR-H_TILE1", "NIR-H"),
        ("EUC_MER_BGSUB-MOSAIC-NIR_H_TILE1", "NIR_H"),
    ],
)
def test_separators_are_interchangeable_in_both_directions(name, key):
    """`re.escape` escapes `-` but not `_`, so matching on the escaped key rewrote one.

    The hyphen-key-against-underscore-name direction worked and the reverse did not,
    while the documentation promised both.
    """
    assert validate_channel_order_consistency([name], {key: [1.0]}) == [key]
