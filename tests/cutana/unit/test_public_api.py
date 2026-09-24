#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""What `import cutana` promises."""

import fitsbolt
import numpy as np
import pytest
from dotmap import DotMap

import cutana
from cutana import get_default_config, image_processor


def test_everything_named_in_all_resolves_to_the_module_function():
    """`from cutana import *` is otherwise the first thing to find a name that is not bound."""
    for name in cutana.__all__:
        assert hasattr(cutana, name), name

    # Re-exported, not reimplemented.
    assert cutana.combine_channels is image_processor.combine_channels
    assert cutana.apply_normalisation is image_processor.apply_normalisation


def test_a_preview_can_mix_and_stretch_through_the_public_names():
    """The flow the export exists for, run rather than asserted about.

    An export that resolves but no longer works through the public path is the regression an
    import check cannot see.
    """
    config = get_default_config()
    config.channel_weights = {"VIS": [0.0, 0.0, 1.0], "NIR-H": [1.0, 0.0, 0.0]}
    config.normalisation_method = "linear"

    raw = np.zeros((1, 4, 4, 2), dtype=np.float32)
    raw[..., 0] = 10.0
    raw[..., 1] = 1.0

    mixed = cutana.combine_channels(raw, config.channel_weights, ["VIS", "NIR-H"])
    assert mixed.shape == (1, 4, 4, 3)

    display = cutana.apply_normalisation(mixed, config)
    assert display.shape == mixed.shape
    # VIS is ten times brighter and weighs into blue; the order survived the round trip.
    assert display[0, ..., 2].mean() > display[0, ..., 0].mean()


def test_named_weights_follow_reversed_tensor():
    raw = np.zeros((1, 4, 4, 2), dtype=np.float32)
    raw[..., 0] = 7
    raw[..., 1] = 2
    weights = {"VIS": [1.0, 0.0, 0.0], "NIR-H": [0.0, 1.0, 0.0]}
    result = cutana.combine_channels(raw, weights, ["NIR-H", "VIS"])
    np.testing.assert_array_equal(result[0, 0, 0], [2, 7, 0])


def test_a_weight_count_that_does_not_match_the_tensor_is_refused():
    raw = np.zeros((1, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="one weight entry"):
        cutana.combine_channels(raw, {"VIS": [1.0], "NIR-H": [0.0]}, ["VIS", "NIR-H", "NIR-J"])


def test_normalising_does_not_edit_the_config_it_was_given():
    """As public API this is called in a loop with one config, and it was writing into it."""
    external = DotMap(_dynamic=False)
    external.normalisation_method = fitsbolt.NormalisationMethod.ASINH
    external.output_dtype = np.float32
    external.normalisation = DotMap(_dynamic=False)
    external.normalisation.asinh_scale = [0.7]
    external.normalisation.asinh_clip = [99.8]
    # This is the key that makes the function write crop settings into the config it was handed.
    external.normalisation.crop_for_maximum_value = (2, 2)

    config = get_default_config()
    config.external_fitsbolt_cfg = external
    before = config.normalisation.toDict()

    cutana.apply_normalisation(np.zeros((1, 4, 4, 1), dtype=np.float32), config)

    assert config.normalisation.toDict() == before
