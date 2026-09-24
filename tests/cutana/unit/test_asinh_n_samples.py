#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for wiring the asinh percentile subsample size through to fitsbolt.

The ``asinh_n_samples`` knob lets the asinh percentile bounds be estimated from a
deterministic pixel subsample (fitsbolt ``norm_asinh_n_samples``) instead of every
pixel. Cutana carries a conservative default; AnomalyMatch may override it via the
external fitsbolt config.
"""

import fitsbolt
import numpy as np
import pytest
from dotmap import DotMap

from cutana.get_default_config import get_default_config
from cutana.image_processor import apply_normalisation
from cutana.normalisation_parameters import (
    NormalisationDefaults,
    build_fitsbolt_params_from_external_cfg,
    convert_cfg_to_fitsbolt_cfg,
    get_default_normalisation_config,
)


class TestAsinhNSamplesWiring:
    """The asinh subsample size must reach fitsbolt for asinh and only asinh."""

    def test_default_normalisation_config_has_asinh_n_samples(self):
        """The default config exposes the conservative subsample default."""
        norm = get_default_normalisation_config()
        assert norm.asinh_n_samples == NormalisationDefaults.ASINH_N_SAMPLES

    def test_convert_cfg_threads_asinh_n_samples(self):
        """convert_cfg_to_fitsbolt_cfg forwards the value for the asinh method."""
        cfg = get_default_config()
        cfg.normalisation_method = "asinh"
        cfg.normalisation.asinh_n_samples = 4000

        params = convert_cfg_to_fitsbolt_cfg(cfg, num_channels=3)

        assert params["norm_asinh_n_samples"] == 4000

    def test_convert_cfg_omits_asinh_n_samples_for_other_methods(self):
        """Non-asinh methods must not carry the asinh subsample parameter."""
        for method in ("log", "zscale", "linear"):
            cfg = get_default_config()
            cfg.normalisation_method = method
            params = convert_cfg_to_fitsbolt_cfg(cfg, num_channels=3)
            assert "norm_asinh_n_samples" not in params

    def test_external_cfg_forwards_asinh_n_samples_when_present(self):
        """An external (AnomalyMatch) config may drive the subsample aggressively."""
        external = DotMap(_dynamic=False)
        external.normalisation_method = fitsbolt.NormalisationMethod.ASINH
        external.output_dtype = "float32"
        external.normalisation = DotMap(_dynamic=False)
        external.normalisation.asinh_scale = [0.7, 0.7, 0.7]
        external.normalisation.asinh_clip = [99.8, 99.8, 99.8]
        external.normalisation.asinh_n_samples = 2000

        params = build_fitsbolt_params_from_external_cfg(external, num_channels=3)

        assert params["norm_asinh_n_samples"] == 2000

    def test_external_cfg_omits_asinh_n_samples_when_absent(self):
        """A legacy external config without the key must fall back to fitsbolt's default."""
        external = DotMap(_dynamic=False)
        external.normalisation_method = fitsbolt.NormalisationMethod.ASINH
        external.output_dtype = "float32"
        external.normalisation = DotMap(_dynamic=False)
        external.normalisation.asinh_scale = [0.7, 0.7, 0.7]
        external.normalisation.asinh_clip = [99.8, 99.8, 99.8]

        params = build_fitsbolt_params_from_external_cfg(external, num_channels=3)

        assert "norm_asinh_n_samples" not in params


class TestAsinhNSamplesMissingFromConfig:
    """A config that lost the key must fail here, not inside fitsbolt.

    Nothing else reads ``asinh_n_samples``, so a config that drops it looks healthy until
    an asinh run reaches this conversion. Dot access on a dynamic DotMap invents an empty
    DotMap for a missing attribute, and that object used to travel all the way into
    fitsbolt, which rejected it as its own ``normalisation.asinh_n_samples must be an
    integer, got DotMap``. ``apply_normalisation`` then wrapped that in a message about
    ``external_fitsbolt_cfg`` and channel counts, neither of which was involved.
    """

    @staticmethod
    def _dynamic_config(**normalisation_overrides):
        """An asinh config whose normalisation block is dynamic, as a round trip leaves it.

        A preview copy or a worker serialisation rebuilds the config from a plain dict,
        which is what restores the auto-creating behaviour that hid the missing key.
        """
        cfg = get_default_config()
        cfg.normalisation_method = "asinh"
        as_dict = cfg.toDict()
        as_dict["normalisation"].update(normalisation_overrides)
        return DotMap(as_dict)

    def test_convert_cfg_names_the_lost_key(self):
        """The lost key is named here rather than surfacing as a fitsbolt type error."""
        cfg = self._dynamic_config()
        del cfg.normalisation["asinh_n_samples"]

        with pytest.raises(ValueError, match="normalisation.asinh_n_samples is missing"):
            convert_cfg_to_fitsbolt_cfg(cfg, num_channels=4)

    def test_convert_cfg_names_the_lost_key_on_a_non_dynamic_config_too(self):
        """The shape the widget itself produced must get the same message, not a KeyError.

        Bracket access on a non-dynamic DotMap raises before any check of ours can run, so
        the guard has to test membership on the plain dict rather than read the key first.
        """
        cfg = DotMap(self._dynamic_config().toDict(), _dynamic=False)
        del cfg.normalisation["asinh_n_samples"]

        with pytest.raises(ValueError, match="normalisation.asinh_n_samples is missing"):
            convert_cfg_to_fitsbolt_cfg(cfg, num_channels=4)

    def test_convert_cfg_rejects_a_bool(self):
        """`True` passes ``isinstance(x, int)`` and would reach fitsbolt as one pixel."""
        cfg = self._dynamic_config(asinh_n_samples=True)

        with pytest.raises(ValueError, match="must be an int or None, got bool"):
            convert_cfg_to_fitsbolt_cfg(cfg, num_channels=4)

    def test_convert_cfg_rejects_a_key_an_earlier_access_invented(self):
        """Presence is not enough: dot access stores the empty DotMap it auto-creates.

        A config that lost the key can therefore carry one under it and pass an `in`
        check, which is why the guard tests the value.
        """
        cfg = self._dynamic_config()
        del cfg.normalisation["asinh_n_samples"]
        cfg.normalisation.asinh_n_samples  # noqa: B018  # the access that invents the key
        assert "asinh_n_samples" in cfg.normalisation.toDict()

        with pytest.raises(ValueError, match="must be an int or None, got dict"):
            convert_cfg_to_fitsbolt_cfg(cfg, num_channels=4)

    def test_convert_cfg_still_forwards_none_through_a_round_trip(self):
        """The exact-percentile default must survive the same round trip, not be rejected.

        ``None`` and a missing key look alike through dot access, so the guard has to tell
        them apart or it would refuse every default configuration.
        """
        cfg = self._dynamic_config(asinh_n_samples=None)

        params = convert_cfg_to_fitsbolt_cfg(cfg, num_channels=4)

        assert params["norm_asinh_n_samples"] is None

    def test_apply_normalisation_runs_asinh_over_four_channels(self):
        """The reported failure, from the conversion through fitsbolt's own validation.

        Asserts the stretch actually ran: a shape check alone would pass on an untouched
        input, which is the one outcome that would hide a silently skipped normalisation.
        """
        cfg = self._dynamic_config()
        cfg.data_type = "float32"
        images = np.random.default_rng(0).random((3, 32, 32, 4)).astype(np.float32)

        normalised = apply_normalisation(images, cfg)

        assert normalised.shape == images.shape
        assert normalised.dtype == np.float32
        assert not np.allclose(normalised, images), "asinh left the pixels untouched"
        assert normalised.min() >= 0.0 and normalised.max() <= 1.0
