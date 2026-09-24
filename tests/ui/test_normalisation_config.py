#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""The config the UI hands the backend must keep the normalisation keys it does not render.

`NormalisationConfigWidget` renders part of the normalisation block, and `asinh_n_samples`
is not one of the fields it shows. While it built its block from nothing, assigning that
block over the config's own dropped the key, and an asinh run then failed inside fitsbolt
with a message about fitsbolt's config rather than cutana's.
"""

import pytest

from cutana import get_default_config
from cutana_ui.widgets.configuration_widget import SharedConfigurationWidget

# A non-default subsample size. The default is `None`, so a block that carried the key
# with the wrong value -- seeded as `None`, or re-invented by a dynamic dot access, which
# stores the empty DotMap it creates -- is indistinguishable from one that carried it
# properly. A real value separates "came through" from "happens to be there".
CARRIED_ASINH_N_SAMPLES = 5000


def _widget():
    """A configuration widget with the advanced panel showing, set to asinh."""
    config = get_default_config()
    config.normalisation_method = "asinh"
    config.normalisation.asinh_n_samples = CARRIED_ASINH_N_SAMPLES
    return SharedConfigurationWidget(
        config=config,
        compact=False,
        show_extensions=True,
        show_matrix=False,
        show_advanced_params=True,
    )


@pytest.mark.parametrize("mode", ["normal", "raw_cutout", "flux_conserved"])
def test_ui_config_keeps_every_default_normalisation_key(mode):
    """No key the widget does not render may be lost, on any of the three branches.

    `get_current_config` assigns the widget's block in a different place for each mode, so
    a fix applied to one of them would leave the other two dropping the key.
    """
    widget = _widget()
    if mode == "raw_cutout":
        widget.do_only_cutout_checkbox.value = True
    elif mode == "flux_conserved":
        widget.normalisation_widget.flux_conserved_checkbox.value = True
    defaults = set(get_default_config().normalisation.toDict())

    from_ui = widget.get_current_config().normalisation.toDict()

    assert defaults - set(from_ui) == set(), (
        f"UI dropped normalisation keys: {defaults - set(from_ui)}"
    )
    assert from_ui["asinh_n_samples"] == CARRIED_ASINH_N_SAMPLES, (
        "the key is present but not the value the config carried"
    )


def test_raw_cutout_mode_keeps_its_forced_normalisation_method():
    """Raw extraction forces `none`; nothing after the branch may put the dropdown back.

    The blanket update that used to close `get_current_config` copied the widget's
    `normalisation_method` over the forced value, leaving a config that asked for raw
    cutouts and asinh at once.
    """
    widget = _widget()
    widget.do_only_cutout_checkbox.value = True

    config = widget.get_current_config()

    assert config.do_only_cutout_extraction is True
    assert config.normalisation_method == "none"
    assert config.flux_conserved_resizing is False


def test_raw_cutout_mode_works_without_the_advanced_panel():
    """The start screen has no normalisation widget, and raw extraction must still work.

    The same blanket update read a name that is only bound when the advanced panel is
    showing, so this combination raised `NameError` before reaching the caller.
    """
    config = get_default_config()
    widget = SharedConfigurationWidget(
        config=config,
        compact=False,
        show_extensions=True,
        show_matrix=False,
        show_advanced_params=False,
    )
    widget.do_only_cutout_checkbox.value = True

    assert widget.get_current_config().do_only_cutout_extraction is True
