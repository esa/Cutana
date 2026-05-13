#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Shared browser test helpers for waiting on UI state transitions."""


def wait_for_analysis(page, timeout=60000):
    """Wait for Voila kernel boot and analysis to complete.

    Waits for the Start button to become visible, which only happens
    after the notebook has loaded, widgets have rendered, and the
    auto-analysis has finished.
    """
    page.locator("button").filter(has_text="Start").first.wait_for(state="visible", timeout=timeout)


def wait_for_main_screen(page, timeout=30000):
    """Wait for the main screen to fully render after clicking Start.

    Waits for the Processing Status panel header, which is one of the
    last elements to appear on the main screen.
    """
    page.locator("text=/Processing Status/i").first.wait_for(state="visible", timeout=timeout)
