#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Cutana UI - Interactive interface for astronomical cutout creation."""

try:
    from .app import start

    __all__ = ["start"]
except ImportError as e:
    # Handle missing dependencies during testing or when not in Jupyter environment
    _error_msg = str(e)

    def start():
        raise ImportError(f"Cutana UI requires ipywidgets and related dependencies: {_error_msg}")

    __all__ = ["start"]
