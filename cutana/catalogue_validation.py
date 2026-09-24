#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""The catalogue validation error, in a module both the reader and the analyser can import.

It lives here rather than in ``catalogue_preprocessor`` because ``catalogue_sample`` raises
it too, and ``catalogue_preprocessor`` imports ``catalogue_sample``.
"""


class CatalogueValidationError(Exception):
    """Exception raised when catalogue validation fails."""

    pass
