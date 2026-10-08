#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Python module to hold constants"""

JANSKY_AB_ZEROPONT = 3631.0  # jansky https://en.wikipedia.org/wiki/AB_magnitude

# Default chunk size for streaming catalogue reads (rows per chunk).
# Used by catalogue_streamer for memory-efficient catalogue processing.
DEFAULT_CATALOGUE_CHUNK_SIZE = 100000

# === Cutout pixel-unit labels ===
# Values of the legacy ``UNIT`` primary-header keyword. ``UNIT`` is descriptive
# rather than machine-readable (``approx Jy`` is not a parseable unit string), so
# it is kept only for consumers written against the 0.3.2 schema; new readers
# should use the standard ``BUNIT`` on the image HDUs instead.
UNIT_JANSKY = "Jy"
# Pixels keep whatever unit the parent tile carried: flux conversion was disabled,
# so cutana cannot name the unit without reading it back from the parent header.
UNIT_ORIGINAL = "OriginalUnit"
# A ``user_flux_conversion_function`` replaced the AB-zeropoint maths, so the unit
# is whatever that function produces. Distinct from UNIT_ORIGINAL: the pixels were
# converted, they just were not converted to Jy by cutana.
UNIT_USER_CONVERSION = "UserConversionUnit"
# Normalisation stretches pixels into the ``data_type`` range, which discards the
# physical scale entirely — the values are dimensionless, not "approximately Jy".
UNIT_NORMALISED = "normalised"
# Prefix applied to a physical unit when the resize did not conserve flux, so the
# pixel values are only approximately in that unit.
UNIT_APPROX_PREFIX = "approx "
