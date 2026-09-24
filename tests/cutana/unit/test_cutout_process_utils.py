#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for the original_cutout_size computation in cutout_process_utils."""

import warnings

import numpy as np
import pytest

from cutana.cutout_process_utils import _compute_original_sizes

NAN = np.nan


class TestComputeOriginalSizes:
    """Fail-vs-warn behaviour of the per-source size resolution.

    The catalogue carries either diameter_pixel or diameter_arcsec (possibly mixed
    per source). diameter_pixel always wins; arcsec falls back to round(arcsec/scale).
    Anything without a usable size resolves to 0 (→ None downstream) and is counted
    for a single per-batch warning rather than warned per source.
    """

    def _arr(self, values):
        return np.array(values, dtype=float)

    def test_diameter_pixel_takes_precedence_independent_of_scale(self):
        """diameter_pixel is used as-is even when no pixel scale is available."""
        result = _compute_original_sizes(self._arr([64, 128]), self._arr([NAN, NAN]), None)
        assert result.sizes.tolist() == [64, 128]
        assert result.n_no_scale == 0
        assert result.n_subpixel == 0

    def test_arcsec_converts_with_scale(self):
        """diameter_arcsec / pixel_scale rounds to the expected pixel size."""
        # 6.4" at 0.1"/px -> 64 px.
        result = _compute_original_sizes(self._arr([NAN]), self._arr([6.4]), 0.1)
        assert result.sizes.tolist() == [64]
        assert result.n_subpixel == 0

    def test_pixel_wins_over_arcsec_when_both_present(self):
        """When a source has both columns, diameter_pixel wins."""
        result = _compute_original_sizes(self._arr([64]), self._arr([6.4]), 0.1)
        assert result.sizes.tolist() == [64]

    def test_no_cast_warning_for_all_nan_arcsec(self):
        """A diameter_pixel-only batch must not emit the 'invalid value in cast' warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = _compute_original_sizes(self._arr([64, 64]), self._arr([NAN, NAN]), 0.1)
        assert result.sizes.tolist() == [64, 64]

    def test_subpixel_arcsec_counted_not_sized(self):
        """A sub-pixel arcsec value rounds to 0 and is counted, size left at 0."""
        # 0.05" at 0.1"/px rounds to 0 px.
        result = _compute_original_sizes(self._arr([NAN, NAN]), self._arr([0.05, 0.05]), 0.1)
        assert result.sizes.tolist() == [0, 0]
        assert result.n_subpixel == 2
        assert result.n_no_scale == 0

    def test_arcsec_without_scale_counted_no_scale(self):
        """arcsec-sized sources with no pixel scale are counted, not converted."""
        result = _compute_original_sizes(self._arr([NAN, 32]), self._arr([6.4, NAN]), None)
        # The diameter_pixel source still resolves; the arcsec one is left undefined.
        assert result.sizes.tolist() == [0, 32]
        assert result.n_no_scale == 1
        assert result.n_subpixel == 0

    def test_no_diameter_at_all_resolves_to_zero(self):
        """A source with neither diameter resolves to 0 and is not miscounted."""
        result = _compute_original_sizes(self._arr([NAN]), self._arr([NAN]), 0.1)
        assert result.sizes.tolist() == [0]
        assert result.n_no_scale == 0
        assert result.n_subpixel == 0

    @pytest.mark.parametrize("bad_scale", [0.0, -0.1, np.inf, np.nan])
    def test_degenerate_pixel_scale_fails_hard(self, bad_scale):
        """A broken WCS (<=0 / non-finite scale) used for conversion raises, not warns."""
        with pytest.raises(ValueError, match="Degenerate pixel scale"):
            _compute_original_sizes(self._arr([NAN]), self._arr([6.4]), bad_scale)

    def test_degenerate_scale_ignored_when_no_arcsec_conversion_needed(self):
        """A degenerate scale is harmless when no source needs arcsec conversion."""
        # Only diameter_pixel sources → the scale is never used, so no raise.
        result = _compute_original_sizes(self._arr([64]), self._arr([NAN]), 0.0)
        assert result.sizes.tolist() == [64]
