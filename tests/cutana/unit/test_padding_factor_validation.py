#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for padding_factor parameter validation."""

import pytest

from cutana.get_default_config import get_default_config
from cutana.validate_config import validate_config


class TestPaddingFactorValidation:
    """Test validation of padding_factor parameter."""

    def test_padding_factor_default_value(self):
        """Test that default config has valid padding_factor."""
        cfg = get_default_config()
        cfg.source_catalogue = "dummy_catalogue.csv"  # Set required field
        assert hasattr(cfg, "padding_factor")
        assert cfg.padding_factor == 1.0

        # Should validate without errors
        validate_config(cfg, check_paths=False)

    def test_padding_factor_valid_range(self):
        """Test valid padding_factor values."""
        cfg = get_default_config()
        cfg.source_catalogue = "dummy_catalogue.csv"  # Set required field

        # Test minimum valid value
        cfg.padding_factor = 0.25
        validate_config(cfg, check_paths=False)

        # Test maximum valid value
        cfg.padding_factor = 10.0
        validate_config(cfg, check_paths=False)

        # Test some values in between
        for value in [0.5, 0.75, 1.0, 2.0, 5.5, 7.25]:
            cfg.padding_factor = value
            validate_config(cfg, check_paths=False)

    def test_padding_factor_invalid_too_small(self):
        """Test that padding_factor below minimum raises error."""
        cfg = get_default_config()
        cfg.source_catalogue = "dummy_catalogue.csv"  # Set required field
        cfg.padding_factor = 0.1  # Below minimum of 0.25

        with pytest.raises(ValueError, match="padding_factor must be >= 0.25"):
            validate_config(cfg, check_paths=False)

    def test_padding_factor_invalid_too_large(self):
        """Test that padding_factor above maximum raises error."""
        cfg = get_default_config()
        cfg.source_catalogue = "dummy_catalogue.csv"  # Set required field
        cfg.padding_factor = 10.5  # Above maximum of 10.0

        with pytest.raises(ValueError, match="padding_factor must be <= 10.0"):
            validate_config(cfg, check_paths=False)

    def test_padding_factor_invalid_type(self):
        """Test that non-numeric padding_factor raises error."""
        cfg = get_default_config()
        cfg.source_catalogue = "dummy_catalogue.csv"  # Set required field
        cfg.padding_factor = "invalid"

        with pytest.raises(ValueError, match="padding_factor must be a number"):
            validate_config(cfg, check_paths=False)

    def test_padding_factor_missing(self):
        """Test that missing padding_factor raises error (it's mandatory)."""
        cfg = get_default_config()
        cfg.source_catalogue = "dummy_catalogue.csv"  # Set required field
        del cfg.padding_factor

        with pytest.raises(ValueError, match="Missing required parameter: padding_factor"):
            validate_config(cfg, check_paths=False)

    def test_padding_factor_integer_accepted(self):
        """Test that integer values are accepted for padding_factor."""
        cfg = get_default_config()
        cfg.source_catalogue = "dummy_catalogue.csv"  # Set required field
        cfg.padding_factor = 2  # Integer should be accepted as float
        validate_config(cfg, check_paths=False)

        cfg.padding_factor = 5
        validate_config(cfg, check_paths=False)
