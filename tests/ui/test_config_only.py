#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for configuration management without widget dependencies."""

import tempfile
from pathlib import Path

# Test backend configuration functions which don't depend on ipywidgets
from cutana.get_default_config import get_default_config, save_config_with_timestamp


class TestConfigBackendStandalone:
    """Test the backend configuration functions without UI dependencies."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()

        # Check that it returns a DotMap
        from dotmap import DotMap

        assert isinstance(config, DotMap)

        # Check that it has required attributes
        assert hasattr(config, "apply_flux_conversion")
        assert hasattr(config, "max_workers")
        assert hasattr(config, "target_resolution")

    def test_save_config_with_timestamp(self):
        """Test saving configuration with timestamp."""
        config = get_default_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = save_config_with_timestamp(config, temp_dir)

            # Check that file was created
            assert Path(config_path).exists()
            assert "cutana_config_" in config_path
            assert config_path.endswith(".toml")

            # Check that timestamp was added to config
            assert hasattr(config, "timestamp")
            assert hasattr(config, "created_at")
