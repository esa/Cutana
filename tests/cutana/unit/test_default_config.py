#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for default configuration validation."""

import pytest
from dotmap import DotMap

from cutana import get_default_config, validate_config, validate_config_for_processing


class TestDefaultConfig:
    """Test suite for default configuration."""

    def test_default_config_returns_dotmap(self):
        """Test that get_default_config returns a DotMap."""
        config = get_default_config()
        assert isinstance(config, DotMap)

    def test_default_config_has_required_parameters(self):
        """Test that default config has all required parameters."""
        config = get_default_config()

        # Required parameters that must be present
        required_params = [
            "name",
            "log_level",
            "output_dir",
            "output_format",
            "data_type",
            "max_workers",
            "loadbalancer",
            "target_resolution",
            "normalisation_method",
            "interpolation",
            "fits_extensions",
            "apply_flux_conversion",
        ]

        for param in required_params:
            assert hasattr(config, param), f"Missing required parameter: {param}"
            assert getattr(config, param) is not None, f"Parameter {param} is None"

    def test_default_config_validates_successfully(self):
        """Test that default configuration passes validation."""
        config = get_default_config()

        # Set required values for validation
        config.source_catalogue = "test.csv"  # Required for validation

        # This should not raise any exceptions
        validate_config(config, check_paths=False)

    def test_default_config_parameter_types(self):
        """Test that default config parameters have correct types."""
        config = get_default_config()

        # String parameters
        assert isinstance(config.name, str)
        assert isinstance(config.log_level, str)
        assert isinstance(config.output_dir, str)
        assert isinstance(config.output_format, str)
        assert isinstance(config.data_type, str)
        assert isinstance(config.normalisation_method, str)
        assert isinstance(config.interpolation, str)

        # Integer parameters (required)
        assert isinstance(config.max_workers, int)
        assert isinstance(config.target_resolution, int)
        assert isinstance(config.N_batch_cutout_process, int)
        assert isinstance(config.max_workflow_time_seconds, int)

        # Integer parameters (optional - can be None or int)
        assert config.loadbalancer.max_sources_per_process is None or isinstance(
            config.loadbalancer.max_sources_per_process, int
        )
        assert config.process_threads is None or isinstance(config.process_threads, int)

        # Boolean parameters
        assert isinstance(config.apply_flux_conversion, bool)
        assert isinstance(config.loadbalancer.skip_memory_calibration_wait, bool)

        # List parameters
        assert isinstance(config.fits_extensions, list)
        assert isinstance(config.selected_extensions, list)

    def test_default_config_parameter_ranges(self):
        """Test that default config parameters are within valid ranges."""
        config = get_default_config()

        # Test numeric ranges (required parameters)
        assert 1 <= config.max_workers <= 32
        assert 16 <= config.target_resolution <= 2048
        assert 10 <= config.N_batch_cutout_process <= 10000
        assert 60 <= config.max_workflow_time_seconds <= 5e6

        # Test optional numeric parameters (None is acceptable)
        if config.loadbalancer.max_sources_per_process is not None:
            assert 100 <= config.loadbalancer.max_sources_per_process <= 150000
        if config.process_threads is not None:
            assert 1 <= config.process_threads <= 128

        # Test string values
        assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"]
        assert config.output_format in ["zarr", "fits"]
        assert config.data_type in ["float32", "float64", "int32", "int16", "uint16"]
        assert config.normalisation_method in ["linear", "log", "asinh", "zscale"]
        assert config.interpolation in ["bilinear", "nearest", "cubic"]

    def test_config_for_processing_requires_additional_fields(self):
        """Test that validate_config_for_processing requires additional fields."""
        config = get_default_config()

        # Default config should not validate for processing (missing required fields)
        with pytest.raises(ValueError, match="required for processing"):
            validate_config_for_processing(config)

        # Add required fields for processing
        config.source_catalogue = "test_sources.csv"
        config.selected_extensions = [{"name": "VIS", "ext": "PRIMARY"}]

        # Should still fail due to file not existing, but different error
        with pytest.raises(ValueError, match="does not exist"):
            validate_config_for_processing(config)
