#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Test for deployment validation module.

This test ensures the deployment validation function is importable and executable.
"""

import os
from unittest.mock import patch

import pytest


def test_deployment_validation_importable():
    """Test that deployment validation can be imported."""
    from cutana import deployment_validation

    assert deployment_validation is not None
    assert callable(deployment_validation)


def test_deployment_validation_returns_dict():
    """Test that deployment validation returns a dictionary."""
    from cutana.deployment_validator import DeploymentValidator

    # Create validator with quiet mode
    validator = DeploymentValidator(verbose=False)

    # Mock methods to avoid actual system checks during testing
    with patch.object(validator, "validate_conda_environment", return_value=True):
        with patch.object(validator, "validate_dependencies", return_value=True):
            with patch.object(validator, "validate_configuration", return_value=True):
                with patch.object(validator, "run_minimal_e2e_test", return_value=True):
                    with patch.object(validator, "check_git_access", return_value=True):
                        results = validator.run_all_validations()

    assert isinstance(results, dict)
    assert "conda_environment" in results
    assert "dependencies" in results
    assert "configuration" in results
    assert "end_to_end" in results
    assert "git_access" in results


def test_deployment_validator_class_exists():
    """Test that DeploymentValidator class exists and is instantiable."""
    from cutana.deployment_validator import DeploymentValidator

    validator = DeploymentValidator(verbose=False)
    assert validator is not None
    assert hasattr(validator, "validate_conda_environment")
    assert hasattr(validator, "validate_dependencies")
    assert hasattr(validator, "validate_configuration")
    assert hasattr(validator, "run_minimal_e2e_test")
    assert hasattr(validator, "check_git_access")
    assert hasattr(validator, "run_all_validations")


def test_deployment_validation_handles_missing_conda_env():
    """Test that validation handles missing conda environment gracefully."""
    from cutana.deployment_validator import DeploymentValidator

    validator = DeploymentValidator(verbose=False)

    # Mock environment without 'cutana' conda env
    with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "some_other_env"}):
        result = validator.validate_conda_environment()

    assert result is False  # Should return False when not in cutana env


def test_deployment_validation_configuration_check():
    """Test that configuration validation works."""
    from cutana.deployment_validator import DeploymentValidator

    validator = DeploymentValidator(verbose=False)

    # This should work with the real config system
    result = validator.validate_configuration()

    # Configuration should be valid (the function creates valid temp files)
    assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
